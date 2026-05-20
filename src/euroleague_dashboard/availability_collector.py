from __future__ import annotations

import argparse
import html
import re
import sqlite3
import unicodedata
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

from .config import DB_PATH, PROJECT_ROOT
import sys

ML_RUNTIME_DIR = Path(__file__).resolve().parent / "ml_runtime"
if str(ML_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(ML_RUNTIME_DIR))

from .ml_runtime.feature_engineering import PLAYER_ROLE_OVERRIDES, infer_player_roles, parse_minutes


BASKETSTORIES_INJURIES_URL = "https://www.basketstories.net/datacenter/injuries.php"
NEWS_FEEDS = [
    ("Eurohoops", "https://www.eurohoops.net/en/euroleague/feed/"),
    ("TalkBasket", "https://www.talkbasket.net/euroleague/feed"),
]
STATUS_FACTORS = {
    "out": 0.0,
    "injured": 0.0,
    "inactive": 0.0,
    "dnp": 0.0,
    "doubtful": 0.25,
    "questionable": 0.65,
    "probable": 0.9,
    "available": 1.0,
}
BASKETSTORIES_LOGO_TEAM_CODES = {
    "ASVEL2.png": "ASV",
    "Armani_Milano.png": "MIL",
    "Dubai_Basketball.png": "DUB",
    "Efes_Istanbul.png": "IST",
    "Maccabi_Tel_Aviv.png": "TEL",
    "Olympiacos_Piraeus.png": "OLY",
    "Partizan_Belgrade.png": "PRS",
    "Red_Star.png": "RED",
    "Virtus_Segafredo2.png": "VIR",
}
BASKETSTORIES_ROLE_MAP = {
    "Point Guards": "Guard",
    "Shooting Guards": "Guard",
    "Small Forwards": "Forward",
    "Power Forwards": "Forward",
    "Centers": "Center",
}
OUT_PATTERNS = [
    r"\b(out|ruled out|sidelined|will miss|misses|to miss|won't play|will not play|unavailable|injured|undergo(?:es)? surgery|injury list)\b",
]
DOUBTFUL_PATTERNS = [
    r"\b(doubtful|questionable|game-time decision|uncertain|day-to-day|could miss|might miss)\b",
]
AVAILABLE_PATTERNS = [
    r"\b(available|cleared|returns?|back in the mix|will suit up|ready to play|set to play)\b",
]


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", normalized.upper()).strip()


def player_display_name(player_name: str) -> str:
    parts = [part.strip() for part in str(player_name).replace(".", "").split(",")]
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}".strip().upper()
    return str(player_name).strip().upper()


def player_key(player_name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", normalize_text(player_display_name(player_name))).strip()


def player_last_token_first_key(player_name: str) -> str:
    tokens = re.findall(r"[A-Z0-9]+", normalize_text(player_display_name(player_name)))
    if len(tokens) < 2:
        return player_key(player_name)
    return " ".join([tokens[-1], *tokens[:-1]])


def player_variants(player_name: str) -> set[str]:
    raw = str(player_name).strip()
    display = player_display_name(raw)
    variants = {raw.upper(), display.upper()}
    if "," in raw:
        last, first = [part.strip() for part in raw.split(",", 1)]
        variants.add(f"{first} {last}".upper())
        variants.add(f"{last} {first}".upper())
    parts = display.split()
    if len(parts) >= 2:
        variants.add(" ".join(parts[:2]).upper())
    return {re.sub(r"\s+", " ", normalize_text(value)).strip() for value in variants if len(value.strip()) >= 4}


def infer_status(text: str) -> tuple[str | None, float | None]:
    lowered = str(text).lower()
    if any(re.search(pattern, lowered) for pattern in AVAILABLE_PATTERNS):
        return "available", 1.0
    if any(re.search(pattern, lowered) for pattern in OUT_PATTERNS):
        return "out", 0.0
    if any(re.search(pattern, lowered) for pattern in DOUBTFUL_PATTERNS):
        return "questionable", 0.65
    return None, None


def html_cell_text(cell: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(cell))
    return html.unescape(re.sub(r"\s+", " ", text).strip())


def latest_roster(db_path: Path = DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        boxscores = pd.read_sql(
            """
            SELECT b.season, b.game_code, g.round, b.player_id, b.player_name, b.team_code, b.minutes,
                   points, field_goals_attempted_2, field_goals_attempted_3,
                   free_throws_attempted, total_rebounds, assists, blocks_favour, pir
            FROM player_boxscores b
            LEFT JOIN games g ON b.season = g.season AND b.game_code = g.game_code
            WHERE b.player_id NOT IN ('Total', 'Team')
              AND b.player_name NOT IN ('Total', 'Team')
            """,
            conn,
        )
        profiles = pd.read_sql(
            """
            SELECT season, player_id, player_name, team_code, team_name
            FROM player_profiles
            """,
            conn,
        )
    if boxscores.empty:
        return pd.DataFrame()
    boxscores["minutes_float"] = boxscores["minutes"].map(parse_minutes)
    boxscores = boxscores[boxscores["minutes_float"].fillna(0) > 0].copy()
    latest_season = int(boxscores["season"].max())
    season_logs = boxscores[boxscores["season"].astype(int) == latest_season].copy()
    latest_round = pd.to_numeric(season_logs["round"], errors="coerce").max()
    if pd.notna(latest_round):
        active_logs = season_logs[pd.to_numeric(season_logs["round"], errors="coerce") >= int(latest_round) - 4].copy()
    else:
        active_logs = season_logs

    latest_log_team = (
        season_logs.sort_values(["player_id", "season", "game_code"])
        .drop_duplicates("player_id", keep="last")
        [["player_id", "player_name", "team_code"]]
        .rename(columns={"player_name": "log_player_name", "team_code": "log_team_code"})
    )
    if not profiles.empty:
        latest_profile_season = int(profiles["season"].max())
        profile_roster = (
            profiles[profiles["season"].astype(int) == latest_profile_season]
            .sort_values(["player_id", "team_code"])
            .drop_duplicates("player_id")
            [["player_id", "player_name", "team_code"]]
        )
        latest = profile_roster.merge(latest_log_team, on="player_id", how="left")
        multi_team = latest["team_code"].astype(str).str.contains(";", regex=False)
        latest.loc[multi_team & latest["log_team_code"].notna(), "team_code"] = latest.loc[multi_team & latest["log_team_code"].notna(), "log_team_code"]
        latest.loc[latest["player_name"].isna() & latest["log_player_name"].notna(), "player_name"] = latest.loc[latest["player_name"].isna() & latest["log_player_name"].notna(), "log_player_name"]
        latest = latest[["player_id", "player_name", "team_code"]]
        active_ids = set(active_logs["player_id"].astype(str))
        profile_ids = set(latest["player_id"].astype(str))
        extra_logs = latest_log_team[
            latest_log_team["player_id"].astype(str).isin(active_ids - profile_ids)
        ].rename(columns={"log_player_name": "player_name", "log_team_code": "team_code"})
        latest = pd.concat([latest, extra_logs], ignore_index=True)
    else:
        latest = latest_log_team.rename(columns={"log_player_name": "player_name", "log_team_code": "team_code"})
    roles = infer_player_roles(boxscores)
    latest = latest.merge(roles, on="player_id", how="left")
    latest["player_role"] = latest["player_role"].fillna("Forward")
    latest["player_role"] = latest.apply(
        lambda row: PLAYER_ROLE_OVERRIDES.get(str(row["player_id"]), row["player_role"]),
        axis=1,
    )
    return latest


def fetch_basketstories() -> pd.DataFrame:
    try:
        response = requests.get(BASKETSTORIES_INJURIES_URL, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
    except Exception:
        return pd.DataFrame()
    records = []
    pattern = re.compile(r'<div class="description1"[^>]*>(?P<role>[^<]+)</div>\s*<table[^>]*>(?P<table>.*?)</table>', re.S)
    for match in pattern.finditer(response.text):
        role = BASKETSTORIES_ROLE_MAP.get(html_cell_text(match.group("role")), "")
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", match.group("table"), flags=re.S)[1:]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)
            if len(cells) < 4:
                continue
            logo_match = re.search(r"/images/logos/([^\"'\s>]+)", cells[1])
            team_code = BASKETSTORIES_LOGO_TEAM_CODES.get(logo_match.group(1), "") if logo_match else ""
            raw_status = html_cell_text(cells[2]).lower()
            status = "doubtful" if "doubt" in raw_status or "αμφ" in raw_status else "out"
            records.append({
                "player": html_cell_text(cells[0]),
                "player_id": "",
                "team_code": team_code,
                "status": status,
                "impact": STATUS_FACTORS.get(status, 0.0),
                "note": raw_status,
                "updated": html_cell_text(cells[3]),
                "role": role,
                "source": "BasketStories",
                "source_url": BASKETSTORIES_INJURIES_URL,
            })
    return pd.DataFrame(records)


def fetch_news(roster: pd.DataFrame) -> pd.DataFrame:
    if roster.empty:
        return pd.DataFrame()
    lookup = []
    for row in roster.itertuples(index=False):
        lookup.append({
            "player_id": str(row.player_id),
            "player": str(row.player_name),
            "team_code": str(row.team_code).upper(),
            "role": str(row.player_role),
            "variants": player_variants(str(row.player_name)),
        })
    records = []
    for source, feed_url in NEWS_FEEDS:
        try:
            response = requests.get(feed_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue
        for item in root.findall("./channel/item")[:50]:
            title = html.unescape(item.findtext("title") or "")
            description = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
            text = f"{title} {description}"
            status, impact = infer_status(text)
            if status is None:
                continue
            normalized = normalize_text(text)
            for player in lookup:
                if not any(re.search(rf"\b{re.escape(variant)}\b", normalized) for variant in player["variants"]):
                    continue
                records.append({
                    "player": player["player"],
                    "player_id": player["player_id"],
                    "team_code": player["team_code"],
                    "status": status,
                    "impact": impact,
                    "note": title,
                    "updated": item.findtext("pubDate") or "",
                    "role": player["role"],
                    "source": source,
                    "source_url": item.findtext("link") or feed_url,
                })
    return pd.DataFrame(records)


def normalize_availability(frame: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["player", "player_id", "team_code", "status", "impact", "note", "updated", "role", "source", "source_url"])
    out = frame.copy()
    for col in ["player", "player_id", "team_code", "status", "impact", "note", "updated", "role", "source", "source_url"]:
        if col not in out.columns:
            out[col] = ""
    out["key"] = out["player"].map(player_key)
    roster_keys = pd.concat(
        [
            roster.assign(key=roster["player_name"].map(player_key)),
            roster.assign(key=roster["player_name"].map(player_last_token_first_key)),
        ],
        ignore_index=True,
    ).drop_duplicates(["player_id", "key"])
    out = out.merge(
        roster_keys[["player_id", "player_name", "team_code", "player_role", "key"]].rename(columns={
            "player_id": "matched_player_id",
            "team_code": "matched_team_code",
        }),
        on="key",
        how="left",
    )
    out["player_id"] = out["player_id"].replace("", pd.NA).fillna(out["matched_player_id"]).fillna("")
    out["team_code"] = out["team_code"].replace("", pd.NA).fillna(out["matched_team_code"]).fillna("").astype(str).str.upper()
    out["player"] = out["player"].replace("", pd.NA).fillna(out["player_name"]).fillna("")
    out["role"] = out["role"].replace("", pd.NA).fillna(out["player_role"]).fillna("")
    out["status"] = out["status"].fillna("available").astype(str).str.lower().str.strip()
    out["impact"] = pd.to_numeric(out["impact"], errors="coerce").fillna(out["status"].map(STATUS_FACTORS)).fillna(1.0).clip(0, 1)
    out["collected_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    out = out.sort_values(["source", "updated"]).drop_duplicates(["player_id", "key", "team_code"], keep="last")
    return out[["player", "player_id", "team_code", "status", "impact", "note", "updated", "role", "source", "source_url", "collected_at"]]


def build_rotation_impact(roster: pd.DataFrame, availability: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    if roster.empty:
        return pd.DataFrame()
    with sqlite3.connect(db_path) as conn:
        logs = pd.read_sql(
            """
            SELECT season, game_code, player_id, player_name, team_code, minutes,
                   field_goals_attempted_2, field_goals_attempted_3, free_throws_attempted, pir
            FROM player_boxscores
            WHERE player_id NOT IN ('Total', 'Team')
              AND player_name NOT IN ('Total', 'Team')
            """,
            conn,
        )
    logs["minutes_float"] = logs["minutes"].map(parse_minutes)
    logs = logs[logs["minutes_float"].fillna(0) > 0].copy().sort_values(["player_id", "season", "game_code"])
    logs["shot_load"] = (
        logs["field_goals_attempted_2"].fillna(0)
        + logs["field_goals_attempted_3"].fillna(0)
        + logs["free_throws_attempted"].fillna(0)
    )
    recent = logs.groupby("player_id").tail(5).groupby("player_id", as_index=False).agg(
        base_minutes=("minutes_float", "mean"),
        base_usage=("shot_load", "mean"),
        base_pir=("pir", "mean"),
    )
    base = roster.merge(recent, on="player_id", how="left")
    base = base.merge(
        availability[["player_id", "status", "impact", "source", "note"]].rename(columns={
            "status": "availability_status",
            "impact": "availability_impact",
            "source": "availability_source",
            "note": "availability_note",
        }),
        on="player_id",
        how="left",
    )
    base["availability_status"] = base["availability_status"].fillna("available")
    base["availability_impact"] = pd.to_numeric(base["availability_impact"], errors="coerce").fillna(1.0).clip(0, 1)
    base["base_minutes"] = pd.to_numeric(base["base_minutes"], errors="coerce").fillna(0)
    base["base_usage"] = pd.to_numeric(base["base_usage"], errors="coerce").fillna(0)
    base["missing_minutes"] = base["base_minutes"] * (1.0 - base["availability_impact"])
    base["missing_usage"] = base["base_usage"] * (1.0 - base["availability_impact"])
    team_missing = base.groupby("team_code", as_index=False).agg(
        team_missing_minutes=("missing_minutes", "sum"),
        team_missing_usage=("missing_usage", "sum"),
    )
    role_missing = base.groupby(["team_code", "player_role"], as_index=False).agg(
        role_missing_minutes=("missing_minutes", "sum"),
        role_missing_usage=("missing_usage", "sum"),
    )
    out = base.merge(team_missing, on="team_code", how="left").merge(role_missing, on=["team_code", "player_role"], how="left")
    out["available_role_minutes"] = out.groupby(["team_code", "player_role"])["base_minutes"].transform(
        lambda values: values[out.loc[values.index, "availability_impact"] > 0.25].sum()
    )
    out["role_share"] = (out["base_minutes"] / out["available_role_minutes"].replace(0, pd.NA)).fillna(0)
    out["same_role_minutes_boost"] = (out["role_missing_minutes"] * 0.42 * out["role_share"]).clip(0, 7)
    out["cross_role_minutes_boost"] = ((out["team_missing_minutes"] - out["role_missing_minutes"]).clip(lower=0) * 0.08 * out["role_share"]).clip(0, 3)
    out["minutes_penalty"] = ((1.0 - out["availability_impact"]) * out["base_minutes"] * 0.55).clip(0, 30)
    out["net_minutes_delta"] = out["same_role_minutes_boost"] + out["cross_role_minutes_boost"] - out["minutes_penalty"]
    out["rotation_impact_score"] = (out["net_minutes_delta"] / out["base_minutes"].replace(0, pd.NA)).fillna(0).clip(-1, 1)
    out["collected_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    return out[[
        "player_id", "player_name", "team_code", "player_role",
        "base_minutes", "base_usage", "base_pir",
        "availability_status", "availability_impact", "availability_source", "availability_note",
        "team_missing_minutes", "team_missing_usage", "role_missing_minutes", "role_missing_usage",
        "same_role_minutes_boost", "cross_role_minutes_boost", "minutes_penalty",
        "net_minutes_delta", "rotation_impact_score", "collected_at",
    ]]


def collect(db_path: Path = DB_PATH, output_dir: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir = output_dir or PROJECT_ROOT / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    roster = latest_roster(db_path)
    frames = [fetch_basketstories(), fetch_news(roster)]
    availability = normalize_availability(pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if any(not frame.empty for frame in frames) else pd.DataFrame(), roster)
    rotation = build_rotation_impact(roster, availability, db_path)
    availability.to_csv(output_dir / "player_availability_collected.csv", index=False)
    rotation.to_csv(output_dir / "rotation_impact.csv", index=False)
    return availability, rotation


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect EuroLeague player availability and rotation impact data.")
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    args = parser.parse_args()
    availability, rotation = collect(args.db, args.output_dir)
    print(f"availability_rows={len(availability)}")
    print(f"rotation_rows={len(rotation)}")
    if not availability.empty:
        print(availability.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
