from __future__ import annotations

import base64
import html
import re
import sqlite3
import xml.etree.ElementTree as ET
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

requests.packages.urllib3.disable_warnings()  # type: ignore[attr-defined]

from src.euroleague_dashboard.analytics import (
    PLAYER_STAT_COLUMNS,
    available_date_bounds,
    database_summary,
    player_game_logs,
    player_profiles,
    season_and_total_averages,
    shot_chart_data,
    team_game_logs,
    team_leaderboard,
)
from src.euroleague_dashboard.config import DB_PATH, PROJECT_ROOT
from src.euroleague_dashboard.ml_predictor import clear_ml_caches, predict_player, predict_players
from src.euroleague_dashboard.storage import connect


st.set_page_config(page_title="EuroLeague Intelligence", layout="wide", initial_sidebar_state="expanded")

ASSET_DIR = PROJECT_ROOT / "assets" / "logos"
TEAM_LOGO_DIR = ASSET_DIR / "teams"
COACH_PHOTO_DIR = PROJECT_ROOT / "assets" / "coach_photos"
PLAYER_PHOTO_DIR = PROJECT_ROOT / "assets" / "player_photos"

PLAYER_STATS = PLAYER_STAT_COLUMNS
TEAM_STATS = ["points", "points_allowed", "point_diff", "pir", "pir_allowed", "assists", "total_rebounds"]
STAT_LABELS = {
    "points": "Points",
    "points_allowed": "Pts Allowed",
    "point_diff": "Point Diff",
    "total_rebounds": "Rebounds",
    "assists": "Assists",
    "steals": "Steals",
    "blocks_favour": "Blocks",
    "turnovers": "Turnovers",
    "fg_pct": "FG%",
    "three_pct": "3PT%",
    "minutes_avg": "Minutes",
    "pir": "PIR",
    "pir_allowed": "PIR Allowed",
}
PHASE_LABELS = {"RS": "Regular Season", "PI": "Play-In", "PO": "Playoffs", "FF": "Final Four"}
PHASE_CODES = {label: code for code, label in PHASE_LABELS.items()}
VENUE_CODES = {"All": None, "Home": 1, "Away": 0}
CHART_COLORS = ["#f26a21", "#2f80ed", "#16a3b8", "#8b5cf6", "#1eb6a0", "#d69d26", "#d85c5c"]
COMPARE_LEFT_DARK = "#f26a21"
COMPARE_RIGHT_DARK = "#38bdf8"
COMPARE_LEFT_LIGHT = "#dc5f1f"
COMPARE_RIGHT_LIGHT = "#2563eb"
OFFICIAL_SCHEDULE_URL = "https://api-live.euroleague.net/v2/competitions/E/seasons/E{season}/games"
DUNKEST_STATS_URL = "https://www.dunkest.com/api/stats/table"
BASKETSTORIES_INJURIES_URL = "https://www.basketstories.net/datacenter/injuries.php"
AVAILABILITY_NEWS_FEEDS = [
    ("Eurohoops", "https://www.eurohoops.net/en/euroleague/feed/"),
    ("TalkBasket", "https://www.talkbasket.net/euroleague/feed"),
]
DUNKEST_SEASON_ID = 23
DUNKEST_TEAMS = list(range(32, 49)) + [56, 60, 75]
DUNKEST_POSITIONS = [1, 2, 3]
DUNKEST_LOCAL_FILES = [
    PROJECT_ROOT / "data" / "dunkest_players.json",
    PROJECT_ROOT / "data" / "dunkest_players.csv",
    PROJECT_ROOT / "data" / "dunkest_players.txt",
    PROJECT_ROOT / "data" / "fantasy_data.json",
]
COACH_LOCAL_FILES = [
    PROJECT_ROOT / "data" / "dunkest_coaches.csv",
    PROJECT_ROOT / "data" / "fantasy_coaches.csv",
    PROJECT_ROOT / "data" / "coaches.csv",
]
AVAILABILITY_LOCAL_FILES = [
    PROJECT_ROOT / "data" / "player_availability_collected.csv",
    PROJECT_ROOT / "data" / "player_availability.csv",
    PROJECT_ROOT / "data" / "injuries.csv",
]
ROTATION_IMPACT_PATH = PROJECT_ROOT / "data" / "rotation_impact.csv"
COACH_TEAM_CODE_ALIASES = {"RMB": "MAD", "PAO": "PAN", "VBC": "PAM", "ASM": "MCO", "FNB": "ULK"}
AVAILABILITY_STATUS_FACTORS = {
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
NEWS_OUT_PATTERNS = [
    r"\b(out|ruled out|sidelined|will miss|misses|to miss|won't play|will not play|unavailable|injured|undergo(?:es)? surgery|injury list)\b",
]
NEWS_DOUBTFUL_PATTERNS = [
    r"\b(doubtful|questionable|game-time decision|uncertain|day-to-day|could miss|might miss)\b",
]
NEWS_AVAILABLE_PATTERNS = [
    r"\b(available|cleared|returns?|back in the mix|will suit up|ready to play|set to play)\b",
]
FANTASY_FORMATIONS = [(2, 2, 1), (1, 2, 2), (2, 1, 2), (1, 3, 1), (3, 1, 1)]
FANTASY_POSITION_TARGETS = {"Guard": 4, "Forward": 4, "Center": 2}
FANTASY_CAPTAIN_MULTIPLIER = 2.0
FANTASY_TEAM_WIN_BOOST = 1.10
DUNKEST_EMPTY_COLUMNS = ["dunkest_key", "Dunkest CR", "Dunkest Base CR", "Dunkest PDK", "Dunkest GP", "Dunkest PLUS"]
FANTASY_OPTIMIZER_ROLE_LIMITS = {"Guard": 16, "Forward": 18, "Center": 10}
PLAYER_PHOTO_FALLBACKS = {
    "P008161": "assets/player_photos/P008161.png",
}
COACH_EUROLEAGUE_PROFILE_IDS = {
    "georgios bartzokas": "001869",
    "sergio scariolo": "wav",
    "saras jasikevicius": "adg",
    "sarunas jasikevicius": "adg",
    "ergin ataman": "wcl",
    "pedro martinez": "cwx",
    "tomas masiulis": "acx",
    "dimitris itoudis": "cag",
    "manuchar markoishvili": "bed",
}
PLAYER_ROLE_OVERRIDES = {
    "P003469": "Forward",  # Sasha Vezenkov
    "P014124": "Guard",    # Talen Horton-Tucker
    "P008161": "Forward",  # Juhann Begarin
}


def image_data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def project_asset_path(path_value) -> Path | None:
    if not isinstance(path_value, str) or not path_value.strip():
        return None
    normalized = Path(path_value.strip().replace("\\", "/"))
    return normalized if normalized.is_absolute() else PROJECT_ROOT / normalized


def player_name_to_display_name(player_name: str) -> str:
    parts = [part.strip() for part in str(player_name).replace(".", "").split(",")]
    if len(parts) >= 2:
        return f"{parts[1]} {parts[0]}".strip().upper()
    return str(player_name).strip().upper()


def player_name_to_slug(player_name: str) -> str:
    display = player_name_to_display_name(player_name).lower()
    normalized = unicodedata.normalize("NFKD", display).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def coach_photo_key(coach_name: str) -> str:
    normalized = unicodedata.normalize("NFKD", str(coach_name)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "_", normalized.lower()).strip("_") or "coach"


def player_name_match_key(player_name: str) -> str:
    display = player_name_to_display_name(player_name)
    normalized = unicodedata.normalize("NFKD", display).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^A-Z0-9]+", " ", normalized.upper()).strip()


def player_name_last_token_first_key(player_name: str) -> str:
    display = player_name_to_display_name(player_name)
    tokens = re.findall(r"[A-Z0-9]+", unicodedata.normalize("NFKD", display).encode("ascii", "ignore").decode("ascii").upper())
    if len(tokens) < 2:
        return player_name_match_key(player_name)
    return " ".join([tokens[-1], *tokens[:-1]])


@st.cache_data(show_spinner=False)
def fetch_and_cache_official_player_photo(player_id: str, player_name: str) -> str:
    player_code = str(player_id).replace("P", "").lstrip("0")
    if not player_code:
        return ""
    padded_code = player_code.zfill(6)
    page_url = f"https://www.euroleaguebasketball.net/euroleague/players/{player_name_to_slug(player_name)}/{padded_code}/"
    try:
        response = requests.get(page_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        response.raise_for_status()
    except requests.RequestException:
        return ""
    display_name = re.escape(player_name_to_display_name(player_name))
    match = re.search(
        rf'<img[^>]+data-srcset="([^"]+)"[^>]+alt="{display_name}"',
        response.text,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    image_url = html.unescape(match.group(1).split()[0])
    try:
        image_response = requests.get(image_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        image_response.raise_for_status()
    except requests.RequestException:
        return image_url
    path = PROJECT_ROOT / "assets" / "player_photos" / f"{player_id}.png"
    path.write_bytes(image_response.content)
    return image_data_uri(path)


@st.cache_data(show_spinner=False)
def fetch_and_cache_official_coach_photo(coach_name: str, profile_id: str) -> str:
    if not profile_id:
        return ""
    page_url = f"https://www.euroleaguebasketball.net/en/euroleague/players/{player_name_to_slug(coach_name)}/profile/{profile_id}/"
    try:
        response = requests.get(page_url, timeout=12, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        response.raise_for_status()
    except requests.RequestException:
        return ""
    match = re.search(r'"photo":"([^"]+)"', response.text)
    if not match:
        return ""
    image_url = match.group(1).replace("\\u002F", "/")
    if not image_url or "default" in image_url.lower() or '"isDefaultPhoto":true' in response.text:
        return ""
    try:
        image_response = requests.get(image_url, timeout=20, headers={"User-Agent": "Mozilla/5.0"}, verify=False)
        image_response.raise_for_status()
    except requests.RequestException:
        return image_url
    COACH_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = COACH_PHOTO_DIR / f"{coach_photo_key(coach_name)}.png"
    path.write_bytes(image_response.content)
    return image_data_uri(path)


def team_logo_uri(team_code: str) -> str:
    svg = TEAM_LOGO_DIR / f"{team_code}.svg"
    if svg.exists():
        return image_data_uri(svg)
    return image_data_uri(TEAM_LOGO_DIR / f"{team_code}.png")


def euroleague_logo_uri() -> str:
    return image_data_uri(ASSET_DIR / "euroleague_logo.svg")


def player_photo_uri(player_id: str, profiles_df: pd.DataFrame) -> str:
    fallback_path = PLAYER_PHOTO_FALLBACKS.get(str(player_id))
    asset_path = PLAYER_PHOTO_DIR / f"{str(player_id)}.png"
    if profiles_df.empty or not player_id:
        return image_data_uri(asset_path) or (image_data_uri(PROJECT_ROOT / fallback_path) if fallback_path else "")
    match = profiles_df[profiles_df["player_id"] == player_id]
    if match.empty:
        return image_data_uri(asset_path) or (image_data_uri(PROJECT_ROOT / fallback_path) if fallback_path else "")
    local_path = match.iloc[0].get("local_image_path")
    resolved_path = project_asset_path(local_path)
    local_photo = image_data_uri(resolved_path) if resolved_path else ""
    if not local_photo:
        asset_photo = image_data_uri(asset_path)
        if asset_photo:
            return asset_photo
        image_url = match.iloc[0].get("image_url")
        if isinstance(image_url, str) and image_url:
            return image_url
        return image_data_uri(PROJECT_ROOT / fallback_path) if fallback_path else ""
    return local_photo


def player_photo_uri_for_name(player_id: str, player_name: str, profiles_df: pd.DataFrame) -> str:
    photo = player_photo_uri(player_id, profiles_df)
    if photo:
        return photo
    fetched = fetch_and_cache_official_player_photo(player_id, player_name)
    if fetched:
        return fetched
    fallback_path = PLAYER_PHOTO_FALLBACKS.get(str(player_id))
    return image_data_uri(PROJECT_ROOT / fallback_path) if fallback_path else ""


def coach_photo_uri(coach_name: str, team_code: str = "") -> str:
    COACH_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    if team_code:
        paths.append(COACH_PHOTO_DIR / f"{str(team_code).upper()}.png")
    paths.append(COACH_PHOTO_DIR / f"{coach_photo_key(coach_name)}.png")
    for path in paths:
        if path.exists():
            return image_data_uri(path)
    profile_id = COACH_EUROLEAGUE_PROFILE_IDS.get(coach_photo_key(coach_name).replace("_", " "))
    return fetch_and_cache_official_coach_photo(coach_name, profile_id) if profile_id else ""


def minutes_to_float(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    if ":" in text:
        mins, secs = text.split(":", 1)
        return float(mins or 0) + float(secs or 0) / 60
    parsed = pd.to_numeric(text, errors="coerce")
    return float(parsed) if pd.notna(parsed) else 0.0


def is_light_theme() -> bool:
    return st.session_state.get("app_theme", "Dark") == "Light"


def compare_palette() -> tuple[str, str]:
    if is_light_theme():
        return COMPARE_LEFT_LIGHT, COMPARE_RIGHT_LIGHT
    return COMPARE_LEFT_DARK, COMPARE_RIGHT_DARK


def season_label(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, str) and "-" in value:
        return value
    start_year = int(value)
    return f"{start_year}-{start_year + 1}"


def season_axis_label(value) -> str:
    return season_label(value)


def plotly_dark(height: int = 420) -> dict:
    if is_light_theme():
        return {
            "height": height,
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(255,255,255,.72)",
            "font": {"color": "#172033", "family": "Outfit, Arial, sans-serif"},
            "colorway": ["#e85d04", "#1d4ed8", "#0891b2", "#7c3aed", "#0f766e", "#ca8a04", "#dc2626"],
            "xaxis": {
                "gridcolor": "rgba(71,85,105,.18)",
                "zeroline": False,
                "linecolor": "rgba(71,85,105,.28)",
                "tickfont": {"color": "#334155"},
                "title": {"font": {"color": "#172033"}},
            },
            "yaxis": {
                "gridcolor": "rgba(71,85,105,.18)",
                "zeroline": False,
                "linecolor": "rgba(71,85,105,.28)",
                "tickfont": {"color": "#334155"},
                "title": {"font": {"color": "#172033"}},
            },
            "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "bgcolor": "rgba(0,0,0,0)", "font": {"color": "#172033"}, "title": {"font": {"color": "#172033"}}},
            "hoverlabel": {"bgcolor": "#ffffff", "bordercolor": "#cbd5e1", "font": {"color": "#172033"}},
            "margin": {"l": 42, "r": 26, "t": 62, "b": 42},
        }
    return {
        "height": height,
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#dbeafe", "family": "Outfit, Arial, sans-serif"},
        "colorway": CHART_COLORS,
        "xaxis": {
            "gridcolor": "rgba(148,163,184,.12)",
            "zeroline": False,
            "linecolor": "rgba(148,163,184,.18)",
            "tickfont": {"color": "#cbd5e1"},
            "title": {"font": {"color": "#eaf2ff"}},
        },
        "yaxis": {
            "gridcolor": "rgba(148,163,184,.12)",
            "zeroline": False,
            "linecolor": "rgba(148,163,184,.18)",
            "tickfont": {"color": "#cbd5e1"},
            "title": {"font": {"color": "#eaf2ff"}},
        },
        "legend": {"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "bgcolor": "rgba(0,0,0,0)", "font": {"color": "#dbeafe"}, "title": {"font": {"color": "#eaf2ff"}}},
        "hoverlabel": {"bgcolor": "#0d1b2e", "bordercolor": "#2b3d56", "font": {"color": "#eaf2ff"}},
        "margin": {"l": 42, "r": 26, "t": 62, "b": 42},
    }


def polish_plotly_text(fig: go.Figure) -> go.Figure:
    if is_light_theme():
        fig.update_layout(
            font=dict(color="#172033", family="Outfit, Arial, sans-serif"),
            title_font=dict(color="#172033"),
            legend_font=dict(color="#334155"),
            legend_title_font=dict(color="#172033"),
        )
        fig.update_xaxes(tickfont=dict(color="#334155"), title_font=dict(color="#172033"))
        fig.update_yaxes(tickfont=dict(color="#334155"), title_font=dict(color="#172033"))
        fig.for_each_annotation(lambda annotation: annotation.update(font=dict(color="#172033")))
        return fig
    fig.update_layout(
        font=dict(color="#dbeafe", family="Outfit, Arial, sans-serif"),
        title_font=dict(color="#eaf2ff"),
        legend_font=dict(color="#dbeafe"),
        legend_title_font=dict(color="#eaf2ff"),
    )
    fig.update_xaxes(tickfont=dict(color="#cbd5e1"), title_font=dict(color="#eaf2ff"))
    fig.update_yaxes(tickfont=dict(color="#cbd5e1"), title_font=dict(color="#eaf2ff"))
    fig.for_each_annotation(lambda annotation: annotation.update(font=dict(color="#eaf2ff")))
    return fig


def style_line_chart(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    title_color = "#182033" if is_light_theme() else "#eaf2ff"
    tick_color = "#4b5c72" if is_light_theme() else "#cbd5e1"
    grid_color = "rgba(71,85,105,.16)" if is_light_theme() else "rgba(148,163,184,.13)"
    marker_line = "#ffffff" if is_light_theme() else "#07111d"
    layout = plotly_dark(height)
    layout.update(
        title=dict(text=title, x=0, y=.98, font=dict(size=21, color=title_color)),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(color=title_color, size=12)),
        hovermode="x unified",
        margin=dict(l=46, r=28, t=86, b=46),
        xaxis=dict(title="", gridcolor=grid_color, zeroline=False, linecolor=grid_color, tickfont=dict(color=tick_color)),
        yaxis=dict(title="", gridcolor=grid_color, zeroline=False, linecolor=grid_color, tickfont=dict(color=tick_color)),
    )
    fig.update_traces(
        mode="lines+markers",
        line=dict(width=3.4, shape="spline", smoothing=.55),
        marker=dict(size=8, line=dict(color=marker_line, width=1.5)),
        hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:.2f}<extra></extra>",
    )
    fig.update_layout(**layout)
    return polish_plotly_text(fig)


def style_bar_chart(fig: go.Figure, title: str, height: int = 380, horizontal: bool = False) -> go.Figure:
    title_color = "#182033" if is_light_theme() else "#eaf2ff"
    tick_color = "#4b5c72" if is_light_theme() else "#cbd5e1"
    grid_color = "rgba(71,85,105,.14)" if is_light_theme() else "rgba(148,163,184,.12)"
    layout = plotly_dark(height)
    layout.update(
        title=dict(text=title, x=0, font=dict(size=21, color=title_color)),
        legend_title_text="",
        legend=dict(orientation="h", yanchor="bottom", y=1.09, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)", font=dict(color=title_color, size=12)),
        bargap=.28,
        bargroupgap=.08,
        margin=dict(l=86 if horizontal else 48, r=42, t=88, b=42),
        xaxis=dict(title="", gridcolor=grid_color, zeroline=False, linecolor=grid_color, tickfont=dict(color=tick_color)),
        yaxis=dict(title="", gridcolor="rgba(0,0,0,0)" if horizontal else grid_color, zeroline=False, linecolor=grid_color, tickfont=dict(color=tick_color)),
    )
    fig.update_traces(
        marker_line_width=0,
        opacity=.94,
        texttemplate="%{text:.1f}",
        textposition="outside",
        cliponaxis=False,
    )
    fig.update_layout(**layout)
    return polish_plotly_text(fig)


def apply_styles(theme: str = "Dark") -> None:
    light_overrides = ""
    if theme == "Light":
        light_overrides = """
        :root {
            --bg: #f7f8fb;
            --panel: #ffffff;
            --panel-2: #f3f6fa;
            --panel-3: #e9eef6;
            --line: #d7deea;
            --text: #182033;
            --muted: #65738a;
            --muted-2: #7b8799;
            --orange: #dc5f1f;
            --orange-soft: #f08a42;
            --green: #138a76;
            --red: #c94f4f;
            --shadow: 0 18px 46px rgba(24,32,51,.08);
        }
        [data-testid="stAppViewContainer"] {background: radial-gradient(circle at 12% 0%, rgba(220,95,31,.08), transparent 31%), linear-gradient(135deg, #f9fafc 0%, #eef3f8 58%, #f7f8fb 100%) !important;}
        [data-testid="stAppViewContainer"]:before {background-image: linear-gradient(rgba(71,85,105,.052) 1px, transparent 1px), linear-gradient(90deg, rgba(71,85,105,.044) 1px, transparent 1px) !important;}
        h1, h2, h3, p, label, .stMarkdown, .stMarkdown p {color: var(--text) !important;}
        [data-testid="stSidebar"] {background: linear-gradient(180deg, #ffffff 0%, #f4f7fb 100%) !important; border-right: 1px solid var(--line) !important; box-shadow: 18px 0 50px rgba(24,32,51,.08) !important;}
        [data-testid="stSidebar"] *, [data-testid="stSidebar"] p {color: #334155 !important;}
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(220,95,31,.08) !important; color: var(--text) !important;}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background: linear-gradient(90deg, rgba(220,95,31,.14), rgba(220,95,31,.04)) !important; color: var(--orange) !important;}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {background: #ffffff !important; color: var(--text) !important; border: 1px solid #cbd5e1 !important; box-shadow: inset 0 1px 0 rgba(255,255,255,.65) !important;}
        [data-baseweb="select"] span, [data-baseweb="select"] div, [data-baseweb="input"] input {color: var(--text) !important;}
        [data-baseweb="popover"] ul, [data-baseweb="menu"], [data-baseweb="menu"] li, [role="option"] {background: #ffffff !important; color: var(--text) !important; border-color: var(--line) !important;}
        .topbar {background: rgba(255,255,255,.88) !important; border-bottom-color: var(--line) !important;}
        .topbar-title, .identity-title {color: var(--text) !important; text-shadow: none !important;}
        .topbar-pill {background: #ffffff !important; color: var(--muted) !important; border-color: var(--line) !important;}
        .refresh-status {
            background: #ffffff !important;
            color: #64748b !important;
            border-color: var(--line) !important;
            box-shadow: 0 10px 24px rgba(15,23,42,.06) !important;
        }
        .refresh-status strong {color: #334155 !important;}
        .refresh-row span:first-child {color: #64748b !important;}
        .refresh-row span:last-child {color: #172033 !important;}
        .brand-sub, .hero-copy, .player-sub {color: #64748b !important;}
        .apply-button {color: #ffffff !important;}
        .hero-card, .identity-card, .section-card, .metric-card {
            background: linear-gradient(180deg, rgba(255,255,255,.98), rgba(246,249,253,.98)) !important;
            border-color: var(--line) !important;
            box-shadow: var(--shadow) !important;
        }
        .hero-card:after {border-color: rgba(71,85,105,.16) !important;}
        .hero-title, .section-title, .player-main {color: var(--text) !important; text-shadow: none !important;}
        .section-card-header {background: rgba(24,32,51,.025) !important; border-bottom-color: var(--line) !important;}
        .metric-label {color: #64748b !important;}
        .metric-value, .player-score {color: var(--text) !important; text-shadow: none !important;}
        .dark-table, .overview-table {color: #172033 !important;}
        .dark-table th, .overview-table th {background: #e8eef6 !important; color: #475569 !important;}
        .dark-table td, .overview-table td, .top-player-row {background: #ffffff !important; border-top-color: #d8e0ec !important; color: #172033 !important;}
        .dark-table tr:nth-child(even) td, .overview-table tr:nth-child(even) td {background: #f7f9fc !important;}
        .dark-table tbody tr:hover td, .overview-table tbody tr:hover td, .top-player-row:hover {background: #eef4fb !important;}
        .avatar {background: linear-gradient(135deg, #e8eef6, #ffffff) !important; color: #172033 !important; border-color: #cbd5e1 !important;}
        .player-photo {background: #ffffff !important; border-color: #cbd5e1 !important;}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {background: #ffffff !important; border-color: var(--line) !important;}
        [data-testid="stTabs"] [data-baseweb="tab"] {color: #64748b !important;}
        [data-testid="stTabs"] [aria-selected="true"] {color: var(--orange) !important; background: #fff7ed !important;}
        .stButton button, [data-testid="stDownloadButton"] button {
            background: #ffffff !important;
            color: var(--text) !important;
            border: 1px solid #cbd5e1 !important;
            box-shadow: 0 10px 24px rgba(15,23,42,.08) !important;
        }
        .stButton button:hover, [data-testid="stDownloadButton"] button:hover {
            border-color: rgba(220,95,31,.52) !important;
            color: var(--orange) !important;
            background: #fff7ed !important;
        }
        [data-testid="stSlider"] [data-baseweb="slider"] > div {color: var(--orange) !important;}
        [data-testid="stExpander"] {
            background: #ffffff !important;
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            box-shadow: 0 12px 30px rgba(15,23,42,.06) !important;
        }
        [data-testid="stExpander"] * {color: #172033 !important;}
        .match-fixture-value span {color: #64748b !important;}
        .fantasy-wrap {
            background: linear-gradient(180deg, #ffffff, #f6f0ff) !important;
            color: #172033 !important;
            border-color: #d8e0ec !important;
            box-shadow: 0 18px 46px rgba(15,23,42,.10) !important;
        }
        .fantasy-title {color: #172033 !important;}
        .fantasy-pill, .fantasy-bench, .fantasy-coach {background: rgba(255,255,255,.72) !important; border-color: var(--line) !important; color: #172033 !important;}
        .fantasy-section-label, .fantasy-note, .fantasy-pill span {color: #64748b !important;}
        .js-plotly-plot .gtitle, .js-plotly-plot .xtitle, .js-plotly-plot .ytitle, .js-plotly-plot .legendtitletext, .js-plotly-plot .annotation-text {fill: #172033 !important; color: #172033 !important;}
        .js-plotly-plot .xtick text, .js-plotly-plot .ytick text, .js-plotly-plot .legendtext {fill: #334155 !important; color: #334155 !important;}
        """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;700;800;900&family=JetBrains+Mono:wght@500;700&display=swap');
        :root {
            --bg: #07111d;
            --panel: #101d2f;
            --panel-2: #15243a;
            --panel-3: #1b2d47;
            --line: #2b3a50;
            --text: #eaf2ff;
            --muted: #9baac0;
            --muted-2: #7e8ca2;
            --orange: #f26a21;
            --orange-soft: #f39a4a;
            --green: #1eb6a0;
            --red: #e05f5f;
            --shadow: 0 20px 56px rgba(0,0,0,.22);
        }
        html, body, [class*="css"] {font-family: 'Outfit', Arial, sans-serif;}
        [data-testid="stAppViewContainer"] {
            background: radial-gradient(circle at 9% 0%, rgba(242,106,33,.12), transparent 29%), linear-gradient(135deg, #070d16 0%, #091522 48%, #0b1724 100%);
        }
        [data-testid="stAppViewContainer"]:before {
            content: "";
            position: fixed;
            inset: 0;
            pointer-events: none;
            background-image:
                linear-gradient(rgba(148,163,184,.035) 1px, transparent 1px),
                linear-gradient(90deg, rgba(148,163,184,.03) 1px, transparent 1px);
            background-size: 44px 44px;
            mask-image: linear-gradient(to bottom, black, transparent 72%);
        }
        .block-container {max-width: 1360px; padding: 0 1.75rem 2.75rem;}
        #MainMenu, footer {visibility: hidden;}
        header {visibility: visible;}
        [data-testid="stHeader"] {background: transparent;}
        [data-testid="stHeader"] [data-testid="stToolbar"] {visibility: visible;}
        [data-testid="stHeader"] [data-testid="stToolbar"] button:not([data-testid="stExpandSidebarButton"]) {
            visibility: hidden;
        }
        [data-testid="stHeader"] [data-testid="stToolbar"] div:has([data-testid="stExpandSidebarButton"]) {
            visibility: visible !important;
        }
        [data-testid="collapsedControl"],
        [data-testid="stSidebarCollapsedControl"],
        [data-testid="stSidebarCollapseButton"],
        [data-testid="stExpandSidebarButton"] {
            visibility: visible !important;
            opacity: 1 !important;
            pointer-events: auto !important;
            z-index: 9999 !important;
        }
        h1, h2, h3, p, label, .stMarkdown {color: var(--text);}
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #101827 0%, #0b1524 100%);
            border-right: 1px solid rgba(148,163,184,.16);
            width: 304px !important;
            box-shadow: 18px 0 50px rgba(0,0,0,.2);
        }
        [data-testid="stSidebar"] > div {padding: .75rem 0 1rem;}
        [data-testid="stSidebar"] * {color: #cbd5e1;}
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {padding: 0 28px;}
        [data-testid="stSidebar"] [data-testid="stRadio"] {margin: 0 28px 14px; width: calc(100% - 56px);}
        [data-testid="stSidebar"] [role="radiogroup"],
        [data-testid="stSidebar"] [role="radiogroup"] > div,
        [data-testid="stSidebar"] [role="radiogroup"] > div > label,
        [data-testid="stSidebar"] [role="radiogroup"] label [data-testid="stMarkdownContainer"] {width: 100% !important;}
        [data-testid="stSidebar"] [role="radiogroup"] {gap: 0;}
        [data-testid="stSidebar"] [role="radiogroup"] label {
            min-height: 52px;
            padding: 0 18px !important;
            margin: 0 !important;
            border-radius: 10px !important;
            display: flex !important;
            align-items: center !important;
            border: 1px solid transparent;
            color: #94a3b8 !important;
            box-sizing: border-box;
            transition: transform .18s ease, border-color .18s ease, background .18s ease;
        }
        [data-testid="stSidebar"] [role="radiogroup"] label:hover {background: rgba(255,255,255,.045); color: #eaf2ff !important; border-color: rgba(148,163,184,.14);}
        [data-testid="stSidebar"] [role="radiogroup"] label:active {transform: translateY(1px) scale(.99);}
        [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
            background: linear-gradient(90deg, rgba(242,106,33,.20), rgba(242,106,33,.055));
            border-color: rgba(242,106,33,.34);
            color: var(--orange) !important;
            font-weight: 900;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
        }
        [data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display: none !important;}
        [data-testid="stSidebar"] [role="radiogroup"] p {font-size: 1.02rem; font-weight: 800; width: 100%;}
        [data-testid="stSidebar"] [data-baseweb="select"] > div,
        [data-testid="stSidebar"] [data-baseweb="input"] > div,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: var(--panel-2) !important;
            color: #eaf2ff !important;
            border: 1px solid var(--line) !important;
            border-radius: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
        }
        [data-baseweb="select"] span,
        [data-baseweb="select"] div,
        [data-baseweb="input"] input {color: #eaf2ff !important;}
        [data-baseweb="popover"] ul, [data-baseweb="menu"] {
            background: var(--panel) !important;
            border: 1px solid var(--line) !important;
            color: #eaf2ff !important;
        }
        [data-baseweb="menu"] li, [role="option"] {background: var(--panel) !important; color: #eaf2ff !important;}
        [data-baseweb="menu"] li:hover, [role="option"]:hover, [aria-selected="true"] {
            background: rgba(242,106,33,.16) !important;
            color: #ffffff !important;
        }
        [data-testid="stSidebar"] [data-baseweb="tag"] {background: var(--orange); color: white;}
        .topbar {
            height: 76px;
            border-bottom: 1px solid rgba(148,163,184,.16);
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 0 -1.75rem 1.65rem;
            padding: 0 1.75rem;
            background: rgba(7,13,22,.82);
            backdrop-filter: blur(18px);
            position: sticky;
            top: 0;
            z-index: 5;
        }
        .topbar-title {font-size: 1.12rem; font-weight: 900; letter-spacing: .08em; color: white;}
        .topbar-pill {
            border: 1px solid var(--line);
            border-radius: 999px;
            padding: 8px 12px;
            color: var(--muted);
            font-weight: 800;
            font-size: .78rem;
            background: rgba(16,29,47,.82);
        }
        .brand-block {display: flex; align-items: center; gap: 14px; padding: 4px 26px 22px;}
        .brand-block img {width: 58px; height: 58px; object-fit: contain; border-radius: 999px; filter: drop-shadow(0 8px 18px rgba(0,0,0,.35));}
        .brand-main {color: var(--orange); font-size: 1.58rem; line-height: .96; font-weight: 900;}
        .brand-sub {color: #94a3b8; font-size: .82rem; margin-top: 5px;}
        .refresh-status {
            margin: -8px 28px 18px;
            padding: 12px 14px;
            border: 1px solid var(--line);
            border-radius: 10px;
            background: rgba(16,29,47,.72);
            color: var(--muted);
            font-size: .76rem;
            line-height: 1.45;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.025);
        }
        .refresh-status strong {
            display: block;
            color: #eaf2ff;
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: .06em;
            margin-bottom: 4px;
        }
        .refresh-row {display: flex; justify-content: space-between; gap: 10px;}
        .refresh-row span:last-child {color: #ffffff; font-weight: 800; text-align: right;}
        .apply-button {
            background: linear-gradient(135deg, #d95f22, #f08a42);
            color: white;
            border-radius: 10px;
            text-align: center;
            padding: 12px 16px;
            font-weight: 800;
            margin: 2px 28px 24px;
            box-shadow: 0 14px 24px rgba(242,106,33,.14);
        }
        .sidebar-search-label {
            color: #eaf2ff;
            font-weight: 900;
            letter-spacing: .08em;
            text-transform: uppercase;
            font-size: .75rem;
            margin: 0 28px 8px;
        }
        .hero-card {
            position: relative;
            overflow: hidden;
            min-height: 194px;
            border: 1px solid var(--line);
            background:
                linear-gradient(100deg, rgba(16,29,47,.98), rgba(16,29,47,.80)),
                repeating-linear-gradient(120deg, rgba(148,163,184,.045) 0 1px, transparent 1px 18px);
            border-radius: 12px;
            padding: 40px 40px;
            margin-bottom: 28px;
            box-shadow: var(--shadow);
        }
        .hero-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 6px;
            background: linear-gradient(180deg, var(--orange), var(--orange-soft));
            opacity: .95;
        }
        .hero-card:after {
            content: "";
            position: absolute;
            right: -80px;
            bottom: -110px;
            width: 520px;
            height: 280px;
            border: 2px solid rgba(148,163,184,.14);
            border-radius: 50%;
            transform: rotate(-8deg);
        }
        .hero-title {
            font-size: clamp(2.05rem, 5vw, 3.05rem);
            line-height: 1.02;
            font-weight: 900;
            color: #dbeafe;
            letter-spacing: 0;
            text-shadow: 0 2px 0 rgba(0,0,0,.36);
        }
        .hero-copy {color: #c8d3e5; font-size: 1.04rem; line-height: 1.55; max-width: 760px; margin-top: 16px;}
        .section-title {font-size: 1.24rem; font-weight: 900; color: #eaf2ff; margin: 0;}
        .section-card {
            border: 1px solid var(--line);
            border-radius: 12px;
            background: linear-gradient(180deg, rgba(16,29,47,.98), rgba(13,27,44,.98));
            overflow: hidden;
            margin-bottom: 24px;
            box-shadow: var(--shadow);
        }
        .section-card-header {
            padding: 18px 22px;
            border-bottom: 1px solid var(--line);
            display: flex;
            align-items: center;
            justify-content: space-between;
            background: rgba(255,255,255,.018);
            gap: 18px;
        }
        .metric-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(178px, 1fr)); gap: 16px; margin-bottom: 28px;}
        .metric-card {
            min-height: 132px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background:
                linear-gradient(135deg, rgba(255,255,255,.035), transparent 34%),
                linear-gradient(180deg, rgba(21,36,58,.98), rgba(17,31,50,.98));
            padding: 18px 18px 16px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,.04), 0 14px 34px rgba(0,0,0,.12);
            position: relative;
            overflow: hidden;
            transition: transform .22s cubic-bezier(.16,1,.3,1), border-color .18s ease, background .18s ease;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            min-width: 0;
        }
        .metric-card:before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 3px;
            background: linear-gradient(90deg, var(--orange), rgba(243,154,74,.55));
            opacity: .8;
        }
        .metric-card:hover {transform: translateY(-2px); border-color: rgba(242,106,33,.42);}
        .metric-label {color: #dbe4f4; opacity: .88; font-weight: 900; letter-spacing: .06em; text-transform: uppercase; font-size: .74rem; line-height: 1.15; overflow-wrap: anywhere;}
        .metric-value {
            color: #f8fbff;
            font-size: clamp(1.32rem, 2.25vw, 2.05rem);
            font-weight: 900;
            margin-top: 18px;
            text-shadow: 0 2px 0 rgba(0,0,0,.35);
            font-family: 'JetBrains Mono', 'Outfit', monospace;
            line-height: 1.06;
            overflow-wrap: anywhere;
            word-break: break-word;
            max-width: 100%;
        }
        .metric-card:after {
            content: "";
            position: absolute;
            right: -38px;
            bottom: -54px;
            width: 138px;
            height: 138px;
            border: 1px solid rgba(148,163,184,.14);
            border-radius: 50%;
            pointer-events: none;
        }
        .metric-value.metric-long {font-size: clamp(1.02rem, 1.65vw, 1.42rem); letter-spacing: 0;}
        .metric-value.metric-xl {font-size: clamp(.9rem, 1.35vw, 1.12rem); letter-spacing: 0;}
        .metric-value img {max-width: 100%; flex-shrink: 0;}
        .match-logo-value {display:flex; align-items:center; justify-content:center; min-height:72px;}
        .match-logo-value img {width:86px; height:86px; object-fit:contain;}
        .match-fixture-value {display:flex; align-items:center; justify-content:center; gap:12px; min-width:0;}
        .match-fixture-value img {width:74px; height:74px; object-fit:contain; min-width:0;}
        .match-fixture-value span {color:#9fb0c8; font-size:.82rem; font-weight:900; letter-spacing:.04em;}
        .metric-card:has(.match-logo-value), .metric-card:has(.match-fixture-value) {
            align-items: center;
            text-align: center;
        }
        .identity-card {
            display: flex;
            align-items: center;
            gap: 18px;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: linear-gradient(135deg, var(--panel), var(--panel-2));
            padding: 22px 24px;
            margin-bottom: 22px;
            box-shadow: var(--shadow);
        }
        .identity-card img {width: 74px; height: 74px; object-fit: contain;}
        .identity-card img.player-headshot {object-fit: cover; object-position: top center; border-radius: 999px; background: #26364b;}
        .identity-kicker {color: var(--orange); text-transform: uppercase; font-weight: 900; letter-spacing: .08em; font-size: .76rem;}
        .identity-title {font-size: 1.55rem; font-weight: 900; color: white;}
        .dark-table, .overview-table {width: 100%; border-collapse: collapse; color: #eaf2ff;}
        .dark-table {font-size: .92rem;}
        .dark-table th, .overview-table th {
            background: var(--panel-3);
            color: #c3ccda;
            text-transform: uppercase;
            letter-spacing: .06em;
            font-size: .74rem;
            padding: 14px 16px;
            text-align: left;
        }
        .overview-table th {padding: 16px 24px;}
        .dark-table td, .overview-table td {
            border-top: 1px solid var(--line);
            background: #111f32;
        }
        .dark-table td {padding: 14px 16px;}
        .overview-table td {padding: 18px 24px; font-size: 1rem;}
        .dark-table tr:nth-child(even) td, .overview-table tr:nth-child(even) td {background: #14243a;}
        .dark-table tbody tr:hover td, .overview-table tbody tr:hover td {background: #1b2d46;}
        .js-plotly-plot .gtitle,
        .js-plotly-plot .xtitle,
        .js-plotly-plot .ytitle,
        .js-plotly-plot .legendtitletext,
        .js-plotly-plot .annotation-text {fill: #eaf2ff !important; color: #eaf2ff !important;}
        .js-plotly-plot .xtick text,
        .js-plotly-plot .ytick text,
        .js-plotly-plot .legendtext {fill: #cbd5e1 !important; color: #cbd5e1 !important;}
        .top-player-row {
            display: grid;
            grid-template-columns: 52px 1fr auto;
            gap: 12px;
            align-items: center;
            padding: 18px 20px;
            border-top: 1px solid var(--line);
            background: #122033;
        }
        .top-player-row:hover {background: #1a2b43;}
        .avatar {
            width: 44px;
            height: 44px;
            border-radius: 999px;
            background: linear-gradient(135deg, #2b3d56, #0f172a);
            border: 1px solid #3d516d;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 900;
            color: #dbeafe;
        }
        .player-main {font-weight: 900; color: #dbeafe; font-size: 1.05rem;}
        .player-sub {color: #cbd5e1; font-size: .86rem; margin-top: 2px;}
        .player-score {font-size: 1.28rem; color: #ffd9c2; font-weight: 900; font-family: 'JetBrains Mono', monospace;}
        .player-photo {
            width: 44px;
            height: 44px;
            border-radius: 999px;
            object-fit: cover;
            object-position: top center;
            border: 1px solid #3d516d;
            background: #26364b;
        }
        .team-cell {display: flex; align-items: center; gap: 12px; font-weight: 800;}
        .team-cell img {width: 34px; height: 34px; object-fit: contain;}
        .highlight-good {background: rgba(30,182,160,.22) !important; color: #dffcf7 !important; font-weight: 900;}
        .stButton button, [data-testid="stDownloadButton"] button {
            border-radius: 10px !important;
            min-height: 42px;
            font-weight: 800 !important;
            transition: transform .18s cubic-bezier(.16,1,.3,1), border-color .18s ease, background .18s ease !important;
        }
        .stButton button:active, [data-testid="stDownloadButton"] button:active {transform: translateY(1px) scale(.99);}
        [data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 6px;
            border: 1px solid var(--line);
            border-radius: 12px;
            padding: 6px;
            background: rgba(16,29,47,.72);
        }
        [data-testid="stTabs"] [data-baseweb="tab"] {
            border-radius: 9px;
            padding: 10px 14px;
            color: var(--muted);
            font-weight: 800;
        }
        [data-testid="stTabs"] [aria-selected="true"] {
            background: rgba(242,106,33,.14);
            color: #f7c6a5 !important;
        }
        [data-testid="stExpander"] {
            border: 1px solid var(--line) !important;
            border-radius: 12px !important;
            overflow: hidden;
            background: rgba(16,29,47,.84) !important;
        }
        [data-testid="stAlert"] {
            border-radius: 12px;
            border: 1px solid var(--line);
        }
        @media (max-width: 1100px) {
            .metric-grid {grid-template-columns: repeat(2, minmax(160px, 1fr));}
            .hero-card {padding: 34px 30px;}
        }
        @media (min-width: 1180px) {
            .metric-grid {grid-template-columns: repeat(5, minmax(0, 1fr));}
        }
        @media (max-width: 760px) {
            .block-container {padding: 0 1rem 2rem;}
            .topbar {height: auto; min-height: 66px; margin: 0 -1rem 1.25rem; padding: 14px 1rem; align-items: flex-start; gap: 10px; flex-direction: column;}
            .topbar-title {font-size: .96rem;}
            .hero-card {padding: 28px 22px; min-height: 0;}
            .section-card-header {align-items: flex-start; flex-direction: column;}
            .identity-card {align-items: flex-start; flex-direction: column;}
        }
        @media (max-width: 620px) {
            .metric-grid {grid-template-columns: 1fr; gap: 12px;}
            .metric-card{min-height:116px;}
            .match-fixture-value img{width:58px;height:58px;}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if light_overrides:
        st.markdown(f"<style>{light_overrides}</style>", unsafe_allow_html=True)


def metric_cards(items: list[tuple[str, str]]) -> None:
    def value_class(value: str) -> str:
        text_value = re.sub(r"<[^>]+>", "", str(value)).strip()
        if len(text_value) > 24:
            return "metric-xl"
        if len(text_value) > 12:
            return "metric-long"
        return ""

    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value {value_class(value)}">{html.escape(value)}</div></div>'
        for label, value in items
    )
    st.html(f"<div class='metric-grid'>{cards}</div>")


def match_metric_cards(items: list[tuple[str, str]]) -> None:
    def value_class(value: str) -> str:
        text_value = re.sub(r"<[^>]+>", "", str(value)).strip()
        if len(text_value) > 24:
            return "metric-xl"
        if len(text_value) > 12:
            return "metric-long"
        return ""

    cards = "".join(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value {value_class(value)}">{value}</div></div>'
        for label, value in items
    )
    st.html(f"<div class='metric-grid'>{cards}</div>")


def fixture_logo_html(home_code: str, away_code: str) -> str:
    return (
        '<div class="match-fixture-value">'
        f'<img src="{team_logo_uri(home_code)}" alt="{html.escape(home_code)}" />'
        '<span>VS</span>'
        f'<img src="{team_logo_uri(away_code)}" alt="{html.escape(away_code)}" />'
        "</div>"
    )


def winner_logo_html(winner_code: str) -> str:
    return (
        '<div class="match-logo-value">'
        f'<img src="{team_logo_uri(winner_code)}" alt="{html.escape(winner_code)}" />'
        "</div>"
    )


def overview_leaderboard_html(board: pd.DataFrame) -> str:
    rows = []
    for _, row in board.iterrows():
        logo = team_logo_uri(str(row["Code"]))
        rows.append(
            f"""
            <tr>
                <td>{row["RK"]}</td>
                <td><div class="team-cell"><img src="{logo}" /><span>{html.escape(str(row["Team"]))}</span></div></td>
                <td>{row["W"]}</td>
                <td>{row["L"]}</td>
                <td>{float(row["WIN %"]):.1f}%</td>
                <td>{float(row["PIR"]):.1f}</td>
            </tr>
            """
        )
    return (
        '<table class="overview-table"><thead><tr><th>RK</th><th>Team</th><th>W</th><th>L</th><th>Win %</th><th>PIR</th></tr></thead><tbody>'
        + "".join(rows)
        + "</tbody></table>"
    )


def top_players_html(players_df: pd.DataFrame, profiles_df: pd.DataFrame) -> str:
    if players_df.empty:
        return '<div class="top-player-row"><div class="avatar">-</div><div><div class="player-main">No player data</div><div class="player-sub">Try a broader filter set</div></div><div class="player-score">-</div></div>'
    rows = []
    for row in players_df.itertuples(index=False):
        initials = "".join(part[:1] for part in str(row.Player).replace(",", " ").split()[:2]).upper()
        photo = player_photo_uri_for_name(str(row.player_id), str(row.Player), profiles_df) if hasattr(row, "player_id") else ""
        visual = f'<img class="player-photo" src="{photo}" />' if photo else f'<div class="avatar">{html.escape(initials)}</div>'
        rows.append(
            f"""
            <div class="top-player-row">
                {visual}
                <div>
                    <div class="player-main">{html.escape(str(row.Player))}</div>
                    <div class="player-sub">{html.escape(str(row.Team))}</div>
                </div>
                <div class="player-score">{float(row.PIR):.1f}</div>
            </div>
            """
        )
    return "".join(rows)


def dark_html_table(
    df: pd.DataFrame,
    max_rows: int = 20,
    logo_col: str | None = None,
    player_col: str | None = None,
    highlight: dict[str, str] | None = None,
) -> str:
    if df.empty:
        return '<div class="section-card"><div class="section-card-header">No data</div></div>'
    view = df.head(max_rows).copy()
    highlight = highlight or {}
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in view.columns if col != "player_id")
    rows = []
    for _, row in view.iterrows():
        cells = []
        for col in view.columns:
            if col == "player_id":
                continue
            value = row[col]
            css = highlight.get(f"{row.name}:{col}", "")
            if col == logo_col:
                logo = team_logo_uri(str(row.get("Code", "")))
                cell = f'<td class="{css}"><div class="team-cell"><img src="{logo}" /><span>{html.escape(str(value))}</span></div></td>'
            elif col == player_col:
                player_id = str(row.get("player_id", ""))
                if str(row.get("Role", "")).lower() == "coach" or player_id.startswith("COACH-"):
                    photo = coach_photo_uri(str(value), str(row.get("Team Code", "")))
                else:
                    photo = player_photo_uri_for_name(player_id, str(value), profiles)
                initials = "".join(part[:1] for part in str(value).replace(",", " ").split()[:2]).upper()
                visual = f'<img class="player-photo" src="{photo}" />' if photo else f'<div class="avatar">{html.escape(initials)}</div>'
                cell = f'<td class="{css}"><div class="team-cell">{visual}<span>{html.escape(str(value))}</span></div></td>'
            else:
                if str(col).lower() in {"season", "lastseason"}:
                    text = season_label(value)
                else:
                    text = f"{value:.2f}" if isinstance(value, float) else str(value)
                cell = f'<td class="{css}">{html.escape(text)}</td>'
            cells.append(cell)
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="dark-table"><thead><tr>{headers}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def render_html_table(df: pd.DataFrame, height: int = 460, **kwargs) -> None:
    if is_light_theme():
        table_tokens = {
            "scroll_bg": "linear-gradient(180deg, #ffffff, #f7f9fc)",
            "scroll_border": "#d7deea",
            "scrollbar_track": "#eef2f7",
            "scrollbar_thumb": "#c9d3e1",
            "scrollbar_hover": "#dc5f1f",
            "text": "#182033",
            "head_bg": "#e9eef6",
            "head_text": "#536176",
            "cell_bg": "#ffffff",
            "cell_alt_bg": "#f7f9fc",
            "cell_hover_bg": "#eef4fb",
            "cell_border": "#d7deea",
            "avatar_bg": "linear-gradient(135deg, #e9eef6, #ffffff)",
            "avatar_border": "#cbd5e1",
            "avatar_text": "#182033",
            "photo_bg": "#ffffff",
            "highlight_bg": "rgba(19,138,118,.14)",
            "highlight_text": "#0f766e",
            "shadow": "inset 0 1px 0 rgba(255,255,255,.72), 0 10px 28px rgba(24,32,51,.06)",
        }
    else:
        table_tokens = {
            "scroll_bg": "linear-gradient(180deg, #101d2f, #0f1b2c)",
            "scroll_border": "#2b3a50",
            "scrollbar_track": "#101d2f",
            "scrollbar_thumb": "#334155",
            "scrollbar_hover": "#f26a21",
            "text": "#eaf2ff",
            "head_bg": "#1b2d47",
            "head_text": "#c3ccda",
            "cell_bg": "#111f32",
            "cell_alt_bg": "#14243a",
            "cell_hover_bg": "#1b2d46",
            "cell_border": "#2b3a50",
            "avatar_bg": "linear-gradient(135deg, #2b3a50, #0f172a)",
            "avatar_border": "#3d516d",
            "avatar_text": "#dbeafe",
            "photo_bg": "#26364b",
            "highlight_bg": "rgba(30,182,160,.22)",
            "highlight_text": "#dffcf7",
            "shadow": "inset 0 1px 0 rgba(255,255,255,.025)",
        }
    html_doc = f"""
    <style>
    .table-scroll {{ max-height: {height}px; overflow: auto; background: {table_tokens["scroll_bg"]}; border: 1px solid {table_tokens["scroll_border"]}; border-radius: 12px; scrollbar-color: {table_tokens["scrollbar_thumb"]} {table_tokens["scrollbar_track"]}; scrollbar-width: thin; box-shadow: {table_tokens["shadow"]}; }}
    .table-scroll::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    .table-scroll::-webkit-scrollbar-track {{ background: {table_tokens["scrollbar_track"]}; }}
    .table-scroll::-webkit-scrollbar-thumb {{ background: {table_tokens["scrollbar_thumb"]}; border-radius: 999px; border: 2px solid {table_tokens["scrollbar_track"]}; }}
    .table-scroll::-webkit-scrollbar-thumb:hover {{ background: {table_tokens["scrollbar_hover"]}; }}
    .dark-table {{ width: 100%; border-collapse: collapse; color: {table_tokens["text"]}; font-size: 14px; }}
    .dark-table th {{ background: {table_tokens["head_bg"]}; color: {table_tokens["head_text"]}; text-transform: uppercase; letter-spacing: .06em; font-size: 11px; padding: 14px 16px; text-align: left; position: sticky; top: 0; z-index: 2; }}
    .dark-table td {{ padding: 14px 16px; border-top: 1px solid {table_tokens["cell_border"]}; background: {table_tokens["cell_bg"]}; }}
    .dark-table tr:nth-child(even) td {{ background: {table_tokens["cell_alt_bg"]}; }}
    .dark-table tbody tr:hover td {{ background: {table_tokens["cell_hover_bg"]}; }}
    .team-cell {{ display: flex; align-items: center; gap: 12px; font-weight: 800; }}
    .team-cell img {{ width: 34px; height: 34px; object-fit: contain; }}
    .team-cell .player-photo {{ width: 36px; height: 36px; border-radius: 999px; object-fit: cover; object-position: top center; border: 1px solid {table_tokens["avatar_border"]}; background: {table_tokens["photo_bg"]}; }}
    .team-cell .avatar {{ width: 36px; height: 36px; border-radius: 999px; background: {table_tokens["avatar_bg"]}; border: 1px solid {table_tokens["avatar_border"]}; display: flex; align-items: center; justify-content: center; font-size: .78rem; font-weight: 900; color: {table_tokens["avatar_text"]}; }}
    .highlight-good {{ background: {table_tokens["highlight_bg"]} !important; color: {table_tokens["highlight_text"]} !important; font-weight: 900; }}
    </style>
    <div class="table-scroll">{dark_html_table(df, **kwargs)}</div>
    """
    st.html(html_doc)


def render_fantasy_lineup_court(lineup: pd.DataFrame, summary: dict, profiles_df: pd.DataFrame, title: str) -> None:
    if lineup.empty:
        return

    def starter_position_class(role: str, idx: int, count: int) -> str:
        role_key = str(role).lower()
        if role_key == "center":
            positions = {
                1: ["pos-c"],
                2: ["pos-cl", "pos-cr"],
                3: ["pos-cl", "pos-c", "pos-cr"],
            }
        elif role_key == "forward":
            positions = {
                1: ["pos-fc"],
                2: ["pos-fl", "pos-fr"],
                3: ["pos-fl", "pos-fc", "pos-fr"],
            }
        else:
            positions = {
                1: ["pos-gc"],
                2: ["pos-gl", "pos-gr"],
                3: ["pos-gl", "pos-gc", "pos-gr"],
            }
        selected = positions.get(min(max(count, 1), 3), positions[1])
        return selected[min(idx, len(selected) - 1)]

    def player_card(row: pd.Series, css_class: str = "") -> str:
        is_coach = str(row.get("Role", "")) == "Coach"
        player_id = str(row.get("player_id", ""))
        name = str(row.get("Player", ""))
        photo = coach_photo_uri(name, str(row.get("Team Code", ""))) if is_coach else player_photo_uri_for_name(player_id, name, profiles_df)
        initials = "".join(part[:1] for part in name.replace(",", " ").split()[:2]).upper() or name[:1].upper()
        visual = (
            f'<img class="fantasy-photo" src="{photo}" />'
            if photo
            else f'<div class="fantasy-avatar">{html.escape(initials)}</div>'
        )
        slot = str(row.get("Slot", ""))
        badge = "C" if slot == "Captain" else "6" if slot == "6th Man" else "HC" if is_coach else ""
        badge_html = f'<span class="fantasy-badge">{badge}</span>' if badge else ""
        team = str(row.get("Team", ""))
        credits = float(row.get("Credits", 0) or 0)
        projection = float(row.get("Projection", 0) or 0)
        return f"""
        <div class="fantasy-card {css_class}">
            {badge_html}
            <div class="fantasy-visual">{visual}</div>
            <div class="fantasy-name">{html.escape(name)}</div>
            <div class="fantasy-meta"><span>{html.escape(str(row.get("Role", ""))[:1])}</span><span>{credits:.1f} CR</span></div>
            <div class="fantasy-sub">{html.escape(team)}</div>
            <div class="fantasy-proj">{projection:.1f} PIR</div>
        </div>
        """

    starters = lineup[lineup["Slot"].isin(["Captain", "Starter"])].copy()
    sixth = lineup[lineup["Slot"].eq("6th Man")].head(1)
    bench = lineup[lineup["Slot"].eq("Bench")].head(4)
    coach = lineup[lineup["Slot"].eq("Coach")].head(1)

    starter_rows = []
    role_order = ["Center", "Forward", "Guard"]
    for role in role_order:
        role_rows = starters[starters["Role"].astype(str) == role].sort_values("Projection", ascending=False)
        cards = []
        for idx, (_, row) in enumerate(role_rows.iterrows()):
            cards.append(player_card(row, starter_position_class(role, idx, len(role_rows))))
        label = "Centers" if role == "Center" else "Forwards" if role == "Forward" else "Guards"
        starter_rows.append(
            f'<div class="fantasy-role-row fantasy-role-{role.lower()}"><div class="fantasy-lane-label">{label}</div><div class="fantasy-role-cards">{"".join(cards)}</div></div>'
        )
    starter_rows_html = "".join(starter_rows)
    sixth_cards = "".join(player_card(row, "sixth-card") for _, row in sixth.iterrows())
    bench_cards = "".join(player_card(row, "bench-card") for _, row in bench.iterrows())
    coach_cards = "".join(player_card(row, "coach-card") for _, row in coach.iterrows())
    coach_note = "Coach credits from CSV" if summary.get("Coach Source") == "CSV" else "Coach placeholder until coach credits are loaded"

    html_doc = f"""
    <style>
    .fantasy-wrap {{
        background: radial-gradient(circle at 50% 0%, rgba(92,24,125,.78), #25003d 58%, #170027);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 16px;
        padding: 18px;
        color: #fff;
        box-shadow: 0 22px 54px rgba(0,0,0,.34);
    }}
    .fantasy-top {{
        display: grid;
        grid-template-columns: 1fr auto auto;
        gap: 14px;
        align-items: center;
        margin-bottom: 12px;
        font-family: Outfit, Arial, sans-serif;
    }}
    .fantasy-title {{ font-size: 1.05rem; font-weight: 900; text-transform: uppercase; }}
    .fantasy-pill {{
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 8px;
        padding: 8px 11px;
        font-weight: 900;
        text-align: center;
        min-width: 82px;
    }}
    .fantasy-pill span {{ display:block; color:#c7aed8; font-size:.68rem; text-transform:uppercase; margin-top:2px; }}
    .fantasy-court {{
        position: relative;
        min-height: 520px;
        border-radius: 18px;
        overflow: visible;
        background:
            radial-gradient(ellipse at top, rgba(255,255,255,.42), transparent 31%),
            repeating-linear-gradient(90deg, rgba(255,255,255,.10) 0 18px, rgba(255,255,255,.03) 18px 44px),
            linear-gradient(135deg, #f4c46f, #e39a43);
        border: 2px solid rgba(255,255,255,.35);
        display: grid;
        grid-template-rows: 1fr 1fr 1fr;
        gap: 14px;
        padding: 38px 36px 28px;
    }}
    .fantasy-court::before {{
        content:"";
        position:absolute;
        left:20%;
        right:20%;
        top:-18%;
        height:52%;
        border: 3px solid rgba(255,255,255,.56);
        border-bottom-left-radius: 50% 80%;
        border-bottom-right-radius: 50% 80%;
    }}
    .fantasy-card {{
        width: min(156px, 100%);
        min-height: 166px;
        position: relative;
        text-align: center;
        font-family: Outfit, Arial, sans-serif;
        filter: drop-shadow(0 12px 16px rgba(33,0,53,.34));
        z-index: 1;
    }}
    .fantasy-visual {{ height: 82px; display:flex; align-items:end; justify-content:center; }}
    .fantasy-photo {{
        width: 82px;
        height: 82px;
        border-radius: 999px;
        object-fit: cover;
        object-position: top center;
        background: #e7edf5;
        border: 2px solid #fff;
    }}
    .fantasy-avatar {{
        width: 82px;
        height: 82px;
        border-radius: 999px;
        background: linear-gradient(135deg, #4c1d95, #111827);
        border: 2px solid #fff;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight: 1000;
        font-size: 1rem;
    }}
    .fantasy-name {{
        background: #3b075a;
        color: #fff;
        font-size: .86rem;
        line-height: 1.06;
        font-weight: 900;
        padding: 7px 8px;
        min-height: 42px;
        display:flex;
        align-items:center;
        justify-content:center;
    }}
    .fantasy-meta {{
        background: #fff;
        color: #2d0a3c;
        display:flex;
        justify-content:space-between;
        gap: 4px;
        padding: 4px 7px 1px;
        font-size: .82rem;
        font-weight: 900;
    }}
    .fantasy-sub {{
        background: #f0f6ff;
        color: #6b7280;
        font-size: .72rem;
        font-weight: 800;
        padding: 0 6px 4px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    .fantasy-proj {{ color:#16c784; font-size:.82rem; font-weight:900; margin-top:5px; text-shadow:0 1px 0 rgba(0,0,0,.16); }}
    .fantasy-badge {{
        position:absolute;
        top: 64px;
        right: 10px;
        z-index: 2;
        width: 21px;
        height: 21px;
        border-radius: 999px;
        background: #d8ff2f;
        color: #3b075a;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:.68rem;
        font-weight:1000;
        border:1px solid #fff;
    }}
    .fantasy-role-row {{
        position: relative;
        display: grid;
        grid-template-columns: 92px 1fr;
        gap: 14px;
        align-items: center;
        z-index: 1;
    }}
    .fantasy-lane-label {{
        color: rgba(45,10,60,.62);
        font-size: .74rem;
        font-weight: 1000;
        letter-spacing: .08em;
        text-transform: uppercase;
        writing-mode: vertical-rl;
        transform: rotate(180deg);
        justify-self: center;
    }}
    .fantasy-role-cards {{
        display: flex;
        justify-content: center;
        align-items: end;
        gap: clamp(14px, 4vw, 48px);
        min-width: 0;
    }}
    .fantasy-lower {{
        display:grid;
        grid-template-columns: minmax(110px, .8fr) 1.2fr;
        gap: 12px;
        margin-top: 14px;
    }}
    .fantasy-bench, .fantasy-coach {{
        background: rgba(255,255,255,.08);
        border: 1px solid rgba(255,255,255,.13);
        border-radius: 12px;
        padding: 12px;
    }}
    .fantasy-section-label {{
        color:#c7aed8;
        text-transform:uppercase;
        font-size:.72rem;
        font-weight:900;
        text-align:center;
        margin-bottom:8px;
    }}
    .fantasy-bench-grid {{ display:grid; grid-template-columns: repeat(4, minmax(126px, 1fr)); gap: 12px; }}
    .fantasy-bench-grid .fantasy-card, .fantasy-coach .fantasy-card {{
        position:relative;
        left:auto;
        top:auto;
        transform:none;
        width:100%;
        margin:auto;
    }}
    .fantasy-sixth {{ margin-bottom: 12px; display:flex; justify-content:center; }}
    .fantasy-note {{ color:#c7aed8; font-size:.72rem; text-align:center; margin-top:7px; }}
    @media (max-width: 900px) {{
        .fantasy-court {{ min-height: auto; padding: 26px 12px; }}
        .fantasy-role-row {{ grid-template-columns: 1fr; gap: 8px; }}
        .fantasy-lane-label {{ writing-mode: horizontal-tb; transform:none; }}
        .fantasy-card {{ width: min(150px, 100%); }}
        .fantasy-role-cards {{ gap: 10px; flex-wrap: wrap; }}
        .fantasy-lower {{ grid-template-columns:1fr; }}
        .fantasy-bench-grid {{ grid-template-columns: repeat(2, 1fr); }}
    }}
    </style>
    <div class="fantasy-wrap">
        <div class="fantasy-top">
            <div class="fantasy-title">{html.escape(title)}</div>
            <div class="fantasy-pill">{float(summary.get("Credits", 0)):.1f}/{float(summary.get("Cap", 0)):.0f}<span>Credits</span></div>
            <div class="fantasy-pill">{float(summary.get("Projected", 0)):.1f}<span>Projected</span></div>
            <div class="fantasy-pill">{html.escape(str(summary.get("Formation", "-")))}<span>Formation</span></div>
            <div class="fantasy-pill">{html.escape(str(summary.get("Optimizer", "Heuristic")))}<span>Optimizer</span></div>
        </div>
        <div class="fantasy-court">{starter_rows_html}</div>
        <div class="fantasy-lower">
            <div class="fantasy-coach">
                <div class="fantasy-section-label">Head Coach</div>
                {coach_cards}
                <div class="fantasy-note">{html.escape(coach_note)}</div>
            </div>
            <div class="fantasy-bench">
                <div class="fantasy-section-label">6th Man (100% FPT)</div>
                <div class="fantasy-sixth">{sixth_cards}</div>
                <div class="fantasy-section-label">Bench (50% FPT)</div>
                <div class="fantasy-bench-grid">{bench_cards}</div>
            </div>
        </div>
    </div>
    """
    st.html(html_doc)


def stat_average(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    available = [col for col in cols if col in df.columns]
    if not available or df.empty:
        return pd.Series(dtype=float)
    return df[available].mean(numeric_only=True)


def average_minutes(df: pd.DataFrame) -> float:
    if df.empty or "minutes" not in df.columns:
        return 0.0
    return float(df["minutes"].map(minutes_to_float).mean())


def dunkest_row_for_player(player_name: str) -> pd.Series:
    dunkest = load_dunkest_player_stats()
    if dunkest.empty:
        return pd.Series(dtype=object)
    key = player_name_match_key(player_name)
    rows = dunkest[dunkest["dunkest_key"] == key]
    return rows.iloc[0] if not rows.empty else pd.Series(dtype=object)


def _candidate_pool_for_lineups(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or "Dunkest CR" not in predictions.columns:
        return pd.DataFrame()
    pool = predictions.copy()
    pool["Credits"] = pd.to_numeric(pool["Dunkest CR"], errors="coerce")
    projection_col = "Fantasy Score" if "Fantasy Score" in pool.columns else "Fantasy PIR" if "Fantasy PIR" in pool.columns else "Expected PIR"
    pool["Projection"] = pd.to_numeric(pool[projection_col], errors="coerce")
    pool = pool[
        pool["Credits"].notna()
        & pool["Projection"].notna()
        & pool["Role"].isin(FANTASY_POSITION_TARGETS)
    ].copy()
    pool = pool.sort_values("Projection", ascending=False).drop_duplicates("player_id", keep="first")
    selected_indices = set()
    for role, group in pool.groupby("Role"):
        limit = FANTASY_OPTIMIZER_ROLE_LIMITS.get(str(role), 18)
        ranked = group.assign(value=group["Projection"] / group["Credits"].clip(lower=.1))
        selected_indices.update(ranked.sort_values("Projection", ascending=False).head(max(6, limit // 2)).index)
        selected_indices.update(ranked.sort_values("value", ascending=False).head(max(6, limit // 2)).index)
        selected_indices.update(ranked.sort_values("Credits", ascending=True).head(4).index)
    pool = pool.loc[sorted(selected_indices)].copy()
    pool["value"] = pool["Projection"] / pool["Credits"].clip(lower=.1)
    capped = []
    for role, group in pool.groupby("Role"):
        limit = FANTASY_OPTIMIZER_ROLE_LIMITS.get(str(role), 18)
        capped.append(group.sort_values(["Projection", "value"], ascending=[False, False]).head(limit))
    return pd.concat(capped, ignore_index=True).drop(columns=["value"], errors="ignore") if capped else pd.DataFrame()


def _best_assignment(roster: list[dict]) -> tuple[float, list[dict]]:
    best_score = -1.0
    best_rows: list[dict] = []
    by_role = {role: sorted([p for p in roster if p["Role"] == role], key=lambda p: p["Projection"], reverse=True) for role in FANTASY_POSITION_TARGETS}
    for guards, forwards, centers in FANTASY_FORMATIONS:
        counts = {"Guard": guards, "Forward": forwards, "Center": centers}
        if any(len(by_role[role]) < count for role, count in counts.items()):
            continue
        starters = []
        for role, count in counts.items():
            starters.extend(by_role[role][:count])
        starter_ids = {p["player_id"] for p in starters}
        remaining = sorted([p for p in roster if p["player_id"] not in starter_ids], key=lambda p: p["Projection"], reverse=True)
        if len(remaining) < 5:
            continue
        sixth = remaining[0]
        bench = remaining[1:]
        captain = max(starters, key=lambda p: p["Projection"])
        weighted_score = (
            sum(p["Projection"] for p in starters)
            + sixth["Projection"]
            + 0.5 * sum(p["Projection"] for p in bench)
            + (FANTASY_CAPTAIN_MULTIPLIER - 1.0) * captain["Projection"]
        )
        if weighted_score > best_score:
            assigned = []
            for p in starters:
                assigned.append({**p, "Slot": "Captain" if p["player_id"] == captain["player_id"] else "Starter", "Weight": FANTASY_CAPTAIN_MULTIPLIER if p["player_id"] == captain["player_id"] else 1.0})
            assigned.append({**sixth, "Slot": "6th Man", "Weight": 1.0})
            assigned.extend({**p, "Slot": "Bench", "Weight": 0.5} for p in bench)
            best_score = weighted_score
            best_rows = assigned
    return best_score, best_rows


def _team_projection_score(team_rows: pd.DataFrame, home_bonus: float = 0.0) -> float:
    if team_rows.empty:
        return home_bonus
    ordered = team_rows.sort_values(["parsed_date", "season", "game_code"]).tail(8).copy()
    point_diff = pd.to_numeric(ordered.get("point_diff"), errors="coerce").mean()
    pir = pd.to_numeric(ordered.get("pir"), errors="coerce").mean()
    pir_allowed = pd.to_numeric(ordered.get("pir_allowed"), errors="coerce").mean()
    win_rate = pd.to_numeric(ordered.get("won"), errors="coerce").mean()
    return (
        (0 if pd.isna(point_diff) else float(point_diff))
        + 0.15 * (0 if pd.isna(pir) or pd.isna(pir_allowed) else float(pir - pir_allowed))
        + 4.0 * (0 if pd.isna(win_rate) else float(win_rate))
        + home_bonus
    )


def projected_win_boosts(team_logs: pd.DataFrame, games_df: pd.DataFrame) -> dict[tuple[int, str], float]:
    boosts: dict[tuple[int, str], float] = {}
    if team_logs.empty or games_df.empty:
        return boosts
    for game in games_df.itertuples(index=False):
        game_code = int(game.game_code)
        home_code = str(game.home_code)
        away_code = str(game.away_code)
        home_rows = team_logs[team_logs["team_code"].astype(str) == home_code]
        away_rows = team_logs[team_logs["team_code"].astype(str) == away_code]
        home_score = _team_projection_score(home_rows, home_bonus=1.5)
        away_score = _team_projection_score(away_rows)
        if abs(home_score - away_score) < 0.1:
            continue
        projected_winner = home_code if home_score > away_score else away_code
        boosts[(game_code, projected_winner)] = FANTASY_TEAM_WIN_BOOST
    return boosts


def projected_team_margins(team_logs: pd.DataFrame, games_df: pd.DataFrame) -> dict[tuple[int, str], float]:
    margins: dict[tuple[int, str], float] = {}
    if team_logs.empty or games_df.empty:
        return margins
    for game in games_df.itertuples(index=False):
        game_code = int(game.game_code)
        home_code = str(game.home_code)
        away_code = str(game.away_code)
        home_rows = team_logs[team_logs["team_code"].astype(str) == home_code]
        away_rows = team_logs[team_logs["team_code"].astype(str) == away_code]
        home_score = _team_projection_score(home_rows, home_bonus=1.5)
        away_score = _team_projection_score(away_rows)
        projected_margin = home_score - away_score
        margins[(game_code, home_code)] = projected_margin
        margins[(game_code, away_code)] = -projected_margin
    return margins


def fantasy_coach_score_from_margin(margin: float) -> float:
    if margin >= 20:
        return 25.0
    if margin >= 11:
        return 20.0
    if margin >= 0:
        return 10.0
    if margin >= -10:
        return -5.0
    if margin >= -20:
        return -10.0
    return -20.0


def enrich_ml_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    out = predictions.copy()
    for col in ["Expected PIR", "Low", "High", "Pred MIN", "GP", "PIR", "H2H", "H2H GP", "Win Boost", "Dunkest CR"]:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    interval_width = (out["High"] - out["Low"]).clip(lower=0)
    interval_radius = interval_width / 2
    player_gp = out.get("GP", pd.Series(0, index=out.index)).fillna(0)
    h2h_gp = out.get("H2H GP", pd.Series(0, index=out.index)).fillna(0)
    pred_min = out.get("Pred MIN", pd.Series(0, index=out.index)).fillna(0)
    expected = out["Expected PIR"].fillna(0)
    floor = (expected - 0.55 * interval_radius).clip(lower=0)
    upside = expected + 0.35 * interval_radius

    sample_score = (player_gp.clip(0, 15) / 15 * 45) + (h2h_gp.clip(0, 4) / 4 * 15)
    minutes_score = (pred_min.clip(0, 28) / 28 * 25)
    stability_score = (15 - interval_radius.clip(0, 15)).clip(lower=0)
    confidence_score = (sample_score + minutes_score + stability_score).clip(0, 100)
    risk_penalty = interval_radius * (1.05 - confidence_score / 140).clip(lower=0.35, upper=1.05)
    safe_pir = (expected - 0.38 * risk_penalty).clip(lower=0)
    win_boost = out.get("Win Boost", pd.Series(1.0, index=out.index)).fillna(1.0)
    out["Floor PIR"] = floor
    out["Upside PIR"] = upside
    out["Confidence Score"] = confidence_score
    out["Safe PIR"] = safe_pir
    out["Fantasy Score"] = safe_pir * win_boost
    out["Value Score"] = out["Fantasy Score"] / out.get("Dunkest CR", pd.Series(pd.NA, index=out.index)).clip(lower=.1)
    out["Risk"] = pd.cut(
        confidence_score,
        bins=[-1, 44, 69, 100],
        labels=["High", "Medium", "Low"],
    ).astype(str)
    out["ML Note"] = ""
    out.loc[player_gp < 5, "ML Note"] = "Low sample"
    out.loc[(player_gp >= 5) & (pred_min < 12), "ML Note"] = "Minutes risk"
    out.loc[(out["ML Note"].eq("")) & (interval_radius > 8.5), "ML Note"] = "Wide range"
    out.loc[out["ML Note"].eq(""), "ML Note"] = "Stable"
    return out


def optimize_fantasy_lineup(predictions: pd.DataFrame, credit_cap: float) -> tuple[pd.DataFrame, dict]:
    pool = _candidate_pool_for_lineups(predictions)
    if pool.empty:
        return pd.DataFrame(), {}
    records = []
    for row in pool.itertuples(index=False):
        records.append(
            {
                "player_id": str(row.player_id),
                "Player": str(row.Player),
                "Team": str(row.Team),
                "Team Code": str(getattr(row, "team_code", "")),
                "Role": str(row.Role),
                "Credits": float(row.Credits),
                "Projection": float(row.Projection),
            }
        )
    role_order = {"Guard": 0, "Forward": 1, "Center": 2}
    target_counts = (4, 4, 2)
    cap_tenths = int(round(credit_cap * 10))
    beam = 2
    states: dict[tuple[tuple[int, int, int], int], list[tuple[float, tuple[int, ...]]]] = {((0, 0, 0), 0): [(0.0, tuple())]}
    for idx, rec in enumerate(records):
        role_idx = role_order[rec["Role"]]
        cost = int(round(rec["Credits"] * 10))
        new_states = {key: list(value) for key, value in states.items()}
        for (counts, budget), rosters in states.items():
            if counts[role_idx] >= target_counts[role_idx] or budget + cost > cap_tenths:
                continue
            new_counts = list(counts)
            new_counts[role_idx] += 1
            new_key = (tuple(new_counts), budget + cost)
            bucket = new_states.setdefault(new_key, [])
            for score, selected in rosters:
                bucket.append((score + rec["Projection"], selected + (idx,)))
        states = {}
        for key, rosters in new_states.items():
            dedup = {}
            for score, selected in rosters:
                previous = dedup.get(selected)
                if previous is None or score > previous:
                    dedup[selected] = score
            pruned = sorted(((score, selected) for selected, score in dedup.items()), reverse=True)[:beam]
            states[key] = pruned

    best_score = -1.0
    best_assignment: list[dict] = []
    best_budget = 0.0
    for (counts, budget), rosters in states.items():
        if counts != target_counts:
            continue
        for _, selected in rosters:
            roster = [records[i] for i in selected]
            score, assignment = _best_assignment(roster)
            if score > best_score:
                best_score = score
                best_assignment = assignment
                best_budget = budget / 10
    if not best_assignment:
        return pd.DataFrame(), {}
    lineup = pd.DataFrame(best_assignment)
    lineup["Weighted Projection"] = lineup["Projection"] * lineup["Weight"]
    lineup = lineup[["Slot", "player_id", "Player", "Team", "Role", "Credits", "Projection", "Weight", "Weighted Projection"]]
    slot_order = pd.Categorical(lineup["Slot"], categories=["Captain", "Starter", "6th Man", "Bench"], ordered=True)
    lineup = lineup.assign(_slot=slot_order).sort_values(["_slot", "Projection"], ascending=[True, False]).drop(columns="_slot").round(2)
    starter_counts = lineup[lineup["Slot"].isin(["Captain", "Starter"])]["Role"].value_counts()
    formation = f"{int(starter_counts.get('Guard', 0))}-{int(starter_counts.get('Forward', 0))}-{int(starter_counts.get('Center', 0))}"
    summary = {"Credits": round(best_budget, 1), "Projected": round(best_score, 2), "Cap": credit_cap, "Formation": formation}
    return lineup, summary


def coach_candidates_for_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame()
    coaches = load_fantasy_coaches()
    rows = []
    for team_code, group in predictions.groupby("team_code"):
        team_code = str(team_code)
        team_name = str(group["team_name"].dropna().iloc[0]) if "team_name" in group and not group["team_name"].dropna().empty else team_code
        projected_margin = pd.to_numeric(group.get("Projected Team Margin"), errors="coerce").mean()
        if pd.notna(projected_margin):
            projection = fantasy_coach_score_from_margin(float(projected_margin))
        else:
            win_boost = pd.to_numeric(group.get("Win Boost"), errors="coerce").max()
            projection = 10.0 if pd.notna(win_boost) and float(win_boost) > 1.0 else -5.0
            projected_margin = 1.0 if projection > 0 else -1.0
        coach_name = f"{team_name} Coach"
        credits = 0.0
        source = "Estimated"
        if not coaches.empty:
            match = coaches[coaches["Team Code"].astype(str).str.upper() == team_code.upper()]
            if not match.empty:
                coach_name = str(match.iloc[0].get("Coach", coach_name))
                parsed_credits = pd.to_numeric(pd.Series([match.iloc[0].get("Coach CR")]), errors="coerce").iloc[0]
                credits = float(parsed_credits) if pd.notna(parsed_credits) else 0.0
                source = "CSV"
        rows.append(
            {
                "Slot": "Coach",
                "player_id": f"COACH-{team_code}",
                "Player": coach_name,
                "Team": team_name,
                "Team Code": team_code,
                "Role": "Coach",
                "Credits": credits,
                "Projection": projection,
                "Projected Margin": round(float(projected_margin), 1),
                "Weight": 1.0,
                "Weighted Projection": projection,
                "Coach Source": source,
            }
        )
    return pd.DataFrame(rows).sort_values(["Projection", "Credits"], ascending=[False, True])


def optimize_fantasy_lineup_exact(predictions: pd.DataFrame, credit_cap: float) -> tuple[pd.DataFrame, dict]:
    pool = _candidate_pool_for_lineups(predictions)
    coaches = coach_candidates_for_predictions(predictions)
    if pool.empty or coaches.empty:
        return pd.DataFrame(), {}

    players = []
    for row in pool.itertuples(index=False):
        players.append(
            {
                "player_id": str(row.player_id),
                "Player": str(row.Player),
                "Team": str(row.Team),
                "Team Code": str(getattr(row, "team_code", "")),
                "Role": str(row.Role),
                "Credits": float(row.Credits),
                "Projection": float(row.Projection),
            }
        )
    coach_rows = [row.to_dict() for _, row in coaches.sort_values(["Projection", "Credits"], ascending=[False, True]).iterrows()]
    n_players = len(players)
    n_coaches = len(coach_rows)
    if n_players < 10 or n_coaches < 1:
        return pd.DataFrame(), {}

    selected_offset = 0
    starter_offset = n_players
    sixth_offset = 2 * n_players
    captain_offset = 3 * n_players
    coach_offset = 4 * n_players
    n_vars = 4 * n_players + n_coaches

    objective = np.zeros(n_vars)
    for idx, player in enumerate(players):
        projection = float(player["Projection"])
        objective[selected_offset + idx] = -0.5 * projection
        objective[starter_offset + idx] = -0.5 * projection
        objective[sixth_offset + idx] = -0.5 * projection
        objective[captain_offset + idx] = -(FANTASY_CAPTAIN_MULTIPLIER - 1.0) * projection
    for idx, coach in enumerate(coach_rows):
        objective[coach_offset + idx] = -float(coach.get("Projection", 0.0))

    best_solution = None
    best_fun = np.inf
    best_formation = ""
    for guards, forwards, centers in FANTASY_FORMATIONS:
        rows = []
        lower = []
        upper = []

        def add_constraint(coeffs: dict[int, float], lb: float, ub: float) -> None:
            rows.append(coeffs)
            lower.append(lb)
            upper.append(ub)

        add_constraint(
            {
                **{selected_offset + i: float(players[i]["Credits"]) for i in range(n_players)},
                **{coach_offset + i: float(coach_rows[i].get("Credits", 0.0)) for i in range(n_coaches)},
            },
            -np.inf,
            float(credit_cap),
        )
        add_constraint({selected_offset + i: 1.0 for i in range(n_players)}, 10.0, 10.0)
        for role, target in zip(["Guard", "Forward", "Center"], [4, 4, 2]):
            add_constraint({selected_offset + i: 1.0 for i, player in enumerate(players) if player["Role"] == role}, float(target), float(target))
        add_constraint({starter_offset + i: 1.0 for i in range(n_players)}, 5.0, 5.0)
        for role, target in zip(["Guard", "Forward", "Center"], [guards, forwards, centers]):
            add_constraint({starter_offset + i: 1.0 for i, player in enumerate(players) if player["Role"] == role}, float(target), float(target))
        add_constraint({sixth_offset + i: 1.0 for i in range(n_players)}, 1.0, 1.0)
        add_constraint({captain_offset + i: 1.0 for i in range(n_players)}, 1.0, 1.0)
        add_constraint({coach_offset + i: 1.0 for i in range(n_coaches)}, 1.0, 1.0)

        for idx in range(n_players):
            add_constraint({starter_offset + idx: 1.0, selected_offset + idx: -1.0}, -np.inf, 0.0)
            add_constraint({sixth_offset + idx: 1.0, selected_offset + idx: -1.0}, -np.inf, 0.0)
            add_constraint({captain_offset + idx: 1.0, starter_offset + idx: -1.0}, -np.inf, 0.0)
            add_constraint({starter_offset + idx: 1.0, sixth_offset + idx: 1.0, selected_offset + idx: -1.0}, -np.inf, 0.0)

        matrix = lil_matrix((len(rows), n_vars), dtype=float)
        for row_idx, coeffs in enumerate(rows):
            for col_idx, value in coeffs.items():
                matrix[row_idx, col_idx] = value

        result = milp(
            c=objective,
            integrality=np.ones(n_vars),
            bounds=Bounds(np.zeros(n_vars), np.ones(n_vars)),
            constraints=LinearConstraint(matrix.tocsr(), np.array(lower), np.array(upper)),
            options={"time_limit": 8.0},
        )
        if result.success and result.x is not None and float(result.fun) < best_fun:
            best_solution = result.x
            best_fun = float(result.fun)
            best_formation = f"{guards}-{forwards}-{centers}"

    if best_solution is None:
        return pd.DataFrame(), {}

    solution = best_solution
    assigned = []
    for idx, player in enumerate(players):
        if solution[selected_offset + idx] < 0.5:
            continue
        if solution[captain_offset + idx] >= 0.5:
            slot = "Captain"
            weight = FANTASY_CAPTAIN_MULTIPLIER
        elif solution[starter_offset + idx] >= 0.5:
            slot = "Starter"
            weight = 1.0
        elif solution[sixth_offset + idx] >= 0.5:
            slot = "6th Man"
            weight = 1.0
        else:
            slot = "Bench"
            weight = 0.5
        assigned.append({**player, "Slot": slot, "Weight": weight})

    coach_idx = next((idx for idx in range(n_coaches) if solution[coach_offset + idx] >= 0.5), None)
    if coach_idx is None:
        return pd.DataFrame(), {}
    assigned.append({**coach_rows[coach_idx], "Slot": "Coach", "Weight": 1.0})

    lineup = pd.DataFrame(assigned)
    lineup["Weighted Projection"] = pd.to_numeric(lineup["Projection"], errors="coerce") * pd.to_numeric(lineup["Weight"], errors="coerce")
    slot_order = pd.Categorical(lineup["Slot"], categories=["Captain", "Starter", "6th Man", "Bench", "Coach"], ordered=True)
    lineup = lineup.assign(_slot=slot_order).sort_values(["_slot", "Projection"], ascending=[True, False]).drop(columns="_slot")
    credits = float(pd.to_numeric(lineup["Credits"], errors="coerce").sum())
    projected = float(pd.to_numeric(lineup["Weighted Projection"], errors="coerce").sum())
    summary = {
        "Credits": round(credits, 1),
        "Projected": round(projected, 2),
        "Cap": credit_cap,
        "Coach": str(coach_rows[coach_idx].get("Player", "")),
        "Coach Source": str(coach_rows[coach_idx].get("Coach Source", "Estimated")),
        "Optimizer": "Exact MILP",
        "Formation": best_formation,
    }
    return lineup.round(2), summary


def optimize_fantasy_lineup_with_coach(predictions: pd.DataFrame, credit_cap: float) -> tuple[pd.DataFrame, dict]:
    exact_lineup, exact_summary = optimize_fantasy_lineup_exact(predictions, credit_cap)
    if not exact_lineup.empty:
        return exact_lineup, exact_summary
    coaches = coach_candidates_for_predictions(predictions)
    if coaches.empty:
        lineup, summary = optimize_fantasy_lineup(predictions, credit_cap)
        return lineup, summary
    coaches = coaches.sort_values(["Projection", "Credits"], ascending=[False, True]).head(12)
    best_lineup = pd.DataFrame()
    best_summary: dict = {}
    best_score = -999.0
    for _, coach in coaches.iterrows():
        remaining_cap = credit_cap - float(coach["Credits"])
        if remaining_cap <= 0:
            continue
        lineup, summary = optimize_fantasy_lineup(predictions, remaining_cap)
        if lineup.empty:
            continue
        total_score = float(summary.get("Projected", 0)) + float(coach["Projection"])
        if total_score > best_score:
            coach_df = pd.DataFrame([coach.to_dict()])
            lineup = pd.concat([lineup, coach_df], ignore_index=True, sort=False)
            best_score = total_score
            best_lineup = lineup
            best_summary = {
                "Credits": round(float(summary.get("Credits", 0)) + float(coach["Credits"]), 1),
                "Projected": round(total_score, 2),
                "Cap": credit_cap,
                "Coach": str(coach["Player"]),
                "Coach Source": str(coach.get("Coach Source", "Estimated")),
                "Formation": str(summary.get("Formation", "-")),
            }
    return best_lineup, best_summary


def winning_streak(rows: pd.DataFrame) -> int:
    if rows.empty or "won" not in rows.columns:
        return 0
    ordered = rows.sort_values(["parsed_date", "season", "game_code"], ascending=[False, False, False])
    streak = 0
    for won in ordered["won"].fillna(False).astype(bool):
        if not won:
            break
        streak += 1
    return streak


def filter_logs(
    df: pd.DataFrame,
    seasons: list[int],
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
    teams_filter: list[str] | None = None,
    phases: list[str] | None = None,
    venue: int | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["season"].isin(seasons)].copy()
    if teams_filter:
        out = out[out["team_code"].isin(teams_filter)]
    if phases and "phase" in out.columns:
        out = out[out["phase"].isin(phases)]
    if venue is not None and "home" in out.columns:
        out = out[out["home"].fillna(False).astype(int) == venue]
    if start_date is not None:
        out = out[out["parsed_date"] >= start_date]
    if end_date is not None:
        out = out[out["parsed_date"] <= end_date]
    return out


def player_role_profiles(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame(columns=["player_id", "player_role"])
    profile = (
        logs.groupby(["player_id", "player_name"], as_index=False)
        .agg(games=("game_code", "nunique"), assists=("assists", "mean"), rebounds=("total_rebounds", "mean"), blocks=("blocks_favour", "mean"))
    )
    profile["player_role"] = "Forward"
    profile.loc[(profile["assists"] >= 3.5) | (profile["assists"] >= profile["rebounds"] + profile["blocks"]), "player_role"] = "Guard"
    profile.loc[(profile["blocks"] >= 0.7) | ((profile["rebounds"] >= 7.0) & (profile["assists"] < 2.0)), "player_role"] = "Center"
    profile["player_role"] = profile.apply(
        lambda row: PLAYER_ROLE_OVERRIDES.get(str(row["player_id"]), row["player_role"]),
        axis=1,
    )
    return profile[["player_id", "player_role"]]


def with_player_roles(logs: pd.DataFrame) -> pd.DataFrame:
    if logs.empty:
        return logs
    roles = player_role_profiles(players)
    out = logs.merge(roles, on="player_id", how="left")
    out["player_role"] = out["player_role"].fillna("Forward")
    return out


def current_roster_player_ids(logs: pd.DataFrame, profiles_df: pd.DataFrame) -> set[str]:
    latest_log_ids: set[str] = set()
    if not logs.empty:
        latest_season = int(logs["season"].dropna().max())
        latest_logs = logs[logs["season"].astype(int) == latest_season]
        latest_round = pd.to_numeric(latest_logs["round"], errors="coerce").max() if "round" in latest_logs.columns else pd.NA
        if pd.notna(latest_round):
            active_logs = latest_logs[pd.to_numeric(latest_logs["round"], errors="coerce") >= int(latest_round) - 4]
        else:
            active_logs = latest_logs
        latest_log_ids = set(active_logs["player_id"].astype(str).unique())
    if not profiles_df.empty and {"season", "player_id", "team_code"}.issubset(profiles_df.columns):
        latest_profile_season = int(profiles_df["season"].dropna().max())
        roster = profiles_df[profiles_df["season"].astype(int) == latest_profile_season].copy()
        roster = roster[roster["team_code"].notna() & roster["player_id"].notna()]
        return set(roster["player_id"].astype(str)) | latest_log_ids
    return latest_log_ids


def current_roster_table(logs: pd.DataFrame, profiles_df: pd.DataFrame) -> pd.DataFrame:
    roster_ids = current_roster_player_ids(logs, profiles_df)
    if logs.empty or not roster_ids:
        return pd.DataFrame()
    role_logs = with_player_roles(logs[logs["player_id"].astype(str).isin(roster_ids)].copy())
    latest_log_team = (
        role_logs.sort_values(["player_id", "season", "game_code"])
        .drop_duplicates("player_id", keep="last")
        [["player_id", "player_name", "team_code", "team_name"]]
        .rename(columns={"player_name": "log_player_name", "team_code": "log_team_code", "team_name": "log_team_name"})
    )
    if not profiles_df.empty and {"season", "player_id", "team_code", "team_name", "player_name"}.issubset(profiles_df.columns):
        latest_profile_season = int(profiles_df["season"].dropna().max())
        latest_team = (
            profiles_df[profiles_df["season"].astype(int) == latest_profile_season]
            .sort_values(["player_id", "team_code"])
            .drop_duplicates("player_id")
            [["player_id", "player_name", "team_code", "team_name"]]
        )
        latest_team = latest_team.merge(latest_log_team, on="player_id", how="left")
        multi_team = latest_team["team_code"].astype(str).str.contains(";", regex=False)
        latest_team.loc[multi_team & latest_team["log_team_code"].notna(), "team_code"] = latest_team.loc[multi_team & latest_team["log_team_code"].notna(), "log_team_code"]
        latest_team.loc[multi_team & latest_team["log_team_name"].notna(), "team_name"] = latest_team.loc[multi_team & latest_team["log_team_name"].notna(), "log_team_name"]
        latest_team.loc[latest_team["player_name"].isna() & latest_team["log_player_name"].notna(), "player_name"] = latest_team.loc[latest_team["player_name"].isna() & latest_team["log_player_name"].notna(), "log_player_name"]
        latest_team = latest_team[["player_id", "player_name", "team_code", "team_name"]]
        profile_ids = set(latest_team["player_id"].astype(str))
        latest_from_logs = latest_log_team[~latest_log_team["player_id"].astype(str).isin(profile_ids)].rename(
            columns={"log_player_name": "player_name", "log_team_code": "team_code", "log_team_name": "team_name"}
        )
        latest_team = pd.concat([latest_team, latest_from_logs], ignore_index=True)
        latest_team = latest_team.drop_duplicates("player_id", keep="first")
        latest_team = latest_team.merge(
            role_logs[["player_id", "player_role"]].drop_duplicates("player_id"),
            on="player_id",
            how="left",
        )
        latest_team["team_name"] = latest_team["team_name"].fillna(latest_team["team_code"])
    else:
        latest_team = (
            role_logs.sort_values(["season", "game_code"], ascending=[False, False])
            .drop_duplicates("player_id")
            [["player_id", "player_name", "team_code", "team_name", "player_role"]]
        )
    latest_team["player_role"] = latest_team["player_role"].fillna("Forward")
    summary = (
        role_logs.groupby("player_id", as_index=False)
        .agg(GP=("game_code", "nunique"), LastSeason=("season", "max"), PIR=("pir", "mean"), MIN=("minutes", lambda values: values.map(minutes_to_float).mean()))
    )
    roster = latest_team.merge(summary, on="player_id", how="left")
    dunkest = load_dunkest_player_stats()
    if not dunkest.empty:
        roster["dunkest_key"] = roster["player_name"].map(player_name_match_key)
        roster = roster.merge(dunkest, on="dunkest_key", how="left").drop(columns=["dunkest_key"])
    return roster


def normalize_availability_rows(availability: pd.DataFrame, source: str) -> pd.DataFrame:
    if availability.empty:
        return pd.DataFrame(columns=["Availability Player", "availability_player_id", "Availability Team", "Availability Status", "Availability Impact", "Availability Note", "availability_key", "Availability Source", "Availability Updated", "Availability Role"])
    try:
        player_col = next((col for col in ["player", "Player", "player_name", "Player Name", "name", "Name"] if col in availability.columns), None)
        player_id_col = next((col for col in ["player_id", "Player ID", "id", "ID"] if col in availability.columns), None)
        team_col = next((col for col in ["team_code", "Team Code", "team", "Team"] if col in availability.columns), None)
        status_col = next((col for col in ["status", "Status", "availability", "Availability"] if col in availability.columns), None)
        impact_col = next((col for col in ["impact", "Impact", "availability_impact", "Availability Impact"] if col in availability.columns), None)
        note_col = next((col for col in ["note", "Note", "notes", "Notes", "reason", "Reason"] if col in availability.columns), None)
        if not (player_col or player_id_col):
            return pd.DataFrame()
        status = availability[status_col].fillna("available").astype(str).str.lower().str.strip() if status_col else pd.Series("available", index=availability.index)
        impact = pd.to_numeric(availability[impact_col], errors="coerce") if impact_col else status.map(AVAILABILITY_STATUS_FACTORS)
        impact = impact.fillna(status.map(AVAILABILITY_STATUS_FACTORS)).fillna(1.0).clip(0, 1)
        out = pd.DataFrame(
            {
                "Availability Player": availability[player_col].astype(str).str.strip() if player_col else "",
                "availability_player_id": availability[player_id_col].astype(str).str.strip() if player_id_col else "",
                "Availability Team": availability[team_col].astype(str).str.upper().str.strip() if team_col else "",
                "Availability Status": status,
                "Availability Impact": impact,
                "Availability Note": availability[note_col].fillna("").astype(str).str.strip() if note_col else "",
                "Availability Source": source,
                "Availability Updated": availability["updated"].astype(str).str.strip() if "updated" in availability.columns else "",
                "Availability Role": availability["role"].astype(str).str.strip() if "role" in availability.columns else "",
            }
        )
        out["availability_key"] = out["Availability Player"].map(player_name_match_key)
        out = out[(out["availability_player_id"].ne("")) | (out["availability_key"].ne(""))]
        return out.drop_duplicates(["availability_player_id", "availability_key", "Availability Team"], keep="last")
    except Exception:
        return pd.DataFrame()


def parse_availability_updated(value: str, latest_game_date: pd.Timestamp) -> pd.Timestamp | pd.NaT:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    parsed = pd.to_datetime(text, errors="coerce", utc=True)
    if pd.notna(parsed):
        return parsed
    match = re.match(r"^(\d{1,2})/(\d{1,2})$", text)
    if not match or pd.isna(latest_game_date):
        return pd.NaT
    day, month = (int(part) for part in match.groups())
    dated = pd.Timestamp(year=int(latest_game_date.year), month=month, day=day, tz="UTC")
    if dated > latest_game_date + pd.Timedelta(days=7):
        dated = dated - pd.DateOffset(years=1)
    return dated


@st.cache_data(show_spinner=False)
def load_latest_played_dates() -> pd.DataFrame:
    try:
        with sqlite3.connect(DB_PATH) as conn:
            logs = pd.read_sql(
                """
                SELECT b.player_id, b.team_code, b.minutes, g.game_date
                FROM player_boxscores b
                LEFT JOIN games g ON b.season = g.season AND b.game_code = g.game_code
                WHERE b.player_id NOT IN ('Total', 'Team')
                  AND b.player_name NOT IN ('Total', 'Team')
                """,
                conn,
            )
    except Exception:
        return pd.DataFrame(columns=["player_id", "team_code", "latest_played_at"])
    if logs.empty:
        return pd.DataFrame(columns=["player_id", "team_code", "latest_played_at"])
    logs["minutes_float"] = logs["minutes"].map(minutes_to_float)
    logs["played_at"] = pd.to_datetime(logs["game_date"], errors="coerce", utc=True)
    played = logs[(logs["minutes_float"].fillna(0) > 0) & logs["played_at"].notna()].copy()
    if played.empty:
        return pd.DataFrame(columns=["player_id", "team_code", "latest_played_at"])
    return played.groupby(["player_id", "team_code"], as_index=False).agg(latest_played_at=("played_at", "max"))


def remove_stale_availability_rows(availability: pd.DataFrame) -> pd.DataFrame:
    if availability.empty:
        return availability
    played = load_latest_played_dates()
    if played.empty:
        return availability
    latest_game_date = played["latest_played_at"].max()
    out = availability.merge(
        played,
        left_on=["availability_player_id", "Availability Team"],
        right_on=["player_id", "team_code"],
        how="left",
    )
    out["updated_at"] = out["Availability Updated"].map(lambda value: parse_availability_updated(value, latest_game_date))
    unavailable = pd.to_numeric(out["Availability Impact"], errors="coerce").fillna(1.0) < 1.0
    stale = unavailable & out["updated_at"].notna() & out["latest_played_at"].notna() & (out["latest_played_at"] > out["updated_at"])
    return out.loc[~stale, availability.columns].copy()


def infer_news_status(text: str) -> tuple[str | None, float | None]:
    lowered = str(text).lower()
    if any(re.search(pattern, lowered) for pattern in NEWS_AVAILABLE_PATTERNS):
        return "available", 1.0
    if any(re.search(pattern, lowered) for pattern in NEWS_OUT_PATTERNS):
        return "out", 0.0
    if any(re.search(pattern, lowered) for pattern in NEWS_DOUBTFUL_PATTERNS):
        return "questionable", 0.65
    return None, None


def player_name_variants(player_name: str) -> set[str]:
    raw = str(player_name).strip()
    display = player_name_to_display_name(raw)
    variants = {raw.upper(), display.upper()}
    if "," in raw:
        last, first = [part.strip() for part in raw.split(",", 1)]
        variants.add(f"{first} {last}".upper())
        variants.add(f"{last} {first}".upper())
    parts = display.split()
    if len(parts) >= 2:
        variants.add(" ".join(parts[:2]).upper())
    return {re.sub(r"\s+", " ", value).strip() for value in variants if len(value.strip()) >= 4}


@st.cache_data(show_spinner=False, ttl=60 * 20)
def fetch_news_availability(roster_names: tuple[tuple[str, str, str], ...]) -> pd.DataFrame:
    if not roster_names:
        return pd.DataFrame()
    player_lookup = []
    for player_id, player_name, team_code in roster_names:
        player_lookup.append(
            {
                "player_id": str(player_id),
                "player": str(player_name),
                "team_code": str(team_code).upper(),
                "variants": player_name_variants(str(player_name)),
            }
        )
    records = []
    for source, feed_url in AVAILABILITY_NEWS_FEEDS:
        try:
            response = requests.get(feed_url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue
        for item in root.findall("./channel/item")[:35]:
            title = html.unescape(item.findtext("title") or "")
            description = html.unescape(re.sub(r"<[^>]+>", " ", item.findtext("description") or ""))
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            text = f"{title} {description}"
            status, impact = infer_news_status(text)
            if status is None:
                continue
            normalized_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").upper()
            for player in player_lookup:
                if not any(re.search(rf"\b{re.escape(variant)}\b", normalized_text) for variant in player["variants"]):
                    continue
                records.append(
                    {
                        "player": player["player"],
                        "player_id": player["player_id"],
                        "team_code": player["team_code"],
                        "status": status,
                        "impact": impact,
                        "note": f"{title} | {link}",
                        "updated": pub_date,
                    }
                )
    if not records:
        return pd.DataFrame()
    return normalize_availability_rows(pd.DataFrame(records), "News")


def text_from_html_cell(cell: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(cell))
    return html.unescape(re.sub(r"\s+", " ", text).strip())


@st.cache_data(show_spinner=False, ttl=60 * 30)
def fetch_basketstories_availability() -> pd.DataFrame:
    try:
        response = requests.get(BASKETSTORIES_INJURIES_URL, timeout=15)
        response.raise_for_status()
    except Exception:
        return pd.DataFrame()
    page_html = response.text
    records = []
    pattern = re.compile(r'<div class="description1"[^>]*>(?P<role>[^<]+)</div>\s*<table[^>]*>(?P<table>.*?)</table>', re.S)
    for match in pattern.finditer(page_html):
        role_label = text_from_html_cell(match.group("role"))
        fantasy_role = BASKETSTORIES_ROLE_MAP.get(role_label, "")
        table_html = match.group("table")
        for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", table_html, flags=re.S)[1:]:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, flags=re.S)
            if len(cells) < 4:
                continue
            player = text_from_html_cell(cells[0])
            team_cell = cells[1]
            status_raw = text_from_html_cell(cells[2]).lower()
            updated = text_from_html_cell(cells[3])
            logo_match = re.search(r"/images/logos/([^\"'\s>]+)", team_cell)
            team_code = BASKETSTORIES_LOGO_TEAM_CODES.get(logo_match.group(1), "") if logo_match else ""
            status = "doubtful" if "αμφ" in status_raw or "doubt" in status_raw else "out"
            records.append(
                {
                    "player": player,
                    "team_code": team_code,
                    "status": status,
                    "impact": AVAILABILITY_STATUS_FACTORS.get(status, 0.0),
                    "note": status_raw,
                    "updated": updated,
                    "role": fantasy_role,
                }
            )
    return normalize_availability_rows(pd.DataFrame(records), "BasketStories")


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_player_availability(roster_names: tuple[tuple[str, str, str], ...] = tuple()) -> pd.DataFrame:
    frames = []
    remote = fetch_basketstories_availability()
    if not remote.empty:
        frames.append(remote)
    news = fetch_news_availability(roster_names)
    if not news.empty:
        frames.append(news)
    for path in AVAILABILITY_LOCAL_FILES:
        if not path.exists():
            continue
        try:
            local = pd.read_csv(path)
        except Exception:
            continue
        normalized = normalize_availability_rows(local, path.name)
        if not normalized.empty:
            frames.append(normalized)
    if not frames:
        return pd.DataFrame(columns=["Availability Player", "availability_player_id", "Availability Team", "Availability Status", "Availability Impact", "Availability Note", "availability_key", "Availability Source", "Availability Updated", "Availability Role"])
    combined = pd.concat(frames, ignore_index=True)
    combined = remove_stale_availability_rows(combined)
    combined["_priority"] = combined["Availability Source"].map({"BasketStories": 0, "News": 1}).fillna(2)
    combined = combined.sort_values("_priority").drop_duplicates(["availability_player_id", "availability_key", "Availability Team"], keep="last")
    return combined.drop(columns="_priority")


def apply_player_availability(roster: pd.DataFrame) -> pd.DataFrame:
    out = roster.copy()
    out["Availability Status"] = "available"
    out["Availability Impact"] = 1.0
    out["Availability Note"] = ""
    out["Availability Source"] = ""
    out["Availability Updated"] = ""
    out["availability_key"] = out["player_name"].map(player_name_match_key)
    roster_names = tuple(
        out[["player_id", "player_name", "team_code"]]
        .dropna(subset=["player_id", "player_name"])
        .astype(str)
        .itertuples(index=False, name=None)
    )
    availability = load_player_availability(roster_names)
    if availability.empty:
        return out.drop(columns=["availability_key"], errors="ignore")

    by_id = availability[availability["availability_player_id"].astype(str).str.len() > 0]
    if not by_id.empty:
        out = out.merge(
            by_id[["availability_player_id", "Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]].rename(columns={"availability_player_id": "player_id"}),
            on="player_id",
            how="left",
            suffixes=("", "_file"),
        )
        for col in ["Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]:
            out[col] = out[f"{col}_file"].combine_first(out[col])
            out = out.drop(columns=f"{col}_file")

    by_name = availability[availability["availability_key"].astype(str).str.len() > 0].copy()
    if not by_name.empty:
        by_name_team = by_name[by_name["Availability Team"].astype(str).str.len() > 0].rename(columns={"Availability Team": "team_code"})
        by_name_any_team = by_name[by_name["Availability Team"].astype(str).str.len() == 0].copy()
    else:
        by_name_team = pd.DataFrame()
        by_name_any_team = pd.DataFrame()
    if not by_name_team.empty:
        out = out.merge(
            by_name_team[["availability_key", "team_code", "Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]],
            on=["availability_key", "team_code"],
            how="left",
            suffixes=("", "_name_file"),
        )
        for col in ["Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]:
            out[col] = out[f"{col}_name_file"].combine_first(out[col])
            out = out.drop(columns=f"{col}_name_file")
    if not by_name_any_team.empty:
        out = out.merge(
            by_name_any_team[["availability_key", "Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]],
            on="availability_key",
            how="left",
            suffixes=("", "_any_file"),
        )
        for col in ["Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]:
            out[col] = out[f"{col}_any_file"].combine_first(out[col])
            out = out.drop(columns=f"{col}_any_file")

    out["Availability Status"] = out["Availability Status"].fillna("available").astype(str).str.lower()
    out["Availability Impact"] = pd.to_numeric(out["Availability Impact"], errors="coerce").fillna(1.0).clip(0, 1)
    out["Availability Note"] = out["Availability Note"].fillna("").astype(str)
    out["Availability Source"] = out["Availability Source"].fillna("").astype(str)
    out["Availability Updated"] = out["Availability Updated"].fillna("").astype(str)
    return out.drop(columns=["availability_key"], errors="ignore")


@st.cache_data(show_spinner=False, ttl=60 * 10)
def load_rotation_impact() -> pd.DataFrame:
    if not ROTATION_IMPACT_PATH.exists():
        return pd.DataFrame()
    try:
        impact = pd.read_csv(ROTATION_IMPACT_PATH)
    except Exception:
        return pd.DataFrame()
    for col in ["player_id", "team_code", "player_role"]:
        if col in impact.columns:
            impact[col] = impact[col].fillna("").astype(str)
    for col in [
        "same_role_minutes_boost", "cross_role_minutes_boost", "minutes_penalty",
        "net_minutes_delta", "rotation_impact_score", "team_missing_minutes",
        "role_missing_minutes", "availability_impact",
    ]:
        if col in impact.columns:
            impact[col] = pd.to_numeric(impact[col], errors="coerce").fillna(0.0)
    return impact


def apply_availability_projection_adjustments(predictions: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    if predictions.empty or roster.empty or "Availability Impact" not in roster.columns:
        return predictions
    out = predictions.copy()
    rotation_impact = load_rotation_impact()
    if not rotation_impact.empty and "player_id" in out.columns:
        impact_cols = [
            "player_id", "same_role_minutes_boost", "cross_role_minutes_boost",
            "minutes_penalty", "net_minutes_delta", "rotation_impact_score",
            "team_missing_minutes", "role_missing_minutes", "availability_status",
            "availability_impact",
        ]
        out = out.merge(rotation_impact[[col for col in impact_cols if col in rotation_impact.columns]], on="player_id", how="left")
        for col in impact_cols:
            if col not in {"player_id", "availability_status"} and col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
        if "net_minutes_delta" in out.columns:
            out["Injury Boost MIN"] = out["net_minutes_delta"].clip(lower=-8.0, upper=8.0)
            status = out.get("availability_status", pd.Series("", index=out.index)).fillna("").astype(str).str.strip().str.lower()
            impact = pd.to_numeric(out.get("availability_impact", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0)
            out_mask = status.isin(["out", "injured", "inactive", "dnp"]) | (impact <= 0.25)
            out.loc[out_mask, "Injury Boost MIN"] = -pd.to_numeric(out.loc[out_mask, "Pred MIN"], errors="coerce").fillna(0.0)
            pir_per_min = (out["Expected PIR"] / out["Pred MIN"].replace(0, pd.NA)).fillna(out.get("PIR", 0) / out.get("MIN", 1)).clip(lower=0.12, upper=0.9)
            out["Pred MIN"] = (out["Pred MIN"] + out["Injury Boost MIN"]).clip(lower=0)
            out["Injury Adj PIR"] = (out["Injury Boost MIN"] * pir_per_min).round(2)
            for col in ["Expected PIR", "Low", "High"]:
                out[col] = pd.to_numeric(out[col], errors="coerce") + out["Injury Adj PIR"]
            return out

    roster_view = roster.copy()
    roster_view["Availability Impact"] = pd.to_numeric(roster_view["Availability Impact"], errors="coerce").fillna(1.0).clip(0, 1)
    roster_view["MIN"] = pd.to_numeric(roster_view.get("MIN"), errors="coerce").fillna(0)
    unavailable = roster_view[roster_view["Availability Impact"] <= 0.25].copy()
    out["Injury Boost MIN"] = 0.0
    out["Injury Adj PIR"] = 0.0
    if unavailable.empty:
        return out

    missing_role_minutes = unavailable.groupby(["team_code", "player_role"], as_index=False).agg(MissingRoleMIN=("MIN", "sum"))
    missing_team_minutes = unavailable.groupby("team_code", as_index=False).agg(MissingTeamMIN=("MIN", "sum"))
    role_available_minutes = (
        out.groupby(["team_code", "Role"], as_index=False)
        .agg(AvailableRolePredMIN=("Pred MIN", "sum"))
        .rename(columns={"Role": "player_role"})
    )
    out = out.merge(missing_role_minutes.rename(columns={"player_role": "Role"}), on=["team_code", "Role"], how="left")
    out = out.merge(missing_team_minutes, on="team_code", how="left")
    out = out.merge(role_available_minutes.rename(columns={"player_role": "Role"}), on=["team_code", "Role"], how="left")
    for col in ["MissingRoleMIN", "MissingTeamMIN", "AvailableRolePredMIN"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)

    share = (out["Pred MIN"] / out["AvailableRolePredMIN"].replace(0, pd.NA)).fillna(0)
    same_role_boost = out["MissingRoleMIN"] * 0.42 * share
    other_role_boost = (out["MissingTeamMIN"] - out["MissingRoleMIN"]).clip(lower=0) * 0.08 * share
    out["Injury Boost MIN"] = (same_role_boost + other_role_boost).clip(lower=0, upper=7.0)

    availability_impact = pd.to_numeric(out.get("Availability Impact", pd.Series(1.0, index=out.index)), errors="coerce").fillna(1.0).clip(0, 1)
    availability_penalty_min = (1.0 - availability_impact) * out["Pred MIN"] * 0.55
    pir_per_min = (out["Expected PIR"] / out["Pred MIN"].replace(0, pd.NA)).fillna(out.get("PIR", 0) / out.get("MIN", 1)).clip(lower=0.12, upper=0.9)
    out["Pred MIN"] = (out["Pred MIN"] + out["Injury Boost MIN"] - availability_penalty_min).clip(lower=0)
    out["Injury Adj PIR"] = ((out["Injury Boost MIN"] - availability_penalty_min) * pir_per_min).round(2)
    for col in ["Expected PIR", "Low", "High"]:
        out[col] = pd.to_numeric(out[col], errors="coerce") + out["Injury Adj PIR"]
    return out.drop(columns=["MissingRoleMIN", "MissingTeamMIN", "AvailableRolePredMIN"], errors="ignore")


@st.cache_data(show_spinner=False)
def fetch_official_schedule(season: int) -> pd.DataFrame:
    url = OFFICIAL_SCHEDULE_URL.format(season=season)
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame()

    raw = pd.json_normalize(rows)
    schedule = pd.DataFrame(
        {
            "season": season,
            "game_code": pd.to_numeric(raw.get("gameCode"), errors="coerce"),
            "game_id": raw.get("identifier"),
            "phase": raw.get("phaseType.code"),
            "round": pd.to_numeric(raw.get("round"), errors="coerce"),
            "game_date": raw.get("localDate").fillna(raw.get("date")),
            "group_name": raw.get("group.rawName").fillna(raw.get("group.name")),
            "home_team": raw.get("local.club.name"),
            "home_code": raw.get("local.club.code"),
            "home_score": pd.to_numeric(raw.get("local.score"), errors="coerce").fillna(0),
            "away_team": raw.get("road.club.name"),
            "away_code": raw.get("road.club.code"),
            "away_score": pd.to_numeric(raw.get("road.score"), errors="coerce").fillna(0),
            "played": raw.get("played").fillna(False).astype(bool).astype(int),
            "schedule_source": "Official EuroLeague v2",
        }
    )
    schedule["game_time"] = pd.to_datetime(schedule["game_date"], errors="coerce").dt.strftime("%H:%M")
    schedule["game_time"] = schedule["game_time"].fillna("")
    schedule = schedule.dropna(subset=["game_code", "round", "home_code", "away_code"])
    schedule["game_code"] = schedule["game_code"].astype(int)
    schedule["round"] = schedule["round"].astype(int)
    return schedule


@st.cache_data(show_spinner=False)
def load_ml_schedule() -> pd.DataFrame:
    with connect(DB_PATH) as conn:
        games = pd.read_sql_query(
            """
            SELECT season, game_code, phase, round, game_date, game_time, group_name,
                   home_team, home_code, away_team, away_code, played,
                   home_score, away_score
            FROM games
            ORDER BY season DESC, round DESC, game_code DESC
            """,
            conn,
        )
    if not games.empty:
        games["schedule_source"] = "Local SQLite"

    seasons = games["season"].dropna().astype(int).unique().tolist() if not games.empty else []
    if seasons:
        latest_season = max(seasons)
        try:
            official = fetch_official_schedule(latest_season)
        except Exception:
            official = pd.DataFrame()
        if not official.empty:
            games = pd.concat([games, official], ignore_index=True)
            games["_source_priority"] = games["schedule_source"].eq("Official EuroLeague v2").astype(int)
            games = games.sort_values(["season", "game_code", "_source_priority"])
            games = games.drop_duplicates(["season", "game_code"], keep="last")
            games = games.drop(columns=["_source_priority"])

    if games.empty:
        return games
    games["parsed_date"] = pd.to_datetime(games["game_date"], errors="coerce", format="mixed")
    games["played"] = games["played"].fillna(0).astype(int)
    return games.sort_values(["season", "round", "game_code"], ascending=[False, False, False])


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_dunkest_player_stats() -> pd.DataFrame:
    local = load_local_dunkest_player_stats()
    if not local.empty:
        return local

    params = {
        "season_id": DUNKEST_SEASON_ID,
        "mode": "dunkest",
        "stats_type": "tot",
        "weeks[]": list(range(1, 41)),
        "rounds[]": [1, 2, 3],
        "teams[]": DUNKEST_TEAMS,
        "positions[]": DUNKEST_POSITIONS,
        "player_search": "",
        "min_cr": 4,
        "max_cr": 35,
        "sort_by": "pdk",
        "sort_order": "desc",
        "iframe": "yes",
    }
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/plain, */*",
        "Referer": "https://www.dunkest.com/en/euroleague/stats/players/table/season/2025-2026",
    }
    try:
        response = requests.get(DUNKEST_STATS_URL, params=params, headers=headers, timeout=25, verify=False)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)
    dunkest = pd.DataFrame(data)
    if dunkest.empty:
        return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)
    if {"first_name", "last_name"}.issubset(dunkest.columns):
        dunkest["dunkest_player"] = dunkest["first_name"].fillna("").astype(str).str.strip() + " " + dunkest["last_name"].fillna("").astype(str).str.strip()
    elif "player" in dunkest.columns:
        dunkest["dunkest_player"] = dunkest["player"].astype(str)
    else:
        return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)
    dunkest["dunkest_key"] = dunkest["dunkest_player"].map(player_name_match_key)
    base_cr = pd.to_numeric(dunkest.get("cr"), errors="coerce")
    plus = pd.to_numeric(dunkest.get("plus"), errors="coerce") if "plus" in dunkest.columns else pd.Series(0, index=dunkest.index)
    current_cr = (base_cr.fillna(0) + plus.fillna(0)).round(1)
    current_cr = current_cr.where(base_cr.notna(), pd.NA)
    out = pd.DataFrame(
        {
            "dunkest_key": dunkest["dunkest_key"],
            "Dunkest CR": current_cr,
            "Dunkest Base CR": base_cr,
            "Dunkest PDK": pd.to_numeric(dunkest.get("pdk"), errors="coerce"),
            "Dunkest GP": pd.to_numeric(dunkest.get("gp"), errors="coerce"),
            "Dunkest PLUS": plus,
        }
    )
    if "team_code" in dunkest.columns:
        out["Dunkest Team"] = dunkest["team_code"].astype(str)
    if "position" in dunkest.columns:
        out["Dunkest Position"] = dunkest["position"].astype(str)
    out = out.dropna(subset=["dunkest_key"]).drop_duplicates("dunkest_key", keep="first")
    return out


def normalize_dunkest_rows(dunkest: pd.DataFrame) -> pd.DataFrame:
    if dunkest.empty:
        return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)
    if {"first_name", "last_name"}.issubset(dunkest.columns):
        dunkest["dunkest_player"] = dunkest["first_name"].fillna("").astype(str).str.strip() + " " + dunkest["last_name"].fillna("").astype(str).str.strip()
    elif "player" in dunkest.columns:
        dunkest["dunkest_player"] = dunkest["player"].astype(str)
    elif "name" in dunkest.columns:
        dunkest["dunkest_player"] = dunkest["name"].astype(str)
    else:
        return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)
    dunkest["dunkest_key"] = dunkest["dunkest_player"].map(player_name_match_key)
    base_cr = pd.to_numeric(dunkest.get("cr", dunkest.get("credits", dunkest.get("price"))), errors="coerce")
    plus = pd.to_numeric(dunkest.get("plus"), errors="coerce") if "plus" in dunkest.columns else pd.Series(0, index=dunkest.index)
    current_cr = (base_cr.fillna(0) + plus.fillna(0)).round(1)
    current_cr = current_cr.where(base_cr.notna(), pd.NA)
    out = pd.DataFrame(
        {
            "dunkest_key": dunkest["dunkest_key"],
            "Dunkest CR": current_cr,
            "Dunkest Base CR": base_cr,
            "Dunkest PDK": pd.to_numeric(dunkest.get("pdk"), errors="coerce"),
            "Dunkest GP": pd.to_numeric(dunkest.get("gp"), errors="coerce"),
            "Dunkest PLUS": plus,
        }
    )
    if "team_code" in dunkest.columns:
        out["Dunkest Team"] = dunkest["team_code"].astype(str)
    elif "team" in dunkest.columns:
        out["Dunkest Team"] = dunkest["team"].astype(str)
    if "position" in dunkest.columns:
        out["Dunkest Position"] = dunkest["position"].astype(str)
    elif "position_norm" in dunkest.columns:
        out["Dunkest Position"] = dunkest["position_norm"].astype(str)
    elif "pos" in dunkest.columns:
        out["Dunkest Position"] = dunkest["pos"].astype(str)
    alternate = out.copy()
    alternate["dunkest_key"] = dunkest["dunkest_player"].map(player_name_last_token_first_key)
    out = pd.concat([out, alternate], ignore_index=True)
    return out.dropna(subset=["dunkest_key"]).drop_duplicates("dunkest_key", keep="first")


def load_local_dunkest_player_stats() -> pd.DataFrame:
    for path in DUNKEST_LOCAL_FILES:
        if not path.exists():
            continue
        try:
            if path.suffix.lower() == ".csv":
                rows = pd.read_csv(path)
            else:
                raw = path.read_text(encoding="utf-8").strip()
                payload = json.loads(raw)
                if isinstance(payload, dict):
                    payload = payload.get("players", payload.get("data", payload.get("rows", [])))
                rows = pd.DataFrame(payload)
            normalized = normalize_dunkest_rows(rows)
            if not normalized.empty:
                return normalized
        except Exception:
            continue
    return pd.DataFrame(columns=DUNKEST_EMPTY_COLUMNS)


@st.cache_data(show_spinner=False, ttl=60 * 60)
def load_fantasy_coaches() -> pd.DataFrame:
    for path in COACH_LOCAL_FILES:
        if not path.exists():
            continue
        try:
            coaches = pd.read_csv(path)
        except Exception:
            continue
        if coaches.empty:
            continue
        name_col = next((col for col in ["coach", "Coach", "name", "Name", "manager", "Manager"] if col in coaches.columns), None)
        team_col = next((col for col in ["team_code", "Team Code", "team", "Team"] if col in coaches.columns), None)
        credit_col = next((col for col in ["credits", "Credits", "cr", "CR", "value", "Value"] if col in coaches.columns), None)
        if not team_col:
            continue
        out = pd.DataFrame(
            {
                "Team Code": coaches[team_col].astype(str).str.upper().str.strip(),
                "Coach": coaches[name_col].astype(str).str.strip() if name_col else coaches[team_col].astype(str).str.upper().str.strip() + " Coach",
                "Coach CR": pd.to_numeric(coaches[credit_col], errors="coerce") if credit_col else pd.NA,
            }
        )
        out["Team Code"] = out["Team Code"].replace(COACH_TEAM_CODE_ALIASES)
        return out.dropna(subset=["Team Code"]).drop_duplicates("Team Code", keep="first")
    return pd.DataFrame(columns=["Team Code", "Coach", "Coach CR"])


def schedule_rest_days(schedule: pd.DataFrame, season: int, game_code: int, team_code: str) -> int:
    if schedule.empty or "parsed_date" not in schedule.columns:
        return 4
    current_rows = schedule[
        (schedule["season"].astype(int) == int(season))
        & (schedule["game_code"].astype(int) == int(game_code))
        & ((schedule["home_code"].astype(str) == str(team_code)) | (schedule["away_code"].astype(str) == str(team_code)))
    ]
    if current_rows.empty or pd.isna(current_rows.iloc[0]["parsed_date"]):
        return 4
    current_date = pd.Timestamp(current_rows.iloc[0]["parsed_date"])
    team_games = schedule[
        (schedule["season"].astype(int) == int(season))
        & schedule["parsed_date"].notna()
        & ((schedule["home_code"].astype(str) == str(team_code)) | (schedule["away_code"].astype(str) == str(team_code)))
        & (schedule["parsed_date"] < current_date)
    ].copy()
    if team_games.empty:
        return 4
    previous_date = pd.Timestamp(team_games.sort_values("parsed_date").iloc[-1]["parsed_date"])
    rest = (current_date.normalize() - previous_date.normalize()).days
    return max(0, min(int(rest), 30))


def player_form_table(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    ordered = rows.sort_values(["parsed_date", "season", "game_code"], ascending=False)
    records = []
    for window in [3, 5, 10]:
        slice_ = ordered.head(window)
        records.append(
            {
                "Window": f"Last {window}",
                "GP": int(slice_["game_code"].nunique()),
                "PIR": slice_["pir"].mean(),
                "PTS": slice_["points"].mean(),
                "REB": slice_["total_rebounds"].mean(),
                "AST": slice_["assists"].mean(),
                "STL": slice_["steals"].mean(),
                "TOV": slice_["turnovers"].mean(),
                "MIN": slice_["minutes"].map(minutes_to_float).mean(),
            }
        )
    return pd.DataFrame(records).round(2)


def opponent_role_allowance(logs: pd.DataFrame, opponent_code: str) -> pd.DataFrame:
    if logs.empty:
        return pd.DataFrame()
    role_logs = with_player_roles(logs)
    view = role_logs[role_logs["opponent_code"] == opponent_code]
    if view.empty:
        return pd.DataFrame()
    table = (
        view.groupby("player_role", as_index=False)
        .agg(
            GP=("game_code", "nunique"),
            PIR_Allowed=("pir", "mean"),
            PTS_Allowed=("points", "mean"),
            REB_Allowed=("total_rebounds", "mean"),
            AST_Allowed=("assists", "mean"),
            STL_Allowed=("steals", "mean"),
            TOV_Forced=("turnovers", "mean"),
        )
        .rename(columns={"player_role": "Role"})
        .round(2)
    )
    order = pd.Categorical(table["Role"], categories=["Guard", "Forward", "Center"], ordered=True)
    return table.assign(_order=order).sort_values("_order").drop(columns="_order")


def game_role_pir_table(player_rows: pd.DataFrame) -> pd.DataFrame:
    if player_rows.empty:
        return pd.DataFrame()
    role_rows = with_player_roles(player_rows)
    role_rows["player_role"] = pd.Categorical(role_rows["player_role"], categories=["Guard", "Forward", "Center"], ordered=True)
    role_team = (
        role_rows.groupby(["team_code", "team_name", "player_role"], observed=True, as_index=False)
        .agg(PIR_For=("pir", "sum"), Players=("player_id", "nunique"))
    )
    opponent = role_team[["team_code", "player_role", "PIR_For"]].rename(columns={"team_code": "opponent_code", "PIR_For": "PIR_Against"})
    teams_in_game = player_rows[["team_code", "opponent_code"]].drop_duplicates()
    out = role_team.merge(teams_in_game, on="team_code", how="left")
    out = out.merge(opponent, on=["opponent_code", "player_role"], how="left")
    out["PIR_Against"] = out["PIR_Against"].fillna(0)
    out = out.rename(columns={"team_name": "Team", "player_role": "Role"})
    out["Role"] = out["Role"].astype("string").replace({"Guard": "Guards", "Forward": "Forwards", "Center": "Centers"})
    out = out.rename(columns={"PIR_For": "PIR Created", "PIR_Against": "PIR Allowed"})
    return out[["Team", "Role", "Players", "PIR Created", "PIR Allowed"]].round(2)


@st.cache_data(show_spinner=False)
def cached_ml_predictions(
    player_ids: tuple[str, ...],
    opponent_code: str,
    home: int,
    phase: str,
    rest_days: int,
    team_code: str,
) -> pd.DataFrame:
    return predict_players(list(player_ids), opponent_code, home, phase, rest_days, team_code)


def radar_chart(left_name: str, left_values: pd.Series, right_name: str, right_values: pd.Series, columns: list[str], title: str) -> go.Figure:
    labels = [STAT_LABELS.get(col, col) for col in columns]
    left = [float(left_values.get(col, 0) or 0) for col in columns]
    right = [float(right_values.get(col, 0) or 0) for col in columns]
    max_values = [max(l, r, 1) for l, r in zip(left, right)]
    left_color, right_color = compare_palette()
    grid_color = "rgba(71,85,105,.24)" if is_light_theme() else "rgba(148,163,184,.24)"
    tick_color = "#4b5c72" if is_light_theme() else "#b9c6d8"
    radial_color = "rgba(71,85,105,.32)" if is_light_theme() else "rgba(148,163,184,.28)"
    panel_bg = "rgba(255,255,255,.58)" if is_light_theme() else "rgba(9,21,34,.18)"
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[l / m * 100 for l, m in zip(left, max_values)],
            theta=labels,
            customdata=np.array(left).reshape(-1, 1),
            fill="toself",
            name=left_name,
            line=dict(color=left_color, width=3),
            marker=dict(size=7, color=left_color, line=dict(color="#ffffff", width=1)),
            fillcolor="rgba(220,95,31,.18)" if is_light_theme() else "rgba(242,106,33,.20)",
            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{customdata[0]:.2f}<br>Relative: %{r:.0f}%<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[r / m * 100 for r, m in zip(right, max_values)],
            theta=labels,
            customdata=np.array(right).reshape(-1, 1),
            fill="toself",
            name=right_name,
            line=dict(color=right_color, width=3),
            marker=dict(size=7, color=right_color, line=dict(color="#ffffff", width=1)),
            fillcolor="rgba(37,99,235,.15)" if is_light_theme() else "rgba(56,189,248,.16)",
            hovertemplate="<b>%{fullData.name}</b><br>%{theta}: %{customdata[0]:.2f}<br>Relative: %{r:.0f}%<extra></extra>",
        )
    )
    radar_layout = plotly_dark(480)
    radar_layout.update(
        title=dict(text=title, x=0, font=dict(size=21, color="#182033" if is_light_theme() else "#eaf2ff")),
        showlegend=False,
        polar=dict(
            bgcolor=panel_bg,
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor=radial_color,
                linecolor=radial_color,
                tickvals=[25, 50, 75, 100],
                ticktext=["25", "50", "75", "100"],
                tickfont=dict(color=tick_color, size=10),
                angle=90,
                showticklabels=True,
                ticks="",
            ),
            angularaxis=dict(
                gridcolor=grid_color,
                linecolor=grid_color,
                tickfont=dict(color=tick_color, size=12),
                showticklabels=True,
            ),
        ),
        margin=dict(l=46, r=46, t=86, b=42),
    )
    fig.update_layout(**radar_layout)
    return fig


def compare_side_card(name: str, image_uri: str, color: str, subtitle: str = "", image_kind: str = "photo") -> str:
    initials = "".join(part[:1] for part in str(name).replace(",", " ").split()[:2]).upper() or str(name)[:1].upper()
    image_class = "compare-logo" if image_kind == "logo" else "compare-photo"
    scope = f"compare-card-{coach_photo_key(str(subtitle) + '-' + str(name) + '-' + color)}"
    visual = f'<img class="{image_class}" src="{image_uri}" />' if image_uri else f'<div class="compare-fallback">{html.escape(initials)}</div>'
    if is_light_theme():
        card_bg = "linear-gradient(180deg, rgba(255,255,255,.96), rgba(246,249,253,.96))"
        card_border = "#d7deea"
        media_bg = "#ffffff"
        media_border = "#cbd5e1"
        name_color = "#182033"
        sub_color = "#65738a"
        fallback_bg = "linear-gradient(135deg, #e9eef6, #ffffff)"
        shadow = "0 18px 42px rgba(24,32,51,.08)"
        color_shadow = f"0 8px 24px {color}44"
    else:
        card_bg = "linear-gradient(180deg, rgba(21,36,58,.72), rgba(12,24,39,.78))"
        card_border = "rgba(148,163,184,.18)"
        media_bg = "#26364b"
        media_border = "rgba(219,234,254,.38)"
        name_color = "#eaf2ff"
        sub_color = "#9baac0"
        fallback_bg = "linear-gradient(135deg, #2b3d56, #0f172a)"
        shadow = "0 18px 42px rgba(0,0,0,.18)"
        color_shadow = f"0 8px 24px {color}55"
    return f"""
    <style>
    .{scope} {{
        min-height: 430px;
        display:flex;
        flex-direction:column;
        align-items:center;
        justify-content:center;
        gap:14px;
        text-align:center;
        border:1px solid {card_border};
        border-radius:12px;
        background:{card_bg};
        padding:20px 14px;
        box-shadow:{shadow};
        position:relative;
        overflow:hidden;
    }}
    .{scope}:before {{
        content:"";
        position:absolute;
        inset:0;
        border-top:4px solid {color};
        opacity:.95;
        pointer-events:none;
    }}
    .{scope} .compare-photo, .{scope} .compare-logo, .{scope} .compare-fallback {{
        width:142px;
        height:142px;
        background:{media_bg};
        border:1px solid {media_border};
        box-shadow:{color_shadow};
        position:relative;
        z-index:1;
    }}
    .{scope} .compare-photo {{ border-radius:999px; object-fit:cover; object-position:top center; }}
    .{scope} .compare-logo {{ border-radius:14px; object-fit:contain; padding:12px; background:{media_bg}; }}
    .{scope} .compare-fallback {{
        border-radius:999px;
        display:flex;
        align-items:center;
        justify-content:center;
        color:{name_color};
        font-weight:900;
        font-size:1.35rem;
        background:{fallback_bg};
    }}
    .{scope} .compare-name {{
        color:{name_color};
        font-size:1.05rem;
        font-weight:900;
        line-height:1.12;
        overflow-wrap:anywhere;
        position:relative;
        z-index:1;
    }}
    .{scope} .compare-subtitle {{
        color:{sub_color};
        font-size:.78rem;
        font-weight:800;
        text-transform:uppercase;
        letter-spacing:.06em;
        position:relative;
        z-index:1;
    }}
    .{scope} .compare-swatch {{
        width:112px;
        height:8px;
        border-radius:999px;
        background:{color};
        box-shadow:{color_shadow};
        position:relative;
        z-index:1;
    }}
    </style>
    <div class="{scope}">
        {visual}
        <div class="compare-swatch"></div>
        <div class="compare-name">{html.escape(str(name))}</div>
        <div class="compare-subtitle">{html.escape(str(subtitle))}</div>
    </div>
    """


def highlight_best_rows(df: pd.DataFrame, label_col: str, lower_is_better: set[str] | None = None) -> dict[str, str]:
    lower_is_better = lower_is_better or set()
    highlights: dict[str, str] = {}
    numeric_cols = [col for col in df.select_dtypes(include="number").columns if col != label_col]
    for col in numeric_cols:
        target = df[col].min() if col in lower_is_better else df[col].max()
        if (df[col] == target).sum() != 1:
            continue
        for idx, value in df[col].items():
            if value == target:
                highlights[f"{idx}:{col}"] = "highlight-good"
    return highlights


def show_identity(title: str, kicker: str, logo_uri: str | None = None, initials: str | None = None, headshot: bool = False) -> None:
    image_class = ' class="player-headshot"' if headshot else ""
    visual = f'<img{image_class} src="{logo_uri}" />' if logo_uri else f'<div style="width:74px;height:74px;border-radius:999px;background:#26364b;display:flex;align-items:center;justify-content:center;font-size:2rem;font-weight:900;color:white;">{html.escape(initials or title[:1])}</div>'
    st.markdown(
        f"""
        <div class="identity-card">
            {visual}
            <div>
                <div class="identity-kicker">{html.escape(kicker)}</div>
                <div class="identity-title">{html.escape(title)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_ml_player_identity(title: str, kicker: str, player_id: str) -> None:
    photo = player_photo_uri_for_name(player_id, title, profiles)
    initials = "".join(part[:1] for part in title.replace(",", " ").split()[:2]).upper() or title[:1]
    visual = (
        f'<img class="ml-player-photo" src="{photo}" />'
        if photo
        else f'<div class="ml-player-fallback">{html.escape(initials)}</div>'
    )
    st.markdown(
        f"""
        <div class="identity-card" style="align-items:center; gap:28px; padding:28px 32px;">
            {visual}
            <div>
                <div class="identity-kicker">{html.escape(kicker)}</div>
                <div class="identity-title" style="font-size:2rem;">{html.escape(title)}</div>
            </div>
        </div>
        <style>
        .ml-player-photo {{
            width: 132px;
            height: 132px;
            border-radius: 999px;
            object-fit: cover;
            object-position: top center;
            border: 2px solid #3d516d;
            background: #26364b;
            box-shadow: 0 18px 38px rgba(0,0,0,.28);
        }}
        .ml-player-fallback {{
            width: 132px;
            height: 132px;
            border-radius: 999px;
            background: linear-gradient(135deg, #2b3d56, #0f172a);
            display: flex;
            align-items: center;
            justify-content: center;
            color: #eaf2ff;
            font-size: 2.2rem;
            font-weight: 900;
            border: 2px solid #3d516d;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_player_logs() -> pd.DataFrame:
    return player_game_logs(DB_PATH)


@st.cache_data(show_spinner=False)
def load_player_profiles() -> pd.DataFrame:
    return player_profiles(DB_PATH)


@st.cache_data(show_spinner=False)
def load_team_logs() -> pd.DataFrame:
    return team_game_logs(DB_PATH)


@st.cache_data(show_spinner=False)
def load_shots() -> pd.DataFrame:
    return shot_chart_data(DB_PATH)


@st.cache_data(show_spinner=False)
def load_summary() -> dict[str, int]:
    return database_summary(DB_PATH)


@st.cache_data(show_spinner=False)
def load_date_bounds():
    return available_date_bounds(DB_PATH)


def format_refresh_timestamp(timestamp) -> str:
    if timestamp is None or pd.isna(timestamp):
        return "Not available"
    parsed = pd.to_datetime(timestamp, errors="coerce", utc=True)
    if pd.isna(parsed):
        return "Not available"
    return parsed.tz_convert("Europe/Athens").strftime("%d/%m/%Y %H:%M")


def latest_file_timestamp(paths: list[Path]):
    mtimes = [path.stat().st_mtime for path in paths if path.exists()]
    if not mtimes:
        return None
    return pd.Timestamp(max(mtimes), unit="s", tz="UTC")


@st.cache_data(show_spinner=False)
def load_refresh_status() -> dict[str, str]:
    data_timestamps = []
    if DB_PATH.exists():
        file_timestamp = latest_file_timestamp([DB_PATH])
        if file_timestamp is not None:
            data_timestamps.append(file_timestamp)
        try:
            with sqlite3.connect(DB_PATH) as conn:
                row = conn.execute(
                    """
                    SELECT finished_at
                    FROM ingestion_runs
                    WHERE status = 'success' AND finished_at IS NOT NULL
                    ORDER BY id DESC
                    LIMIT 1
                    """
                ).fetchone()
            if row:
                data_timestamps.append(pd.to_datetime(row[0], errors="coerce", utc=True))
        except sqlite3.Error:
            pass

    availability_timestamp = latest_file_timestamp(
        [*AVAILABILITY_LOCAL_FILES[:2], ROTATION_IMPACT_PATH]
    )
    return {
        "data": format_refresh_timestamp(max(data_timestamps) if data_timestamps else None),
        "availability": format_refresh_timestamp(availability_timestamp),
    }


with st.sidebar:
    st.caption("Appearance")
    theme_mode = st.radio("Theme", ["Dark", "Light"], horizontal=True, key="app_theme")
apply_styles(theme_mode)

players = load_player_logs()
profiles = load_player_profiles()
teams = load_team_logs()
shots = load_shots()
summary = load_summary()

if players.empty or teams.empty:
    st.warning("No usable boxscore data found. Run update_data.bat first.")
    st.stop()

all_seasons = sorted(players["season"].dropna().astype(int).unique().tolist())
team_lookup = dict(
    zip(
        teams[["team_name", "team_code"]].drop_duplicates().sort_values("team_name")["team_name"],
        teams[["team_name", "team_code"]].drop_duplicates().sort_values("team_name")["team_code"],
    )
)
code_to_team = {code: name for name, code in team_lookup.items()}
min_date, max_date = load_date_bounds()
refresh_status = load_refresh_status()

with st.sidebar:
    st.markdown(
        f"""
        <div class="brand-block">
            <img src="{euroleague_logo_uri()}" />
            <div>
                <div class="brand-main">Pro<br/>Analytics</div>
                <div class="brand-sub">EuroLeague 2023-2026</div>
            </div>
        </div>
        <div class="refresh-status">
            <strong>Last Refresh</strong>
            <div class="refresh-row"><span>Data</span><span>{html.escape(refresh_status["data"])}</span></div>
            <div class="refresh-row"><span>Injury</span><span>{html.escape(refresh_status["availability"])}</span></div>
        </div>
        <div class="apply-button">Live Filters</div>
        """,
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", ["Overview", "Player Dashboard", "Team Dashboard", "Match Center", "ML PIR Predictor", "Shot Chart", "Compare"], label_visibility="collapsed")
    st.markdown('<div class="sidebar-search-label">Search</div>', unsafe_allow_html=True)
    team_suggestions = sorted(teams["team_name"].dropna().unique().tolist())
    player_suggestions = sorted(players["player_name"].dropna().unique().tolist())
    search_options = [""] + [f"Team | {name}" for name in team_suggestions] + [f"Player | {name}" for name in player_suggestions]
    selected_search = st.selectbox("Search players or teams", search_options, format_func=lambda value: "Search players, teams..." if value == "" else value, label_visibility="collapsed")
    st.divider()
    st.subheader("Global Filters")
    selected_seasons = st.multiselect("Seasons", all_seasons, default=all_seasons, format_func=season_label)
    if not selected_seasons:
        selected_seasons = all_seasons
    available_phase_codes = [code for code in PHASE_LABELS if code in set(teams["phase"].dropna().unique())]
    available_phase_labels = [PHASE_LABELS[code] for code in available_phase_codes]
    selected_phase_labels = st.multiselect("Competition Phase", available_phase_labels, default=available_phase_labels)
    selected_phases = [PHASE_CODES[label] for label in selected_phase_labels] or available_phase_codes
    selected_venue_label = st.selectbox("Venue", list(VENUE_CODES.keys()))
    selected_venue = VENUE_CODES[selected_venue_label]
    selected_team_names = st.multiselect("Focus Teams", sorted(team_lookup.keys()))
    selected_team_codes = [team_lookup[name] for name in selected_team_names]
    use_dates = st.toggle("Date Precision")
    start_date = end_date = None
    if use_dates and min_date is not None and max_date is not None:
        dates = st.date_input("Range", value=(min_date.date(), max_date.date()), min_value=min_date.date(), max_value=max_date.date())
        if isinstance(dates, tuple) and len(dates) == 2:
            start_date, end_date = pd.Timestamp(dates[0]), pd.Timestamp(dates[1])

filtered_players = filter_logs(players, selected_seasons, start_date, end_date, selected_team_codes, selected_phases, selected_venue)
filtered_teams = filter_logs(teams, selected_seasons, start_date, end_date, selected_team_codes, selected_phases, selected_venue)
filtered_shots = filter_logs(shots, selected_seasons, start_date, end_date, selected_team_codes, selected_phases, selected_venue) if not shots.empty else shots

if selected_search:
    search_kind, search_value = selected_search.split(" | ", 1)
    if search_kind == "Team":
        search_team_codes = teams.loc[teams["team_name"] == search_value, "team_code"].dropna().unique().tolist()
        filtered_players = filtered_players[filtered_players["team_code"].isin(search_team_codes)].copy()
        filtered_teams = filtered_teams[filtered_teams["team_code"].isin(search_team_codes)].copy()
        if not filtered_shots.empty:
            filtered_shots = filtered_shots[filtered_shots["team_code"].isin(search_team_codes)].copy()
    else:
        player_mask = filtered_players["player_name"].eq(search_value)
        player_team_codes = filtered_players.loc[player_mask, "team_code"].dropna().unique().tolist()
        filtered_players = filtered_players[player_mask].copy()
        filtered_teams = filtered_teams[filtered_teams["team_code"].isin(player_team_codes)].copy()
        if not filtered_shots.empty:
            filtered_shots = filtered_shots[filtered_shots["player_name"].eq(search_value)].copy()

st.markdown(
    """
    <div class="topbar">
        <div class="topbar-title">EUROLEAGUE INTELLIGENCE</div>
        <div style="display:flex; gap:10px; align-items:center;">
            <div class="topbar-pill">LOCAL CACHE</div>
            <div class="topbar-pill">2023-2026</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


def overview_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">EuroLeague Analytics Dashboard</div>
            <div class="hero-copy">Comprehensive statistical overview of recent EuroLeague seasons. Track team performance, player metrics, league trends and shot distribution from the local data cache.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_cards(
        [
            ("Games", f"{filtered_teams['game_code'].nunique():,}"),
            ("Player Games", f"{len(filtered_players):,}"),
            ("Teams", f"{filtered_teams['team_code'].nunique():,}"),
            ("Players", f"{filtered_players['player_name'].nunique():,}"),
            ("Total Shots", f"{len(filtered_shots):,}"),
        ]
    )
    board = team_leaderboard(filtered_teams).head(12)
    if board.empty:
        board_display = pd.DataFrame(columns=["RK", "Code", "Team", "GP", "W", "L", "WIN %", "+/-", "PIR"])
    else:
        board_display = board.rename(
            columns={"rank": "RK", "team_name": "Team", "games": "GP", "wins": "W", "losses": "L", "win_pct": "WIN %", "point_diff": "+/-", "pir_for": "PIR"}
        )[["RK", "team_code", "Team", "GP", "W", "L", "WIN %", "+/-", "PIR"]].rename(columns={"team_code": "Code"})
    top_players = (
        filtered_players.groupby(["player_id", "player_name", "team_name"], as_index=False)
        .agg(GP=("game_code", "nunique"), PIR=("pir", "mean"), PTS=("points", "mean"))
        .sort_values("PIR", ascending=False)
    )
    if not top_players.empty:
        min_games = 5 if top_players["GP"].ge(5).any() else 1
        top_players = (
            top_players.query("GP >= @min_games")
            .head(10)
            .rename(columns={"player_name": "Player", "team_name": "Team"})
            .round(2)
        )
    left, right = st.columns([1.4, 1])
    with left:
        st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Team Leaderboard</div><div style="color:var(--muted);font-weight:800;">Ranked by wins</div></div>', unsafe_allow_html=True)
        render_html_table(board_display.head(8), height=450, logo_col="Team")
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Top Players by PIR</div><div style="color:#9fb0c8;">Avg / Game</div></div>', unsafe_allow_html=True)
        st.markdown(top_players_html(top_players.head(5), profiles), unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    season_trend = filtered_teams.groupby("season", as_index=False).agg(Points=("points", "mean"), PIR=("pir", "mean")).round(2)
    season_trend["Period"] = season_trend["season"].map(season_axis_label)
    fig = px.line(season_trend, x="Period", y=["Points", "PIR"], markers=True, color_discrete_sequence=["#f26a21", "#2f80ed"])
    styled = style_line_chart(fig, "League Averages by Season", 440)
    styled.update_xaxes(type="category", categoryorder="array", categoryarray=season_trend["Period"].tolist())
    st.plotly_chart(styled, width="stretch")


def player_page() -> None:
    if filtered_players.empty:
        st.info("No players match the current filters.")
        return
    selected = st.selectbox("Select Athlete", sorted(filtered_players["player_name"].dropna().unique().tolist()))
    rows = filtered_players[filtered_players["player_name"] == selected].copy()
    avg = stat_average(rows, PLAYER_STATS)
    avg_min = average_minutes(rows)
    dunkest = dunkest_row_for_player(selected)
    selected_player_id = str(rows.sort_values(["season", "game_code"], ascending=[False, False]).iloc[0]["player_id"])
    photo = player_photo_uri_for_name(selected_player_id, selected, profiles)
    show_identity(selected, "Professional Athlete", logo_uri=photo or None, initials=selected[:1], headshot=bool(photo))
    role = with_player_roles(rows)["player_role"].mode()
    role_text = role.iloc[0] if not role.empty else "Forward"
    metric_cards(
        [
            ("Games", f"{rows['game_code'].nunique():,}"),
            ("Points", f"{avg.get('points', 0):.1f}"),
            ("PIR", f"{avg.get('pir', 0):.1f}"),
            ("Minutes", f"{avg_min:.1f}"),
            ("Dunkest CR", f"{float(dunkest.get('Dunkest CR')):.1f}" if pd.notna(dunkest.get("Dunkest CR", pd.NA)) else "-"),
            ("Rebounds", f"{avg.get('total_rebounds', 0):.1f}"),
            ("Assists", f"{avg.get('assists', 0):.1f}"),
        ]
    )
    st.markdown(
        f'<div class="section-card"><div class="section-card-header"><div><div class="section-title">Player Form</div><div style="color:#9fb0c8;margin-top:4px;">Role proxy: {html.escape(role_text)}. Rolling view from recent games.</div></div></div>',
        unsafe_allow_html=True,
    )
    f1, f2 = st.columns([1, 1.2])
    with f1:
        render_html_table(player_form_table(rows), height=190)
    with f2:
        recent = rows.sort_values(["parsed_date", "season", "game_code"]).tail(10).copy()
        recent["Game"] = recent["parsed_date"].dt.strftime("%d/%m/%y").fillna(recent["game_code"].astype(str))
        fig = px.line(
            recent,
            x="Game",
            y=["pir", "points", "assists", "total_rebounds"],
            markers=True,
            color_discrete_map={
                "pir": "#2f80ed",
                "points": "#f26a21",
                "assists": "#06b6d4",
                "total_rebounds": "#c084fc",
            },
        )
        st.plotly_chart(style_line_chart(fig, "Last 10 Game Trend", 320), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    season_avg, _ = season_and_total_averages(rows, "player_name", selected)
    season_minutes = rows.groupby("season")["minutes"].apply(lambda values: values.map(minutes_to_float).mean()).reset_index(name="minutes_avg")
    season_avg = season_avg.merge(season_minutes, on="season", how="left")
    left, right = st.columns([1, 1.2])
    with left:
        table = season_avg.rename(columns={"season": "Season", **{c: STAT_LABELS.get(c, c) for c in season_avg.columns}})
        table = table.rename(columns={"minutes_avg": "MIN"})
        render_html_table(table, height=360)
    with right:
        chart_cols = [c for c in ["points", "pir", "minutes_avg", "assists", "total_rebounds", "steals", "blocks_favour", "turnovers"] if c in season_avg.columns]
        trend = season_avg.sort_values("season").rename(columns={c: STAT_LABELS.get(c, c) for c in chart_cols})
        trend["Season"] = trend["season"].map(season_axis_label)
        y_cols = [STAT_LABELS.get(c, c) for c in chart_cols]
        fig = px.line(
            trend,
            x="Season",
            y=y_cols,
            markers=True,
            color_discrete_map={
                "Points": "#f26a21",
                "PIR": "#2f80ed",
                "Minutes": "#f6c344",
                "Assists": "#06b6d4",
                "Rebounds": "#c084fc",
                "Steals": "#14b8a6",
                "Blocks": "#f6c344",
                "Turnovers": "#e05d5d",
            },
        )
        styled = style_line_chart(fig, "Player Trend by Season", 430)
        styled.update_xaxes(type="category", categoryorder="array", categoryarray=trend["Season"].tolist())
        st.plotly_chart(styled, width="stretch")
    game_log = rows[["season", "phase", "group_name", "team_name", "opponent_name", "minutes", "points", "total_rebounds", "assists", "steals", "blocks_favour", "turnovers", "fg_pct", "three_pct", "pir", "plus_minus"]].sort_values("season", ascending=False)
    game_log["phase"] = game_log["phase"].map(PHASE_LABELS).fillna(game_log["phase"])
    game_log = game_log.rename(columns={"season": "Season", "phase": "Phase", "group_name": "Stage", "team_name": "Team", "opponent_name": "Opponent", "minutes": "MIN", "points": "PTS", "total_rebounds": "REB", "assists": "AST", "steals": "STL", "blocks_favour": "BLK", "turnovers": "TOV", "fg_pct": "FG%", "three_pct": "3PT%", "pir": "PIR", "plus_minus": "+/-"})
    render_html_table(game_log, height=520)


def team_page() -> None:
    if filtered_teams.empty:
        st.info("No teams match the current filters.")
        return
    selected = st.selectbox("Select Franchise", sorted(filtered_teams["team_name"].dropna().unique().tolist()))
    code = team_lookup[selected]
    rows = filtered_teams[filtered_teams["team_code"] == code].copy()
    avg = stat_average(rows, TEAM_STATS)
    wins = int(rows["won"].sum())
    games = int(rows["game_code"].nunique())
    streak = winning_streak(rows)
    show_identity(selected, "EuroLeague Franchise", logo_uri=team_logo_uri(code))
    metric_cards(
        [
            ("Record", f"{wins}-{games - wins}"),
            ("Win Rate", f"{wins / games * 100:.1f}%" if games else "0.0%"),
            ("Winning Streak", f"W{streak}" if streak else "0"),
            ("Points", f"{avg.get('points', 0):.1f}"),
            ("PIR", f"{avg.get('pir', 0):.1f}"),
            ("PIR Allowed", f"{avg.get('pir_allowed', 0):.1f}"),
        ]
    )
    board = team_leaderboard(filtered_teams).rename(columns={"rank": "RK", "team_name": "Team", "games": "GP", "wins": "W", "losses": "L", "win_pct": "WIN %", "point_diff": "+/-", "pir_for": "PIR", "pir_against": "PIR Allowed"})
    board = board[["RK", "team_code", "Team", "GP", "W", "L", "WIN %", "+/-", "PIR", "PIR Allowed"]].rename(columns={"team_code": "Code"})
    render_html_table(board, height=430, logo_col="Team")
    allowance = opponent_role_allowance(filtered_players, code)
    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Opponent View by Role</div><div style="color:#9fb0c8;margin-top:4px;">What this team allows to guard, forward and center profiles</div></div></div>', unsafe_allow_html=True)
    if allowance.empty:
        st.info("No opponent role data available for the current filters.")
    else:
        a1, a2 = st.columns([1, 1.2])
        with a1:
            render_html_table(allowance, height=230)
        with a2:
            role_plot = allowance.melt(id_vars="Role", value_vars=["PIR_Allowed", "PTS_Allowed", "REB_Allowed", "AST_Allowed"], var_name="Metric", value_name="Value")
            role_plot["Metric"] = role_plot["Metric"].replace({"PIR_Allowed": "PIR", "PTS_Allowed": "Points", "REB_Allowed": "Rebounds", "AST_Allowed": "Assists"})
            fig = px.bar(role_plot, y="Role", x="Value", color="Metric", text="Value", barmode="group", color_discrete_map={"PIR": "#f26a21", "Points": "#2f80ed", "Rebounds": "#8b5cf6", "Assists": "#16a3b8"})
            styled = style_bar_chart(fig, "Allowed Production by Role", 380, horizontal=True)
            styled.update_yaxes(categoryorder="array", categoryarray=["Guard", "Forward", "Center"])
            styled.update_traces(hovertemplate="<b>%{y}</b><br>%{fullData.name}: %{x:.2f}<extra></extra>")
            st.plotly_chart(styled, width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    season = rows.groupby("season", as_index=False).agg(Points=("points", "mean"), Points_Allowed=("points_allowed", "mean"), PIR=("pir", "mean"), PIR_Allowed=("pir_allowed", "mean")).round(2)
    season["Period"] = season["season"].map(season_axis_label)
    fig = px.line(season, x="Period", y=["Points", "Points_Allowed", "PIR", "PIR_Allowed"], markers=True, color_discrete_sequence=["#f26a21", "#d85c5c", "#2f80ed", "#7e8ca2"])
    styled = style_line_chart(fig, "Team Trend by Season", 420)
    styled.update_xaxes(type="category", categoryorder="array", categoryarray=season["Period"].tolist())
    st.plotly_chart(styled, width="stretch")
    game_log = rows[["season", "phase", "group_name", "team_name", "opponent_name", "points", "points_allowed", "point_diff", "total_rebounds", "assists", "pir", "pir_allowed", "won"]].sort_values("season", ascending=False)
    game_log["phase"] = game_log["phase"].map(PHASE_LABELS).fillna(game_log["phase"])
    game_log = game_log.rename(columns={"season": "Season", "phase": "Phase", "group_name": "Stage", "team_name": "Team", "opponent_name": "Opponent", "points": "PTS", "points_allowed": "PTS Allowed", "point_diff": "+/-", "total_rebounds": "REB", "assists": "AST", "pir": "PIR", "pir_allowed": "PIR Allowed", "won": "Win"})
    render_html_table(game_log, height=500)


def shot_page() -> None:
    if filtered_shots.empty:
        st.info("No shot data available.")
        return
    c1, c2, c3 = st.columns(3)
    shot_team = c1.selectbox("Shot Team", ["All"] + sorted(filtered_teams["team_name"].dropna().unique().tolist()))
    view = filtered_shots.copy()
    if shot_team != "All":
        view = view[view["team_code"] == team_lookup[shot_team]]
    player = c2.selectbox("Shot Player", ["All"] + sorted(view["player_name"].dropna().unique().tolist()))
    result = c3.selectbox("Result", ["All", "Made", "Missed"])
    if player != "All":
        view = view[view["player_name"] == player]
    if result == "Made":
        view = view[view["made"]]
    elif result == "Missed":
        view = view[~view["made"]]
    field_goal_mask = ~view.get("is_ft", pd.Series(False, index=view.index))
    attempts = int(field_goal_mask.sum())
    made = int((field_goal_mask & view["made"]).sum())
    three_attempts = int(view.get("is_3pt", pd.Series(False, index=view.index)).sum())
    three_made = int((view.get("is_3pt", pd.Series(False, index=view.index)) & view["made"]).sum())
    ft_attempts = int(view.get("is_ft", pd.Series(False, index=view.index)).sum())
    ft_made = int((view.get("is_ft", pd.Series(False, index=view.index)) & view["made"]).sum())
    metric_cards([
        ("Attempts", f"{attempts:,}"),
        ("Made", f"{made:,}"),
        ("FG%", f"{made / attempts * 100:.1f}%" if attempts else "0.0%"),
        ("3PT%", f"{three_made / three_attempts * 100:.1f}%" if three_attempts else "0.0%"),
        ("FT%", f"{ft_made / ft_attempts * 100:.1f}%" if ft_attempts else "0.0%"),
        ("Zones", f"{view['zone'].nunique():,}"),
    ])
    shot_view = view.dropna(subset=["coord_x", "coord_y"]).copy()
    shot_view["Season"] = shot_view["season"].map(season_label)
    fig = px.scatter(shot_view, x="coord_x", y="coord_y", color="made", color_discrete_map={True: "#14b8a6", False: "#e05d5d"}, opacity=.62, hover_data=["Season", "team_code", "player_name", "action", "zone"])
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(**plotly_dark(640), title="Field Goal Distribution", xaxis_visible=False, yaxis_visible=False)
    st.plotly_chart(polish_plotly_text(fig), width="stretch")


def ml_pir_page() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">ML PIR Predictor</div>
            <div class="hero-copy">RidgeStacking model with minutes prediction, calibration and conformal ranges. Rank players by role for a selected matchup context, or inspect one player in detail.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if filtered_players.empty:
        st.info("No players match the current filters.")
        return

    schedule = load_ml_schedule()
    candidate_base = current_roster_table(players, profiles)
    if candidate_base.empty:
        st.info("No current roster candidates found. Refresh player profiles first.")
        return
    candidate_base = apply_player_availability(candidate_base)

    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Prediction Context</div><div style="color:#9fb0c8;margin-top:4px;">Reads matchup context from the official EuroLeague schedule when available, then falls back to local historical fixtures.</div></div></div>', unsafe_allow_html=True)
    if schedule.empty:
        st.info("No schedule rows found in the database.")
        return
    today = pd.Timestamp.now().normalize()
    future = schedule[(schedule["played"] == 0) & schedule["parsed_date"].notna() & (schedule["parsed_date"] >= today)].copy()
    upcoming = future if not future.empty else schedule[schedule["played"] == 0].copy()
    schedule_source = "Official upcoming fixtures" if not upcoming.empty else "Historical fixtures"
    schedule_view = upcoming if not upcoming.empty else schedule.copy()
    season_options = sorted(schedule_view["season"].dropna().astype(int).unique().tolist(), reverse=True)
    c1, c2, c3, c4 = st.columns([.85, .85, 1.35, .9])
    default_season = int(schedule_view.sort_values("parsed_date").iloc[0]["season"]) if not upcoming.empty else season_options[0]
    season_index = season_options.index(default_season) if default_season in season_options else 0
    schedule_season = int(c1.selectbox("Season", season_options, index=season_index, key="ml_schedule_season", format_func=season_label))
    season_schedule = schedule_view[schedule_view["season"].astype(int) == schedule_season].copy()
    round_options = sorted(season_schedule["round"].dropna().astype(int).unique().tolist(), reverse=upcoming.empty)
    if not upcoming.empty and not season_schedule.empty:
        default_round = int(season_schedule.sort_values("parsed_date").iloc[0]["round"])
    else:
        default_round = round_options[0]
    round_index = round_options.index(default_round) if default_round in round_options else 0
    selected_round = int(c2.selectbox("Round", round_options, index=round_index, key="ml_schedule_round"))
    round_games = season_schedule[season_schedule["round"].astype(int) == selected_round].copy()
    round_games = round_games.sort_values(["parsed_date", "game_code"], ascending=[True, True])
    fixture_labels = ["All fixtures in round"] + [
        f"{row.home_code} vs {row.away_code} ({pd.to_datetime(row.parsed_date).strftime('%d/%m %H:%M') if pd.notna(row.parsed_date) else 'TBD'} - {PHASE_LABELS.get(row.phase, row.phase)} - G{int(row.game_code)})"
        for row in round_games.itertuples(index=False)
    ]
    selected_fixture_label = c3.selectbox("Fixture", fixture_labels, key="ml_fixture")
    phase_label = c4.selectbox("Phase Override", ["From schedule"] + list(PHASE_LABELS.values()), index=0, key="ml_phase_override")
    c5, c6, c7 = st.columns([1.05, 1, 1])
    min_gp = int(c5.slider("Minimum Games", min_value=1, max_value=20, value=5, step=1, key="ml_min_gp"))
    roster_scope = c6.selectbox("Roster Scope", ["Fixture teams", "All current rosters"], key="ml_roster_scope")
    refresh = c7.button("Refresh ML Cache", key="ml_refresh")
    if refresh:
        cached_ml_predictions.clear()
        load_ml_schedule.clear()
        load_player_availability.clear()
        fetch_basketstories_availability.clear()
        fetch_news_availability.clear()
        clear_ml_caches()
        st.success("ML cache refreshed.")
    unavailable_view = candidate_base[pd.to_numeric(candidate_base["Availability Impact"], errors="coerce").fillna(1.0) <= 0.65].copy()
    if not unavailable_view.empty:
        st.markdown(
            '<div style="color:#9fb0c8;margin:8px 24px 2px;font-weight:800;">Availability adjustments loaded automatically, with local CSV overrides when present.</div>',
            unsafe_allow_html=True,
        )
        render_html_table(
            unavailable_view[["player_id", "player_name", "team_code", "player_role", "Availability Status", "Availability Impact", "Availability Source", "Availability Updated", "Availability Note"]]
            .rename(columns={"player_name": "Player", "team_code": "Team", "player_role": "Role"})
            .round(2),
            height=170,
            player_col="Player",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    candidates = candidate_base[candidate_base["GP"] >= min_gp].copy()
    candidates = candidates[pd.to_numeric(candidates["Availability Impact"], errors="coerce").fillna(1.0) > 0.05].copy()
    if selected_fixture_label != "All fixtures in round":
        selected_idx = fixture_labels.index(selected_fixture_label) - 1
        prediction_games = round_games.iloc[[selected_idx]].copy()
    else:
        prediction_games = round_games.copy()

    if prediction_games.empty:
        st.info("No fixture available for the selected round.")
        return

    if roster_scope == "Fixture teams":
        fixture_team_codes = set(prediction_games["home_code"].dropna().astype(str)) | set(prediction_games["away_code"].dropna().astype(str))
        candidates = candidates[candidates["team_code"].astype(str).isin(fixture_team_codes)].copy()

    if candidates.empty:
        st.info("No current roster players match the selected fixture/round.")
        return

    prediction_frames = []
    with st.spinner("Running model predictions..."):
        for game in prediction_games.itertuples(index=False):
            fixture_phase = PHASE_CODES[phase_label] if phase_label != "From schedule" else str(game.phase)
            matchups = [
                (str(game.home_code), str(game.away_code), 1),
                (str(game.away_code), str(game.home_code), 0),
            ]
            for team_code, opponent_code, home_flag in matchups:
                team_candidates = candidates[candidates["team_code"].astype(str) == team_code].copy()
                if team_candidates.empty:
                    continue
                rest_days = schedule_rest_days(schedule, int(game.season), int(game.game_code), team_code)
                player_ids = tuple(team_candidates["player_id"].astype(str).tolist())
                frame = cached_ml_predictions(player_ids, opponent_code, home_flag, fixture_phase, rest_days, team_code)
                if frame.empty:
                    continue
                frame["fixture"] = f"{game.home_code} vs {game.away_code}"
                frame["rest_days"] = rest_days
                frame["game_code"] = int(game.game_code)
                frame["round"] = int(game.round)
                frame["fixture_phase"] = fixture_phase
                frame["fixture_source"] = schedule_source
                prediction_frames.append(frame)
    predictions = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()

    if predictions.empty:
        st.warning("The model did not return predictions.")
        return
    if "error" in predictions.columns and predictions["error"].notna().all():
        st.error("All ML predictions failed. Check model artifacts and feature runtime.")
        render_html_table(predictions[["player_id", "error"]], height=260)
        return

    predictions = predictions[predictions.get("error", pd.Series(index=predictions.index)).isna()].copy()
    for col, default in {
        "Availability Status": "available",
        "Availability Impact": 1.0,
        "Availability Note": "",
        "Availability Source": "",
        "Availability Updated": "",
    }.items():
        if col not in candidates.columns:
            candidates[col] = default
        if col not in candidate_base.columns:
            candidate_base[col] = default

    candidate_cols = ["player_id", "team_name", "player_role", "GP", "PIR", "MIN", "Availability Status", "Availability Impact", "Availability Note", "Availability Source", "Availability Updated"]
    candidate_cols.extend([col for col in ["Dunkest CR", "Dunkest PDK", "Dunkest GP", "Dunkest PLUS"] if col in candidates.columns])
    predictions = predictions.merge(
        candidates[candidate_cols],
        on="player_id",
        how="left",
    )
    predictions["Team"] = predictions["team_name"].fillna(predictions["team_code"])
    predictions["Role"] = predictions["player_role"].fillna("Forward")
    predictions["Player"] = predictions["player_name"]
    predictions["Expected PIR"] = predictions["predicted_pir"].astype(float)
    predictions["Low"] = predictions["interval_low"].astype(float)
    predictions["High"] = predictions["interval_high"].astype(float)
    predictions["Pred MIN"] = predictions["predicted_minutes"].astype(float)
    predictions["Confidence"] = predictions["confidence"]
    predictions["H2H"] = predictions["h2h_avg_pir"].astype(float)
    predictions["H2H GP"] = predictions["h2h_games"].astype(int)
    predictions = apply_availability_projection_adjustments(predictions, candidate_base)
    for col, default in {
        "Availability Status": "available",
        "Availability Impact": 1.0,
        "Availability Note": "",
        "Availability Source": "",
        "Availability Updated": "",
        "Injury Boost MIN": 0.0,
        "Injury Adj PIR": 0.0,
    }.items():
        if col not in predictions.columns:
            predictions[col] = default
    boost_map = projected_win_boosts(teams, prediction_games)
    margin_map = projected_team_margins(teams, prediction_games)
    predictions["Win Boost"] = predictions.apply(
        lambda row: boost_map.get((int(row["game_code"]), str(row["team_code"])), 1.0),
        axis=1,
    )
    predictions["Projected Team Margin"] = predictions.apply(
        lambda row: margin_map.get((int(row["game_code"]), str(row["team_code"])), 0.0),
        axis=1,
    )
    predictions["Coach Projection"] = predictions["Projected Team Margin"].apply(fantasy_coach_score_from_margin)
    predictions["Fantasy PIR"] = predictions["Expected PIR"] * predictions["Win Boost"]
    predictions = enrich_ml_predictions(predictions)
    predictions = predictions.sort_values("Fantasy Score", ascending=False)

    metric_cards(
        [
            ("Predicted Players", f"{len(predictions):,}"),
            ("Best ML Score", f"{predictions['Fantasy Score'].max():.1f}"),
            ("Best Raw PIR", f"{predictions['Fantasy PIR'].max():.1f}"),
            ("Avg Floor", f"{predictions['Floor PIR'].mean():.1f}"),
            ("Round", f"{season_label(schedule_season)} / {selected_round}"),
            ("Schedule", schedule_source),
            ("Rest Days", "From schedule"),
            ("Model", str(predictions["model_name"].dropna().iloc[0]) if "model_name" in predictions and not predictions["model_name"].dropna().empty else "RidgeStacking"),
        ]
    )

    table_cols = ["player_id", "fixture", "Player", "Team", "Role", "Fantasy Score", "Fantasy PIR", "Expected PIR", "model_expected_pir", "tail_risk_adjustment", "high_pir_probability", "low_pir_probability", "context_adjustment", "Floor PIR", "Upside PIR", "Risk", "Confidence Score", "Win Boost", "Projected Team Margin", "Coach Projection", "Pred MIN", "Injury Boost MIN", "Injury Adj PIR", "Availability Status", "Availability Source", "Rest"]
    table_cols.extend([col for col in ["Dunkest CR", "Dunkest PDK"] if col in predictions.columns])
    table_cols.extend(["Value Score", "recent_3_pir", "recent_5_pir", "pir_volatility", "H2H", "H2H GP", "GP", "PIR", "MIN", "ML Note"])
    predictions["Rest"] = predictions["rest_days"].astype(int)
    table_cols = [col for col in table_cols if col in predictions.columns]
    role_specs = [("Center", 1, "Top Center"), ("Forward", 3, "Top 3 Forwards"), ("Guard", 3, "Top 3 Guards")]
    for role_name, limit, title in role_specs:
        role_table = predictions[predictions["Role"] == role_name][table_cols].head(limit).round(2)
        role_table = role_table.rename(columns={"fixture": "Fixture"})
        st.markdown(
            f'<div class="section-card"><div class="section-card-header"><div><div class="section-title">{title}</div><div style="color:#9fb0c8;margin-top:4px;">Fantasy PIR includes projected +10% team-win boost, round {selected_round}</div></div></div>',
            unsafe_allow_html=True,
        )
        if role_table.empty:
            st.info(f"No {role_name.lower()} candidates in the current pool.")
        else:
            render_html_table(role_table, height=250, player_col="Player")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Full Player Prediction Pool</div><div style="color:#9fb0c8;margin-top:4px;">Use this as the manual-swap board: every player keeps the ML projection, fantasy score, credits, risk and availability context.</div></div></div>', unsafe_allow_html=True)
    render_html_table(predictions[table_cols].rename(columns={"fixture": "Fixture"}).round(2), height=430, player_col="Player", max_rows=120)
    st.markdown("</div>", unsafe_allow_html=True)

    coach_predictions = coach_candidates_for_predictions(predictions)
    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Coach Predictions</div><div style="color:#9fb0c8;margin-top:4px;">Coach score follows the official fantasy margin buckets: +25, +20, +10, -5, -10, -20.</div></div></div>', unsafe_allow_html=True)
    if coach_predictions.empty:
        st.info("No coach candidates available for the selected fixtures.")
    else:
        coach_table = coach_predictions[["player_id", "Player", "Team", "Team Code", "Credits", "Projection", "Projected Margin", "Coach Source"]].rename(
            columns={"Projection": "Coach Projection"}
        )
        render_html_table(coach_table.round(2), height=260, player_col="Player", max_rows=30)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Fantasy Lineup Optimizer</div><div style="color:#9fb0c8;margin-top:4px;">Exact optimizer: 10 players + 1 coach, 4 guards, 4 forwards, 2 centers. Captain x2, starters and sixth man 100%, bench 50%, coach by projected margin.</div></div></div>', unsafe_allow_html=True)
    lineup_tabs = st.tabs(["100 CR", "105 CR"])
    for container, cap in zip(lineup_tabs, [100.0, 105.0]):
        lineup, lineup_summary = optimize_fantasy_lineup_with_coach(predictions, cap)
        with container:
            if lineup.empty:
                st.info(f"No valid lineup under {cap:.0f} credits. Dunkest credits may be unavailable.")
            else:
                render_fantasy_lineup_court(lineup, lineup_summary, profiles, f"Cap {cap:.0f} CR lineup")
                with st.expander(f"Lineup table - cap {cap:.0f}"):
                    table_lineup = lineup.drop(columns=["Coach Source"], errors="ignore")
                    render_html_table(table_lineup, height=360, player_col="Player")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">Single Player Inspector</div><div style="color:#9fb0c8;margin-top:4px;">Select a player and optionally override starter status.</div></div></div>', unsafe_allow_html=True)
    s1, s2, s3 = st.columns([1.5, .8, .8])
    inspector_options = predictions.sort_values(["Player", "fixture"]).copy()
    inspector_options["Inspector Label"] = inspector_options.apply(
        lambda row: f"{row['Player']} | {row['Team']} | {row['fixture']}",
        axis=1,
    )
    selected_player = s1.selectbox("Player", inspector_options["Inspector Label"].tolist(), key="ml_selected_player")
    starter_mode = s2.selectbox("Starter Override", ["Auto", "Starter", "Bench"], key="ml_starter_override")
    run_single = s3.button("Run Player", key="ml_run_single")
    selected_row = inspector_options[inspector_options["Inspector Label"] == selected_player].iloc[0]
    selected_player_id = str(selected_row["player_id"])
    selected_opponent_code = str(selected_row["opponent_code"])
    selected_home = int(selected_row["home"])
    selected_phase = str(selected_row["fixture_phase"])
    starter_override = None if starter_mode == "Auto" else 1 if starter_mode == "Starter" else 0
    if run_single or selected_player:
        with st.spinner("Scoring player..."):
            selected_rest_days = int(selected_row.get("rest_days", 4))
            result = predict_player(selected_player_id, selected_opponent_code, selected_home, selected_phase, starter_override, selected_rest_days, str(selected_row["team_code"]))
        show_ml_player_identity(result["player_name"], f"{result['team_code']} vs {result['opponent_code']} - {result['phase']}", selected_player_id)
        metric_cards(
            [
                ("Expected PIR", f"{result['predicted_pir']:.1f}"),
                ("ML Score", f"{float(selected_row.get('Fantasy Score', selected_row.get('Fantasy PIR', result['predicted_pir']))):.1f}"),
                ("Range", f"{result['interval_low']:.1f}-{result['interval_high']:.1f}"),
                ("Risk", str(selected_row.get("Risk", result["confidence"]))),
                ("Confidence", f"{float(selected_row.get('Confidence Score', 0)):.0f}/100"),
                ("Pred Minutes", f"{result['predicted_minutes']:.1f}"),
                ("Rest Days", str(selected_rest_days)),
                ("H2H", f"{result['h2h_avg_pir']:.1f} / {result['h2h_games']} GP"),
            ]
        )
        details = pd.DataFrame([{
            "Player": result["player_name"],
            "Team": result["team_code"],
            "Opponent": result["opponent_code"],
            "Venue": "Home" if result["home"] else "Away",
            "Starter": "Yes" if result["is_starter"] else "No",
            "Raw PIR": result["raw_predicted_pir"],
            "Calibration": result["calibration_adjustment"],
            "Model PIR": result.get("model_expected_pir"),
            "Tail Risk Adj": result.get("tail_risk_adjustment"),
            "20+ PIR Prob": result.get("high_pir_probability"),
            "<=0 PIR Prob": result.get("low_pir_probability"),
            "Context Adj": result.get("context_adjustment"),
            "Expected PIR": result["predicted_pir"],
            "Recent 3 PIR": result.get("recent_3_pir"),
            "Recent 5 PIR": result.get("recent_5_pir"),
            "PIR Volatility": result.get("pir_volatility"),
            "Low": result["interval_low"],
            "High": result["interval_high"],
            "Confidence": result["confidence"],
        }])
        render_html_table(details, height=160)

        player_history = players[players["player_id"].astype(str) == selected_player_id].copy()
        player_history = player_history.sort_values(["parsed_date", "season", "game_code"], ascending=False)
        form_cols = ["parsed_date", "team_name", "opponent_name", "home", "minutes", "points", "total_rebounds", "assists", "steals", "turnovers", "pir"]
        last5 = player_history[[col for col in form_cols if col in player_history.columns]].head(5).copy()
        if not last5.empty:
            last5["Date"] = pd.to_datetime(last5["parsed_date"], errors="coerce").dt.strftime("%d/%m/%Y")
            last5["Venue"] = last5["home"].fillna(0).astype(int).map({1: "Home", 0: "Away"})
            last5 = last5.rename(
                columns={
                    "team_name": "Team",
                    "opponent_name": "Opponent",
                    "minutes": "MIN",
                    "points": "PTS",
                    "total_rebounds": "REB",
                    "assists": "AST",
                    "steals": "STL",
                    "turnovers": "TOV",
                    "pir": "PIR",
                }
            )
            last5 = last5[["Date", "Team", "Opponent", "Venue", "MIN", "PTS", "REB", "AST", "STL", "TOV", "PIR"]]

        h2h = player_history[player_history["opponent_code"] == selected_opponent_code].head(2).copy()
        if not h2h.empty:
            h2h["Date"] = pd.to_datetime(h2h["parsed_date"], errors="coerce").dt.strftime("%d/%m/%Y")
            h2h["Venue"] = h2h["home"].fillna(0).astype(int).map({1: "Home", 0: "Away"})
            h2h = h2h.rename(
                columns={
                    "team_name": "Team",
                    "opponent_name": "Opponent",
                    "minutes": "MIN",
                    "points": "PTS",
                    "total_rebounds": "REB",
                    "assists": "AST",
                    "steals": "STL",
                    "turnovers": "TOV",
                    "pir": "PIR",
                }
            )
            h2h = h2h[["Date", "Team", "Opponent", "Venue", "MIN", "PTS", "REB", "AST", "STL", "TOV", "PIR"]]

        t1, t2 = st.columns([1.25, 1])
        with t1:
            st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Last 5 Games</div></div>', unsafe_allow_html=True)
            if last5.empty:
                st.info("No recent game log available.")
            else:
                render_html_table(last5, height=330)
            st.markdown("</div>", unsafe_allow_html=True)
        with t2:
            st.markdown(f'<div class="section-card"><div class="section-card-header"><div class="section-title">Last 2 vs {html.escape(selected_opponent_code)}</div></div>', unsafe_allow_html=True)
            if h2h.empty:
                st.info("No recent H2H games against this opponent.")
            else:
                render_html_table(h2h, height=250)
            st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


def match_center_page() -> None:
    if filtered_teams.empty:
        st.info("No games match the current filters.")
        return
    games = (
        filtered_teams[["season", "game_code", "phase", "group_name", "round", "parsed_date", "home_code", "away_code", "home_score", "away_score", "winner_code", "point_spread"]]
        .drop_duplicates(["season", "game_code"])
        .sort_values(["parsed_date", "season", "game_code"], ascending=[False, False, False])
        .copy()
    )
    games["label"] = games.apply(
        lambda row: f"{season_label(row['season'])} - {PHASE_LABELS.get(row['phase'], row['phase'])} - {code_to_team.get(row['home_code'], row['home_code'])} {int(row['home_score'])} - {int(row['away_score'])} {code_to_team.get(row['away_code'], row['away_code'])}",
        axis=1,
    )
    selected_label = st.selectbox("Select Game", games["label"].tolist())
    game = games[games["label"] == selected_label].iloc[0]
    game_key = (int(game["season"]), int(game["game_code"]))
    home_name = code_to_team.get(game["home_code"], game["home_code"])
    away_name = code_to_team.get(game["away_code"], game["away_code"])
    phase_name = PHASE_LABELS.get(game["phase"], game["phase"])
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">{html.escape(str(home_name))} vs {html.escape(str(away_name))}</div>
            <div class="hero-copy">{html.escape(str(phase_name))} - {html.escape(str(game['group_name']))} - Round {int(game['round']) if pd.notna(game['round']) else '-'} - {pd.Timestamp(game['parsed_date']).strftime('%d/%m/%Y') if pd.notna(game['parsed_date']) else ''}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    match_metric_cards(
        [
            ("Score", html.escape(f"{int(game['home_score'])}-{int(game['away_score'])}")),
            ("Winner", winner_logo_html(str(game["winner_code"]))),
            ("Spread", html.escape(f"{float(game['point_spread']):+.0f}")),
            ("Fixture", fixture_logo_html(str(game["home_code"]), str(game["away_code"]))),
            ("Phase", html.escape(str(phase_name))),
        ]
    )
    team_rows = filtered_teams[(filtered_teams["season"] == game_key[0]) & (filtered_teams["game_code"] == game_key[1])].copy()
    team_table = team_rows[["team_code", "team_name", "points", "points_allowed", "point_diff", "total_rebounds", "assists", "pir", "pir_allowed", "won"]].rename(columns={"team_code": "Code", "team_name": "Team", "points": "PTS", "points_allowed": "PTS Allowed", "point_diff": "+/-", "total_rebounds": "REB", "assists": "AST", "pir": "PIR", "pir_allowed": "PIR Allowed", "won": "Win"})
    left, right = st.columns([1, 1])
    with left:
        st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Team Boxscore</div></div>', unsafe_allow_html=True)
        render_html_table(team_table, height=210, logo_col="Team")
        st.markdown("</div>", unsafe_allow_html=True)
    player_rows = filtered_players[(filtered_players["season"] == game_key[0]) & (filtered_players["game_code"] == game_key[1])].copy()
    leaders = (
        player_rows[["player_id", "player_name", "team_name", "minutes", "points", "total_rebounds", "assists", "pir", "plus_minus"]]
        .sort_values("pir", ascending=False)
        .head(12)
        .rename(columns={"player_name": "Player", "team_name": "Team", "minutes": "MIN", "points": "PTS", "total_rebounds": "REB", "assists": "AST", "pir": "PIR", "plus_minus": "+/-"})
    )
    with right:
        st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Player Leaders</div></div>', unsafe_allow_html=True)
        render_html_table(leaders, height=360, player_col="Player")
        st.markdown("</div>", unsafe_allow_html=True)
    role_pir = game_role_pir_table(player_rows)
    if not role_pir.empty:
        st.markdown('<div class="section-card"><div class="section-card-header"><div><div class="section-title">PIR by Role</div><div style="color:#9fb0c8;margin-top:4px;">PIR created and conceded by guard, forward and center profiles in this game</div></div></div>', unsafe_allow_html=True)
        r1, r2 = st.columns([1, 1.25])
        with r1:
            render_html_table(role_pir, height=260)
        with r2:
            role_plot = role_pir.melt(id_vars=["Team", "Role"], value_vars=["PIR Created", "PIR Allowed"], var_name="Metric", value_name="PIR")
            fig = px.bar(
                role_plot,
                x="Role",
                y="PIR",
                color="Metric",
                facet_col="Team",
                text="PIR",
                barmode="group",
                category_orders={"Role": ["Guards", "Forwards", "Centers"], "Metric": ["PIR Created", "PIR Allowed"]},
                color_discrete_map={"PIR Created": "#f26a21", "PIR Allowed": "#2f80ed"},
            )
            fig = style_bar_chart(fig, "Role PIR Split", 390)
            fig.update_traces(hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f} PIR<extra></extra>")
            fig.for_each_annotation(lambda annotation: annotation.update(text=annotation.text.replace("Team=", "")))
            st.plotly_chart(fig, width="stretch")
        st.markdown("</div>", unsafe_allow_html=True)
    if not filtered_shots.empty:
        game_shots = filtered_shots[(filtered_shots["season"] == game_key[0]) & (filtered_shots["game_code"] == game_key[1])]
        if not game_shots.empty:
            fig = px.scatter(game_shots.dropna(subset=["coord_x", "coord_y"]), x="coord_x", y="coord_y", color="made", color_discrete_map={True: "#14b8a6", False: "#e05d5d"}, opacity=.7, hover_data=["team_code", "player_name", "action", "zone"])
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(**plotly_dark(560), title="Game Shot Map", xaxis_visible=False, yaxis_visible=False)
            st.plotly_chart(polish_plotly_text(fig), width="stretch")


def compare_page() -> None:
    if filtered_teams.empty or filtered_players.empty:
        st.info("Comparison needs data in the current filters.")
        return
    left_color, right_color = compare_palette()
    st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Team Comparison</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    names = sorted(filtered_teams["team_name"].dropna().unique().tolist())
    a_name = c1.selectbox("Team A", names)
    b_name = c2.selectbox("Team B", names, index=min(1, len(names) - 1))
    a = filtered_teams[filtered_teams["team_name"] == a_name]
    b = filtered_teams[filtered_teams["team_name"] == b_name]
    team_cols = st.columns([.18, .64, .18])
    with team_cols[0]:
        st.html(compare_side_card(a_name, team_logo_uri(str(a["team_code"].dropna().iloc[0])) if not a.empty else "", left_color, "Team A", "logo"))
    with team_cols[1]:
        st.plotly_chart(polish_plotly_text(radar_chart(a_name, stat_average(a, TEAM_STATS), b_name, stat_average(b, TEAM_STATS), TEAM_STATS, "Team Spider Chart")), width="stretch")
    with team_cols[2]:
        st.html(compare_side_card(b_name, team_logo_uri(str(b["team_code"].dropna().iloc[0])) if not b.empty else "", right_color, "Team B", "logo"))
    team_compare = pd.DataFrame([{"Team": a_name, **stat_average(a, TEAM_STATS).to_dict()}, {"Team": b_name, **stat_average(b, TEAM_STATS).to_dict()}]).rename(columns={c: STAT_LABELS.get(c, c) for c in TEAM_STATS})
    render_html_table(team_compare, height=170, highlight=highlight_best_rows(team_compare, "Team", {"Pts Allowed", "PIR Allowed"}))
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="section-card"><div class="section-card-header"><div class="section-title">Player Comparison</div></div>', unsafe_allow_html=True)
    p1, p2 = st.columns(2)
    player_names = sorted(filtered_players["player_name"].dropna().unique().tolist())
    pa_name = p1.selectbox("Player A", player_names)
    pb_name = p2.selectbox("Player B", player_names, index=min(1, len(player_names) - 1))
    pa = filtered_players[filtered_players["player_name"] == pa_name]
    pb = filtered_players[filtered_players["player_name"] == pb_name]
    pa_id = str(pa.sort_values(["season", "game_code"]).iloc[-1]["player_id"]) if not pa.empty else ""
    pb_id = str(pb.sort_values(["season", "game_code"]).iloc[-1]["player_id"]) if not pb.empty else ""
    player_cols = st.columns([.18, .64, .18])
    with player_cols[0]:
        st.html(compare_side_card(pa_name, player_photo_uri_for_name(pa_id, pa_name, profiles), left_color, "Player A"))
    with player_cols[1]:
        st.plotly_chart(polish_plotly_text(radar_chart(pa_name, stat_average(pa, PLAYER_STATS), pb_name, stat_average(pb, PLAYER_STATS), PLAYER_STATS, "Player Spider Chart")), width="stretch")
    with player_cols[2]:
        st.html(compare_side_card(pb_name, player_photo_uri_for_name(pb_id, pb_name, profiles), right_color, "Player B"))
    player_compare = pd.DataFrame([{"Player": pa_name, **stat_average(pa, PLAYER_STATS).to_dict()}, {"Player": pb_name, **stat_average(pb, PLAYER_STATS).to_dict()}]).rename(columns={c: STAT_LABELS.get(c, c) for c in PLAYER_STATS})
    render_html_table(player_compare, height=170, highlight=highlight_best_rows(player_compare, "Player", {"Turnovers"}))
    st.markdown("</div>", unsafe_allow_html=True)


if page == "Overview":
    overview_page()
elif page == "Player Dashboard":
    player_page()
elif page == "Team Dashboard":
    team_page()
elif page == "Match Center":
    match_center_page()
elif page == "ML PIR Predictor":
    ml_pir_page()
elif page == "Shot Chart":
    shot_page()
else:
    compare_page()
