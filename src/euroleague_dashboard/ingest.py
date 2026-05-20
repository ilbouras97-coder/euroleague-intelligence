from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import requests
import typer
from rich.console import Console

from euroleague_api.EuroLeagueData import EuroLeagueData
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.player_stats import PlayerStats
from euroleague_api.shot_data import ShotData

from .config import COMPETITION, DB_PATH, DEFAULT_END_SEASON, DEFAULT_START_SEASON
from .config import PLAYER_PHOTO_DIR, RAW_DIR
from .storage import (
    begin_ingestion_run,
    connect,
    finish_ingestion_run,
    initialize_database,
    load_or_fetch_raw,
)


app = typer.Typer(help="Ingest Euroleague API data into local cache and SQLite.")
console = Console()

DATASETS = {"games", "player_boxscores", "team_quarter_scores", "shots", "player_profiles"}


def season_range(start_season: int, end_season: int) -> list[int]:
    if end_season < start_season:
        raise typer.BadParameter("end-season must be greater than or equal to start-season")
    return list(range(start_season, end_season + 1))


def normalize_games(df: pd.DataFrame, season: int) -> pd.DataFrame:
    games = df.copy()
    games["season"] = season
    games["winner_code"] = None
    played = games.get("played", False).astype(bool)
    home_win = games["homescore"] > games["awayscore"]
    games.loc[played & home_win, "winner_code"] = games.loc[played & home_win, "homecode"]
    games.loc[played & ~home_win, "winner_code"] = games.loc[played & ~home_win, "awaycode"]
    games["point_spread"] = games["homescore"] - games["awayscore"]

    return games.rename(
        columns={
            "gameCode": "game_code",
            "gamecode": "game_id",
            "Phase": "phase",
            "Round": "round",
            "date": "game_date",
            "time": "game_time",
            "group": "group_name",
            "hometeam": "home_team",
            "homecode": "home_code",
            "homescore": "home_score",
            "awayteam": "away_team",
            "awaycode": "away_code",
            "awayscore": "away_score",
        }
    )[
        [
            "season",
            "game_code",
            "game_id",
            "phase",
            "round",
            "game_date",
            "game_time",
            "group_name",
            "home_team",
            "home_code",
            "home_score",
            "away_team",
            "away_code",
            "away_score",
            "played",
            "winner_code",
            "point_spread",
        ]
    ]


def played_game_codes(games_raw: pd.DataFrame, max_games: int | None = None) -> list[int]:
    games = games_raw.copy()
    games = games[games["played"].astype(bool)].sort_values("gameCode")
    if max_games is not None:
        games = games.head(max_games)
    return [int(game_code) for game_code in games["gameCode"].tolist()]


def fetch_per_game(season: int, game_codes: list[int], fetch_game) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for index, game_code in enumerate(game_codes, start=1):
        console.print(f"Fetching game {game_code} ({index}/{len(game_codes)})")
        try:
            df = fetch_game(season, game_code)
        except Exception as exc:
            console.print(f"[yellow]Skipping game {game_code}: {exc}[/yellow]")
            continue
        if not df.empty:
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def fetch_per_game_cached(
    dataset: str,
    season: int,
    game_codes: list[int],
    fetch_game,
    force_refresh: bool = False,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    cache_dir = RAW_DIR / "games" / dataset / f"E{season}"
    cache_dir.mkdir(parents=True, exist_ok=True)

    for index, game_code in enumerate(game_codes, start=1):
        cache_path = cache_dir / f"{game_code}.parquet"
        console.print(f"{dataset}: game {game_code} ({index}/{len(game_codes)})")

        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
        else:
            try:
                df = fetch_game(season, game_code)
            except Exception as exc:
                console.print(f"[yellow]Skipping game {game_code}: {exc}[/yellow]")
                continue
            if not df.empty:
                df.to_parquet(cache_path, index=False)

        if not df.empty:
            frames.append(df)

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    season_cache = RAW_DIR / f"{dataset}_E{season}.parquet"
    if not combined.empty:
        combined.to_parquet(season_cache, index=False)
    return combined


def normalize_player_boxscores(df: pd.DataFrame) -> pd.DataFrame:
    players = df.copy()
    players = players.rename(
        columns={
            "Season": "season",
            "Gamecode": "game_code",
            "Home": "home",
            "Player_ID": "player_id",
            "IsStarter": "is_starter",
            "IsPlaying": "is_playing",
            "Team": "team_code",
            "Dorsal": "dorsal",
            "Player": "player_name",
            "Minutes": "minutes",
            "Points": "points",
            "FieldGoalsMade2": "field_goals_made_2",
            "FieldGoalsAttempted2": "field_goals_attempted_2",
            "FieldGoalsMade3": "field_goals_made_3",
            "FieldGoalsAttempted3": "field_goals_attempted_3",
            "FreeThrowsMade": "free_throws_made",
            "FreeThrowsAttempted": "free_throws_attempted",
            "OffensiveRebounds": "offensive_rebounds",
            "DefensiveRebounds": "defensive_rebounds",
            "TotalRebounds": "total_rebounds",
            "Assistances": "assists",
            "Steals": "steals",
            "Turnovers": "turnovers",
            "BlocksFavour": "blocks_favour",
            "BlocksAgainst": "blocks_against",
            "FoulsCommited": "fouls_committed",
            "FoulsReceived": "fouls_received",
            "Valuation": "pir",
            "Plusminus": "plus_minus",
        }
    )
    for col in ["player_id", "team_code", "player_name", "dorsal"]:
        if col in players.columns:
            players[col] = players[col].astype(str).str.strip()
    return players[
        [
            "season",
            "game_code",
            "player_id",
            "team_code",
            "player_name",
            "home",
            "is_starter",
            "is_playing",
            "dorsal",
            "minutes",
            "points",
            "field_goals_made_2",
            "field_goals_attempted_2",
            "field_goals_made_3",
            "field_goals_attempted_3",
            "free_throws_made",
            "free_throws_attempted",
            "offensive_rebounds",
            "defensive_rebounds",
            "total_rebounds",
            "assists",
            "steals",
            "turnovers",
            "blocks_favour",
            "blocks_against",
            "fouls_committed",
            "fouls_received",
            "pir",
            "plus_minus",
        ]
    ]


def payload_table(
    df: pd.DataFrame,
    season: int,
    key_columns: Iterable[str],
) -> pd.DataFrame:
    rows = df.copy().reset_index(drop=True)
    rows["season"] = season
    rows["payload_json"] = rows.apply(
        lambda row: json.dumps(row.dropna().to_dict(), ensure_ascii=True, default=str),
        axis=1,
    )
    for col in key_columns:
        if col not in rows.columns:
            rows[col] = None
    return rows


def normalize_team_quarter_scores(df: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = payload_table(df, season, ["Gamecode", "Team"])
    rows = rows.rename(columns={"Gamecode": "game_code", "Team": "team_code"})
    if "team_code" not in rows or rows["team_code"].isna().all():
        rows["team_code"] = rows.groupby(["season", "game_code"]).cumcount().astype(str)
    return rows[["season", "game_code", "team_code", "payload_json"]]


def normalize_shots(df: pd.DataFrame, season: int) -> pd.DataFrame:
    rows = payload_table(df, season, ["Gamecode", "TEAM", "ID_PLAYER", "PLAYER"])
    rows = rows.rename(
        columns={
            "Gamecode": "game_code",
            "TEAM": "team_code",
            "ID_PLAYER": "player_id",
            "PLAYER": "player_name",
        }
    )
    rows["row_id"] = rows.groupby(["season", "game_code"]).cumcount()
    return rows[
        ["season", "game_code", "row_id", "team_code", "player_id", "player_name", "payload_json"]
    ]


def cached_player_photo(player_id: str, image_url: str | None, force_refresh: bool = False) -> str | None:
    if not image_url or not isinstance(image_url, str):
        return None

    suffix = Path(image_url.split("?", 1)[0]).suffix.lower() or ".png"
    if suffix not in {".png", ".jpg", ".jpeg", ".webp"}:
        suffix = ".png"
    path = PLAYER_PHOTO_DIR / f"{player_id}{suffix}"
    if path.exists() and not force_refresh:
        return str(path.relative_to(PLAYER_PHOTO_DIR.parents[1]))

    try:
        response = requests.get(image_url, timeout=20)
        response.raise_for_status()
    except requests.RequestException as exc:
        console.print(f"[yellow]Photo unavailable for {player_id}: {exc}[/yellow]")
        return None

    path.write_bytes(response.content)
    return str(path.relative_to(PLAYER_PHOTO_DIR.parents[1]))


def normalize_player_profiles(df: pd.DataFrame, season: int, force_refresh: bool = False) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "player_id",
                "player_code",
                "player_name",
                "age",
                "team_code",
                "team_name",
                "image_url",
                "local_image_path",
            ]
        )

    rows = df.copy()
    rows["season"] = season
    rows["player_code"] = rows.get("player.code", "").astype(str).str.strip()
    rows["player_id"] = "P" + rows["player_code"]
    rows["player_name"] = rows.get("player.name")
    rows["age"] = rows.get("player.age")
    rows["team_code"] = rows.get("player.team.code")
    rows["team_name"] = rows.get("player.team.name")
    rows["image_url"] = rows.get("player.imageUrl")
    rows["local_image_path"] = rows.apply(
        lambda row: cached_player_photo(str(row["player_id"]), row.get("image_url"), force_refresh),
        axis=1,
    )
    return rows[
        [
            "season",
            "player_id",
            "player_code",
            "player_name",
            "age",
            "team_code",
            "team_name",
            "image_url",
            "local_image_path",
        ]
    ].drop_duplicates(["season", "player_id", "team_code"])


def write_table(conn, table: str, df: pd.DataFrame) -> None:
    if df.empty:
        console.print(f"[yellow]Skipping empty table:[/yellow] {table}")
        return
    df.to_sql(table, conn, if_exists="append", index=False)
    console.print(f"[green]Loaded[/green] {len(df):,} rows into {table}")


def delete_season_rows(conn, season: int, datasets: list[str]) -> None:
    tables = []
    if "games" in datasets:
        tables.extend(["player_boxscores", "team_quarter_scores", "shots", "games"])
        if "player_profiles" in datasets:
            tables.append("player_profiles")
    else:
        table_map = {
            "player_boxscores": "player_boxscores",
            "team_quarter_scores": "team_quarter_scores",
            "shots": "shots",
            "player_profiles": "player_profiles",
        }
        tables.extend(table_map[dataset] for dataset in datasets)

    for table in dict.fromkeys(tables):
        conn.execute(f"DELETE FROM {table} WHERE season = ?", (season,))
    conn.commit()


@app.command()
def main(
    start_season: int = typer.Option(DEFAULT_START_SEASON, help="First season start year."),
    end_season: int = typer.Option(DEFAULT_END_SEASON, help="Last season start year."),
    datasets: str = typer.Option(
        "games,player_boxscores,team_quarter_scores,shots",
        help=f"Comma-separated datasets. Available: {', '.join(sorted(DATASETS))}",
    ),
    force_refresh: bool = typer.Option(False, help="Ignore Parquet cache and call API again."),
    max_games_per_season: int | None = typer.Option(
        None,
        help="Development/testing shortcut. Fetch only the first N played games per season.",
    ),
) -> None:
    selected = [item.strip() for item in datasets.split(",") if item.strip()]
    unknown = sorted(set(selected) - DATASETS)
    if unknown:
        raise typer.BadParameter(f"Unknown datasets: {', '.join(unknown)}")
    game_dependent_datasets = {"player_boxscores", "team_quarter_scores", "shots"}
    if any(item in game_dependent_datasets for item in selected) and "games" not in selected:
        selected = ["games", *selected]

    seasons = season_range(start_season, end_season)
    conn = connect(DB_PATH)
    initialize_database(conn)
    run_id = begin_ingestion_run(conn, seasons, selected)

    metadata_api = EuroLeagueData(COMPETITION)
    boxscore_api = BoxScoreData(COMPETITION)
    shot_api = ShotData(COMPETITION)
    player_stats_api = PlayerStats(COMPETITION)

    try:
        for season in seasons:
            console.rule(f"E{season}")
            delete_season_rows(conn, season, selected)

            if "games" in selected:
                games_raw = load_or_fetch_raw(
                    "games",
                    season,
                    metadata_api.get_gamecodes_season,
                    force_refresh=force_refresh,
                )
                write_table(conn, "games", normalize_games(games_raw, season))
            else:
                games_raw = load_or_fetch_raw(
                    "games",
                    season,
                    metadata_api.get_gamecodes_season,
                    force_refresh=False,
                )

            game_codes = played_game_codes(games_raw, max_games_per_season)
            cache_suffix = f"_first_{max_games_per_season}" if max_games_per_season else ""

            if "player_boxscores" in selected:
                players_raw = load_or_fetch_raw(
                    f"player_boxscores{cache_suffix}",
                    season,
                    lambda selected_season: fetch_per_game_cached(
                        "player_boxscores",
                        selected_season,
                        game_codes,
                        boxscore_api.get_players_boxscore_stats,
                        force_refresh=force_refresh,
                    ),
                    force_refresh=force_refresh,
                )
                write_table(conn, "player_boxscores", normalize_player_boxscores(players_raw))

            if "team_quarter_scores" in selected:
                team_scores_raw = load_or_fetch_raw(
                    f"team_quarter_scores{cache_suffix}",
                    season,
                    lambda selected_season: fetch_per_game_cached(
                        "team_quarter_scores",
                        selected_season,
                        game_codes,
                        boxscore_api.get_teams_boxscore_quarter_scores,
                        force_refresh=force_refresh,
                    ),
                    force_refresh=force_refresh,
                )
                write_table(
                    conn,
                    "team_quarter_scores",
                    normalize_team_quarter_scores(team_scores_raw, season),
                )

            if "shots" in selected:
                shots_raw = load_or_fetch_raw(
                    f"shots{cache_suffix}",
                    season,
                    lambda selected_season: fetch_per_game_cached(
                        "shots",
                        selected_season,
                        game_codes,
                        shot_api.get_game_shot_data,
                        force_refresh=force_refresh,
                    ),
                    force_refresh=force_refresh,
                )
                write_table(conn, "shots", normalize_shots(shots_raw, season))

            if "player_profiles" in selected:
                profiles_raw = load_or_fetch_raw(
                    "player_profiles",
                    season,
                    lambda selected_season: player_stats_api.get_player_stats_single_season(
                        "traditional",
                        selected_season,
                    ),
                    force_refresh=force_refresh,
                )
                write_table(
                    conn,
                    "player_profiles",
                    normalize_player_profiles(profiles_raw, season, force_refresh),
                )

            conn.commit()

        finish_ingestion_run(conn, run_id, "success")
        console.print(f"[bold green]Done.[/bold green] SQLite database: {DB_PATH}")
    except Exception as exc:
        finish_ingestion_run(conn, run_id, "failed", str(exc))
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    app()
