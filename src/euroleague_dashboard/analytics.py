from __future__ import annotations

import sqlite3
import json
from pathlib import Path

import pandas as pd

from .config import DB_PATH
from .storage import connect


STAT_COLUMNS = ["points", "total_rebounds", "assists", "pir"]
PLAYER_STAT_COLUMNS = [
    "points",
    "total_rebounds",
    "assists",
    "steals",
    "blocks_favour",
    "turnovers",
    "fg_pct",
    "three_pct",
    "pir",
]


def read_table(table: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)


def player_profiles(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        table_exists = conn.execute(
            """
            SELECT 1
            FROM sqlite_master
            WHERE type = 'table' AND name = 'player_profiles'
            """
        ).fetchone()
        if not table_exists:
            return pd.DataFrame()
        profiles = pd.read_sql_query("SELECT * FROM player_profiles", conn)

    if profiles.empty:
        return profiles
    profiles = profiles.sort_values(["player_id", "season"], ascending=[True, False])
    return profiles.drop_duplicates("player_id")


def games_with_dates(conn: sqlite3.Connection) -> pd.DataFrame:
    games = pd.read_sql_query("SELECT * FROM games", conn)
    if games.empty:
        return games

    games["parsed_date"] = pd.to_datetime(games["game_date"], errors="coerce")
    return games


def team_lookup(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        games = pd.read_sql_query(
            """
            SELECT season, game_code, home_code AS team_code, home_team AS team_name FROM games
            UNION ALL
            SELECT season, game_code, away_code AS team_code, away_team AS team_name FROM games
            """,
            conn,
        )
    if games.empty:
        return pd.DataFrame(columns=["team_code", "team_name"])
    games["team_name"] = games["team_name"].str.title()
    games = games.sort_values(["team_code", "season", "game_code"], ascending=[True, False, False])
    return games.drop_duplicates("team_code")[["team_code", "team_name"]].sort_values("team_name")


def add_team_names(df: pd.DataFrame, db_path: Path = DB_PATH) -> pd.DataFrame:
    if df.empty:
        return df
    lookup = team_lookup(db_path)
    named = df.merge(lookup, on="team_code", how="left")
    named["team_name"] = named["team_name"].fillna(named["team_code"])
    if "opponent_code" in named.columns:
        opponent_lookup = lookup.rename(
            columns={"team_code": "opponent_code", "team_name": "opponent_name"}
        )
        named = named.merge(opponent_lookup, on="opponent_code", how="left")
        named["opponent_name"] = named["opponent_name"].fillna(named["opponent_code"])
    return named


def player_game_logs(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        players = pd.read_sql_query("SELECT * FROM player_boxscores", conn)
        games = games_with_dates(conn)

    if players.empty or games.empty:
        return pd.DataFrame()

    players = players[
        ~players["player_id"].isin(["Team", "Total"])
        & ~players["player_name"].isin(["Team", "Total"])
    ].copy()
    if "is_playing" in players.columns:
        players = players[players["is_playing"].fillna(0).astype(int) == 1].copy()

    merged = players.merge(
        games[
            [
                "season",
                "game_code",
                "phase",
                "group_name",
                "round",
                "parsed_date",
                "home_code",
                "away_code",
                "home_score",
                "away_score",
                "winner_code",
                "point_spread",
            ]
        ],
        on=["season", "game_code"],
        how="left",
    )
    merged["opponent_code"] = merged.apply(
        lambda row: row["away_code"] if int(row["home"] or 0) == 1 else row["home_code"],
        axis=1,
    )
    fga = (
        merged["field_goals_attempted_2"].fillna(0).astype(float)
        + merged["field_goals_attempted_3"].fillna(0).astype(float)
    )
    fgm = (
        merged["field_goals_made_2"].fillna(0).astype(float)
        + merged["field_goals_made_3"].fillna(0).astype(float)
    )
    merged["fg_pct"] = (fgm / fga).where(fga > 0, 0) * 100
    three_attempts = merged["field_goals_attempted_3"].fillna(0).astype(float)
    merged["three_pct"] = (
        merged["field_goals_made_3"].fillna(0).astype(float) / three_attempts
    ).where(three_attempts > 0, 0) * 100
    merged = add_team_names(merged, db_path)
    return merged.sort_values(["season", "game_code", "team_code", "player_name"])


def team_game_logs(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        boxscores = pd.read_sql_query("SELECT * FROM player_boxscores", conn)
        games = games_with_dates(conn)

    if boxscores.empty or games.empty:
        return pd.DataFrame()

    team_games = boxscores[boxscores["player_id"] == "Total"].copy()
    team_games = team_games.merge(
        games[
            [
                "season",
                "game_code",
                "phase",
                "group_name",
                "round",
                "parsed_date",
                "home_code",
                "away_code",
                "home_score",
                "away_score",
                "winner_code",
                "point_spread",
            ]
        ],
        on=["season", "game_code"],
        how="left",
    )
    team_games["opponent_code"] = team_games.apply(
        lambda row: row["away_code"] if int(row["home"] or 0) == 1 else row["home_code"],
        axis=1,
    )
    team_games["won"] = team_games["team_code"] == team_games["winner_code"]
    opponent_pir = team_games[["season", "game_code", "team_code", "pir"]].rename(
        columns={"team_code": "opponent_code", "pir": "pir_allowed"}
    )
    team_games = team_games.merge(
        opponent_pir,
        on=["season", "game_code", "opponent_code"],
        how="left",
    )
    team_games["points_allowed"] = team_games.apply(
        lambda row: row["away_score"] if int(row["home"] or 0) == 1 else row["home_score"],
        axis=1,
    )
    team_games["point_diff"] = team_games["points"] - team_games["points_allowed"]
    team_games = add_team_names(team_games, db_path)
    return team_games.sort_values(["season", "game_code", "team_code"])


def apply_optional_date_filter(
    df: pd.DataFrame,
    start_date: pd.Timestamp | None,
    end_date: pd.Timestamp | None,
) -> pd.DataFrame:
    if df.empty or (start_date is None and end_date is None):
        return df

    filtered = df.copy()
    if start_date is not None:
        filtered = filtered[filtered["parsed_date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        filtered = filtered[filtered["parsed_date"] <= pd.Timestamp(end_date)]
    return filtered


def season_and_total_averages(
    df: pd.DataFrame,
    entity_column: str,
    entity_value: str,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    selected = df[df[entity_column] == entity_value].copy()
    selected = apply_optional_date_filter(selected, start_date, end_date)
    if selected.empty:
        return pd.DataFrame(), pd.DataFrame()

    stat_columns = [
        col
        for col in PLAYER_STAT_COLUMNS
        if col in selected.columns and pd.api.types.is_numeric_dtype(selected[col])
    ]
    season_avg = (
        selected.groupby("season", as_index=False)[stat_columns]
        .mean()
        .sort_values("season")
        .round(2)
    )
    total_avg = selected[stat_columns].mean().to_frame().T.round(2)
    total_avg.insert(0, "scope", "Selected dates" if start_date or end_date else "All available games")
    total_avg.insert(1, "games", len(selected))
    return season_avg, total_avg


def team_leaderboard(team_logs: pd.DataFrame) -> pd.DataFrame:
    if team_logs.empty:
        return pd.DataFrame()
    board = (
        team_logs.groupby(["team_code", "team_name"], as_index=False)
        .agg(
            games=("game_code", "nunique"),
            wins=("won", "sum"),
            points_for=("points", "mean"),
            points_against=("points_allowed", "mean"),
            point_diff=("point_diff", "mean"),
            pir_for=("pir", "mean"),
            pir_against=("pir_allowed", "mean"),
        )
    )
    board["losses"] = board["games"] - board["wins"]
    board["win_pct"] = (board["wins"] / board["games"]).where(board["games"] > 0, 0) * 100
    board = board.sort_values(["wins", "point_diff", "pir_for"], ascending=False).reset_index(drop=True)
    board.insert(0, "rank", board.index + 1)
    return board.round(2)


def available_date_bounds(db_path: Path = DB_PATH) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
    with connect(db_path) as conn:
        games = games_with_dates(conn)
    if games.empty or games["parsed_date"].dropna().empty:
        return None, None
    return games["parsed_date"].min(), games["parsed_date"].max()


def database_summary(db_path: Path = DB_PATH) -> dict[str, int]:
    with connect(db_path) as conn:
        return {
            "games": conn.execute("SELECT COUNT(*) FROM games").fetchone()[0],
            "played_games": conn.execute("SELECT COUNT(*) FROM games WHERE played = 1").fetchone()[0],
            "player_rows": conn.execute("SELECT COUNT(*) FROM player_boxscores").fetchone()[0],
            "shots": conn.execute("SELECT COUNT(*) FROM shots").fetchone()[0],
            "teams": conn.execute(
                "SELECT COUNT(DISTINCT team_code) FROM player_boxscores WHERE team_code IS NOT NULL"
            ).fetchone()[0],
            "players": conn.execute(
                """
                SELECT COUNT(DISTINCT player_id)
                FROM player_boxscores
                WHERE player_id NOT IN ('Team', 'Total')
                """
            ).fetchone()[0],
        }


def shot_chart_data(db_path: Path = DB_PATH) -> pd.DataFrame:
    with connect(db_path) as conn:
        shots = pd.read_sql_query("SELECT season, game_code, payload_json FROM shots", conn)
        games = games_with_dates(conn)

    if shots.empty:
        return pd.DataFrame()

    records = []
    for row in shots.itertuples(index=False):
        payload = json.loads(row.payload_json)
        records.append(
            {
                "season": row.season,
                "game_code": row.game_code,
                "team_code": payload.get("TEAM"),
                "player_id": payload.get("ID_PLAYER"),
                "player_name": payload.get("PLAYER"),
                "action_id": payload.get("ID_ACTION"),
                "action": payload.get("ACTION"),
                "points": payload.get("POINTS"),
                "coord_x": payload.get("COORD_X"),
                "coord_y": payload.get("COORD_Y"),
                "zone": payload.get("ZONE"),
            }
        )

    shot_df = pd.DataFrame(records)
    shot_df["made"] = shot_df["points"].fillna(0).astype(float) > 0
    shot_df["action_id"] = shot_df["action_id"].astype(str)
    shot_df["is_3pt"] = shot_df["action_id"].str.startswith("3FG")
    shot_df["is_ft"] = shot_df["action_id"].str.startswith("FT")
    shot_df["coord_x"] = pd.to_numeric(shot_df["coord_x"], errors="coerce")
    shot_df["coord_y"] = pd.to_numeric(shot_df["coord_y"], errors="coerce")

    if not games.empty:
        shot_df = shot_df.merge(
            games[["season", "game_code", "phase", "group_name", "parsed_date", "home_code", "away_code"]],
            on=["season", "game_code"],
            how="left",
        )
        shot_df["home"] = shot_df["team_code"] == shot_df["home_code"]
    return shot_df
