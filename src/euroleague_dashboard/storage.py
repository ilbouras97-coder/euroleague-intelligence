from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DB_PATH, RAW_DIR, ensure_data_dirs


def connect(db_path: Path = DB_PATH) -> sqlite3.Connection:
    ensure_data_dirs()
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def initialize_database(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS games (
            season INTEGER NOT NULL,
            game_code INTEGER NOT NULL,
            game_id TEXT,
            phase TEXT,
            round INTEGER,
            game_date TEXT,
            game_time TEXT,
            group_name TEXT,
            home_team TEXT,
            home_code TEXT,
            home_score INTEGER,
            away_team TEXT,
            away_code TEXT,
            away_score INTEGER,
            played INTEGER,
            winner_code TEXT,
            point_spread INTEGER,
            PRIMARY KEY (season, game_code)
        );

        CREATE TABLE IF NOT EXISTS player_boxscores (
            season INTEGER NOT NULL,
            game_code INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            team_code TEXT,
            player_name TEXT,
            home INTEGER,
            is_starter INTEGER,
            is_playing INTEGER,
            dorsal TEXT,
            minutes TEXT,
            points REAL,
            field_goals_made_2 REAL,
            field_goals_attempted_2 REAL,
            field_goals_made_3 REAL,
            field_goals_attempted_3 REAL,
            free_throws_made REAL,
            free_throws_attempted REAL,
            offensive_rebounds REAL,
            defensive_rebounds REAL,
            total_rebounds REAL,
            assists REAL,
            steals REAL,
            turnovers REAL,
            blocks_favour REAL,
            blocks_against REAL,
            fouls_committed REAL,
            fouls_received REAL,
            pir REAL,
            plus_minus REAL,
            PRIMARY KEY (season, game_code, player_id, team_code),
            FOREIGN KEY (season, game_code) REFERENCES games(season, game_code)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS team_quarter_scores (
            season INTEGER NOT NULL,
            game_code INTEGER NOT NULL,
            team_code TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (season, game_code, team_code),
            FOREIGN KEY (season, game_code) REFERENCES games(season, game_code)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS shots (
            season INTEGER NOT NULL,
            game_code INTEGER NOT NULL,
            row_id INTEGER NOT NULL,
            team_code TEXT,
            player_id TEXT,
            player_name TEXT,
            payload_json TEXT NOT NULL,
            PRIMARY KEY (season, game_code, row_id),
            FOREIGN KEY (season, game_code) REFERENCES games(season, game_code)
                ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS player_profiles (
            season INTEGER NOT NULL,
            player_id TEXT NOT NULL,
            player_code TEXT,
            player_name TEXT,
            age REAL,
            team_code TEXT,
            team_name TEXT,
            image_url TEXT,
            local_image_path TEXT,
            PRIMARY KEY (season, player_id, team_code)
        );

        CREATE TABLE IF NOT EXISTS ingestion_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            seasons TEXT NOT NULL,
            datasets TEXT NOT NULL,
            status TEXT NOT NULL,
            message TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_games_team_round
            ON games(season, round, home_code, away_code);
        CREATE INDEX IF NOT EXISTS idx_player_boxscores_player
            ON player_boxscores(player_id, season, game_code);
        CREATE INDEX IF NOT EXISTS idx_player_boxscores_team
            ON player_boxscores(team_code, season, game_code);
        CREATE INDEX IF NOT EXISTS idx_player_profiles_player
            ON player_profiles(player_id, season);
        """
    )
    conn.commit()


def raw_cache_path(dataset: str, season: int) -> Path:
    ensure_data_dirs()
    return RAW_DIR / f"{dataset}_E{season}.parquet"


def load_or_fetch_raw(
    dataset: str,
    season: int,
    fetcher: Any,
    force_refresh: bool = False,
) -> pd.DataFrame:
    path = raw_cache_path(dataset, season)
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)

    df = fetcher(season)
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    df.to_parquet(path, index=False)
    return df


def begin_ingestion_run(
    conn: sqlite3.Connection,
    seasons: list[int],
    datasets: list[str],
) -> int:
    cur = conn.execute(
        """
        INSERT INTO ingestion_runs (started_at, seasons, datasets, status)
        VALUES (?, ?, ?, ?)
        """,
        (
            datetime.now(UTC).isoformat(),
            json.dumps(seasons),
            json.dumps(datasets),
            "running",
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def finish_ingestion_run(
    conn: sqlite3.Connection,
    run_id: int,
    status: str,
    message: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE ingestion_runs
        SET finished_at = ?, status = ?, message = ?
        WHERE id = ?
        """,
        (datetime.now(UTC).isoformat(), status, message, run_id),
    )
    conn.commit()
