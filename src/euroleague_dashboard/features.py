from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from .analytics import player_game_logs, team_game_logs
from .config import DB_PATH, FEATURES_DIR, ensure_data_dirs


app = typer.Typer(help="Build leakage-safe feature datasets for EuroLeague models.")


def minutes_to_float(value) -> float:
    if pd.isna(value):
        return 0.0
    text = str(value)
    if ":" in text:
        minutes, seconds = text.split(":", 1)
        return float(minutes or 0) + float(seconds or 0) / 60
    parsed = pd.to_numeric(text, errors="coerce")
    return float(parsed) if pd.notna(parsed) else 0.0


def add_player_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["player_id", "parsed_date", "season", "game_code"]).copy()
    base_cols = ["pir", "minutes_float", "points", "total_rebounds", "assists", "steals", "turnovers"]
    for window in [3, 5, 10]:
        for col in base_cols:
            ordered[f"player_last{window}_{col}"] = (
                ordered.groupby("player_id")[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )

    for col in base_cols:
        ordered[f"player_season_to_date_{col}"] = (
            ordered.groupby(["player_id", "season"])[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )

    ordered["player_games_before"] = ordered.groupby("player_id").cumcount()
    ordered["season_games_before"] = ordered.groupby(["player_id", "season"]).cumcount()
    ordered["prev_game_date"] = ordered.groupby("player_id")["parsed_date"].shift(1)
    ordered["rest_days"] = (ordered["parsed_date"] - ordered["prev_game_date"]).dt.days
    return ordered


def opponent_difficulty_features(team_logs: pd.DataFrame) -> pd.DataFrame:
    ordered = team_logs.sort_values(["team_code", "parsed_date", "season", "game_code"]).copy()
    cols = ["pir_allowed", "points_allowed", "total_rebounds", "assists", "point_diff"]
    for col in cols:
        ordered[f"opp_last5_{col}"] = (
            ordered.groupby("team_code")[col]
            .transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
        )
        ordered[f"opp_season_to_date_{col}"] = (
            ordered.groupby(["team_code", "season"])[col]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
    keep = ["season", "game_code", "team_code"] + [col for col in ordered.columns if col.startswith("opp_")]
    return ordered[keep].rename(columns={"team_code": "opponent_code"})


def build_player_game_features(db_path: Path = DB_PATH) -> pd.DataFrame:
    players = player_game_logs(db_path)
    teams = team_game_logs(db_path)
    if players.empty:
        return pd.DataFrame()

    features = players.copy()
    features["minutes_float"] = features["minutes"].map(minutes_to_float)
    features["is_home"] = features["home"].fillna(0).astype(int)
    features["is_starter"] = features["is_starter"].fillna(0).astype(int)
    features["target_pir"] = features["pir"].astype(float)

    features = add_player_rolling_features(features)
    opponent_features = opponent_difficulty_features(teams)
    features = features.merge(
        opponent_features,
        on=["season", "game_code", "opponent_code"],
        how="left",
    )

    model_cols = [
        "season",
        "game_code",
        "parsed_date",
        "phase",
        "round",
        "player_id",
        "player_name",
        "team_code",
        "team_name",
        "opponent_code",
        "opponent_name",
        "is_home",
        "is_starter",
        "minutes_float",
        "player_games_before",
        "season_games_before",
        "rest_days",
        "target_pir",
    ]
    engineered = [
        col
        for col in features.columns
        if col.startswith("player_last")
        or col.startswith("player_season_to_date")
        or col.startswith("opp_")
    ]
    out = features[model_cols + engineered].copy()
    out = out[out["player_games_before"] > 0].copy()
    numeric_cols = out.select_dtypes(include="number").columns
    out[numeric_cols] = out[numeric_cols].fillna(0)
    return out.sort_values(["parsed_date", "season", "game_code", "team_code", "player_name"])


@app.command()
def main(
    output: Path = typer.Option(
        FEATURES_DIR / "features_player_game.parquet",
        help="Output Parquet path.",
    ),
) -> None:
    ensure_data_dirs()
    output.parent.mkdir(parents=True, exist_ok=True)
    features = build_player_game_features(DB_PATH)
    features.to_parquet(output, index=False)
    typer.echo(f"Wrote {len(features):,} rows and {len(features.columns):,} columns to {output}")


if __name__ == "__main__":
    app()
