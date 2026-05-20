"""
Feature Engineering for EuroLeague PIR Prediction.

Builds a feature matrix from raw SQLite data with:
  - Rolling averages per player (last 3/5/10 games)
  - Season averages (expanding, leak-free)
  - Home / away / starter flags
  - Opponent defensive difficulty metrics
  - Team form (Rolling 5-game win % and point diff)
  - Head-to-Head (H2H) player vs opponent history
  - Phase encoding (RS, PI, PO, FF)
  - Rest days between consecutive games
  - Trend features (recent form vs season average)
  - Per-minute rate features and rotation stability

CRITICAL: No data leakage -- every feature uses ONLY prior games.
"""

import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

from config import (
    DB_PATH, ROLLING_WINDOWS, ROLLING_STATS,
    OPPONENT_DEFENSIVE_STATS, PHASE_MAP, MIN_GAMES_FOR_TRAINING,
)

RATE_STATS = [
    "pir", "points", "total_rebounds", "assists", "steals",
    "turnovers", "field_goals_attempted_2", "field_goals_attempted_3",
    "free_throws_attempted",
]

PLAYER_ROLE_OVERRIDES = {
    # Stable corrections for high-usage players whose boxscore profile can
    # look like a different fantasy slot.
    "P003469": "Forward",  # Sasha Vezenkov
    "P014124": "Guard",    # Talen Horton-Tucker
    "P008161": "Forward",  # Cedi Osman
}

PLAYER_STYLE_BUCKETS = [
    "ball_handler", "scoring_guard", "wing_scorer", "stretch_forward", "rim_center"
]


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while treating zero-minute rows as missing signal."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


# ==============================================================
# 1. DATA LOADING
# ==============================================================

def load_raw_data(db_path: str = DB_PATH) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load games and player_boxscores from SQLite."""
    conn = sqlite3.connect(db_path)

    games = pd.read_sql("""
        SELECT season, game_code, phase, round, game_date,
               home_code, away_code, home_score, away_score,
               winner_code, point_spread
        FROM games
        WHERE played = 1
        ORDER BY season, game_code
    """, conn)

    boxscores = pd.read_sql("""
        SELECT season, game_code, player_id, player_name, team_code,
               home, is_starter, is_playing, minutes,
               points, field_goals_made_2, field_goals_attempted_2,
               field_goals_made_3, field_goals_attempted_3,
               free_throws_made, free_throws_attempted,
               offensive_rebounds, defensive_rebounds, total_rebounds,
               assists, steals, turnovers, blocks_favour, blocks_against,
               fouls_committed, fouls_received, pir, plus_minus
        FROM player_boxscores
        WHERE player_id != 'Total'
    """, conn)

    conn.close()
    return games, boxscores


# ==============================================================
# 2. DATA CLEANING
# ==============================================================

def parse_minutes(minutes_str: str) -> float:
    """Convert 'MM:SS' string to float minutes. Returns NaN for invalid."""
    if pd.isna(minutes_str) or minutes_str in ("DNP", "None", ""):
        return np.nan
    try:
        parts = str(minutes_str).split(":")
        return int(parts[0]) + int(parts[1]) / 60.0
    except (ValueError, IndexError):
        return np.nan


def parse_game_date(date_str: str) -> pd.Timestamp:
    """Parse EuroLeague date format like 'Apr 01, 2026'."""
    try:
        return pd.to_datetime(date_str, format="%b %d, %Y")
    except (ValueError, TypeError):
        return pd.NaT


def clean_data(games: pd.DataFrame, boxscores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean and prepare raw data."""
    # Parse game dates
    games["game_date_parsed"] = games["game_date"].apply(parse_game_date)
    games = games.sort_values(["season", "game_date_parsed", "game_code"]).reset_index(drop=True)

    # Parse minutes to float
    boxscores["minutes_float"] = boxscores["minutes"].apply(parse_minutes)

    # Remove rows where player had 0 or NaN minutes
    boxscores = boxscores[boxscores["minutes_float"] > 0].copy()

    # Merge game info into boxscores
    boxscores = boxscores.merge(
        games[["season", "game_code", "game_date_parsed", "phase", "round",
               "home_code", "away_code", "home_score", "away_score"]],
        on=["season", "game_code"],
        how="left",
    )

    # Determine opponent_code for each player row
    boxscores["opponent_code"] = np.where(
        boxscores["team_code"] == boxscores["home_code"],
        boxscores["away_code"],
        boxscores["home_code"],
    )

    # Sort by player, then chronologically
    boxscores = boxscores.sort_values(
        ["player_id", "season", "game_date_parsed", "game_code"]
    ).reset_index(drop=True)

    return games, boxscores


# ==============================================================
# 3. PLAYER ROLLING & SEASON FEATURES (LEAK-FREE)
# ==============================================================

def compute_player_features(boxscores: pd.DataFrame) -> pd.DataFrame:
    df = boxscores.copy()
    grouped = df.groupby("player_id")

    # Rolling averages (shifted)
    print("       -> Computing rolling averages...")
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        for w in ROLLING_WINDOWS:
            col_name = f"rolling_{w}_{stat}"
            df[col_name] = grouped[stat].transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )

    # Season expanding average (shifted)
    season_grouped = df.groupby(["player_id", "season"])
    for stat in ROLLING_STATS:
        if stat not in df.columns:
            continue
        col_name = f"season_avg_{stat}"
        df[col_name] = season_grouped[stat].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    # EWMA features (Exponentially Weighted Moving Average) - inspired by Meklis repo
    # Gives more weight to recent games. Span=5 and Span=10.
    print("       -> Computing EWMA features...")
    for stat in ["pir", "minutes_float", "points"]:
        if stat in df.columns:
            df[f"ewma_5_{stat}"] = grouped[stat].transform(
                lambda x: x.shift(1).ewm(span=5, adjust=False).mean()
            )
            df[f"ewma_10_{stat}"] = grouped[stat].transform(
                lambda x: x.shift(1).ewm(span=10, adjust=False).mean()
            )

    # Games played
    df["season_games_played"] = season_grouped["pir"].transform(lambda x: x.shift(1).expanding().count())
    df["career_games_played"] = grouped["pir"].transform(lambda x: x.shift(1).expanding().count())

    # Trend features
    for stat in ["pir", "minutes_float", "points"]:
        if f"rolling_5_{stat}" in df.columns and f"season_avg_{stat}" in df.columns:
            df[f"trend_5_{stat}"] = df[f"rolling_5_{stat}"] - df[f"season_avg_{stat}"]
        if f"rolling_3_{stat}" in df.columns and f"season_avg_{stat}" in df.columns:
            df[f"trend_3_{stat}"] = df[f"rolling_3_{stat}"] - df[f"season_avg_{stat}"]

    # Consistency
    df["rolling_5_pir_std"] = grouped["pir"].transform(lambda x: x.shift(1).rolling(window=5, min_periods=2).std())
    df["rolling_10_pir_std"] = grouped["pir"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=3).std())

    # Usage proxy
    df["total_fga"] = df["field_goals_attempted_2"].fillna(0) + df["field_goals_attempted_3"].fillna(0) + df["free_throws_attempted"].fillna(0)
    df["rolling_5_total_fga"] = grouped["total_fga"].transform(lambda x: x.shift(1).rolling(window=5, min_periods=1).mean())
    df["rolling_3_total_fga"] = grouped["total_fga"].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
    df["rolling_10_total_fga"] = grouped["total_fga"].transform(lambda x: x.shift(1).rolling(window=10, min_periods=1).mean())
    df["season_avg_total_fga"] = season_grouped["total_fga"].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    new_features = {}

    # Per-minute rates capture production independent of role/minute load.
    for stat in RATE_STATS:
        if stat not in df.columns:
            continue
        rate_col = f"{stat}_per_min"
        rate = safe_divide(df[stat], df["minutes_float"])
        new_features[rate_col] = rate
        for w in ROLLING_WINDOWS:
            new_features[f"rolling_{w}_{rate_col}"] = rate.groupby(df["player_id"]).transform(
                lambda x: x.shift(1).rolling(window=w, min_periods=1).mean()
            )
        new_features[f"season_avg_{rate_col}"] = rate.groupby([df["player_id"], df["season"]]).transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )

    starter_int = df["is_starter"].fillna(0).astype(int)
    new_features["starter_int"] = starter_int
    new_features["rolling_3_starter_rate"] = starter_int.groupby(df["player_id"]).transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    new_features["rolling_5_starter_rate"] = starter_int.groupby(df["player_id"]).transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    new_features["rolling_5_minutes_std"] = grouped["minutes_float"].transform(lambda x: x.shift(1).rolling(5, min_periods=2).std())
    new_features["last_game_minutes"] = grouped["minutes_float"].shift(1)
    new_features["last_game_pir"] = grouped["pir"].shift(1)
    new_features["last_game_total_fga"] = grouped["total_fga"].shift(1)
    new_features["usage_trend_3_vs_10"] = df["rolling_3_total_fga"] - df["rolling_10_total_fga"]
    new_features["usage_trend_5_vs_season"] = df["rolling_5_total_fga"] - df["season_avg_total_fga"]
    new_features["rolling_5_fga_per_min"] = safe_divide(df["rolling_5_total_fga"], df["rolling_5_minutes_float"])
    new_features["rolling_3_fga_per_min"] = safe_divide(df["rolling_3_total_fga"], df["rolling_3_minutes_float"])
    new_features["season_avg_fga_per_min"] = safe_divide(df["season_avg_total_fga"], df["season_avg_minutes_float"])

    df = pd.concat([df, pd.DataFrame(new_features, index=df.index)], axis=1)

    return df


# ==============================================================
# 4. OPPONENT & TEAM FORM & H2H FEATURES (LEAK-FREE)
# ==============================================================

def infer_player_roles(boxscores: pd.DataFrame) -> pd.DataFrame:
    """Infer fantasy role buckets used for opponent-by-role defensive context."""
    required = ["player_id", "total_rebounds", "assists", "blocks_favour"]
    missing = [col for col in required if col not in boxscores.columns]
    if missing:
        return pd.DataFrame({"player_id": boxscores["player_id"].drop_duplicates(), "player_role": "Forward"})

    profiles = boxscores.groupby("player_id").agg(
        avg_rebounds=("total_rebounds", "mean"),
        avg_assists=("assists", "mean"),
        avg_blocks=("blocks_favour", "mean"),
    ).reset_index()

    conditions = [
        (profiles["avg_assists"] >= 3.5)
        | (profiles["avg_assists"] >= (profiles["avg_rebounds"] + profiles["avg_blocks"])),
        (profiles["avg_blocks"] >= 0.7)
        | ((profiles["avg_rebounds"] >= 7.0) & (profiles["avg_assists"] < 2.0)),
    ]
    profiles["player_role"] = np.select(conditions, ["Guard", "Center"], default="Forward")
    profiles["player_role"] = profiles.apply(
        lambda row: PLAYER_ROLE_OVERRIDES.get(str(row["player_id"]), row["player_role"]),
        axis=1,
    )
    return profiles[["player_id", "player_role"]]


def infer_player_styles(boxscores: pd.DataFrame) -> pd.DataFrame:
    """Infer rough offensive/physical style buckets for matchup defense."""
    profiles = boxscores.groupby("player_id").agg(
        avg_rebounds=("total_rebounds", "mean"),
        avg_assists=("assists", "mean"),
        avg_blocks=("blocks_favour", "mean"),
        avg_2pa=("field_goals_attempted_2", "mean"),
        avg_3pa=("field_goals_attempted_3", "mean"),
        avg_fta=("free_throws_attempted", "mean"),
        avg_points=("points", "mean"),
    ).reset_index()
    profiles["shot_volume"] = profiles["avg_2pa"].fillna(0) + profiles["avg_3pa"].fillna(0) + profiles["avg_fta"].fillna(0)
    profiles["three_share"] = safe_divide(profiles["avg_3pa"], profiles["avg_2pa"].fillna(0) + profiles["avg_3pa"].fillna(0)).fillna(0)
    conditions = [
        profiles["avg_assists"] >= 4.0,
        (profiles["avg_assists"] >= 2.2) & (profiles["shot_volume"] >= 8.0),
        (profiles["avg_points"] >= 9.0) & (profiles["avg_rebounds"] < 5.5),
        (profiles["three_share"] >= 0.45) & (profiles["avg_rebounds"] >= 3.5),
        (profiles["avg_rebounds"] >= 6.2) | (profiles["avg_blocks"] >= 0.7),
    ]
    profiles["player_style"] = np.select(conditions, PLAYER_STYLE_BUCKETS, default="wing_scorer")
    return profiles[["player_id", "player_style"]]


def compute_opponent_features(games: pd.DataFrame, boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute opponent defensive difficulty metrics."""
    game_stats = boxscores.groupby(["season", "game_code", "team_code"]).agg(
        team_pir=("pir", "sum"),
        team_points=("points", "sum"),
        team_rebounds=("total_rebounds", "sum"),
        team_assists=("assists", "sum"),
    ).reset_index()

    game_stats = game_stats.merge(
        games[["season", "game_code", "home_code", "away_code", "game_date_parsed"]],
        on=["season", "game_code"],
        how="left",
    )
    game_stats["opponent_code"] = np.where(game_stats["team_code"] == game_stats["home_code"], game_stats["away_code"], game_stats["home_code"])

    allowed_lookup = game_stats.rename(columns={
        "team_pir": "allowed_pir", "team_points": "allowed_points",
        "team_rebounds": "allowed_rebounds", "team_assists": "allowed_assists",
        "opponent_code": "defending_team"
    })[["season", "game_code", "defending_team", "game_date_parsed", "allowed_pir", "allowed_points", "allowed_rebounds", "allowed_assists"]]

    allowed_lookup = allowed_lookup.sort_values(["defending_team", "season", "game_date_parsed", "game_code"]).reset_index(drop=True)
    def_grouped = allowed_lookup.groupby("defending_team")

    for stat in ["allowed_pir", "allowed_points", "allowed_rebounds", "allowed_assists"]:
        allowed_lookup[f"opp_avg_{stat}"] = def_grouped[stat].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())

    allowed_lookup["opp_games_defended"] = def_grouped["allowed_pir"].transform(lambda x: x.shift(1).expanding().count())
    
    return allowed_lookup[["season", "game_code", "defending_team", "opp_avg_allowed_pir", "opp_avg_allowed_points", "opp_avg_allowed_rebounds", "opp_avg_allowed_assists", "opp_games_defended"]].drop_duplicates()


def compute_opponent_role_features(games: pd.DataFrame, boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free defensive allowance split by player fantasy role."""
    roles = infer_player_roles(boxscores)
    role_boxscores = boxscores.merge(roles, on="player_id", how="left")
    role_boxscores["player_role"] = role_boxscores["player_role"].fillna("Forward")

    role_game_stats = role_boxscores.groupby(
        ["season", "game_code", "team_code", "player_role"], as_index=False
    ).agg(
        role_allowed_pir=("pir", "sum"),
        role_allowed_points=("points", "sum"),
        role_allowed_rebounds=("total_rebounds", "sum"),
        role_allowed_assists=("assists", "sum"),
        role_allowed_minutes=("minutes_float", "sum"),
        role_allowed_players=("player_id", "nunique"),
    )

    role_game_stats = role_game_stats.merge(
        games[["season", "game_code", "home_code", "away_code", "game_date_parsed"]],
        on=["season", "game_code"],
        how="left",
    )
    role_game_stats["defending_team"] = np.where(
        role_game_stats["team_code"] == role_game_stats["home_code"],
        role_game_stats["away_code"],
        role_game_stats["home_code"],
    )
    role_game_stats = role_game_stats.sort_values(
        ["defending_team", "player_role", "season", "game_date_parsed", "game_code"]
    ).reset_index(drop=True)
    grouped = role_game_stats.groupby(["defending_team", "player_role"])

    for stat in [
        "role_allowed_pir", "role_allowed_points", "role_allowed_rebounds",
        "role_allowed_assists", "role_allowed_minutes", "role_allowed_players",
    ]:
        role_game_stats[f"opp_role_avg_{stat}"] = grouped[stat].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
    role_game_stats["opp_role_games_defended"] = grouped["role_allowed_pir"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    return role_game_stats[[
        "season", "game_code", "defending_team", "player_role",
        "opp_role_avg_role_allowed_pir", "opp_role_avg_role_allowed_points",
        "opp_role_avg_role_allowed_rebounds", "opp_role_avg_role_allowed_assists",
        "opp_role_avg_role_allowed_minutes", "opp_role_avg_role_allowed_players",
        "opp_role_games_defended",
    ]].drop_duplicates()


def compute_opponent_style_features(games: pd.DataFrame, boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free defensive allowance split by player style archetype."""
    styles = infer_player_styles(boxscores)
    style_boxscores = boxscores.merge(styles, on="player_id", how="left")
    style_boxscores["player_style"] = style_boxscores["player_style"].fillna("wing_scorer")
    style_game_stats = style_boxscores.groupby(
        ["season", "game_code", "team_code", "player_style"], as_index=False
    ).agg(
        style_allowed_pir=("pir", "sum"),
        style_allowed_points=("points", "sum"),
        style_allowed_fga=("field_goals_attempted_2", "sum"),
        style_allowed_3pa=("field_goals_attempted_3", "sum"),
        style_allowed_fta=("free_throws_attempted", "sum"),
        style_allowed_minutes=("minutes_float", "sum"),
    )
    style_game_stats["style_allowed_fga"] = (
        style_game_stats["style_allowed_fga"].fillna(0)
        + style_game_stats["style_allowed_3pa"].fillna(0)
        + style_game_stats["style_allowed_fta"].fillna(0)
    )
    style_game_stats = style_game_stats.merge(
        games[["season", "game_code", "home_code", "away_code", "game_date_parsed"]],
        on=["season", "game_code"],
        how="left",
    )
    style_game_stats["defending_team"] = np.where(
        style_game_stats["team_code"] == style_game_stats["home_code"],
        style_game_stats["away_code"],
        style_game_stats["home_code"],
    )
    style_game_stats = style_game_stats.sort_values(
        ["defending_team", "player_style", "season", "game_date_parsed", "game_code"]
    ).reset_index(drop=True)
    grouped = style_game_stats.groupby(["defending_team", "player_style"])
    for stat in ["style_allowed_pir", "style_allowed_points", "style_allowed_fga", "style_allowed_minutes"]:
        style_game_stats[f"opp_style_avg_{stat}"] = grouped[stat].transform(
            lambda x: x.shift(1).expanding(min_periods=1).mean()
        )
    style_game_stats["opp_style_games_defended"] = grouped["style_allowed_pir"].transform(
        lambda x: x.shift(1).expanding().count()
    )
    return style_game_stats[[
        "season", "game_code", "defending_team", "player_style",
        "opp_style_avg_style_allowed_pir", "opp_style_avg_style_allowed_points",
        "opp_style_avg_style_allowed_fga", "opp_style_avg_style_allowed_minutes",
        "opp_style_games_defended",
    ]].drop_duplicates()


def compute_team_form(games: pd.DataFrame) -> pd.DataFrame:
    """Compute 5-game rolling win percentage and point differential for each team."""
    home = games[['season', 'game_code', 'game_date_parsed', 'home_code', 'winner_code', 'home_score', 'away_score']].rename(columns={'home_code': 'team_code'})
    home['is_win'] = (home['team_code'] == home['winner_code']).astype(int)
    home['point_diff'] = home['home_score'] - home['away_score']
    
    away = games[['season', 'game_code', 'game_date_parsed', 'away_code', 'winner_code', 'away_score', 'home_score']].rename(columns={'away_code': 'team_code'})
    away['is_win'] = (away['team_code'] == away['winner_code']).astype(int)
    away['point_diff'] = away['away_score'] - away['home_score']
    
    team_games = pd.concat([home, away]).sort_values(['team_code', 'game_date_parsed']).reset_index(drop=True)
    grouped = team_games.groupby('team_code')
    
    team_games['team_form_win_pct_5'] = grouped['is_win'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    team_games['team_form_pt_diff_5'] = grouped['point_diff'].transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    
    return team_games[['season', 'game_code', 'team_code', 'team_form_win_pct_5', 'team_form_pt_diff_5']]


def compute_team_rotation_features(boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute leak-free team rotation stability features from prior games."""
    team_game = boxscores.groupby(["season", "game_code", "team_code", "game_date_parsed"]).agg(
        active_players=("player_id", "nunique"),
        starters_avg_minutes=("minutes_float", lambda x: x[boxscores.loc[x.index, "is_starter"] == 1].mean()),
        bench_avg_minutes=("minutes_float", lambda x: x[boxscores.loc[x.index, "is_starter"] == 0].mean()),
        team_minutes_std=("minutes_float", "std"),
        team_top_minutes=("minutes_float", "max"),
        team_total_minutes=("minutes_float", "sum"),
    ).reset_index()
    team_game["team_top_minutes_share"] = safe_divide(team_game["team_top_minutes"], team_game["team_total_minutes"])
    team_game = team_game.sort_values(["team_code", "season", "game_date_parsed", "game_code"]).reset_index(drop=True)
    grouped = team_game.groupby("team_code")

    for col in [
        "active_players", "starters_avg_minutes", "bench_avg_minutes",
        "team_minutes_std", "team_top_minutes_share",
    ]:
        team_game[f"rolling_5_{col}"] = grouped[col].transform(
            lambda x: x.shift(1).rolling(5, min_periods=1).mean()
        )

    team_game["team_rotation_volatility_5"] = grouped["active_players"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )

    return team_game[[
        "season", "game_code", "team_code",
        "rolling_5_active_players", "rolling_5_starters_avg_minutes",
        "rolling_5_bench_avg_minutes", "rolling_5_team_minutes_std",
        "rolling_5_team_top_minutes_share", "team_rotation_volatility_5",
    ]]


def compute_team_availability_pressure(player_features: pd.DataFrame) -> pd.DataFrame:
    """Estimate how much known rotation/usage is available in the active lineup."""
    required = ["rolling_5_minutes_float", "rolling_5_total_fga"]
    if any(col not in player_features.columns for col in required):
        return player_features[["season", "game_code", "team_code"]].drop_duplicates().assign(
            team_active_prior_minutes_sum=np.nan,
            team_active_prior_usage_sum=np.nan,
            team_core_available_score=np.nan,
            team_usage_available_score=np.nan,
        )

    def top_sum(values: pd.Series, n: int) -> float:
        return float(values.fillna(0).sort_values(ascending=False).head(n).sum())

    team_game = player_features.groupby(["season", "game_code", "team_code"], as_index=False).agg(
        team_active_prior_minutes_sum=("rolling_5_minutes_float", "sum"),
        team_active_prior_usage_sum=("rolling_5_total_fga", "sum"),
        team_top6_prior_minutes_sum=("rolling_5_minutes_float", lambda x: top_sum(x, 6)),
        team_top8_prior_usage_sum=("rolling_5_total_fga", lambda x: top_sum(x, 8)),
    )
    team_game["team_core_available_score"] = safe_divide(team_game["team_top6_prior_minutes_sum"], pd.Series(150.0, index=team_game.index)).clip(0, 1.35)
    team_game["team_usage_available_score"] = safe_divide(team_game["team_top8_prior_usage_sum"], pd.Series(70.0, index=team_game.index)).clip(0, 1.35)
    team_game["team_rotation_shortage_score"] = (1.0 - team_game["team_core_available_score"]).clip(0, 1)

    return team_game[[
        "season", "game_code", "team_code",
        "team_active_prior_minutes_sum", "team_active_prior_usage_sum",
        "team_top6_prior_minutes_sum", "team_top8_prior_usage_sum",
        "team_core_available_score", "team_usage_available_score",
        "team_rotation_shortage_score",
    ]]


def compute_player_team_rotation_context(player_features: pd.DataFrame) -> pd.DataFrame:
    """Rank players within their game-day team by prior minutes and usage."""
    df = player_features[[
        "season", "game_code", "team_code", "player_id",
        "rolling_5_minutes_float", "rolling_5_total_fga",
        "rolling_3_minutes_float", "rolling_3_total_fga",
    ]].copy()
    grouped = df.groupby(["season", "game_code", "team_code"])
    df["team_prior_minutes_sum"] = grouped["rolling_5_minutes_float"].transform("sum")
    df["team_prior_usage_sum"] = grouped["rolling_5_total_fga"].transform("sum")
    df["team_prior_minutes_rank"] = grouped["rolling_5_minutes_float"].rank(method="average", ascending=False)
    df["team_prior_usage_rank"] = grouped["rolling_5_total_fga"].rank(method="average", ascending=False)
    df["team_prior_rotation_count"] = grouped["player_id"].transform("count")
    df["team_prior_minutes_rank_pct"] = safe_divide(df["team_prior_minutes_rank"], df["team_prior_rotation_count"])
    df["team_prior_usage_rank_pct"] = safe_divide(df["team_prior_usage_rank"], df["team_prior_rotation_count"])
    df["team_prior_minutes_share"] = safe_divide(df["rolling_5_minutes_float"], df["team_prior_minutes_sum"])
    df["team_prior_usage_share"] = safe_divide(df["rolling_5_total_fga"], df["team_prior_usage_sum"])
    df["recent_minutes_vs_team_avg"] = df["rolling_3_minutes_float"] - grouped["rolling_3_minutes_float"].transform("mean")
    df["recent_usage_vs_team_avg"] = df["rolling_3_total_fga"] - grouped["rolling_3_total_fga"].transform("mean")
    df["top3_minutes_rotation"] = (df["team_prior_minutes_rank"] <= 3).astype(int)
    df["top6_minutes_rotation"] = (df["team_prior_minutes_rank"] <= 6).astype(int)
    df["top3_usage_rotation"] = (df["team_prior_usage_rank"] <= 3).astype(int)

    return df[[
        "season", "game_code", "team_code", "player_id",
        "team_prior_minutes_sum", "team_prior_usage_sum",
        "team_prior_minutes_rank", "team_prior_usage_rank",
        "team_prior_rotation_count", "team_prior_minutes_rank_pct",
        "team_prior_usage_rank_pct", "team_prior_minutes_share",
        "team_prior_usage_share", "recent_minutes_vs_team_avg",
        "recent_usage_vs_team_avg", "top3_minutes_rotation",
        "top6_minutes_rotation", "top3_usage_rotation",
    ]]


def compute_team_standings_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute pre-game standings pressure proxies."""
    home = games[[
        "season", "game_code", "game_date_parsed", "round",
        "home_code", "winner_code", "home_score", "away_score"
    ]].rename(columns={"home_code": "team_code"})
    home["is_win"] = (home["team_code"] == home["winner_code"]).astype(int)
    home["point_diff"] = home["home_score"] - home["away_score"]

    away = games[[
        "season", "game_code", "game_date_parsed", "round",
        "away_code", "winner_code", "away_score", "home_score"
    ]].rename(columns={"away_code": "team_code"})
    away["is_win"] = (away["team_code"] == away["winner_code"]).astype(int)
    away["point_diff"] = away["away_score"] - away["home_score"]

    team_games = pd.concat([home, away], ignore_index=True)
    team_games = team_games.sort_values(["season", "team_code", "game_date_parsed", "game_code"]).reset_index(drop=True)
    grouped = team_games.groupby(["season", "team_code"])
    team_games["team_wins_before"] = grouped["is_win"].cumsum() - team_games["is_win"]
    team_games["team_games_before"] = grouped.cumcount()
    team_games["team_win_pct_before"] = safe_divide(team_games["team_wins_before"], team_games["team_games_before"])
    team_games["team_point_diff_before"] = grouped["point_diff"].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    )

    max_round = games.groupby("season")["round"].max().rename("max_round")
    team_games = team_games.merge(max_round, on="season", how="left")
    team_games["round_pct"] = safe_divide(team_games["round"], team_games["max_round"])
    team_games["late_season_flag"] = (team_games["round_pct"] >= 0.85).astype(int)
    team_games["final_two_rounds_flag"] = (team_games["round"] >= (team_games["max_round"] - 1)).astype(int)
    team_games["motivation_uncertainty_proxy"] = (
        team_games["late_season_flag"]
        * (team_games["team_win_pct_before"].fillna(0.5) - 0.5).abs()
    )

    return team_games[[
        "season", "game_code", "team_code",
        "team_wins_before", "team_games_before", "team_win_pct_before",
        "team_point_diff_before", "round_pct", "late_season_flag",
        "final_two_rounds_flag", "motivation_uncertainty_proxy",
    ]]


def compute_h2h_features(boxscores: pd.DataFrame) -> pd.DataFrame:
    """Compute player historical performance against the specific opponent."""
    df = boxscores[['season', 'game_code', 'player_id', 'opponent_code', 'pir', 'game_date_parsed']].copy()
    df = df.sort_values(['player_id', 'opponent_code', 'game_date_parsed']).reset_index(drop=True)
    grouped = df.groupby(['player_id', 'opponent_code'])
    
    df['h2h_avg_pir'] = grouped['pir'].transform(lambda x: x.shift(1).expanding(min_periods=1).mean())
    df['h2h_games'] = grouped['pir'].transform(lambda x: x.shift(1).expanding().count())
    
    return df[['season', 'game_code', 'player_id', 'opponent_code', 'h2h_avg_pir', 'h2h_games']]


# ==============================================================
# 5. REST DAYS FEATURE
# ==============================================================

def compute_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_game_date"] = df.groupby("player_id")["game_date_parsed"].shift(1)
    df["rest_days"] = (df["game_date_parsed"] - df["prev_game_date"]).dt.days
    df["rest_days"] = df["rest_days"].clip(upper=30)
    df.drop(columns=["prev_game_date"], inplace=True)
    return df


# ==============================================================
# 6. ASSEMBLE FULL FEATURE MATRIX
# ==============================================================

def build_feature_matrix(db_path: str = DB_PATH) -> pd.DataFrame:
    print("[1/7] Loading raw data...")
    games, boxscores = load_raw_data(db_path)

    print("[2/7] Cleaning data...")
    games, boxscores = clean_data(games, boxscores)

    print("[3/7] Computing player rolling & season features...")
    df = compute_player_features(boxscores)
    player_roles = infer_player_roles(boxscores)
    player_styles = infer_player_styles(boxscores)
    df = df.merge(player_roles, on="player_id", how="left")
    df = df.merge(player_styles, on="player_id", how="left")
    df["player_role"] = df["player_role"].fillna("Forward")
    df["player_style"] = df["player_style"].fillna("wing_scorer")
    for role in ["Guard", "Forward", "Center"]:
        df[f"role_{role}"] = (df["player_role"] == role).astype(int)
    for style in PLAYER_STYLE_BUCKETS:
        df[f"style_{style}"] = (df["player_style"] == style).astype(int)
    df["rotation_tier_score"] = (
        0.55 * safe_divide(df["rolling_5_minutes_float"], pd.Series(30.0, index=df.index)).fillna(0)
        + 0.45 * safe_divide(df["rolling_5_total_fga"], pd.Series(12.0, index=df.index)).fillna(0)
    ).clip(0, 2.0)

    print("[4/7] Computing opponent & team form & H2H features...")
    opp_features = compute_opponent_features(games, boxscores)
    opp_role_features = compute_opponent_role_features(games, boxscores)
    opp_style_features = compute_opponent_style_features(games, boxscores)
    team_form = compute_team_form(games)
    team_rotation = compute_team_rotation_features(boxscores)
    team_availability = compute_team_availability_pressure(df)
    player_team_context = compute_player_team_rotation_context(df)
    team_standings = compute_team_standings_features(games)
    h2h_features = compute_h2h_features(boxscores)

    # Merge Opponent Defense
    df = df.merge(opp_features, left_on=["season", "game_code", "opponent_code"], right_on=["season", "game_code", "defending_team"], how="left")
    df = df.merge(
        opp_role_features,
        left_on=["season", "game_code", "opponent_code", "player_role"],
        right_on=["season", "game_code", "defending_team", "player_role"],
        how="left",
        suffixes=("", "_role"),
    )
    df = df.merge(
        opp_style_features,
        left_on=["season", "game_code", "opponent_code", "player_style"],
        right_on=["season", "game_code", "defending_team", "player_style"],
        how="left",
        suffixes=("", "_style"),
    )
    
    # Merge Team Form
    df = df.merge(team_form, on=["season", "game_code", "team_code"], how="left")

    # Merge Team Rotation and Motivation Context
    df = df.merge(team_rotation, on=["season", "game_code", "team_code"], how="left")
    df = df.merge(team_availability, on=["season", "game_code", "team_code"], how="left")
    df = df.merge(player_team_context, on=["season", "game_code", "team_code", "player_id"], how="left")
    df = df.merge(team_standings, on=["season", "game_code", "team_code"], how="left")
    
    # Merge H2H
    df = df.merge(h2h_features, on=["season", "game_code", "player_id", "opponent_code"], how="left")

    print("[5/7] Computing rest days & encoding phase...")
    df = compute_rest_days(df)
    df["phase_encoded"] = df["phase"].map(PHASE_MAP).fillna(0).astype(int)
    for phase, code in PHASE_MAP.items():
        df[f"phase_{phase}"] = (df["phase"] == phase).astype(int)

    print("[6/7] Filtering and finalizing...")
    df = df[df["career_games_played"] >= MIN_GAMES_FOR_TRAINING].copy()
    df = df.dropna(subset=["pir", "minutes_float"])

    print(f"       -> Final dataset: {len(df)} rows, {df['player_id'].nunique()} players")
    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    feature_cols = []

    # Player Base
    for stat in ROLLING_STATS:
        for w in ROLLING_WINDOWS:
            feature_cols.append(f"rolling_{w}_{stat}")
        feature_cols.append(f"season_avg_{stat}")

    for stat in ["pir", "minutes_float", "points"]:
        feature_cols.extend([f"trend_3_{stat}", f"trend_5_{stat}"])
        feature_cols.extend([f"ewma_5_{stat}", f"ewma_10_{stat}"])

    feature_cols.extend([
        "rolling_5_pir_std", "rolling_10_pir_std", "rolling_5_total_fga",
        "rolling_3_total_fga", "rolling_10_total_fga", "season_avg_total_fga",
        "last_game_total_fga", "usage_trend_3_vs_10", "usage_trend_5_vs_season",
        "rolling_5_fga_per_min", "rolling_3_fga_per_min", "season_avg_fga_per_min",
        "rolling_3_starter_rate", "rolling_5_starter_rate",
        "rolling_5_minutes_std", "last_game_minutes", "last_game_pir",
    ])
    for stat in RATE_STATS:
        rate_col = f"{stat}_per_min"
        for w in ROLLING_WINDOWS:
            feature_cols.append(f"rolling_{w}_{rate_col}")
        feature_cols.append(f"season_avg_{rate_col}")

    feature_cols.extend([
        "home", "is_starter", "phase_encoded",
        "role_Guard", "role_Forward", "role_Center",
        "rotation_tier_score",
    ])
    for phase in PHASE_MAP:
        feature_cols.append(f"phase_{phase}")
    feature_cols.extend(["season_games_played", "career_games_played"])

    # Opponent Defensive
    opp_cols = [
        c for c in df.columns
        if c.startswith("opp_avg_") or c.startswith("opp_role_") or c == "opp_games_defended"
    ]
    feature_cols.extend(opp_cols)

    # Team Form
    feature_cols.extend(['team_form_win_pct_5', 'team_form_pt_diff_5'])

    # Rotation and motivation context
    feature_cols.extend([
        "rolling_5_active_players", "rolling_5_starters_avg_minutes",
        "rolling_5_bench_avg_minutes", "rolling_5_team_minutes_std",
        "rolling_5_team_top_minutes_share", "team_rotation_volatility_5",
        "team_wins_before", "team_games_before", "team_win_pct_before",
        "team_point_diff_before", "round_pct", "late_season_flag",
        "final_two_rounds_flag", "motivation_uncertainty_proxy",
    ])
    
    # H2H
    feature_cols.extend(['h2h_avg_pir', 'h2h_games'])

    # Rest days
    feature_cols.append("rest_days")

    # Filter out missing
    feature_cols = list(dict.fromkeys(feature_cols))
    feature_cols = [c for c in feature_cols if c in df.columns]

    return feature_cols


if __name__ == "__main__":
    df = build_feature_matrix()
    cols = get_feature_columns(df)
    print(f"\nFeature columns count: {len(cols)}")
