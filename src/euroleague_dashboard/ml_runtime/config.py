from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BASE_DIR = PROJECT_ROOT
DB_PATH = str(PROJECT_ROOT / "data" / "euroleague.sqlite")
MODELS_DIR = str(PROJECT_ROOT / "data" / "ml_models")
OUTPUT_DIR = str(PROJECT_ROOT / "data" / "ml_output")

TRAIN_SEASONS = [2023, 2024]
VAL_SEASON = 2025
VAL_TEST_SPLIT_RATIO = 0.7

ROLLING_WINDOWS = [3, 5, 10]
ROLLING_STATS = [
    "pir", "minutes_float", "points", "total_rebounds",
    "assists", "steals", "turnovers", "blocks_favour",
    "fouls_committed", "plus_minus",
    "field_goals_made_2", "field_goals_attempted_2",
    "field_goals_made_3", "field_goals_attempted_3",
    "free_throws_made", "free_throws_attempted",
]
OPPONENT_DEFENSIVE_STATS = ["pir", "points", "total_rebounds", "assists"]
PHASE_MAP = {"RS": 0, "PI": 1, "PO": 2, "FF": 3}
MIN_GAMES_FOR_TRAINING = 5
TARGET_COL = "pir"
