from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DB_PATH = DATA_DIR / "euroleague.sqlite"
FEATURES_DIR = DATA_DIR / "features"
ASSET_DIR = PROJECT_ROOT / "assets"
PLAYER_PHOTO_DIR = ASSET_DIR / "player_photos"

DEFAULT_START_SEASON = 2023
DEFAULT_END_SEASON = 2025
COMPETITION = "E"


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    PLAYER_PHOTO_DIR.mkdir(parents=True, exist_ok=True)
