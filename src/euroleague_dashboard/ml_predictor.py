from __future__ import annotations

import json
import sys
import warnings
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .config import DB_PATH, PROJECT_ROOT

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


ML_RUNTIME_DIR = Path(__file__).resolve().parent / "ml_runtime"
ML_MODELS_DIR = PROJECT_ROOT / "data" / "ml_models"
ROTATION_IMPACT_PATH = PROJECT_ROOT / "data" / "rotation_impact.csv"

if str(ML_RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(ML_RUNTIME_DIR))

from feature_engineering import (  # type: ignore  # noqa: E402
    RATE_STATS,
    ROLLING_STATS,
    ROLLING_WINDOWS,
    PHASE_MAP,
    PLAYER_STYLE_BUCKETS,
    clean_data,
    compute_h2h_features,
    compute_opponent_features,
    compute_opponent_role_features,
    compute_opponent_style_features,
    compute_player_features,
    compute_team_availability_pressure,
    compute_team_form,
    compute_player_team_rotation_context,
    compute_team_rotation_features,
    compute_team_standings_features,
    infer_player_roles,
    infer_player_styles,
    load_raw_data,
)


MINUTES_BUCKETS = [0, 10, 20, 30, 60]
MINUTES_BUCKET_LABELS = ["0-10", "10-20", "20-30", "30+"]
UNAVAILABLE_STATUSES = {"out", "injured", "inactive", "suspended"}
LIMITED_STATUSES = {"doubtful", "questionable", "day-to-day", "probable"}


def _minutes_bucket(value: float) -> str:
    return pd.cut(
        pd.Series([value]),
        bins=MINUTES_BUCKETS,
        labels=MINUTES_BUCKET_LABELS,
        include_lowest=True,
    ).astype(str).iloc[0]


@lru_cache(maxsize=1)
def load_ml_artifacts():
    model = joblib.load(ML_MODELS_DIR / "best_model_ridgestacking.joblib")
    minutes_model = joblib.load(ML_MODELS_DIR / "model_minutes_lgbm.joblib")
    feature_cols = json.loads((ML_MODELS_DIR / "feature_columns.json").read_text())
    residual_calibrator = json.loads((ML_MODELS_DIR / "residual_calibrator.json").read_text())
    interval_calibrator = json.loads((ML_MODELS_DIR / "interval_calibrator.json").read_text())
    risk_path = ML_MODELS_DIR / "risk_models.joblib"
    risk_models = joblib.load(risk_path) if risk_path.exists() else {}
    manifest_path = ML_MODELS_DIR / "model_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return model, minutes_model, feature_cols, residual_calibrator, interval_calibrator, risk_models, manifest


@lru_cache(maxsize=1)
def load_prediction_context(db_path: str = str(DB_PATH)) -> dict:
    games, boxscores = load_raw_data(db_path)
    games, boxscores = clean_data(games, boxscores)
    player_features = compute_player_features(boxscores)
    return {
        "games": games,
        "boxscores": boxscores,
        "player_features": player_features,
        "player_roles": infer_player_roles(boxscores),
        "player_styles": infer_player_styles(boxscores),
        "opponent_features": compute_opponent_features(games, boxscores),
        "opponent_role_features": compute_opponent_role_features(games, boxscores),
        "opponent_style_features": compute_opponent_style_features(games, boxscores),
        "team_form": compute_team_form(games),
        "team_availability": compute_team_availability_pressure(player_features),
        "player_team_context": compute_player_team_rotation_context(player_features),
        "team_rotation": compute_team_rotation_features(boxscores),
        "team_standings": compute_team_standings_features(games),
        "h2h": compute_h2h_features(boxscores),
    }


def clear_ml_caches() -> None:
    load_ml_artifacts.cache_clear()
    load_prediction_context.cache_clear()
    load_current_availability.cache_clear()
    load_rotation_impact.cache_clear()


def _normalize_name(value: str) -> str:
    return " ".join(str(value or "").upper().replace(",", " ").split())


@lru_cache(maxsize=1)
def load_current_availability() -> pd.DataFrame:
    frames = []
    for path in [PROJECT_ROOT / "data" / "player_availability_collected.csv", PROJECT_ROOT / "data" / "player_availability.csv"]:
        if not path.exists():
            continue
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame(columns=["player", "team_code", "status", "impact", "note"])
    frame = pd.concat(frames, ignore_index=True)
    for col in ["player", "team_code", "status", "impact", "note"]:
        if col not in frame.columns:
            frame[col] = np.nan
    frame["player_norm"] = frame["player"].map(_normalize_name)
    frame["team_code"] = frame["team_code"].astype(str).str.upper().str.strip()
    frame["status_norm"] = frame["status"].astype(str).str.lower().str.strip()
    frame["impact_num"] = pd.to_numeric(frame["impact"], errors="coerce")
    return frame


@lru_cache(maxsize=1)
def load_rotation_impact() -> pd.DataFrame:
    if not ROTATION_IMPACT_PATH.exists():
        return pd.DataFrame()
    try:
        frame = pd.read_csv(ROTATION_IMPACT_PATH)
    except Exception:
        return pd.DataFrame()
    if "player_id" not in frame.columns:
        return pd.DataFrame()
    frame["player_id"] = frame["player_id"].fillna("").astype(str)
    for col in [
        "net_minutes_delta",
        "rotation_impact_score",
        "team_missing_minutes",
        "role_missing_minutes",
        "availability_impact",
    ]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0.0)
    return frame


def _status_impact(row: pd.Series) -> float:
    status = str(row.get("status_norm", "")).lower()
    explicit = pd.to_numeric(pd.Series([row.get("impact_num")]), errors="coerce").iloc[0]
    if pd.notna(explicit):
        return float(max(0.0, min(1.0, explicit)))
    if status in UNAVAILABLE_STATUSES:
        return 1.0
    if status == "doubtful":
        return 0.65
    if status in LIMITED_STATUSES:
        return 0.35
    return 0.0


def _availability_minutes_adjustment(latest: pd.Series, team_code: str, pred_minutes: float) -> tuple[float, dict]:
    rotation_impact = load_rotation_impact()
    if not rotation_impact.empty:
        player_rows = rotation_impact[rotation_impact["player_id"] == str(latest.get("player_id", ""))]
        if not player_rows.empty:
            row = player_rows.iloc[-1]
            status = str(row.get("availability_status", "Available"))
            availability_impact = _safe_float(row.get("availability_impact"), 1.0)
            if status.strip().lower() in {"out", "injured", "inactive", "dnp"} or availability_impact <= 0.25:
                return 0.0, {
                    "availability_minutes_delta": -pred_minutes,
                    "availability_status": status,
                    "team_absence_impact": _safe_float(row.get("team_missing_minutes"), 0.0),
                }
            delta = max(-8.0, min(8.0, _safe_float(row.get("net_minutes_delta"), 0.0)))
            adjusted = min(35.0, max(0.0, pred_minutes + delta))
            return adjusted, {
                "availability_minutes_delta": adjusted - pred_minutes,
                "availability_status": status,
                "team_absence_impact": _safe_float(row.get("team_missing_minutes"), 0.0),
            }

    availability = load_current_availability()
    if availability.empty:
        return pred_minutes, {"availability_minutes_delta": 0.0, "availability_status": "Available", "team_absence_impact": 0.0}

    player_name = _normalize_name(str(latest.get("player_name", "")))
    team_rows = availability[availability["team_code"] == str(team_code).upper()].copy()
    if team_rows.empty:
        return pred_minutes, {"availability_minutes_delta": 0.0, "availability_status": "Available", "team_absence_impact": 0.0}
    team_rows["impact_weight"] = team_rows.apply(_status_impact, axis=1)

    own_rows = team_rows[team_rows["player_norm"] == player_name]
    if not own_rows.empty:
        own_impact = float(own_rows["impact_weight"].max())
        own_status = str(own_rows.iloc[0].get("status", "Limited"))
        if own_impact >= 0.95:
            return 0.0, {
                "availability_minutes_delta": -pred_minutes,
                "availability_status": own_status,
                "team_absence_impact": float(team_rows["impact_weight"].sum()),
            }
        limited_minutes = pred_minutes * (1.0 - 0.45 * own_impact)
        return limited_minutes, {
            "availability_minutes_delta": limited_minutes - pred_minutes,
            "availability_status": own_status,
            "team_absence_impact": float(team_rows["impact_weight"].sum()),
        }

    teammate_impact = float(team_rows["impact_weight"].sum())
    if teammate_impact <= 0:
        return pred_minutes, {"availability_minutes_delta": 0.0, "availability_status": "Available", "team_absence_impact": 0.0}
    role = str(latest.get("player_role", "Forward"))
    starter_rate = _safe_float(latest.get("rolling_5_starter_rate"), 0.0)
    usage_rate = _safe_float(latest.get("rolling_5_fga_per_min"), 0.0)
    beneficiary = 0.75 + 0.20 * starter_rate + min(0.25, usage_rate)
    if role == "Guard":
        beneficiary += 0.10
    minute_delta = min(3.2, teammate_impact * beneficiary)
    adjusted = min(35.0, pred_minutes + minute_delta)
    return adjusted, {
        "availability_minutes_delta": adjusted - pred_minutes,
        "availability_status": "Available",
        "team_absence_impact": teammate_impact,
    }


def _correction_for_prediction(pred_minutes: float, is_starter: int, calibrator: dict) -> float:
    bucket = _minutes_bucket(pred_minutes)
    combo_key = f"{bucket}|{int(is_starter)}"
    combo = calibrator.get("combo_corrections", {}).get(combo_key)
    if combo and combo["count"] >= calibrator.get("min_combo_samples", 30):
        return float(combo["correction"])

    bucket_info = calibrator.get("bucket_corrections", {}).get(bucket)
    if bucket_info and bucket_info["count"] >= calibrator.get("min_bucket_samples", 30):
        return float(bucket_info["correction"])

    starter_info = calibrator.get("starter_corrections", {}).get(str(int(is_starter)))
    if starter_info:
        return float(starter_info["correction"])

    return float(calibrator.get("global_correction", 0.0))


def _interval_for_prediction(predicted_pir: float, pred_minutes: float, is_starter: int, calibrator: dict) -> dict:
    bucket = _minutes_bucket(pred_minutes)
    combo_key = f"{bucket}|{int(is_starter)}"
    combo = calibrator.get("combo_quantiles", {}).get(combo_key)
    if combo and combo["count"] >= calibrator.get("min_combo_samples", 30):
        radius = float(combo["q"])
    else:
        bucket_info = calibrator.get("bucket_quantiles", {}).get(bucket)
        if bucket_info and bucket_info["count"] >= calibrator.get("min_bucket_samples", 30):
            radius = float(bucket_info["q"])
        else:
            radius = float(calibrator.get("global_abs_error_quantile", 0.0))

    confidence = "High" if radius <= 5.5 else "Medium" if radius <= 8.5 else "Low"
    return {
        "interval_low": predicted_pir - radius,
        "interval_high": predicted_pir + radius,
        "interval_radius": radius,
        "confidence": confidence,
    }


def _apply_player_latest_stats(latest: pd.Series, player_history: pd.DataFrame) -> pd.Series:
    current_season = int(player_history["season"].iloc[-1])
    for stat in ROLLING_STATS:
        if stat not in player_history.columns:
            continue
        for window in ROLLING_WINDOWS:
            latest[f"rolling_{window}_{stat}"] = player_history[stat].tail(window).mean()
        latest[f"season_avg_{stat}"] = player_history[player_history["season"] == current_season][stat].mean()

    for stat in ["pir", "minutes_float", "points"]:
        latest[f"trend_5_{stat}"] = latest.get(f"rolling_5_{stat}", np.nan) - latest.get(f"season_avg_{stat}", np.nan)
        latest[f"trend_3_{stat}"] = latest.get(f"rolling_3_{stat}", np.nan) - latest.get(f"season_avg_{stat}", np.nan)
        latest[f"ewma_5_{stat}"] = player_history[stat].ewm(span=5, adjust=False).mean().iloc[-1]
        latest[f"ewma_10_{stat}"] = player_history[stat].ewm(span=10, adjust=False).mean().iloc[-1]

    pir_values = player_history["pir"].values
    latest["rolling_5_pir_std"] = np.std(pir_values[-5:]) if len(pir_values) >= 2 else 0
    latest["rolling_10_pir_std"] = np.std(pir_values[-10:]) if len(pir_values) >= 3 else 0

    total_fga = (
        player_history["field_goals_attempted_2"].fillna(0)
        + player_history["field_goals_attempted_3"].fillna(0)
        + player_history["free_throws_attempted"].fillna(0)
    )
    latest["rolling_5_total_fga"] = total_fga.tail(5).mean()
    latest["rolling_3_total_fga"] = total_fga.tail(3).mean()
    latest["rolling_10_total_fga"] = total_fga.tail(10).mean()
    latest["season_avg_total_fga"] = total_fga[player_history["season"] == current_season].mean()
    latest["last_game_total_fga"] = total_fga.iloc[-1]
    latest["usage_trend_3_vs_10"] = latest["rolling_3_total_fga"] - latest["rolling_10_total_fga"]
    latest["usage_trend_5_vs_season"] = latest["rolling_5_total_fga"] - latest["season_avg_total_fga"]
    latest["rolling_5_fga_per_min"] = _safe_float(latest["rolling_5_total_fga"]) / max(_safe_float(latest.get("rolling_5_minutes_float")), 0.1)
    latest["rolling_3_fga_per_min"] = _safe_float(latest["rolling_3_total_fga"]) / max(_safe_float(latest.get("rolling_3_minutes_float")), 0.1)
    latest["season_avg_fga_per_min"] = _safe_float(latest["season_avg_total_fga"]) / max(_safe_float(latest.get("season_avg_minutes_float")), 0.1)

    for stat in RATE_STATS:
        if stat not in player_history.columns:
            continue
        rate = player_history[stat] / player_history["minutes_float"].replace(0, np.nan)
        for window in ROLLING_WINDOWS:
            latest[f"rolling_{window}_{stat}_per_min"] = rate.tail(window).mean()
        latest[f"season_avg_{stat}_per_min"] = rate[player_history["season"] == current_season].mean()

    starter_rate = player_history["is_starter"].fillna(0).astype(int)
    latest["rolling_3_starter_rate"] = starter_rate.tail(3).mean()
    latest["rolling_5_starter_rate"] = starter_rate.tail(5).mean()
    latest["rolling_5_minutes_std"] = player_history["minutes_float"].tail(5).std()
    latest["last_game_minutes"] = player_history["minutes_float"].iloc[-1]
    latest["last_game_pir"] = player_history["pir"].iloc[-1]
    latest["season_games_played"] = len(player_history[player_history["season"] == current_season])
    latest["career_games_played"] = len(player_history)
    for window in [3, 5, 10]:
        latest[f"current_{window}_pir"] = player_history["pir"].tail(window).mean()
        latest[f"current_{window}_minutes"] = player_history["minutes_float"].tail(window).mean()
        latest[f"current_{window}_pir_std"] = player_history["pir"].tail(window).std()
    return latest


def _safe_float(value, default: float = 0.0) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(parsed) if pd.notna(parsed) else default


def _tail_risk_adjustment(expected_pir: float, x_final: pd.DataFrame, risk_models: dict) -> tuple[float, dict]:
    if not risk_models:
        return expected_pir, {"high_pir_probability": 0.0, "low_pir_probability": 0.0, "tail_risk_adjustment": 0.0}
    high_model = risk_models.get("high_classifier")
    low_model = risk_models.get("low_classifier")
    params = risk_models.get("params", {})
    if high_model is None or low_model is None or not params:
        return expected_pir, {"high_pir_probability": 0.0, "low_pir_probability": 0.0, "tail_risk_adjustment": 0.0}
    high_prob = float(high_model.predict_proba(x_final)[0, 1])
    low_prob = float(low_model.predict_proba(x_final)[0, 1])
    if not bool(params.get("enabled", False)):
        return expected_pir, {
            "high_pir_probability": high_prob,
            "low_pir_probability": low_prob,
            "tail_risk_adjustment": 0.0,
        }
    high_delta = max(0.0, high_prob - float(params.get("high_rate", 0.0)))
    low_delta = max(0.0, low_prob - float(params.get("low_rate", 0.0)))
    adjustment = float(params.get("high_weight", 0.0)) * high_delta - float(params.get("low_weight", 0.0)) * low_delta
    adjusted = expected_pir + adjustment
    return adjusted, {
        "high_pir_probability": high_prob,
        "low_pir_probability": low_prob,
        "tail_risk_adjustment": adjustment,
    }


def _playoff_model_adjustment(expected_pir: float, phase: str, x_final: pd.DataFrame, risk_models: dict) -> tuple[float, float]:
    params = risk_models.get("playoff_params", {}) if risk_models else {}
    model = risk_models.get("playoff_adjustment_model") if risk_models else None
    if model is None or not bool(params.get("enabled", False)):
        return expected_pir, 0.0
    if str(phase).upper() not in set(params.get("phases", ["PI", "PO", "FF"])):
        return expected_pir, 0.0
    raw_adjustment = float(model.predict(x_final)[0])
    clipped = max(-float(params.get("clip", 4.0)), min(float(params.get("clip", 4.0)), raw_adjustment))
    adjustment = clipped * float(params.get("shrink", 0.55))
    return expected_pir + adjustment, adjustment


def _blend_prediction_with_current_context(
    expected_pir: float,
    pred_minutes: float,
    is_starter: int,
    phase: str,
    latest: pd.Series,
    interval: dict,
) -> tuple[float, dict]:
    recent_3 = _safe_float(latest.get("current_3_pir"), expected_pir)
    recent_5 = _safe_float(latest.get("current_5_pir"), recent_3)
    season_avg = _safe_float(latest.get("season_avg_pir"), recent_5)
    h2h_games = _safe_float(latest.get("h2h_games"), 0.0)
    h2h_avg = _safe_float(latest.get("h2h_avg_pir"), season_avg)
    recent_3_minutes = _safe_float(latest.get("current_3_minutes"), pred_minutes)
    recent_5_minutes = _safe_float(latest.get("current_5_minutes"), pred_minutes)
    pir_volatility = max(
        _safe_float(latest.get("current_5_pir_std"), 0.0),
        _safe_float(latest.get("rolling_5_pir_std"), 0.0),
    )
    minutes_volatility = max(
        _safe_float(latest.get("rolling_5_minutes_std"), 0.0),
        abs(pred_minutes - recent_3_minutes),
    )
    radius = _safe_float(interval.get("interval_radius"), 0.0)

    if h2h_games >= 3:
        context_prior = 0.42 * recent_3 + 0.28 * recent_5 + 0.20 * h2h_avg + 0.10 * season_avg
    else:
        context_prior = 0.48 * recent_3 + 0.32 * recent_5 + 0.20 * season_avg

    high_pressure_phase = str(phase).upper() in {"PI", "PO", "FF"}
    if high_pressure_phase:
        model_weight = 0.48 if radius >= 8.5 else 0.58
        volatility_penalty = max(0.0, radius - 6.5) * 0.38 + max(0.0, pir_volatility - 7.0) * 0.16
        minutes_penalty = max(0.0, pred_minutes - recent_5_minutes - 3.0) * 0.22
    else:
        model_weight = 0.74 if radius >= 8.5 else 0.82
        volatility_penalty = max(0.0, radius - 8.0) * 0.16 + max(0.0, pir_volatility - 9.0) * 0.07
        minutes_penalty = max(0.0, pred_minutes - recent_5_minutes - 5.0) * 0.12

    adjusted = model_weight * expected_pir + (1.0 - model_weight) * context_prior
    adjusted -= volatility_penalty + minutes_penalty

    if high_pressure_phase and is_starter and radius >= 8.5:
        current_floor = min(recent_3, recent_5, season_avg)
        downside_anchor = 0.55 * current_floor + 0.45 * expected_pir
        adjusted = min(adjusted, downside_anchor)

    if not is_starter:
        adjusted = 0.70 * adjusted + 0.30 * recent_3
        adjusted -= max(0.0, 14.0 - pred_minutes) * 0.05

    adjusted = max(-5.0, adjusted)
    info = {
        "context_prior": context_prior,
        "recent_3_pir": recent_3,
        "recent_5_pir": recent_5,
        "recent_3_minutes": recent_3_minutes,
        "recent_5_minutes": recent_5_minutes,
        "pir_volatility": pir_volatility,
        "minutes_volatility": minutes_volatility,
        "playoff_adjustment": adjusted - expected_pir,
    }
    return adjusted, info


def _build_latest_features(context: dict, player_id: str, opponent_code: str, team_code_override: str | None = None) -> pd.Series:
    boxscores = context["boxscores"]
    player_history = boxscores[boxscores["player_id"] == player_id].copy()
    if player_history.empty:
        raise ValueError(f"Player {player_id} not found in boxscores.")

    player_features = context["player_features"]
    latest_rows = player_features[player_features["player_id"] == player_id]
    latest = latest_rows.iloc[-1].copy()
    latest = _apply_player_latest_stats(latest, player_history)

    if team_code_override:
        latest["team_code"] = team_code_override
    team_code = latest["team_code"]

    player_roles = context.get("player_roles", pd.DataFrame())
    role_rows = player_roles[player_roles["player_id"] == player_id] if not player_roles.empty else pd.DataFrame()
    latest["player_role"] = role_rows.iloc[-1]["player_role"] if not role_rows.empty else "Forward"
    player_styles = context.get("player_styles", pd.DataFrame())
    style_rows = player_styles[player_styles["player_id"] == player_id] if not player_styles.empty else pd.DataFrame()
    latest["player_style"] = style_rows.iloc[-1]["player_style"] if not style_rows.empty else "wing_scorer"
    for role in ["Guard", "Forward", "Center"]:
        latest[f"role_{role}"] = 1 if latest["player_role"] == role else 0
    for style in PLAYER_STYLE_BUCKETS:
        latest[f"style_{style}"] = 1 if latest["player_style"] == style else 0
    latest["rotation_tier_score"] = min(
        2.0,
        0.55 * (_safe_float(latest.get("rolling_5_minutes_float"), 0.0) / 30.0)
        + 0.45 * (_safe_float(latest.get("rolling_5_total_fga"), 0.0) / 12.0),
    )

    opp_data = context["opponent_features"][context["opponent_features"]["defending_team"] == opponent_code]
    for key in ["opp_avg_allowed_pir", "opp_avg_allowed_points", "opp_avg_allowed_rebounds", "opp_avg_allowed_assists", "opp_games_defended"]:
        latest[key] = opp_data.iloc[-1].get(key, np.nan) if not opp_data.empty else np.nan

    opp_role_features = context.get("opponent_role_features", pd.DataFrame())
    if not opp_role_features.empty:
        opp_role_data = opp_role_features[
            (opp_role_features["defending_team"] == opponent_code)
            & (opp_role_features["player_role"] == latest["player_role"])
        ]
    else:
        opp_role_data = pd.DataFrame()
    for key in [
        "opp_role_avg_role_allowed_pir", "opp_role_avg_role_allowed_points",
        "opp_role_avg_role_allowed_rebounds", "opp_role_avg_role_allowed_assists",
        "opp_role_avg_role_allowed_minutes", "opp_role_avg_role_allowed_players",
        "opp_role_games_defended",
    ]:
        latest[key] = opp_role_data.iloc[-1].get(key, np.nan) if not opp_role_data.empty else np.nan

    opp_style_features = context.get("opponent_style_features", pd.DataFrame())
    if not opp_style_features.empty:
        opp_style_data = opp_style_features[
            (opp_style_features["defending_team"] == opponent_code)
            & (opp_style_features["player_style"] == latest["player_style"])
        ]
    else:
        opp_style_data = pd.DataFrame()
    for key in [
        "opp_style_avg_style_allowed_pir", "opp_style_avg_style_allowed_points",
        "opp_style_avg_style_allowed_fga", "opp_style_avg_style_allowed_minutes",
        "opp_style_games_defended",
    ]:
        latest[key] = opp_style_data.iloc[-1].get(key, np.nan) if not opp_style_data.empty else np.nan

    team_form = context["team_form"][context["team_form"]["team_code"] == team_code]
    latest["team_form_win_pct_5"] = team_form.iloc[-1].get("team_form_win_pct_5", np.nan) if not team_form.empty else np.nan
    latest["team_form_pt_diff_5"] = team_form.iloc[-1].get("team_form_pt_diff_5", np.nan) if not team_form.empty else np.nan

    team_rotation = context["team_rotation"][context["team_rotation"]["team_code"] == team_code]
    for key in [
        "rolling_5_active_players", "rolling_5_starters_avg_minutes",
        "rolling_5_bench_avg_minutes", "rolling_5_team_minutes_std",
        "rolling_5_team_top_minutes_share", "team_rotation_volatility_5",
    ]:
        latest[key] = team_rotation.iloc[-1].get(key, np.nan) if not team_rotation.empty else np.nan

    team_availability = context.get("team_availability", pd.DataFrame())
    team_availability = team_availability[team_availability["team_code"] == team_code] if not team_availability.empty else pd.DataFrame()
    for key in [
        "team_active_prior_minutes_sum", "team_active_prior_usage_sum",
        "team_top6_prior_minutes_sum", "team_top8_prior_usage_sum",
        "team_core_available_score", "team_usage_available_score",
        "team_rotation_shortage_score",
    ]:
        latest[key] = team_availability.iloc[-1].get(key, np.nan) if not team_availability.empty else np.nan

    player_team_context = context.get("player_team_context", pd.DataFrame())
    if not player_team_context.empty:
        player_team_context = player_team_context[
            (player_team_context["team_code"] == team_code)
            & (player_team_context["player_id"] == player_id)
        ]
    for key in [
        "team_prior_minutes_sum", "team_prior_usage_sum",
        "team_prior_minutes_rank", "team_prior_usage_rank",
        "team_prior_rotation_count", "team_prior_minutes_rank_pct",
        "team_prior_usage_rank_pct", "team_prior_minutes_share",
        "team_prior_usage_share", "recent_minutes_vs_team_avg",
        "recent_usage_vs_team_avg", "top3_minutes_rotation",
        "top6_minutes_rotation", "top3_usage_rotation",
    ]:
        latest[key] = player_team_context.iloc[-1].get(key, np.nan) if not player_team_context.empty else np.nan

    team_standings = context["team_standings"][context["team_standings"]["team_code"] == team_code]
    for key in [
        "team_wins_before", "team_games_before", "team_win_pct_before",
        "team_point_diff_before", "round_pct", "late_season_flag",
        "final_two_rounds_flag", "motivation_uncertainty_proxy",
    ]:
        latest[key] = team_standings.iloc[-1].get(key, np.nan) if not team_standings.empty else np.nan

    matchups = player_history[player_history["opponent_code"] == opponent_code]
    latest["h2h_avg_pir"] = matchups["pir"].mean() if not matchups.empty else latest.get("season_avg_pir", np.nan)
    latest["h2h_games"] = len(matchups)
    return latest


def predict_player(
    player_id: str,
    opponent_code: str,
    home: int,
    phase: str,
    is_starter: int | None = None,
    rest_days: int = 4,
    team_code_override: str | None = None,
) -> dict:
    model, minutes_model, feature_cols, residual_calibrator, interval_calibrator, risk_models, manifest = load_ml_artifacts()
    context = load_prediction_context(str(DB_PATH))
    latest = _build_latest_features(context, player_id, opponent_code, team_code_override)

    starter_value = int(latest.get("is_starter", 0)) if is_starter is None else int(is_starter)
    latest["home"] = int(home)
    latest["is_starter"] = starter_value
    latest["rest_days"] = min(int(rest_days), 30)
    latest["phase_encoded"] = PHASE_MAP.get(phase, 0)
    for phase_code in PHASE_MAP:
        latest[f"phase_{phase_code}"] = 1 if phase_code == phase else 0

    base_cols = [col for col in feature_cols if col != "pred_minutes"]
    x_base = pd.DataFrame([latest])[base_cols]
    pred_minutes = float(minutes_model.predict(x_base)[0])
    pred_minutes, availability_info = _availability_minutes_adjustment(latest, str(latest.get("team_code", "")), pred_minutes)

    latest["pred_minutes"] = pred_minutes
    x_final = pd.DataFrame([latest])[feature_cols]
    raw_pir = float(model.predict(x_final)[0])
    correction = _correction_for_prediction(pred_minutes, starter_value, residual_calibrator)
    model_expected_pir = raw_pir + correction
    risk_expected_pir, tail_info = _tail_risk_adjustment(model_expected_pir, x_final, risk_models)
    playoff_expected_pir, playoff_model_delta = _playoff_model_adjustment(risk_expected_pir, phase, x_final, risk_models)
    raw_interval = _interval_for_prediction(model_expected_pir, pred_minutes, starter_value, interval_calibrator)
    expected_pir, context_adjustment = _blend_prediction_with_current_context(
        playoff_expected_pir,
        pred_minutes,
        starter_value,
        phase,
        latest,
        raw_interval,
    )
    interval = _interval_for_prediction(expected_pir, pred_minutes, starter_value, interval_calibrator)
    status_norm = str(availability_info["availability_status"]).strip().lower()
    if pred_minutes <= 0.05 or status_norm in {"out", "injured", "inactive", "dnp"}:
        expected_pir = 0.0
        interval = {
            "interval_low": 0.0,
            "interval_high": 0.0,
            "interval_radius": 0.0,
            "confidence": "Out",
        }

    return {
        "player_id": player_id,
        "player_name": str(latest.get("player_name", player_id)),
        "team_code": str(latest.get("team_code", "")),
        "opponent_code": opponent_code,
        "home": int(home),
        "phase": phase,
        "is_starter": starter_value,
        "predicted_minutes": round(pred_minutes, 1),
        "availability_status": availability_info["availability_status"],
        "availability_minutes_delta": round(float(availability_info["availability_minutes_delta"]), 2),
        "team_absence_impact": round(float(availability_info["team_absence_impact"]), 2),
        "raw_predicted_pir": round(raw_pir, 2),
        "calibration_adjustment": round(correction, 2),
        "model_expected_pir": round(model_expected_pir, 2),
        "tail_risk_adjustment": round(float(tail_info["tail_risk_adjustment"]), 2),
        "playoff_model_adjustment": round(float(playoff_model_delta), 2),
        "high_pir_probability": round(float(tail_info["high_pir_probability"]), 3),
        "low_pir_probability": round(float(tail_info["low_pir_probability"]), 3),
        "context_adjustment": round(float(context_adjustment["playoff_adjustment"]), 2),
        "predicted_pir": round(expected_pir, 2),
        "interval_low": round(float(interval["interval_low"]), 2),
        "interval_high": round(float(interval["interval_high"]), 2),
        "interval_radius": round(float(interval["interval_radius"]), 2),
        "confidence": interval["confidence"],
        "h2h_avg_pir": round(float(latest.get("h2h_avg_pir", 0) or 0), 2),
        "h2h_games": int(latest.get("h2h_games", 0) or 0),
        "recent_3_pir": round(float(context_adjustment["recent_3_pir"]), 2),
        "recent_5_pir": round(float(context_adjustment["recent_5_pir"]), 2),
        "recent_3_minutes": round(float(context_adjustment["recent_3_minutes"]), 2),
        "pir_volatility": round(float(context_adjustment["pir_volatility"]), 2),
        "model_name": manifest.get("best_model_name", "RidgeStacking"),
    }


def predict_players(
    player_ids: list[str],
    opponent_code: str,
    home: int,
    phase: str,
    rest_days: int = 4,
    team_code_override: str | None = None,
) -> pd.DataFrame:
    records = []
    for player_id in player_ids:
        try:
            records.append(predict_player(player_id, opponent_code, home, phase, None, rest_days, team_code_override))
        except Exception as exc:
            records.append({"player_id": player_id, "error": str(exc)})
    return pd.DataFrame(records)
