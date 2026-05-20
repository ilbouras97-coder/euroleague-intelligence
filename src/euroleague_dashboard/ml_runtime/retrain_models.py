from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

RUNTIME_DIR = Path(__file__).resolve().parent
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from .config import DB_PATH, MODELS_DIR, TARGET_COL
from .ensemble_models import CalibratedMinutesRegressor, StarterSegmentMinutesRegressor, WeightedBlendRegressor
from .feature_engineering import build_feature_matrix, get_feature_columns


MODELS_PATH = Path(MODELS_DIR)
MINUTES_BUCKETS = [0, 10, 20, 30, 60]
MINUTES_LABELS = ["0-10", "10-20", "20-30", "30+"]


def metric_row(model: str, y_true: pd.Series, y_pred: np.ndarray) -> dict:
    return {
        "model": model,
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def make_minutes_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMRegressor(
                    n_estimators=360,
                    learning_rate=0.03,
                    num_leaves=20,
                    max_depth=5,
                    subsample=0.84,
                    colsample_bytree=0.86,
                    min_child_samples=20,
                    reg_lambda=1.5,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )


def make_segmented_minutes_model(df: pd.DataFrame, base_cols: list[str], weights: pd.Series | None = None) -> StarterSegmentMinutesRegressor:
    """Fit a global minutes model plus starter/bench models."""
    global_model = make_minutes_model()
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    global_model.fit(df[base_cols], df["minutes_float"], **fit_kwargs)

    segment_models = {}
    starter_values = df["is_starter"].fillna(0).astype(int)
    for starter_value in [0, 1]:
        segment_idx = df.index[starter_values.eq(starter_value)]
        if len(segment_idx) < 500:
            continue
        model = make_minutes_model()
        segment_weights = weights.loc[segment_idx] if weights is not None else None
        segment_kwargs = {"model__sample_weight": segment_weights} if segment_weights is not None else {}
        model.fit(df.loc[segment_idx, base_cols], df.loc[segment_idx, "minutes_float"], **segment_kwargs)
        segment_models[str(starter_value)] = model

    return StarterSegmentMinutesRegressor(global_model, segment_models)


def fit_minutes_model(df: pd.DataFrame, base_cols: list[str], weights: pd.Series | None = None, segmented: bool = False):
    if segmented:
        return make_segmented_minutes_model(df, base_cols, weights)
    model = make_minutes_model()
    fit_kwargs = {"model__sample_weight": weights} if weights is not None else {}
    model.fit(df[base_cols], df["minutes_float"], **fit_kwargs)
    return model


def make_pir_models() -> dict[str, Pipeline]:
    return {
        "Ridge": Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", Ridge(alpha=6.0))]),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=420,
                        max_depth=9,
                        min_samples_leaf=3,
                        max_features=0.58,
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "LightGBM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=360,
                        learning_rate=0.028,
                        num_leaves=18,
                        max_depth=5,
                        subsample=0.78,
                        colsample_bytree=0.86,
                        min_child_samples=22,
                        reg_lambda=1.0,
                        random_state=42,
                        verbosity=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=320,
                        learning_rate=0.035,
                        max_depth=4,
                        min_child_weight=5,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        reg_lambda=2.0,
                        objective="reg:squarederror",
                        n_jobs=-1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def make_tail_classifier(scale_pos_weight: float = 1.0) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=260,
                    learning_rate=0.035,
                    num_leaves=18,
                    max_depth=5,
                    subsample=0.82,
                    colsample_bytree=0.86,
                    min_child_samples=18,
                    reg_lambda=1.0,
                    scale_pos_weight=scale_pos_weight,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )


def make_playoff_adjustment_model() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                LGBMRegressor(
                    n_estimators=180,
                    learning_rate=0.03,
                    num_leaves=10,
                    max_depth=3,
                    subsample=0.84,
                    colsample_bytree=0.82,
                    min_child_samples=14,
                    reg_lambda=2.0,
                    random_state=42,
                    verbosity=-1,
                ),
            ),
        ]
    )


def classifier_metrics(name: str, y_true: pd.Series, proba: np.ndarray) -> dict:
    positive_rate = float(pd.Series(y_true).mean())
    try:
        auc = float(roc_auc_score(y_true, proba)) if len(pd.Series(y_true).unique()) > 1 else None
    except Exception:
        auc = None
    return {
        "model": name,
        "positive_rate": round(positive_rate, 4),
        "AUC": round(auc, 4) if auc is not None else None,
        "avg_probability": round(float(np.mean(proba)), 4),
    }


def tail_risk_params(train_df: pd.DataFrame) -> dict:
    high = train_df[train_df[TARGET_COL] >= 20][TARGET_COL]
    low = train_df[train_df[TARGET_COL] <= 0][TARGET_COL]
    mid = train_df[(train_df[TARGET_COL] > 0) & (train_df[TARGET_COL] < 20)][TARGET_COL]
    mid_mean = float(mid.mean()) if not mid.empty else float(train_df[TARGET_COL].mean())
    high_mean = float(high.mean()) if not high.empty else 24.0
    low_mean = float(low.mean()) if not low.empty else -2.0
    high_rate = float((train_df[TARGET_COL] >= 20).mean())
    low_rate = float((train_df[TARGET_COL] <= 0).mean())
    return {
        "high_threshold": 20.0,
        "low_threshold": 0.0,
        "high_rate": high_rate,
        "low_rate": low_rate,
        "high_mean": high_mean,
        "mid_mean": mid_mean,
        "low_mean": low_mean,
        "high_weight": max(2.5, min(7.0, (high_mean - mid_mean) * 0.35)),
        "low_weight": max(2.0, min(5.5, (mid_mean - low_mean) * 0.34)),
    }


def apply_tail_risk_adjustment(base_pred: np.ndarray, high_prob: np.ndarray, low_prob: np.ndarray, params: dict) -> np.ndarray:
    high_delta = np.maximum(0.0, high_prob - float(params["high_rate"]))
    low_delta = np.maximum(0.0, low_prob - float(params["low_rate"]))
    return base_pred + float(params["high_weight"]) * high_delta - float(params["low_weight"]) * low_delta


def apply_playoff_adjustment(base_pred: np.ndarray, adjustment: np.ndarray, phase: pd.Series, enabled: bool) -> np.ndarray:
    if not enabled:
        return base_pred
    high_pressure = phase.astype(str).str.upper().isin(["PI", "PO", "FF"]).to_numpy()
    clipped = np.clip(adjustment, -4.0, 4.0) * 0.55
    return base_pred + np.where(high_pressure, clipped, 0.0)


def minutes_calibration_frame(df: pd.DataFrame, prediction: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(index=df.index)
    frame["prediction"] = prediction.astype(float).clip(0, 45)
    frame["actual"] = df["minutes_float"].astype(float)
    frame["residual"] = frame["actual"] - frame["prediction"]
    frame["pred_bucket"] = pd.cut(
        frame["prediction"],
        bins=[0, 8, 14, 20, 26, 60],
        labels=["0-8", "8-14", "14-20", "20-26", "26+"],
        include_lowest=True,
    ).astype(str)
    frame["starter"] = df["is_starter"].fillna(0).astype(int).astype(str)
    frame["role"] = np.select(
        [df.get("role_Guard", 0).fillna(0).astype(int).eq(1), df.get("role_Center", 0).fillna(0).astype(int).eq(1)],
        ["Guard", "Center"],
        default="Forward",
    )
    frame["phase_pressure"] = df["phase"].astype(str).str.upper().isin(["PI", "PO", "FF"]).astype(int).astype(str)
    return frame


def build_minutes_corrections(frame: pd.DataFrame) -> list[dict]:
    corrections = []
    specs = [
        (["pred_bucket", "starter", "role", "phase_pressure"], 12, 0.7),
    ]
    for keys, min_rows, shrink in specs:
        grouped = frame.groupby(keys, dropna=False).agg(count=("residual", "size"), correction=("residual", "mean")).reset_index()
        grouped = grouped[grouped["count"] >= min_rows]
        for _, row in grouped.iterrows():
            corrections.append({
                "keys": keys,
                "values": {key: str(row[key]) for key in keys},
                "count": int(row["count"]),
                "correction": float(row["correction"]) * shrink,
                "raw_correction": float(row["correction"]),
                "shrink": float(shrink),
            })
    return corrections


def apply_minutes_corrections(df: pd.DataFrame, prediction: pd.Series, corrections: list[dict]) -> pd.Series:
    frame = minutes_calibration_frame(df, prediction)
    adjusted = frame["prediction"].copy()
    for spec in corrections:
        mask = pd.Series(True, index=frame.index)
        for key in spec["keys"]:
            mask &= frame[key].eq(str(spec["values"][key]))
        adjusted.loc[mask] += float(spec["correction"])
    return adjusted.clip(0, 45)


def add_oof_predicted_minutes(df: pd.DataFrame, base_cols: list[str]) -> tuple[pd.DataFrame, Pipeline]:
    ordered = df.sort_values(["game_date_parsed", "season", "game_code"]).copy()
    oof = pd.Series(np.nan, index=ordered.index, dtype=float)
    split_count = min(5, max(2, ordered["game_code"].nunique() // 18))
    splitter = TimeSeriesSplit(n_splits=split_count)
    for train_idx, test_idx in splitter.split(ordered):
        train_rows = ordered.iloc[train_idx]
        test_rows = ordered.iloc[test_idx]
        if train_rows.empty or test_rows.empty:
            continue
        weights = train_rows["season"].map(
            lambda season: 0.7 if season < ordered["season"].max() - 1 else 0.9 if season < ordered["season"].max() else 1.0
        ).astype(float)
        model = fit_minutes_model(train_rows, base_cols, weights, segmented=False)
        oof.loc[test_rows.index] = model.predict(test_rows[base_cols])
    fallback = ordered["season_avg_minutes_float"].fillna(ordered["rolling_5_minutes_float"]).fillna(ordered["minutes_float"])
    ordered["pred_minutes"] = oof.fillna(fallback).clip(lower=0, upper=45)

    final_weights = ordered["season"].map(
        lambda season: 0.7 if season < ordered["season"].max() - 1 else 0.9 if season < ordered["season"].max() else 1.0
    ).astype(float)
    final_minutes_model = fit_minutes_model(ordered, base_cols, final_weights, segmented=False)
    calibrated_model = CalibratedMinutesRegressor(
        final_minutes_model,
        [],
        bins=[0, 8, 14, 20, 26, 60],
        labels=["0-8", "8-14", "14-20", "20-26", "26+"],
    )
    return ordered, calibrated_model


def calibration_payload(y_true: pd.Series, y_pred: pd.Series, pred_minutes: pd.Series, is_starter: pd.Series) -> tuple[dict, dict]:
    residual = y_true.astype(float) - y_pred.astype(float)
    buckets = pd.cut(pred_minutes, bins=MINUTES_BUCKETS, labels=MINUTES_LABELS, include_lowest=True).astype(str)
    starter = is_starter.fillna(0).astype(int).astype(str)
    frame = pd.DataFrame({"residual": residual, "abs_residual": residual.abs(), "bucket": buckets, "starter": starter})

    residual_calibrator = {
        "strategy": "mean_validation_residual_by_pred_minutes_bucket_and_starter",
        "min_combo_samples": 30,
        "min_bucket_samples": 30,
        "minutes_bins": MINUTES_BUCKETS,
        "minutes_labels": MINUTES_LABELS,
        "global_correction": float(frame["residual"].mean()),
        "bucket_corrections": {},
        "starter_corrections": {},
        "combo_corrections": {},
    }
    interval_calibrator = {
        "strategy": "split_conformal_abs_residual_by_pred_minutes_bucket_and_starter",
        "coverage": 0.8,
        "min_combo_samples": 30,
        "min_bucket_samples": 30,
        "minutes_bins": MINUTES_BUCKETS,
        "minutes_labels": MINUTES_LABELS,
        "global_abs_error_quantile": float(frame["abs_residual"].quantile(0.8)),
        "bucket_quantiles": {},
        "combo_quantiles": {},
    }
    for bucket, group in frame.groupby("bucket", dropna=False):
        residual_calibrator["bucket_corrections"][str(bucket)] = {"count": int(len(group)), "correction": float(group["residual"].mean())}
        interval_calibrator["bucket_quantiles"][str(bucket)] = {"count": int(len(group)), "q": float(group["abs_residual"].quantile(0.8))}
    for starter_value, group in frame.groupby("starter", dropna=False):
        residual_calibrator["starter_corrections"][str(starter_value)] = {"count": int(len(group)), "correction": float(group["residual"].mean())}
    for (bucket, starter_value), group in frame.groupby(["bucket", "starter"], dropna=False):
        key = f"{bucket}|{starter_value}"
        residual_calibrator["combo_corrections"][key] = {"count": int(len(group)), "correction": float(group["residual"].mean())}
        interval_calibrator["combo_quantiles"][key] = {"count": int(len(group)), "q": float(group["abs_residual"].quantile(0.8))}
    return residual_calibrator, interval_calibrator


def backup_models() -> Path | None:
    if not MODELS_PATH.exists():
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MODELS_PATH.parent / f"ml_models_backup_{timestamp}"
    shutil.copytree(MODELS_PATH, backup)
    return backup


def main() -> None:
    MODELS_PATH.mkdir(parents=True, exist_ok=True)
    df = build_feature_matrix(DB_PATH)
    if df.empty:
        raise RuntimeError("Feature matrix is empty.")
    df = df.sort_values(["game_date_parsed", "season", "game_code"]).reset_index(drop=True)
    base_cols = get_feature_columns(df)
    base_cols = [col for col in base_cols if col != "pred_minutes"]
    df, minutes_model = add_oof_predicted_minutes(df, base_cols)
    feature_cols = base_cols + ["pred_minutes"]

    latest_season = int(df["season"].max())
    latest_round = int(pd.to_numeric(df.loc[df["season"] == latest_season, "round"], errors="coerce").max())
    holdout_mask = (df["season"] == latest_season) & (pd.to_numeric(df["round"], errors="coerce") >= latest_round - 1)
    train_df = df[~holdout_mask].copy()
    holdout_df = df[holdout_mask].copy()
    if len(holdout_df) < 40:
        holdout_df = df.tail(max(80, int(len(df) * 0.08))).copy()
        train_df = df.drop(index=holdout_df.index).copy()
    minutes_validation_metrics = metric_row("MinutesModel", holdout_df["minutes_float"], holdout_df["pred_minutes"])

    train_weights = train_df["season"].map(lambda season: 0.65 if season < latest_season - 1 else 0.85 if season < latest_season else 1.0).astype(float)
    models = make_pir_models()
    validation_metrics = []
    fitted_models = {}
    for name, model in models.items():
        model.fit(train_df[feature_cols], train_df[TARGET_COL], **({"model__sample_weight": train_weights} if name != "Ridge" else {}))
        fitted_models[name] = model
        validation_metrics.append(metric_row(name, holdout_df[TARGET_COL], model.predict(holdout_df[feature_cols])))

    blend_weights = {"Ridge": 0.18, "RandomForest": 0.26, "LightGBM": 0.34, "XGBoost": 0.22}
    holdout_blend = WeightedBlendRegressor(fitted_models, blend_weights)
    holdout_pred = holdout_blend.predict(holdout_df[feature_cols])
    validation_metrics.append(metric_row("WeightedBlend", holdout_df[TARGET_COL], holdout_pred))
    base_predictions = {
        name: model.predict(holdout_df[feature_cols])
        for name, model in fitted_models.items()
    }
    base_predictions["WeightedBlend"] = holdout_pred
    best_base_metric = min(validation_metrics[-5:], key=lambda row: float(row["MAE"]))
    selected_base_model_name = str(best_base_metric["model"])
    if selected_base_model_name != "WeightedBlend":
        holdout_pred = base_predictions[selected_base_model_name]
        blend_weights = {name: (1.0 if name == selected_base_model_name else 0.0) for name in fitted_models}

    pressure_train_mask = train_df["phase"].astype(str).str.upper().isin(["PI", "PO", "FF"])
    pressure_holdout_mask = holdout_df["phase"].astype(str).str.upper().isin(["PI", "PO", "FF"])
    train_blend_pred = holdout_blend.predict(train_df[feature_cols])
    playoff_model = make_playoff_adjustment_model()
    playoff_enabled = False
    holdout_playoff_pred = holdout_pred.copy()
    playoff_metrics = {
        "eligible_train_rows": int(pressure_train_mask.sum()),
        "eligible_holdout_rows": int(pressure_holdout_mask.sum()),
        "enabled": False,
    }
    if pressure_train_mask.sum() >= 250 and pressure_holdout_mask.sum() >= 30:
        residual_target = train_df.loc[pressure_train_mask, TARGET_COL] - train_blend_pred[pressure_train_mask.to_numpy()]
        playoff_model.fit(train_df.loc[pressure_train_mask, feature_cols], residual_target)
        holdout_adjustment = playoff_model.predict(holdout_df[feature_cols])
        candidate_pred = apply_playoff_adjustment(holdout_pred, holdout_adjustment, holdout_df["phase"], True)
        candidate_metric = metric_row("WeightedBlend + PlayoffAdj", holdout_df[TARGET_COL], candidate_pred)
        validation_metrics.append(candidate_metric)
        playoff_enabled = float(candidate_metric["MAE"]) < float(best_base_metric["MAE"])
        playoff_metrics.update({
            "enabled": playoff_enabled,
            "candidate_MAE": float(candidate_metric["MAE"]),
            "base_MAE": float(best_base_metric["MAE"]),
        })
        if playoff_enabled:
            holdout_playoff_pred = candidate_pred
    else:
        validation_metrics.append(metric_row("WeightedBlend + PlayoffAdj", holdout_df[TARGET_COL], holdout_playoff_pred))

    high_target = (train_df[TARGET_COL] >= 20).astype(int)
    low_target = (train_df[TARGET_COL] <= 0).astype(int)
    high_weight = float((len(high_target) - high_target.sum()) / max(high_target.sum(), 1))
    low_weight = float((len(low_target) - low_target.sum()) / max(low_target.sum(), 1))
    high_classifier = make_tail_classifier(high_weight)
    low_classifier = make_tail_classifier(low_weight)
    high_classifier.fit(train_df[feature_cols], high_target)
    low_classifier.fit(train_df[feature_cols], low_target)
    holdout_high_prob = high_classifier.predict_proba(holdout_df[feature_cols])[:, 1]
    holdout_low_prob = low_classifier.predict_proba(holdout_df[feature_cols])[:, 1]
    tail_params = tail_risk_params(train_df)
    holdout_tail_base = holdout_playoff_pred
    holdout_tail_pred = apply_tail_risk_adjustment(holdout_tail_base, holdout_high_prob, holdout_low_prob, tail_params)
    blend_metric = metric_row("WeightedBlend + TailRisk", holdout_df[TARGET_COL], holdout_tail_pred)
    validation_metrics.append(blend_metric)
    base_blend_metric = metric_row("WeightedBlend + ActiveAdjustments", holdout_df[TARGET_COL], holdout_tail_base)
    tail_enabled = float(blend_metric["MAE"]) < float(base_blend_metric["MAE"])
    if not tail_enabled:
        tail_params["enabled"] = False
        calibration_pred = holdout_tail_base
    else:
        tail_params["enabled"] = True
        calibration_pred = holdout_tail_pred
    tail_classifier_metrics = [
        classifier_metrics("20+ PIR classifier", (holdout_df[TARGET_COL] >= 20).astype(int), holdout_high_prob),
        classifier_metrics("<=0 PIR classifier", (holdout_df[TARGET_COL] <= 0).astype(int), holdout_low_prob),
    ]
    residual_calibrator, interval_calibrator = calibration_payload(
        holdout_df[TARGET_COL],
        pd.Series(calibration_pred, index=holdout_df.index),
        holdout_df["pred_minutes"],
        holdout_df["is_starter"],
    )

    final_models = make_pir_models()
    final_weights = df["season"].map(lambda season: 0.65 if season < latest_season - 1 else 0.85 if season < latest_season else 1.0).astype(float)
    for name, model in final_models.items():
        model.fit(df[feature_cols], df[TARGET_COL], **({"model__sample_weight": final_weights} if name != "Ridge" else {}))
    final_blend = WeightedBlendRegressor(final_models, blend_weights)
    final_pressure_mask = df["phase"].astype(str).str.upper().isin(["PI", "PO", "FF"])
    final_playoff_model = make_playoff_adjustment_model()
    if int(final_pressure_mask.sum()) >= 250:
        final_base_pred = final_blend.predict(df[feature_cols])
        final_residual = df.loc[final_pressure_mask, TARGET_COL] - final_base_pred[final_pressure_mask.to_numpy()]
        final_playoff_model.fit(df.loc[final_pressure_mask, feature_cols], final_residual)
    final_high_target = (df[TARGET_COL] >= 20).astype(int)
    final_low_target = (df[TARGET_COL] <= 0).astype(int)
    final_high_weight = float((len(final_high_target) - final_high_target.sum()) / max(final_high_target.sum(), 1))
    final_low_weight = float((len(final_low_target) - final_low_target.sum()) / max(final_low_target.sum(), 1))
    final_high_classifier = make_tail_classifier(final_high_weight)
    final_low_classifier = make_tail_classifier(final_low_weight)
    final_high_classifier.fit(df[feature_cols], final_high_target)
    final_low_classifier.fit(df[feature_cols], final_low_target)
    final_tail_params = tail_risk_params(df)
    final_tail_params["enabled"] = tail_enabled

    backup = backup_models()
    joblib.dump(final_blend, MODELS_PATH / "best_model_ridgestacking.joblib")
    joblib.dump(minutes_model, MODELS_PATH / "model_minutes_lgbm.joblib")
    joblib.dump(
        {
            "high_classifier": final_high_classifier,
            "low_classifier": final_low_classifier,
            "playoff_adjustment_model": final_playoff_model,
            "params": final_tail_params,
            "playoff_params": {
                "enabled": playoff_enabled,
                "phases": ["PI", "PO", "FF"],
                "clip": 4.0,
                "shrink": 0.55,
            },
        },
        MODELS_PATH / "risk_models.joblib",
    )
    (MODELS_PATH / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")
    (MODELS_PATH / "residual_calibrator.json").write_text(json.dumps(residual_calibrator, indent=2), encoding="utf-8")
    (MODELS_PATH / "interval_calibrator.json").write_text(json.dumps(interval_calibrator, indent=2), encoding="utf-8")
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "target": TARGET_COL,
        "best_model_name": selected_base_model_name,
        "feature_count": len(feature_cols),
        "rows": int(len(df)),
        "train_rows_for_backtest": int(len(train_df)),
        "holdout_rows": int(len(holdout_df)),
        "holdout_season": latest_season,
        "holdout_round_min": int(pd.to_numeric(holdout_df["round"], errors="coerce").min()),
        "holdout_round_max": int(pd.to_numeric(holdout_df["round"], errors="coerce").max()),
        "latest_data_season": latest_season,
        "latest_data_round": latest_round,
        "blend_weights": blend_weights,
        "selected_base_model": selected_base_model_name,
        "validation_metrics": validation_metrics,
        "minutes_validation_metrics": minutes_validation_metrics,
        "tail_classifier_metrics": tail_classifier_metrics,
        "tail_risk_params": tail_params,
        "playoff_adjustment": playoff_metrics,
        "backup_dir": str(backup) if backup else None,
        "notes": [
            "Retrained from local euroleague.sqlite using corrected minutes-based player filtering.",
            "Final model is fitted on all played rows after holdout evaluation.",
            "Calibration payloads are computed from the latest holdout rounds.",
        ],
    }
    (MODELS_PATH / "model_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
