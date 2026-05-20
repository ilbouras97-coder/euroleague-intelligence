from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RUNTIME_DIR = Path(__file__).resolve().parent
if str(RUNTIME_DIR) not in sys.path:
    sys.path.insert(0, str(RUNTIME_DIR))

from .config import DB_PATH, MODELS_DIR, TARGET_COL
from .feature_engineering import build_feature_matrix, get_feature_columns


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "ml_validation"
MODELS_PATH = Path(MODELS_DIR)


def metric_summary(y_true: pd.Series, y_pred: pd.Series) -> dict:
    errors = y_pred.astype(float) - y_true.astype(float)
    abs_errors = errors.abs()
    return {
        "rows": int(len(y_true)),
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(mean_squared_error(y_true, y_pred) ** 0.5), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else None,
        "Bias": round(float(errors.mean()), 4),
        "MedianAE": round(float(abs_errors.median()), 4),
        "P80_AE": round(float(abs_errors.quantile(0.8)), 4),
        "P90_AE": round(float(abs_errors.quantile(0.9)), 4),
        "Hit_Within_3": round(float((abs_errors <= 3).mean() * 100), 2),
        "Hit_Within_5": round(float((abs_errors <= 5).mean() * 100), 2),
        "Hit_Within_8": round(float((abs_errors <= 8).mean() * 100), 2),
    }


def add_role_proxy(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Role"] = "Forward"
    assists = pd.to_numeric(out.get("season_avg_assists"), errors="coerce").fillna(pd.to_numeric(out.get("rolling_5_assists"), errors="coerce"))
    rebounds = pd.to_numeric(out.get("season_avg_total_rebounds"), errors="coerce").fillna(pd.to_numeric(out.get("rolling_5_total_rebounds"), errors="coerce"))
    blocks = pd.to_numeric(out.get("season_avg_blocks_favour"), errors="coerce").fillna(pd.to_numeric(out.get("rolling_5_blocks_favour"), errors="coerce"))
    out.loc[(assists >= 3.5) | (assists >= rebounds + blocks), "Role"] = "Guard"
    out.loc[(blocks >= 0.7) | ((rebounds >= 7.0) & (assists < 2.0)), "Role"] = "Center"
    overrides = {
        "P003469": "Forward",
        "P014124": "Guard",
        "P008161": "Forward",
    }
    out["Role"] = out.apply(lambda row: overrides.get(str(row.get("player_id")), row["Role"]), axis=1)
    return out


def bucketize(df: pd.DataFrame) -> pd.DataFrame:
    out = add_role_proxy(df)
    out["minutes_bucket"] = pd.cut(out["pred_minutes"], bins=[0, 10, 20, 30, 60], labels=["0-10", "10-20", "20-30", "30+"], include_lowest=True).astype(str)
    out["actual_minutes_bucket"] = pd.cut(out["minutes_float"], bins=[0, 10, 20, 30, 60], labels=["0-10", "10-20", "20-30", "30+"], include_lowest=True).astype(str)
    out["actual_pir_bucket"] = pd.cut(out[TARGET_COL], bins=[-50, 0, 5, 10, 20, 80], labels=["<=0", "0-5", "5-10", "10-20", "20+"], include_lowest=True).astype(str)
    out["starter_bucket"] = np.where(out["is_starter"].fillna(0).astype(int).eq(1), "Starter", "Bench")
    out["volatility_bucket"] = pd.cut(pd.to_numeric(out.get("rolling_5_pir_std"), errors="coerce").fillna(0), bins=[-1, 4, 8, 12, 99], labels=["Low", "Medium", "High", "Extreme"]).astype(str)
    out["career_bucket"] = pd.cut(pd.to_numeric(out.get("career_games_played"), errors="coerce").fillna(0), bins=[0, 15, 50, 120, 1000], labels=["5-15", "16-50", "51-120", "120+"], include_lowest=True).astype(str)
    out["sample_bucket"] = pd.cut(pd.to_numeric(out.get("season_games_played"), errors="coerce").fillna(0), bins=[0, 5, 15, 30, 80], labels=["0-5", "6-15", "16-30", "30+"], include_lowest=True).astype(str)
    return out


def grouped_metrics(df: pd.DataFrame, group_col: str, min_rows: int = 8) -> pd.DataFrame:
    rows = []
    for value, group in df.groupby(group_col, dropna=False):
        if len(group) < min_rows:
            continue
        summary = metric_summary(group[TARGET_COL], group["prediction"])
        summary[group_col] = value
        summary["Avg Actual"] = round(float(group[TARGET_COL].mean()), 3)
        summary["Avg Pred"] = round(float(group["prediction"].mean()), 3)
        rows.append(summary)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("MAE", ascending=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(MODELS_PATH / "best_model_ridgestacking.joblib")
    minutes_model = joblib.load(MODELS_PATH / "model_minutes_lgbm.joblib")
    feature_cols = json.loads((MODELS_PATH / "feature_columns.json").read_text(encoding="utf-8"))
    manifest = json.loads((MODELS_PATH / "model_manifest.json").read_text(encoding="utf-8"))

    df = build_feature_matrix(DB_PATH).sort_values(["game_date_parsed", "season", "game_code"]).reset_index(drop=True)
    base_cols = [col for col in feature_cols if col != "pred_minutes"]
    missing_cols = [col for col in base_cols if col not in df.columns]
    if missing_cols:
        raise RuntimeError(f"Missing feature columns: {missing_cols[:10]}")

    latest_season = int(manifest.get("holdout_season", df["season"].max()))
    round_min = int(manifest.get("holdout_round_min", pd.to_numeric(df.loc[df["season"] == latest_season, "round"], errors="coerce").max() - 1))
    round_max = int(manifest.get("holdout_round_max", pd.to_numeric(df.loc[df["season"] == latest_season, "round"], errors="coerce").max()))
    holdout = df[(df["season"].astype(int) == latest_season) & (pd.to_numeric(df["round"], errors="coerce").between(round_min, round_max))].copy()
    if holdout.empty:
        raise RuntimeError("Holdout is empty.")

    holdout["pred_minutes"] = minutes_model.predict(holdout[base_cols]).clip(min=0, max=45)
    holdout["prediction"] = model.predict(holdout[feature_cols])
    holdout["error"] = holdout["prediction"] - holdout[TARGET_COL]
    holdout["abs_error"] = holdout["error"].abs()
    holdout = bucketize(holdout)

    overall = metric_summary(holdout[TARGET_COL], holdout["prediction"])
    groups = {}
    for col in ["phase", "round", "Role", "starter_bucket", "minutes_bucket", "actual_minutes_bucket", "actual_pir_bucket", "volatility_bucket", "career_bucket", "sample_bucket"]:
        groups[col] = grouped_metrics(holdout, col).to_dict(orient="records")

    worst_cols = [
        "season", "game_code", "game_date_parsed", "phase", "round", "player_id", "player_name",
        "team_code", "opponent_code", "Role", "starter_bucket", "minutes_float", "pred_minutes",
        TARGET_COL, "prediction", "error", "abs_error", "rolling_5_pir", "rolling_5_pir_std",
        "season_avg_pir", "h2h_avg_pir", "h2h_games",
    ]
    worst = holdout.sort_values("abs_error", ascending=False)[[col for col in worst_cols if col in holdout.columns]].head(35).copy()
    worst["game_date_parsed"] = worst["game_date_parsed"].astype(str)

    predictions_path = OUTPUT_DIR / "holdout_predictions.csv"
    worst_path = OUTPUT_DIR / "worst_errors.csv"
    report_path = OUTPUT_DIR / "validation_report.json"
    holdout.to_csv(predictions_path, index=False)
    worst.to_csv(worst_path, index=False)
    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_created_at": manifest.get("created_at"),
        "model_name": manifest.get("best_model_name"),
        "holdout": {"season": latest_season, "round_min": round_min, "round_max": round_max, "rows": int(len(holdout))},
        "overall": overall,
        "groups": groups,
        "files": {"predictions": str(predictions_path), "worst_errors": str(worst_path)},
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
