"""Small serializable ensemble helpers for PIR prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin


class WeightedBlendRegressor(RegressorMixin, BaseEstimator):
    """Blend already-fitted regressors with fixed weights."""

    def __init__(self, models: dict, weights: dict):
        self.models = models
        self.weights = weights

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = []
        weights = []
        for name, model in self.models.items():
            preds.append(model.predict(X))
            weights.append(self.weights[name])

        pred_matrix = np.column_stack(preds)
        weight_array = np.array(weights, dtype=float)
        weight_array = weight_array / weight_array.sum()
        return pred_matrix @ weight_array


class CalibratedMinutesRegressor(RegressorMixin, BaseEstimator):
    """Apply grouped residual corrections on top of a fitted minutes model."""

    def __init__(self, model, corrections: list[dict], bins: list[float], labels: list[str], clip_low: float = 0.0, clip_high: float = 45.0):
        self.model = model
        self.corrections = corrections
        self.bins = bins
        self.labels = labels
        self.clip_low = clip_low
        self.clip_high = clip_high

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def _roles(self, X: pd.DataFrame) -> pd.Series:
        if "role_Guard" in X.columns and "role_Center" in X.columns:
            return pd.Series(
                np.select(
                    [X["role_Guard"].fillna(0).astype(float).eq(1), X["role_Center"].fillna(0).astype(float).eq(1)],
                    ["Guard", "Center"],
                    default="Forward",
                ),
                index=X.index,
            )
        return pd.Series("Forward", index=X.index)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        base = np.asarray(self.model.predict(X), dtype=float)
        frame = pd.DataFrame(index=X.index)
        frame["prediction"] = np.clip(base, self.clip_low, self.clip_high)
        frame["pred_bucket"] = pd.cut(
            frame["prediction"],
            bins=self.bins,
            labels=self.labels,
            include_lowest=True,
        ).astype(str)
        frame["starter"] = X.get("is_starter", pd.Series(0, index=X.index)).fillna(0).astype(int).astype(str)
        frame["role"] = self._roles(X)
        phase_flags = [col for col in ["phase_PI", "phase_PO", "phase_FF"] if col in X.columns]
        if phase_flags:
            frame["phase_pressure"] = X[phase_flags].fillna(0).sum(axis=1).gt(0).astype(int).astype(str)
        else:
            frame["phase_pressure"] = "0"

        correction = pd.Series(0.0, index=X.index)
        for spec in self.corrections:
            keys = spec["keys"]
            mask = pd.Series(True, index=X.index)
            for key in keys:
                mask &= frame[key].eq(str(spec["values"][key]))
            correction.loc[mask] = float(spec["correction"])

        return np.clip(frame["prediction"].to_numpy() + correction.to_numpy(), self.clip_low, self.clip_high)


class StarterSegmentMinutesRegressor(RegressorMixin, BaseEstimator):
    """Use starter/bench-specific minutes models with a global fallback."""

    def __init__(self, global_model, segment_models: dict, clip_low: float = 0.0, clip_high: float = 45.0):
        self.global_model = global_model
        self.segment_models = segment_models
        self.clip_low = clip_low
        self.clip_high = clip_high

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.asarray(self.global_model.predict(X), dtype=float)
        if "is_starter" not in X.columns:
            return np.clip(pred, self.clip_low, self.clip_high)

        starter = X["is_starter"].fillna(0).astype(int)
        for key, model in self.segment_models.items():
            target = 1 if str(key) in {"1", "S", "Starter"} else 0
            mask = starter.eq(target).to_numpy()
            if mask.any():
                pred[mask] = np.asarray(model.predict(X.loc[mask]), dtype=float)
        return np.clip(pred, self.clip_low, self.clip_high)


class StackedRegressor(RegressorMixin, BaseEstimator):
    """Stack fitted base regressors through a fitted meta model."""

    def __init__(self, models: dict, meta_model):
        self.models = models
        self.meta_model = meta_model

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        meta_X = np.column_stack([model.predict(X) for model in self.models.values()])
        return self.meta_model.predict(meta_X)
