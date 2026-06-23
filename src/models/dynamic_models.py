"""Persistence-adjusted dynamic models for coral cover projection."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge

from src.models.baseline_features import HierarchicalGeographyEncoder
from src.models.baseline_models import make_baseline_estimator
from src.models.dynamic_features import DYNAMIC_INPUT_COLS

DynamicModelName = Literal[
    "persist_residual_linear",
    "persist_residual_ridge",
    "persist_residual_xgboost",
    "persist_ar_env",
]

DYNAMIC_MODEL_NAMES: tuple[DynamicModelName, ...] = (
    "persist_residual_linear",
    "persist_residual_ridge",
    "persist_residual_xgboost",
    "persist_ar_env",
)

DYNAMIC_DISPLAY_NAMES: dict[str, str] = {
    "persist_residual_linear": "Persist + linear residual",
    "persist_residual_ridge": "Persist + ridge residual",
    "persist_residual_xgboost": "Persist + XGBoost residual",
    "persist_ar_env": "Persist + AR + env",
    "dynamic_projection": "Dynamic projection",
}


def _clip_unit(y: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(y, dtype=float), 0.0, 1.0)


def _fit_residual_estimator(
    name: DynamicModelName,
    *,
    random_state: int,
    n_jobs: int = -1,
):
    if name == "persist_residual_linear":
        return LinearRegression()
    if name == "persist_residual_ridge":
        return Ridge(alpha=1.0)
    if name == "persist_residual_xgboost":
        return make_baseline_estimator("xgboost", random_state=random_state, n_jobs=n_jobs)
    raise ValueError(f"Not a residual estimator model: {name}")


def _fit_ar_phi(
    y: np.ndarray,
    y_persist: np.ndarray,
    y_lag: np.ndarray,
    has_prior: np.ndarray,
) -> float:
    """Estimate AR weight on lag deviation from persistence (clipped to [0, 1])."""
    mask = has_prior & np.isfinite(y_lag) & np.isfinite(y_persist) & np.isfinite(y)
    if int(mask.sum()) < 5:
        return 0.5
    resid = y[mask] - y_persist[mask]
    lag_dev = y_lag[mask] - y_persist[mask]
    denom = float(np.dot(lag_dev, lag_dev))
    if denom < 1e-12:
        return 0.5
    phi = float(np.dot(resid, lag_dev) / denom)
    return float(np.clip(phi, 0.0, 1.0))


class PersistenceDynamicModel:
    """Persistence baseline with optional AR blending and env residual correction."""

    def __init__(
        self,
        name: DynamicModelName,
        *,
        random_state: int = 42,
        n_jobs: int = -1,
        use_geo_encoder: bool = True,
        smoothing: float = 20.0,
    ):
        self.name = name
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.use_geo_encoder = use_geo_encoder
        self.smoothing = smoothing
        self.phi_: float = 0.5
        self.residual_model_: Any = None
        self._geo_encoder_: HierarchicalGeographyEncoder | None = None
        self._imputer_: SimpleImputer | None = None
        self.feature_columns_: list[str] = []

    def _transform_features(self, X: pd.DataFrame, *, y: np.ndarray | None = None) -> np.ndarray:
        X = X[self.feature_columns_]
        dynamic = X[list(DYNAMIC_INPUT_COLS)].to_numpy(dtype=float)
        if self._geo_encoder_ is not None:
            if y is not None:
                self._geo_encoder_.fit(X, y)
            geo = self._geo_encoder_.transform(X)
            design = np.column_stack([geo, dynamic])
        else:
            base_cols = [c for c in X.columns if c not in DYNAMIC_INPUT_COLS]
            design = np.column_stack([X[base_cols].to_numpy(dtype=float), dynamic])
        if self._imputer_ is not None:
            if y is not None:
                return self._imputer_.fit_transform(design)
            return self._imputer_.transform(design)
        return design

    def fit(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        *,
        y_persist: np.ndarray,
        y_lag: np.ndarray,
        has_prior: np.ndarray,
    ) -> PersistenceDynamicModel:
        y = np.asarray(y, dtype=float)
        y_persist = np.asarray(y_persist, dtype=float)
        y_lag = np.asarray(y_lag, dtype=float)
        has_prior = np.asarray(has_prior, dtype=bool)
        self.feature_columns_ = list(X.columns)
        self._imputer_ = SimpleImputer(strategy="median")

        if self.use_geo_encoder:
            self._geo_encoder_ = HierarchicalGeographyEncoder(smoothing=self.smoothing)
        else:
            self._geo_encoder_ = None

        if self.name == "persist_ar_env":
            self.phi_ = _fit_ar_phi(y, y_persist, y_lag, has_prior)
            base = y_persist.copy()
            lag_mask = has_prior & np.isfinite(y_lag)
            base[lag_mask] = (1.0 - self.phi_) * y_persist[lag_mask] + self.phi_ * y_lag[lag_mask]
            residual = y - base
            X_mat = self._transform_features(X, y=y)
            est = LinearRegression()
            est.fit(X_mat, residual)
            self.residual_model_ = est
            return self

        residual = y - y_persist
        X_mat = self._transform_features(X, y=y)
        est = _fit_residual_estimator(
            self.name,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        est.fit(X_mat, residual)
        self.residual_model_ = est
        return self

    def _predict_residual(self, X: pd.DataFrame) -> np.ndarray:
        if self.residual_model_ is None:
            return np.zeros(len(X), dtype=float)
        X_mat = self._transform_features(X)
        return np.asarray(self.residual_model_.predict(X_mat), dtype=float)

    def predict(
        self,
        X: pd.DataFrame,
        *,
        y_persist: np.ndarray,
        y_lag: np.ndarray | None = None,
        has_prior: np.ndarray | None = None,
    ) -> np.ndarray:
        y_persist = np.asarray(y_persist, dtype=float)
        X = X[self.feature_columns_]

        if self.name == "persist_ar_env":
            base = y_persist.copy()
            if y_lag is not None and has_prior is not None:
                y_lag = np.asarray(y_lag, dtype=float)
                has_prior = np.asarray(has_prior, dtype=bool)
                lag_mask = has_prior & np.isfinite(y_lag)
                base[lag_mask] = (
                    (1.0 - self.phi_) * y_persist[lag_mask] + self.phi_ * y_lag[lag_mask]
                )
            return _clip_unit(base + self._predict_residual(X))

        return _clip_unit(y_persist + self._predict_residual(X))

    def fit_params(self) -> dict[str, Any]:
        return {
            "model": self.name,
            "phi": self.phi_,
            "use_geo_encoder": self.use_geo_encoder,
        }


def make_dynamic_model(
    name: str,
    *,
    random_state: int = 42,
    n_jobs: int = -1,
) -> PersistenceDynamicModel:
    if name not in DYNAMIC_MODEL_NAMES:
        raise ValueError(
            f"Unknown dynamic model '{name}'. Expected one of: {DYNAMIC_MODEL_NAMES}"
        )
    return PersistenceDynamicModel(
        name,  # type: ignore[arg-type]
        random_state=random_state,
        n_jobs=n_jobs,
    )
