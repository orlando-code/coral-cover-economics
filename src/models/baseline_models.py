"""Baseline regressors for coral cover from environmental covariates.

These models are used in cross-validation with preprocessing (e.g. scaling)
handled by the training / evaluation script.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

BaselineName = Literal[
    "linear", "random_forest", "xgboost", "neural_network", "survey_mean"
]

BASELINE_MODEL_NAMES: tuple[BaselineName, ...] = (
    "linear",
    "random_forest",
    "xgboost",
    "neural_network",
    "survey_mean",
)

PERSISTENCE_BASELINE_NAMES: frozenset[str] = frozenset({"survey_mean"})


def is_persistence_baseline(name: str) -> bool:
    return name in PERSISTENCE_BASELINE_NAMES


DISPLAY_NAMES: dict[BaselineName, str] = {
    "linear": "Linear",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural network",
    "survey_mean": "Prior survey mean",
}

# String keys avoid skopt failures on tuple-valued categoricals.
HIDDEN_LAYER_KEYS: tuple[str, ...] = (
    "64",
    "64-32",
    "128-64",
    "128-64-32",
    "256-128",
    "256-128-64",
)


def _clip_unit_interval(y: np.ndarray) -> np.ndarray:
    return np.clip(y, 0.0, 1.0)


def _parse_hidden_layers(key: str) -> tuple[int, ...]:
    return tuple(int(part) for part in str(key).split("-"))


def _as_int(value: float, *, low: int = 1, high: int | None = None) -> int:
    out = int(round(float(value)))
    out = max(low, out)
    if high is not None:
        out = min(high, out)
    return out


def _parse_max_features(value: str | float) -> str | float:
    if isinstance(value, str):
        if value in {"sqrt", "log2"}:
            return value
        return float(value)
    return float(value)


def _parse_batch_size_key(key: str) -> int | str:
    return "auto" if str(key) == "auto" else int(key)


class TunedRandomForestRegressor(BaseEstimator, RegressorMixin):
    """RF wrapper accepting continuous skopt draws; rounds to valid integers."""

    def __init__(
        self,
        *,
        n_estimators: float = 500.0,
        max_depth: float = 20.0,
        min_samples_leaf: float = 3.0,
        min_samples_split: float = 4.0,
        max_features: str | float = "sqrt",
        n_jobs: int = 16,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        self.estimator_ = RandomForestRegressor(
            n_estimators=_as_int(self.n_estimators, low=50, high=1200),
            max_depth=_as_int(self.max_depth, low=2, high=80),
            min_samples_leaf=_as_int(self.min_samples_leaf, low=1, high=30),
            min_samples_split=_as_int(self.min_samples_split, low=2, high=40),
            max_features=_parse_max_features(self.max_features),
            n_jobs=int(self.n_jobs),
            random_state=int(self.random_state),
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class TunedXGBRegressor(BaseEstimator, RegressorMixin):
    """XGB wrapper accepting continuous skopt draws; rounds discrete params."""

    def __init__(
        self,
        *,
        n_estimators: float = 600.0,
        max_depth: float = 6.0,
        learning_rate: float = 0.05,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        min_child_weight: float = 3.0,
        reg_lambda: float = 1.0,
        reg_alpha: float = 0.0,
        n_jobs: int = 16,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_lambda = reg_lambda
        self.reg_alpha = reg_alpha
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X, y):
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for the XGBoost baseline. Install with: pip install xgboost"
            )
        self.estimator_ = XGBRegressor(
            n_estimators=_as_int(self.n_estimators, low=50, high=1200),
            max_depth=_as_int(self.max_depth, low=2, high=16),
            learning_rate=float(self.learning_rate),
            subsample=float(self.subsample),
            colsample_bytree=float(self.colsample_bytree),
            min_child_weight=_as_int(self.min_child_weight, low=1, high=20),
            reg_lambda=float(self.reg_lambda),
            reg_alpha=float(self.reg_alpha),
            objective="reg:squarederror",
            random_state=int(self.random_state),
            n_jobs=int(self.n_jobs),
        )
        self.estimator_.fit(X, y)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class TunedMLPRegressor(BaseEstimator, RegressorMixin):
    """MLP wrapper exposing architecture as a string key for hyperparameter search."""

    def __init__(
        self,
        *,
        hidden_layers_key: str = "128-64",
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        learning_rate: str = "adaptive",
        max_iter: int = 2000,
        batch_size_key: str = "auto",
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 30,
        random_state: int = 42,
    ):
        self.hidden_layers_key = hidden_layers_key
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.batch_size_key = batch_size_key
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state

    def fit(self, X, y):
        if self.hidden_layers_key not in HIDDEN_LAYER_KEYS:
            raise ValueError(
                f"Unknown hidden_layers_key={self.hidden_layers_key!r}; "
                f"expected one of {HIDDEN_LAYER_KEYS}"
            )
        self.estimator_ = MLPRegressor(
            hidden_layer_sizes=_parse_hidden_layers(self.hidden_layers_key),
            activation=self.activation,
            solver="adam",
            alpha=float(self.alpha),
            learning_rate_init=float(self.learning_rate_init),
            learning_rate=self.learning_rate,
            max_iter=int(self.max_iter),
            batch_size=_parse_batch_size_key(self.batch_size_key),
            early_stopping=bool(self.early_stopping),
            validation_fraction=float(self.validation_fraction),
            n_iter_no_change=int(self.n_iter_no_change),
            random_state=int(self.random_state),
        )
        self.estimator_.fit(X, y)
        self.n_iter_ = int(self.estimator_.n_iter_)
        self.loss_ = float(self.estimator_.loss_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


def make_baseline_estimator(
    name: BaselineName, *, n_jobs: int = 16, random_state: int = 42
) -> Any:
    """Return an unfitted sklearn-compatible estimator."""
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return TunedRandomForestRegressor(
            n_estimators=500.0,
            max_depth=20.0,
            min_samples_leaf=3.0,
            min_samples_split=4.0,
            max_features="sqrt",
            n_jobs=n_jobs,
            random_state=random_state,
        )
    if name == "xgboost":
        if not HAS_XGBOOST:
            raise ImportError(
                "xgboost is required for the XGBoost baseline. Install with: pip install xgboost"
            )
        return TunedXGBRegressor(
            n_estimators=600.0,
            max_depth=6.0,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3.0,
            reg_lambda=1.0,
            random_state=random_state,
            n_jobs=n_jobs,
        )
    if name == "neural_network":
        return TunedMLPRegressor(
            hidden_layers_key="128-64",
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            learning_rate="adaptive",
            max_iter=2000,
            batch_size_key="auto",
            early_stopping=False,
            random_state=random_state,
        )
    if name == "survey_mean":
        raise ValueError(
            "survey_mean is a persistence baseline; use predict_survey_mean_baseline() "
            "or run via cross-validation with model name 'survey_mean'."
        )
    raise ValueError(f"Unknown baseline model: {name}")


def baseline_param_grid(name: BaselineName) -> dict[str, list[Any]]:
    """Hyperparameter grid for random search within CV.

    Returned keys match the *bare estimator* returned by :func:`make_baseline_estimator`.
    When used inside a sklearn ``Pipeline`` step named ``model``, prefix keys with
    ``model__``.
    """
    if name == "linear":
        return {}
    if name == "random_forest":
        return {
            "n_estimators": [300, 500, 700, 900],
            "max_depth": [8.0, 12.0, 20.0, 30.0, 50.0],
            "min_samples_leaf": [1, 2, 5, 10],
            "min_samples_split": [2, 4, 8, 16],
            "max_features": ["sqrt", "log2", "0.3", "0.5", "0.7", "1.0"],
        }
    if name == "xgboost":
        return {
            "n_estimators": [300, 500, 700, 900],
            "max_depth": [3, 4, 6, 8, 10],
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.8, 1.0],
            "min_child_weight": [1, 3, 5, 10],
            "reg_lambda": [0.1, 1.0, 5.0, 10.0],
        }
    if name == "neural_network":
        return {
            "hidden_layers_key": list(HIDDEN_LAYER_KEYS),
            "activation": ["relu", "tanh"],
            "alpha": [1e-5, 1e-4, 1e-3, 1e-2],
            "learning_rate_init": [1e-4, 3e-4, 1e-3, 3e-3],
            "batch_size_key": ["64", "128", "256", "auto"],
        }
    if name == "survey_mean":
        return {}
    raise ValueError(f"Unknown baseline model: {name}")


def fit_baseline_model(
    name: BaselineName,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    random_state: int = 42,
) -> Any:
    """Fit a baseline model and return the fitted estimator."""
    model = make_baseline_estimator(name, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def predict_coral_cover(model: Any, X: np.ndarray) -> np.ndarray:
    """Predict coral cover (proportion), clipped to [0, 1]."""
    return _clip_unit_interval(np.asarray(model.predict(X), dtype=float))
