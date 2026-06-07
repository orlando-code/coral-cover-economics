"""Hyperparameter tuning for baseline models (Bayesian optimization via skopt)."""

from __future__ import annotations

import time
from typing import Any, Literal

import numpy as np

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Real

    HAS_SKOPT = True
except ImportError:
    BayesSearchCV = None  # type: ignore[misc, assignment]
    Categorical = Real = None  # type: ignore[misc, assignment]
    HAS_SKOPT = False

from sklearn.model_selection import RandomizedSearchCV

from src.models.baseline_models import (
    HIDDEN_LAYER_KEYS,
    BaselineName,
    baseline_param_grid,
)

TuningMethod = Literal["bayes", "random"]

# String-only categoricals avoid numpy scalar duplicates in skopt's point cache.
_RF_MAX_FEATURES = ("sqrt", "log2", "0.25", "0.4", "0.6", "0.8", "1.0")
_NN_BATCH_SIZE_KEYS = ("64", "128", "256", "auto")


def _prefixed(space: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in space.items()}


def _bayes_optimizer_kwargs(n_iter: int, seed: int) -> dict[str, Any]:
    """Optimizer settings that reduce duplicate-point collisions."""
    return {
        "random_state": int(seed),
        # Explore randomly before GP starts collapsing to repeated corners.
        "n_initial_points": int(min(15, max(8, n_iter // 4))),
        "acq_func": "gp_hedge",
    }


def baseline_search_space(
    name: BaselineName, *, prefix: str = "model__"
) -> dict[str, Any]:
    """Continuous search space for Bayesian optimization.

    Numeric tree/NN hyperparameters use :class:`Real` (not :class:`Integer`)
    so the GP explores a smooth surface; estimators round to valid ints.
    Categoricals use native Python strings only.
    """
    if not HAS_SKOPT:
        return {}

    if name == "linear":
        return {}

    if name == "random_forest":
        space = {
            "n_estimators": Real(250.0, 1000.0, prior="uniform"),
            "max_depth": Real(6.0, 40.0, prior="uniform"),
            "min_samples_leaf": Real(1.0, 15.0, prior="uniform"),
            "min_samples_split": Real(2.0, 20.0, prior="uniform"),
            "max_features": Categorical(_RF_MAX_FEATURES),
        }
    elif name == "xgboost":
        space = {
            "n_estimators": Real(250.0, 1000.0, prior="uniform"),
            "max_depth": Real(3.0, 12.0, prior="uniform"),
            "learning_rate": Real(0.005, 0.2, prior="log-uniform"),
            "subsample": Real(0.6, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Real(1.0, 15.0, prior="uniform"),
            "reg_lambda": Real(1e-2, 20.0, prior="log-uniform"),
            "reg_alpha": Real(1e-3, 10.0, prior="log-uniform"),
        }
    elif name == "neural_network":
        space = {
            "hidden_layers_key": Categorical(HIDDEN_LAYER_KEYS),
            "activation": Categorical(("relu", "tanh")),
            "alpha": Real(1e-6, 1e-1, prior="log-uniform"),
            "learning_rate_init": Real(5e-5, 1e-2, prior="log-uniform"),
            "batch_size_key": Categorical(_NN_BATCH_SIZE_KEYS),
        }
    else:
        raise ValueError(f"Unknown baseline model: {name}")

    return _prefixed(space, prefix=prefix)


def _tuning_diagnostics(
    search: Any,
    *,
    method: str,
    n_iter_requested: int,
    elapsed_sec: float,
) -> dict[str, Any]:
    cv_results = getattr(search, "cv_results_", {})
    n_completed = len(cv_results.get("params", []))
    mean_scores = cv_results.get("mean_test_score", [])
    std_scores = cv_results.get("std_test_score", [])
    rank_scores = cv_results.get("rank_test_score", [])

    diagnostics: dict[str, Any] = {
        "tuning_method": method,
        "n_trials_requested": int(n_iter_requested),
        "n_trials_completed": int(n_completed),
        "best_cv_r2": float(search.best_score_),
        "elapsed_sec": round(float(elapsed_sec), 2),
    }
    if n_completed:
        diagnostics["mean_cv_r2"] = round(float(np.mean(mean_scores)), 4)
        diagnostics["std_cv_r2"] = round(float(np.std(mean_scores)), 4)
        diagnostics["worst_cv_r2"] = round(float(np.min(mean_scores)), 4)
        if len(std_scores):
            diagnostics["mean_inner_cv_std"] = round(float(np.mean(std_scores)), 4)
        if len(rank_scores):
            diagnostics["best_trial_rank"] = int(np.min(rank_scores))
    return diagnostics


def tune_baseline_estimator(
    estimator: Any,
    name: BaselineName,
    *,
    X_train,
    y_train,
    groups_train,
    inner_cv,
    n_iter: int,
    n_jobs: int,
    seed: int,
    method: TuningMethod = "bayes",
    verbose: int = 0,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Inner-CV hyperparameter search; returns refitted best estimator + diagnostics."""
    n_iter = max(1, int(n_iter))
    grid = baseline_param_grid(name)

    if not grid:
        t0 = time.perf_counter()
        fitted = estimator.fit(X_train, y_train)
        return fitted, {}, {
            "tuning_method": "none",
            "n_trials_requested": 0,
            "n_trials_completed": 0,
            "best_cv_r2": np.nan,
            "elapsed_sec": round(time.perf_counter() - t0, 2),
        }

    use_bayes = method == "bayes" and HAS_SKOPT
    if not use_bayes:
        max_iter = int(np.prod([len(v) for v in grid.values()]))
        n_iter = min(n_iter, max_iter)
    n_iter = int(n_iter)

    if method == "bayes" and not HAS_SKOPT:
        import warnings

        warnings.warn(
            "scikit-optimize not installed; falling back to RandomizedSearchCV. "
            "Install with: pip install scikit-optimize",
            stacklevel=2,
        )

    if use_bayes:
        search = BayesSearchCV(
            estimator,
            search_spaces=baseline_search_space(name),
            n_iter=n_iter,
            n_points=1,
            cv=inner_cv,
            n_jobs=n_jobs,
            scoring="r2",
            random_state=int(seed),
            optimizer_kwargs=_bayes_optimizer_kwargs(n_iter, seed),
            refit=True,
            verbose=verbose,
        )
        method_label = "bayes"
    else:
        search = RandomizedSearchCV(
            estimator,
            param_distributions=_prefixed(grid, prefix="model__"),
            n_iter=n_iter,
            scoring="r2",
            cv=inner_cv,
            n_jobs=n_jobs,
            refit=True,
            random_state=seed,
            verbose=verbose,
        )
        method_label = "random"

    t0 = time.perf_counter()
    search.fit(X_train, y_train, groups=groups_train)
    elapsed = time.perf_counter() - t0

    diagnostics = _tuning_diagnostics(
        search,
        method=method_label,
        n_iter_requested=n_iter,
        elapsed_sec=elapsed,
    )
    diagnostics["inner_cv_splits"] = int(getattr(inner_cv, "n_splits", 0) or 0)

    n_completed = diagnostics["n_trials_completed"]
    if n_completed < n_iter:
        import warnings

        warnings.warn(
            f"{name}: completed {n_completed}/{n_iter} tuning trials "
            f"(method={method_label}).",
            stacklevel=2,
        )

    return search.best_estimator_, dict(search.best_params_), diagnostics
