"""Baseline features aligned with hierarchical beta-GLMM cross-validation.

Fixed effects use the same train-only standardized environmental covariates as the
beta model (``CV_PREDICTORS``). Geographic structure is approximated with a
fold-safe hierarchical site intercept encoder on the logit scale, with fallbacks
site → region → global mean (mirroring beta posterior prediction logic).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from src.dataloading.build_model_ready_data import to_hbb_frame
from src.models.hbb._config import CV_PREDICTORS
from src.models.hbb.data import standardize_train_test
from src.models.hbb.indices import make_dense_site_region

DIV_COL = "diversity.standardized"
_LOGIT_EPS = 1e-4


def _diversity_col(df: pd.DataFrame) -> str:
    return DIV_COL if DIV_COL in df.columns else "diversity"


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _LOGIT_EPS, 1.0 - _LOGIT_EPS)
    return np.log(p / (1.0 - p))


def attach_test_dense_indices(
    test_df: pd.DataFrame,
    *,
    site_dense_map: dict[str, int],
    region_dense_map: dict[str, int],
) -> pd.DataFrame:
    """Map test rows to train-fold dense site/region indices (NaN when unseen)."""
    out = test_df.copy()
    out["site_dense"] = out["site"].astype(str).map(site_dense_map)
    out["region_dense"] = out["region"].astype(str).map(region_dense_map)
    return out


def prepare_baseline_fold_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Per-fold frames for baseline pipelines (matches beta CV preprocessing).

    Returns train/test DataFrames with standardized env covariates plus dense
    site/region indices for the geography encoder.
    """
    train_h = to_hbb_frame(train_df.reset_index(drop=True))
    test_h = to_hbb_frame(test_df.reset_index(drop=True))
    std = standardize_train_test(train_h, test_h)
    tr_raw, te_raw = std["train"], std["test"]

    dense = make_dense_site_region(tr_raw)
    tr = dense["data"]
    te = attach_test_dense_indices(
        te_raw,
        site_dense_map=dense["site_dense_map"],
        region_dense_map=dense["region_dense_map"],
    )

    env_cols = [p for p in CV_PREDICTORS if p in tr.columns]
    if len(env_cols) != len(CV_PREDICTORS):
        missing = sorted(set(CV_PREDICTORS) - set(env_cols))
        raise ValueError(f"Missing standardized predictors after fold prep: {missing}")

    div = _diversity_col(tr)
    pipe_cols = env_cols + [div, "site_dense", "region_dense"]
    return tr[pipe_cols].copy(), te[pipe_cols].copy(), env_cols


def baseline_feature_spec() -> dict[str, Any]:
    """Document how baseline features align with the beta model."""
    return {
        "fixed_effects": list(CV_PREDICTORS),
        "geography": "hierarchical_site_logit_intercept_zscored",
        "geography_fallback": ["site", "region", "diversity", "global"],
        "standardization": "train_only_standardize_train_test",
        "notes": (
            "Approximates beta site/region random intercepts with fold-safe "
            "smoothed training means on the logit scale; refit inside each CV fold."
        ),
    }


class HierarchicalGeographyEncoder(BaseEstimator, TransformerMixin):
    """
    Add a hierarchical site intercept feature on the logit scale.

    Known training sites receive a smoothed logit-mean offset (shrunk toward the
    training region mean). Unseen sites fall back to region then global means,
    matching the beta model's site → region → diversity/global hierarchy.
    """

    def __init__(self, *, smoothing: float = 20.0):
        self.smoothing = smoothing

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("HierarchicalGeographyEncoder requires target y at fit time.")
        X = self._as_frame(X)
        y = np.asarray(y, dtype=float)
        y_logit = _logit(y)

        self.div_col_ = _diversity_col(X)
        self.env_columns_ = [
            c
            for c in X.columns
            if c not in {self.div_col_, "site_dense", "region_dense"}
        ]
        self.global_mean_ = float(np.mean(y_logit))
        self.diversity_fallback_slope_ = 0.0
        self.diversity_fallback_intercept_ = self.global_mean_
        if self.div_col_ in X.columns:
            div_vals = pd.to_numeric(X[self.div_col_], errors="coerce").to_numpy()
            mask = np.isfinite(div_vals)
            if int(mask.sum()) >= 2:
                slope, intercept = np.polyfit(div_vals[mask], y_logit[mask], 1)
                self.diversity_fallback_slope_ = float(slope)
                self.diversity_fallback_intercept_ = float(intercept)

        region_series = pd.to_numeric(X["region_dense"], errors="coerce")
        self.region_means_: dict[int, float] = {}
        for region in sorted(region_series.dropna().unique()):
            mask = region_series == region
            self.region_means_[int(region)] = float(np.mean(y_logit[mask.to_numpy()]))

        site_series = pd.to_numeric(X["site_dense"], errors="coerce")
        self.site_offsets_: dict[int, float] = {}
        m = float(self.smoothing)
        for site in sorted(site_series.dropna().unique()):
            site = int(site)
            mask = site_series == site
            site_mean = float(np.mean(y_logit[mask.to_numpy()]))
            n_s = int(mask.sum())
            region_val = region_series.loc[mask].iloc[0]
            if pd.isna(region_val):
                region_mean = self.global_mean_
            else:
                region_mean = self.region_means_.get(int(region_val), self.global_mean_)
            smoothed = (n_s * site_mean + m * region_mean) / (n_s + m)
            self.site_offsets_[site] = float(smoothed)

        train_offsets = self._encode_offsets(X)
        self.site_hier_mean_ = float(np.mean(train_offsets))
        std = float(np.std(train_offsets))
        self.site_hier_std_ = std if std > 1e-8 else 1.0
        self.feature_names_out_ = [*self.env_columns_, "site_hier_logit_stzd"]
        return self

    def _encode_offsets(self, X: pd.DataFrame) -> np.ndarray:
        offsets = np.empty(len(X), dtype=float)
        site_series = pd.to_numeric(X["site_dense"], errors="coerce")
        region_series = pd.to_numeric(X["region_dense"], errors="coerce")
        div_series = (
            pd.to_numeric(X[self.div_col_], errors="coerce")
            if self.div_col_ in X.columns
            else pd.Series(np.nan, index=X.index)
        )

        for i in range(len(X)):
            site = site_series.iloc[i]
            region = region_series.iloc[i]
            if pd.notna(site) and int(site) in self.site_offsets_:
                offsets[i] = self.site_offsets_[int(site)]
            elif pd.notna(region) and int(region) in self.region_means_:
                offsets[i] = self.region_means_[int(region)]
            elif pd.notna(div_series.iloc[i]):
                d = float(div_series.iloc[i])
                offsets[i] = (
                    self.diversity_fallback_intercept_
                    + self.diversity_fallback_slope_ * d
                )
            else:
                offsets[i] = self.global_mean_
        return offsets

    def transform(self, X):
        X = self._as_frame(X)
        env = X[self.env_columns_].to_numpy(dtype=float)
        offsets = self._encode_offsets(X)
        offsets = (offsets - self.site_hier_mean_) / self.site_hier_std_
        return np.column_stack([env, offsets])

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)

    @staticmethod
    def _as_frame(X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("HierarchicalGeographyEncoder expects a pandas DataFrame input.")


def make_baseline_pipeline(model, *, smoothing: float = 20.0) -> Pipeline:
    """Pipeline: beta-aligned features + estimator (no extra scaling step)."""
    return Pipeline(
        [
            ("features", HierarchicalGeographyEncoder(smoothing=smoothing)),
            ("model", model),
        ]
    )
