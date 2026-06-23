"""PyMC hierarchical beta regression model."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Optional

from src.models.hbb._config import ModelSpec

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.special import expit as inv_logit

from src.models.hbb._config import HAS_PYMC
from src.plots.hb_beta_plots import (
    plot_coefficient_traces_and_posteriors,
    save_trace_diagnostics,
)

if HAS_PYMC:
    import arviz as az
    import pymc as pm

_PI_EPS = 1e-6


def _sample_with_parallel_fallback(sample_kw: dict[str, Any]) -> Any:
    """Run ``pm.sample`` with spawn-based parallelism and serial fallback."""
    try:
        return pm.sample(**sample_kw)
    except EOFError:
        if int(sample_kw.get("cores", 1)) <= 1:
            raise
        warnings.warn(
            "Parallel PyMC sampling failed with EOFError; retrying with cores=1.",
            stacklevel=3,
        )
        serial_kw = dict(sample_kw)
        serial_kw["cores"] = 1
        serial_kw.pop("mp_ctx", None)
        return pm.sample(**serial_kw)


def resolve_pymc_ncores(*, ncores: int | None, n_chains: int) -> int:
    """Resolve PyMC worker count (capped by chain count)."""
    import os

    if ncores is None:
        return max(1, min(int(n_chains), os.cpu_count() or 1))
    return max(1, min(int(ncores), int(n_chains)))


def _parallel_sample_kwargs(
    sample_kw: dict[str, Any],
    *,
    ncores: int,
    n_chains: int,
    mp_ctx: str | None = None,
) -> dict[str, Any]:
    """Attach PyMC core count and a safe multiprocessing context."""
    cores = max(1, min(int(ncores), int(n_chains)))
    out = dict(sample_kw)
    out["cores"] = cores
    if cores > 1:
        # fork is unreliable with PyMC on recent Python/macOS (EOFError in workers).
        out["mp_ctx"] = mp_ctx or "spawn"
        out["blas_cores"] = 1
    return out


class HierarchicalBetaModel:
    """
    Hierarchical Bayesian Beta Regression Model for Coral Cover.

    This model implements a beta regression with:
    - Fixed effects for environmental predictors
    - Random intercepts for sites nested within ecoregions
    - Ecoregion-level random effects with diversity as a predictor

    Attributes
    ----------
    trace : arviz.InferenceData
        MCMC trace from model fitting
    model : pymc.Model
        PyMC model object
    summary : pd.DataFrame
        Summary statistics for model parameters
    """

    def __init__(self):
        self.trace = None
        self.model = None
        self.summary = None
        self.X = None
        self.y = None
        self.site_idx = None
        self.region_idx = None
        self.col_names = None
        self.n_samples = None
        self.n_tune = None
        self.n_chains = None
        self.target_accept = None
        self.max_treedepth = None
        self.random_seed = None
        # Mapping from reef_id to site index - critical for consistent predictions
        self.reef_to_site_map = None
        self.n_observations = None  # Store for inverse_transform_beta

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        site_idx: np.ndarray,
        region_idx: np.ndarray,
        site_to_region: np.ndarray,
        reef_to_site_map: Optional[dict] = None,
        diversity: Optional[np.ndarray] = None,
        col_names: Optional[list[str]] = None,
        n_samples: int = 2000,
        n_tune: int = 1000,
        n_chains: int = 6,
        target_accept: float = 0.95,
        cores: int = 6,
        max_treedepth: int = 15,
        random_seed: int = 42,
        spec: ModelSpec = "reparam",
        ncores: int = 6,
        progressbar: bool = True,
        mp_ctx: str | None = None,
        use_site_hierarchy: bool | None = None,
        use_ecoregion_hierarchy: bool | None = None,
        use_hierarchy: bool | None = None,
        use_diversity: bool = True,
    ) -> "HierarchicalBetaModel":
        """
        Fit the hierarchical beta regression model.

        Args:
            X (np.ndarray): Design matrix (n_obs x n_predictors)
            y (np.ndarray): Response variable (coral cover, transformed to (0,1))
            site_idx (np.ndarray): Site index for each observation
            region_idx (np.ndarray): Region index for each observation
            site_to_region (np.ndarray): Mapping from site index to region index
            reef_to_site_map (dict, optional): Mapping from reef_id to site index. CRITICAL for consistent predictions.
            diversity (np.ndarray, optional): Standardized diversity values for each region
            col_names (list, optional): Names of columns in X
            n_samples (int): Number of posterior samples per chain
            n_tune (int): Number of tuning samples
            n_chains (int): Number of MCMC chains
            target_accept (float): Target acceptance rate for NUTS sampler
            max_treedepth (int): Maximum tree depth for NUTS sampler
            random_seed (int): Random seed for reproducibility
            spec (str): ``reparam`` (non-centered, no intercept in X),
                ``centered`` (centered hierarchy, intercept optional), or
                ``legacy_r`` (centered JAGS like ``my_1_run_the_beta_model.Rmd``;
                requires ``Intercept`` in ``X``).
            ncores (int): Number of cores to use for sampling
            progressbar (bool): Show PyMC sampling progress bar
            mp_ctx (str, optional): Multiprocessing start method when ``ncores > 1``
                (default: ``spawn``, avoids EOFError on macOS / Python 3.12+).

        Returns:
            self: HierarchicalBetaModel
        """
        if not HAS_PYMC:
            raise ImportError(
                "\tPyMC is required for model fitting. Install with: pip install pymc"
            )

        self.X = X
        self.y = y
        self.site_idx = site_idx
        self.region_idx = region_idx
        self.site_to_region = site_to_region
        self.reef_to_site_map = reef_to_site_map
        self.n_observations = len(y)  # Store for inverse_transform_beta
        self.diversity = diversity
        self.col_names = col_names
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.target_accept = target_accept
        self.max_treedepth = max_treedepth
        self.random_seed = random_seed
        self.mp_ctx = mp_ctx
        if use_site_hierarchy is None and use_ecoregion_hierarchy is None:
            if use_hierarchy is None:
                use_site_hierarchy = True
                use_ecoregion_hierarchy = True
            else:
                use_site_hierarchy = use_hierarchy
                use_ecoregion_hierarchy = use_hierarchy
        else:
            use_site_hierarchy = (
                True if use_site_hierarchy is None else use_site_hierarchy
            )
            use_ecoregion_hierarchy = (
                True if use_ecoregion_hierarchy is None else use_ecoregion_hierarchy
            )
        self.use_site_hierarchy = use_site_hierarchy
        self.use_ecoregion_hierarchy = use_ecoregion_hierarchy
        self.use_hierarchy = use_site_hierarchy or use_ecoregion_hierarchy
        self.use_diversity = use_diversity and use_ecoregion_hierarchy

        n_obs, n_predictors = X.shape
        n_sites = len(np.unique(site_idx))
        n_regions = len(np.unique(region_idx))

        if diversity is None or not self.use_diversity:
            diversity = np.zeros(n_regions)
        if spec == "legacy_r" and not (use_site_hierarchy and use_ecoregion_hierarchy):
            raise ValueError(
                "spec='legacy_r' requires use_site_hierarchy=True and "
                "use_ecoregion_hierarchy=True (JAGS site/ecoregion structure)."
            )
        if spec == "legacy_r" and (col_names is None or col_names[0] != "Intercept"):
            raise ValueError(
                "spec='legacy_r' requires add_intercept=True (R model.matrix default)."
            )

        self.model_spec = spec
        self.trace = self._sample_pymc_model(
            X,
            n_predictors,
            diversity,
            n_regions,
            n_sites,
            site_to_region,
            spec,
            ncores,
            progressbar=progressbar,
            mp_ctx=mp_ctx,
            use_site_hierarchy=use_site_hierarchy,
            use_ecoregion_hierarchy=use_ecoregion_hierarchy,
            use_diversity=self.use_diversity,
        )
        self.summary = az.summary(self.trace)
        return self

    @staticmethod
    def _legacy_initvals(
        n_predictors: int,
        n_regions: int,
        n_sites: int,
        n_chains: int,
        seed: int,
        *,
        use_diversity: bool = True,
        use_site_hierarchy: bool = True,
        use_ecoregion_hierarchy: bool = True,
    ) -> list[dict[str, Any]]:
        """Match the centered JAGS initial values in ``1_run_the_beta_model.Rmd``."""
        inits: list[dict[str, Any]] = []
        for c in range(n_chains):
            rng = np.random.default_rng(seed + c * 1000)

            init: dict[str, Any] = {
                "beta": rng.normal(0, 0.1, n_predictors).astype(float),
                "mu_global": float(rng.normal(0, 0.1)),
                "sigma_ecoregion_num": float(rng.normal(0, 25)),
                "sigma_ecoregion_denom": float(rng.normal(0, 1)),
                "sigma_num": float(rng.normal(0, 25)),
                "sigma_denom": float(rng.normal(0, 1)),
                "theta_num": float(rng.normal(0, 25)),
                "theta_denom": float(rng.normal(0, 1)),
            }
            if use_ecoregion_hierarchy:
                init["ecoregion"] = rng.normal(0, 0.1, n_regions).astype(float)
            if use_site_hierarchy:
                init["site_effect"] = rng.normal(0, 0.1, n_sites).astype(float)
            if use_diversity:
                init["beta_diversity"] = float(rng.normal(0, 0.1))
            inits.append(init)
        return inits

    def _sample_pymc_model(
        self,
        X: np.ndarray,
        n_predictors: int,
        diversity: np.ndarray,
        n_regions: int,
        n_sites: int,
        site_to_region: np.ndarray,
        spec: ModelSpec,
        ncores: int,
        *,
        progressbar: bool = True,
        mp_ctx: str | None = None,
        use_site_hierarchy: bool = True,
        use_ecoregion_hierarchy: bool = True,
        use_diversity: bool = True,
    ):
        """PyMC graph: ``reparam`` (default) or centered ``legacy_r`` (my_1 JAGS)."""
        use_hierarchy = use_site_hierarchy or use_ecoregion_hierarchy
        with pm.Model() as self.model:
            beta = pm.Normal("beta", mu=0, sigma=100, shape=n_predictors)

            if not use_hierarchy:
                theta = pm.HalfCauchy("theta", beta=25)
                eta = pm.math.dot(X, beta)
                fast_sample = True
            else:
                mu_global = pm.Normal("mu_global", mu=0, sigma=100)
                if use_ecoregion_hierarchy and use_diversity:
                    beta_diversity = pm.Normal("beta_diversity", mu=0, sigma=100)
                    g = mu_global + beta_diversity * diversity
                else:
                    g = mu_global

                def _bugs_halfcauchy(name: str):
                    num = pm.Normal(f"{name}_num", mu=0, sigma=25)
                    denom = pm.Normal(f"{name}_denom", mu=0, sigma=1)
                    return pm.Deterministic(name, pm.math.abs(num / denom))

                ecoregion_effect = None
                if use_ecoregion_hierarchy:
                    if spec in {"legacy_r", "centered"}:
                        ecoregion_effect = pm.Normal(
                            "ecoregion",
                            mu=g,
                            sigma=_bugs_halfcauchy("sigma_ecoregion"),
                            shape=n_regions,
                        )
                    else:
                        sigma_ecoregion = pm.HalfCauchy("sigma_ecoregion", beta=25)
                        ecoregion_offset = pm.Normal(
                            "ecoregion_offset", mu=0, sigma=1, shape=n_regions
                        )
                        ecoregion_effect = pm.Deterministic(
                            "ecoregion", g + sigma_ecoregion * ecoregion_offset
                        )

                if use_site_hierarchy:
                    if use_ecoregion_hierarchy:
                        if spec in {"legacy_r", "centered"}:
                            site_effect = pm.Normal(
                                "site_effect",
                                mu=ecoregion_effect[site_to_region],
                                sigma=_bugs_halfcauchy("sigma"),
                                shape=n_sites,
                            )
                        else:
                            sigma_site = pm.HalfCauchy("sigma_site", beta=25)
                            site_offset = pm.Normal(
                                "site_offset", mu=0, sigma=1, shape=n_sites
                            )
                            site_effect = pm.Deterministic(
                                "site_effect",
                                ecoregion_effect[site_to_region]
                                + sigma_site * site_offset,
                            )
                    elif spec in {"legacy_r", "centered"}:
                        site_effect = pm.Normal(
                            "site_effect",
                            mu=g,
                            sigma=_bugs_halfcauchy("sigma"),
                            shape=n_sites,
                        )
                    else:
                        sigma_site = pm.HalfCauchy("sigma_site", beta=25)
                        site_offset = pm.Normal(
                            "site_offset", mu=0, sigma=1, shape=n_sites
                        )
                        site_effect = pm.Deterministic(
                            "site_effect", g + sigma_site * site_offset
                        )

                if spec in {"legacy_r", "centered"}:
                    theta = _bugs_halfcauchy("theta")
                    fast_sample = False
                else:
                    theta = pm.HalfCauchy("theta", beta=25)
                    fast_sample = True

                if use_site_hierarchy:
                    eta = pm.math.dot(X, beta) + site_effect[self.site_idx]
                else:
                    eta = pm.math.dot(X, beta) + ecoregion_effect[self.region_idx]

            pi = pm.math.clip(
                pm.math.invlogit(eta), _PI_EPS, 1.0 - _PI_EPS
            )  # JAGS: max(1e-6, min(0.999999, pi_raw))
            pm.Beta(
                "y_obs",
                alpha=theta * pi,
                beta=theta * (1 - pi),
                observed=self.y,
            )

            step = pm.NUTS(
                target_accept=self.target_accept, max_treedepth=self.max_treedepth
            )
            sample_kw: dict = dict(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                step=step,
                random_seed=self.random_seed,
                return_inferencedata=True,
                progressbar=progressbar,
            )
            if fast_sample:
                sample_kw["idata_kwargs"] = {"log_likelihood": False}
            elif use_hierarchy and spec in {"legacy_r", "centered"}:
                sample_kw["initvals"] = self._legacy_initvals(
                    n_predictors,
                    n_regions,
                    n_sites,
                    self.n_chains,
                    self.random_seed,
                    use_diversity=use_diversity,
                    use_site_hierarchy=use_site_hierarchy,
                    use_ecoregion_hierarchy=use_ecoregion_hierarchy,
                )
            sample_kw = _parallel_sample_kwargs(
                sample_kw,
                ncores=ncores,
                n_chains=self.n_chains,
                mp_ctx=mp_ctx,
            )
            return _sample_with_parallel_fallback(sample_kw)

    def predict(
        self,
        X_new: np.ndarray,
        site_idx: np.ndarray,
        n_samples: int = 1000,
        verbose: bool = False,
        region_idx: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Generate predictions from the fitted model.

        NOTE: Predictions are returned in BETA-TRANSFORMED space.
        Use inverse_transform_beta() to convert back to original [0,1] scale.

        Args:
            X_new (np.ndarray): New design matrix
            site_idx (np.ndarray): Site indices for new observations. MUST match training site indices!
            Use model.reef_to_site_map to ensure consistent mapping.
            n_samples (int): Number of samples from posterior predictive
            verbose (bool): If True, print validation information

        Returns:
            dict[str, np.ndarray]: Dictionary of distribution statistics for each observation, with keys:
            Dictionary of distribution statistics for each observation, with keys:
            - 'mean': posterior predictive mean
            - 'std': posterior predictive standard deviation
            - 'median': posterior predictive median
            - 'ci_lower_95', 'ci_upper_95': 95% credible interval bounds (2.5%, 97.5%)
            - 'ci_lower_50', 'ci_upper_50': 50% credible interval bounds (25%, 75%)
            - 'min', 'max': minimum and maximum over posterior predictive draws
        """
        if self.trace is None:
            raise ValueError("\tModel must be fit before prediction")

        # Extract posterior samples
        # Get number of predictors from trace shape (more reliable than self.X which may not be saved)
        n_predictors = self.trace.posterior["beta"].shape[-1]
        beta_samples = self.trace.posterior["beta"].values.reshape(-1, n_predictors)

        use_site_hierarchy = getattr(self, "use_site_hierarchy", True)
        use_ecoregion_hierarchy = getattr(self, "use_ecoregion_hierarchy", True)
        use_hierarchy = use_site_hierarchy or use_ecoregion_hierarchy

        site_samples = None
        eco_samples = None
        if use_site_hierarchy and "site_effect" in self.trace.posterior:
            n_sites_trained = self.trace.posterior["site_effect"].shape[-1]
            site_samples = self.trace.posterior["site_effect"].values.reshape(
                -1, n_sites_trained
            )
        else:
            n_sites_trained = 0
        if use_ecoregion_hierarchy and "ecoregion" in self.trace.posterior:
            n_regions_trained = self.trace.posterior["ecoregion"].shape[-1]
            eco_samples = self.trace.posterior["ecoregion"].values.reshape(
                -1, n_regions_trained
            )
        else:
            n_regions_trained = 0
        theta_samples = self.trace.posterior["theta"].values.flatten()

        if region_idx is None:
            region_idx = self.region_idx
        if region_idx is not None and len(region_idx) != X_new.shape[0]:
            raise ValueError(
                "region_idx length must match X_new rows when provided explicitly."
            )

        if verbose:
            print("\n  🔍 PREDICT VALIDATION:")
            print(f"      X_new shape: {X_new.shape}")
            print(f"      Expected predictors: {n_predictors}")
            if use_site_hierarchy:
                print(f"      Trained sites: {n_sites_trained}")
                print(f"      Input unique sites: {len(np.unique(site_idx))}")
                print(f"      Site idx range: [{site_idx.min()}, {site_idx.max()}]")
            if use_ecoregion_hierarchy and region_idx is not None:
                print(f"      Trained regions: {n_regions_trained}")
                print(f"      Input unique regions: {len(np.unique(region_idx))}")

        site_idx_mapped = site_idx
        if use_site_hierarchy and site_samples is not None:
            unique_sites = np.unique(site_idx)
            if len(unique_sites) > n_sites_trained or max(unique_sites) >= n_sites_trained:
                site_idx_mapped = site_idx % n_sites_trained
                if len(unique_sites) > n_sites_trained:
                    print(
                        f"⚠️  Warning: {len(unique_sites)} unique sites in site_idx but model trained on {n_sites_trained} sites."
                    )
                    print("    Using modulo mapping - site effects may not be accurate!")
                    print(
                        "    Consider using model.reef_to_site_map for consistent site indices."
                    )
            elif verbose:
                print("      ✅ All site indices within valid range")
        elif verbose and not use_hierarchy:
            print("      Fixed-effects-only model (no random effects)")

        region_idx_mapped = region_idx
        if (
            use_ecoregion_hierarchy
            and eco_samples is not None
            and region_idx is not None
        ):
            unique_regions = np.unique(region_idx)
            if (
                len(unique_regions) > n_regions_trained
                or max(unique_regions) >= n_regions_trained
            ):
                region_idx_mapped = region_idx % n_regions_trained
                if len(unique_regions) > n_regions_trained:
                    print(
                        f"⚠️  Warning: {len(unique_regions)} unique regions in region_idx "
                        f"but model trained on {n_regions_trained} regions."
                    )
                    print("    Using modulo mapping - ecoregion effects may not be accurate!")

        # Randomly select samples
        n_total = len(beta_samples)
        idx = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)

        predictions = []
        for i in idx:
            eta = X_new @ beta_samples[i]
            if use_site_hierarchy and site_samples is not None:
                eta = eta + site_samples[i, site_idx_mapped]
            elif use_ecoregion_hierarchy and eco_samples is not None and region_idx_mapped is not None:
                eta = eta + eco_samples[i, region_idx_mapped]
            pi = inv_logit(eta)

            # Sample from beta distribution
            alpha = theta_samples[i] * pi
            beta_param = theta_samples[i] * (1 - pi)
            y_pred = np.random.beta(alpha, beta_param)
            predictions.append(y_pred)

        predictions = np.array(predictions)  # shape: (n_draws, n_obs)

        if verbose:
            print(
                f"      Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}"
            )

        # Compute distribution statistics along the draws axis
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0, ddof=0)
        median = np.median(predictions, axis=0)
        ci_lower_95 = np.percentile(predictions, 2.5, axis=0)
        ci_upper_95 = np.percentile(predictions, 97.5, axis=0)
        ci_lower_50 = np.percentile(predictions, 25.0, axis=0)
        ci_upper_50 = np.percentile(predictions, 75.0, axis=0)
        min_vals = predictions.min(axis=0)
        max_vals = predictions.max(axis=0)

        # Compute significance flags based on 95% interval excluding reference (default 0)
        significant = self._compute_significance(
            ci_lower_95, ci_upper_95, reference=0.0
        )

        return {
            "mean": mean,
            "std": std,
            "median": median,
            "ci_lower_95": ci_lower_95,
            "ci_upper_95": ci_upper_95,
            "ci_lower_50": ci_lower_50,
            "ci_upper_50": ci_upper_50,
            "min": min_vals,
            "max": max_vals,
            "significant": significant,
        }

    @staticmethod
    def _compute_significance(
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        reference: float = 0.0,
    ) -> np.ndarray:
        """
        Determine significance for each variable based on a credible interval.

        A variable is marked significant if its credible interval does not
        include the reference value (default 0.0), i.e. the entire interval is
        either above or below the reference.

        Parameters
        ----------
        ci_lower : np.ndarray
            Lower bounds of the credible interval (e.g. 2.5%).
        ci_upper : np.ndarray
            Upper bounds of the credible interval (e.g. 97.5%).
        reference : float, optional
            Reference value to test against (default is 0.0).

        Returns
        -------
        np.ndarray
            Boolean array indicating significance for each variable.
        """
        above_ref = ci_lower > reference
        below_ref = ci_upper < reference
        return np.logical_or(above_ref, below_ref)

    def get_coefficient_summary(self) -> pd.DataFrame:
        """
        Get summary statistics for regression coefficients.

        Returns
        -------
        pd.DataFrame
            Summary with mean, std, and credible intervals
        """
        if self.trace is None:
            raise ValueError("\tModel must be fit first")

        # Use 90% HDI so we get 5% / 95% bounds, to mirror the
        # MyBUGSOutput-style summaries (2.5/97.5 and 25/75 in R).
        var_names = ["beta"]
        if getattr(self, "use_diversity", True) and "beta_diversity" in self.trace.posterior:
            var_names.append("beta_diversity")
        beta_summary = az.summary(self.trace, var_names=var_names, hdi_prob=0.90)

        if self.col_names is not None:
            # Rename index to use column names
            new_index = []
            for idx in beta_summary.index:
                if "beta[" in idx:
                    i = int(idx.split("[")[1].split("]")[0])
                    new_index.append(self.col_names[i])
                else:
                    new_index.append(idx)
            beta_summary.index = new_index

        # Add a significance flag: interval does not include 0
        # (strictly above or strictly below zero).
        hdi_lower_cols = [c for c in beta_summary.columns if c.startswith("hdi_")][0:1]
        hdi_upper_cols = [c for c in beta_summary.columns if c.startswith("hdi_")][1:2]
        if hdi_lower_cols and hdi_upper_cols:
            lower_col = hdi_lower_cols[0]
            upper_col = hdi_upper_cols[0]
            beta_summary["significant"] = (beta_summary[lower_col] > 0) | (
                beta_summary[upper_col] < 0
            )

        return beta_summary

    def plot_coefficient_traces_and_posteriors(
        self,
        output_dir: Optional[Path] = None,
        show: bool = False,
        **kwargs: Any,
    ) -> dict[str, Figure]:
        """
        Plot coefficient MCMC traces, ESS, and posteriors for a fitted model.

        Delegates to :func:`src.plots.hb_beta_plots.plot_coefficient_traces_and_posteriors`.
        """
        if self.trace is None:
            raise ValueError("Model must be fit first")
        return plot_coefficient_traces_and_posteriors(
            self.trace,
            col_names=self.col_names,
            output_dir=output_dir,
            show=show,
            **kwargs,
        )

    def save_diagnostics(self, output_path: Path) -> None:
        """
        Save diagnostics for the model.

        Delegates to :func:`src.plots.hb_beta_plots.save_trace_diagnostics`.
        """
        if self.trace is None:
            raise ValueError("Model must be fit first")
        save_trace_diagnostics(self.trace, output_path)

    def save_model(self, output_path: Path) -> None:
        """
        Save the trained model to disk.

        Parameters
        ----------
        output_path : Path
            Directory path to save model files
        """
        if self.trace is None:
            raise ValueError("Model must be fit before saving")

        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        print(f"Saving model to {output_path}")

        # Save trace (MCMC samples)
        trace_path = output_path / "model_trace.nc"
        print(f"\tSaving trace to {trace_path}")
        if not self.trace.posterior.data_vars:
            raise ValueError("Cannot save trace with empty posterior.")
        self.trace.to_netcdf(trace_path)
        loaded = az.from_netcdf(trace_path)
        if not loaded.posterior.data_vars:
            raise RuntimeError(
                f"model_trace.nc at {trace_path} failed post-save validation. "
                "Load with arviz.from_netcdf(), not xarray.open_dataset()."
            )

        # Save metadata
        import json

        # Get n_predictors from trace if X not available
        n_predictors = (
            self.trace.posterior["beta"].shape[-1]
            if self.X is None
            else self.X.shape[1]
        )
        metadata = {
            "col_names": self.col_names,
            "n_observations": len(self.X) if self.X is not None else None,
            "n_predictors": n_predictors,
            "n_sites": len(np.unique(self.site_idx))
            if self.site_idx is not None
            else None,
            "n_regions": len(np.unique(self.region_idx))
            if self.region_idx is not None
            else None,
            "n_samples": self.n_samples,
            "n_tune": self.n_tune,
            "n_chains": self.n_chains,
            "target_accept": self.target_accept,
            "max_treedepth": self.max_treedepth,
            "random_seed": self.random_seed,
            "model_spec": getattr(self, "model_spec", "reparam"),
        }

        metadata_path = output_path / "model_metadata.json"
        print(f"\tSaving metadata to {metadata_path}")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Save site and region indices
        if self.site_idx is not None:
            np.save(output_path / "site_idx.npy", self.site_idx)
        if self.region_idx is not None:
            np.save(output_path / "region_idx.npy", self.region_idx)
        if self.site_to_region is not None:
            np.save(output_path / "site_to_region.npy", self.site_to_region)
        if self.diversity is not None:
            np.save(output_path / "diversity.npy", self.diversity)

        # Save reef_to_site_map - CRITICAL for consistent predictions
        if self.reef_to_site_map is not None:
            reef_map_path = output_path / "reef_to_site_map.json"
            print(f"\tSaving reef_to_site_map to {reef_map_path}")
            # Convert keys to strings for JSON serialization
            reef_map_json = {str(k): int(v) for k, v in self.reef_to_site_map.items()}
            with open(reef_map_path, "w") as f:
                json.dump(reef_map_json, f)

        # Save n_observations for inverse_transform_beta
        if self.n_observations is not None:
            np.save(output_path / "n_observations.npy", np.array([self.n_observations]))

        # Save standardization stats if available
        if (
            hasattr(self, "standardization_stats")
            and self.standardization_stats is not None
        ):
            import json

            # Convert numpy types to native Python types for JSON serialization
            std_stats_json = {}
            for key, value in self.standardization_stats.items():
                if isinstance(value, tuple) and len(value) == 2:
                    std_stats_json[key] = (float(value[0]), float(value[1]))
                else:
                    std_stats_json[key] = value

            std_stats_path = output_path / "standardization_stats.json"
            print(f"\tSaving standardization stats to {std_stats_path}")
            with open(std_stats_path, "w") as f:
                json.dump(std_stats_json, f, indent=2)

        print("Model saved successfully")

    @classmethod
    def load_model(cls, model_path: Path) -> "HierarchicalBetaModel":
        """
        Load a trained model from disk.

        Parameters
        ----------
        model_path : Path
            Directory path containing model files

        Returns
        -------
        HierarchicalBetaModel
            Loaded model instance
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        print(f"Loading model from {model_path}")

        model = cls()

        # Load trace
        trace_path = model_path / "model_trace.nc"
        if not trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        print(f"\tLoading trace from {trace_path}")
        model.trace = az.from_netcdf(trace_path)

        # Load metadata
        metadata_path = model_path / "model_metadata.json"
        if metadata_path.exists():
            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            model.col_names = metadata.get("col_names")

        # Load indices
        if (model_path / "site_idx.npy").exists():
            model.site_idx = np.load(model_path / "site_idx.npy")
        if (model_path / "region_idx.npy").exists():
            model.region_idx = np.load(model_path / "region_idx.npy")
        if (model_path / "site_to_region.npy").exists():
            model.site_to_region = np.load(model_path / "site_to_region.npy")
        if (model_path / "diversity.npy").exists():
            model.diversity = np.load(model_path / "diversity.npy")

        # Load reef_to_site_map - CRITICAL for consistent predictions
        reef_map_path = model_path / "reef_to_site_map.json"
        if reef_map_path.exists():
            import json

            with open(reef_map_path, "r") as f:
                reef_map_json = json.load(f)
            # Convert string keys back to original type (try int first, then keep as string)
            model.reef_to_site_map = {}
            for k, v in reef_map_json.items():
                try:
                    model.reef_to_site_map[int(k)] = v
                except ValueError:
                    model.reef_to_site_map[k] = v
            print(
                f"\tLoaded reef_to_site_map with {len(model.reef_to_site_map)} entries"
            )
        else:
            print(
                "\t⚠️  WARNING: reef_to_site_map.json not found. Site effects may be incorrect for predictions!"
            )
            model.reef_to_site_map = None

        # Load n_observations for inverse_transform_beta
        if (model_path / "n_observations.npy").exists():
            model.n_observations = int(np.load(model_path / "n_observations.npy")[0])
            print(f"\tLoaded n_observations: {model.n_observations}")
        else:
            print(
                "\t⚠️  WARNING: n_observations.npy not found. inverse_transform_beta may not work correctly!"
            )
            model.n_observations = None

        # Load standardization stats if available
        if (model_path / "standardization_stats.json").exists():
            import json

            with open(model_path / "standardization_stats.json", "r") as f:
                model.standardization_stats = json.load(f)

        print("Model loaded successfully")
        return model

    def get_coefficient_samples(self) -> dict[str, np.ndarray]:
        """
        Get posterior samples of coefficients.

        Returns
        -------
        dict
            dictionary with coefficient arrays
        """
        if self.trace is None:
            raise ValueError("Model must be fit first")

        samples: dict[str, np.ndarray] = {
            "beta": self.trace.posterior["beta"].values.reshape(-1, self.X.shape[1]),
            "theta": self.trace.posterior["theta"].values.flatten(),
        }
        if getattr(self, "use_diversity", True) and "beta_diversity" in self.trace.posterior:
            samples["beta_diversity"] = self.trace.posterior[
                "beta_diversity"
            ].values.flatten()
        if getattr(self, "use_ecoregion_hierarchy", True) and "ecoregion" in self.trace.posterior:
            samples["ecoregion"] = self.trace.posterior["ecoregion"].values.reshape(
                -1, len(np.unique(self.region_idx))
            )
        if getattr(self, "use_site_hierarchy", True) and "site_effect" in self.trace.posterior:
            samples["site_effect"] = self.trace.posterior["site_effect"].values.reshape(
                -1, len(np.unique(self.site_idx))
            )

        return samples
