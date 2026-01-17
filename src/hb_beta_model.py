"""
Hierarchical Bayesian Beta Model for Coral Cover Analysis

This module implements the coral cover beta regression model from Sully et al.,
originally written in R/JAGS, now translated to Python using PyMC.

The model predicts coral cover as a function of environmental variables
(temperature, turbidity, cyclones, etc.) with hierarchical random effects
for sites nested within ecoregions.

Reference: Sully et al. - "Present and future bright and dark spots for
coral reefs through climate change"

Author: Adapted from R code by Shannon Sully
"""

import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.special import expit as inv_logit

# Optional imports for Bayesian modeling
try:
    import arviz as az
    import pymc as pm

    HAS_PYMC = True
except ImportError:
    HAS_PYMC = False
    warnings.warn("PyMC not installed. Bayesian modeling will not be available.")


# =============================================================================
# CONFIGURATION
# =============================================================================

from src import config

SULLY_DATA_DIR = config.data_dir / "sully_2022"
OUTPUT_DIR = config.figures_dir / "hbb"
COVARIATE_LABELS_DICT = {
    "lat_stzd": "Latitude",
    "historical_sst_max_stzd": "Max. historical SST",
    "ssta_dhwmax_stzd": "Max. SSTA DHW",
    "tsa_freqstdev_stzd": "TSA standard deviation",
    "ssta_min_stzd": "Min. SST anomaly",
    "tsa_max_stzd": "Max. TSA anomaly",
    "beta_diversity": "Beta diversity",
    "sst_mean_stzd": "Mean SST",
    "depth_stzd": "Depth",
    "ssta_mean_stzd": "Mean SSTA",
    "cyclone_stzd": "Cyclone frequency",
    "ssta_freqstdev_stzd": "SSTA frequency",
    "human_pop_stzd": "Local human population",
    "turbidity_mean_stzd": "Mean turbidity",
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_data(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load coral cover data from CSV.

    Parameters
    ----------
    filepath : Path, optional
        Path to the data.csv file. If None, uses default location.

    Returns
    -------
    pd.DataFrame
        Loaded coral cover data
    """
    if filepath is None:
        filepath = SULLY_DATA_DIR / "data_for_maps.csv"

    df = pd.read_csv(filepath)

    # Create reef column for compatibility
    df["reef"] = df["Reef_ID"]
    df["lat"] = df["Latitude.Degrees"]
    df["lon"] = df["Longitude.Degrees"]
    df["diversity"] = df["diversity.standardized"]

    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove sites with NA values or implausible values.

    Parameters
    ----------
    df : pd.DataFrame
        Raw coral cover data

    Returns
    -------
    pd.DataFrame
        Cleaned data with NAs and invalid values removed
    """
    df = df.copy()

    df.rename(columns=str.lower, inplace=True)

    # Remove rows with missing or invalid coral cover
    df = df[df["average_coral_cover"].notna()]
    df = df[df["average_coral_cover"] > 0]

    # Remove rows with missing environmental variables – these are the variables used in the final model, although there are more in the analysis
    required_cols = [
        "sst_mean",
        "sst_stdev",
        "sst_freqmax",
        "sst_freqmean",
        "turbidity_mean",
        "cyclone",
        "depth",
        "historical_sst_max",
        "sst_mean_rcp85_2100",
    ]

    for col in required_cols:
        if col in df.columns:
            df = df[df[col].notna()]

    # Remove points on land (high turbidity indicates land)
    if "turbidity_mean" in df.columns:
        df = df[df["turbidity_mean"] < 0.35]

    # Remove rows with missing ecoregion
    if "erg" in df.columns:
        df = df[df["erg"].notna()]

    return df.reset_index(drop=True)


def standardize_variables(
    df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """
    Standardize explanatory variables (zero mean, unit variance).

    Parameters
    ----------
    df : pd.DataFrame
        Data with variables to standardize
    columns : list of str
        Column names to standardize

    Returns
    -------
    pd.DataFrame
        Data with standardized columns
    dict
        Dictionary of (mean, std) for each column for later un-standardization
    """
    df = df.copy()
    stats_dict = {}

    for col in columns:
        if col in df.columns:
            mean_val = df[col].dropna().mean()
            std_val = df[col].dropna().std()
            stats_dict[col] = (mean_val, std_val)
            df[f"{col}_stzd"] = (df[col] - mean_val) / std_val

    return df, stats_dict


def create_output_dir_path(output_path: Path) -> Path:
    """
    Create output directory path.

    Parameters
    ----------
    output_path : Path
        Path to save diagnostics
    """
    # save with a timestamp in the output path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir_path = output_path / f"model_{timestamp}"
    output_dir_path.mkdir(parents=True, exist_ok=True)
    return output_dir_path


# =============================================================================
# CORRELATION ANALYSIS
# =============================================================================


def compute_correlation_matrix(
    df: pd.DataFrame, columns: Optional[list[str]] = None
) -> pd.DataFrame:
    """
    Compute correlation matrix for specified columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input data
    columns : list of str, optional
        Columns to include. If None, uses default environmental variables.

    Returns
    -------
    pd.DataFrame
        Correlation matrix
    """
    if columns is None:
        columns = [
            "lat",
            "depth",
            "human_pop",
            "cyclone",
            "sst_mean",
            "sst_min",
            "sst_max",
            "sst_stdev",
            "ssta_mean",
            "ssta_min",
            "ssta_max",
            "ssta_stdev",
            "ssta_freqmax",
            "ssta_freqstdev",
            "ssta_freqmean",
            "ssta_dhwmax",
            "ssta_dhwmean",
            "ssta_dhwstdev",
            "tsa_min",
            "tsa_max",
            "tsa_stdev",
            "tsa_mean",
            "tsa_freqmax",
            "tsa_freqstdev",
            "tsa_freqmean",
            "tsa_dhwmax",
            "tsa_dhwmean",
            "tsa_dhwstdev",
            "turbidity_mean",
            "turbidity_max",
            "turbidity_min",
            "historical_sst_mean",
            "historical_sst_max",
            "historical_sst_sd",
        ]

    # Filter to available columns
    available_cols = [c for c in columns if c in df.columns]
    if len(set(columns) - set(available_cols)) > 0:
        print(
            f"\tThe following columns were unavailable: {set(columns) - set(available_cols)}"
        )
    else:
        print("\tAll necessary columns were available")

    return df[available_cols].corr()


def plot_correlation_matrix(
    corr_matrix: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (12, 10),
    cmap: str = "RdBu_r",
) -> plt.Figure:
    """
    Create correlation plot.

    Parameters
    ----------
    corr_matrix : pd.DataFrame
        Correlation matrix
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size
    cmap : str
        Colormap name

    Returns
    -------
    matplotlib.figure.Figure
        The figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap=cmap,
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.5},
        annot=False,
        ax=ax,
    )
    # # TODO: update colorbar to be between -1 and 1
    # cbar = ax.collections[0].colorbar
    # cbar.set_clim(-1, 1)

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# BETA REGRESSION MODEL
# =============================================================================


def transform_to_beta(y: np.ndarray, n: int) -> np.ndarray:
    """
    Transform [0,1] bounded data for beta distribution.

    Applies the transformation: y_beta = (y * (n-1) + 0.5) / n
    This avoids exact 0s and 1s which are problematic for beta distribution.

    Parameters
    ----------
    y : np.ndarray
        Original response values in [0, 1]
    n : int
        Sample size

    Returns
    -------
    np.ndarray
        Transformed values in (0, 1)
    """
    return (y * (n - 1) + 0.5) / n


def inverse_transform_beta(y_beta: np.ndarray, n: int) -> np.ndarray:
    """
    Inverse of beta transformation.

    Parameters
    ----------
    y_beta : np.ndarray
        Transformed values
    n : int
        Sample size used in original transformation

    Returns
    -------
    np.ndarray
        Original scale values
    """
    return (y_beta * n - 0.5) / (n - 1)


def build_design_matrix(
    df: pd.DataFrame, predictors: Optional[list[str]] = None, add_intercept: bool = True
) -> tuple[np.ndarray, list[str]]:
    """
    Build design matrix for beta regression.

    Parameters
    ----------
    df : pd.DataFrame
        Data with standardized predictors
    predictors : list of str, optional
        Predictor column names (should be standardized versions)
    add_intercept : bool
        Whether to add intercept column

    Returns
    -------
    np.ndarray
        Design matrix X
    list
        Column names
    """
    if predictors is None:
        predictors = [
            "lat_std",
            "depth_std",
            "human_pop_std",
            "cyclone_std",
            "sst_mean_std",
            "ssta_mean_std",
            "ssta_min_std",
            "ssta_freqstdev_std",
            "ssta_dhwmax_std",
            "tsa_max_std",
            "tsa_freqstdev_std",
            "turbidity_mean_std",
            "historical_sst_max_std",
        ]

    # Filter to available columns
    available_predictors = [p for p in predictors if p in df.columns]

    X = df[available_predictors].values.astype(float)
    col_names = available_predictors.copy()

    if add_intercept:
        X = np.column_stack([np.ones(len(df)), X])
        col_names = ["Intercept"] + col_names

    return X, col_names


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
        max_treedepth: int = 15,
        random_seed: int = 42,
    ) -> "HierarchicalBetaModel":
        """
        Fit the hierarchical beta regression model.

        Parameters
        ----------
        X : np.ndarray
            Design matrix (n_obs x n_predictors)
        y : np.ndarray
            Response variable (coral cover, transformed to (0,1))
        site_idx : np.ndarray
            Site index for each observation
        region_idx : np.ndarray
            Region index for each observation
        site_to_region : np.ndarray
            Mapping from site index to region index
        reef_to_site_map : dict, optional
            Mapping from reef_id to site index. CRITICAL for consistent predictions.
            If not provided, predictions may use incorrect site effects.
        diversity : np.ndarray, optional
            Standardized diversity values for each region
        col_names : list, optional
            Names of columns in X
        n_samples : int
            Number of posterior samples per chain
        n_tune : int
            Number of tuning samples
        n_chains : int
            Number of MCMC chains
        target_accept : float
            Target acceptance rate for NUTS sampler
        max_treedepth : int
            Maximum tree depth for NUTS sampler
        random_seed : int
            Random seed for reproducibility

        Returns
        -------
        self
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

        n_obs, n_predictors = X.shape
        n_sites = len(np.unique(site_idx))
        n_regions = len(np.unique(region_idx))

        if diversity is None:
            diversity = np.zeros(n_regions)

        with pm.Model() as self.model:
            # Fixed effects priors (weakly informative, matching R's dnorm(0, 0.0001) = SD 100)
            beta = pm.Normal("beta", mu=0, sigma=100, shape=n_predictors)

            # Diversity effect on ecoregion means
            beta_diversity = pm.Normal("beta_diversity", mu=0, sigma=100)
            mu_global = pm.Normal("mu_global", mu=0, sigma=100)

            # Ecoregion-level random effects - non-centered parameterization
            sigma_ecoregion = pm.HalfCauchy("sigma_ecoregion", beta=25)
            g = mu_global + beta_diversity * diversity
            ecoregion_offset = pm.Normal(
                "ecoregion_offset", mu=0, sigma=1, shape=n_regions
            )
            ecoregion_effect = pm.Deterministic(
                "ecoregion", g + sigma_ecoregion * ecoregion_offset
            )

            # Site-level random effects - non-centered parameterization
            sigma_site = pm.HalfCauchy("sigma_site", beta=25)
            site_offset = pm.Normal("site_offset", mu=0, sigma=1, shape=n_sites)
            site_effect = pm.Deterministic(
                "site_effect",
                ecoregion_effect[site_to_region] + sigma_site * site_offset,
            )

            # Precision parameter (theta in original notation)
            theta = pm.HalfCauchy("theta", beta=25)

            # Linear predictor
            eta = pm.math.dot(X, beta) + site_effect[site_idx]

            # Mean (pi) via inverse logit
            pi = pm.math.invlogit(eta)

            # Beta distribution parameterized by mean and precision
            alpha = theta * pi
            beta_param = theta * (1 - pi)

            # Likelihood
            y_obs = pm.Beta("y_obs", alpha=alpha, beta=beta_param, observed=y)

            # custom step size for better sampling
            step = pm.NUTS(target_accept=target_accept, max_treedepth=max_treedepth)
            # Sample
            self.trace = pm.sample(
                n_samples,
                tune=n_tune,
                chains=n_chains,
                step=step,
                random_seed=random_seed,
                return_inferencedata=True,
                # initvals=map_estimate
            )

        self.summary = az.summary(self.trace)

        return self

    def predict(
        self,
        X_new: np.ndarray,
        site_idx: np.ndarray,
        n_samples: int = 1000,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions from the fitted model.

        NOTE: Predictions are returned in BETA-TRANSFORMED space.
        Use inverse_transform_beta() to convert back to original [0,1] scale.

        Parameters
        ----------
        X_new : np.ndarray
            New design matrix
        site_idx : np.ndarray
            Site indices for new observations. MUST match training site indices!
            Use model.reef_to_site_map to ensure consistent mapping.
        n_samples : int
            Number of samples from posterior predictive
        verbose : bool
            If True, print validation information

        Returns
        -------
        np.ndarray
            Mean predictions (in beta-transformed space)
        np.ndarray
            Standard deviation of predictions
        """
        if self.trace is None:
            raise ValueError("\tModel must be fit before prediction")

        # Extract posterior samples
        # Get number of predictors from trace shape (more reliable than self.X which may not be saved)
        n_predictors = self.trace.posterior["beta"].shape[-1]
        beta_samples = self.trace.posterior["beta"].values.reshape(-1, n_predictors)

        # Get number of sites from trace
        n_sites_trained = self.trace.posterior["site_effect"].shape[-1]
        site_samples = self.trace.posterior["site_effect"].values.reshape(
            -1, n_sites_trained
        )
        theta_samples = self.trace.posterior["theta"].values.flatten()

        if verbose:
            print("\n  🔍 PREDICT VALIDATION:")
            print(f"      X_new shape: {X_new.shape}")
            print(f"      Expected predictors: {n_predictors}")
            print(f"      Trained sites: {n_sites_trained}")
            print(f"      Input unique sites: {len(np.unique(site_idx))}")
            print(f"      Site idx range: [{site_idx.min()}, {site_idx.max()}]")

        # Handle site_idx: map to valid range if needed
        unique_sites = np.unique(site_idx)
        if len(unique_sites) > n_sites_trained or max(unique_sites) >= n_sites_trained:
            # Map site_idx to valid range (0 to n_sites_trained-1) using modulo
            # This handles cases where new data has different site indices
            site_idx_mapped = site_idx % n_sites_trained
            if len(unique_sites) > n_sites_trained:
                print(
                    f"⚠️  Warning: {len(unique_sites)} unique sites in site_idx but model trained on {n_sites_trained} sites."
                )
                print("    Using modulo mapping - site effects may not be accurate!")
                print(
                    "    Consider using model.reef_to_site_map for consistent site indices."
                )
        else:
            site_idx_mapped = site_idx
            if verbose:
                print("      ✅ All site indices within valid range")

        # Randomly select samples
        n_total = len(beta_samples)
        idx = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)

        predictions = []
        for i in idx:
            eta = X_new @ beta_samples[i] + site_samples[i, site_idx_mapped]
            pi = inv_logit(eta)

            # Sample from beta distribution
            alpha = theta_samples[i] * pi
            beta_param = theta_samples[i] * (1 - pi)
            y_pred = np.random.beta(alpha, beta_param)
            predictions.append(y_pred)

        predictions = np.array(predictions)

        if verbose:
            print(
                f"      Predictions: mean={predictions.mean():.4f}, std={predictions.std():.4f}"
            )

        return predictions.mean(axis=0), predictions.std(axis=0)

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

        beta_summary = az.summary(self.trace, var_names=["beta", "beta_diversity"])

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

        return beta_summary

    def plot_coefficients(
        self, output_path: Optional[Path] = None, figsize: tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Create coefficient plot with credible intervals.

        Parameters
        ----------
        output_path : Path, optional
            Path to save figure
        figsize : tuple
            Figure size

        Returns
        -------
        matplotlib.figure.Figure
        """
        coef_summary = self.get_coefficient_summary()

        # Exclude intercept for plotting
        if "Intercept" in coef_summary.index:
            coef_summary = coef_summary.drop("Intercept")

        fig, ax = plt.subplots(figsize=figsize)

        # Sort by coefficient magnitude
        coef_summary = coef_summary.sort_values("mean")

        y_pos = np.arange(len(coef_summary))

        # Determine colors based on significance
        colors = []
        for (
            _,
            row,
        ) in (
            coef_summary.iterrows()
        ):  # TODO: why 3 and 97? # TODO: bar plot for significance
            if row["hdi_3%"] > 0:
                colors.append("blue")
            elif row["hdi_97%"] < 0:
                colors.append("red")
            else:
                colors.append("gray")

        # Plot error bars (94% HDI)
        ax.hlines(
            y_pos,
            coef_summary["hdi_3%"],
            coef_summary["hdi_97%"],
            color="black",
            linewidth=1,
        )
        # dummy label for point significant legend
        ax.scatter(
            [],
            [],
            c="blue",
            s=100,
            zorder=5,
            edgecolor="black",
            label="Significant positive",
        )
        ax.scatter(
            [],
            [],
            c="red",
            s=100,
            zorder=5,
            edgecolor="black",
            label="Significant negative",
        )
        ax.scatter(
            [],
            [],
            c="gray",
            s=100,
            zorder=5,
            edgecolor="black",
            label="Not significant",
        )

        # Plot points
        ax.scatter(
            coef_summary["mean"], y_pos, c=colors, s=100, zorder=5, edgecolor="black"
        )

        # Add zero line
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=1)

        ax.set_yticks(y_pos)
        ax.set_yticklabels([COVARIATE_LABELS_DICT[ind] for ind in coef_summary.index])
        ax.set_xlabel("Estimated coefficient")
        # ax.set_title('Beta Regression Coefficients')
        ax.set_xlim(-0.5, 0.5)
        ax.legend(loc="lower right")
        ax.grid(which="major", axis="x", linestyle="--", linewidth=0.5)
        plt.tight_layout()

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")

        return fig

    def save_diagnostics(self, output_path: Path) -> None:
        """
        Save diagnostics for the model.

        Parameters
        ----------
        output_path : Path
            Path to save diagnostics
        """
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving diagnostics to {output_path}")

        print("\t[DIAGNOSTIC 1/5] Plotting trace...")
        az.plot_trace(self.trace)
        plt.savefig(output_path / "trace.png")
        plt.close()

        print("\t[DIAGNOSTIC 2/5] Plotting pair...")
        az.plot_pair(self.trace, var_names=["beta", "beta_diversity"])
        plt.savefig(output_path / "pair.png")
        plt.close()

        print("\t[DIAGNOSTIC 3/5] Plotting posterior...")
        az.plot_posterior(self.trace, var_names=["beta", "beta_diversity"])
        plt.savefig(output_path / "posterior.png")
        plt.close()

        print("\t[DIAGNOSTIC 4/5] Plotting autocorrelation...")
        az.plot_autocorr(self.trace, var_names=["beta", "beta_diversity"])
        plt.savefig(output_path / "autocorr.png")
        plt.close()

        print("\t[DIAGNOSTIC 5/5] Plotting ESS...")
        az.plot_ess(self.trace, var_names=["beta", "beta_diversity"])
        plt.savefig(output_path / "ess.png")
        plt.close()

        # az.plot_rhat(self.trace, var_names=['beta', 'beta_diversity'])    # apparently no plot_rhat
        # plt.savefig(output_path / "rhat.png")
        # plt.close()

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
        self.trace.to_netcdf(trace_path)

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

        samples = {
            "beta": self.trace.posterior["beta"].values.reshape(-1, self.X.shape[1]),
            "beta_diversity": self.trace.posterior["beta_diversity"].values.flatten(),
            "ecoregion": self.trace.posterior["ecoregion"].values.reshape(
                -1, len(np.unique(self.region_idx))
            ),
            "site_effect": self.trace.posterior["site_effect"].values.reshape(
                -1, len(np.unique(self.site_idx))
            ),
            "theta": self.trace.posterior["theta"].values.flatten(),
        }

        return samples


# =============================================================================
# FUTURE PROJECTIONS
# =============================================================================


def build_current_design_matrix(
    model: "HierarchicalBetaModel",
    df: pd.DataFrame,
    standardization_stats: Optional[dict[str, tuple[float, float]]] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the current design matrix from a dataframe using the model's predictors.

    Parameters
    ----------
    model : HierarchicalBetaModel
        Trained model (must have col_names and reef_to_site_map attributes)
    df : pd.DataFrame
        Dataframe with current/historical data
    standardization_stats : dict, optional
        Standardization statistics. If None, uses model.standardization_stats
    verbose : bool
        If True, print validation information

    Returns
    -------
    tuple
        (X_current, site_idx) - Design matrix and site indices
    """
    if verbose:
        print("=" * 60)
        print("BUILD CURRENT DESIGN MATRIX - VALIDATION")
        print("=" * 60)

    if model.col_names is None:
        raise ValueError("Model must have col_names attribute")

    if standardization_stats is None:
        if hasattr(model, "standardization_stats"):
            standardization_stats = model.standardization_stats
        else:
            raise ValueError(
                "standardization_stats must be provided or model must have standardization_stats"
            )

    # Get predictors (exclude 'Intercept' if present)
    predictors = [p for p in model.col_names if p != "Intercept"]

    if verbose:
        print(
            f"\n📊 Model predictors ({len(predictors)}): {predictors[:5]}..."
            if len(predictors) > 5
            else f"\n📊 Model predictors ({len(predictors)}): {predictors}"
        )

    # Standardize variables if needed
    df_work = df.copy()
    standardized_vars = []
    for pred in predictors:
        base_name = pred.replace("_stzd", "")
        if base_name not in standardization_stats:
            continue

        # Check if already standardized
        if pred not in df_work.columns:
            # Need to standardize
            if base_name in df_work.columns:
                mean_val, std_val = standardization_stats[base_name]
                df_work[pred] = (df_work[base_name] - mean_val) / std_val
                standardized_vars.append(pred)

                if verbose:
                    raw_vals = df_work[base_name]
                    std_vals = df_work[pred]
                    print(f"  📐 Standardizing '{base_name}' -> '{pred}':")
                    print(
                        f"      Raw:  min={raw_vals.min():.3f}, max={raw_vals.max():.3f}, mean={raw_vals.mean():.3f}"
                    )
                    print(
                        f"      Std:  min={std_vals.min():.3f}, max={std_vals.max():.3f}, mean={std_vals.mean():.3f}"
                    )
                    print(f"      Stats: mean={mean_val:.3f}, std={std_val:.3f}")
            else:
                raise ValueError(
                    f"Column '{base_name}' or '{pred}' not found in dataframe"
                )

    # Build design matrix
    X_current, _ = build_design_matrix(df_work, predictors, add_intercept=True)

    if verbose:
        print(f"\n📐 Design matrix shape: {X_current.shape}")
        print(
            f"    Column means: {X_current.mean(axis=0)[:5]}..."
            if X_current.shape[1] > 5
            else f"    Column means: {X_current.mean(axis=0)}"
        )

    # Get site indices - USE SAVED MAPPING IF AVAILABLE
    if model.reef_to_site_map is not None and "reef_id" in df_work.columns:
        # Use the saved mapping from training for consistent site effects
        reef_ids = df_work["reef_id"].values
        site_idx = np.array([model.reef_to_site_map.get(rid, -1) for rid in reef_ids])

        # Check for unmapped reef_ids
        unmapped_count = np.sum(site_idx == -1)
        n_sites_trained = len(model.reef_to_site_map)

        if verbose:
            print(
                f"\n🔗 Using saved reef_to_site_map ({n_sites_trained} sites in training)"
            )
            print(
                f"    Mapped {len(reef_ids) - unmapped_count}/{len(reef_ids)} reef_ids successfully"
            )

        if unmapped_count > 0:
            print(
                f"    ⚠️  WARNING: {unmapped_count} reef_ids not found in training data!"
            )
            print("       These will use mean site effect (index 0)")
            site_idx[site_idx == -1] = 0  # Default to site 0 for unknown sites

    elif "site" in df_work.columns:
        site_idx = df_work["site"].values
        if verbose:
            print("\n🔗 Using 'site' column directly")
            print(f"    Unique sites: {len(np.unique(site_idx))}")
    elif "reef_id" in df_work.columns:
        # FALLBACK: Create new codes - WARNING: may not match training!
        print("\n⚠️  WARNING: No reef_to_site_map available. Creating new site indices.")
        print("    Site effects may NOT match training data!")
        site_idx = pd.Categorical(df_work["reef_id"]).codes
        if verbose:
            print(f"    Created {len(np.unique(site_idx))} unique site indices")
    else:
        # Create dummy site indices if not available
        print("\n⚠️  WARNING: No site information available. Using dummy indices.")
        site_idx = np.zeros(len(df_work), dtype=int)

    if verbose:
        print("\n✅ Design matrix and site indices ready")
        print("=" * 60)

    return X_current, site_idx


def project_future_coral_cover(
    model: "HierarchicalBetaModel",
    df: pd.DataFrame,
    X_current: np.ndarray,
    site_idx: np.ndarray,
    changed_columns: dict[str, str],
    standardization_stats: Optional[dict[str, tuple[float, float]]] = None,
    output_path: Optional[Path] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Project future coral cover by updating specified columns in the design matrix.

    This is a simplified interface: just specify which standardized variable columns
    should be updated with which dataframe columns, and the function handles the rest.

    Parameters
    ----------
    model : HierarchicalBetaModel
        Fitted model (must have col_names attribute)
    df : pd.DataFrame
        Dataframe containing both current and future variable columns
    X_current : np.ndarray
        Current design matrix (n_obs x n_predictors)
    site_idx : np.ndarray
        Site indices for observations
    changed_columns : dict
        Dictionary mapping standardized variable names (as in model.col_names)
        to dataframe column names containing future values.
        Example: {'sst_mean_stzd': 'sst_mean_ssp585_2100', 'human_pop_stzd': 'human_pop_2100'}
    standardization_stats : dict, optional
        Dictionary mapping variable names (without _stzd suffix) to (mean, std) tuples.
        If None, will try to extract from model if available.
    output_path : Path, optional
        Path to save results CSV file
    verbose : bool
        If True, print validation information

    Returns
    -------
    pd.DataFrame
        DataFrame with current and projected coral cover, including:
        - Y_current: Current predictions (inverse-transformed to original scale)
        - Y_future: Future predictions (inverse-transformed to original scale)
        - Y_future_std: Standard deviation of future predictions
        - Y_change: Absolute change
        - Y_relative_change: Relative change (%)
    """
    if verbose:
        print("=" * 60)
        print("PROJECT FUTURE CORAL COVER - VALIDATION")
        print("=" * 60)

    if model.col_names is None:
        raise ValueError(
            "Model must have col_names attribute. Ensure model was fit with col_names parameter."
        )

    if standardization_stats is None:
        if hasattr(model, "standardization_stats"):
            standardization_stats = model.standardization_stats
        else:
            raise ValueError(
                "standardization_stats must be provided or model must have standardization_stats attribute"
            )

    # Get n_observations for inverse_transform_beta
    if hasattr(model, "n_observations") and model.n_observations is not None:
        n_obs = model.n_observations
        if verbose:
            print(f"\n📊 Using n_observations from model: {n_obs}")
    else:
        n_obs = len(df)
        if verbose:
            print(f"\n⚠️  n_observations not saved in model, using df length: {n_obs}")

    # Create future design matrix
    X_future = X_current.copy()

    if verbose:
        print(f"\n🔄 Updating {len(changed_columns)} variables for future projection:")

    # Update variables in design matrix
    for std_var_name, future_col_name in changed_columns.items():
        if std_var_name not in model.col_names:
            print(
                f"⚠️  Warning: Variable '{std_var_name}' not in model columns. Skipping."
            )
            continue

        if future_col_name not in df.columns:
            raise ValueError(
                f"Future column '{future_col_name}' not found in dataframe."
            )

        # Find column index in design matrix
        col_idx = model.col_names.index(std_var_name)

        # Get base variable name (without _stzd suffix)
        base_var_name = std_var_name.replace("_stzd", "")

        # Standardize future values using original statistics
        if base_var_name in standardization_stats:
            mean_val, std_val = standardization_stats[base_var_name]
            future_values = df[future_col_name].values

            # Check for NaN values
            nan_count = np.sum(np.isnan(future_values))

            # Current standardized values
            current_std_values = X_current[:, col_idx]

            # Standardize future values
            future_values_std = (future_values - mean_val) / std_val

            # Update design matrix
            X_future[:, col_idx] = future_values_std

            if verbose:
                print(f"\n  📐 {std_var_name} <- {future_col_name}:")
                print(f"      Standardization: mean={mean_val:.4f}, std={std_val:.4f}")
                print(
                    f"      Future raw: min={np.nanmin(future_values):.4f}, max={np.nanmax(future_values):.4f}, mean={np.nanmean(future_values):.4f}"
                )
                print(
                    f"      Current std: min={current_std_values.min():.4f}, max={current_std_values.max():.4f}, mean={current_std_values.mean():.4f}"
                )
                print(
                    f"      Future std:  min={np.nanmin(future_values_std):.4f}, max={np.nanmax(future_values_std):.4f}, mean={np.nanmean(future_values_std):.4f}"
                )
                print(
                    f"      Delta std:   mean={np.nanmean(future_values_std - current_std_values):.4f}"
                )
                if nan_count > 0:
                    print(f"      ⚠️  {nan_count} NaN values in future data")
        else:
            raise ValueError(f"Standardization stats not found for {base_var_name}.")

    # Get predictions (in beta-transformed space)
    if verbose:
        print("\n🔮 Generating predictions...")
    print("Generating current predictions...")
    y_current_beta, y_current_std = model.predict(
        X_current, site_idx, n_samples=1000, verbose=verbose
    )

    print("Generating future predictions...")
    y_future_beta, y_future_std = model.predict(
        X_future, site_idx, n_samples=1000, verbose=verbose
    )

    # Apply inverse_transform_beta to convert from beta space to original scale
    if verbose:
        print(f"\n🔄 Applying inverse_transform_beta (n={n_obs})...")
        print(
            f"    Before transform - Current: mean={y_current_beta.mean():.4f}, Future: mean={y_future_beta.mean():.4f}"
        )

    y_current = inverse_transform_beta(y_current_beta, n_obs)
    y_future = inverse_transform_beta(y_future_beta, n_obs)

    if verbose:
        print(
            f"    After transform  - Current: mean={y_current.mean():.4f}, Future: mean={y_future.mean():.4f}"
        )

    # Calculate changes
    y_change = y_future - y_current
    y_relative_percent_change = np.where(
        y_current > 0, (y_future - y_current) / y_current * 100, np.nan
    )

    if verbose:
        print("\n📊 PROJECTION SUMMARY:")
        print(
            f"    Current coral cover:  mean={y_current.mean() * 100:.2f}%, median={np.median(y_current) * 100:.2f}%"
        )
        print(
            f"    Future coral cover:   mean={y_future.mean() * 100:.2f}%, median={np.median(y_future) * 100:.2f}%"
        )
        print(
            f"    Absolute change:      mean={y_change.mean() * 100:.2f}%, median={np.median(y_change) * 100:.2f}%"
        )
        print(
            f"    Relative change:      mean={np.nanmean(y_relative_percent_change):.2f}%, median={np.nanmedian(y_relative_percent_change):.2f}%"
        )
        print("=" * 60)
    # Create results dataframe
    results = pd.DataFrame(
        {
            "Y_current": y_current,
            "Y_current_std": y_current_std,
            "Y_future": y_future,
            "Y_future_std": y_future_std,
            "Y_change": y_change,
            "Y_relative_percent_change": y_relative_percent_change,
        }
    )

    # Add metadata columns if available
    if "lat" in df.columns:
        results["latitude"] = df["lat"].values
    if "lon" in df.columns:
        results["longitude"] = df["lon"].values
    if "site" in df.columns:
        results["site"] = df["site"].values
    if "reef_id" in df.columns:
        results["reef_id"] = df["reef_id"].values

    # Save results if path provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    return results


def load_model_and_project(
    model_path: Path,
    df: pd.DataFrame,
    X: np.ndarray,
    site_idx: np.ndarray,
    scenarios: list[tuple[str, int]],
    update_variables: Optional[dict[str, str]] = None,
    output_dir: Optional[Path] = None,
) -> dict[tuple[str, int], pd.DataFrame]:
    """
    Convenience function to load a trained model and run projections for multiple scenarios.

    Parameters
    ----------
    model_path : Path
        Path to saved model directory
    df : pd.DataFrame
        Data with future projections
    X : np.ndarray
        Current design matrix
    site_idx : np.ndarray
        Site indices
    scenarios : list of tuples
        list of (scenario, year) tuples, e.g., [('ssp585', 2050), ('ssp585', 2100)]
    update_variables : dict, optional
        dictionary mapping standardized variable names to future column names
    output_dir : Path, optional
        Directory to save projection results

    Returns
    -------
    dict
        dictionary mapping (scenario, year) tuples to result DataFrames
    """
    # Load model
    print(f"Loading model from {model_path}")
    model = HierarchicalBetaModel.load_model(model_path)

    # Run projections for each scenario
    results = {}
    for scenario, year in scenarios:
        print(f"\n{'=' * 80}")
        print(f"Projecting for scenario: {scenario}, year: {year}")
        print(f"{'=' * 80}")

        # Build update_variables dict from scenario/year if not provided
        if update_variables is None:
            update_variables = {
                "sst_mean_stzd": f"sst_mean_{scenario}_{year}",
                "human_pop_stzd": f"human_pop_{year}_vals",
            }

        result = project_future_coral_cover(
            model=model,
            df=df,
            X_current=X,
            site_idx=site_idx,
            changed_columns=update_variables,
            standardization_stats=model.standardization_stats,
            output_path=output_dir / f"projections_{scenario}_{year}.csv"
            if output_dir
            else None,
        )

        results[(scenario, year)] = result

    return results


# =============================================================================
# BRIGHT AND DARK SPOTS ANALYSIS
# =============================================================================


def identify_bright_dark_spots(
    observed: np.ndarray, expected: np.ndarray, threshold_sd: float = 1.5
) -> pd.DataFrame:
    """
    Identify bright spots (observed >> expected) and dark spots (observed << expected).

    Parameters
    ----------
    observed : np.ndarray
        Observed coral cover values
    expected : np.ndarray
        Model-predicted expected coral cover
    threshold_sd : float
        Number of standard deviations for classification

    Returns
    -------
    pd.DataFrame
        Classification results
    """
    deviation = observed - expected
    sd = np.std(observed)

    classification = np.where(
        deviation > threshold_sd * sd,
        "bright_spot",
        np.where(deviation < -threshold_sd * sd, "dark_spot", "normal"),
    )

    return pd.DataFrame(
        {
            "observed": observed,
            "expected": expected,
            "deviation": deviation,
            "deviation_normalized": deviation / sd,
            "classification": classification,
        }
    )


def plot_observed_vs_expected(
    observed: np.ndarray,
    expected: np.ndarray,
    classification: Optional[np.ndarray] = None,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (8, 8),
) -> plt.Figure:
    """
    Create observed vs expected coral cover plot.

    Parameters
    ----------
    observed : np.ndarray
        Observed coral cover
    expected : np.ndarray
        Expected coral cover
    classification : np.ndarray, optional
        Bright/dark spot classification
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Color points by classification
    if classification is not None:  # TODO: fix and update this assignment
        colors = np.where(
            classification == "bright_spot",
            "yellow",
            np.where(classification == "dark_spot", "black", "gray"),
        )
    else:
        colors = "gray"

    # Plot normal points first
    if classification is not None:
        mask_normal = classification == "normal"
        ax.scatter(
            observed[mask_normal] * 100,
            expected[mask_normal] * 100,
            c="gray",
            alpha=0.5,
            s=30,
            label="Normal",
        )

        mask_bright = classification == "bright_spot"
        ax.scatter(
            observed[mask_bright] * 100,
            expected[mask_bright] * 100,
            c="yellow",
            edgecolor="orange",
            s=50,
            label="Bright spot",
        )

        mask_dark = classification == "dark_spot"
        ax.scatter(
            observed[mask_dark] * 100,
            expected[mask_dark] * 100,
            c="black",
            alpha=0.7,
            s=40,
            label="Dark spot",
        )
    else:
        ax.scatter(observed * 100, expected * 100, c="gray", alpha=0.5, s=30)

    # Add 1:1 line
    ax.plot([0, 100], [0, 100], "k--", linewidth=1, label="1:1 line")

    # Add threshold lines
    sd = np.std(observed) * 100
    ax.plot([0, 100], [1.5 * sd, 100 + 1.5 * sd], "r-", linewidth=0.5, alpha=0.5)
    ax.plot([0, 100], [-1.5 * sd, 100 - 1.5 * sd], "r-", linewidth=0.5, alpha=0.5)

    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Observed % coral cover")
    ax.set_ylabel("Expected % coral cover")
    ax.set_title("Observed vs Expected Coral Cover")
    ax.legend(loc="upper left")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_bright_dark_spots_map(
    df: pd.DataFrame,
    classification: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (16, 8),
) -> plt.Figure:
    """
    Create map of bright and dark spots.

    Parameters
    ----------
    df : pd.DataFrame
        Data with latitude and longitude
    classification : np.ndarray
        Bright/dark spot classification
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        has_cartopy = True
    except ImportError:
        has_cartopy = False

    if has_cartopy:
        fig, ax = plt.subplots(
            figsize=figsize,
            subplot_kw={"projection": ccrs.Robinson(central_longitude=150)},
        )

        ax.add_feature(cfeature.LAND, facecolor="lightgreen", edgecolor="darkgreen")
        ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)

        # Plot points
        transform = ccrs.PlateCarree()

        mask_normal = classification == "normal"
        ax.scatter(
            df.loc[mask_normal, "lon"],
            df.loc[mask_normal, "lat"],
            c="gray",
            alpha=0.5,
            s=20,
            transform=transform,
        )

        mask_bright = classification == "bright_spot"
        ax.scatter(
            df.loc[mask_bright, "lon"],
            df.loc[mask_bright, "lat"],
            c="yellow",
            edgecolor="orange",
            s=40,
            transform=transform,
            label="Bright spot",
        )

        mask_dark = classification == "dark_spot"
        ax.scatter(
            df.loc[mask_dark, "lon"],
            df.loc[mask_dark, "lat"],
            c="black",
            alpha=0.7,
            s=30,
            transform=transform,
            label="Dark spot",
        )

        ax.set_global()
        ax.legend(loc="lower left")

    else:
        # Fallback without cartopy
        fig, ax = plt.subplots(figsize=figsize)

        mask_normal = classification == "normal"
        ax.scatter(
            df.loc[mask_normal, "lon"],
            df.loc[mask_normal, "lat"],
            c="gray",
            alpha=0.5,
            s=20,
        )

        mask_bright = classification == "bright_spot"
        ax.scatter(
            df.loc[mask_bright, "lon"],
            df.loc[mask_bright, "lat"],
            c="yellow",
            edgecolor="orange",
            s=40,
            label="Bright spot",
        )

        mask_dark = classification == "dark_spot"
        ax.scatter(
            df.loc[mask_dark, "lon"],
            df.loc[mask_dark, "lat"],
            c="black",
            alpha=0.7,
            s=30,
            label="Dark spot",
        )

        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.legend()

    ax.set_title("Bright and Dark Spots for Coral Reefs")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# CLIMATE SCENARIO ANALYSIS
# =============================================================================


def plot_coral_cover_change_histogram(
    change: np.ndarray,
    current_cover: np.ndarray,
    scenario: str,
    year: int,
    relative: bool = True,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (6, 5),
) -> plt.Figure:
    """
    Plot histogram of coral cover change under climate scenario.

    Parameters
    ----------
    change : np.ndarray
        Change in coral cover
    current_cover : np.ndarray
        Current coral cover values
    scenario : str
        Climate scenario name
    year : int
        Projection year
    relative : bool
        If True, show relative change; if False, show absolute change
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if relative:
        values = 100 * change / current_cover
        values = np.clip(values, -100, 100)
        xlabel = "Relative coral cover change (%)"
    else:
        values = change * 100
        xlabel = "Absolute coral cover change (% points)"

    bins = np.linspace(-100, 0, 11) if relative else np.linspace(-25, 0, 11)

    ax.hist(values, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(f"{scenario.upper()} year {year}")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


def plot_absolute_vs_relative_change(
    current_cover: np.ndarray,
    absolute_change: np.ndarray,
    scenarios: list[tuple[str, int]],
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (10, 10),
) -> plt.Figure:
    """
    Create 2x2 plot of absolute and relative change for two scenarios.

    Parameters
    ----------
    current_cover : np.ndarray
        Current coral cover
    absolute_change : dict
        dictionary mapping (scenario, year) to absolute change arrays
    scenarios : list of tuple
        list of (scenario, year) tuples to plot
    output_path : Path, optional
        Path to save figure
    figsize : tuple
        Figure size

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, (ax, (scenario, year)) in enumerate(zip(axes, scenarios)):
        key = (scenario, year)
        if callable(absolute_change):
            change = absolute_change(scenario, year)
        else:
            change = absolute_change.get(key, np.zeros_like(current_cover))

        # Clip change
        change_clipped = np.clip(change, -1, 0)

        if idx % 2 == 0:
            # Absolute change
            ax.scatter(current_cover * 100, change_clipped * 100, alpha=0.5, s=10)
            ax.set_ylabel("Absolute change in % coral cover")
            ax.set_ylim(-25, 0)
        else:
            # Relative change
            relative = change / current_cover
            relative_clipped = np.clip(relative, -1, 0)
            ax.scatter(current_cover * 100, relative_clipped * 100, alpha=0.5, s=10)
            ax.set_ylabel("Relative change in % coral cover")
            ax.set_ylim(-100, 0)

        ax.set_xlim(0, 100)
        ax.set_xlabel("Modern observed % coral cover")
        ax.set_title(f"{scenario.upper()} year {year}")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    return fig


# =============================================================================
# OCEAN/REGION STATISTICS
# =============================================================================


def calculate_coral_cover_by_ocean(
    df: pd.DataFrame,
    reef_id_col: str = "reef_id",
    ocean_col: str = "ocean",
    cover_col: str = "average_coral_cover",
    date_col: str = "days_since_19811231",
) -> pd.DataFrame:
    """
    Calculate mean coral cover per ocean, avoiding pseudo-replication.

    Uses the most recent survey for each reef.

    Parameters
    ----------
    df : pd.DataFrame
        Coral cover data
    reef_id_col : str
        Column name for reef ID
    ocean_col : str
        Column name for ocean
    cover_col : str
        Column name for coral cover
    date_col : str
        Column name for survey date

    Returns
    -------
    pd.DataFrame
        Summary statistics by ocean
    """
    # Get most recent survey for each reef
    idx = df.groupby(reef_id_col)[date_col].idxmax()
    reef_data = df.loc[idx].copy()

    # Calculate statistics by ocean
    stats = reef_data.groupby(ocean_col)[cover_col].agg(["mean", "std", "count"])
    stats.columns = ["mean_cover", "std_cover", "n_reefs"]

    # Add overall statistics
    overall = pd.DataFrame(
        {
            "mean_cover": [reef_data[cover_col].mean()],
            "std_cover": [reef_data[cover_col].std()],
            "n_reefs": [len(reef_data)],
        },
        index=["Overall"],
    )

    return pd.concat([stats, overall])


# =============================================================================
# MAIN PIPELINE
# =============================================================================


def run_full_analysis(
    data_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    fit_bayesian_model: bool = True,
    save_diagnostics: bool = True,
    n_samples: int = 2000,
    n_tune: int = 1000,
    n_chains: int = 6,
    target_accept: float = 0.95,
    max_treedepth: int = 15,
    random_seed: int = 42,
) -> dict[str, Any]:
    """
    Run the complete coral cover analysis pipeline.

    Parameters
    ----------
    data_path : Path, optional
        Path to data.csv
    output_dir : Path, optional
        Directory for outputs
    fit_bayesian_model : bool
        Whether to fit the full Bayesian model
    save_diagnostics : bool
        Whether to save model diagnostics
    n_samples : int
        Number of posterior samples
    n_tune : int
        Number of tuning samples
    n_chains : int
        Number of chains for MCMC sampling
    target_accept : float
        Target acceptance rate for NUTS sampler
    max_treedepth : int
        Maximum tree depth for NUTS sampler
    random_seed : int
        Random seed for reproducibility

    Returns
    -------
    dict
        dictionary containing all results
    """

    # 0. Create output directory and return path
    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir = create_output_dir_path(output_dir)
    # if output_dir is None:
    #     output_dir = OUTPUT_DIR
    # output_dir = Path(output_dir)
    # output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Load and clean data
    print("Loading data...")
    df = load_data(data_path)
    df = clean_data(df)
    results["n_observations"] = len(df)
    print(f"Loaded {len(df)} observations")

    # 2. Compute and plot correlation matrix
    print("Computing correlation matrix...")
    corr_matrix = compute_correlation_matrix(df)
    results["correlation_matrix"] = corr_matrix
    print("Plotting correlation matrix...")
    plot_correlation_matrix(corr_matrix, output_dir / "corrplot.png")
    print("Saving correlation matrix to CSV...")
    corr_matrix.to_csv(output_dir / "corrplot.csv")

    # 3. Standardize variables
    print("Standardizing variables...")
    vars_to_standardize = [
        "lon",
        "lat",
        "depth",
        "human_pop",
        "cyclone",
        "sst_mean",
        "sst_max",
        "sst_stdev",
        "ssta_min",
        "ssta_max",
        "ssta_mean",
        "ssta_stdev",
        "ssta_freqmax",
        "ssta_freqstdev",
        "ssta_dhwmean",
        "ssta_dhwmax",
        "tsa_min",
        "tsa_max",
        "tsa_mean",
        "tsa_freqstdev",
        "tsa_dhwmean",
        "tsa_dhwmax",
        "tsa_dhwstdev",
        "turbidity_mean",
        "turbidity_max",
        "historical_sst_max",
        "historical_sst_mean",
        "historical_sst_sd",
    ]
    df["lat"] = np.abs(df["lat"])  # Use absolute latitude
    df, std_stats = standardize_variables(df, vars_to_standardize)
    results["standardization_stats"] = std_stats

    # 4. Build design matrix
    print("Building design matrix...")
    predictors = [
        "lat_stzd",
        "depth_stzd",
        "human_pop_stzd",
        "cyclone_stzd",
        "sst_mean_stzd",
        "ssta_mean_stzd",
        "ssta_min_stzd",
        "ssta_freqstdev_stzd",
        "ssta_dhwmax_stzd",
        "tsa_max_stzd",
        "tsa_freqstdev_stzd",
        "turbidity_mean_stzd",
        "historical_sst_max_stzd",
    ]
    # Filter to available predictors
    predictors = [p for p in predictors if p in df.columns]
    X, col_names = build_design_matrix(df, predictors)
    results["col_names"] = col_names

    # Prepare indices
    # CRITICAL: Use R's pre-computed indices from data_for_maps.csv if available
    # R creates region indices from alphabetical order of Ecoregion NAME, not ERG CODE
    # Using different orderings would misalign diversity values with regions!

    if "site" in df.columns and "region" in df.columns:
        # Use R's pre-computed indices (convert to 0-based)
        print("Using R's pre-computed site and region indices from data file")

        # Map R's sparse site indices (1-2949) to dense 0-based (0-2948)
        unique_r_sites = sorted(df["site"].unique())
        r_site_to_py = {r_site: i for i, r_site in enumerate(unique_r_sites)}
        df["site_idx"] = df["site"].map(r_site_to_py)

        # Map R's sparse region indices (2-150 with gaps) to dense 0-based (0-82)
        unique_r_regions = sorted(df["region"].unique())
        r_region_to_py = {r_region: i for i, r_region in enumerate(unique_r_regions)}
        df["region_idx"] = df["region"].map(r_region_to_py)

        site_idx = df["site_idx"].values
        region_idx = df["region_idx"].values

        print(
            f"   Sites: {len(unique_r_sites)} unique (R indices {min(unique_r_sites)}-{max(unique_r_sites)} -> 0-{len(unique_r_sites) - 1})"
        )
        print(
            f"   Regions: {len(unique_r_regions)} unique (R indices {min(unique_r_regions)}-{max(unique_r_regions)} -> 0-{len(unique_r_regions) - 1})"
        )

        # Site to region mapping (using Python's 0-based indices)
        site_region_map = (
            df.groupby("site_idx")["region_idx"].first().sort_index().values
        )

        # Diversity per region (indexed by Python's 0-based region indices)
        if "diversity" in df.columns:
            diversity = (
                df.groupby("region_idx")["diversity"].first().sort_index().values
            )
        else:
            diversity = None
    else:
        # Fallback: Create indices from scratch (may not match R!)
        print("⚠️  WARNING: Using Python-generated indices. May not match R's model!")
        df["site_idx"] = pd.Categorical(df["reef_id"]).codes
        df["region_idx"] = pd.Categorical(df["erg"]).codes if "erg" in df.columns else 0
        site_idx = df["site_idx"].values
        region_idx = df["region_idx"].values
        site_region_map = df.groupby("site_idx")["region_idx"].first().values
        diversity = (
            df.groupby("region_idx")["diversity"].first().values
            if "diversity" in df.columns
            else None
        )

    # Create reef_id to site_idx mapping - CRITICAL for consistent predictions
    reef_to_site_map = dict(zip(df["reef_id"], df["site_idx"]))
    print(f"Created reef_to_site_map with {len(reef_to_site_map)} unique reef_ids")
    results["reef_to_site_map"] = reef_to_site_map

    # Response variable
    n = len(df)
    y = transform_to_beta(df["average_coral_cover"].values, n)

    # 5. Fit model
    if fit_bayesian_model and HAS_PYMC:
        print(
            f"\nFitting Bayesian model (n_samples={n_samples}, n_tune={n_tune}, target_accept={target_accept}, max_treedepth={max_treedepth}, random_seed={random_seed})"
        )
        model = HierarchicalBetaModel()
        model.fit(
            X,
            y,
            site_idx,
            region_idx,
            site_region_map,
            reef_to_site_map=reef_to_site_map,  # CRITICAL for consistent predictions
            diversity=diversity,
            col_names=col_names,
            n_samples=n_samples,
            n_tune=n_tune,
            n_chains=n_chains,
            target_accept=target_accept,
            max_treedepth=max_treedepth,
            random_seed=random_seed,
        )
        # Store standardization stats in model for future use
        model.standardization_stats = std_stats
        results["model"] = model

        # Save model
        model_save_path = output_dir / "trained_model"
        print("Saving trained model...")
        model.save_model(model_save_path)

        # Save coefficient summary
        coef_summary = model.get_coefficient_summary()
        print("Saving coefficient summary...")
        coef_summary.to_csv(output_dir / "beta_est.csv")
        results["coefficient_summary"] = coef_summary

        # Save model diagnostics
        if save_diagnostics:
            print("Saving model diagnostics...")
            model.save_diagnostics(output_dir / "diagnostics")

        # Plot coefficients
        print("Plotting coefficients...")
        model.plot_coefficients(output_dir / "Beta_coeff_plot.png")

        # Predictions
        print("Making predictions...")
        y_pred, y_pred_std = model.predict(X, site_idx)
        y_new = inverse_transform_beta(y_pred, n)
        results["Y_New"] = y_new

    else:
        raise ValueError("Alternative (non-Bayesian) method not yet implemented")
        # print("Fitting simple MLE model...")
        # model = SimpleBetaModel()
        # model.fit(X, y)
        # results['model'] = model

        # y_new = model.predict(X)
        # results['Y_New'] = y_new

    # 6. Identify bright and dark spots
    print("Identifying bright and dark spots...")
    spots_df = identify_bright_dark_spots(df["average_coral_cover"].values, y_new)
    results["spots"] = spots_df

    # Plot observed vs expected
    plot_observed_vs_expected(
        df["average_coral_cover"].values,
        y_new,
        spots_df["classification"].values,
        output_dir / "observed_vs_expected_coral_cover.png",
    )

    # Plot map
    plot_bright_dark_spots_map(
        df, spots_df["classification"].values, output_dir / "bright_dark_spots_map.png"
    )

    # 7. Calculate ocean statistics
    print("Calculating ocean statistics...")
    ocean_stats = calculate_coral_cover_by_ocean(df)
    ocean_stats.to_csv(output_dir / "coral_cover_by_ocean.csv")
    results["ocean_stats"] = ocean_stats

    # 8. Save processed data
    print("Saving results...")
    df["y_new"] = y_new
    df["deviation_from_expected"] = spots_df["deviation_normalized"].values
    df["classification"] = spots_df["classification"].values
    df.to_csv(output_dir / "data_processed.csv", index=False)

    print(f"Analysis complete. Results saved to {output_dir}")

    return results


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run coral cover beta regression analysis"
    )
    parser.add_argument(
        "--data", "-d", type=str, default=None, help="Path to data.csv file"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--save-diagnostics",
        "-sd",
        action="store_true",
        default=False,
        help="Save model diagnostics",
    )
    parser.add_argument(
        "--max-treedepth",
        "-mt",
        type=int,
        default=15,
        help="Maximum tree depth for NUTS sampler",
    )
    parser.add_argument(
        "--target-accept",
        "-ta",
        type=float,
        default=0.9,
        help="Target acceptance rate for NUTS sampler",
    )
    parser.add_argument(
        "--num_chains",
        "-nc",
        type=int,
        default=6,
        help="Number of chains for MCMC sampling",
    )

    parser.add_argument(
        "--samples", "-s", type=int, default=2000, help="Number of posterior samples"
    )
    parser.add_argument(
        "--tune", "-t", type=int, default=1000, help="Number of tuning samples"
    )

    parser.add_argument(
        "--random-seed",
        "-rs",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    data_path = Path(args.data) if args.data else None
    output_dir = Path(args.output) if args.output else None

    results = run_full_analysis(
        data_path=data_path,
        output_dir=output_dir,
        save_diagnostics=args.save_diagnostics,
        n_samples=args.samples,
        n_tune=args.tune,
        n_chains=args.num_chains,
        target_accept=args.target_accept,
        max_treedepth=args.max_treedepth,
        random_seed=args.random_seed,
    )

    print("\nSummary:")
    print(f"  Observations: {results['n_observations']}")
    if "coefficient_summary" in results:
        print("\nCoefficient estimates:")
        print(results["coefficient_summary"][["mean", "hdi_3%", "hdi_97%"]])


# DEPRECATED
# def load_diversity_data(filepath: Optional[Path] = None) -> pd.DataFrame:
#     """
#     Load coral diversity data.

#     Parameters
#     ----------
#     filepath : Path, optional
#         Path to the diversity CSV file.

#     Returns
#     -------
#     pd.DataFrame
#         Diversity data with ecoregion information
#     """
#     if filepath is None:
#         filepath = SULLY_DATA_DIR / "coral_diversity_for_coral_cover.csv"

#     if not filepath.exists():
#         warnings.warn(f"Diversity file not found at {filepath}. Returning empty DataFrame.")
#         return pd.DataFrame()

#     diversity = pd.read_csv(filepath)
#     diversity.columns = ['Ecoregion'] + list(diversity.columns[1:])
#     diversity['Region'] = diversity['Ecoregion']
#     diversity = diversity.sort_values('Ecoregion')

#     return diversity


# # =============================================================================
# # SIMPLIFIED MODEL (Without PyMC - using Maximum Likelihood)
# # =============================================================================

# class SimpleBetaModel:
#     """
#     Simplified Beta Regression using Maximum Likelihood Estimation.

#     This is a fallback when PyMC is not available. Uses scipy for optimization.
#     Does not include hierarchical structure.
#     """

#     def __init__(self):
#         self.coefficients = None
#         self.theta = None
#         self.se = None

#     def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleBetaModel':
#         """
#         Fit beta regression using MLE.

#         Parameters
#         ----------
#         X : np.ndarray
#             Design matrix
#         y : np.ndarray
#             Response (must be in (0, 1))

#         Returns
#         -------
#         self
#         """
#         from scipy.optimize import minimize
#         from scipy.special import gammaln

#         n_predictors = X.shape[1]

#         def neg_log_lik(params):
#             beta = params[:-1]
#             log_theta = params[-1]
#             theta = np.exp(log_theta)

#             eta = X @ beta
#             pi = inv_logit(eta)

#             # Beta log-likelihood
#             alpha = theta * pi
#             beta_param = theta * (1 - pi)

#             ll = (
#                 gammaln(alpha + beta_param) - gammaln(alpha) - gammaln(beta_param)
#                 + (alpha - 1) * np.log(y)
#                 + (beta_param - 1) * np.log(1 - y)
#             )

#             return -np.sum(ll)

#         # Initial values
#         init_params = np.zeros(n_predictors + 1)
#         init_params[-1] = 2.0  # log(theta) ≈ 7.4

#         # Optimize
#         result = minimize(
#             neg_log_lik,
#             init_params,
#             method='L-BFGS-B',
#             options={'maxiter': 1000}
#         )

#         self.coefficients = result.x[:-1]
#         self.theta = np.exp(result.x[-1])

#         return self

#     def predict(self, X_new: np.ndarray, return_mean: bool = True) -> np.ndarray:
#         """
#         Generate predictions.

#         Parameters
#         ----------
#         X_new : np.ndarray
#             New design matrix
#         return_mean : bool
#             If True, return expected value; if False, sample from beta

#         Returns
#         -------
#         np.ndarray
#             Predictions
#         """
#         eta = X_new @ self.coefficients
#         pi = inv_logit(eta)

#         if return_mean:
#             return pi
#         else:
#             alpha = self.theta * pi
#             beta_param = self.theta * (1 - pi)
#             return np.random.beta(alpha, beta_param)
