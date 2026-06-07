"""
Cumulative impact calculations for coral reef economics.

This module provides functions to:
1. Interpolate coral cover trajectories between observation points
2. Calculate cumulative economic losses over time
3. Compare linear vs exponential decline scenarios

The key insight is that annual losses compound over time - a reef that loses
coral cover gradually accumulates economic losses each year, not just at the endpoint.
"""

import json
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .depreciation_models import DepreciationModel, get_model

# Forward reference for type hints
if TYPE_CHECKING:
    pass

# =============================================================================
# TRAJECTORY INTERPOLATION
# =============================================================================


@dataclass
class TrajectoryPoint:
    """A point in time with coral cover value."""

    year: int
    cover: float  # Coral cover as fraction (0-1)


@dataclass
class CoverTrajectory:
    """A trajectory of coral cover over time."""

    years: np.ndarray
    covers: np.ndarray
    scenario: str
    interpolation_method: str

    @property
    def start_year(self) -> int:
        return int(self.years[0])

    @property
    def end_year(self) -> int:
        return int(self.years[-1])

    @property
    def total_change(self) -> float:
        """Total change in coral cover (negative = loss)."""
        return self.covers[-1] - self.covers[0]

    def get_cover_at(self, year: int) -> float:
        """Get interpolated cover at a specific year."""
        return np.interp(year, self.years, self.covers)

    def get_annual_changes(self) -> np.ndarray:
        """Get year-over-year changes in cover."""
        return np.diff(self.covers)


def interpolate_linear(
    baseline_year: int,
    baseline_cover: float,
    points: List[TrajectoryPoint],
    annual_resolution: bool = True,
) -> CoverTrajectory:
    """
    Linear interpolation between observation points.

    Parameters
    ----------
    baseline_year : int
        Starting year (e.g., 2013).
    baseline_cover : float
        Coral cover at baseline (fraction).
    points : list of TrajectoryPoint
        Future observation points (e.g., 2050, 2100).
    annual_resolution : bool
        If True, return annual values. If False, return only observation points.

    Returns
    -------
    CoverTrajectory
        Interpolated trajectory.
    """
    # Build observation points including baseline
    all_years = [baseline_year] + [p.year for p in points]
    all_covers = [baseline_cover] + [p.cover for p in points]

    if annual_resolution:
        # Create annual time series
        years = np.arange(baseline_year, max(all_years) + 1)
        covers = np.interp(years, all_years, all_covers)
    else:
        years = np.array(all_years)
        covers = np.array(all_covers)

    # Determine scenario name from points
    scenario = points[-1].year if points else "unknown"

    return CoverTrajectory(
        years=years,
        covers=covers,
        scenario=f"to_{scenario}",
        interpolation_method="linear",
    )


def interpolate_exponential(
    baseline_year: int,
    baseline_cover: float,
    points: List[TrajectoryPoint],
    annual_resolution: bool = True,
) -> CoverTrajectory:
    """
    Exponential decay interpolation - more realistic for ecosystem decline.

    This models the decline as exponential decay, where the rate of loss
    is proportional to the remaining cover. This is often more realistic
    for ecosystem collapse scenarios.

    Parameters
    ----------
    baseline_year : int
        Starting year (e.g., 2013).
    baseline_cover : float
        Coral cover at baseline (fraction).
    points : list of TrajectoryPoint
        Future observation points (e.g., 2050, 2100).
    annual_resolution : bool
        If True, return annual values.

    Returns
    -------
    CoverTrajectory
        Interpolated trajectory using exponential decay.
    """
    if annual_resolution:
        end_year = max(p.year for p in points)
        years = np.arange(baseline_year, end_year + 1)
    else:
        years = np.array([baseline_year] + [p.year for p in points])

    # Build piecewise exponential decay
    covers = np.zeros_like(years, dtype=float)
    covers[0] = baseline_cover

    # Sort points by year
    sorted_points = sorted(points, key=lambda p: p.year)

    # Calculate decay rate for each segment
    prev_year = baseline_year
    prev_cover = baseline_cover

    for point in sorted_points:
        # Years in this segment
        segment_mask = (years > prev_year) & (years <= point.year)
        segment_years = years[segment_mask]

        if len(segment_years) == 0:
            prev_year = point.year
            prev_cover = point.cover
            continue

        # Calculate decay rate: cover(t) = cover_0 * exp(-lambda * t)
        # Solve for lambda: lambda = -ln(cover_end/cover_start) / dt
        dt = point.year - prev_year

        if prev_cover > 0 and point.cover > 0:
            # Exponential decay
            decay_rate = -np.log(point.cover / prev_cover) / dt
            t_relative = segment_years - prev_year
            covers[segment_mask] = prev_cover * np.exp(-decay_rate * t_relative)
        else:
            # Fall back to linear if cover goes to zero
            covers[segment_mask] = np.interp(
                segment_years, [prev_year, point.year], [prev_cover, point.cover]
            )

        prev_year = point.year
        prev_cover = point.cover

    # Set the exact endpoint values
    for point in sorted_points:
        idx = np.where(years == point.year)[0]
        if len(idx) > 0:
            covers[idx[0]] = point.cover

    scenario = sorted_points[-1].year if sorted_points else "unknown"

    return CoverTrajectory(
        years=years,
        covers=covers,
        scenario=f"to_{scenario}",
        interpolation_method="exponential",
    )


# =============================================================================
# CUMULATIVE IMPACT CALCULATION
# =============================================================================


@dataclass
class CumulativeImpactResult:
    """
    Results from cumulative impact calculation.

    Loss definitions:
    - annual_losses / annual_value_lost: Value lost each year (year-over-year decline)
      This decreases over time as less value remains to lose.
    - annual_opportunity_cost: Opportunity cost (baseline revenue we could have earned)
      This stays high after collapse: baseline_value - curr_value
    - cumulative_losses: Cumulative sum of annual_opportunity_cost
      This represents total revenue lost over time.
    """

    trajectory: CoverTrajectory
    annual_values: np.ndarray  # Economic value each year
    annual_losses: (
        np.ndarray
    )  # Value lost each year (year-over-year decline, for display)
    cumulative_losses: (
        np.ndarray
    )  # Cumulative sum of opportunity cost (annual_lost_revenue)

    baseline_value: float
    model: DepreciationModel
    value_type: str
    discount_rate: float = 0.0

    # Explicit tracking of loss components (optional for backward compatibility)
    annual_value_lost: Optional[np.ndarray] = field(
        default=None
    )  # Year-over-year decline in value (prev_value - curr_value, decreases over time)
    annual_opportunity_cost: Optional[np.ndarray] = field(
        default=None
    )  # Opportunity cost (baseline_value - curr_value, stays high after collapse)

    def __post_init__(self):
        """Ensure new fields are set if not provided (backward compatibility)."""
        if self.annual_value_lost is None:
            # Default: assume all loss is value_lost (no opportunity cost separation)
            self.annual_value_lost = self.annual_losses.copy()
        if self.annual_opportunity_cost is None:
            self.annual_opportunity_cost = np.zeros_like(self.annual_losses)

    @property
    def years(self) -> np.ndarray:
        return self.trajectory.years

    @property
    def total_cumulative_loss(self) -> float:
        """Total cumulative loss over the entire period."""
        return self.cumulative_losses[-1]

    @property
    def annual_loss_at_end(self) -> float:
        """Value lost in the final year (year-over-year decline)."""
        return self.annual_losses[-1]

    @property
    def average_annual_loss(self) -> float:
        """Average annual loss over the period."""
        return self.annual_losses.mean()

    @property
    def years_to_50pct_loss(self) -> Optional[int]:
        """Years until 50% of baseline value is lost (if applicable)."""
        half_value = self.baseline_value * 0.5
        idx = np.where(self.annual_values <= half_value)[0]
        if len(idx) > 0:
            return int(self.years[idx[0]] - self.years[0])
        return None

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for analysis."""
        return pd.DataFrame(
            {
                "year": self.years,
                "coral_cover": self.trajectory.covers,
                "annual_value": self.annual_values,
                "annual_loss": self.annual_losses,
                "annual_value_lost": self.annual_value_lost,
                "annual_opportunity_cost": self.annual_opportunity_cost,
                "cumulative_loss": self.cumulative_losses,
            }
        )

    def save(self, path: Path) -> None:
        """Save cumulative impact result to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save the full object
        with open(path, "wb") as f:
            pickle.dump(self, f)

        # Also save metadata and trajectory data as JSON
        metadata = {
            "scenario": self.trajectory.scenario,
            "interpolation_method": self.trajectory.interpolation_method,
            "model_name": self.model.name,
            "model_type": type(self.model).__name__,
            "value_type": self.value_type,
            "baseline_value": float(self.baseline_value),
            "discount_rate": float(self.discount_rate),
            "start_year": int(self.trajectory.start_year),
            "end_year": int(self.trajectory.end_year),
            "total_cumulative_loss": float(self.total_cumulative_loss),
            "annual_loss_at_end": float(self.annual_loss_at_end),
            "average_annual_loss": float(self.average_annual_loss),
        }

        metadata_path = path.with_suffix(".json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "CumulativeImpactResult":
        """Load cumulative impact result from disk."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cumulative result file not found: {path}")

        with open(path, "rb") as f:
            result = pickle.load(f)

        # Backward compatibility: if old format, compute new fields from annual_losses
        if not hasattr(result, "annual_value_lost") or result.annual_value_lost is None:
            # For old files, we can't perfectly reconstruct, but we can approximate:
            # Assume all loss is value_lost (no opportunity cost separation)
            result.annual_value_lost = result.annual_losses.copy()
            result.annual_opportunity_cost = np.zeros_like(result.annual_losses)

        return result


def calculate_cumulative_impact(
    baseline_cover: float,
    baseline_value: float,
    trajectory: CoverTrajectory,
    model: Union[str, DepreciationModel] = "compound",
    discount_rate: float = 0.0,
    value_type: str = "tourism",
    **model_kwargs,
) -> CumulativeImpactResult:
    """
    Calculate cumulative economic impact over a coral cover trajectory.

    Parameters
    ----------
    baseline_cover : float
        Coral cover at baseline (fraction, 0-1).
    baseline_value : float
        Economic value at baseline (e.g., USD).
    trajectory : CoverTrajectory
        Coral cover trajectory over time.
    model : str or DepreciationModel
        Depreciation model to use. If string, will be instantiated.
    discount_rate : float
        Annual discount rate for NPV calculation (0.0 = no discounting).
    value_type : str
        Type of economic value.
    **model_kwargs
        Additional arguments for model instantiation (only used if model is string).

    Returns
    -------
    CumulativeImpactResult
        Cumulative impact results.

    Notes
    -----
    - For TippingPointModel, original_cc is set to baseline_cover (the starting coral cover level).
      This ensures the tipping point threshold is evaluated relative to the original ecosystem state.
    - All models calculate delta_cc as (current_cover - baseline_cover), so each year's value is
      calculated independently relative to the baseline.
    - Annual losses are separated into:
      * value_lost_this_year: Year-over-year decline (prev_value - curr_value), decreases over time
      * annual_lost_revenue: Opportunity cost (baseline_value - curr_value), stays high after collapse
    - Cumulative loss = cumulative sum of annual_lost_revenue (opportunity cost)
    - Annual loss (for display) = value_lost_this_year (value lost each year)
    - This ensures cumulative loss correctly reflects total revenue lost over time,
      and models with higher endpoint loss_fraction will have higher cumulative loss.
    """
    # Ensure model is instantiated
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)
    elif not isinstance(model, DepreciationModel):
        raise TypeError(f"model must be str or DepreciationModel, got {type(model)}")

    # Vectorized calculation: process all years at once
    covers = trajectory.covers

    # Calculate delta_cc for all years at once (vectorized)
    # delta_cc is negative when cover decreases from baseline
    delta_cc_array = covers - baseline_cover

    # Calculate remaining values for all years using vectorized model calculation
    # Handle tipping point model which requires original_cc parameter
    if model.model_type == "tipping_point":
        threshold = getattr(model, "threshold_cc", 0.1)
        # For tipping point, pass original_cc (baseline) and threshold
        # The model will evaluate collapse based on remaining_cc = original_cc + delta_cc
        annual_values = model.calculate(
            delta_cc_array,
            baseline_value,
            original_cc=baseline_cover,
            threshold=threshold,
        )
    else:
        # Standard models (linear, compound) use simple signature
        annual_values = model.calculate(delta_cc_array, baseline_value)

    # Calculate annual losses with explicit separation:
    # 1. value_lost_this_year: Year-over-year decline in value (decreases as less value remains)
    #    This represents the actual value decline: prev_value - curr_value
    # 2. annual_lost_revenue: Opportunity cost = baseline revenue we could have earned
    #    This represents: baseline_value - curr_value (stays high after collapse)
    #
    # Interpretation:
    # - value_lost_this_year: The actual decline this year (decreases over time)
    # - annual_lost_revenue: The opportunity cost (what we could have earned, stays high)
    # - Cumulative loss = cumulative sum of annual_lost_revenue (opportunity cost)
    # - Annual loss (for display) = value_lost_this_year (value lost each year)
    #
    # This ensures:
    # 1. Cumulative loss = sum of opportunity cost (high and constant after collapse)
    # 2. Annual loss = value lost each year (decreases as less value remains)
    # 3. Models with higher endpoint loss_fraction have higher cumulative loss

    value_lost_this_year = np.zeros_like(annual_values)
    value_lost_this_year[0] = np.maximum(baseline_value - annual_values[0], 0.0)
    if len(annual_values) > 1:
        value_lost_this_year[1:] = np.maximum(
            annual_values[:-1] - annual_values[1:], 0.0
        )
    annual_lost_revenue = np.maximum(baseline_value - annual_values, 0.0)

    # # Apply discounting if specified (the extent to which future years are 'worth less')
    # if discount_rate > 0:
    #     years_from_start = trajectory.years - trajectory.years[0]
    #     discount_factors = (1 + discount_rate) ** (-years_from_start)
    #     annual_losses_discounted = annual_losses * discount_factors
    #     annual_value_lost_discounted = annual_value_lost * discount_factors
    #     annual_opportunity_cost_discounted = annual_opportunity_cost * discount_factors
    # else:
    #     annual_losses_discounted = annual_losses
    #     annual_value_lost_discounted = annual_value_lost
    #     annual_opportunity_cost_discounted = annual_opportunity_cost

    # Apply discounting if specified
    if discount_rate > 0:
        years_from_start = trajectory.years - trajectory.years[0]
        discount_factors = (1 + discount_rate) ** (-years_from_start)
        value_lost_this_year_discounted = value_lost_this_year * discount_factors
        annual_lost_revenue_discounted = annual_lost_revenue * discount_factors
    else:
        value_lost_this_year_discounted = value_lost_this_year
        annual_lost_revenue_discounted = annual_lost_revenue

    # Calculate cumulative losses
    # Cumulative loss = cumulative sum of opportunity cost (annual_lost_revenue)
    cumulative_losses = np.cumsum(annual_lost_revenue_discounted)

    return CumulativeImpactResult(
        trajectory=trajectory,
        annual_values=annual_values,
        annual_losses=value_lost_this_year_discounted,  # Annual loss = value lost each year
        annual_value_lost=value_lost_this_year_discounted,  # Year-over-year decline
        annual_opportunity_cost=annual_lost_revenue_discounted,  # Opportunity cost (baseline revenue lost)
        cumulative_losses=cumulative_losses,  # Cumulative sum of opportunity cost
        baseline_value=baseline_value,
        model=model,
        value_type=value_type,
        discount_rate=discount_rate,
    )


def calculate_cumulative_impacts_multi_scenario(
    baseline_year: int,
    baseline_cover: float,
    baseline_value: float,
    scenario_endpoints: Dict[str, Dict[int, float]],
    model: Union[str, DepreciationModel] = "compound",
    interpolation_methods: List[str] = None,
    discount_rate: float = 0.0,
    value_type: str = "tourism",
    **model_kwargs,
) -> Dict[str, CumulativeImpactResult]:
    """
    Calculate cumulative impacts for multiple scenarios and interpolation methods.

    Parameters
    ----------
    baseline_year : int
        Starting year.
    baseline_cover : float
        Coral cover at baseline.
    baseline_value : float
        Economic value at baseline.
    scenario_endpoints : dict
        Mapping of scenario name to {year: cover} dict.
        E.g., {"RCP45": {2050: 0.28, 2100: 0.25}, "RCP85": {2050: 0.25, 2100: 0.18}}
    model : str or DepreciationModel
        Depreciation model. If string, will be instantiated with model_kwargs.
    interpolation_methods : list
        List of interpolation methods to use. Default: ["linear", "exponential"]
    discount_rate : float
        Discount rate for NPV.
    value_type : str
        Type of economic value.
    **model_kwargs
        Additional arguments for model instantiation (only used if model is string).

    Returns
    -------
    dict
        Mapping of "{scenario}_{method}" to CumulativeImpactResult.

    Notes
    -----
    - Each scenario/model combination creates a separate CumulativeImpactResult.
    - Models are instantiated once per call (not per scenario), ensuring consistency.
    """
    if interpolation_methods is None:
        interpolation_methods = ["linear", "exponential"]

    # Instantiate model once (ensures consistency across scenarios)
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)
    elif not isinstance(model, DepreciationModel):
        raise TypeError(f"model must be str or DepreciationModel, got {type(model)}")

    results = {}

    for scenario_name, endpoints in scenario_endpoints.items():
        # Convert endpoints to TrajectoryPoints
        points = [
            TrajectoryPoint(year=year, cover=cover)
            for year, cover in sorted(endpoints.items())
        ]

        for method in interpolation_methods:
            # Generate trajectory
            if method == "linear":
                trajectory = interpolate_linear(baseline_year, baseline_cover, points)
            elif method == "exponential":
                trajectory = interpolate_exponential(
                    baseline_year, baseline_cover, points
                )
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            trajectory.scenario = scenario_name
            trajectory.interpolation_method = method

            # Calculate cumulative impact
            result = calculate_cumulative_impact(
                baseline_cover=baseline_cover,
                baseline_value=baseline_value,
                trajectory=trajectory,
                model=model,  # Pass the instantiated model
                discount_rate=discount_rate,
                value_type=value_type,
            )

            key = f"{scenario_name}_{method}"
            results[key] = result

    return results


def calculate_cumulative_impacts_multi_scenario_per_site(
    baseline_year: int,
    baseline_covers: np.ndarray,
    baseline_values: np.ndarray,
    scenario_endpoints_per_site: Dict[str, Dict[int, np.ndarray]],
    model: Union[str, DepreciationModel] = "compound",
    interpolation_methods: List[str] = None,
    discount_rate: float = 0.0,
    value_type: str = "tourism",
    verbose: bool = False,
    n_jobs: Union[int, str, None] = "auto",
    chunk_size: int = 25000,
    **model_kwargs,
) -> Dict[str, CumulativeImpactResult]:
    """
    Calculate cumulative impacts per-site, then aggregate.

    **CRITICAL FOR TIPPING POINT MODELS**: Each site uses its own baseline_cover
    as original_cc, which determines when it crosses the threshold.

    Logic:
    1. For each site individually:
       - Use site's own baseline_cover as original_cc (for tipping point models)
       - Use site's own baseline_value
       - Use site's own projected_cover for each scenario/year
       - Generate site-specific trajectory
       - Calculate site's cumulative impact with its own original_cc
    2. Aggregate all site results:
       - Sum annual_values across sites
       - Sum annual_losses across sites
       - Sum annual_value_lost across sites
       - Sum annual_opportunity_cost across sites
       - Sum cumulative_losses across sites
    3. Create aggregated CumulativeImpactResult with combined trajectory
       (using mean covers for trajectory, but summed values)

    Parameters
    ----------
    baseline_year : int
        Starting year.
    baseline_covers : np.ndarray
        Per-site baseline coral covers (fraction, 0-1).
    baseline_values : np.ndarray
        Per-site baseline economic values.
    scenario_endpoints_per_site : dict
        Mapping of scenario name to {year: array of covers} dict.
        E.g., {"rcp45": {2050: array([0.28, 0.30, ...]), 2100: array([0.25, 0.27, ...])}}
        Each array has one value per site.
    model : str or DepreciationModel
        Depreciation model. If string, will be instantiated with model_kwargs.
    interpolation_methods : list
        List of interpolation methods to use. Default: ["linear", "exponential"]
    discount_rate : float
        Discount rate for NPV.
    value_type : str
        Type of economic value.
    verbose : bool
        Print progress information.
    **model_kwargs
        Additional arguments for model instantiation (only used if model is string).

    Returns
    -------
    dict
        Mapping of "{scenario}_{method}" to CumulativeImpactResult.
        Each result represents the aggregated (summed) impact across all sites.

    Notes
    -----
    - For TippingPointModel, each site uses its own baseline_cover as original_cc.
      This ensures sites with different initial covers tip at different times.
    - Sites with high original_cc (e.g., 70%) collapse later
    - Sites with low original_cc (e.g., 15%) collapse early
    - Total cumulative loss = sum of all these different trajectories
    """
    if interpolation_methods is None:
        interpolation_methods = ["linear", "exponential"]
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    # Instantiate model once (ensures consistency across scenarios)
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)
    elif not isinstance(model, DepreciationModel):
        raise TypeError(f"model must be str or DepreciationModel, got {type(model)}")

    # Validate inputs
    n_sites = len(baseline_covers)
    if len(baseline_values) != n_sites:
        raise ValueError(
            f"baseline_covers ({len(baseline_covers)}) and "
            f"baseline_values ({len(baseline_values)}) must have same length"
        )

    # Validate scenario_endpoints_per_site structure
    for rcp, endpoints in scenario_endpoints_per_site.items():
        for year, covers_array in endpoints.items():
            if len(covers_array) != n_sites:
                raise ValueError(
                    f"scenario_endpoints_per_site['{rcp}'][{year}] has length {len(covers_array)}, "
                    f"expected {n_sites}"
                )

    total_baseline_value = baseline_values.sum()

    def _compute_cover_matrix(
        chunk_baseline_covers: np.ndarray,
        endpoint_years: List[int],
        endpoint_cover_arrays: List[np.ndarray],
        years: np.ndarray,
        method: str,
    ) -> np.ndarray:
        row_count = len(chunk_baseline_covers)
        cover_matrix = np.empty((row_count, len(years)), dtype=float)

        prev_year = baseline_year
        prev_covers = chunk_baseline_covers
        first_segment = True

        for year, covers in zip(endpoint_years, endpoint_cover_arrays):
            current_covers = covers
            if first_segment:
                seg_mask = (years >= prev_year) & (years <= year)
            else:
                seg_mask = (years > prev_year) & (years <= year)
            seg_years = years[seg_mask]
            if len(seg_years) == 0:
                prev_year = year
                prev_covers = current_covers
                first_segment = False
                continue

            if method == "linear":
                t = (seg_years - prev_year) / (year - prev_year)
                segment_vals = prev_covers[:, None] + (
                    current_covers - prev_covers
                )[:, None] * t[None, :]
            elif method == "exponential":
                dt = year - prev_year
                t_rel = seg_years - prev_year
                can_exp = (prev_covers > 0) & (current_covers > 0)
                ratio = np.ones_like(prev_covers, dtype=float)
                ratio[can_exp] = current_covers[can_exp] / prev_covers[can_exp]
                decay = np.zeros_like(prev_covers, dtype=float)
                decay[can_exp] = -np.log(ratio[can_exp]) / dt
                exp_vals = prev_covers[:, None] * np.exp(-decay[:, None] * t_rel[None, :])
                lin_t = t_rel / dt
                lin_vals = prev_covers[:, None] + (
                    current_covers - prev_covers
                )[:, None] * lin_t[None, :]
                segment_vals = np.where(can_exp[:, None], exp_vals, lin_vals)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")

            cover_matrix[:, seg_mask] = segment_vals
            prev_year = year
            prev_covers = current_covers
            first_segment = False

        return cover_matrix

    def _compute_one_task(
        scenario_name: str,
        method: str,
        sorted_endpoints: List[Tuple[int, np.ndarray]],
    ) -> Tuple[str, Optional[CumulativeImpactResult], int]:
        if verbose:
            print(f"      Calculating {scenario_name} ({method}) for {n_sites} sites...")

        if not sorted_endpoints:
            return f"{scenario_name}_{method}", None, 0

        endpoint_years = [year for year, _ in sorted_endpoints]
        years = np.arange(baseline_year, max(endpoint_years) + 1)
        n_years = len(years)
        discount_factors = None
        if discount_rate > 0:
            years_from_start = years - years[0]
            discount_factors = (1 + discount_rate) ** (-years_from_start)

        aggregated_annual_values = np.zeros(n_years, dtype=float)
        aggregated_annual_value_lost = np.zeros(n_years, dtype=float)
        aggregated_annual_opportunity_cost = np.zeros(n_years, dtype=float)
        aggregated_trajectory_covers_sum = np.zeros(n_years, dtype=float)
        sites_processed = 0

        for start in range(0, n_sites, chunk_size):
            end = min(start + chunk_size, n_sites)
            chunk_baseline_covers = baseline_covers[start:end]
            chunk_baseline_values = baseline_values[start:end]
            chunk_endpoint_cover_arrays = [covers[start:end] for _, covers in sorted_endpoints]

            cover_matrix = _compute_cover_matrix(
                chunk_baseline_covers=chunk_baseline_covers,
                endpoint_years=endpoint_years,
                endpoint_cover_arrays=chunk_endpoint_cover_arrays,
                years=years,
                method=method,
            )

            delta_cc_matrix = cover_matrix - chunk_baseline_covers[:, None]
            if model.model_type == "tipping_point":
                threshold = getattr(model, "threshold_cc", 0.1)
                annual_values_chunk = model.calculate(
                    delta_cc_matrix,
                    chunk_baseline_values[:, None],
                    original_cc=chunk_baseline_covers[:, None],
                    threshold=threshold,
                )
            else:
                annual_values_chunk = model.calculate(
                    delta_cc_matrix, chunk_baseline_values[:, None]
                )

            annual_value_lost_chunk = np.zeros_like(annual_values_chunk)
            annual_value_lost_chunk[:, 0] = np.maximum(
                chunk_baseline_values - annual_values_chunk[:, 0], 0.0
            )
            if n_years > 1:
                annual_value_lost_chunk[:, 1:] = np.maximum(
                    annual_values_chunk[:, :-1] - annual_values_chunk[:, 1:], 0.0
                )

            annual_opportunity_cost_chunk = np.maximum(
                chunk_baseline_values[:, None] - annual_values_chunk, 0.0
            )

            if discount_factors is not None:
                annual_value_lost_chunk *= discount_factors[None, :]
                annual_opportunity_cost_chunk *= discount_factors[None, :]

            aggregated_annual_values += annual_values_chunk.sum(axis=0)
            aggregated_annual_value_lost += annual_value_lost_chunk.sum(axis=0)
            aggregated_annual_opportunity_cost += annual_opportunity_cost_chunk.sum(axis=0)
            aggregated_trajectory_covers_sum += cover_matrix.sum(axis=0)
            sites_processed += len(chunk_baseline_covers)

        if sites_processed == 0:
            return f"{scenario_name}_{method}", None, 0

        aggregated_trajectory_covers = aggregated_trajectory_covers_sum / sites_processed
        aggregated_cumulative_losses = np.cumsum(aggregated_annual_opportunity_cost)
        aggregated_trajectory = CoverTrajectory(
            years=years,
            covers=aggregated_trajectory_covers,
            scenario=scenario_name,
            interpolation_method=method,
        )
        result = CumulativeImpactResult(
            trajectory=aggregated_trajectory,
            annual_values=aggregated_annual_values,
            annual_losses=aggregated_annual_value_lost,
            annual_value_lost=aggregated_annual_value_lost,
            annual_opportunity_cost=aggregated_annual_opportunity_cost,
            cumulative_losses=aggregated_cumulative_losses,
            baseline_value=total_baseline_value,
            model=model,
            value_type=value_type,
            discount_rate=discount_rate,
        )
        return f"{scenario_name}_{method}", result, sites_processed

    tasks: List[Tuple[str, str, List[Tuple[int, np.ndarray]]]] = []
    for scenario_name, endpoints_dict in scenario_endpoints_per_site.items():
        sorted_endpoints = sorted(endpoints_dict.items())
        for method in interpolation_methods:
            tasks.append((scenario_name, method, sorted_endpoints))

    if n_jobs in (None, "auto"):
        max_workers = min(len(tasks), max(1, os.cpu_count() or 1))
    else:
        max_workers = max(1, int(n_jobs))
    max_workers = min(max_workers, len(tasks)) if tasks else 1

    if n_sites >= 200000:
        max_workers = 1

    results = {}
    if max_workers == 1 or len(tasks) <= 1:
        for scenario_name, method, sorted_endpoints in tasks:
            key, result, sites_processed = _compute_one_task(
                scenario_name, method, sorted_endpoints
            )
            if result is None:
                if verbose:
                    print(
                        f"        ⚠️  No valid sites processed for {scenario_name} ({method})"
                    )
                continue
            results[key] = result
            if verbose:
                print(
                    f"        ✓ Processed {sites_processed} sites, "
                    f"cumulative loss: ${result.total_cumulative_loss / 1e12:.2f}T"
                )
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_compute_one_task, scenario_name, method, sorted_endpoints): (
                    scenario_name,
                    method,
                )
                for scenario_name, method, sorted_endpoints in tasks
            }
            for future in as_completed(futures):
                scenario_name, method = futures[future]
                key, result, sites_processed = future.result()
                if result is None:
                    if verbose:
                        print(
                            f"        ⚠️  No valid sites processed for {scenario_name} ({method})"
                        )
                    continue
                results[key] = result
                if verbose:
                    print(
                        f"        ✓ Processed {sites_processed} sites, "
                        f"cumulative loss: ${result.total_cumulative_loss / 1e12:.2f}T"
                    )

    return results


# =============================================================================
# SUMMARY STATISTICS
# =============================================================================


def summarize_cumulative_impacts(
    results: Dict[str, CumulativeImpactResult],
) -> pd.DataFrame:
    """
    Generate summary table comparing cumulative impacts across scenarios.

    Parameters
    ----------
    results : dict
        Mapping of scenario names to CumulativeImpactResult.

    Returns
    -------
    DataFrame
        Summary statistics for each scenario.
    """
    rows = []

    for key, result in results.items():
        rows.append(
            {
                "scenario": result.trajectory.scenario,
                "interpolation": result.trajectory.interpolation_method,
                "model": result.model.name,
                "start_year": result.trajectory.start_year,
                "end_year": result.trajectory.end_year,
                "baseline_cover": result.trajectory.covers[0],
                "final_cover": result.trajectory.covers[-1],
                "cover_change_pp": (
                    result.trajectory.covers[-1] - result.trajectory.covers[0]
                )
                * 100,
                "baseline_value": result.baseline_value,
                "final_annual_value": result.annual_values[-1],
                "annual_loss_at_end": result.annual_loss_at_end,
                "annual_loss_pct_at_end": 100
                * result.annual_loss_at_end
                / result.baseline_value,
                "total_cumulative_loss": result.total_cumulative_loss,
                "average_annual_loss": result.average_annual_loss,
                "discount_rate": result.discount_rate,
            }
        )

    return pd.DataFrame(rows)


def print_cumulative_summary(
    results: Dict[str, CumulativeImpactResult],
    # top_n: int = 5,
) -> None:
    """Print formatted summary of cumulative impacts."""
    summarize_cumulative_impacts(results)

    print(f"\n{'=' * 70}")
    print("CUMULATIVE IMPACT SUMMARY")
    print(f"{'=' * 70}")

    # Sort results for consistent output: by value_type, then scenario, then model, then interpolation
    sorted_items = sorted(
        results.items(),
        key=lambda x: (
            x[1].value_type or "",
            x[1].trajectory.scenario,
            x[1].model.name,
            x[1].trajectory.interpolation_method,
        ),
    )

    for key, result in sorted_items:
        traj = result.trajectory
        # Extract value type from key if available, otherwise use result.value_type
        # Keys are formatted as: "{value_type}_{scenario}_{method}_{model.name}"
        value_type_display = (
            result.value_type.title() if result.value_type else "Unknown"
        )

        print(
            f"\n📈 {value_type_display} - {traj.scenario} ({traj.interpolation_method.title()} Interpolation)"
        )
        print(f"   Period: {traj.start_year} → {traj.end_year}")
        print(f"   Model: {result.model.name}")
        print(f"   Discount rate: {result.discount_rate * 100:.1f}%")
        print("\n   Coral Cover:")
        print(f"      Baseline: {traj.covers[0] * 100:.1f}%")
        print(f"      Final: {traj.covers[-1] * 100:.1f}%")
        print(f"      Change: {traj.total_change * 100:+.1f} pp")
        print("\n   Economic Impact:")
        print(f"      Baseline value: ${result.baseline_value / 1e9:.2f}B/year")
        print(f"      Final annual value: ${result.annual_values[-1] / 1e9:.2f}B/year")
        print(
            f"      Annual loss at end: ${result.annual_loss_at_end / 1e9:.2f}B/year ({100 * result.annual_loss_at_end / result.baseline_value:.1f}%)"
        )
        print(f"\n   Cumulative Impact ({traj.end_year - traj.start_year} years):")
        print(
            f"      Total cumulative loss: ${result.total_cumulative_loss / 1e12:.2f}T"
        )
        print(
            f"      Average annual loss: ${result.average_annual_loss / 1e9:.2f}B/year"
        )

        if result.years_to_50pct_loss:
            print(f"      Years to 50% value loss: {result.years_to_50pct_loss}")

    print(f"\n{'=' * 70}\n")


# =============================================================================
# SPATIAL CUMULATIVE IMPACT CALCULATION
# =============================================================================


def add_spatial_cumulative_losses(
    result,
    baseline_year: int = 2013,
    discount_rate: float = 0.0,
    interpolation_method: str = "linear",
):
    """
    Calculate and add spatial cumulative loss columns to a DepreciationResult's gdf.

    This function computes cumulative losses for each polygon based on its own
    baseline cover, baseline value, and coral cover trajectory. This provides
    true spatial cumulative losses rather than proportionally distributed global totals.

    **REFACTORED**: Now uses vectorized operations for efficiency instead of
    looping through each polygon individually.

    Parameters
    ----------
    result : DepreciationResult
        Result containing gdf with original_value, coral_change, and baseline cover.
    baseline_year : int
        Starting year for cumulative calculations (default: 2013).
    discount_rate : float
        Annual discount rate for NPV calculation (0.0 = no discounting).
    interpolation_method : str
        Interpolation method for trajectory ("linear" or "exponential").

    Returns
    -------
    DepreciationResult
        Result with added cumulative loss columns in gdf.
        Columns added: cumulative_loss_{scenario}_{model_name}
        e.g., "cumulative_loss_rcp45_2050_Linear_3-81pct_pp"

    Notes
    -----
    - Vectorized implementation processes all polygons simultaneously.
    - Missing or invalid data (NaN, zero values) are handled gracefully.
    - For TippingPointModel, original_cc is set to each polygon's baseline_cover.
    """
    # Import here to avoid circular dependency
    from .analysis import DepreciationResult

    gdf = result.gdf.copy()

    # Find baseline coral cover column
    baseline_cover_col = None
    for col in ["nearest_average_coral_cover", "average_coral_cover", "nearest_y_new"]:
        if col in gdf.columns:
            baseline_cover_col = col
            break

    if baseline_cover_col is None:
        raise ValueError(
            "Baseline coral cover column not found. "
            "Expected 'nearest_average_coral_cover', 'average_coral_cover', or 'nearest_y_new'"
        )

    # Parse scenario from result to determine RCP and target year
    scenario_lower = result.scenario.lower()
    if "rcp45" in scenario_lower:
        rcp = "rcp45"
    elif "rcp85" in scenario_lower:
        rcp = "rcp85"
    else:
        raise ValueError(f"Could not determine RCP from scenario: {result.scenario}")

    if "2050" in scenario_lower:
        target_year = 2050
    elif "2100" in scenario_lower:
        target_year = 2100
    else:
        raise ValueError(
            f"Could not determine target year from scenario: {result.scenario}"
        )

    # Find projected coral cover column (try multiple naming conventions)
    proj_col = None
    possible_names = [
        f"nearest_Y_future_{rcp.upper()}_yr_{target_year}",
        f"nearest_y_future_{rcp.lower()}_yr_{target_year}",
        f"nearest_Y_future_{rcp.upper()}_{target_year}",
        f"nearest_y_future_{rcp.lower()}_{target_year}",
    ]

    for name in possible_names:
        if name in gdf.columns:
            proj_col = name
            break

    if proj_col is None:
        # Try to find any column that matches the pattern
        matching_cols = [
            c
            for c in gdf.columns
            if f"{rcp.lower()}" in c.lower()
            and str(target_year) in c
            and "future" in c.lower()
        ]
        if matching_cols:
            proj_col = matching_cols[0]
        else:
            raise ValueError(
                f"Projected coral cover column not found for {rcp} {target_year}. "
                f"Tried: {possible_names}. "
                f"Available columns containing '{rcp}': {[c for c in gdf.columns if rcp.lower() in c.lower()]}"
            )

    # Get model instance
    model = result.model
    model_name = model.name

    # Column name for cumulative loss (sanitize model name for column)
    # Replace special characters that might cause issues
    safe_model_name = (
        model_name.replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("%", "pct")
        .replace("/", "_")
        .replace(".", "-")
    )
    cum_col = f"cumulative_loss_{rcp}_{target_year}_{safe_model_name}"

    # Extract arrays (vectorized)
    baseline_covers = gdf[baseline_cover_col].fillna(0).values
    projected_covers = gdf[proj_col].fillna(0).values
    baseline_values = gdf["original_value"].fillna(0).values

    # Create mask for valid data
    valid_mask = (
        ~np.isnan(baseline_covers)
        & ~np.isnan(projected_covers)
        & (baseline_values > 0)
        & (baseline_covers > 0)
    )

    # Initialize output array
    cumulative_losses = np.zeros(len(gdf), dtype=float)

    # Process valid polygons in a batched vectorized path
    if np.any(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        valid_baseline_covers = baseline_covers[valid_indices]
        valid_projected_covers = projected_covers[valid_indices]
        valid_baseline_values = baseline_values[valid_indices]
        years = np.arange(baseline_year, target_year + 1)
        row_count = len(valid_indices)

        if target_year <= baseline_year:
            cover_matrix = valid_baseline_covers[:, None]
        else:
            dt = target_year - baseline_year
            t_rel = years - baseline_year

            if interpolation_method == "linear":
                t = t_rel / dt
                cover_matrix = valid_baseline_covers[:, None] + (
                    valid_projected_covers - valid_baseline_covers
                )[:, None] * t[None, :]
            elif interpolation_method == "exponential":
                # Use exponential interpolation when both endpoints are positive.
                # Fall back to linear interpolation for zero/non-positive covers.
                can_exp = (valid_baseline_covers > 0) & (valid_projected_covers > 0)
                ratio = np.ones(row_count, dtype=float)
                ratio[can_exp] = (
                    valid_projected_covers[can_exp] / valid_baseline_covers[can_exp]
                )
                decay = np.zeros(row_count, dtype=float)
                decay[can_exp] = -np.log(ratio[can_exp]) / dt
                exp_vals = valid_baseline_covers[:, None] * np.exp(
                    -decay[:, None] * t_rel[None, :]
                )

                t = t_rel / dt
                lin_vals = valid_baseline_covers[:, None] + (
                    valid_projected_covers - valid_baseline_covers
                )[:, None] * t[None, :]
                cover_matrix = np.where(can_exp[:, None], exp_vals, lin_vals)
            else:
                raise ValueError(
                    f"Unknown interpolation method: {interpolation_method}"
                )

        # Calculate annual value trajectory for each polygon/year.
        delta_cc_matrix = cover_matrix - valid_baseline_covers[:, None]
        if model.model_type == "tipping_point":
            threshold = getattr(model, "threshold_cc", 0.1)
            annual_values = model.calculate(
                delta_cc_matrix,
                valid_baseline_values[:, None],
                original_cc=valid_baseline_covers[:, None],
                threshold=threshold,
            )
        else:
            annual_values = model.calculate(
                delta_cc_matrix, valid_baseline_values[:, None]
            )

        annual_opportunity_cost = np.maximum(
            valid_baseline_values[:, None] - annual_values, 0.0
        )
        if discount_rate > 0:
            years_from_start = years - years[0]
            discount_factors = (1 + discount_rate) ** (-years_from_start)
            annual_opportunity_cost *= discount_factors[None, :]

        # Keep behavior aligned with calculate_cumulative_impact:
        # cumulative loss = cumulative sum of annual opportunity cost.
        cumulative_matrix = np.cumsum(annual_opportunity_cost, axis=1)
        cumulative_losses[valid_indices] = cumulative_matrix[:, -1]

    # Add column to gdf
    gdf[cum_col] = cumulative_losses

    # Also add cumulative loss fraction
    cum_fraction_col = f"cumulative_loss_fraction_{rcp}_{target_year}_{safe_model_name}"
    with np.errstate(divide="ignore", invalid="ignore"):
        gdf[cum_fraction_col] = np.where(
            baseline_values > 0,
            cumulative_losses / baseline_values,
            0.0,
        )

    # Create new result with updated gdf
    updated_result = DepreciationResult(
        gdf=gdf,
        scenario=result.scenario,
        model=result.model,
        value_type=result.value_type,
    )

    return updated_result
