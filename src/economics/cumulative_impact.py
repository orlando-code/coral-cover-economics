"""
Cumulative impact calculations for coral reef economics.

This module provides functions to:
1. Interpolate coral cover trajectories between observation points
2. Calculate cumulative economic losses over time
3. Compare linear vs exponential decline scenarios

The key insight is that annual losses compound over time - a reef that loses
coral cover gradually accumulates economic losses each year, not just at the endpoint.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .depreciation_models import DepreciationModel, get_model

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
        Starting year (e.g., 2017).
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
        Starting year (e.g., 2017).
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
    """Results from cumulative impact calculation."""

    trajectory: CoverTrajectory
    annual_values: np.ndarray  # Economic value each year
    annual_losses: np.ndarray  # Loss compared to baseline each year
    cumulative_losses: np.ndarray  # Running total of losses

    baseline_value: float
    model: DepreciationModel
    value_type: str
    discount_rate: float = 0.0

    @property
    def years(self) -> np.ndarray:
        return self.trajectory.years

    @property
    def total_cumulative_loss(self) -> float:
        """Total cumulative loss over the entire period."""
        return self.cumulative_losses[-1]

    @property
    def annual_loss_at_end(self) -> float:
        """Annual loss rate at the end of the trajectory."""
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
                "cumulative_loss": self.cumulative_losses,
            }
        )


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
        Depreciation model to use.
    discount_rate : float
        Annual discount rate for NPV calculation (0.0 = no discounting).
    value_type : str
        Type of economic value.
    **model_kwargs
        Additional arguments for model instantiation.

    Returns
    -------
    CumulativeImpactResult
        Cumulative impact results.
    """
    if isinstance(model, str):
        model = get_model(model, **model_kwargs)

    n_years = len(trajectory.years)
    annual_values = np.zeros(n_years)
    annual_losses = np.zeros(n_years)

    # Calculate value and loss for each year
    for i, (year, cover) in enumerate(zip(trajectory.years, trajectory.covers)):
        # Change in coral cover from baseline
        delta_cc = cover - baseline_cover  # Negative if cover decreased

        # Remaining value after depreciation
        remaining = model.calculate(delta_cc, baseline_value)
        annual_values[i] = remaining

        # Loss compared to baseline
        annual_losses[i] = baseline_value - remaining

    # Apply discounting if specified
    if discount_rate > 0:
        years_from_start = trajectory.years - trajectory.years[0]
        discount_factors = (1 + discount_rate) ** (-years_from_start)
        annual_losses_discounted = annual_losses * discount_factors
    else:
        annual_losses_discounted = annual_losses

    # Calculate cumulative losses
    cumulative_losses = np.cumsum(annual_losses_discounted)

    return CumulativeImpactResult(
        trajectory=trajectory,
        annual_values=annual_values,
        annual_losses=annual_losses_discounted,
        cumulative_losses=cumulative_losses,
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
        Depreciation model.
    interpolation_methods : list
        List of interpolation methods to use. Default: ["linear", "exponential"]
    discount_rate : float
        Discount rate for NPV.
    value_type : str
        Type of economic value.

    Returns
    -------
    dict
        Mapping of "{scenario}_{method}" to CumulativeImpactResult.
    """
    if interpolation_methods is None:
        interpolation_methods = ["linear", "exponential"]

    if isinstance(model, str):
        model = get_model(model, **model_kwargs)

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
                model=model,
                discount_rate=discount_rate,
                value_type=value_type,
            )

            key = f"{scenario_name}_{method}"
            results[key] = result

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

    for _, result in results.items():
        traj = result.trajectory
        print(
            f"\n📈 {traj.scenario} ({traj.interpolation_method.title()} Interpolation)"
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
