"""
Pluggable depreciation models for mapping coral cover change to economic value loss.

All models follow the convention:
- Input: delta_cc (change in coral cover, as proportion, NEGATIVE for decrease)
- Input: value (current economic value, e.g., USD)
- Output: remaining_value (value after depreciation)

The models are designed to be easily swappable and comparable.

Reference:
- Chen et al. (2014): "3.81% value decrease per 1% coral cover decrease"
  DOI: 10.1016/j.gloenvcha.2014.10.011
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np


@dataclass
class DepreciationModel(ABC):
    """
    Abstract base class for depreciation models: populate as necessary.

    All models must implement:
    - calculate(delta_cc, value) -> remaining_value
    - name: human-readable name
    - description: model description
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable model name."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """Model description for documentation."""
        pass

    @abstractmethod
    def calculate(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate remaining value after coral cover change.

        Parameters
        ----------
        delta_cc : float or array
            Change in coral cover as a proportion (e.g., -0.10 = 10pp decrease).
            NEGATIVE values indicate decrease.
        value : float or array
            Current economic value (e.g., USD).

        Returns
        -------
        float or array
            Remaining value after depreciation. Always >= 0.
        """
        pass

    def calculate_loss(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate value LOST (original - remaining).

        Returns
        -------
        float or array
            Value lost due to coral cover change.
        """
        return value - self.calculate(delta_cc, value)

    def calculate_loss_fraction(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Calculate fraction of value lost (0 to 1).

        Returns
        -------
        float or array
            Fraction of value lost (loss / original_value).
        """
        remaining = self.calculate(delta_cc, value)
        # Avoid division by zero
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = 1 - (remaining / value)
            fraction = np.where(value == 0, 0, fraction)
        return fraction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class LinearModel(DepreciationModel):
    """
    Linear depreciation: value decreases proportionally to coral cover loss.

    Formula: remaining = value * (1 + rate_per_percent * delta_cc * 100)

    Default: 3.81% value loss per 1 percentage point coral cover decrease.

    Note: delta_cc is negative for decreases, so the formula adds a negative term.
    """

    rate_per_percent: float = 0.0381  # 3.81% value loss per 1pp cover decrease

    @property
    def name(self) -> str:
        return f"Linear ({self.rate_per_percent * 100:.2f}%/pp)"

    @property
    def description(self) -> str:
        return (
            f"Linear depreciation: {self.rate_per_percent * 100:.2f}% value loss "
            f"per 1 percentage point coral cover decrease. "
            f"Based on Chen et al. (2014)."
        )

    def calculate(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # delta_cc is negative for decrease
        # E.g., delta_cc = -0.10 means 10pp decrease
        # Loss = value * rate * |delta_cc| * 100 = value * 0.0381 * 10 = 38.1% loss

        # Convert proportion to percentage points
        delta_cc_pp = delta_cc * 100  # e.g., -0.10 -> -10

        # Calculate remaining value
        # delta_cc_pp is negative for decrease, so rate_per_percent * delta_cc_pp is negative
        # remaining = value * (1 + negative_number) = value * (1 - loss_fraction)
        remaining = value * (1 + self.rate_per_percent * delta_cc_pp)

        return np.maximum(remaining, 0)  # hard threshold at zero value


@dataclass
class CompoundModel(DepreciationModel):
    """
    Compound depreciation: each percentage point of coral cover loss compounds.

    Formula: remaining = value * (1 - rate_per_percent) ^ |delta_cc * 100|

    This models diminishing marginal value: the first 10% loss is more impactful
    than the last 10%.

    Default: 3.81% compounded loss per 1 percentage point coral cover decrease.
    """

    rate_per_percent: float = 0.0381  # 3.81% compounded per 1pp cover decrease

    @property
    def name(self) -> str:
        return f"Compound ({self.rate_per_percent * 100:.2f}%/pp)"

    @property
    def description(self) -> str:
        return (
            f"Compound depreciation: value multiplied by (1 - {self.rate_per_percent * 100:.2f}%) "
            f"for each percentage point of coral cover decrease. "
            f"Models diminishing marginal loss."
        )

    def calculate(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        # delta_cc is negative for decrease (e.g., -0.10 = 10pp decrease)
        # We want: value * (1 - rate)^|delta_cc_pp|

        delta_cc_pp = np.abs(delta_cc * 100)  # Always positive, in percentage points

        # Only apply depreciation for coral cover DECREASE (delta_cc < 0)
        # For increases (delta_cc > 0), we could model appreciation, but
        # the default behavior is no change (conservative assumption)
        is_decrease = np.asarray(delta_cc) < 0

        decay_factor = (1 - self.rate_per_percent) ** delta_cc_pp

        # Apply decay only to decreases
        remaining = np.where(is_decrease, value * decay_factor, value)

        return np.maximum(remaining, 0)


@dataclass
class TippingPointModel(DepreciationModel):
    """
    Tipping point depreciation: gradual loss until threshold, then rapid collapse.

    Models ecosystem collapse where:
    - Below threshold: linear/gradual depreciation
    - At/beyond threshold: catastrophic value loss

    Parameters
    ----------
    threshold_cc : float
        Coral cover threshold (proportion) below which collapse occurs.
        E.g., 0.10 means collapse when cover drops below 10%.
    pre_threshold_rate : float
        Depreciation rate before threshold (per percentage point).
    post_threshold_loss : float
        Fraction of remaining value lost when threshold is crossed (0-1).
        E.g., 0.9 means 90% of remaining value is lost.
    """

    threshold_cc: float = 0.10  # 10% coral cover threshold
    pre_threshold_rate: float = 0.0381  # 3.81% loss per pp before threshold
    post_threshold_loss: float = 0.90  # 80% of value lost at collapse

    @property
    def name(self) -> str:
        return f"Tipping Point (threshold={self.threshold_cc * 100:.0f}%)"

    @property
    def description(self) -> str:
        return (
            f"Tipping point model: {self.pre_threshold_rate * 100:.1f}% loss per pp "
            f"until coral cover falls below {self.threshold_cc * 100:.0f}%, "
            f"then {self.post_threshold_loss * 100:.0f}% catastrophic loss."
        )

    # def calculate(
    #     self,
    #     delta_cc: Union[float, np.ndarray],
    #     value: Union[float, np.ndarray],
    #     current_cover: Union[float, np.ndarray] = None,
    #     future_cover: Union[float, np.ndarray] = None,
    # ) -> Union[float, np.ndarray]:
    #     """
    #     Calculate remaining value with tipping point dynamics.

    #     The tipping point is crossed when:
    #     - current_cover >= threshold_cc (starts above threshold)
    #     - future_cover < threshold_cc (ends below threshold)

    #     Parameters
    #     ----------
    #     delta_cc : float or array
    #         Change in coral cover (proportion, negative for decrease).
    #     value : float or array
    #         Original economic value.
    #     current_cover : float or array, optional
    #         Current/baseline coral cover (proportion). If not provided, inferred from
    #         future_cover - delta_cc, or assumed to be above threshold.
    #     future_cover : float or array, optional
    #         Future coral cover (proportion). If not provided, calculated as
    #         current_cover + delta_cc.

    #     Returns
    #     -------
    #     float or array
    #         Remaining value after depreciation.
    #     """
    #     delta_cc_pp = np.abs(delta_cc * 100)  # Percentage points of change
    #     is_decrease = np.asarray(delta_cc) < 0

    #     # Determine current and future cover
    #     if future_cover is not None:
    #         future_cover_arr = np.asarray(future_cover)
    #         if current_cover is not None:
    #             current_cover_arr = np.asarray(current_cover)
    #         else:
    #             # Infer current cover from future cover and change
    #             current_cover_arr = future_cover_arr - np.asarray(delta_cc)
    #     elif current_cover is not None:
    #         current_cover_arr = np.asarray(current_cover)
    #         future_cover_arr = current_cover_arr + np.asarray(delta_cc)
    #     else:
    #         # Neither provided: use fallback logic
    #         # Assume large decreases cross threshold
    #         current_cover_arr = None
    #         future_cover_arr = None

    #     # Pre-threshold gradual loss (linear)
    #     gradual_loss_factor = 1 - (self.pre_threshold_rate * delta_cc_pp)
    #     gradual_remaining = value * np.maximum(gradual_loss_factor, 0)

    #     # Check if we've crossed the threshold
    #     # Threshold is crossed if: started above threshold AND ended below threshold
    #     if current_cover_arr is not None and future_cover_arr is not None:
    #         started_above = current_cover_arr >= self.threshold_cc
    #         ended_below = future_cover_arr < self.threshold_cc
    #         crossed_threshold = started_above & ended_below
    #     elif future_cover_arr is not None:
    #         # Only future cover known: check if below threshold
    #         # This is less precise but better than nothing
    #         ended_below = future_cover_arr < self.threshold_cc
    #         # Assume we started above if future is below (conservative)
    #         crossed_threshold = ended_below
    #     else:
    #         # Fallback: assume threshold crossed for large decreases
    #         # (>50pp decrease likely crosses most thresholds)
    #         crossed_threshold = delta_cc_pp > 50

    #     # Apply catastrophic loss at threshold
    #     # If threshold crossed, apply additional loss to the gradually-depreciated value
    #     collapse_remaining = gradual_remaining * (1 - self.post_threshold_loss)

    #     # Combine: use collapse value if threshold crossed, else gradual
    #     remaining = np.where(
    #         is_decrease & crossed_threshold,
    #         collapse_remaining,
    #         np.where(is_decrease, gradual_remaining, value),
    #     )

    #     return np.maximum(remaining, 0)

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        current_cover: Union[float, np.ndarray] = None,
        future_cover: Union[float, np.ndarray] = None,
    ) -> Union[float, np.ndarray]:
        """
        Calculate remaining value with tipping point dynamics.

        The tipping point is crossed when:
        - current_cover >= threshold_cc (starts above threshold)
        - future_cover < threshold_cc (ends below threshold)

        Before threshold: linear depreciation (same as LinearModel)
        At threshold crossing: catastrophic loss applied
        After threshold: continue with linear depreciation from post-collapse value
        """
        delta_cc_pp = np.abs(delta_cc * 100)  # Percentage points of change
        is_decrease = np.asarray(delta_cc) < 0

        # Determine current and future cover
        if future_cover is not None:
            future_cover_arr = np.asarray(future_cover)
            if current_cover is not None:
                current_cover_arr = np.asarray(current_cover)
            else:
                current_cover_arr = future_cover_arr - np.asarray(delta_cc)
        elif current_cover is not None:
            current_cover_arr = np.asarray(current_cover)
            future_cover_arr = current_cover_arr + np.asarray(delta_cc)
        else:
            current_cover_arr = None
            future_cover_arr = None

        # Check if we've crossed the threshold
        if current_cover_arr is not None and future_cover_arr is not None:
            started_above = current_cover_arr >= self.threshold_cc
            ended_below = future_cover_arr < self.threshold_cc
            crossed_threshold = started_above & ended_below
        elif future_cover_arr is not None:
            ended_below = future_cover_arr < self.threshold_cc
            crossed_threshold = ended_below
        else:
            crossed_threshold = delta_cc_pp > 50

        # Calculate value using linear depreciation (same as LinearModel)
        # This gives us the correct gradient before AND after threshold
        delta_cc_pp_signed = (
            np.asarray(delta_cc) * 100
        )  # Keep sign for linear calculation
        linear_remaining = value * (1 + self.pre_threshold_rate * delta_cc_pp_signed)

        # If threshold is crossed, apply catastrophic loss at the crossing point
        # Calculate how much cover change is needed to reach threshold
        if current_cover_arr is not None and future_cover_arr is not None:
            # Calculate cover change to threshold
            cover_to_threshold = self.threshold_cc - current_cover_arr
            # Value at threshold (using linear depreciation)
            value_at_threshold = value * (
                1 + self.pre_threshold_rate * cover_to_threshold * 100
            )
            # Apply catastrophic loss
            value_after_collapse = value_at_threshold * (1 - self.post_threshold_loss)
            # Remaining cover change after threshold
            cover_change_after = future_cover_arr - self.threshold_cc
            # Continue linear depreciation from post-collapse value
            if np.any(crossed_threshold):
                # For points that cross threshold: use post-collapse value + linear depreciation for remaining change
                remaining_after = np.where(
                    crossed_threshold,
                    value_after_collapse
                    * (1 + self.pre_threshold_rate * cover_change_after * 100),
                    linear_remaining,
                )
            else:
                remaining_after = linear_remaining
        else:
            # Fallback: if threshold crossed, apply catastrophic loss to linear value
            remaining_after = np.where(
                crossed_threshold,
                linear_remaining * (1 - self.post_threshold_loss),
                linear_remaining,
            )

        # Only apply to decreases
        remaining = np.where(is_decrease, remaining_after, value)

        return np.maximum(remaining, 0)


@dataclass
class CoastalProtectionModel(DepreciationModel):
    """
    Model for coastal protection value loss.

    Coastal protection depends on reef structure, which degrades differently
    than tourism value. This model can be parameterized with different
    assumptions about structural integrity loss.

    Reference: Beck et al. (2018) - The global flood protection savings
    provided by coral reefs.
    """

    structural_loss_rate: float = 0.05  # 5% protection loss per 1pp cover decrease
    min_protection_fraction: float = (
        0.20  # Reef provides min 20% protection even degraded
    )

    @property
    def name(self) -> str:
        return f"Coastal Protection ({self.structural_loss_rate * 100:.1f}%/pp)"

    @property
    def description(self) -> str:
        return (
            f"Coastal protection depreciation: {self.structural_loss_rate * 100:.1f}% "
            f"protection loss per pp coral cover decrease. "
            f"Minimum {self.min_protection_fraction * 100:.0f}% protection retained."
        )

    def calculate(
        self, delta_cc: Union[float, np.ndarray], value: Union[float, np.ndarray]
    ) -> Union[float, np.ndarray]:
        delta_cc_pp = np.abs(delta_cc * 100)
        is_decrease = np.asarray(delta_cc) < 0

        # Calculate protection factor (fraction of protection remaining)
        protection_factor = 1 - (self.structural_loss_rate * delta_cc_pp)
        protection_factor = np.maximum(protection_factor, self.min_protection_fraction)

        remaining = np.where(is_decrease, value * protection_factor, value)

        return np.maximum(remaining, 0)


# Registry of available models
_MODEL_REGISTRY: Dict[str, type] = {
    "linear": LinearModel,
    "compound": CompoundModel,
    "tipping_point": TippingPointModel,
    "coastal_protection": CoastalProtectionModel,
}


def get_model(name: str, **kwargs) -> DepreciationModel:
    """
    Get a depreciation model by name.

    Parameters
    ----------
    name : str
        Model name: 'linear', 'compound', 'tipping_point', 'coastal_protection'
    **kwargs
        Model-specific parameters (e.g., rate_per_percent, threshold_cc)

    Returns
    -------
    DepreciationModel
        Instantiated model.

    Examples
    --------
    >>> model = get_model("compound", rate_per_percent=0.05)
    >>> model.calculate(delta_cc=-0.10, value=1000)
    598.74  # ~40% loss for 10pp decrease
    """
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> Dict[str, str]:
    """
    List available models with descriptions.

    Returns
    -------
    dict
        Model names mapped to descriptions.
    """
    return {name: cls().description for name, cls in _MODEL_REGISTRY.items()}


def compare_models(
    delta_cc_range: np.ndarray = None, value: float = 100.0, models: list = None
) -> dict:
    """
    Compare multiple models across a range of coral cover changes.

    Parameters
    ----------
    delta_cc_range : array, optional
        Range of delta_cc values to compare. Default: -1.0 to 0 in 0.01 steps.
    value : float
        Base value for comparison.
    models : list, optional
        List of model names or instances. Default: all registered models.

    Returns
    -------
    dict
        Keys: model names, Values: arrays of remaining values.
    """
    if delta_cc_range is None:
        delta_cc_range = np.linspace(-1.0, 0, 101)  # 0 to 100pp decrease

    if models is None:
        models = list(_MODEL_REGISTRY.keys())

    results = {"delta_cc": delta_cc_range}

    for m in models:
        if isinstance(m, str):
            model = get_model(m)
        else:
            model = m
        results[model.name] = model.calculate(delta_cc_range, value)

    return results
