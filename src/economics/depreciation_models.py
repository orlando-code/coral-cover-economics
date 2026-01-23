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
    def model_type(self) -> str:
        """Model type for colour selection in plotting."""
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
    def model_type(self) -> str:
        return "linear"

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
    def model_type(self) -> str:
        return "compound"

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
        """Calculate remaining value after compound depreciation."""
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
    post_threshold_loss: float = 1  # 80% of value lost at collapse

    @property
    def name(self) -> str:
        return f"Tipping Point (threshold={self.threshold_cc * 100:.0f}%)"

    @property
    def model_type(self) -> str:
        return "tipping_point"

    @property
    def description(self) -> str:
        return (
            f"Tipping point model: {self.pre_threshold_rate * 100:.1f}% loss per pp "
            f"until coral cover falls below {self.threshold_cc * 100:.0f}%, "
            f"then {self.post_threshold_loss * 100:.0f}% catastrophic loss."
        )

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        threshold: float = None,  # If None, uses self.threshold_cc
        original_cc: Union[float, np.ndarray] = 0.5,
    ) -> Union[float, np.ndarray]:
        """
        Calculate remaining value with tipping point behavior.

        Parameters
        ----------
        delta_cc : float or array
            Change in coral cover (negative for decrease).
        value : float or array
            Current economic value.
        threshold : float, optional
            Coral cover threshold. If None, uses self.threshold_cc.
        original_cc : float or array, optional
            Original/baseline coral cover. Default: 0.5.

        Returns
        -------
        float or array
            Remaining value after depreciation.
        """
        # Use model's threshold_cc if threshold not provided
        if threshold is None:
            threshold = self.threshold_cc

        # delta_cc is negative for decrease (e.g., -0.10 = 10pp decrease)
        delta_cc_pp = np.abs(delta_cc * 100)  # Always positive, in percentage points

        # Only apply depreciation for coral cover DECREASE (delta_cc < 0)
        is_decrease = np.asarray(delta_cc) < 0

        # Baseline compound decline before threshold
        decay_factor = (1 - self.pre_threshold_rate) ** delta_cc_pp

        # Apply depreciation
        remaining_value = np.where(is_decrease, value * decay_factor, value)

        # Calculate remaining coral cover
        remaining_cc = np.maximum(original_cc + delta_cc, 0)

        # Apply tipping point collapse: if remaining_cc < threshold, catastrophic loss
        # Uses post_threshold_loss to determine how much value is lost
        collapse_mask = remaining_cc < threshold
        if np.any(collapse_mask):
            # Apply catastrophic loss: lose post_threshold_loss fraction
            remaining_value = np.where(
                collapse_mask,
                remaining_value * (1 - self.post_threshold_loss),
                remaining_value,
            )

        return np.maximum(remaining_value, 0)


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
    delta_cc_range: np.ndarray = None,
    value: float = 100.0,
    models: list = None,
    original_cc: float = 0.5,
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
    original_cc : float, optional
        Original coral cover (for tipping point model). Default: 0.5.

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

        # TippingPointModel requires original_cc parameter
        if model.model_type == "tipping_point":
            threshold = getattr(model, "threshold_cc", 0.1)
            results[model.name] = model.calculate(
                delta_cc_range, value, original_cc=original_cc, threshold=threshold
            )
        else:
            # Standard models use simple signature
            results[model.name] = model.calculate(delta_cc_range, value)

    return results
