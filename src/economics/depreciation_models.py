"""
Pluggable depreciation models for mapping coral cover change to economic value loss.

Sector-specific Chen et al. valuation (tourism elasticity; fisheries/coastal 1:1 relative)
is the default ``linear`` model.  Compound and tipping-point models remain available
for sensitivity analysis.

Reference:
- Chen et al. (2014/2015): tourism elasticity ~3.81% value loss per 1% relative coral loss
  DOI: 10.1016/j.gloenvcha.2014.10.011
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

# Chen et al. quadratic coefficients (Table 5) for exact_nonlinear tourism
_CHEN_BETA1 = 429.665
_CHEN_BETA2 = 214.167
_CHEN_MEAN_TV = 108319.0

_SECTOR_KEYS = {
    "tourism": "tourism",
    "fisheries": "fisheries",
    "coastal_protection": "coastal_protection",
}


def compute_coral_valuation_change(
    initial_cover: float,
    final_cover: float,
    baseline_tourism: float = 0.0,
    baseline_fisheries: float = 0.0,
    baseline_coastal_protection: float = 0.0,
    method: str = "elasticity",
) -> Dict:
    """
    Projected economic impact of coral cover change across three sectors (Chen et al.).

    Parameters
    ----------
    initial_cover, final_cover : float
        Live coral cover on a **percentage scale** (e.g. 35.0 for 35%).
    baseline_* : float
        Baseline economic value per sector (same currency units).
    method : str
        ``elasticity`` (default) or ``exact_nonlinear`` (quadratic tourism curve).

    Returns
    -------
    dict
        Per-sector percentage and absolute changes plus combined totals.
    """
    if initial_cover <= 0:
        raise ValueError("Initial coral cover must be greater than 0%.")

    relative_cover_change = (final_cover - initial_cover) / initial_cover

    if method == "elasticity":
        tourism_pct_change = relative_cover_change * 3.8069
    elif method == "exact_nonlinear":
        delta_tv = ( _CHEN_BETA1 * (final_cover - initial_cover)
            + _CHEN_BETA2 * (final_cover**2 - initial_cover**2)
        )
        tourism_pct_change = delta_tv / _CHEN_MEAN_TV
    else:
        raise ValueError("method must be 'elasticity' or 'exact_nonlinear'")

    tourism_abs_change = baseline_tourism * tourism_pct_change
    fisheries_pct_change = relative_cover_change
    fisheries_abs_change = baseline_fisheries * fisheries_pct_change
    coastal_pct_change = relative_cover_change
    coastal_abs_change = baseline_coastal_protection * coastal_pct_change

    total_baseline = baseline_tourism + baseline_fisheries + baseline_coastal_protection
    total_abs_change = tourism_abs_change + fisheries_abs_change + coastal_abs_change
    total_pct_change = (total_abs_change / total_baseline) if total_baseline > 0 else 0.0

    return {
        "metadata": {
            "initial_cover_pct": initial_cover,
            "final_cover_pct": final_cover,
            "relative_cover_change_pct": relative_cover_change * 100,
            "calculation_method": method,
        },
        "tourism": {
            "percentage_change": tourism_pct_change * 100,
            "absolute_change": tourism_abs_change,
        },
        "fisheries": {
            "percentage_change": fisheries_pct_change * 100,
            "absolute_change": fisheries_abs_change,
        },
        "coastal_protection": {
            "percentage_change": coastal_pct_change * 100,
            "absolute_change": coastal_abs_change,
        },
        "total_combined": {
            "percentage_change": total_pct_change * 100,
            "absolute_change": total_abs_change,
        },
    }


def _chen_fractional_change(
    initial_cover: Union[float, np.ndarray],
    delta_cc: Union[float, np.ndarray],
    value_type: str,
    method: str = "elasticity",
) -> np.ndarray:
    """
    Fractional economic change (e.g. -0.38 = 38% loss) for one sector.

    ``initial_cover`` and ``delta_cc`` are proportions on [0, 1] scale.
    """
    initial = np.asarray(initial_cover, dtype=float)
    delta = np.asarray(delta_cc, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        relative = np.where(initial > 0, delta / initial, 0.0)

    sector = _SECTOR_KEYS.get(value_type, value_type)

    if sector == "tourism":
        if method == "elasticity":
            return relative * 3.8069
        if method == "exact_nonlinear":
            initial_pct = initial * 100.0
            final_pct = (initial + delta) * 100.0
            delta_tv = (
                _CHEN_BETA1 * (final_pct - initial_pct)
                + _CHEN_BETA2 * (final_pct**2 - initial_pct**2)
            )
            return np.where(_CHEN_MEAN_TV > 0, delta_tv / _CHEN_MEAN_TV, 0.0)
        raise ValueError("method must be 'elasticity' or 'exact_nonlinear'")

    if sector in ("fisheries", "coastal_protection"):
        return relative

    return relative


def chen_remaining_value(
    delta_cc: Union[float, np.ndarray],
    value: Union[float, np.ndarray],
    value_type: str,
    initial_cover: Union[float, np.ndarray],
    method: str = "elasticity",
) -> np.ndarray:
    """Remaining value after Chen sector-specific depreciation."""
    frac_change = _chen_fractional_change(initial_cover, delta_cc, value_type, method)
    remaining = np.asarray(value, dtype=float) * (1.0 + frac_change)
    return np.maximum(remaining, 0.0)


def uses_chen_valuation(model: "DepreciationModel") -> bool:
    """True for models backed by compute_coral_valuation_change."""
    return model.model_type in ("linear", "chen_exact")


def apply_depreciation_model(
    model: "DepreciationModel",
    delta_cc: Union[float, np.ndarray],
    value: Union[float, np.ndarray],
    *,
    value_type: str = "tourism",
    initial_cover: Union[float, np.ndarray, None] = None,
    original_cc: Union[float, np.ndarray, None] = None,
    threshold: float = None,
) -> Union[float, np.ndarray]:
    """Call ``model.calculate`` with the keyword arguments required by its type."""
    if model.model_type == "tipping_point":
        baseline = original_cc if original_cc is not None else initial_cover
        return model.calculate(
            delta_cc,
            value,
            value_type=value_type,
            initial_cover=initial_cover,
            original_cc=baseline,
            threshold=threshold,
        )
    if uses_chen_valuation(model):
        if initial_cover is None:
            raise ValueError(f"{model.name} requires initial_cover.")
        return model.calculate(
            delta_cc,
            value,
            value_type=value_type,
            initial_cover=initial_cover,
        )
    return model.calculate(delta_cc, value, value_type=value_type)


@dataclass
class DepreciationModel(ABC):
    """Abstract depreciation model."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def model_type(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @abstractmethod
    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        *,
        value_type: str = "tourism",
        initial_cover: Union[float, np.ndarray, None] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        pass

    def calculate_change(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        **kwargs,
    ) -> Union[float, np.ndarray]:
        return value - self.calculate(delta_cc, value, **kwargs)

    def calculate_change_fraction(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        **kwargs,
    ) -> Union[float, np.ndarray]:
        remaining = self.calculate(delta_cc, value, **kwargs)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = 1 - (remaining / value)
            fraction = np.where(value == 0, 0, fraction)
        return fraction

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


@dataclass
class LinearModel(DepreciationModel):
    """
    Chen et al. sector-specific valuation (registered as ``linear``).

    - Tourism: 3.81% value loss per 1% **relative** coral cover loss (elasticity).
    - Fisheries / coastal protection: 1:1 with relative coral cover change.
    """

    method: str = "elasticity"

    @property
    def name(self) -> str:
        return "Linear (3.81%/rel%)"

    @property
    def model_type(self) -> str:
        return "linear"

    @property
    def description(self) -> str:
        return (
            "Chen et al. valuation: tourism elasticity 3.81% per 1% relative coral loss; "
            "fisheries and coastal protection scale 1:1 with relative coral change."
        )

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        *,
        value_type: str = "tourism",
        initial_cover: Union[float, np.ndarray, None] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        if initial_cover is None:
            raise ValueError(
                f"{self.name} requires initial_cover (baseline coral cover, proportion 0–1)."
            )
        return chen_remaining_value(
            delta_cc, value, value_type, initial_cover, method=self.method
        )


@dataclass
class ChenExactModel(DepreciationModel):
    """Chen et al. quadratic tourism curve (exact_nonlinear); other sectors 1:1 relative."""

    @property
    def name(self) -> str:
        return "Chen Exact (quadratic)"

    @property
    def model_type(self) -> str:
        return "chen_exact"

    @property
    def description(self) -> str:
        return (
            "Chen et al. exact quadratic tourism meta-regression; "
            "fisheries and coastal protection scale 1:1 with relative coral change."
        )

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        *,
        value_type: str = "tourism",
        initial_cover: Union[float, np.ndarray, None] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        if initial_cover is None:
            raise ValueError(f"{self.name} requires initial_cover.")
        return chen_remaining_value(
            delta_cc, value, value_type, initial_cover, method="exact_nonlinear"
        )


@dataclass
class CompoundModel(DepreciationModel):
    """Compound depreciation per percentage point of absolute coral cover loss."""

    rate_per_percent: float = 0.0381

    @property
    def name(self) -> str:
        return f"Compound ({self.rate_per_percent * 100:.2f}%/pp)"

    @property
    def model_type(self) -> str:
        return "compound"

    @property
    def description(self) -> str:
        return (
            f"Compound: value × (1 − {self.rate_per_percent * 100:.2f}%) "
            f"per percentage point of absolute coral cover decrease."
        )

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        *,
        value_type: str = "tourism",
        initial_cover: Union[float, np.ndarray, None] = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        delta_cc_pp = np.abs(np.asarray(delta_cc) * 100)
        is_decrease = np.asarray(delta_cc) < 0
        decay_factor = (1 - self.rate_per_percent) ** delta_cc_pp
        remaining = np.where(is_decrease, value * decay_factor, value)
        return np.maximum(remaining, 0)


@dataclass
class TippingPointModel(DepreciationModel):
    """Gradual compound loss until a coral cover threshold, then catastrophic loss."""

    threshold_cc: float = 0.10
    pre_threshold_rate: float = 0.0381
    post_threshold_loss: float = 1.0

    @property
    def name(self) -> str:
        return f"Tipping Point (threshold={self.threshold_cc * 100:.0f}%)"

    @property
    def model_type(self) -> str:
        return "tipping_point"

    @property
    def description(self) -> str:
        return (
            f"Tipping point: {self.pre_threshold_rate * 100:.1f}% compound loss per pp "
            f"until cover < {self.threshold_cc * 100:.0f}%, "
            f"then {self.post_threshold_loss * 100:.0f}% catastrophic loss."
        )

    def calculate(
        self,
        delta_cc: Union[float, np.ndarray],
        value: Union[float, np.ndarray],
        *,
        value_type: str = "tourism",
        initial_cover: Union[float, np.ndarray, None] = None,
        original_cc: Union[float, np.ndarray, None] = None,
        threshold: float = None,
        **kwargs,
    ) -> Union[float, np.ndarray]:
        if threshold is None:
            threshold = self.threshold_cc
        if original_cc is None:
            original_cc = initial_cover if initial_cover is not None else 0.5

        delta_cc_pp = np.abs(np.asarray(delta_cc) * 100)
        is_decrease = np.asarray(delta_cc) < 0
        decay_factor = (1 - self.pre_threshold_rate) ** delta_cc_pp
        remaining_value = np.where(is_decrease, value * decay_factor, value)

        remaining_cc = np.maximum(original_cc + delta_cc, 0)
        collapse_mask = remaining_cc < threshold
        if np.any(collapse_mask):
            remaining_value = np.where(
                collapse_mask,
                remaining_value * (1 - self.post_threshold_loss),
                remaining_value,
            )
        return np.maximum(remaining_value, 0)


_MODEL_REGISTRY: Dict[str, type] = {
    "linear": LinearModel,
    "chen_exact": ChenExactModel,
    "compound": CompoundModel,
    "tipping_point": TippingPointModel,
}


def get_model(name: str, **kwargs) -> DepreciationModel:
    if name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[name](**kwargs)


def list_models() -> Dict[str, str]:
    return {name: cls().description for name, cls in _MODEL_REGISTRY.items()}


def compare_models(
    delta_cc_range: np.ndarray = None,
    value: float = 100.0,
    models: list = None,
    original_cc: float = 0.35,
    value_type: str = "tourism",
) -> dict:
    if delta_cc_range is None:
        delta_cc_range = np.linspace(-1.0, 0, 101)

    if models is None:
        models = list(_MODEL_REGISTRY.keys())

    results = {"delta_cc": delta_cc_range}

    for m in models:
        if isinstance(m, str):
            model = get_model(m)
        else:
            model = m

        if model.model_type == "tipping_point":
            threshold = getattr(model, "threshold_cc", 0.1)
            results[model.name] = model.calculate(
                delta_cc_range,
                value,
                value_type=value_type,
                initial_cover=original_cc,
                original_cc=original_cc,
                threshold=threshold,
            )
        elif uses_chen_valuation(model):
            results[model.name] = model.calculate(
                delta_cc_range,
                value,
                value_type=value_type,
                initial_cover=original_cc,
            )
        else:
            results[model.name] = model.calculate(delta_cc_range, value)

    return results
