"""Distance-marginalized gravitational-wave likelihood in the (rho_mf, v*) basis."""
from ._prior import FitWarning, PriorFit, fit_prior
from .core import (
    REGIME_HIGH_SNR,
    REGIME_NAMES,
    REGIME_PRIOR_DOMINATED,
    REGIME_QUADRATURE,
    DistanceMarginalization,
)

__all__ = [
    "DistanceMarginalization",
    "FitWarning",
    "PriorFit",
    "fit_prior",
    "REGIME_HIGH_SNR",
    "REGIME_PRIOR_DOMINATED",
    "REGIME_QUADRATURE",
    "REGIME_NAMES",
]
__version__ = "0.1.0"
