"""Canonical interpretation of the tabulated prior (IMPLEMENTATION_NOTES §3.0).

All set-up is host-side numpy float64:

1. ``s_i = -ln d_i`` and ``g_i = ln_prior_i + 2 ln d_i`` (unnormalized ``ln pi_v``);
2. ``t = (2 s - s_a - s_b) / (s_b - s_a)`` with ``s_a = -ln d_max``, ``s_b = -ln d_min``;
3. unweighted least-squares Chebyshev fit of degree ``K`` (auto: the smallest ``K``
   in ``{8, 16, ..., 128}`` with max ``|residual| < fit_tol``);
4. ``ln Z = ln int exp(g~(s) + s) ds`` by 256-node Gauss-Legendre; ``g = g~ - ln Z``;
5. derivative coefficients and prior moments.
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy.special import logsumexp

DEGREE_CANDIDATES = tuple(range(8, 129, 8))
N_NORM_NODES = 256
N_MOMENTS = 8  # E[v^k], k = 0..8 reported in FitInfo
N_CENTRAL = 24  # central moments of z = nu or nu^2 used by the cumulant series


class FitWarning(UserWarning):
    """Warn that the Chebyshev fit of ``ln pi_v`` did not reach ``fit_tol``."""


@dataclass(frozen=True)
class PriorFit:
    """Chebyshev representation of the prior density in ``v = 1/d_L``.

    Attributes
    ----------
    d_min, d_max : float
        Support of the prior in luminosity distance.
    s_a, s_b : float
        Support in ``s = ln v``: ``-ln d_max`` and ``-ln d_min``.
    degree : int
        Chebyshev degree of the fit.
    max_residual : float
        Maximum absolute fit residual at the input points.
    raw_coeffs : numpy.ndarray
        Least-squares coefficients of the unnormalized ``ln pi_v(t)``.
    ln_norm : float
        Log normalization subtracted from ``raw_coeffs[0]``.
    coeffs : numpy.ndarray
        Coefficients of the normalized ``ln pi_v(t)``.
    d1, d2 : numpy.ndarray
        Coefficients of ``d ln pi_v / ds`` and ``d^2 ln pi_v / ds^2`` as series in ``t``.
    moments : numpy.ndarray
        ``E_pi[v^k]`` for ``k = 0..8``.
    zbar, central : dict
        For ``z = nu = v / v_max`` (key ``"nu"``) and ``z = nu^2`` (key ``"nu2"``):
        ``E[z]`` and the central moments ``E[(z - E z)^k]``, ``k = 0..24``.
    """

    d_min: float
    d_max: float
    s_a: float
    s_b: float
    degree: int
    max_residual: float
    raw_coeffs: np.ndarray
    ln_norm: float
    coeffs: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    moments: np.ndarray
    zbar: dict = field(default_factory=dict)
    central: dict = field(default_factory=dict)

    @property
    def v_min(self):
        """float: Lower edge of the support in ``v``, ``1 / d_max``."""
        return 1.0 / self.d_max

    @property
    def v_max(self):
        """float: Upper edge of the support in ``v``, ``1 / d_min``."""
        return 1.0 / self.d_min

    def t_of_s(self, s):
        """Map ``s = ln v`` onto the Chebyshev domain ``[-1, 1]``.

        Parameters
        ----------
        s : array or float
            Log inverse distance.

        Returns
        -------
        array or float
            ``(2 s - s_a - s_b) / (s_b - s_a)``.
        """
        return (2.0 * s - self.s_a - self.s_b) / (self.s_b - self.s_a)

    def ln_pi_v(self, s):
        """Evaluate the normalized log prior density in ``v``.

        Parameters
        ----------
        s : array_like
            ``ln v`` inside ``[s_a, s_b]``.

        Returns
        -------
        numpy.ndarray
            ``ln pi_v(e^s)``.
        """
        return cheb.chebval(self.t_of_s(np.asarray(s, dtype=float)), self.coeffs)

    def fit_info(self) -> dict:
        """Summarize the fit in the cross-implementation fixture format.

        Returns
        -------
        dict
            Keys ``degree``, ``max_residual``, ``coeffs`` (raw), ``ln_norm`` and
            ``moments`` (``E[v^k]``, ``k <= 8``).
        """
        return dict(
            degree=self.degree,
            max_residual=self.max_residual,
            coeffs=[float(c) for c in self.raw_coeffs],
            ln_norm=self.ln_norm,
            moments=[float(m) for m in self.moments],
        )


def _to_host(x):
    """Convert an array from any namespace to numpy.

    Parameters
    ----------
    x : array_like
        Array, possibly a torch tensor or CuPy array.

    Returns
    -------
    numpy.ndarray
        Host copy of ``x``.
    """
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def validate_prior_arrays(distance, ln_prior):
    """Check and convert the prior arrays.

    Parameters
    ----------
    distance : array_like
        Luminosity distances: 1-D, finite, ``> 0`` and strictly increasing.
    ln_prior : array_like
        Log prior density at ``distance``: finite everywhere.

    Returns
    -------
    distance, ln_prior : numpy.ndarray
        float64 copies.

    Raises
    ------
    ValueError
        If any condition is violated.
    """
    d = np.asarray(_to_host(distance), dtype=np.float64)
    lp = np.asarray(_to_host(ln_prior), dtype=np.float64)
    if d.ndim != 1 or lp.ndim != 1 or d.shape != lp.shape:
        raise ValueError("distance and ln_prior must be 1-D arrays of equal length")
    if d.size < 2:
        raise ValueError("at least two prior points are required")
    if not np.all(np.isfinite(d)) or np.any(d <= 0):
        raise ValueError("distance must be finite and > 0")
    if np.any(np.diff(d) <= 0):
        raise ValueError("distance must be strictly increasing")
    if not np.all(np.isfinite(lp)):
        raise ValueError("ln_prior must be finite everywhere (zero prior density is not allowed)")
    return d, lp


def _lstsq_fit(t, g, degree):
    """Fit a Chebyshev series by unweighted least squares.

    Parameters
    ----------
    t : numpy.ndarray
        Abscissae in ``[-1, 1]``.
    g : numpy.ndarray
        Values to fit.
    degree : int
        Series degree.

    Returns
    -------
    coeffs : numpy.ndarray
        Least-squares coefficients (SVD of the Vandermonde matrix, not normal equations).
    resid : float
        Maximum absolute residual.
    """
    vander = cheb.chebvander(t, degree)
    coeffs, *_ = np.linalg.lstsq(vander, g, rcond=None)
    return coeffs, float(np.max(np.abs(vander @ coeffs - g)))


def fit_prior(distance, ln_prior, fit_degree=None, fit_tol=1e-8) -> PriorFit:
    """Fit, normalize and summarize a tabulated luminosity-distance prior.

    Parameters
    ----------
    distance : array_like
        Luminosity distances (finite, ``> 0``, strictly increasing).
    ln_prior : array_like
        Log prior density at ``distance`` (finite; need not be normalized).
    fit_degree : int, optional
        Chebyshev degree.  ``None`` picks the smallest of ``8, 16, ..., 128`` with
        max residual below ``fit_tol``; only degrees with
        ``len(distance) >= 2 (K + 1)`` are considered.
    fit_tol : float, optional
        Residual at or above which a :class:`FitWarning` is emitted.

    Returns
    -------
    PriorFit
        The fitted prior.

    Raises
    ------
    ValueError
        For invalid arrays or too few points for the requested degree.

    Warns
    -----
    FitWarning
        If the fit residual is ``>= fit_tol``; the prior is then likely non-smooth
        or noisily tabulated and the high-SNR curvature model is unreliable.
    """
    d, lp = validate_prior_arrays(distance, ln_prior)
    n = d.size
    s = -np.log(d)
    g = lp + 2.0 * np.log(d)
    s_a, s_b = -math.log(d[-1]), -math.log(d[0])
    t = (2.0 * s - s_a - s_b) / (s_b - s_a)

    if fit_degree is not None:
        degree = int(fit_degree)
        if degree < 0:
            raise ValueError("fit_degree must be >= 0")
        if n < 2 * (degree + 1):
            raise ValueError(f"need len(distance) >= 2(K+1) = {2 * (degree + 1)} for fit_degree={degree}")
        raw, resid = _lstsq_fit(t, g, degree)
    else:
        allowed = [k for k in DEGREE_CANDIDATES if n >= 2 * (k + 1)]
        if not allowed:
            raise ValueError(f"need at least {2 * (DEGREE_CANDIDATES[0] + 1)} prior points for automatic degree selection")
        for degree in allowed:
            raw, resid = _lstsq_fit(t, g, degree)
            if resid < fit_tol:
                break
    if not resid < fit_tol:
        warnings.warn(
            f"Chebyshev fit of ln pi_v has max residual {resid:.3g} >= fit_tol={fit_tol:.3g} "
            f"(degree {degree}); the prior may be non-smooth or noisily tabulated and the "
            "high-SNR curvature model is unreliable",
            FitWarning,
            stacklevel=3,
        )

    # normalization in s: dv = e^s ds
    x_gl, w_gl = np.polynomial.legendre.leggauss(N_NORM_NODES)
    half = 0.5 * (s_b - s_a)
    s_nodes = 0.5 * (s_a + s_b) + half * x_gl
    log_terms = np.log(half * w_gl) + cheb.chebval(x_gl, raw) + s_nodes
    ln_norm = float(logsumexp(log_terms))

    coeffs = raw.copy()
    coeffs[0] -= ln_norm
    scale = 2.0 / (s_b - s_a)
    d1 = cheb.chebder(coeffs, scl=scale)
    d2 = cheb.chebder(coeffs, m=2, scl=scale)

    # prior weights at the Gauss-Legendre nodes
    pw = np.exp(log_terms - ln_norm)
    v = np.exp(s_nodes)
    moments = np.array([np.sum(pw * v**k) for k in range(N_MOMENTS + 1)])

    pw_n = pw / np.sum(pw)
    nu = v * d[0]  # v / v_max
    zbar, central = {}, {}
    for key, z in (("nu", nu), ("nu2", nu * nu)):
        zb = float(np.sum(pw_n * z))
        dz = z - zb
        zbar[key] = zb
        central[key] = np.array([np.sum(pw_n * dz**k) for k in range(N_CENTRAL + 1)])
        central[key][0], central[key][1] = 1.0, 0.0

    return PriorFit(
        d_min=float(d[0]),
        d_max=float(d[-1]),
        s_a=s_a,
        s_b=s_b,
        degree=degree,
        max_residual=resid,
        raw_coeffs=raw,
        ln_norm=ln_norm,
        coeffs=coeffs,
        d1=d1,
        d2=d2,
        moments=moments,
        zbar=zbar,
        central=central,
    )
