"""Reference quadrature for tests and benchmarks (IMPLEMENTATION_NOTES §5).

Composite Gauss-Legendre in ``s = ln v`` over the union of a global grid, a dense
grid around the clipped peak and, when the peak lies outside the support, a dense
grid over the edge tail scale ``lambda = sigma^2 / |v* - v_edge|``.  This is
deliberately much denser than the six-panel rule used in evaluation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import i0e, logsumexp

from ._prior import PriorFit

_PEAK_MULTIPLES = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 25, 30, 40, 50, 60, 80])
_TAIL_MULTIPLES = np.concatenate([np.geomspace(1e-3, 1, 13), np.arange(1.5, 10.5, 0.5), np.arange(12, 82, 2)])


def reference_ln_likelihood_ratio(fit: PriorFit, phase: bool, rho_mf: float, rho_opt: float, d_ref: float,
                                  n_global: int = 600, order: int = 12) -> float:
    """Compute ``ln(L_D / Z_N)`` by dense composite Gauss-Legendre quadrature.

    Parameters
    ----------
    fit : PriorFit
        Fitted prior (the same continuous prior the evaluation uses).
    phase : bool
        Phase marginalization.
    rho_mf, rho_opt, d_ref : float
        As in :meth:`distmarg.DistanceMarginalization.ln_likelihood_ratio`.
    n_global : int, optional
        Number of uniform panels in ``s`` over the whole support.
    order : int, optional
        Gauss-Legendre nodes per panel.

    Returns
    -------
    float
        Reference value, accurate to ~1e-12 relative to the integral.
    """
    rm = abs(float(rho_mf)) if phase else float(rho_mf)
    k = float(rho_opt) * float(d_ref)
    amp, bb = rm * k, k * k
    sigma = 1.0 / k
    vstar = rm * sigma
    vmin, vmax = fit.v_min, fit.v_max
    centre = min(max(vstar, vmin), vmax)
    pts = [np.exp(np.linspace(fit.s_a, fit.s_b, n_global + 1))]
    pts.append(centre + sigma * np.concatenate([-_PEAK_MULTIPLES, [0.0], _PEAK_MULTIPLES]))
    dist = abs(vstar - centre)
    if dist > 0:
        lam = sigma * sigma / dist
        sign = 1.0 if centre == vmin else -1.0
        pts.append(centre + sign * lam * _TAIL_MULTIPLES)
    s = np.unique(np.clip(np.log(np.clip(np.concatenate(pts), vmin, vmax)), fit.s_a, fit.s_b))
    s[0], s[-1] = fit.s_a, fit.s_b
    x, w = np.polynomial.legendre.leggauss(order)
    half = 0.5 * np.diff(s)
    mid = 0.5 * (s[1:] + s[:-1])
    keep = half > 0
    half, mid = half[keep], mid[keep]
    nodes = (mid[:, None] + half[:, None] * x).ravel()
    weights = (half[:, None] * w).ravel()
    vn = np.exp(nodes)
    f = fit.ln_pi_v(nodes) + nodes + amp * vn - 0.5 * bb * vn * vn
    if phase:
        f = f + np.log(i0e(amp * vn))
    return float(logsumexp(f, b=weights))
