"""Special-function helpers built on :mod:`scipy.special`.

SciPy provides ``log_ndtr``, ``i0e``, ``i1e``, ``erfcx`` and ``logsumexp``; with
``SCIPY_ARRAY_API=1`` set before SciPy is imported, they dispatch to the native
library of JAX, torch and CuPy arrays and are traceable by ``jax.jit``/``jax.grad``.
This module only adds what SciPy does not provide:

- :func:`log1mexp` and :func:`log_diff_ndtr` (``ln(Phi(b) - Phi(a))`` without
  cancellation);
- :func:`mills`, the zeroth to second moments of ``exp(-a t - t^2/2)`` on
  ``[0, inf)``, needed for a truncated-normal variance that stays accurate in the far
  tails (``scipy.stats.truncnorm.stats`` returns a negative variance at
  ``a = 1e4``, and ``scipy.stats.truncate(Normal())`` is not JAX-traceable);
- :func:`ln_i0_series_coefficients`, the power series of ``ln I0``.
"""
from __future__ import annotations

import math
import os
from fractions import Fraction

from scipy.special import erfcx, i0e, i1e, log_ndtr, logsumexp  # noqa: F401 (re-exported)

LN_2PI = math.log(2.0 * math.pi)
HALF_LN_2PI = 0.5 * LN_2PI
LN2 = math.log(2.0)
SQRT_HALF_PI = math.sqrt(0.5 * math.pi)

#: Mills-ratio continued fraction: switch point and (static) depth.
MILLS_SWITCH = 3.0
MILLS_DEPTH = 60


def scipy_array_api_enabled() -> bool:
    """Report whether SciPy's array API support is active.

    Returns
    -------
    bool
        ``True`` if SciPy was imported with ``SCIPY_ARRAY_API=1``.  Without it,
        :mod:`scipy.special` converts non-numpy arrays to numpy, which breaks
        JAX tracing and torch autograd.
    """
    try:
        from scipy._lib._array_api import SCIPY_ARRAY_API

        return bool(SCIPY_ARRAY_API)
    except ImportError:  # pragma: no cover - private location moved
        return os.environ.get("SCIPY_ARRAY_API", "0") == "1"


def log1mexp(x, xp):
    """Compute ``ln(1 - exp(x))`` accurately for ``x <= 0``.

    Parameters
    ----------
    x : array
        Non-positive argument.
    xp : namespace
        Array namespace of ``x``.

    Returns
    -------
    array
        ``ln(1 - exp(x))``, using ``log(-expm1(x))`` for ``x > -ln 2`` and
        ``log1p(-exp(x))`` otherwise (Maechler 2012).  Both branches are evaluated
        on clipped inputs so neither produces NaN gradients.
    """
    x_lo = xp.clip(x, None, -LN2)
    x_hi = xp.clip(x, -LN2, -1e-300)
    return xp.where(x > -LN2, xp.log(-xp.expm1(x_hi)), xp.log1p(-xp.exp(x_lo)))


def ln_phi(x):
    """Compute the log of the standard normal density.

    Parameters
    ----------
    x : array or float
        Argument.

    Returns
    -------
    array or float
        ``-x^2/2 - ln sqrt(2 pi)``.
    """
    return -0.5 * x * x - HALF_LN_2PI


def log_diff_ndtr(a, b, xp):
    """Compute ``ln(Phi(b) - Phi(a))`` for ``a < b``.

    Parameters
    ----------
    a, b : array
        Lower and upper limits with ``a < b``.
    xp : namespace
        Array namespace of the inputs.

    Returns
    -------
    array
        The log of the standard normal mass on ``[a, b]``.  When ``a > 0`` the
        interval is reflected to ``(-b, -a)`` so both CDFs are evaluated in the
        lower tail.  Very narrow intervals (``b - a << 1``) lose relative accuracy
        but stay finite.
    """
    flip = a > 0
    lo = xp.where(flip, -b, a)
    hi = xp.where(flip, -a, b)
    la = log_ndtr(lo)
    lb = log_ndtr(hi)
    return lb + log1mexp(xp.clip(la - lb, None, 0.0), xp)


def mills(a, xp):
    """Compute moments of the truncated exponential-Gaussian kernel for ``a >= 0``.

    Parameters
    ----------
    a : array
        Non-negative standardized truncation point.
    xp : namespace
        Array namespace of ``a``.

    Returns
    -------
    m, u, w : array
        ``m = int_0^inf e^{-a t - t^2/2} dt`` (the Mills ratio
        ``sqrt(pi/2) erfcx(a/sqrt 2)``), ``u = int t e^{...} dt = 1 - a m`` and
        ``w = int t^2 e^{...} dt = m - a u``.

    Notes
    -----
    For ``a >= MILLS_SWITCH`` the differences ``1 - a m`` and ``m - a u`` cancel
    catastrophically, so the Laplace continued fraction ``m = 1/(a + D_1)``,
    ``D_k = k/(a + D_{k+1})`` is used instead, giving ``u = m D_1`` and
    ``w = m D_1 D_2``.  The loop has a static length so it traces under JAX.
    """
    a_s = xp.clip(a, 0.0, MILLS_SWITCH)
    m_s = SQRT_HALF_PI * erfcx(a_s / math.sqrt(2.0))
    u_s = 1.0 - a_s * m_s
    w_s = m_s - a_s * u_s

    a_l = xp.clip(a, MILLS_SWITCH, None)
    d_next = xp.zeros_like(a_l)
    d2 = d_next
    for k in range(MILLS_DEPTH, 0, -1):
        d_k = k / (a_l + d_next)
        if k == 2:
            d2 = d_k
        d_next = d_k
    d1 = d_next
    m_l = 1.0 / (a_l + d1)
    u_l = m_l * d1
    w_l = m_l * d1 * d2

    small = a < MILLS_SWITCH
    return (
        xp.where(small, m_s, m_l),
        xp.where(small, u_s, u_l),
        xp.where(small, w_s, w_l),
    )


def logsumexp_lastaxis(x, xp):
    """Compute ``ln sum exp(x)`` over the last axis with a max shift.

    Parameters
    ----------
    x : array
        Finite log terms.
    xp : namespace
        Array namespace of ``x``.

    Returns
    -------
    array
        Reduction over the last axis.

    Notes
    -----
    :func:`scipy.special.logsumexp` (SciPy 1.18, ``SCIPY_ARRAY_API=1``) returns a NaN
    JAX gradient when every term except the maximum underflows (terms spanning
    ~1e5 nats, as in the outer quadrature panels at moderate SNR).  This plain
    max-shifted form has well-defined gradients there.
    """
    mx = xp.max(x, axis=-1, keepdims=True)
    return mx[..., 0] + xp.log(xp.sum(xp.exp(x - mx), axis=-1))


def ln_i0_series_coefficients(n_terms: int) -> list[float]:
    """Compute the power-series coefficients of ``ln I0``.

    Parameters
    ----------
    n_terms : int
        Number of coefficients to return.

    Returns
    -------
    list of float
        ``c_1 .. c_n`` with ``ln I0(x) = sum_k c_k x^{2k}``: ``1/4, -1/64, 1/576,
        -11/49152, ...``.  They are computed exactly with :class:`fractions.Fraction`
        from the logarithm of ``I0 = sum (x^2/4)^k / (k!)^2``.
    """
    a = [Fraction(1, 4**k * math.factorial(k) ** 2) for k in range(n_terms + 1)]
    f = [Fraction(0)] * (n_terms + 1)
    for n in range(1, n_terms + 1):
        acc = a[n]
        for k in range(1, n):
            acc -= Fraction(k, n) * f[k] * a[n - k]
        f[n] = acc
    return [float(c) for c in f[1:]]
