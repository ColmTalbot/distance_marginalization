"""Distance-marginalized likelihood ratio, ``ln(L_D / Z_N)``.

See ``IMPLEMENTATION_NOTES.md`` (§2-§4) and ``SPEC_DECISIONS.md`` at the
repository root for the mathematics and the exact regime-selection rule shared
with the Rust implementation.

Special functions come from :mod:`scipy.special`.  For non-numpy arrays (JAX,
torch, CuPy) SciPy's array API support must be enabled by setting
``SCIPY_ARRAY_API=1`` before SciPy is first imported.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from array_api_compat import array_namespace, is_numpy_namespace
from numpy.polynomial.chebyshev import chebval
from scipy.special import i0e, i1e

from . import _special as sp
from ._prior import PriorFit, fit_prior

REGIME_HIGH_SNR = 0
REGIME_PRIOR_DOMINATED = 1
REGIME_QUADRATURE = 2
REGIME_NAMES = {REGIME_HIGH_SNR: "high_snr", REGIME_PRIOR_DOMINATED: "prior_dominated", REGIME_QUADRATURE: "quadrature"}

#: Break points of the middle-regime quadrature, in units of the peak scale.
WINDOW = (-40.0, -8.0, 0.0, 8.0, 40.0)
#: Number of terms of ``ln I0(x) = sum c_k x^{2k}`` kept in the phase cumulant series.
N_LN_I0_TERMS = 6
#: Input types routed to the Rust scalar backend (float32 is excluded: it must warn).
_SCALAR_TYPES = (float, int, np.float64, np.int64)


def _is_scalar(x):
    """Report whether ``x`` can take the Rust scalar path.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
        True for Python ``float``/``int``, ``numpy.float64``/``int64`` and 0-d
        float64 numpy arrays.
    """
    t = type(x)
    return t in _SCALAR_TYPES or (t is np.ndarray and x.ndim == 0 and x.dtype == np.float64)


def _is_numpy64(x):
    """Report whether ``x`` can take the Rust array path.

    Parameters
    ----------
    x : object

    Returns
    -------
    bool
        True for scalars accepted by :func:`_is_scalar` and numpy arrays of dtype
        float64 or integer (float32 is excluded so that it keeps its warning).
    """
    if _is_scalar(x):
        return True
    return type(x) is np.ndarray and (x.dtype == np.float64 or x.dtype.kind in "iu")


#: Standardized interval length beyond which the far truncation point is irrelevant.
_L_CLIP = 60.0


class DistanceMarginalization:
    r"""Distance-marginalized likelihood ratio for a tabulated luminosity-distance prior.

    Parameters
    ----------
    distance, ln_prior : array_like
        Luminosity distance (finite, ``> 0``, strictly increasing) and the natural-log
        prior density at those points (finite everywhere; need not be normalized).
        Any array namespace is accepted; set-up is done in numpy float64.  Pass a
        dense, smooth tabulation (2-4k points log-spaced in distance).
    phase_marginalization : bool, optional
        Marginalize analytically over a uniform (2,2)-mode phase.
    fit_degree : int, optional
        Chebyshev degree of the fit of ``ln pi_v(s)``; ``None`` selects the smallest
        of 8, 16, ..., 128 whose max residual is below ``fit_tol``.
    fit_tol : float, optional
        Fit residual above which a :class:`distmarg.FitWarning` is emitted.
    rho_high : float, optional
        ``|rho_mf|`` threshold for the high-SNR (Laplace + curvature) branch.
    kappa2_low, cumulant_range_max : float, optional
        The prior-dominated (cumulant-series) branch is used when
        ``kappa_2 = Var_pi[ln ell] < kappa2_low`` *and* the range of ``ln ell`` over
        the support is ``<= cumulant_range_max``.
    cumulant_order : int, optional
        Order of the cumulant series (1-4).
    quad_nodes : int, optional
        Gauss-Legendre nodes per panel of the six-panel middle-regime quadrature.
    quad_grid : int, optional
        Points of the uniform grid in ``s`` on which ``ln pi_v`` and its derivative
        are tabulated for cubic-Hermite interpolation inside the quadrature;
        ``0`` evaluates the Chebyshev series directly (slower, exact).
    high_max_rel_width : float, optional
        The high-SNR branch also requires ``sqrt(Var_TN) <= high_max_rel_width * vbar``.
    phase_series_xmax : float, optional
        With phase marginalization the cumulant branch also requires
        ``|<d|h>| d_ref / d_min <= phase_series_xmax`` (``ln I0`` series validity).
    backend : {"auto", "rust", "python"}, optional
        Backend for numpy (and plain Python) inputs.  ``"rust"`` uses the compiled
        ``distmarg_rs`` extension (built from ``rust/distmarg`` with maturin):
        scalars avoid the ~0.5 ms numpy dispatch overhead of a single-point call and
        float64 arrays are evaluated element-wise in Rust (zero-copy, GIL released,
        multi-threaded).  ``"auto"`` uses it when importable and otherwise falls back
        to the pure-Python path.  JAX, torch, CuPy and float32 inputs always use the
        Python path.  The two agree to ~1e-9: they are the independent
        implementations cross-checked by ``tools/cross_check.py``.

    Notes
    -----
    Inputs to :meth:`ln_likelihood_ratio`:

    - ``rho_opt = sqrt(<h|h>)`` evaluated at ``d_ref`` (must be ``> 0``);
    - ``rho_mf = Re<d|h> / rho_opt`` (may be negative) without phase
      marginalization, or ``|<d|h>| / rho_opt`` with it (the absolute value is taken).

    The output excludes the noise log-likelihood ``ln Z_N``.  From Bilby's
    ``d_inner_h`` and ``optimal_snr_squared`` at the reference distance use
    ``rho_mf = d_inner_h / sqrt(optimal_snr_squared)`` and
    ``rho_opt = sqrt(optimal_snr_squared)``.

    Evaluation has no value-dependent Python control flow, so ``jax.jit``,
    ``jax.vmap`` and ``jax.grad`` work.  Use float64: at ``rho = 1e4``,
    ``rho^2 / 2 = 5e7`` and float32 has an absolute error of ~4 there.  With JAX
    enable ``jax_enable_x64``.  For time marginalization evaluate on arrays over
    time bins and log-sum-exp the result.
    """

    def __init__(
        self,
        distance,
        ln_prior,
        phase_marginalization: bool = False,
        *,
        fit_degree: int | None = None,
        fit_tol: float = 1e-8,
        rho_high: float = 30.0,
        kappa2_low: float = 0.05,
        cumulant_range_max: float = 1.0,
        cumulant_order: int = 4,
        quad_nodes: int = 20,
        quad_grid: int = 64,
        high_max_rel_width: float = 0.01,
        phase_series_xmax: float = 1.0,
        backend: str = "auto",
    ):
        """Fit the prior and store the evaluation constants (see the class docstring)."""
        if not 1 <= int(cumulant_order) <= 4:
            raise ValueError("cumulant_order must be between 1 and 4")
        if int(quad_nodes) < 2:
            raise ValueError("quad_nodes must be >= 2")
        if int(quad_grid) != 0 and int(quad_grid) < 2:
            raise ValueError("quad_grid must be 0 or >= 2")
        self.phase_marginalization = bool(phase_marginalization)
        self.prior: PriorFit = fit_prior(distance, ln_prior, fit_degree=fit_degree, fit_tol=fit_tol)
        self.rho_high = float(rho_high)
        self.kappa2_low = float(kappa2_low)
        self.cumulant_range_max = float(cumulant_range_max)
        self.cumulant_order = int(cumulant_order)
        self.quad_nodes = int(quad_nodes)
        self.quad_grid = int(quad_grid)
        self.high_max_rel_width = float(high_max_rel_width)
        self.phase_series_xmax = float(phase_series_xmax)

        p = self.prior
        self.v_min, self.v_max = p.v_min, p.v_max
        key = "nu2" if self.phase_marginalization else "nu"
        self._zbar = float(p.zbar[key])
        self._central = [float(x) for x in p.central[key]]
        nu0 = self.v_min / self.v_max
        self._z0 = nu0 * nu0 if self.phase_marginalization else nu0
        self._ln_i0_c = sp.ln_i0_series_coefficients(N_LN_I0_TERMS)
        x_gl, w_gl = np.polynomial.legendre.leggauss(self.quad_nodes)
        self._gl_x_np, self._gl_lnw_np = x_gl, np.log(w_gl)
        self._rust = self._make_rust(distance, ln_prior, backend)
        if self.quad_grid:
            s_grid = np.linspace(p.s_a, p.s_b, self.quad_grid)
            self._grid_h = (p.s_b - p.s_a) / (self.quad_grid - 1)
            self._grid_g = chebval(self._t_of_s(s_grid), p.coeffs)
            self._grid_dg = chebval(self._t_of_s(s_grid), p.d1)

    def _make_rust(self, distance, ln_prior, backend):
        """Build the Rust backend.

        Parameters
        ----------
        distance, ln_prior : array_like
            Prior arrays, as passed to the constructor.
        backend : {"auto", "rust", "python"}
            Requested scalar backend.

        Returns
        -------
        distmarg_rs.DistanceMarginalization or None

        Raises
        ------
        ImportError
            If ``backend="rust"`` and the extension is not installed.
        ValueError
            For an unknown ``backend``.
        """
        if backend not in ("auto", "rust", "python"):
            raise ValueError("backend must be 'auto', 'rust' or 'python'")
        if backend == "python":
            return None
        try:
            import distmarg_rs
        except ImportError:
            if backend == "rust":
                raise
            return None
        from ._prior import validate_prior_arrays

        d, lp = validate_prior_arrays(distance, ln_prior)
        return distmarg_rs.DistanceMarginalization(
            d.tolist(), lp.tolist(), self.phase_marginalization,
            fit_degree=self.prior.degree, fit_tol=math.inf, rho_high=self.rho_high,
            kappa2_low=self.kappa2_low, cumulant_range_max=self.cumulant_range_max,
            cumulant_order=self.cumulant_order, quad_nodes=self.quad_nodes,
            quad_grid=self.quad_grid, high_max_rel_width=self.high_max_rel_width,
            phase_series_xmax=self.phase_series_xmax,
        )

    @property
    def backend(self) -> str:
        """str: ``"rust"`` if numpy inputs use the compiled extension, else ``"python"``."""
        return "python" if self._rust is None else "rust"

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_bilby_prior(cls, prior, n: int = 4096, phase_marginalization: bool = False, **kwargs):
        """Build from a Bilby prior on luminosity distance.

        Parameters
        ----------
        prior : bilby.core.prior.Prior
            Luminosity-distance prior; see :func:`distmarg.bilby.prior_to_arrays`.
        n : int, optional
            Number of log-spaced tabulation points.
        phase_marginalization : bool, optional
            Marginalize over a uniform (2,2)-mode phase.
        **kwargs
            Passed to the constructor.

        Returns
        -------
        DistanceMarginalization
        """
        from .bilby import prior_to_arrays

        distance, ln_prior = prior_to_arrays(prior, n=n)
        return cls(distance, ln_prior, phase_marginalization, **kwargs)

    @property
    def fit_info(self) -> dict:
        """dict: Degree, max residual, raw coefficients, ln normalization and ``E[v^k]`` (``k <= 8``)."""
        return self.prior.fit_info()

    # ------------------------------------------------------------------ public API
    def ln_likelihood_ratio(self, rho_mf, rho_opt, d_ref):
        """Evaluate the distance-marginalized log likelihood ratio ``ln(L_D / Z_N)``.

        Parameters
        ----------
        rho_mf : array_like
            Matched-filter SNR ``<d|h> / rho_opt`` (real part without phase
            marginalization, modulus with it).
        rho_opt : array_like
            Optimal SNR at ``d_ref``.
        d_ref : array_like
            Reference distance at which both SNRs were computed.

        Returns
        -------
        array or numpy.float64
            Broadcast result in the namespace of the inputs; NaN where
            ``rho_opt <= 0``, ``d_ref <= 0`` or any input is non-finite.  With the
            Rust backend (see ``backend``) scalar inputs return a ``numpy.float64``
            and numpy inputs a float64 ``numpy.ndarray``.
        """
        if self._rust is not None:
            if _is_scalar(rho_mf) and _is_scalar(rho_opt) and _is_scalar(d_ref):
                return np.float64(self._rust.ln_likelihood_ratio(float(rho_mf), float(rho_opt), float(d_ref)))
            if _is_numpy64(rho_mf) and _is_numpy64(rho_opt) and _is_numpy64(d_ref):
                a, b, c = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (rho_mf, rho_opt, d_ref)))
                out = self._rust.ln_likelihood_ratio_array(
                    np.ascontiguousarray(a).ravel(), np.ascontiguousarray(b).ravel(), np.ascontiguousarray(c).ravel()
                )
                return out.reshape(a.shape)
        out, _ = self._evaluate(rho_mf, rho_opt, d_ref)
        return out

    def regime(self, rho_mf, rho_opt, d_ref):
        """Report which evaluation branch each input uses.

        Parameters
        ----------
        rho_mf, rho_opt, d_ref : array_like
            As in :meth:`ln_likelihood_ratio`.

        Returns
        -------
        array of int
            0 = high SNR, 1 = prior dominated, 2 = quadrature, -1 = invalid input.
        """
        xp, rm, ro, dr = self._as_arrays(rho_mf, rho_opt, d_ref)
        valid, rm, k = self._sanitize(xp, rm, ro, dr)
        reg = self._select(xp, rm, k)
        return xp.where(valid, reg, -1)

    # ------------------------------------------------------------------ internals
    def _as_arrays(self, *args):
        """Convert inputs to broadcast floating arrays of a common namespace.

        Parameters
        ----------
        *args : array_like
            ``rho_mf``, ``rho_opt``, ``d_ref``.

        Returns
        -------
        tuple
            ``(xp, rho_mf, rho_opt, d_ref)``.

        Raises
        ------
        RuntimeError
            For non-numpy arrays when SciPy's array API support is disabled.
        """
        arrays = [a for a in args if hasattr(a, "__array_namespace__") or hasattr(a, "shape")]
        xp = array_namespace(*arrays) if arrays else array_namespace(np.asarray(0.0))
        if not is_numpy_namespace(xp) and not sp.scipy_array_api_enabled():
            raise RuntimeError(
                "non-numpy arrays need SciPy's array API support: set SCIPY_ARRAY_API=1 "
                "before scipy is first imported"
            )
        conv = [xp.asarray(a) for a in args]
        dtype = conv[0].dtype
        for a in conv[1:]:
            dtype = xp.result_type(dtype, a.dtype)
        if not xp.isdtype(dtype, "real floating"):
            dtype = xp.float64
        elif dtype == xp.float32:
            warnings.warn(
                "float32 inputs: ln L is ~rho^2/2 and loses absolute accuracy; use float64 "
                "(with JAX set jax_enable_x64)",
                stacklevel=3,
            )
        conv = [xp.astype(a, dtype) for a in conv]
        conv = xp.broadcast_arrays(*conv)
        return (xp, *conv)

    def _gl(self, xp, like):
        """Return Gauss-Legendre nodes and log weights matching ``like``.

        Parameters
        ----------
        xp : namespace
            Target array namespace.
        like : array
            Array whose dtype and device are used.

        Returns
        -------
        nodes, log_weights : array

        Notes
        -----
        Not cached: under ``jax.jit`` a cached array would be a leaked tracer.
        """
        kwargs = {"dtype": like.dtype}
        dev = getattr(like, "device", None)
        if dev is not None and not is_numpy_namespace(xp):
            kwargs["device"] = dev
        return xp.asarray(self._gl_x_np, **kwargs), xp.asarray(self._gl_lnw_np, **kwargs)

    def _ln_pi_v_grid(self, xp, s):
        """Interpolate ``ln pi_v`` by cubic Hermite on the uniform grid in ``s``.

        Parameters
        ----------
        xp : namespace
        s : array
            ``ln v`` inside ``[s_a, s_b]``.

        Returns
        -------
        array
            ``ln pi_v(e^s)``, C1-continuous in ``s`` (so gradients are smooth).

        Notes
        -----
        With the exact derivative at the knots the interpolation error is
        ``O(h^4 g'''')``; 64 knots are indistinguishable from the Chebyshev series at
        the quadrature's 6e-5 accuracy (SPEC_DECISIONS.md §6).
        """
        n = self.quad_grid
        kwargs = {"dtype": s.dtype}
        dev = getattr(s, "device", None)
        if dev is not None and not is_numpy_namespace(xp):
            kwargs["device"] = dev
        g = xp.asarray(self._grid_g, **kwargs)
        dg = xp.asarray(self._grid_dg, **kwargs)
        h = self._grid_h
        u = (s - self.prior.s_a) / h
        i = xp.clip(xp.floor(u), 0.0, float(n - 2))
        w = u - i
        idx = xp.reshape(xp.astype(i, xp.int64), (-1,))
        shape = s.shape
        g0 = xp.reshape(xp.take(g, idx, axis=0), shape)
        g1 = xp.reshape(xp.take(g, idx + 1, axis=0), shape)
        d0 = xp.reshape(xp.take(dg, idx, axis=0), shape)
        d1 = xp.reshape(xp.take(dg, idx + 1, axis=0), shape)
        w2 = w * w
        w3 = w2 * w
        return ((2.0 * w3 - 3.0 * w2 + 1.0) * g0 + (w3 - 2.0 * w2 + w) * h * d0
                + (3.0 * w2 - 2.0 * w3) * g1 + (w3 - w2) * h * d1)

    def _sanitize(self, xp, rm, ro, dr):
        """Mask invalid inputs and form ``k = rho_opt * d_ref``.

        Parameters
        ----------
        xp : namespace
        rm, ro, dr : array
            ``rho_mf``, ``rho_opt``, ``d_ref``.

        Returns
        -------
        valid : array of bool
            ``rho_opt > 0``, ``d_ref > 0`` and all inputs finite.
        rho : array
            ``rho_mf`` (``|rho_mf|`` with phase marginalization), 1 where invalid.
        k : array
            ``rho_opt * d_ref`` (``1 / sigma_v``), 1 where invalid.
        """
        valid = (ro > 0) & (dr > 0) & xp.isfinite(rm) & xp.isfinite(ro) & xp.isfinite(dr)
        rm = xp.where(valid, rm, 1.0)
        ro = xp.where(valid, ro, 1.0)
        dr = xp.where(valid, dr, 1.0)
        if self.phase_marginalization:
            rm = xp.abs(rm)
        return valid, rm, ro * dr

    def _t_of_s(self, s):
        """Map ``s = ln v`` to the Chebyshev domain.

        Parameters
        ----------
        s : array

        Returns
        -------
        array
        """
        p = self.prior
        return (2.0 * s - p.s_a - p.s_b) / (p.s_b - p.s_a)

    # -------------------------------------------------------------- truncated normal
    def _truncnorm(self, xp, vstar, sigma):
        """Compute the mass, mean and variance of ``N(v*, sigma)`` truncated to the support.

        Parameters
        ----------
        xp : namespace
        vstar : array
            Gaussian centre ``v*``.
        sigma : array
            Gaussian width.

        Returns
        -------
        ln_mass : array
            ``ln[sqrt(2 pi) sigma (Phi(beta) - Phi(alpha))]``.
        vbar : array
            Truncated-normal mean, clipped to ``[v_min, v_max]``.
        var : array
            Truncated-normal variance, clipped to ``[0, (v_max - v_min)^2 / 4]``.

        Notes
        -----
        Neither ``scipy.stats.truncnorm`` (negative variance at ``a = 1e4``) nor
        ``scipy.stats.truncate(Normal())`` (20% variance error there; not
        JAX-traceable) is usable.  When the interval lies in a tail (``a >= 0`` after
        orienting it so ``a + b >= 0``) the moments of ``t = x - a`` are formed from
        :func:`distmarg._special.mills`, which avoids the cancellation of the textbook
        formula; otherwise the textbook formula is used with
        :func:`distmarg._special.log_diff_ndtr`.
        """
        vmin, vmax = self.v_min, self.v_max
        alpha = (vmin - vstar) / sigma
        beta = (vmax - vstar) / sigma
        flip = (alpha + beta) < 0
        a = xp.where(flip, -beta, alpha)
        b = xp.where(flip, -alpha, beta)
        length = b - a
        tail = a >= 0

        # tail case (0 <= a < b): density of t = x - a is ~ exp(-a t - t^2/2) on [0, L]
        a_t = xp.where(tail, a, 0.0)
        l_t = xp.clip(length, None, _L_CLIP)
        b_t = a_t + l_t
        m_a, u_a, w_a = sp.mills(a_t, xp)
        m_b, u_b, w_b = sp.mills(b_t, xp)
        q = xp.exp(-a_t * l_t - 0.5 * l_t * l_t)
        i0 = xp.maximum(m_a - q * m_b, l_t * q)
        i1 = u_a - q * (u_b + l_t * m_b)
        i2 = w_a - q * (w_b + 2.0 * l_t * u_b + l_t * l_t * m_b)
        mean_t = i1 / i0
        var_tail = i2 / i0 - mean_t * mean_t
        mean_tail = a_t + mean_t
        lnz_tail = -0.5 * a_t * a_t + xp.log(i0)  # ln Z + ln sqrt(2 pi)

        # straddling case (a < 0 < b)
        a_s = xp.where(tail, -1.0, a)
        b_s = xp.where(tail, 1.0, b)
        lnz_s = sp.log_diff_ndtr(a_s, b_s, xp)
        r_a = xp.exp(sp.ln_phi(a_s) - lnz_s)
        r_b = xp.exp(sp.ln_phi(b_s) - lnz_s)
        mean_s = r_a - r_b
        var_s = 1.0 + a_s * r_a - b_s * r_b - mean_s * mean_s

        mean = xp.where(tail, mean_tail, mean_s)
        mean = xp.where(flip, -mean, mean)
        var_x = xp.clip(xp.where(tail, var_tail, var_s), 0.0, None)
        lnz = xp.where(tail, lnz_tail, lnz_s + sp.HALF_LN_2PI)

        vbar = xp.clip(vstar + sigma * mean, vmin, vmax)
        var = xp.clip(var_x * sigma * sigma, None, 0.25 * (vmax - vmin) ** 2)
        return lnz + xp.log(sigma), vbar, var

    # -------------------------------------------------------------- regime selection
    def _polynomial(self, a_p, b_p):
        """Express ``ln ell`` as a polynomial in ``z``.

        Parameters
        ----------
        a_p : array
            ``A' = A v_max``.
        b_p : array
            ``B' = B v_max^2``.

        Returns
        -------
        list
            Coefficients ``p_0 .. p_D`` in ``z = nu`` (no phase, ``D = 2``) or
            ``z = nu^2`` (phase, ``D = 6``, truncated ``ln I0`` series).
        """
        if self.phase_marginalization:
            a2 = a_p * a_p
            p = [0.0, self._ln_i0_c[0] * a2 - 0.5 * b_p]
            power = a2
            for c in self._ln_i0_c[1:]:
                power = power * a2
                p.append(c * power)
            return p
        return [0.0, a_p, -0.5 * b_p]

    @staticmethod
    def _horner(p, z):
        """Evaluate a polynomial by Horner's rule.

        Parameters
        ----------
        p : list
            Coefficients, lowest order first.
        z : array

        Returns
        -------
        array
        """
        out = p[-1]
        for c in p[-2::-1]:
            out = out * z + c
        return out

    def _ln_ell_range(self, xp, p):
        """Compute the range of ``ln ell(z)`` over ``[z0, 1]``.

        Parameters
        ----------
        xp : namespace
        p : list
            Polynomial coefficients from :meth:`_polynomial`.

        Returns
        -------
        array
            ``max(P(z0), P(1), P(z_c)) - min(P(z0), P(1))`` with ``z_c`` the clipped
            vertex of the quadratic part.
        """
        z0 = self._z0
        z_c = xp.clip(p[1] / (2.0 * xp.clip(-p[2], 1e-300, None)), z0, 1.0)
        f0 = self._horner(p, z0 + xp.zeros_like(z_c))
        f1 = self._horner(p, 1.0 + xp.zeros_like(z_c))
        fc = self._horner(p, z_c)
        return xp.maximum(xp.maximum(f0, f1), fc) - xp.minimum(f0, f1)

    def _kappa2(self, xp, p):
        """Compute ``kappa_2 = Var_pi[ln ell]`` from the central moments.

        Parameters
        ----------
        xp : namespace
        p : list
            Polynomial coefficients from :meth:`_polynomial`.

        Returns
        -------
        array
            ``sum_ij y_i y_j C_{i+j}`` (a positive semi-definite form, so no
            catastrophic cancellation for large ``A'``, ``B'``).
        """
        y = self._centered_polynomial(xp, p)[1]
        c = self._central
        n = len(y)
        total = 0.0
        for i in range(n):
            for j in range(n):
                if c[i + j] != 0.0:
                    total = total + y[i] * y[j] * c[i + j]
        return total

    def _select(self, xp, rm, k):
        """Apply the regime-selection rule of SPEC_DECISIONS.md §4.

        Parameters
        ----------
        xp : namespace
        rm : array
            ``rho_mf`` (absolute value with phase marginalization).
        k : array
            ``rho_opt * d_ref``.

        Returns
        -------
        array of int
            Regime codes.
        """
        a_p = rm * k * self.v_max
        b_p = (k * self.v_max) ** 2
        if self.phase_marginalization:
            p = self._polynomial(xp.clip(a_p, None, self.phase_series_xmax), b_p)
            prior_dom = a_p <= self.phase_series_xmax
        else:
            p = self._polynomial(a_p, b_p)
            prior_dom = xp.ones_like(a_p, dtype=xp.bool)
        prior_dom = prior_dom & (self._kappa2(xp, p) < self.kappa2_low)
        prior_dom = prior_dom & (self._ln_ell_range(xp, p) <= self.cumulant_range_max)
        sigma = 1.0 / k
        _, vbar, var = self._truncnorm(xp, rm * sigma, sigma)
        high = (xp.abs(rm) >= self.rho_high) & (xp.sqrt(var) <= self.high_max_rel_width * vbar)
        reg = xp.where(high, REGIME_HIGH_SNR, REGIME_QUADRATURE)
        return xp.where(prior_dom, REGIME_PRIOR_DOMINATED, reg)

    # -------------------------------------------------------------- branches
    def _centered_polynomial(self, xp, p):
        """Re-expand ``ln ell = sum p_k z^k`` about ``E[z]``.

        Parameters
        ----------
        xp : namespace
        p : list
            Polynomial coefficients from :meth:`_polynomial`.

        Returns
        -------
        kappa1 : array
            ``E_pi[ln ell]``.
        y : list
            Coefficients of ``ln ell - kappa1`` in ``delta = z - E[z]``.
        """
        deg = len(p) - 1
        zb = self._zbar
        shifted = []
        for j in range(deg + 1):
            acc = 0.0
            for kk in range(max(j, 1), deg + 1):
                acc = acc + math.comb(kk, j) * zb ** (kk - j) * p[kk]
            shifted.append(acc)
        c = self._central
        kappa1 = shifted[0]
        y0 = 0.0
        for j in range(2, deg + 1):
            kappa1 = kappa1 + shifted[j] * c[j]
            y0 = y0 - shifted[j] * c[j]
        return kappa1, [y0] + shifted[1:]

    def _branch_cumulant(self, xp, rm, k):
        """Evaluate the prior-dominated branch (centred cumulant series).

        Parameters
        ----------
        xp : namespace
        rm, k : array
            As in :meth:`_select`.

        Returns
        -------
        array
            ``kappa_1 + kappa_2/2 + kappa_3/6 + kappa_4/24`` up to ``cumulant_order``.
        """
        a_p = rm * k * self.v_max
        b_p = (k * self.v_max) ** 2
        kappa1, y = self._centered_polynomial(xp, self._polynomial(a_p, b_p))
        c = self._central
        out = kappa1
        power = y
        mu = {}
        for n in range(2, self.cumulant_order + 1):
            new = []
            for i in range(len(power) + len(y) - 1):
                acc = 0.0
                for j in range(max(0, i - len(y) + 1), min(i, len(power) - 1) + 1):
                    acc = acc + power[j] * y[i - j]
                new.append(acc)
            power = new
            total = 0.0
            for i, coef in enumerate(power):
                if c[i] != 0.0:
                    total = total + coef * c[i]
            mu[n] = total
        if self.cumulant_order >= 2:
            out = out + mu[2] / 2.0
        if self.cumulant_order >= 3:
            out = out + mu[3] / 6.0
        if self.cumulant_order >= 4:
            out = out + (mu[4] - 3.0 * mu[2] * mu[2]) / 24.0
        return out

    def _branch_high(self, xp, rm, k):
        """Evaluate the high-SNR branch (truncated-normal mass plus curvature).

        Parameters
        ----------
        xp : namespace
        rm, k : array
            As in :meth:`_select`.

        Returns
        -------
        array
            ``rho^2/2 + ln M + ln pi_v(vbar) + ln w(vbar) + Var/2 [h'' + h'^2]``.
        """
        p = self.prior
        sigma = 1.0 / k
        ln_mass, vbar, var = self._truncnorm(xp, rm * sigma, sigma)
        t = self._t_of_s(xp.log(vbar))
        g = chebval(t, p.coeffs)
        g1 = chebval(t, p.d1)
        g2 = chebval(t, p.d2)
        h1 = g1 / vbar
        h2 = (g2 - g1) / (vbar * vbar)
        out = 0.5 * rm * rm + ln_mass + g
        if self.phase_marginalization:
            amp = rm * k
            x = amp * vbar
            i0 = i0e(x)
            r = i1e(x) / i0
            small = x < 1e-8
            r_over_x = xp.where(small, 0.5, r / xp.where(small, 1.0, x))
            h1 = h1 + amp * (r - 1.0)
            h2 = h2 + amp * amp * (1.0 - r_over_x - r * r)
            out = out + xp.log(i0)
        return out + 0.5 * var * (h2 + h1 * h1)

    def _branch_quadrature(self, xp, rm, k):
        """Evaluate the middle regime by six-panel Gauss-Legendre quadrature in ``s``.

        Parameters
        ----------
        xp : namespace
        rm, k : array
            As in :meth:`_select`.

        Returns
        -------
        array
            ``ln int pi_v(v) ell(v) dv``.

        Notes
        -----
        Each panel is reduced with :func:`distmarg._special.logsumexp_lastaxis`
        (:func:`scipy.special.logsumexp` gives NaN JAX gradients for the outer panels,
        whose terms underflow); panels are then combined linearly with their
        half-widths as factors.  Passing the half-widths as ``logsumexp``'s ``b``
        would be shorter, but SciPy returns a zero gradient with respect to ``b`` at
        ``b = 0``, and a zero-width panel must keep its
        derivative with respect to its break points (otherwise the JAX gradient is
        wrong when ``v*`` lies exactly on a prior edge).
        """
        p = self.prior
        vmin, vmax = self.v_min, self.v_max
        sigma = 1.0 / k
        vstar = rm * sigma
        amp = rm * k
        bb = k * k
        centre = xp.clip(vstar, vmin, vmax)
        dist = xp.abs(vstar - centre)
        scale = sigma * sigma / xp.maximum(sigma, dist)
        breaks = [xp.zeros_like(vstar) + p.s_a]
        for w in WINDOW:
            sb = xp.log(xp.clip(centre + w * scale, vmin, vmax))
            breaks.append(xp.clip(sb, p.s_a, p.s_b))
        breaks.append(xp.zeros_like(vstar) + p.s_b)
        gl_x, gl_lnw = self._gl(xp, vstar)
        halves, panel_lse = [], []
        for lo, hi in zip(breaks[:-1], breaks[1:]):
            half = 0.5 * (hi - lo)
            s = 0.5 * (hi + lo)[..., None] + half[..., None] * gl_x
            v = xp.exp(s)
            av = amp[..., None] * v
            ln_pi = self._ln_pi_v_grid(xp, s) if self.quad_grid else chebval(self._t_of_s(s), p.coeffs)
            f = ln_pi + s + av - 0.5 * bb[..., None] * v * v
            if self.phase_marginalization:
                f = f + xp.log(i0e(av))
            halves.append(half)
            panel_lse.append(sp.logsumexp_lastaxis(gl_lnw + f, xp))
        mx = xp.max(xp.stack(panel_lse, axis=-1), axis=-1)
        total = 0.0
        for half, lse in zip(halves, panel_lse):
            total = total + half * xp.exp(lse - mx)
        return mx + xp.log(total)

    # -------------------------------------------------------------- driver
    def _benign(self, xp, like, rho):
        """Return inputs placing the peak at the geometric centre of the support.

        Parameters
        ----------
        xp : namespace
        like : array
            Array giving shape, dtype and device.
        rho : float
            Matched-filter SNR to use.

        Returns
        -------
        rho, k : array
        """
        v_mid = math.sqrt(self.v_min * self.v_max)
        return xp.zeros_like(like) + rho, xp.zeros_like(like) + rho / v_mid

    def _evaluate(self, rho_mf, rho_opt, d_ref, masked=None):
        """Select regimes and evaluate the branches.

        Parameters
        ----------
        rho_mf, rho_opt, d_ref : array_like
            As in :meth:`ln_likelihood_ratio`.
        masked : bool, optional
            Evaluate each branch only on its subset (numpy only).  Default: ``True``
            for numpy, ``False`` otherwise, where every branch is evaluated on every
            element with benign inputs in unselected positions ("double where") and
            the result picked with ``where``.

        Returns
        -------
        out : array
            ``ln(L_D / Z_N)``.
        reg : array of int
            Regime codes.
        """
        xp, rm, ro, dr = self._as_arrays(rho_mf, rho_opt, d_ref)
        valid, rm, k = self._sanitize(xp, rm, ro, dr)
        reg = self._select(xp, rm, k)
        if masked is None:
            masked = is_numpy_namespace(xp)
        if masked:
            out = np.empty(rm.shape, dtype=rm.dtype)
            for code, fn in (
                (REGIME_PRIOR_DOMINATED, self._branch_cumulant),
                (REGIME_HIGH_SNR, self._branch_high),
                (REGIME_QUADRATURE, self._branch_quadrature),
            ):
                idx = reg == code
                if np.any(idx):
                    out[idx] = fn(np, rm[idx], k[idx])
            out = np.asarray(out)
        else:
            sel_c = reg == REGIME_PRIOR_DOMINATED
            sel_h = reg == REGIME_HIGH_SNR
            sel_q = reg == REGIME_QUADRATURE
            zero_r, zero_k = xp.zeros_like(rm), xp.zeros_like(k) + 1.0 / self.v_max
            out_c = self._branch_cumulant(xp, xp.where(sel_c, rm, zero_r), xp.where(sel_c, k, zero_k))
            br, bk = self._benign(xp, rm, max(self.rho_high, 1.0))
            out_h = self._branch_high(xp, xp.where(sel_h, rm, br), xp.where(sel_h, k, bk))
            br, bk = self._benign(xp, rm, 3.0)
            out_q = self._branch_quadrature(xp, xp.where(sel_q, rm, br), xp.where(sel_q, k, bk))
            out = xp.where(sel_c, out_c, xp.where(sel_h, out_h, out_q))
        out = xp.where(valid, out, math.nan)
        return out, reg

    # -------------------------------------------------------------- posterior (host-side numpy)
    def _posterior_grid(self, rho_mf, rho_opt, d_ref, n_grid):
        """Tabulate the conditional distance posterior in ``v`` (numpy).

        Parameters
        ----------
        rho_mf, rho_opt, d_ref : array_like
            As in :meth:`ln_likelihood_ratio`.
        n_grid : int
            Points in each of the global and peak-local grids.

        Returns
        -------
        v : numpy.ndarray
            Sorted grid in ``v`` (trailing axis).
        dens : numpy.ndarray
            Unnormalized density in ``v``.
        cdf : numpy.ndarray
            Cumulative trapezoid integral of ``dens``.
        """
        from scipy.integrate import cumulative_trapezoid

        rm, ro, dr = np.broadcast_arrays(*(np.asarray(x, dtype=np.float64) for x in (rho_mf, rho_opt, d_ref)))
        valid, rm, k = self._sanitize(np, rm, ro, dr)
        if not np.all(valid):
            raise ValueError("rho_opt and d_ref must be > 0 and all inputs finite")
        p = self.prior
        sigma = 1.0 / k
        _, vbar, var = self._truncnorm(np, rm * sigma, sigma)
        u = np.linspace(-1.0, 1.0, n_grid)
        local = np.log(np.clip(vbar[..., None] + 12.0 * np.sqrt(var)[..., None] * u, self.v_min, self.v_max))
        glob = np.broadcast_to(np.linspace(p.s_a, p.s_b, n_grid), local.shape)
        s = np.sort(np.concatenate([glob, local], axis=-1), axis=-1)
        v = np.exp(s)
        amp = (rm * k)[..., None]
        ln_post = p.ln_pi_v(s) + amp * v - 0.5 * (k * k)[..., None] * v * v
        if self.phase_marginalization:
            ln_post = ln_post + np.log(i0e(amp * v))
        dens = np.exp(ln_post - ln_post.max(axis=-1, keepdims=True))
        return v, dens, cumulative_trapezoid(dens, v, axis=-1, initial=0.0)

    def distance_posterior(self, rho_mf, rho_opt, d_ref, n_grid: int = 2000):
        """Tabulate the conditional posterior ``p(d_L | rho_mf, rho_opt, d_ref)`` (numpy).

        Parameters
        ----------
        rho_mf, rho_opt, d_ref : array_like
            As in :meth:`ln_likelihood_ratio`.
        n_grid : int, optional
            Grid resolution.

        Returns
        -------
        distance : numpy.ndarray
            Increasing luminosity distances (trailing axis).
        density : numpy.ndarray
            Posterior density, normalized in luminosity distance.
        """
        v, dens, cdf = self._posterior_grid(rho_mf, rho_opt, d_ref, n_grid)
        d = 1.0 / v[..., ::-1]
        dens_d = (dens * v * v)[..., ::-1] / cdf[..., -1:]
        return d, dens_d

    def sample_distance(self, rho_mf, rho_opt, d_ref, rng=None, n_grid: int = 2000):
        """Draw one luminosity distance per input from the conditional posterior.

        Parameters
        ----------
        rho_mf, rho_opt, d_ref : array_like
            As in :meth:`ln_likelihood_ratio`.
        rng : numpy.random.Generator or int, optional
            Random generator or seed.
        n_grid : int, optional
            Grid resolution of the inverse CDF.

        Returns
        -------
        numpy.ndarray
            Samples with the broadcast input shape.
        """
        rng = np.random.default_rng(rng)
        v, _, cdf = self._posterior_grid(rho_mf, rho_opt, d_ref, n_grid)
        u = rng.uniform(size=cdf.shape[:-1])[..., None] * cdf[..., -1:]
        flat_cdf = cdf.reshape(-1, cdf.shape[-1])
        flat_v = v.reshape(-1, v.shape[-1])
        out = np.array([np.interp(uu, cc, vv) for uu, cc, vv in zip(u.reshape(-1), flat_cdf, flat_v)])
        return 1.0 / out.reshape(cdf.shape[:-1])
