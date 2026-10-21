#!/usr/bin/env python
"""Generate shared data fixtures (IMPLEMENTATION_NOTES §7.7-7.8).

This is a third, independent code path.  It does not import ``distmarg``
except for ``--prior-fits``, which records the *Python* prior fit so that the
Rust implementation can be compared against it.

Outputs (under ``fixtures/``):

- ``priors/*.json``            input prior arrays (distance, ln_prior)
- ``oracle_special.json``      ln Phi, ln(Phi(b)-Phi(a)), ln I0e, I1/I0 at 40 digits
- ``oracle_<prior>.json``      ln(L_D/Z_N) for both flags by mpmath quadrature of the
                               *analytic* prior (not the tabulated arrays)
- ``prior_fit_<prior>.json``   (``--prior-fits``) Python fit: coeffs, ln_norm, moments

Usage::

    python tools/make_fixtures.py --all            # everything (oracle takes minutes)
    python tools/make_fixtures.py --special --priors --prior-fits
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "fixtures"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_prior import H0_DEFAULT, OM0_DEFAULT, power_law_arrays, usf_arrays, write_prior  # noqa: E402

N_PRIOR = 4096
ORACLE_DPS = 25

# name -> (kind, params, d_min, d_max)
PRIORS = {
    "usf_100_5000": ("usf", None, 100.0, 5000.0),
    "usf_10_50000": ("usf", None, 10.0, 50000.0),
    "pl2_50_5000": ("pl", 2.0, 50.0, 5000.0),
    "plm1_50_5000": ("pl", -1.0, 50.0, 5000.0),
}


# --------------------------------------------------------------------------- priors
def make_priors():
    """Write the prior fixtures, including the kinked and noisy ones."""
    (FIX / "priors").mkdir(parents=True, exist_ok=True)
    for name, (kind, par, dmin, dmax) in PRIORS.items():
        if kind == "usf":
            d, lp = usf_arrays(dmin, dmax, n=N_PRIOR)
            desc = f"UniformSourceFrame, flat LCDM H0={H0_DEFAULT} Om0={OM0_DEFAULT} (no radiation)"
        else:
            d, lp = power_law_arrays(par, dmin, dmax, n=N_PRIOR)
            desc = f"pi(d) ∝ d^{par:g}"
        write_prior(FIX / "priors" / f"{name}.json", d, lp, name=name, description=desc)
    # deliberately kinked prior: piecewise-linear interpolant of USF through 12 knots
    knots, lk = usf_arrays(100.0, 5000.0, n=12)
    d = np.geomspace(100.0, 5000.0, N_PRIOR)
    lp = np.log(np.interp(d, knots, np.exp(lk)))
    write_prior(FIX / "priors" / "kinked_100_5000.json", d, lp, name="kinked_100_5000",
                description="piecewise-linear interpolant of USF through 12 knots (must warn)")
    # noisy prior: d^2 with 1e-6 relative noise
    rng = np.random.default_rng(1)
    d = np.geomspace(50.0, 5000.0, N_PRIOR)
    write_prior(FIX / "priors" / "noisy_50_5000.json", d, 2 * np.log(d) + 1e-6 * rng.standard_normal(d.size),
                name="noisy_50_5000", description="d^2 prior with 1e-6 noise in ln pi (must warn)")
    print("wrote priors")


# --------------------------------------------------------------------------- special functions
def make_special():
    """Write the special-function oracle at 40 digits."""
    mp.mp.dps = 40

    def log_ndtr(x):
        """Compute ``ln Phi(x)`` in mpmath.

        Parameters
        ----------
        x : mpmath.mpf

        Returns
        -------
        mpmath.mpf
        """
        return mp.log(mp.ncdf(x))

    xs = sorted(set(np.round(np.linspace(-40, 40, 161), 12).tolist() + [-38.5, -37.6, -20.3, -8.1, -1e-3, 1e-3, 5.3, 8.4, 9.9]))
    out = {"log_ndtr": {"x": xs, "y": [float(log_ndtr(mp.mpf(x))) for x in xs]}}

    rng = np.random.default_rng(7)
    pairs = []
    for _ in range(150):
        a, b = sorted(rng.uniform(-40, 40, 2))
        pairs.append((a, b))
    for a in [-40, -30, -10, -3, -1, -0.1, 0, 0.2, 1, 3, 10, 30, 39]:
        for w in [1e-8, 1e-4, 1e-2, 0.5, 3, 30]:
            pairs.append((a, a + w))
    la, lb, ly = [], [], []
    for a, b in pairs:
        ma, mb = mp.mpf(a), mp.mpf(b)
        if a > 0:  # use survival functions for accuracy
            val = mp.ncdf(-ma) - mp.ncdf(-mb)
        else:
            val = mp.ncdf(mb) - mp.ncdf(ma)
        la.append(float(a))
        lb.append(float(b))
        ly.append(float(mp.log(val)))
    out["log_diff_ndtr"] = {"a": la, "b": lb, "y": ly}

    xb = sorted(set(np.geomspace(1e-6, 1e6, 121).tolist() + [0.5, 1, 2, 5, 10, 19.9, 20, 20.1, 30, 50, 100, 500, 700, 1000]))
    out["ln_i0e"] = {"x": xb, "y": [float(mp.log(mp.besseli(0, x)) - x) for x in xb]}
    out["i1_over_i0"] = {"x": xb, "y": [float(mp.besseli(1, x) / mp.besseli(0, x)) for x in xb]}
    with open(FIX / "oracle_special.json", "w") as f:
        json.dump(out, f, indent=0)
    print("wrote oracle_special.json")


# --------------------------------------------------------------------------- analytic priors in mpmath
class MpPrior:
    """ln pi_v(v) (unnormalized) on [1/d_max, 1/d_min] from the analytic prior."""

    def __init__(self, kind, par, dmin, dmax, n_cheb=90):
        """Build the analytic prior and its normalization.

        Parameters
        ----------
        kind : {"usf", "pl"}
        par : float or None
            Power-law index for ``"pl"``.
        dmin, dmax : float
        n_cheb : int, optional
            Interpolation nodes for the USF prior.
        """
        mp.mp.dps = ORACLE_DPS
        self.kind, self.par = kind, par
        self.dmin, self.dmax = mp.mpf(dmin), mp.mpf(dmax)
        self.vmin, self.vmax = 1 / self.dmax, 1 / self.dmin
        self.s_a, self.s_b = mp.log(self.vmin), mp.log(self.vmax)
        if kind == "usf":
            self._build_cheb(n_cheb)
        f = lambda v: mp.exp(self.ln_pv_unnorm(v))  # noqa: E731
        self.ln_norm = mp.log(mp.quad(f, mp.linspace(self.vmin, self.vmax, 9)))

    # USF: exact density at arbitrary d (slow) -> Chebyshev interpolant in s
    def _usf_exact(self, s):
        """Evaluate the exact USF ``ln pi_v`` at ``s = ln v``.

        Parameters
        ----------
        s : mpmath.mpf

        Returns
        -------
        mpmath.mpf
        """
        dh = mp.mpf(299792.458) / mp.mpf(H0_DEFAULT)
        om = mp.mpf(OM0_DEFAULT)
        inv_e = lambda z: 1 / mp.sqrt(om * (1 + z) ** 3 + 1 - om)  # noqa: E731
        dc = lambda z: dh * mp.quad(inv_e, [0, z])  # noqa: E731
        d = mp.exp(-s)
        z = mp.findroot(lambda z: (1 + z) * dc(z) - d, d / dh)
        dcz = dc(z)
        ddl = dcz + (1 + z) * dh * inv_e(z)
        ln_pd = mp.log(4 * mp.pi * dh * dcz**2 * inv_e(z)) - mp.log(1 + z) - mp.log(ddl)
        return ln_pd + 2 * mp.log(d)

    def _build_cheb(self, n):
        """Build and self-check a Chebyshev interpolant of the USF ``ln pi_v``.

        Parameters
        ----------
        n : int
            Number of Chebyshev nodes.
        """
        mid, half = (self.s_a + self.s_b) / 2, (self.s_b - self.s_a) / 2
        th = [mp.pi * (j + mp.mpf(1) / 2) / n for j in range(n)]
        vals = [self._usf_exact(mid + half * mp.cos(t)) for t in th]
        self.cheb = [2 * mp.fsum(vals[j] * mp.cos(k * th[j]) for j in range(n)) / n for k in range(n)]
        self.cheb[0] /= 2
        # self-check at off-node points
        err = max(abs(self._cheb_eval(s) - self._usf_exact(s)) for s in mp.linspace(self.s_a, self.s_b, 7)[1:-1])
        assert err < mp.mpf("1e-15"), f"USF interpolant error {err}"

    def _cheb_eval(self, s):
        """Evaluate the USF interpolant.

        Parameters
        ----------
        s : mpmath.mpf

        Returns
        -------
        mpmath.mpf
        """
        t = (2 * s - self.s_a - self.s_b) / (self.s_b - self.s_a)
        b1 = b2 = mp.mpf(0)
        for c in reversed(self.cheb[1:]):
            b1, b2 = c + 2 * t * b1 - b2, b1
        return self.cheb[0] + t * b1 - b2

    def ln_pv_unnorm(self, v):
        """Evaluate the unnormalized ``ln pi_v``.

        Parameters
        ----------
        v : mpmath.mpf

        Returns
        -------
        mpmath.mpf
        """
        if self.kind == "pl":
            return -(self.par + 2) * mp.log(v)
        return self._cheb_eval(mp.log(v))


_PRIOR_CACHE = {}


def _get_prior(name):
    """Return the cached :class:`MpPrior` for a fixture prior.

    Parameters
    ----------
    name : str

    Returns
    -------
    MpPrior
    """
    if name not in _PRIOR_CACHE:
        kind, par, dmin, dmax = PRIORS[name]
        _PRIOR_CACHE[name] = MpPrior(kind, par, dmin, dmax)
    return _PRIOR_CACHE[name]


def oracle_value(name, phase, rho_mf, rho_opt, d_ref):
    """Compute ``ln(L_D/Z_N)`` with mpmath quadrature over the analytic prior.

    Parameters
    ----------
    name : str
        Prior name.
    phase : bool
    rho_mf, rho_opt, d_ref : float

    Returns
    -------
    float
    """
    mp.mp.dps = ORACLE_DPS
    pr = _get_prior(name)
    rm = abs(mp.mpf(rho_mf)) if phase else mp.mpf(rho_mf)
    k = mp.mpf(rho_opt) * mp.mpf(d_ref)
    a_amp, b = rm * k, k * k
    sigma = 1 / k
    vstar = rm * sigma
    centre = min(max(vstar, pr.vmin), pr.vmax)

    def lnl(v):
        """Evaluate the log likelihood ratio at fixed distance.

        Parameters
        ----------
        v : mpmath.mpf
            Inverse distance.

        Returns
        -------
        mpmath.mpf
        """
        if phase:
            return mp.log(mp.besseli(0, a_amp * v)) - b * v * v / 2
        return a_amp * v - b * v * v / 2

    shift = lnl(centre)
    pts = {pr.vmin, pr.vmax, centre}
    for m in [0.5, 1, 2, 3, 5, 8, 12, 20, 40]:
        pts.add(centre + m * sigma)
        pts.add(centre - m * sigma)
    dist = abs(vstar - centre)
    if dist > 0:
        lam = sigma**2 / dist
        sgn = 1 if centre == pr.vmin else -1
        for m in [0.05, 0.2, 0.5, 1, 2, 4, 8, 16, 32, 64, 120]:
            pts.add(centre + sgn * m * lam)
    for x in mp.linspace(pr.s_a, pr.s_b, 9):
        pts.add(mp.exp(x))
    pts = sorted(p for p in pts if pr.vmin <= p <= pr.vmax)

    f = lambda v: mp.exp(pr.ln_pv_unnorm(v) - pr.ln_norm + lnl(v) - shift)  # noqa: E731
    total = mp.quad(f, pts)
    return float(mp.log(total) + shift)


def _queries(name):
    """Build the oracle queries for one prior.

    Parameters
    ----------
    name : str

    Returns
    -------
    list of tuple
        ``(rho_mf, rho_opt, d_ref, label)``.
    """
    _, _, dmin, dmax = PRIORS[name]
    dstars = [dmin / 5, dmin / 1.5, dmin * 1.02, math.sqrt(dmin * dmax), dmax / 1.5, dmax * 0.98, 2 * dmax, 5 * dmax]
    rhos = [-300.0, -20.0, -2.0, -0.1, 0.01, 0.5, 2.0, 5.0, 12.0, 40.0, 150.0, 1000.0, 1e4]
    drefs = [1000.0, 300.0, 5000.0]
    q = []
    i = 0
    for ds in dstars:
        for r in rhos:
            d_ref = drefs[i % 3]
            i += 1
            rho_opt = abs(r) * ds / d_ref
            if ds < dmin:
                label = "outside_below_dmin"
            elif ds > dmax:
                label = "outside_above_dmax"
            elif ds < 1.1 * dmin or ds > 0.9 * dmax:
                label = "near_edge"
            else:
                label = "interior"
            if r < 0:
                label = "negative_rho"
            elif abs(r) * ds / dmin < 0.3:
                label = "low_snr"
            q.append((r, rho_opt, d_ref, label))
    return q


def _job(args):
    """Evaluate one oracle value (multiprocessing worker).

    Parameters
    ----------
    args : tuple
        ``(name, phase, rho_mf, rho_opt, d_ref)``.

    Returns
    -------
    float
    """
    name, phase, r, ro, dr = args
    return oracle_value(name, phase, r, ro, dr)


def make_oracle(names, processes):
    """Write the ``ln L_D`` oracle fixtures.

    Parameters
    ----------
    names : list of str
        Prior names.
    processes : int
        Worker processes.
    """
    for name in names:
        q = _queries(name)
        jobs = [(name, ph, r, ro, dr) for ph in (False, True) for (r, ro, dr, _) in q]
        with Pool(processes) as pool:
            vals = pool.map(_job, jobs, chunksize=1)
        n = len(q)
        out = dict(
            prior=name,
            rho_mf=[x[0] for x in q],
            rho_opt=[x[1] for x in q],
            d_ref=[x[2] for x in q],
            label=[x[3] for x in q],
            ln_likelihood_ratio=vals[:n],
            ln_likelihood_ratio_phase=vals[n:],
        )
        with open(FIX / f"oracle_{name}.json", "w") as f:
            json.dump(out, f, indent=0)
        print(f"wrote oracle_{name}.json ({n} queries)", flush=True)


def make_prior_fits():
    """Write the Python prior-fit fixtures used by the Rust tests."""
    sys.path.insert(0, str(ROOT / "python"))
    from distmarg import fit_prior

    for name in PRIORS:
        pr = json.load(open(FIX / "priors" / f"{name}.json"))
        info = fit_prior(pr["distance"], pr["ln_prior"]).fit_info()
        with open(FIX / f"prior_fit_{name}.json", "w") as f:
            json.dump(dict(prior=name, **info), f, indent=0)
    print("wrote prior_fit_*.json")


def main(argv=None):
    """Generate fixtures from the command line.

    Parameters
    ----------
    argv : list of str, optional
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--priors", action="store_true")
    ap.add_argument("--special", action="store_true")
    ap.add_argument("--oracle", nargs="*", default=None, help="prior names (default: all)")
    ap.add_argument("--prior-fits", action="store_true")
    ap.add_argument("--processes", type=int, default=os.cpu_count())
    args = ap.parse_args(argv)
    if args.all or args.priors:
        make_priors()
    if args.all or args.special:
        make_special()
    if args.all or args.prior_fits:
        make_prior_fits()
    if args.all or args.oracle is not None:
        make_oracle(args.oracle or list(PRIORS), args.processes)


if __name__ == "__main__":
    main()
