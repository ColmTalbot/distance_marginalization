#!/usr/bin/env python
"""Compare distance-marginalized likelihoods on the fast-tutorial injection.

For each parameter draw, ``log_likelihood_ratio`` is evaluated with

(a) Bilby's ``GravitationalWaveTransient(distance_marginalization=True)``
    (400 x 800 lookup table, cached under ``examples/outdir/``);
(b) :class:`distmarg_likelihood.DistmargGravitationalWaveTransient`;
(c) a brute-force reference: the non-marginalized likelihood ratio,
    ``Re<d|h>(d) - <h|h>(d)/2`` (or ``ln I0(|<d|h>(d)|) - <h|h>(d)/2`` with phase
    marginalization), integrated over a 2 000 001-point log-spaced grid in ``d_L``
    against the same prior with a log-sum-exp of trapezoid weights in ``ln d_L``
    (converged to ~1e-9 nats; a 200 001-point linear grid is only good to ~1e-6).  ``<d|h>`` and ``<h|h>``
    come from Bilby's non-marginalized likelihood at the drawn distance and are
    rescaled as ``1/d`` and ``1/d^2``; the rescaling is checked against direct
    evaluations of the non-marginalized likelihood.

Two versions of the reference are computed: ``c`` uses Bilby's
``prior.prob`` (the piecewise-linear interpolant Bilby itself uses, i.e. exactly
the prior the lookup table integrates against) and ``c_exact`` uses the analytic
``UniformSourceFrame`` density (as ``distmarg`` does).  Their difference is the
prior-representation error, not a marginalization error.

Two sets of 100 draws are used: draws from the full prior (mostly templates that
do not match the signal, i.e. low-SNR/prior-dominated), and draws near the
injection (``chirp_mass``, ``mass_ratio``, ``theta_jn`` perturbed around the
injected values) which exercise the informative regimes.  Both are done with and
without phase marginalization.  Results are written to
``outdir/likelihood_comparison.{json,png}``.
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import bilby
import numpy as np
from bilby.gw.utils import ln_i0
from scipy.special import logsumexp

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from fast_tutorial_distmarg import INJECTION_PARAMETERS, OUTDIR, make_likelihood, make_setup  # noqa: E402

N_GRID = 2_000_001


def inner_products(likelihood, parameters):
    """Return network ``<d|h>`` (complex) and ``<h|h>`` at ``parameters['luminosity_distance']``.

    Parameters
    ----------
    likelihood : bilby.gw.likelihood.GravitationalWaveTransient
        A non-marginalized likelihood.
    parameters : dict
        Full parameter dictionary.

    Returns
    -------
    d_inner_h : complex
    h_inner_h : float
    """
    parameters = copy.deepcopy(parameters)
    polarizations = likelihood.waveform_generator.frequency_domain_strain(parameters)
    parameters.update(likelihood.get_sky_frame_parameters(parameters))
    d_inner_h, h_inner_h = likelihood._calculate_inner_products(polarizations, parameters)
    return complex(d_inner_h), float(np.real(h_inner_h))


class BruteForce:
    """Brute-force distance (and phase) marginalization on a dense grid.

    Parameters
    ----------
    prior : bilby.core.prior.Prior
        Luminosity-distance prior.
    phase_marginalization : bool
        Use ``ln I0(|<d|h>|)`` instead of ``Re<d|h>``.
    exact_prior : bool
        Use the analytic density from :func:`distmarg.bilby.prior_to_arrays`
        (interpolated in ``ln d`` from 16 384 points) instead of ``prior.prob``.
    n_grid : int, optional
        Number of log-spaced grid points on the prior support (trapezoid rule in ``ln d``).
    """

    def __init__(self, prior, phase_marginalization, exact_prior, n_grid=N_GRID):
        self.phase_marginalization = phase_marginalization
        self.d = np.geomspace(prior.minimum, prior.maximum, n_grid)
        if exact_prior:
            from distmarg.bilby import prior_to_arrays

            dd, lp = prior_to_arrays(prior, n=16384)
            ln_p = np.interp(np.log(self.d), np.log(dd), lp)
        else:
            with np.errstate(divide="ignore"):
                ln_p = np.log(prior.prob(self.d))
        w = np.full(n_grid, np.log(self.d[1] / self.d[0]))
        w[[0, -1]] *= 0.5
        self.ln_w = ln_p + np.log(self.d) + np.log(w)
        self.ln_w -= logsumexp(self.ln_w)

    def __call__(self, d_inner_h, h_inner_h, d0):
        """Evaluate ``ln(L_D / Z_N)``.

        Parameters
        ----------
        d_inner_h : complex
            ``<d|h>`` at distance ``d0``.
        h_inner_h : float
            ``<h|h>`` at distance ``d0``.
        d0 : float
            Distance at which the inner products were computed.

        Returns
        -------
        float
        """
        r = d0 / self.d
        if self.phase_marginalization:
            ln_l = ln_i0(np.abs(d_inner_h) * r) - 0.5 * h_inner_h * r * r
        else:
            ln_l = d_inner_h.real * r - 0.5 * h_inner_h * r * r
        return float(logsumexp(ln_l + self.ln_w))


def draw_samples(priors, n, near_injection, seed):
    """Draw parameter dictionaries.

    Parameters
    ----------
    priors : bilby.core.prior.PriorDict
        Fast-tutorial priors (with a free distance prior).
    n : int
        Number of draws.
    near_injection : bool
        If true, replace ``chirp_mass``, ``mass_ratio`` and ``theta_jn`` by narrow
        Gaussian perturbations about the injection (the distance stays a prior draw).
    seed : int
        Random seed.

    Returns
    -------
    list of dict
    """
    bilby.core.utils.random.seed(seed)
    rng = np.random.default_rng(seed)
    samples = priors.sample(n)
    out = [{k: float(np.atleast_1d(v)[i]) for k, v in samples.items()} for i in range(n)]
    if near_injection:
        inj = bilby.gw.conversion.generate_mass_parameters(dict(INJECTION_PARAMETERS))
        for s in out:
            s["chirp_mass"] = inj["chirp_mass"] + 0.5 * rng.standard_normal()
            s["mass_ratio"] = float(np.clip(inj["mass_ratio"] + 0.1 * rng.standard_normal(), 0.2, 1.0))
            s["theta_jn"] = float(np.clip(INJECTION_PARAMETERS["theta_jn"] + 0.3 * rng.standard_normal(), 0.0, np.pi))
    return out


def time_calls(func, args_list, repeat=1):
    """Mean wall time per call in seconds.

    Parameters
    ----------
    func : callable
    args_list : list of tuple
        Positional arguments for each call.
    repeat : int, optional
        Number of passes over ``args_list``.

    Returns
    -------
    float
    """
    t0 = time.perf_counter()
    for _ in range(repeat):
        for a in args_list:
            func(*a)
    return (time.perf_counter() - t0) / (repeat * len(args_list))


def compare(phase_marginalization, n_samples):
    """Run the comparison for one phase flag.

    Parameters
    ----------
    phase_marginalization : bool
        Marginalize over phase as well.
    n_samples : int
        Draws per sample set.

    Returns
    -------
    dict
        Per-set arrays and summary statistics.
    """
    ifos, waveform_generator, priors = make_setup()
    if phase_marginalization:
        priors["phase"] = bilby.core.prior.Uniform(0, 2 * np.pi, name="phase", boundary="periodic")
    distance_prior = copy.deepcopy(priors["luminosity_distance"])
    stock = make_likelihood("stock", ifos, waveform_generator, copy.deepcopy(priors), phase_marginalization)
    new = make_likelihood("distmarg", ifos, waveform_generator, copy.deepcopy(priors), phase_marginalization)
    plain = make_likelihood("plain", ifos, waveform_generator, copy.deepcopy(priors), False)
    ref_bilby = BruteForce(distance_prior, phase_marginalization, exact_prior=False)
    ref_exact = BruteForce(distance_prior, phase_marginalization, exact_prior=True)
    dm = new.distance_marginalizer
    d_ref = new._ref_dist
    assert np.isclose(stock._ref_dist, d_ref)

    out = {}
    for set_name, near in (("prior", False), ("near_injection", True)):
        samples = draw_samples(priors, n_samples, near, seed=1234 + int(near) + 2 * int(phase_marginalization))
        if phase_marginalization:
            for s in samples:
                s["phase"] = 0.0  # the marginalized likelihoods evaluate at phase = 0
        rows = []
        for s in samples:
            a = stock.log_likelihood_ratio(s)
            b = new.log_likelihood_ratio(s)
            dh, hh = inner_products(plain, s)
            d0 = s["luminosity_distance"]
            c = ref_bilby(dh, hh, d0)
            c_exact = ref_exact(dh, hh, d0)
            # check the 1/d scaling of the inner products against a direct evaluation
            s2 = dict(s, luminosity_distance=1.37 * d0)
            direct = plain.log_likelihood_ratio(s2)
            scaled = dh.real / 1.37 - 0.5 * hh / 1.37**2
            assert abs(direct - scaled) < 1e-8 * max(1.0, abs(direct)), (direct, scaled)
            rho_opt = np.sqrt(hh) * d0 / d_ref
            rho_mf = (abs(dh) if phase_marginalization else dh.real) / np.sqrt(hh)
            rows.append((a, b, c, c_exact, rho_mf, rho_opt, int(dm.regime(rho_mf, rho_opt, d_ref))))
        arr = np.array(rows)
        a, b, c, c_exact = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]

        # timing: full log_likelihood_ratio calls, and the marginalization step alone
        t_full_a = time_calls(stock.log_likelihood_ratio, [(s,) for s in samples])
        t_full_b = time_calls(new.log_likelihood_ratio, [(s,) for s in samples])
        margs = []
        for s in samples:
            dh, hh = inner_products(plain, s)
            margs.append((dh, hh, dict(s)))
        t_marg_a = time_calls(stock.distance_marginalized_likelihood, margs, repeat=20)
        t_marg_b = time_calls(new.distance_marginalized_likelihood, margs, repeat=20)
        t_plain = time_calls(plain.log_likelihood_ratio, [(s,) for s in samples])

        stats = dict(
            n=len(a),
            max_abs_a_c=float(np.max(np.abs(a - c))),
            median_abs_a_c=float(np.median(np.abs(a - c))),
            max_abs_b_c=float(np.max(np.abs(b - c))),
            median_abs_b_c=float(np.median(np.abs(b - c))),
            max_abs_b_cexact=float(np.max(np.abs(b - c_exact))),
            median_abs_b_cexact=float(np.median(np.abs(b - c_exact))),
            max_abs_a_cexact=float(np.max(np.abs(a - c_exact))),
            max_abs_c_cexact=float(np.max(np.abs(c - c_exact))),
            max_abs_a_b=float(np.max(np.abs(a - b))),
            ln_l_range=[float(c.min()), float(c.max())],
            rho_mf_range=[float(arr[:, 4].min()), float(arr[:, 4].max())],
            regime_counts={str(k): int(np.sum(arr[:, 6] == k)) for k in (0, 1, 2)},
            t_full_stock_ms=1e3 * t_full_a,
            t_full_distmarg_ms=1e3 * t_full_b,
            t_full_plain_ms=1e3 * t_plain,
            t_marg_stock_us=1e6 * t_marg_a,
            t_marg_distmarg_us=1e6 * t_marg_b,
        )
        out[set_name] = dict(stats=stats, rows=arr.tolist())
        print(f"phase_marginalization={phase_marginalization} set={set_name}: "
              + ", ".join(f"{k}={v:.3g}" if isinstance(v, float) else f"{k}={v}" for k, v in stats.items()))
    return out


def plot(results, filename):
    """Plot ``|a - c|`` and ``|b - c|`` against the reference value.

    Parameters
    ----------
    results : dict
        Output of :func:`compare` keyed by phase flag.
    filename : str
        Output path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 3.8), sharey=True, squeeze=False)
    for ax, (flag, res) in zip(axes[0], results.items()):
        for set_name, marker in (("prior", "o"), ("near_injection", "s")):
            arr = np.array(res[set_name]["rows"])
            a, b, c = arr[:, 0], arr[:, 1], arr[:, 2]
            floor = 1e-16
            ax.scatter(c, np.abs(a - c) + floor, s=12, marker=marker, color="C0", alpha=0.7,
                       label=f"Bilby lookup table ({set_name})")
            ax.scatter(c, np.abs(b - c) + floor, s=12, marker=marker, color="C1", alpha=0.7,
                       label=f"distmarg ({set_name})")
        ax.set_xscale("symlog", linthresh=1.0)
        ax.set_yscale("log")
        ax.set_xlabel(r"brute-force $\ln(\mathcal{L}_D/\mathcal{Z}_N)$")
        ax.set_title(f"phase_marginalization={flag}")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("|error| vs brute force [nats]")
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)


def main(n_samples=100):
    """Run both comparisons, save JSON and the plot, and print a summary table.

    Parameters
    ----------
    n_samples : int, optional
        Draws per sample set.

    Returns
    -------
    dict
    """
    os.makedirs(OUTDIR, exist_ok=True)
    results = {flag: compare(flag, n_samples) for flag in (False, True)}
    with open(os.path.join(OUTDIR, "likelihood_comparison.json"), "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=1)
    plot(results, os.path.join(OUTDIR, "likelihood_comparison.png"))

    header = (f"{'phase':>5} {'set':>15} {'max|a-c|':>9} {'med|a-c|':>9} {'max|b-c|':>9} {'med|b-c|':>9} "
              f"{'max|b-cx|':>9} {'max|c-cx|':>9} {'t_a ms':>7} {'t_b ms':>7} {'marg_a us':>9} {'marg_b us':>9}")
    print(header)
    for flag, res in results.items():
        for set_name, r in res.items():
            s = r["stats"]
            print(f"{str(flag):>5} {set_name:>15} {s['max_abs_a_c']:9.2e} {s['median_abs_a_c']:9.2e} "
                  f"{s['max_abs_b_c']:9.2e} {s['median_abs_b_c']:9.2e} {s['max_abs_b_cexact']:9.2e} "
                  f"{s['max_abs_c_cexact']:9.2e} {s['t_full_stock_ms']:7.2f} {s['t_full_distmarg_ms']:7.2f} "
                  f"{s['t_marg_stock_us']:9.1f} {s['t_marg_distmarg_us']:9.1f}")
    return results


if __name__ == "__main__":
    bilby.core.utils.setup_logger(outdir=OUTDIR, label="compare_likelihoods", log_level="WARNING")
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100)
