#!/usr/bin/env python
"""Accuracy and speed benchmarks for the Python implementation (IMPLEMENTATION_NOTES §8).

- accuracy: |Δ ln L| against the dense reference quadrature on the stress grid,
  max and quantiles per regime, USF priors, both phase flags;
- speed: numpy (masked branches) and JAX (jit, where-path) at several batch sizes,
  per-regime numpy timing, and set-up time;
- ``--bilby``: comparison with Bilby's lookup table on posterior-like samples
  (needs bilby; builds two ~40 s tables, cached in ``--cache``).

Rust timings come from ``cargo bench`` in ``rust/`` (reported separately).

    python tools/benchmark.py [--quick] [--bilby] [--out benchmarks/report.md]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

os.environ.setdefault("SCIPY_ARRAY_API", "1")  # before SciPy is imported: needed for JAX/torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from distmarg import REGIME_NAMES, DistanceMarginalization  # noqa: E402
from distmarg._quadrature import reference_ln_likelihood_ratio  # noqa: E402


def load(name):
    """Load a fixture prior.

    Parameters
    ----------
    name : str

    Returns
    -------
    dict
    """
    return json.load(open(ROOT / "fixtures" / "priors" / f"{name}.json"))


def stress(dm, n, phase):
    """Build the stress grid.

    Parameters
    ----------
    dm : DistanceMarginalization
    n : int
        Points per axis.
    phase : bool
        Omit negative ``rho_mf`` when True.

    Returns
    -------
    numpy.ndarray
        Rows of ``(rho_mf, rho_opt, d_ref)``.
    """
    dmin, dmax = dm.prior.d_min, dm.prior.d_max
    rows = []
    signs = (1,) if phase else (1, -1)
    for ds in np.geomspace(dmin / 5, dmax * 5, n):
        for r in np.geomspace(1e-3, 1e4, n):
            rows += [(s * r, r * ds / 1000.0, 1000.0) for s in signs]
    for ro in np.geomspace(1e-3, 1e3, n):
        for r in np.geomspace(1e-3, 1e4, n):
            rows += [(s * r, ro * dmin / 1000.0, 1000.0) for s in signs]
    return np.array(rows)


def timeit(fn, min_time=0.5, max_reps=10000):
    """Time a callable.

    Parameters
    ----------
    fn : callable
    min_time : float, optional
        Minimum total measurement time in seconds.
    max_reps : int, optional
        Maximum repetitions.

    Returns
    -------
    float
        Seconds per call (after one warm-up call).
    """
    fn()
    reps, t0 = 0, time.perf_counter()
    while True:
        fn()
        reps += 1
        dt = time.perf_counter() - t0
        if dt > min_time or reps >= max_reps:
            return dt / reps


def accuracy_section(lines, n):
    """Append the accuracy table to the report.

    Parameters
    ----------
    lines : list of str
        Report lines, modified in place.
    n : int
        Stress-grid points per axis.

    Returns
    -------
    dict
        Max error per ``(prior, phase)``.
    """
    lines.append("## Accuracy vs reference quadrature (stress grid)\n")
    lines.append("| prior | phase | regime | N | max abs err | 99% | 50% |")
    lines.append("|---|---|---|---|---|---|---|")
    summary = {}
    for name in ["usf_100_5000", "usf_10_50000"]:
        pr = load(name)
        for phase in (False, True):
            dm = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase)
            q = stress(dm, n, phase)
            got = dm.ln_likelihood_ratio(*q.T)
            reg = dm.regime(*q.T)
            ref = np.array([reference_ln_likelihood_ratio(dm.prior, phase, *row) for row in q])
            err = np.abs(got - ref)
            for code, rname in REGIME_NAMES.items():
                e = err[reg == code]
                if e.size:
                    lines.append(f"| {name} | {phase} | {rname} | {e.size} | {e.max():.1e} | "
                                 f"{np.quantile(e, 0.99):.1e} | {np.median(e):.1e} |")
            lines.append(f"| {name} | {phase} | **all** | {err.size} | **{err.max():.1e}** | "
                         f"{np.quantile(err, 0.99):.1e} | {np.median(err):.1e} |")
            summary[(name, phase)] = float(err.max())
    lines.append("")
    return summary


def speed_section(lines, quick):
    """Append numpy/JAX timings to the report.

    Parameters
    ----------
    lines : list of str
        Report lines, modified in place.
    quick : bool
        Use smaller batches.
    """
    pr = load("usf_100_5000")
    lines.append("## Speed (Python, this machine)\n")
    t0 = time.perf_counter()
    for _ in range(5):
        DistanceMarginalization(pr["distance"], pr["ln_prior"], False)
    lines.append(f"Set-up (fit + normalization + moments, 4096-point prior): "
                 f"{(time.perf_counter() - t0) / 5 * 1e3:.1f} ms\n")
    rng = np.random.default_rng(0)
    sizes = [1, 1000, 100_000] if quick else [1, 1000, 1_000_000]
    lines.append("| backend | phase | batch | time/call | ns/eval |")
    lines.append("|---|---|---|---|---|")
    for phase in (False, True):
        dm = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase)

        def sample(n):
            """Draw random queries over the stress domain.

            Parameters
            ----------
            n : int

            Returns
            -------
            rho_mf, rho_opt, d_ref : numpy.ndarray
            """
            rho = 10 ** rng.uniform(-1, 3, n)
            ds = 20.0 * (25 * 5000 / 100) ** rng.uniform(size=n)
            return rho, rho * ds / 1000.0, np.full(n, 1000.0)

        for n in sizes:
            a = sample(n)
            t = timeit(lambda: dm.ln_likelihood_ratio(*a), min_time=0.3 if n < 1e5 else 0.0, max_reps=200)
            lines.append(f"| numpy | {phase} | {n} | {t * 1e3:.3f} ms | {t / n * 1e9:.0f} |")
        try:
            import jax
            import jax.numpy as jnp

            jax.config.update("jax_enable_x64", True)
            f = jax.jit(dm.ln_likelihood_ratio)
            for n in [1, 1000, 100_000]:
                a = tuple(jnp.asarray(x) for x in sample(n))
                f(*a).block_until_ready()
                t = timeit(lambda: f(*a).block_until_ready(), min_time=0.3, max_reps=200)
                lines.append(f"| jax-cpu jit | {phase} | {n} | {t * 1e3:.3f} ms | {t / n * 1e9:.0f} |")
        except ImportError:
            pass
    # per regime (numpy, batch 1e4)
    dm = DistanceMarginalization(pr["distance"], pr["ln_prior"], False)
    lines.append("\nPer-regime numpy cost (batch 10⁴, no phase):\n")
    for rname, (rho, dstar) in {"high_snr": (200.0, 800.0), "prior_dominated": (0.5, 40.0),
                                "quadrature": (8.0, 900.0)}.items():
        n = 10_000
        a = (np.full(n, rho), np.full(n, rho * dstar / 1000.0), np.full(n, 1000.0))
        assert REGIME_NAMES[int(dm.regime(*(x[0] for x in a)))] == rname
        t = timeit(lambda: dm.ln_likelihood_ratio(*a), min_time=0.3, max_reps=100)
        lines.append(f"- {rname}: {t / n * 1e9:.0f} ns/eval")
    lines.append("")


def backends_section(lines, sizes, prior="usf_100_5000"):
    """Append timings of every array backend, values and gradients, to the report.

    Parameters
    ----------
    lines : list of str
        Report lines, modified in place.
    sizes : sequence of int
        Batch sizes.
    prior : str, optional
        Fixture prior name.

    Notes
    -----
    Inputs are posterior-like (rho_mf in [3, 100], d* inside the prior), which
    puts almost every point in the quadrature regime, the expensive one.  JAX is
    timed after ``jit`` compilation (compile time reported separately) and torch
    eagerly; gradients are with respect to ``rho_mf`` and ``rho_opt`` (``jax.grad``
    of the sum, torch ``backward``).  Backends that are not installed are skipped.
    """
    pr = load(prior)
    rng = np.random.default_rng(0)

    def sample(n):
        rho = 10 ** rng.uniform(0.5, 2, n)
        ds = rng.uniform(1.5 * pr["distance"][0], 0.8 * pr["distance"][-1], n)
        return rho, rho * ds / 1000.0, np.full(n, 1000.0)

    lines.append("## Array backends (values and gradients)\n")
    lines.append(f"Prior {prior}; posterior-like inputs (quadrature regime); µs per call and ns per element.\n")
    lines.append("| backend | phase | batch | value µs/call | value ns/elem | grad µs/call | grad ns/elem | first call (compile) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def row(name, phase, n, t_val, t_grad=None, first=None):
        g = f"{t_grad * 1e6:.1f} | {t_grad / n * 1e9:.0f}" if t_grad is not None else "– | –"
        f = f"{first:.2f} s" if first is not None else "–"
        lines.append(f"| {name} | {phase} | {n} | {t_val * 1e6:.1f} | {t_val / n * 1e9:.0f} | {g} | {f} |")
        print(lines[-1], flush=True)

    try:
        import jax
        import jax.numpy as jnp

        jax.config.update("jax_enable_x64", True)
    except ImportError:
        jax = None
    try:
        import torch
    except ImportError:
        torch = None

    for phase in (False, True):
        dm_rs = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase)
        dm_py = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase, backend="python")
        for n in sizes:
            a = sample(n)
            budget = dict(min_time=0.3, max_reps=200)
            if dm_rs.backend == "rust":
                if n == 1:
                    sc = tuple(float(x[0]) for x in a)
                    row("numpy → Rust (scalar)", phase, n, timeit(lambda: dm_rs.ln_likelihood_ratio(*sc), **budget))
                row("numpy → Rust (array)", phase, n, timeit(lambda: dm_rs.ln_likelihood_ratio(*a), **budget))
            row("numpy, pure Python", phase, n, timeit(lambda: dm_py.ln_likelihood_ratio(*a), **budget))
            if jax is not None:
                ja = tuple(jnp.asarray(x) for x in a)
                f = jax.jit(dm_py.ln_likelihood_ratio)
                g = jax.jit(jax.grad(lambda r, o, d: jnp.sum(dm_py.ln_likelihood_ratio(r, o, d)), argnums=(0, 1)))
                t0 = time.perf_counter()
                f(*ja).block_until_ready()
                first = time.perf_counter() - t0
                jax.block_until_ready(g(*ja))
                row("JAX jit (CPU)", phase, n, timeit(lambda: f(*ja).block_until_ready(), **budget),
                    timeit(lambda: jax.block_until_ready(g(*ja)), **budget), first)
            if torch is not None:
                ta = tuple(torch.tensor(x, dtype=torch.float64) for x in a)

                def tgrad():
                    r = ta[0].clone().requires_grad_(True)
                    o = ta[1].clone().requires_grad_(True)
                    dm_py.ln_likelihood_ratio(r, o, ta[2]).sum().backward()

                with torch.no_grad():
                    t_val = timeit(lambda: dm_py.ln_likelihood_ratio(*ta), **budget)
                row("torch eager (CPU)", phase, n, t_val, timeit(tgrad, **budget))
    lines.append("")


def bilby_section(lines, cache):
    """Append the comparison with Bilby's lookup table to the report.

    Parameters
    ----------
    lines : list of str
        Report lines, modified in place.
    cache : pathlib.Path
        Directory for Bilby's lookup tables.
    """
    import bilby

    bilby.core.utils.logger.setLevel("ERROR")
    cache.mkdir(parents=True, exist_ok=True)
    lines.append("## Comparison with Bilby's lookup table\n")
    prior = bilby.gw.prior.UniformSourceFrame(name="luminosity_distance", minimum=100, maximum=5000)
    rng = np.random.default_rng(42)
    lines.append("Posterior-like samples: true d ~ prior-ish in [200, 4000] Mpc, network SNR ρ₀ ∈ {8, 15, 30, 100}, "
                 "ρ_mf ~ ρ₀ + N(0,1), ρ_opt at the true distance ~ ρ₀(1 + 0.1 N(0,1)).  Errors are against the "
                 "dense reference quadrature with the same (Bilby-cosmology) prior.\n")
    lines.append("| phase | ρ₀ | N | distmarg max | distmarg 99% | bilby max | bilby 99% | bilby median | "
                 "distmarg ns/eval | bilby ns/eval |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for phase in (False, True):
        ifos = bilby.gw.detector.InterferometerList(["H1"])
        ifos.set_strain_data_from_zero_noise(sampling_frequency=512, duration=4, start_time=0)

        def zero_model(f, **kwargs):
            """Return a zero waveform (only the distance-marginalization machinery is used).

            Parameters
            ----------
            f : numpy.ndarray
                Frequencies.
            **kwargs
                Ignored source parameters.

            Returns
            -------
            dict
                Zero ``plus`` and ``cross`` polarizations.
            """
            z = np.zeros_like(f, dtype=complex)
            return {"plus": z, "cross": z}

        wg = bilby.gw.WaveformGenerator(duration=4, sampling_frequency=512, frequency_domain_source_model=zero_model)
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = prior
        lik = bilby.gw.likelihood.GravitationalWaveTransient(
            ifos, wg, priors=priors, distance_marginalization=True, phase_marginalization=phase,
            distance_marginalization_lookup_table=str(cache / f"bilby_usf_100_5000_{'phase' if phase else 'nophase'}.npz"),
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dm = DistanceMarginalization.from_bilby_prior(prior, phase_marginalization=phase)
        d_ref = lik._ref_dist
        for rho0 in [8, 15, 30, 100]:
            n = 400
            d_true = rng.uniform(200, 4000, n)
            rho_mf = rho0 + rng.standard_normal(n)
            rho_opt_true = rho0 * (1 + 0.1 * rng.standard_normal(n))
            rho_opt = rho_opt_true * d_true / d_ref  # at d_ref
            d_inner_h = rho_mf * rho_opt  # at d_ref
            h_inner_h = rho_opt**2
            ours = dm.ln_likelihood_ratio(rho_mf, rho_opt, d_ref)
            theirs = lik.distance_marginalized_likelihood(d_inner_h, h_inner_h, parameters={"luminosity_distance": d_ref})
            ref = np.array([reference_ln_likelihood_ratio(dm.prior, phase, a, b, d_ref) for a, b in zip(rho_mf, rho_opt)])
            eo, eb = np.abs(ours - ref), np.abs(theirs - ref)
            t_o = timeit(lambda: dm.ln_likelihood_ratio(rho_mf, rho_opt, d_ref), min_time=0.2) / n
            t_b = timeit(lambda: lik.distance_marginalized_likelihood(
                d_inner_h, h_inner_h, parameters={"luminosity_distance": d_ref}), min_time=0.2) / n
            lines.append(f"| {phase} | {rho0} | {n} | {eo.max():.1e} | {np.quantile(eo, .99):.1e} | {eb.max():.1e} | "
                         f"{np.quantile(eb, .99):.1e} | {np.median(eb):.1e} | {t_o * 1e9:.0f} | {t_b * 1e9:.0f} |")
    lines.append("")


def main(argv=None):
    """Run the benchmarks from the command line.

    Parameters
    ----------
    argv : list of str, optional
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--bilby", action="store_true")
    ap.add_argument("--backends-only", action="store_true", help="only run the JAX/torch/numpy backend timings")
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 100, 10_000, 1_000_000])
    ap.add_argument("--cache", type=Path, default=ROOT / "benchmarks" / "cache")
    ap.add_argument("--out", type=Path, default=ROOT / "benchmarks" / "python_report.md")
    args = ap.parse_args(argv)
    lines = ["# distmarg Python benchmark report\n"]
    if args.backends_only:
        backends_section(lines, args.sizes)
        args.out = args.out.with_name("backends_report.md")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("\n".join(lines))
        return
    accuracy_section(lines, 14 if args.quick else 30)
    speed_section(lines, args.quick)
    backends_section(lines, [1, 100, 10_000] if args.quick else args.sizes)
    if args.bilby:
        bilby_section(lines, args.cache)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
