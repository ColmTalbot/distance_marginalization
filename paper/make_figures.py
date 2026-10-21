#!/usr/bin/env python
"""Generate the figures and quoted numbers for the distmarg report.

All comparisons use Bilby's ``UniformSourceFrame`` prior on [100, 5000] Mpc
(Planck15 cosmology), so that the Bilby lookup table is compared like for like.
Errors are measured against a dense composite Gauss-Legendre quadrature of the
same continuous prior.

    python paper/make_figures.py            # writes paper/figures/*.pdf and paper/numbers.json

The Bilby lookup tables (one per phase setting, ~40 s each to build) are cached in
``paper/.cache``.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
import warnings
from pathlib import Path

os.environ.setdefault("SCIPY_ARRAY_API", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
CACHE = HERE / ".cache"
ROOT = HERE.parent

D_MIN, D_MAX = 100.0, 5000.0
D_REF = 1000.0
COLORS = {
    "reference": "black",
    "high": "#1b6ca8",
    "prior": "#d1495b",
    "quad": "#3a9a5b",
    "selected": "black",
    "bilby": "#e09f3e",
}
REGIME_COLORS = ["#cfe0f1", "#f6d3d8", "#d6ecdc"]  # high SNR, prior dominated, quadrature


def style():
    """Set a compact serif style matching a two-column physics journal."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 6.5,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "lines.linewidth": 1.1,
        "axes.linewidth": 0.6,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "mathtext.fontset": "cm",
    })


def build():
    """Build the prior and the distmarg objects for both phase settings.

    Returns
    -------
    prior : bilby.gw.prior.UniformSourceFrame
        The comparison prior.
    dms : dict
        ``{phase: DistanceMarginalization}`` with the Rust backend when available.
    dms_py : dict
        The same with the pure-Python backend.
    """
    import bilby
    from distmarg import DistanceMarginalization

    bilby.core.utils.logger.setLevel("ERROR")
    prior = bilby.gw.prior.UniformSourceFrame(name="luminosity_distance", minimum=D_MIN, maximum=D_MAX)
    dms, dms_py = {}, {}
    for phase in (False, True):
        dms[phase] = DistanceMarginalization.from_bilby_prior(prior, phase_marginalization=phase)
        dms_py[phase] = DistanceMarginalization.from_bilby_prior(prior, phase_marginalization=phase, backend="python")
    return prior, dms, dms_py


def reference(dm, phase, rho, rho_opt, d_ref=D_REF):
    """Dense reference quadrature for arrays of inputs.

    Parameters
    ----------
    dm : DistanceMarginalization
    phase : bool
    rho, rho_opt : array_like
    d_ref : float, optional

    Returns
    -------
    numpy.ndarray
    """
    from distmarg._quadrature import reference_ln_likelihood_ratio

    rho, rho_opt = np.broadcast_arrays(np.asarray(rho, float), np.asarray(rho_opt, float))
    return np.array([reference_ln_likelihood_ratio(dm.prior, phase, r, o, d_ref)
                     for r, o in zip(rho.ravel(), rho_opt.ravel())]).reshape(rho.shape)


def branch(dm, name, rho, rho_opt, d_ref=D_REF):
    """Evaluate one evaluation branch regardless of the regime selection.

    Parameters
    ----------
    dm : DistanceMarginalization
    name : {"high", "prior", "quad"}
    rho, rho_opt : numpy.ndarray
    d_ref : float, optional

    Returns
    -------
    numpy.ndarray
    """
    rm = np.abs(rho) if dm.phase_marginalization else np.asarray(rho, float)
    k = np.asarray(rho_opt, float) * d_ref
    fn = {"high": dm._branch_high, "prior": dm._branch_cumulant, "quad": dm._branch_quadrature}[name]
    return fn(np, rm, k)


def bilby_tables():
    """Build (or load) Bilby's lookup-table likelihoods for both phase settings.

    Returns
    -------
    dict
        ``{phase: GravitationalWaveTransient}``.
    """
    import bilby

    CACHE.mkdir(exist_ok=True)

    def zero_model(f, **kwargs):
        z = np.zeros_like(f, dtype=complex)
        return {"plus": z, "cross": z}

    out = {}
    for phase in (False, True):
        ifos = bilby.gw.detector.InterferometerList(["H1"])
        ifos.set_strain_data_from_zero_noise(sampling_frequency=512, duration=4, start_time=0)
        wfg = bilby.gw.WaveformGenerator(duration=4, sampling_frequency=512, frequency_domain_source_model=zero_model)
        priors = bilby.gw.prior.BBHPriorDict()
        priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
            name="luminosity_distance", minimum=D_MIN, maximum=D_MAX)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out[phase] = bilby.gw.likelihood.GravitationalWaveTransient(
                ifos, wfg, priors=priors, distance_marginalization=True, phase_marginalization=phase,
                distance_marginalization_lookup_table=str(CACHE / f"bilby_lookup_{'phase' if phase else 'nophase'}.npz"),
            )
    return out


def bilby_eval(lik, rho, rho_opt, d_ref=D_REF):
    """Evaluate Bilby's lookup table at (rho_mf, rho_opt given at d_ref).

    Parameters
    ----------
    lik : GravitationalWaveTransient
    rho, rho_opt : numpy.ndarray
    d_ref : float, optional

    Returns
    -------
    numpy.ndarray
    """
    scale = d_ref / lik._ref_dist  # move the inner products to Bilby's reference distance
    d_inner_h = rho * rho_opt * scale
    h_inner_h = (rho_opt * scale) ** 2
    return np.asarray(lik.distance_marginalized_likelihood(
        d_inner_h, h_inner_h, parameters={"luminosity_distance": lik._ref_dist}), dtype=float)


def fig_limiting_forms(dms, numbers):
    """Exact ln I across SNR against the two limiting forms and the quadrature."""
    dstar = 1000.0
    rho = np.geomspace(1e-2, 1e3, 160)
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 3.6), sharex=True, gridspec_kw=dict(height_ratios=[1.2, 1], hspace=0.06, wspace=0.28))
    worst = {}
    for col, phase in enumerate((False, True)):
        dm = dms[phase]
        rho_opt = rho * dstar / D_REF
        ref = reference(dm, phase, rho, rho_opt)
        vals = {n: branch(dm, n, rho, rho_opt) for n in ("high", "prior", "quad")}
        sel = np.asarray(dm.ln_likelihood_ratio(rho, rho_opt, D_REF))
        reg = np.asarray(dm.regime(rho, rho_opt, D_REF))
        top, bot = axes[0, col], axes[1, col]
        for ax in (top, bot):
            edges = np.concatenate([[rho[0]], np.sqrt(rho[1:] * rho[:-1]), [rho[-1]]])
            for i, r in enumerate(reg):
                ax.axvspan(edges[i], edges[i + 1], color=REGIME_COLORS[r], lw=0)
        half = 0.5 * rho**2
        top.plot(rho, ref - half, color=COLORS["reference"], lw=1.6, label="exact")
        top.plot(rho, np.where(rho > 3, vals["high"] - half, np.nan), "--", color=COLORS["high"], label="high-SNR expansion")
        top.plot(rho, np.where(rho < 6, vals["prior"] - half, np.nan), ":", color=COLORS["prior"], lw=1.4, label="cumulant expansion")
        top.set_ylim(-14, 1)
        top.set_ylabel(r"$\ln\mathcal{L}_D/\mathcal{Z}_N - \rho_{\rm mf}^2/2$")
        top.set_title("no phase marginalization" if not phase else "phase marginalized")
        floor = 1e-16 + 1e-16 * np.abs(ref)
        for n, lbl, ls in (("high", "high-SNR expansion", "--"), ("prior", "cumulant expansion", ":"), ("quad", "six-panel quadrature", "-")):
            err = np.abs(vals[n] - ref)
            bot.loglog(rho, np.maximum(err, floor), ls, color=COLORS[n], lw=1.0 if n != "prior" else 1.3, label=lbl)
        err_sel = np.abs(sel - ref)
        bot.loglog(rho, np.maximum(err_sel, floor), color=COLORS["selected"], lw=1.6, label="selected")
        bot.axhline(1e-3, color="grey", lw=0.6, ls="-.")
        bot.set_ylim(1e-14, 1e2)
        bot.set_xlabel(r"$\rho_{\rm mf}$")
        bot.set_ylabel(r"$|\Delta\ln\mathcal{L}|$")
        bot.set_xlim(rho[0], rho[-1])
        worst[str(phase)] = float(err_sel.max())
    from matplotlib.patches import Patch
    handles, labels = axes[1, 0].get_legend_handles_labels()
    handles = [axes[0, 0].get_lines()[0]] + handles
    labels = ["exact"] + labels
    handles += [Patch(color=c) for c in REGIME_COLORS]
    labels += ["region: high-SNR", "region: cumulant", "region: quadrature"]
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.995), ncol=5, frameon=False)
    fig.savefig(FIG / "limiting_forms.pdf")
    plt.close(fig)
    numbers["limiting_forms"] = dict(dstar=dstar, worst_selected_error=worst)


def fig_regime_map(dms, numbers):
    """Regime used across the (d*, rho_mf) plane."""
    from matplotlib.colors import ListedColormap

    dstar = np.geomspace(D_MIN / 5, D_MAX * 5, 320)
    rho = np.geomspace(1e-2, 1e3, 320)
    D, R = np.meshgrid(dstar, rho)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True, gridspec_kw=dict(wspace=0.08))
    fractions = {}
    for ax, phase in zip(axes, (False, True)):
        reg = np.asarray(dms[phase].regime(R, R * D / D_REF, D_REF))
        ax.pcolormesh(D, R, reg, cmap=ListedColormap(REGIME_COLORS), vmin=-0.5, vmax=2.5, shading="auto", rasterized=True)
        for d in (D_MIN, D_MAX):
            ax.axvline(d, color="k", lw=0.7, ls="--")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$d_\star$ [Mpc]")
        ax.set_title("no phase marginalization" if not phase else "phase marginalized")
        fractions[str(phase)] = {name: float(np.mean(reg == code)) for code, name in enumerate(("high_snr", "prior_dominated", "quadrature"))}
    axes[0].set_ylabel(r"$\rho_{\rm mf}$")
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(color=c, label=l) for c, l in zip(REGIME_COLORS, ("high-SNR expansion", "cumulant expansion", "six-panel quadrature"))],
                   loc="upper right", frameon=True, framealpha=0.9)
    fig.savefig(FIG / "regime_map.pdf")
    plt.close(fig)
    numbers["regime_fractions"] = fractions


def fig_accuracy(dms, tables, numbers):
    """Worst-case and median error across d* as a function of SNR, against Bilby's lookup table."""
    rho = np.geomspace(1e-1, 1e3, 28)
    dstar = np.geomspace(D_MIN / 5, D_MAX * 5, 33)
    R, D = np.meshgrid(rho, dstar, indexing="ij")
    RO = R * D / D_REF
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
    summary = {}
    for ax, phase in zip(axes, (False, True)):
        dm = dms[phase]
        ref = reference(dm, phase, R, RO)
        ours = np.abs(np.asarray(dm.ln_likelihood_ratio(R, RO, D_REF)) - ref)
        theirs = np.abs(bilby_eval(tables[phase], R, RO) - ref)
        theirs = np.where(np.isfinite(theirs), theirs, np.nan)
        floor = 1e-14
        for err, color, lbl in ((ours, COLORS["selected"], "this work"), (theirs, COLORS["bilby"], "interpolated lookup table")):
            ax.loglog(rho, np.maximum(np.nanmax(err, axis=1), floor), color=color, lw=1.4, label=f"{lbl}, max over $d_\\star$")
            ax.loglog(rho, np.maximum(np.nanmedian(err, axis=1), floor), color=color, lw=1.0, ls="--", label=f"{lbl}, median")
        ax.axhline(1e-3, color="grey", lw=0.6, ls="-.")
        ax.set_xlabel(r"$\rho_{\rm mf}$")
        ax.set_title("no phase marginalization" if not phase else "phase marginalized")
        ax.set_ylim(1e-12, 1e2)
        summary[str(phase)] = dict(
            ours_max=float(np.nanmax(ours)), ours_median=float(np.nanmedian(ours)),
            table_max=float(np.nanmax(theirs)), table_median=float(np.nanmedian(theirs)),
            table_nonfinite=int(np.sum(~np.isfinite(bilby_eval(tables[phase], R, RO)))),
            n=int(R.size),
        )
    axes[0].set_ylabel(r"$|\Delta\ln\mathcal{L}|$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.99), ncol=4, frameon=False)
    fig.savefig(FIG / "accuracy.pdf")
    plt.close(fig)
    numbers["accuracy_vs_table"] = summary


def _best(fn, reps, repeat=5):
    """Return the best mean time per call over ``repeat`` blocks of ``reps`` calls."""
    fn()
    best = np.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        best = min(best, (time.perf_counter() - t0) / reps)
    return best


def fig_timing(dms, dms_py, tables, numbers):
    """Cost per evaluation as a function of SNR for each evaluation path."""
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    rho_grid = np.geomspace(1e-1, 1e3, 13)
    n = 10_000
    rng = np.random.default_rng(1)
    res = {k: [] for k in ("rust_scalar", "rust_batch", "numpy_batch", "jax_batch", "table_scalar", "table_batch")}
    regimes = []
    dm, dm_py, tab = dms[False], dms_py[False], tables[False]
    f_jax = jax.jit(dm_py.ln_likelihood_ratio)
    for r in rho_grid:
        dstar = D_MIN * (D_MAX / D_MIN) ** rng.uniform(size=n)
        rho = np.full(n, r)
        ro = rho * dstar / D_REF
        regimes.append(np.bincount(np.asarray(dm.regime(rho, ro, D_REF)), minlength=3) / n)
        args = [(float(rho[i]), float(ro[i]), D_REF) for i in range(200)]
        res["rust_scalar"].append(_best(lambda: [dm.ln_likelihood_ratio(*a) for a in args], 5) / len(args))
        res["rust_batch"].append(_best(lambda: dm.ln_likelihood_ratio(rho, ro, D_REF), 5) / n)
        res["numpy_batch"].append(_best(lambda: dm_py.ln_likelihood_ratio(rho, ro, D_REF), 1, 3) / n)
        jr, jo = jnp.asarray(rho), jnp.asarray(ro)
        f_jax(jr, jo, D_REF).block_until_ready()
        res["jax_batch"].append(_best(lambda: f_jax(jr, jo, D_REF).block_until_ready(), 3) / n)
        targs = [(rho[i:i + 1], ro[i:i + 1]) for i in range(200)]
        res["table_scalar"].append(_best(lambda: [bilby_eval(tab, a, b) for a, b in targs], 5) / len(targs))
        res["table_batch"].append(_best(lambda: bilby_eval(tab, rho, ro), 5) / n)
    regimes = np.array(regimes)
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    edges = np.concatenate([[rho_grid[0]], np.sqrt(rho_grid[1:] * rho_grid[:-1]), [rho_grid[-1]]])
    for i, frac in enumerate(regimes):
        ax.axvspan(edges[i], edges[i + 1], color=REGIME_COLORS[int(np.argmax(frac))], lw=0)
    styles = {
        "rust_scalar": ("compiled, single call", COLORS["selected"], "-", "o"),
        "rust_batch": ("compiled, batch of $10^4$", COLORS["selected"], "--", "s"),
        "jax_batch": ("JAX (jit, CPU), batch", COLORS["high"], "--", "^"),
        "numpy_batch": ("NumPy, batch", COLORS["quad"], "--", "v"),
        "table_scalar": ("lookup table, single call", COLORS["bilby"], "-", "o"),
        "table_batch": ("lookup table, batch", COLORS["bilby"], "--", "s"),
    }
    for key, (lbl, c, ls, m) in styles.items():
        ax.loglog(rho_grid, np.array(res[key]) * 1e9, ls=ls, marker=m, ms=2.5, color=c, label=lbl)
    ax.set_xlabel(r"$\rho_{\rm mf}$")
    ax.set_ylabel("time per evaluation [ns]")
    ax.set_xlim(rho_grid[0], rho_grid[-1])
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=6)
    fig.savefig(FIG / "timing.pdf")
    plt.close(fig)
    numbers["timing_ns"] = {k: [float(x * 1e9) for x in v] for k, v in res.items()}
    numbers["timing_rho"] = [float(x) for x in rho_grid]
    numbers["timing_regime_fractions"] = regimes.tolist()


def oracle_summary(numbers):
    """Worst error against the independent arbitrary-precision oracle fixtures."""
    from distmarg import DistanceMarginalization

    out = {}
    for name in ("usf_100_5000", "usf_10_50000", "pl2_50_5000", "plm1_50_5000"):
        pr = json.load(open(ROOT / "fixtures" / "priors" / f"{name}.json"))
        o = json.load(open(ROOT / "fixtures" / f"oracle_{name}.json"))
        q = [np.array(o[k]) for k in ("rho_mf", "rho_opt", "d_ref")]
        for phase, key in ((False, "ln_likelihood_ratio"), (True, "ln_likelihood_ratio_phase")):
            dm = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase)
            err = np.abs(np.asarray(dm.ln_likelihood_ratio(*q)) - np.array(o[key]))
            out[f"{name}{'_phase' if phase else ''}"] = float(err.max())
    numbers["oracle_max_error"] = out


def environment(numbers):
    """Record the software and hardware used for the timings."""
    import bilby
    import distmarg
    import jax
    import scipy

    try:
        cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    except OSError:
        cpu = platform.processor()
    numbers["environment"] = dict(
        cpu=cpu, python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
        jax=jax.__version__, bilby=bilby.__version__, distmarg=distmarg.__version__,
    )


def main():
    """Generate all figures and write ``numbers.json``."""
    style()
    FIG.mkdir(exist_ok=True)
    numbers = {}
    environment(numbers)
    _, dms, dms_py = build()
    numbers["fit"] = {str(k): dict(degree=v.fit_info["degree"], max_residual=v.fit_info["max_residual"]) for k, v in dms.items()}
    tables = bilby_tables()
    fig_limiting_forms(dms, numbers)
    fig_regime_map(dms, numbers)
    fig_accuracy(dms, tables, numbers)
    oracle_summary(numbers)
    fig_timing(dms, dms_py, tables, numbers)
    (HERE / "numbers.json").write_text(json.dumps(numbers, indent=1))
    print(json.dumps(numbers, indent=1))


if __name__ == "__main__":
    main()
