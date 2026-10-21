#!/usr/bin/env python
"""Bilby's ``fast_tutorial.py`` with distance marginalization done by ``distmarg``.

This mirrors ``examples/gw_examples/injection_examples/fast_tutorial.py`` from
Bilby: a 4 s, 2048 Hz IMRPhenomPv2 binary-black-hole injection into simulated
H1/L1 design-sensitivity noise, with everything except ``chirp_mass``,
``mass_ratio``, ``theta_jn`` and ``luminosity_distance`` fixed to the injected
values.  ``luminosity_distance`` keeps Bilby's default BBH prior,
``UniformSourceFrame(100, 5000)`` Mpc (Planck15), which contains the injected
2000 Mpc.

Differences from the Bilby example:

- the likelihood is :class:`distmarg_likelihood.DistmargGravitationalWaveTransient`,
  so distance is marginalized analytically/numerically by ``distmarg`` and the
  sampler only sees ``chirp_mass``, ``mass_ratio`` and ``theta_jn``;
- before sampling, :mod:`compare_likelihoods` checks the new likelihood against
  Bilby's lookup table and a brute-force distance integral (skip with
  ``--skip-comparison``);
- dynesty uses fewer live points (``--nlive``, default 500 instead of 1000) and
  a process pool so the example finishes in tens of minutes on a laptop;
- after sampling, distance is reconstructed through Bilby's
  ``generate_all_bbh_parameters`` conversion (which calls the likelihood's
  ``generate_posterior_sample_from_marginalized_likelihood``), so the result has
  ``luminosity_distance`` samples;
- ``--compare-stock`` repeats the run with Bilby's lookup-table likelihood and
  compares ``ln Z`` and the distance posterior.

Run from anywhere with ``python examples/fast_tutorial_distmarg.py``.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import bilby
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from distmarg_likelihood import DistmargGravitationalWaveTransient  # noqa: E402

OUTDIR = os.path.join(HERE, "outdir")
LOOKUP_TABLE = os.path.join(OUTDIR, "distance_marginalization_lookup.npz")

DURATION = 4.0
SAMPLING_FREQUENCY = 2048.0
MINIMUM_FREQUENCY = 20.0
SEED = 88170235

INJECTION_PARAMETERS = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.4,
    a_2=0.3,
    tilt_1=0.5,
    tilt_2=1.0,
    phi_12=1.7,
    phi_jl=0.3,
    luminosity_distance=2000.0,
    theta_jn=0.4,
    psi=2.659,
    phase=1.3,
    geocent_time=1126259642.413,
    ra=1.375,
    dec=-1.2108,
)

FIXED_KEYS = ["a_1", "a_2", "tilt_1", "tilt_2", "phi_12", "phi_jl", "psi", "ra", "dec", "geocent_time", "phase"]


def make_setup(seed=SEED):
    """Create the injection, interferometers, waveform generator and priors.

    Parameters
    ----------
    seed : int, optional
        Seed for Bilby's global random generator (noise realization).

    Returns
    -------
    ifos : bilby.gw.detector.InterferometerList
        H1 and L1 with the injected signal.
    waveform_generator : bilby.gw.WaveformGenerator
        IMRPhenomPv2 frequency-domain generator.
    priors : bilby.gw.prior.BBHPriorDict
        Fast-tutorial priors; ``luminosity_distance`` is ``UniformSourceFrame``
        on [100, 5000] Mpc.  Pass a deep copy to each likelihood because
        marginalized likelihoods modify the dictionary in place.
    """
    bilby.core.utils.random.seed(seed)
    waveform_arguments = dict(
        waveform_approximant="IMRPhenomPv2",
        reference_frequency=50.0,
        minimum_frequency=MINIMUM_FREQUENCY,
    )
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=DURATION,
        sampling_frequency=SAMPLING_FREQUENCY,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
    )
    ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])
    ifos.set_strain_data_from_power_spectral_densities(
        sampling_frequency=SAMPLING_FREQUENCY,
        duration=DURATION,
        start_time=INJECTION_PARAMETERS["geocent_time"] - 2,
    )
    ifos.inject_signal(waveform_generator=waveform_generator, parameters=INJECTION_PARAMETERS)

    priors = bilby.gw.prior.BBHPriorDict()
    for key in FIXED_KEYS:
        priors[key] = INJECTION_PARAMETERS[key]
    priors["luminosity_distance"] = bilby.gw.prior.UniformSourceFrame(
        minimum=100.0, maximum=5000.0, name="luminosity_distance", latex_label="$d_L$", unit="Mpc"
    )
    priors.validate_prior(DURATION, MINIMUM_FREQUENCY)
    return ifos, waveform_generator, priors


def make_likelihood(kind, ifos, waveform_generator, priors, phase_marginalization=False):
    """Build one of the likelihoods used in the example.

    Parameters
    ----------
    kind : {"distmarg", "stock", "plain"}
        ``distmarg``: :class:`DistmargGravitationalWaveTransient`; ``stock``:
        Bilby's lookup-table distance marginalization (table cached in
        ``outdir/``); ``plain``: no distance marginalization.
    ifos, waveform_generator, priors
        As returned by :func:`make_setup`.  ``priors`` is modified in place for
        the marginalized kinds.
    phase_marginalization : bool, optional
        Also marginalize over phase (``phase`` must then have a non-delta prior).

    Returns
    -------
    bilby.gw.likelihood.GravitationalWaveTransient
    """
    common = dict(
        interferometers=ifos,
        waveform_generator=waveform_generator,
        priors=priors,
        phase_marginalization=phase_marginalization,
    )
    if kind == "distmarg":
        return DistmargGravitationalWaveTransient(**common)
    if kind == "stock":
        table = LOOKUP_TABLE.replace(".npz", "_phase.npz" if phase_marginalization else ".npz")
        return bilby.gw.likelihood.GravitationalWaveTransient(
            distance_marginalization=True, distance_marginalization_lookup_table=table, **common
        )
    if kind == "plain":
        return bilby.gw.likelihood.GravitationalWaveTransient(**common)
    raise ValueError(kind)


def run(kind, nlive, npool, label=None):
    """Run dynesty on the fast-tutorial problem with a distance-marginalized likelihood.

    Parameters
    ----------
    kind : {"distmarg", "stock"}
        Which likelihood to use (see :func:`make_likelihood`).
    nlive : int
        Number of live points.
    npool : int
        Number of processes for the likelihood and for reconstruction.
    label : str, optional
        Output label; defaults to ``fast_tutorial_<kind>``.

    Returns
    -------
    result : bilby.gw.result.CBCResult
        Result with reconstructed ``luminosity_distance`` samples.
    wall : float
        Wall-clock time in seconds including reconstruction.
    """
    label = label or f"fast_tutorial_{kind}"
    ifos, waveform_generator, priors = make_setup()
    likelihood = make_likelihood(kind, ifos, waveform_generator, priors)
    t0 = time.perf_counter()
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler="dynesty",
        nlive=nlive,
        sample="acceptance-walk",
        naccept=60,
        npool=npool,
        injection_parameters=INJECTION_PARAMETERS,
        outdir=OUTDIR,
        label=label,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        result_class=bilby.gw.result.CBCResult,
        clean=True,
        seed=SEED,
    )
    wall = time.perf_counter() - t0
    plot_corner(result, label)
    return result, wall


def plot_corner(result, label):
    """Corner plot of the sampled parameters and reconstructed distance with injected values.

    Parameters
    ----------
    result : bilby.core.result.Result
        Result with ``luminosity_distance`` in the posterior.
    label : str
        Output label; the figure is ``outdir/<label>_corner.png``.
    """
    truth = bilby.gw.conversion.generate_mass_parameters(dict(INJECTION_PARAMETERS))
    result.plot_corner(
        parameters={key: truth[key] for key in ["chirp_mass", "mass_ratio", "theta_jn", "luminosity_distance"]},
        filename=os.path.join(OUTDIR, f"{label}_corner.png"),
    )


def summarize(result, wall, name):
    """Print ``ln Z`` and the ``luminosity_distance`` posterior summary.

    Parameters
    ----------
    result : bilby.core.result.Result
        Sampler result with ``luminosity_distance`` in the posterior.
    wall : float
        Wall-clock time in seconds.
    name : str
        Label for the printout.

    Returns
    -------
    dict
        Summary numbers.
    """
    d = result.posterior["luminosity_distance"].to_numpy()
    q05, q50, q95 = np.quantile(d, [0.05, 0.5, 0.95])
    out = dict(
        name=name,
        wall_s=wall,
        sampling_time_s=float(getattr(result, "sampling_time", np.nan) or np.nan),
        ln_z=result.log_evidence,
        ln_z_err=result.log_evidence_err,
        ln_bf=result.log_bayes_factor,
        n_samples=len(d),
        d_median=q50,
        d_5=q05,
        d_95=q95,
        d_injection_quantile=float(np.mean(d < INJECTION_PARAMETERS["luminosity_distance"])),
        n_likelihood_evaluations=result.num_likelihood_evaluations,
    )
    print(
        f"[{name}] wall {wall:.0f} s, ln Z = {out['ln_z']:.3f} +/- {out['ln_z_err']:.3f} "
        f"(ln BF = {out['ln_bf']:.3f}), {len(d)} samples, d_L median {q50:.0f} Mpc, "
        f"90% CI [{q05:.0f}, {q95:.0f}], injected 2000 Mpc at quantile {out['d_injection_quantile']:.2f}"
    )
    return out


def compare_distance_posteriors(results, filename):
    """Overlay the reconstructed ``luminosity_distance`` posteriors.

    Parameters
    ----------
    results : dict
        Mapping of name to result.
    filename : str
        Output figure path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import ks_2samp

    fig, ax = plt.subplots(figsize=(5, 3.5))
    bins = np.linspace(100, 5000, 60)
    for name, res in results.items():
        ax.hist(res.posterior["luminosity_distance"], bins=bins, density=True, histtype="step", lw=1.5,
                label=f"{name} (ln Z = {res.log_evidence:.2f})")
    ax.axvline(INJECTION_PARAMETERS["luminosity_distance"], color="k", ls="--", lw=1, label="injection")
    ax.set_xlabel("$d_L$ [Mpc]")
    ax.set_ylabel("density")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(filename, dpi=120)
    plt.close(fig)
    names = list(results)
    if len(names) == 2:
        a, b = (results[n].posterior["luminosity_distance"] for n in names)
        print(f"KS test {names[0]} vs {names[1]} luminosity_distance: {ks_2samp(a, b)}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nlive", type=int, default=500)
    parser.add_argument("--npool", type=int, default=8)
    parser.add_argument("--n-compare", type=int, default=100)
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--skip-sampling", action="store_true")
    parser.add_argument("--compare-stock", action="store_true", help="also sample with Bilby's lookup table")
    args = parser.parse_args()

    os.makedirs(OUTDIR, exist_ok=True)
    bilby.core.utils.setup_logger(outdir=OUTDIR, label="fast_tutorial_distmarg")

    if not args.skip_comparison:
        import compare_likelihoods

        compare_likelihoods.main(n_samples=args.n_compare)

    if args.skip_sampling:
        return
    results, summaries = {}, []
    kinds = ["distmarg", "stock"] if args.compare_stock else ["distmarg"]
    for kind in kinds:
        result, wall = run(kind, args.nlive, args.npool)
        results[kind] = result
        summaries.append(summarize(result, wall, kind))
    if len(results) > 1:
        compare_distance_posteriors(results, os.path.join(OUTDIR, "luminosity_distance_comparison.png"))
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
