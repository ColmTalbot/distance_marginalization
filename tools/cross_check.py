#!/usr/bin/env python
"""File-based Python <-> Rust cross-check (IMPLEMENTATION_NOTES §7.9).

For each prior and both phase flags:

1. write ``fixtures/stress_queries_<prior>.json`` (stress grid of rho_mf, rho_opt, d_ref),
2. evaluate with Python,
3. run the Rust ``distmarg-eval`` binary on the same files (no Python bindings),
4. compare values, regime labels and prior-fit info; report the worst cases.

Pass criteria: identical regimes away from knife-edge ties; |Δ| <= 1e-9 (relative to
max(1, |value|)); prior-fit coefficients/ln_norm/moments agree to 1e-12.

    python tools/cross_check.py [--n 40] [--priors usf_100_5000 ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from distmarg import REGIME_NAMES, DistanceMarginalization  # noqa: E402

PRIORS = ["usf_100_5000", "usf_10_50000", "pl2_50_5000", "plm1_50_5000"]


def stress_queries(d_min, d_max, n):
    """Build the stress queries for one prior.

    Parameters
    ----------
    d_min, d_max : float
        Prior support.
    n : int
        Points per axis.

    Returns
    -------
    numpy.ndarray
        Rows of ``(rho_mf, rho_opt, d_ref)``.
    """
    rows = []
    for ds in np.geomspace(d_min / 5, d_max * 5, n):
        for r in np.geomspace(1e-3, 1e4, n):
            for s in (1, -1):
                rows.append((s * r, r * ds / 1000.0, 1000.0))
    for ro in np.geomspace(1e-3, 1e3, n):
        for r in np.geomspace(1e-3, 1e4, n):
            for s in (1, -1):
                rows.append((s * r, ro * d_min / 1000.0, 1000.0))
    # a few varied reference distances
    rng = np.random.default_rng(3)
    for _ in range(4 * n):
        r = 10 ** rng.uniform(-3, 4) * rng.choice([-1, 1])
        ds = d_min / 5 * (25 * d_max / d_min) ** rng.uniform()
        dr = 10 ** rng.uniform(1, 4)
        rows.append((r, abs(r) * ds / dr, dr))
    return np.array(rows)


def build_rust():
    """Build the ``distmarg-eval`` binary in release mode.

    Returns
    -------
    pathlib.Path
        Path to the executable.
    """
    subprocess.run(
        ["cargo", "build", "--release", "--features", "cli", "--bin", "distmarg-eval"],
        cwd=ROOT / "rust", check=True,
    )
    return ROOT / "rust" / "target" / "release" / "distmarg-eval"


def main(argv=None):
    """Run the cross-check.

    Parameters
    ----------
    argv : list of str, optional

    Returns
    -------
    int
        Exit status: 0 if every comparison passed.
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--priors", nargs="*", default=PRIORS)
    ap.add_argument("--workdir", type=Path, default=None)
    ap.add_argument("--tol", type=float, default=1e-9)
    args = ap.parse_args(argv)
    work = args.workdir or Path(tempfile.mkdtemp(prefix="distmarg_xcheck_"))
    work.mkdir(parents=True, exist_ok=True)
    exe = build_rust()
    failures = 0
    for name in args.priors:
        prior_path = ROOT / "fixtures" / "priors" / f"{name}.json"
        pr = json.load(open(prior_path))
        q = stress_queries(pr["distance"][0], pr["distance"][-1], args.n)
        qpath = ROOT / "fixtures" / f"stress_queries_{name}.json"
        json.dump({"rho_mf": q[:, 0].tolist(), "rho_opt": q[:, 1].tolist(), "d_ref": q[:, 2].tolist()}, open(qpath, "w"))
        for phase in (False, True):
            tag = f"{name}{'_phase' if phase else ''}"
            out = work / f"results_rust_{tag}.json"
            fit = work / f"fit_rust_{tag}.json"
            cmd = [str(exe), "--prior", str(prior_path), "--queries", str(qpath), "--out", str(out), "--fit-info", str(fit)]
            if phase:
                cmd.append("--phase")
            subprocess.run(cmd, check=True)
            rust = json.load(open(out))
            r_val = np.array([np.nan if x is None else x for x in rust["ln_likelihood_ratio"]], dtype=float)
            r_reg = np.array(rust["regime"])

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dm = DistanceMarginalization(pr["distance"], pr["ln_prior"], phase)
            p_val = np.asarray(dm.ln_likelihood_ratio(*q.T))
            p_reg = np.asarray(dm.regime(*q.T))

            same = p_reg == r_reg
            scale = np.maximum(1.0, np.abs(p_val))
            rel = np.abs(p_val - r_val) / scale
            worst = int(np.argmax(np.where(same, rel, 0)))
            ok_val = np.nanmax(np.where(same, rel, 0)) <= args.tol and np.all(np.isfinite(r_val) == np.isfinite(p_val))
            # mismatched regimes are allowed only at knife-edge ties: the values must still agree
            # to the boundary-continuity tolerance
            mism = np.nonzero(~same)[0]
            ok_mism = len(mism) <= 2 and np.all(np.abs(p_val[mism] - r_val[mism]) < 1e-4)

            rf = json.load(open(fit))
            pf = dm.fit_info
            cmax = np.max(np.abs(pf["coeffs"]))
            fit_ok = (
                rf["degree"] == pf["degree"]
                and np.max(np.abs(np.array(rf["coeffs"]) - pf["coeffs"])) <= 1e-12 * cmax
                and abs(rf["ln_norm"] - pf["ln_norm"]) <= 1e-12 * max(1, abs(pf["ln_norm"]))
                and np.max(np.abs(np.array(rf["moments"]) / pf["moments"] - 1)) <= 1e-12
            )
            status = "OK " if (ok_val and ok_mism and fit_ok) else "FAIL"
            failures += status != "OK "
            counts = {REGIME_NAMES[c]: int(np.sum(p_reg == c)) for c in REGIME_NAMES}
            print(
                f"[{status}] {tag:24s} n={len(q)} regimes={counts} mismatched={len(mism)} "
                f"max rel |Δ|={np.nanmax(np.where(same, rel, 0)):.2e} at {tuple(np.round(q[worst], 6))} "
                f"({REGIME_NAMES.get(int(p_reg[worst]))}); fit ok={fit_ok}"
            )
            for i in mism:
                print(f"        regime mismatch at {tuple(q[i])}: py={p_reg[i]} rust={r_reg[i]} Δ={p_val[i] - r_val[i]:.2e}")
    print(f"results in {work}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
