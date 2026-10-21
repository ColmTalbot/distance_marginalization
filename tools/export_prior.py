#!/usr/bin/env python
"""Write (distance, ln_prior) prior arrays to JSON for both implementations.

Examples
--------
Built-in flat LCDM (H0=67.74, Om0=0.3075, no radiation) UniformSourceFrame prior::

    python tools/export_prior.py --usf 100 5000 --out fixtures/priors/usf_100_5000.json

Power law pi(d) ∝ d^alpha::

    python tools/export_prior.py --power-law 2 50 5000 --out fixtures/priors/pl2_50_5000.json

Any Bilby prior expression (needs bilby; uses the prior's own cosmology)::

    python tools/export_prior.py --bilby "bilby.gw.prior.UniformSourceFrame(name='luminosity_distance', minimum=100, maximum=5000)" --out p.json

Two-column CSV is written instead of JSON when ``--out`` ends in ``.csv``.
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

H0_DEFAULT = 67.74
OM0_DEFAULT = 0.3075
C_KMS = 299792.458


def flat_lcdm(h0=H0_DEFAULT, om0=OM0_DEFAULT):
    """Return the flat LCDM ``1/E(z)`` and Hubble distance.

    Parameters
    ----------
    h0 : float, optional
        Hubble constant in km/s/Mpc.
    om0 : float, optional
        Matter density (radiation neglected).

    Returns
    -------
    inv_efunc : callable
    d_hubble : float
        In Mpc.
    """
    d_hubble = C_KMS / h0

    def inv_efunc(z):
        """Compute ``1/E(z)``.

        Parameters
        ----------
        z : numpy.ndarray

        Returns
        -------
        numpy.ndarray
        """
        return 1.0 / np.sqrt(om0 * (1.0 + z) ** 3 + 1.0 - om0)

    return inv_efunc, d_hubble


def _comoving(z, inv_efunc, d_hubble, n_gl=64):
    """Compute the comoving distance by Gauss-Legendre quadrature.

    Parameters
    ----------
    z : numpy.ndarray
    inv_efunc : callable
    d_hubble : float
    n_gl : int, optional
        Nodes per point.

    Returns
    -------
    numpy.ndarray
        Comoving distance in Mpc.

    Notes
    -----
    Kept independent of astropy so that fixtures do not depend on its version.
    """
    x, w = np.polynomial.legendre.leggauss(n_gl)
    zz = 0.5 * z[:, None] * (x + 1.0)
    return d_hubble * 0.5 * z * np.sum(w * inv_efunc(zz), axis=1)


def usf_arrays(d_min, d_max, n=4096, h0=H0_DEFAULT, om0=OM0_DEFAULT, source_frame=True):
    """Tabulate a UniformSourceFrame (or UniformComovingVolume) prior on luminosity distance.

    Parameters
    ----------
    d_min, d_max : float
        Support in Mpc.
    n : int, optional
        Number of log-spaced points.
    h0, om0 : float, optional
        Flat LCDM parameters (radiation neglected).
    source_frame : bool, optional
        Include the ``(1+z)^{-1}`` time-dilation factor.

    Returns
    -------
    distance, ln_prior : numpy.ndarray
        Distances (Mpc) and unnormalized log density, with ``z(d)`` found by Newton
        iteration.
    """
    inv_efunc, d_hubble = flat_lcdm(h0, om0)
    d = np.geomspace(d_min, d_max, n)
    z = d / d_hubble
    for _ in range(100):
        dc = _comoving(z, inv_efunc, d_hubble)
        ddl = dc + (1.0 + z) * d_hubble * inv_efunc(z)
        step = ((1.0 + z) * dc - d) / ddl
        z = z - step
        if np.max(np.abs(step) / z) < 1e-15:
            break
    dc = _comoving(z, inv_efunc, d_hubble)
    ddl = dc + (1.0 + z) * d_hubble * inv_efunc(z)
    ln_p = np.log(4.0 * np.pi * d_hubble * dc**2 * inv_efunc(z)) - np.log(ddl)
    if source_frame:
        ln_p -= np.log1p(z)
    return d, ln_p


def power_law_arrays(alpha, d_min, d_max, n=4096):
    """Tabulate ``pi(d) ∝ d^alpha``.

    Parameters
    ----------
    alpha : float
    d_min, d_max : float
    n : int, optional

    Returns
    -------
    distance, ln_prior : numpy.ndarray
    """
    d = np.geomspace(d_min, d_max, n)
    return d, alpha * np.log(d)


def write_prior(path, distance, ln_prior, name="", description=""):
    """Write prior arrays to JSON (or CSV if ``path`` ends in ``.csv``).

    Parameters
    ----------
    path : str or pathlib.Path
    distance, ln_prior : array_like
    name, description : str, optional
    """
    distance = np.asarray(distance, dtype=float)
    ln_prior = np.asarray(ln_prior, dtype=float)
    if str(path).endswith(".csv"):
        np.savetxt(path, np.column_stack([distance, ln_prior]), delimiter=",", fmt="%.17e",
                   header="distance,ln_prior", comments="")
        return
    with open(path, "w") as f:
        json.dump(
            dict(name=name, description=description,
                 distance=[float(x) for x in distance], ln_prior=[float(x) for x in ln_prior]),
            f,
        )


def main(argv=None):
    """Export a prior from the command line.

    Parameters
    ----------
    argv : list of str, optional
    """
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--usf", nargs=2, type=float, metavar=("DMIN", "DMAX"))
    g.add_argument("--power-law", nargs=3, type=float, metavar=("ALPHA", "DMIN", "DMAX"))
    g.add_argument("--bilby", type=str, help="Python expression evaluating to a Bilby prior")
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--h0", type=float, default=H0_DEFAULT)
    ap.add_argument("--om0", type=float, default=OM0_DEFAULT)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.usf:
        d, lp = usf_arrays(*args.usf, n=args.n, h0=args.h0, om0=args.om0)
        name = f"usf_{args.usf[0]:g}_{args.usf[1]:g}"
        desc = f"UniformSourceFrame, flat LCDM H0={args.h0} Om0={args.om0} (no radiation)"
    elif args.power_law:
        alpha, dmin, dmax = args.power_law
        d, lp = power_law_arrays(alpha, dmin, dmax, n=args.n)
        name = f"powerlaw_{alpha:g}_{dmin:g}_{dmax:g}"
        desc = f"pi(d) ∝ d^{alpha:g}"
    else:
        import bilby  # noqa: F401

        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "python"))
        from distmarg.bilby import prior_to_arrays

        prior = eval(args.bilby, {"bilby": bilby, "math": math, "np": np})
        d, lp = prior_to_arrays(prior, n=args.n)
        name, desc = type(prior).__name__, args.bilby
    write_prior(args.out, d, lp, name=name, description=desc)
    print(f"wrote {args.out}: {len(d)} points on [{d[0]:g}, {d[-1]:g}]")


if __name__ == "__main__":
    main()
