"""Optional Bilby helper: convert a Bilby distance prior into ``(distance, ln_prior)`` arrays.

Bilby's ``Cosmological`` priors (``UniformSourceFrame``, ``UniformComovingVolume``)
are ``Interped`` priors on a 1000-point redshift grid, so their ``prob`` is a
piecewise-linear interpolant whose kinks would show up as fit residual.  For
those classes the density is recomputed with the prior's own astropy cosmology on
a log-spaced redshift grid:

    pi(z) ∝ (1+z)^{-1} dV_c/dz   (UniformSourceFrame; no (1+z)^{-1} for UniformComovingVolume)
    pi(d_L) = pi(z) / (d d_L/dz),   d d_L/dz = D_M + (1+z) D_H / E(z)  (flat)

The induced luminosity-distance prior is returned whatever parameter
(``redshift``, ``comoving_distance`` or ``luminosity_distance``) the prior is
defined in.  Other priors must be defined on ``luminosity_distance`` and are
evaluated with ``prior.ln_prob``.
"""
from __future__ import annotations

import warnings

import numpy as np


def luminosity_distance_prior_from_cosmology(cosmology, z_min, z_max, source_frame=True, n=4096):
    """Tabulate a uniform-in-comoving-volume prior on luminosity distance.

    Parameters
    ----------
    cosmology : astropy.cosmology.FLRW
        Flat cosmology.
    z_min, z_max : float
        Redshift support.
    source_frame : bool, optional
        Include the ``(1+z)^{-1}`` source-frame time-dilation factor
        (``UniformSourceFrame``); otherwise uniform in comoving volume.
    n : int, optional
        Number of points, log-spaced in redshift (hence close to log-spaced in
        distance).

    Returns
    -------
    distance : numpy.ndarray
        Luminosity distances in Mpc.
    ln_prior : numpy.ndarray
        Unnormalized log prior density in luminosity distance.

    Raises
    ------
    NotImplementedError
        For non-flat cosmologies.
    """
    if abs(getattr(cosmology, "Ok0", 0.0)) > 0:
        raise NotImplementedError("only flat cosmologies are supported")
    z = np.geomspace(z_min, z_max, n)
    d_l = cosmology.luminosity_distance(z).to_value("Mpc")
    d_c = cosmology.comoving_distance(z).to_value("Mpc")
    d_h = cosmology.hubble_distance.to_value("Mpc")
    ddl_dz = d_c + (1.0 + z) * d_h * cosmology.inv_efunc(z)
    ln_pz = np.log(cosmology.differential_comoving_volume(z).value)
    if source_frame:
        ln_pz = ln_pz - np.log1p(z)
    return d_l, ln_pz - np.log(ddl_dz)


def prior_to_arrays(prior, n: int = 4096):
    """Convert a Bilby distance prior to arrays for :class:`distmarg.DistanceMarginalization`.

    Parameters
    ----------
    prior : bilby.core.prior.Prior
        A ``bilby.gw.prior.Cosmological`` prior, or any prior on
        ``luminosity_distance``.
    n : int, optional
        Number of tabulation points.

    Returns
    -------
    distance : numpy.ndarray
        Strictly increasing luminosity distances.
    ln_prior : numpy.ndarray
        Log prior density at ``distance`` (unnormalized).

    Raises
    ------
    ValueError
        If a non-cosmological prior is not on ``luminosity_distance`` or its
        ``ln_prob`` is not finite on the support.

    Warns
    -----
    UserWarning
        For ``Cosmological`` subclasses without a known analytic redshift density,
        which fall back to the interpolated ``ln_prob``.
    """
    from bilby.gw.prior import Cosmological, UniformComovingVolume, UniformSourceFrame

    if isinstance(prior, Cosmological):
        z_min = float(prior._minimum["redshift"])
        z_max = float(prior._maximum["redshift"])
        if isinstance(prior, (UniformSourceFrame, UniformComovingVolume)):
            d, ln_p = luminosity_distance_prior_from_cosmology(
                prior.cosmology, z_min, z_max, isinstance(prior, UniformSourceFrame), n
            )
            # z_at_value round-trips the distance limits only to ~1e-8; pin the support
            # to Bilby's luminosity-distance limits exactly.
            d[0] = float(prior._minimum["luminosity_distance"])
            d[-1] = float(prior._maximum["luminosity_distance"])
            return d, ln_p
        warnings.warn(
            f"{type(prior).__name__}: no analytic redshift density known; using the "
            "interpolated prior.ln_prob, which will likely trigger the fit-residual warning",
            stacklevel=2,
        )
        return _from_ln_prob(prior.get_corresponding_prior("luminosity_distance"), n)

    if getattr(prior, "name", "luminosity_distance") != "luminosity_distance":
        raise ValueError(f"prior is defined on {prior.name!r}; expected luminosity_distance")
    return _from_ln_prob(prior, n)


def _from_ln_prob(prior, n):
    """Tabulate ``prior.ln_prob`` on a log-spaced grid over its support.

    Parameters
    ----------
    prior : bilby.core.prior.Prior
        Prior on luminosity distance.
    n : int
        Number of points.

    Returns
    -------
    distance, ln_prior : numpy.ndarray

    Raises
    ------
    ValueError
        If ``ln_prob`` is not finite everywhere on the grid.
    """
    d = np.geomspace(float(prior.minimum), float(prior.maximum), n)
    ln_p = np.asarray(prior.ln_prob(d), dtype=float)
    if not np.all(np.isfinite(ln_p)):
        raise ValueError("prior.ln_prob is not finite on the whole support")
    return d, ln_p
