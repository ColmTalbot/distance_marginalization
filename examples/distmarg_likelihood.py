"""Bilby ``GravitationalWaveTransient`` with distance marginalization done by ``distmarg``.

Bilby's stock distance marginalization tabulates ``ln(L_D / Z_N)`` on a 400 x 800
grid in ``(<d|h>_ref, <h|h>_ref)`` and interpolates it with a bivariate spline.
:class:`DistmargGravitationalWaveTransient` keeps every other part of Bilby's
likelihood (waveform generation, inner products, time/phase/calibration
marginalization, the delta prior at ``_ref_dist``, posterior reconstruction) and
only swaps the two places that touch the lookup table:

- :meth:`DistmargGravitationalWaveTransient.distance_marginalized_likelihood`,
  which returns ``ln(L_D / Z_N)`` exactly where the interpolated table value was
  returned, so it slots into the plain, time-, phase- and calibration-marginalized
  code paths unchanged;
- :meth:`DistmargGravitationalWaveTransient.generate_distance_sample_from_marginalized_likelihood`,
  which draws ``d_L`` from the conditional posterior with
  :meth:`distmarg.DistanceMarginalization.sample_distance`.

No lookup table is built, loaded or cached.
"""
from __future__ import annotations

import numpy as np
from bilby.core.utils import logger
from bilby.core.utils import random as bilby_random
from bilby.gw.likelihood import GravitationalWaveTransient
from bilby.gw.likelihood.base import _fallback_to_parameters

from distmarg import DistanceMarginalization

__all__ = ["DistmargGravitationalWaveTransient"]


class DistmargGravitationalWaveTransient(GravitationalWaveTransient):
    """Gravitational-wave transient likelihood marginalized over distance with ``distmarg``.

    Parameters
    ----------
    interferometers : list or bilby.gw.detector.InterferometerList
        Interferometers containing data and noise PSDs.
    waveform_generator : bilby.gw.waveform_generator.WaveformGenerator
        Waveform generator.
    priors : bilby.core.prior.PriorDict
        Prior dictionary.  Must contain a non-fixed ``luminosity_distance`` prior
        (or a ``Cosmological`` ``redshift``/``comoving_distance`` prior).  As with
        Bilby's own marginalized likelihoods, the dictionary passed in is modified
        in place: ``luminosity_distance`` is replaced by a delta function at the
        reference distance ``_ref_dist`` (the prior median).
    distance_marginalization : bool, optional
        Must be ``True`` (the default); accepted for signature compatibility.
    distmarg_kwargs : dict, optional
        Extra keyword arguments for
        :meth:`distmarg.DistanceMarginalization.from_bilby_prior`
        (for example ``n`` or ``fit_tol``).
    n_grid : int, optional
        Resolution of the inverse-CDF grid used by
        :meth:`distmarg.DistanceMarginalization.sample_distance` during
        reconstruction.
    **kwargs
        Passed to :class:`bilby.gw.likelihood.GravitationalWaveTransient`
        (``time_marginalization``, ``phase_marginalization``, ...).  The
        ``distance_marginalization_lookup_table`` argument is ignored.

    Attributes
    ----------
    distance_marginalizer : distmarg.DistanceMarginalization
        The object that evaluates the marginalized likelihood ratio.

    Notes
    -----
    With Bilby's inner products rescaled to the reference distance,
    ``rho_opt = sqrt(<h|h>_ref)`` and ``rho_mf = <d|h>_ref / rho_opt`` (real part
    without phase marginalization, modulus with it), and the likelihood ratio is
    ``distance_marginalizer.ln_likelihood_ratio(rho_mf, rho_opt, _ref_dist)``.
    Both this and Bilby's table are normalized over the distance prior, so the two
    agree up to the table's interpolation and quadrature error.
    """

    def __init__(
        self,
        interferometers,
        waveform_generator,
        priors=None,
        distance_marginalization=True,
        distmarg_kwargs=None,
        n_grid=2000,
        **kwargs,
    ):
        if not distance_marginalization:
            raise ValueError(f"{type(self).__name__} always marginalizes over distance")
        kwargs.pop("distance_marginalization_lookup_table", None)
        self._distmarg_kwargs = dict(distmarg_kwargs or {})
        self._n_grid = int(n_grid)
        super().__init__(
            interferometers=interferometers,
            waveform_generator=waveform_generator,
            priors=priors,
            distance_marginalization=True,
            **kwargs,
        )

    # ------------------------------------------------------------------ set-up
    def _setup_distance_marginalization(self, lookup_table=None):
        """Build the ``distmarg`` marginalizer instead of Bilby's lookup table.

        Called from :class:`GravitationalWaveTransient.__init__` after
        ``_ref_dist`` is set and before the distance prior is replaced by a delta
        function.

        Parameters
        ----------
        lookup_table : object, optional
            Ignored; present for signature compatibility.
        """
        prior = self.priors["luminosity_distance"]
        self.distance_marginalizer = DistanceMarginalization.from_bilby_prior(
            prior,
            phase_marginalization=self.phase_marginalization,
            **self._distmarg_kwargs,
        )
        self._interp_dist_margd_loglikelihood = None
        info = self.distance_marginalizer.fit_info
        logger.info(
            "distmarg distance marginalization: prior %s on [%g, %g] Mpc, d_ref = %g Mpc, "
            "fit degree %s, max residual %.2e, phase_marginalization=%s",
            type(prior).__name__, prior.minimum, prior.maximum, self._ref_dist,
            info.get("degree"), info.get("max_residual", np.nan), self.phase_marginalization,
        )

    def cache_lookup_table(self):
        """Do nothing: there is no lookup table to cache."""

    def _create_lookup_table(self):
        """Do nothing: there is no lookup table to build."""

    # ------------------------------------------------------------------ helpers
    def _reference_snrs(self, d_inner_h, h_inner_h, parameters):
        """Convert Bilby inner products to ``(rho_mf, rho_opt)`` at ``_ref_dist``.

        Parameters
        ----------
        d_inner_h : complex or numpy.ndarray
            ``<d|h>`` at ``parameters['luminosity_distance']``; may carry trailing
            time-bin and/or leading calibration axes.
        h_inner_h : float or numpy.ndarray
            ``<h|h>`` at the same distance (scalar, or one value per calibration draw).
        parameters : dict
            Current parameters.

        Returns
        -------
        rho_mf, rho_opt : numpy.ndarray
            Broadcast-compatible arrays (``rho_opt`` gets trailing singleton axes
            when ``d_inner_h`` has more dimensions).
        """
        d_inner_h_ref, h_inner_h_ref = self._setup_rho(d_inner_h, h_inner_h, parameters=parameters)
        d_inner_h_ref = np.asarray(d_inner_h_ref)
        if self.phase_marginalization:
            d_inner_h_ref = np.abs(d_inner_h_ref)
        else:
            d_inner_h_ref = np.real(d_inner_h_ref)
        rho_opt = np.sqrt(np.maximum(np.asarray(np.real(h_inner_h_ref), dtype=float), 0.0))
        extra = d_inner_h_ref.ndim - rho_opt.ndim
        if rho_opt.ndim > 0 and extra > 0:
            rho_opt = rho_opt.reshape(rho_opt.shape + (1,) * extra)
        safe = np.where(rho_opt > 0, rho_opt, 1.0)
        rho_mf = d_inner_h_ref / safe
        return rho_mf, rho_opt

    # ------------------------------------------------------------------ likelihood
    def distance_marginalized_likelihood(self, d_inner_h, h_inner_h, parameters=None):
        """Distance-marginalized log likelihood ratio ``ln(L_D / Z_N)``.

        Drop-in replacement for the lookup-table interpolation in
        :meth:`GravitationalWaveTransient.distance_marginalized_likelihood`.

        Parameters
        ----------
        d_inner_h : complex or numpy.ndarray
            ``<d|h>`` at ``parameters['luminosity_distance']`` (complex; an array
            over time bins and/or calibration draws when those are marginalized).
        h_inner_h : float or numpy.ndarray
            ``<h|h>`` at the same distance.
        parameters : dict, optional
            Current parameters; defaults to ``self.parameters``.

        Returns
        -------
        float or numpy.ndarray
            ``ln(L_D / Z_N)`` with the shape of ``d_inner_h``.  Where ``<h|h> = 0``
            the template is identically zero and the ratio is exactly 0.
        """
        parameters = _fallback_to_parameters(self, parameters)
        rho_mf, rho_opt = self._reference_snrs(d_inner_h, h_inner_h, parameters)
        valid = rho_opt > 0
        out = self.distance_marginalizer.ln_likelihood_ratio(
            rho_mf, np.where(valid, rho_opt, 1.0), self._ref_dist
        )
        out = np.where(valid, out, 0.0)
        return float(out) if np.ndim(out) == 0 else out

    # ------------------------------------------------------------------ reconstruction
    def generate_distance_sample_from_marginalized_likelihood(self, signal_polarizations=None, parameters=None):
        """Draw ``luminosity_distance`` from its conditional posterior.

        Uses :meth:`distmarg.DistanceMarginalization.sample_distance` with Bilby's
        global random generator.  With phase marginalization the conditional is
        already marginal over phase, matching Bilby's own implementation.

        Parameters
        ----------
        signal_polarizations : dict, optional
            Polarizations at the reference distance.  They are rescaled in place
            to the new distance so that later reconstruction steps (phase) use it.
        parameters : dict, optional
            Current parameters; defaults to ``self.parameters``.

        Returns
        -------
        float
            The luminosity-distance sample in Mpc.
        """
        parameters = _fallback_to_parameters(self, parameters)
        parameters.update(self.get_sky_frame_parameters(parameters))
        if signal_polarizations is None:
            signal_polarizations = self.waveform_generator.frequency_domain_strain(parameters)
        d_inner_h, h_inner_h = self._calculate_inner_products(signal_polarizations, parameters=parameters)
        rho_mf, rho_opt = self._reference_snrs(d_inner_h, h_inner_h, parameters)
        new_distance = float(
            self.distance_marginalizer.sample_distance(
                rho_mf, rho_opt, self._ref_dist, rng=bilby_random.rng, n_grid=self._n_grid
            )
        )
        self._rescale_signal(signal_polarizations, new_distance)
        return new_distance
