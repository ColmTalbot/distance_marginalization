import warnings

import numpy as np
import pytest

bilby = pytest.importorskip("bilby")
pytest.importorskip("astropy")

from distmarg import DistanceMarginalization  # noqa: E402
from distmarg.bilby import prior_to_arrays  # noqa: E402


@pytest.fixture(scope="module")
def usf():
    """Return Bilby's UniformSourceFrame prior on [100, 5000] Mpc.

    Returns
    -------
    bilby.gw.prior.UniformSourceFrame
    """
    return bilby.gw.prior.UniformSourceFrame(name="luminosity_distance", minimum=100, maximum=5000)


def test_usf_matches_bilby_density(usf):
    """Check the recomputed USF density against Bilby's interpolated ``prob``."""
    d, lp = prior_to_arrays(usf, n=2048)
    assert d[0] == pytest.approx(100) and d[-1] == pytest.approx(5000)
    # Bilby's own (interpolated) density agrees up to its 1000-point interpolation error
    mid = slice(10, -10)
    ours = lp[mid] - np.log(np.trapezoid(np.exp(lp), d))
    theirs = np.log(usf.prob(d[mid]))
    assert np.max(np.abs(ours - theirs)) < 1e-3


def test_from_bilby_prior_fits_smoothly(usf):
    """Check ``from_bilby_prior`` fits without a :class:`FitWarning`."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no FitWarning: the density is recomputed, not interpolated
        dm = DistanceMarginalization.from_bilby_prior(usf)
    assert dm.fit_info["max_residual"] < 1e-8


def test_interpolated_prior_would_warn(usf):
    """Check that fitting Bilby's piecewise-linear ``prob`` triggers the warning."""
    d = np.geomspace(100, 5000, 4096)
    from distmarg import FitWarning

    with pytest.warns(FitWarning):
        DistanceMarginalization(d, np.log(usf.prob(d)))


def test_redshift_defined_prior_gives_same_distance_prior(usf):
    """Check a redshift-defined prior yields the same distance prior."""
    zpr = usf.get_corresponding_prior("redshift")
    d1, l1 = prior_to_arrays(usf, n=512)
    d2, l2 = prior_to_arrays(zpr, n=512)
    np.testing.assert_allclose(d1, d2, rtol=1e-6)
    np.testing.assert_allclose(l1 - l1[0], l2 - l2[0], atol=1e-9)


def test_power_law_prior():
    """Check a non-cosmological prior is tabulated from ``ln_prob``."""
    pl = bilby.core.prior.PowerLaw(alpha=2, minimum=50, maximum=5000, name="luminosity_distance")
    d, lp = prior_to_arrays(pl, n=1024)
    np.testing.assert_allclose(np.diff(lp), np.diff(2 * np.log(d)), atol=1e-10)


def test_wrong_parameter_rejected():
    """Check priors on other parameters are rejected."""
    with pytest.raises(ValueError):
        prior_to_arrays(bilby.core.prior.Uniform(0, 1, name="mass_ratio"))
