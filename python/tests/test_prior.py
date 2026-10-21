import warnings

import numpy as np
import pytest

from distmarg import DistanceMarginalization, FitWarning, fit_prior
from numpy.polynomial.chebyshev import chebval
from conftest import PRIOR_NAMES, load_json, load_prior


def test_validation():
    """Check invalid prior arrays and fit degrees raise ``ValueError``."""
    d = np.geomspace(10, 1000, 100)
    lp = 2 * np.log(d)
    with pytest.raises(ValueError):
        DistanceMarginalization(d[::-1], lp)
    with pytest.raises(ValueError):
        DistanceMarginalization(np.concatenate([[0.0], d[1:]]), lp)
    bad = lp.copy()
    bad[5] = -np.inf
    with pytest.raises(ValueError):
        DistanceMarginalization(d, bad)
    with pytest.raises(ValueError):
        DistanceMarginalization(d, lp[:-1])
    with pytest.raises(ValueError):
        DistanceMarginalization(d[:10], lp[:10])  # too few points for K = 8
    with pytest.raises(ValueError):
        DistanceMarginalization(d, lp, fit_degree=60)  # needs 122 points


@pytest.mark.parametrize("name", PRIOR_NAMES)
def test_fit_matches_fixture_and_is_normalized(name):
    """Check the fit reproduces the stored fixture and is normalized."""
    pr = load_prior(name)
    fit = fit_prior(pr["distance"], pr["ln_prior"])
    ref = load_json(f"prior_fit_{name}.json")
    assert fit.degree == ref["degree"]
    np.testing.assert_allclose(fit.raw_coeffs, ref["coeffs"], rtol=0, atol=1e-12 * np.max(np.abs(ref["coeffs"])))
    assert abs(fit.ln_norm - ref["ln_norm"]) < 1e-12 * max(1, abs(ref["ln_norm"]))
    np.testing.assert_allclose(fit.moments, ref["moments"], rtol=1e-12)
    assert abs(fit.moments[0] - 1) < 1e-13
    assert fit.max_residual < 1e-8


@pytest.mark.parametrize("name", ["kinked_100_5000", "noisy_50_5000"])
def test_bad_prior_warns(name):
    """Check kinked and noisy priors trigger :class:`FitWarning`."""
    pr = load_prior(name)
    with pytest.warns(FitWarning):
        DistanceMarginalization(pr["distance"], pr["ln_prior"])


def test_power_law_fit_is_exact():
    """Check a power law is fitted exactly, with analytic derivatives and moments."""
    d = np.geomspace(50, 5000, 400)
    fit = fit_prior(d, 2 * np.log(d))
    assert fit.degree == 8
    # pi_v ∝ v^-4 -> g'(s) = -4, g'' = 0 exactly
    s = np.linspace(fit.s_a, fit.s_b, 11)
    t = fit.t_of_s(s)
    np.testing.assert_allclose(chebval(t, fit.d1), -4.0, atol=1e-9)
    np.testing.assert_allclose(chebval(t, fit.d2), 0.0, atol=1e-8)
    # analytic normalization of v^-4 on [1/5000, 1/50]: E[v] = 3/2 (b^-2 - a^-2)/(b^-3 - a^-3)... check moment ratio
    a, b = 1 / 5000, 1 / 50
    z = (a**-3 - b**-3) / 3
    ev = (a**-2 - b**-2) / 2 / z
    assert abs(fit.moments[1] / ev - 1) < 1e-12


def test_chebyshev_derivatives_finite_difference():
    """Check the derivative series against finite differences."""
    pr = load_prior("usf_10_50000")
    fit = fit_prior(pr["distance"], pr["ln_prior"])
    s = np.linspace(fit.s_a + 0.1, fit.s_b - 0.1, 7)
    h = 1e-5
    fd1 = (fit.ln_pi_v(s + h) - fit.ln_pi_v(s - h)) / (2 * h)
    h = 1e-3
    fd2 = (fit.ln_pi_v(s + h) - 2 * fit.ln_pi_v(s) + fit.ln_pi_v(s - h)) / h**2
    np.testing.assert_allclose(chebval(fit.t_of_s(s), fit.d1), fd1, atol=1e-8)
    np.testing.assert_allclose(chebval(fit.t_of_s(s), fit.d2), fd2, atol=1e-5)


def test_usf_local_power_law_index():
    """Diagnostic from notes §3.2: +1.89 / +1.44 / +0.33 at 100 / 707 / 5000 Mpc."""
    pr = load_prior("usf_100_5000")
    fit = fit_prior(pr["distance"], pr["ln_prior"])
    for d, expect in [(100, 1.89), (707, 1.44), (5000, 0.33)]:
        s = -np.log(d)
        alpha = -float(chebval(fit.t_of_s(s), fit.d1)) - 2
        assert abs(alpha - expect) < 0.01
