import warnings

import numpy as np
import pytest

from distmarg import REGIME_HIGH_SNR, REGIME_PRIOR_DOMINATED, REGIME_QUADRATURE
from distmarg._quadrature import reference_ln_likelihood_ratio
from conftest import make_dm


def _grid(dm, n=25, phase=False):
    """Return a stress grid over ``d*`` and ``rho_mf``.

    Parameters
    ----------
    dm : DistanceMarginalization
    n : int, optional
        Points per axis.
    phase : bool, optional
        Omit negative ``rho_mf`` when True.

    Returns
    -------
    rho_mf, rho_opt, d_ref : numpy.ndarray
    """
    dmin, dmax = dm.prior.d_min, dm.prior.d_max
    ds = np.geomspace(dmin / 5, dmax * 5, n)
    rh = np.geomspace(1e-3, 1e4, n)
    D, R = np.meshgrid(ds, rh)
    rm = R.ravel()
    if not phase:
        rm = np.concatenate([rm, -rm])
        D = np.concatenate([D.ravel(), D.ravel()])
    ro = np.abs(rm) * np.ravel(D) / 1000.0
    return rm, ro, np.full_like(rm, 1000.0)


def test_d_ref_invariance(prior_name, phase):
    """Check invariance under ``d_ref -> k d_ref``, ``rho_opt -> rho_opt / k``."""
    dm = make_dm(prior_name, phase)
    rm, ro, dr = _grid(dm, 15, phase)
    a = dm.ln_likelihood_ratio(rm, ro, dr)
    for kscale in [0.1, 7.3]:
        b = dm.ln_likelihood_ratio(rm, ro / kscale, dr * kscale)
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-9)


def test_low_snr_limit(prior_name, phase):
    """Check ``ln L_D -> 0`` as ``rho_mf, rho_opt -> 0``."""
    dm = make_dm(prior_name, phase)
    for eps in [1e-3, 1e-5]:
        val = dm.ln_likelihood_ratio(eps, eps, 1000.0)
        assert abs(float(val)) < 10 * eps
    assert dm.regime(1e-4, 1e-4, 1000.0) == REGIME_PRIOR_DOMINATED


def test_high_snr_matches_laplace():
    """Interior, rho -> inf: ln L -> rho^2/2 + ln[sqrt(2 pi) sigma_v pi_v(v*)]."""
    dm = make_dm("usf_100_5000", False)
    rho, dstar = 1e3, 700.0
    ropt = rho * dstar / 1000.0
    sigma = 1.0 / (ropt * 1000.0)
    laplace = rho**2 / 2 + 0.5 * np.log(2 * np.pi) + np.log(sigma) + dm.prior.ln_pi_v(np.log(1 / dstar))
    assert abs(float(dm.ln_likelihood_ratio(rho, ropt, 1000.0)) - laplace) < 1e-4


def test_phase_takes_abs():
    """Check that phase marginalization uses ``|rho_mf|``."""
    dm = make_dm("usf_100_5000", True)
    assert float(dm.ln_likelihood_ratio(-5.0, 3.0, 1000.0)) == float(dm.ln_likelihood_ratio(5.0, 3.0, 1000.0))


def test_negative_rho_outside_support():
    """Negative rho_mf (no phase) is handled exactly by the TN mass; compare to reference."""
    dm = make_dm("usf_100_5000", False)
    for rho in [-2.0, -40.0, -500.0]:
        for ropt in [0.5, 5.0, 50.0]:
            ref = reference_ln_likelihood_ratio(dm.prior, False, rho, ropt, 1000.0)
            assert abs(float(dm.ln_likelihood_ratio(rho, ropt, 1000.0)) - ref) < 1e-4


def test_invalid_inputs_are_nan():
    """Check invalid inputs give NaN and regime -1."""
    dm = make_dm("usf_100_5000", False)
    out = dm.ln_likelihood_ratio(np.array([1.0, 1.0, np.nan, 1.0]), np.array([0.0, -1.0, 1.0, 1.0]),
                                 np.array([1000.0, 1000.0, 1000.0, -5.0]))
    assert np.all(np.isnan(out))
    assert np.all(dm.regime(1.0, 0.0, 1000.0) == -1)


def test_shapes_and_scalars():
    """Check scalars, 0-d arrays, integers and broadcasting."""
    dm = make_dm("usf_100_5000", False)
    s = dm.ln_likelihood_ratio(3.0, 2.0, 1000.0)
    assert np.shape(s) == ()
    z = dm.ln_likelihood_ratio(np.array(3.0), np.array(2.0), np.array(1000.0))
    assert np.shape(z) == ()
    b = dm.ln_likelihood_ratio(np.array([[1.0], [3.0], [50.0]]), np.array([0.5, 2.0, 30.0, 100.0]), 1000.0)
    assert b.shape == (3, 4)
    assert float(b[1, 1]) == pytest.approx(float(s), abs=1e-13)
    i = dm.ln_likelihood_ratio(3, 2, 1000)
    assert float(i) == pytest.approx(float(s), abs=1e-13)


def test_masked_and_where_paths_agree(prior_name, phase):
    """Check the masked numpy path agrees with the where path."""
    dm = make_dm(prior_name, phase)
    rm, ro, dr = _grid(dm, 12, phase)
    a, ra = dm._evaluate(rm, ro, dr, masked=True)
    b, rb = dm._evaluate(rm, ro, dr, masked=False)
    np.testing.assert_array_equal(ra, rb)
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-11)


def test_float32_warns():
    """Check float32 inputs warn."""
    dm = make_dm("usf_100_5000", False)
    with pytest.warns(UserWarning, match="float32"):
        dm.ln_likelihood_ratio(np.float32(3.0), np.float32(2.0), np.float32(1000.0))


def _boundary_scan(dm, phase, n=4000):
    """Lines through parameter space that cross every regime boundary."""
    dmin, dmax = dm.prior.d_min, dm.prior.d_max
    lines = []
    for dstar in [dmin / 3, dmin * 1.5, np.sqrt(dmin * dmax), dmax * 0.7, dmax * 3]:
        rho = np.geomspace(1e-2, 300, n)
        lines.append((rho, rho * dstar / 1000.0))
    for rho0 in [0.3, 3.0, 40.0]:
        ropt = np.geomspace(1e-3, 1e3, n)
        lines.append((np.full(n, rho0), ropt))
    return lines


def test_regime_boundary_continuity(prior_name, phase):
    """Check the jump across every regime switch is below 1e-4."""
    dm = make_dm(prior_name, phase)
    n_switch = 0
    for rho, ropt in _boundary_scan(dm, phase):
        val = dm.ln_likelihood_ratio(rho, ropt, 1000.0)
        reg = dm.regime(rho, ropt, 1000.0)
        sw = np.nonzero(reg[1:] != reg[:-1])[0]
        n_switch += len(sw)
        for i in sw:
            ref_l = dm._branch_quadrature(np, rho[i : i + 2], ropt[i : i + 2] * 1000.0)
            # jump across the switch, after removing the smooth change between the two points
            jump = abs((val[i + 1] - val[i]) - (ref_l[1] - ref_l[0]))
            assert jump < 1e-4, (rho[i], ropt[i], reg[i], reg[i + 1], jump)
    assert n_switch > 0


def test_all_regimes_exercised():
    """Check the stress grid reaches all three regimes."""
    dm = make_dm("usf_100_5000", False)
    rm, ro, dr = _grid(dm, 25)
    reg = dm.regime(rm, ro, dr)
    assert set(np.unique(reg)) == {REGIME_HIGH_SNR, REGIME_PRIOR_DOMINATED, REGIME_QUADRATURE}


def test_distance_posterior_and_sampling():
    """Check the distance posterior is normalized and samples match its mean."""
    dm = make_dm("usf_100_5000", False)
    rho, dstar = 20.0, 800.0
    ropt = rho * dstar / 1000.0
    d, p = dm.distance_posterior(rho, ropt, 1000.0)
    norm = np.trapezoid(p, d) if hasattr(np, "trapezoid") else np.trapz(p, d)
    assert abs(norm - 1) < 1e-3
    assert abs(d[np.argmax(p)] / dstar - 1) < 0.1
    samples = dm.sample_distance(np.full(4000, rho), np.full(4000, ropt), 1000.0, rng=1)
    mean = np.trapezoid(p * d, d) if hasattr(np, "trapezoid") else np.trapz(p * d, d)
    assert abs(samples.mean() / mean - 1) < 0.01
    assert samples.min() >= 100 and samples.max() <= 5000
