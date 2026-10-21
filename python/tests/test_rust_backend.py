"""Rust backend (``distmarg_rs``) against the pure-Python path."""
import numpy as np
import pytest

from conftest import PRIOR_NAMES, load_prior, make_dm

distmarg_rs = pytest.importorskip("distmarg_rs")


@pytest.mark.parametrize("name", PRIOR_NAMES)
@pytest.mark.parametrize("phase", [False, True], ids=["nophase", "phase"])
def test_rust_matches_python(name, phase):
    """Check Rust scalar and array calls agree with the pure-Python path to 1e-9 relative."""
    dm = make_dm(name, phase)
    assert dm.backend == "rust"
    rng = np.random.default_rng(2)
    n = 400
    rho = 10 ** rng.uniform(-3, 4, n) * np.where(rng.uniform(size=n) < 0.5, -1, 1)
    dstar = dm.prior.d_min / 5 * (25 * dm.prior.d_max / dm.prior.d_min) ** rng.uniform(size=n)
    d_ref = 10 ** rng.uniform(1, 4, n)
    rho_opt = np.abs(rho) * dstar / d_ref
    py = dm._evaluate(rho, rho_opt, d_ref)[0]  # pure-Python path
    arr = dm.ln_likelihood_ratio(rho, rho_opt, d_ref)  # Rust array path
    assert type(arr) is np.ndarray and arr.shape == rho.shape
    sca = np.array([dm.ln_likelihood_ratio(float(a), float(b), float(c)) for a, b, c in zip(rho, rho_opt, d_ref)])
    np.testing.assert_allclose(arr, py, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(sca, arr)


def test_rust_array_broadcasting_and_invalid():
    """Check broadcasting, integer and non-contiguous inputs, and NaN for invalid input."""
    dm = make_dm("usf_100_5000", False)
    rho = np.array([[1.0], [3.0], [50.0]])
    ropt = np.array([0.5, 2.0, 30.0, 0.0])[::-1]  # non-contiguous view, includes rho_opt = 0
    out = dm.ln_likelihood_ratio(rho, ropt, 1000)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out[:, 1:], dm._evaluate(rho, ropt, 1000)[0][:, 1:], rtol=1e-9)
    assert np.all(np.isnan(out[:, 0]))
    ints = dm.ln_likelihood_ratio(np.array([3, 8]), np.array([2, 7]), 1000)
    np.testing.assert_allclose(ints, dm._evaluate(np.array([3.0, 8.0]), np.array([2.0, 7.0]), 1000.0)[0], rtol=1e-9)


def test_scalar_types_and_invalid():
    """Check which inputs take the Rust path and that invalid input gives NaN."""
    dm = make_dm("usf_100_5000", False)
    out = dm.ln_likelihood_ratio(3.0, 2.0, 1000.0)
    assert type(out) is np.float64
    assert dm.ln_likelihood_ratio(np.float64(3.0), 2, 1000) == out
    assert type(dm.ln_likelihood_ratio(np.asarray(3.0), np.asarray(2.0), 1000.0)) is np.float64
    assert np.isnan(dm.ln_likelihood_ratio(1.0, 0.0, 1000.0))
    assert np.isnan(dm.ln_likelihood_ratio(float("nan"), 1.0, 1000.0))


def test_backend_selection():
    """Check ``backend`` options, including the pure-Python fallback."""
    from distmarg import DistanceMarginalization

    pr = load_prior("usf_100_5000")
    py = DistanceMarginalization(pr["distance"], pr["ln_prior"], backend="python")
    assert py.backend == "python"
    assert type(py.ln_likelihood_ratio(np.array([8.0]), 7.2, 1000.0)) is np.ndarray
    rs = DistanceMarginalization(pr["distance"], pr["ln_prior"], backend="rust")
    assert abs(float(py.ln_likelihood_ratio(8.0, 7.2, 1000.0)) - float(rs.ln_likelihood_ratio(8.0, 7.2, 1000.0))) < 1e-9
    with pytest.raises(ValueError):
        DistanceMarginalization(pr["distance"], pr["ln_prior"], backend="fortran")


def test_pickle_round_trip():
    """Check the Rust object and the wrapper survive pickling with identical results."""
    import pickle

    dm = make_dm("usf_10_50000", True)
    args = (8.0, 7.2, 1000.0)
    rs = pickle.loads(pickle.dumps(dm._rust))
    assert rs.ln_likelihood_ratio(*args) == dm._rust.ln_likelihood_ratio(*args)
    assert rs.fit_info["degree"] == dm._rust.fit_info["degree"]
    clone = pickle.loads(pickle.dumps(dm))
    assert clone.backend == "rust"
    assert clone.ln_likelihood_ratio(*args) == dm.ln_likelihood_ratio(*args)


def test_usable_in_multiprocessing_pool():
    """Check the wrapper can be sent to worker processes (spawn start method)."""
    import multiprocessing as mp

    dm = make_dm("usf_100_5000", False)
    with mp.get_context("spawn").Pool(2) as pool:
        out = pool.starmap(dm.ln_likelihood_ratio, [(8.0, 7.2, 1000.0), (3.0, 2.0, 1000.0)])
    assert out[0] == dm.ln_likelihood_ratio(8.0, 7.2, 1000.0)


def test_fallback_without_extension(monkeypatch):
    """Check ``backend="auto"`` falls back to Python when ``distmarg_rs`` is missing."""
    import sys

    from distmarg import DistanceMarginalization

    monkeypatch.setitem(sys.modules, "distmarg_rs", None)  # makes the import fail
    pr = load_prior("usf_100_5000")
    dm = DistanceMarginalization(pr["distance"], pr["ln_prior"])
    assert dm.backend == "python"
    ref = make_dm("usf_100_5000", False)
    assert abs(float(dm.ln_likelihood_ratio(8.0, 7.2, 1000.0)) - float(ref.ln_likelihood_ratio(8.0, 7.2, 1000.0))) < 1e-9
    with pytest.raises(ImportError):
        DistanceMarginalization(pr["distance"], pr["ln_prior"], backend="rust")
