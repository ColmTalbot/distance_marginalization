import numpy as np
import pytest

from distmarg import _special as sp
from conftest import load_json

ORACLE = load_json("oracle_special.json")


def test_log_ndtr():
    """Check ``scipy.special.log_ndtr`` against the oracle fixture."""
    x = np.array(ORACLE["log_ndtr"]["x"])
    y = np.array(ORACLE["log_ndtr"]["y"])
    got = sp.log_ndtr(x)
    err = np.abs(got - y) / np.maximum(1.0, np.abs(y))
    assert err.max() < 1e-14


def test_log_diff_ndtr():
    """Check :func:`log_diff_ndtr` against the oracle fixture."""
    d = ORACLE["log_diff_ndtr"]
    a, b, y = (np.array(d[k]) for k in ("a", "b", "y"))
    got = sp.log_diff_ndtr(a, b, np)
    # width 1e-8 intervals lose ~8 digits by construction (difference of logs); see notes §6
    wide = (b - a) > 1e-3
    assert np.max(np.abs(got - y)[wide] / np.maximum(1, np.abs(y[wide]))) < 1e-12
    assert np.all(np.isfinite(got))
    assert np.max(np.abs(got - y) / np.maximum(1, np.abs(y))) < 1e-6


def test_bessel():
    """Check ``scipy.special.i0e`` and ``i1e`` against the oracle fixture."""
    x = np.array(ORACLE["ln_i0e"]["x"])
    i0 = sp.i0e(x)
    i1 = sp.i1e(x)
    assert np.max(np.abs(np.log(i0) - ORACLE["ln_i0e"]["y"])) < 1e-13
    ratio = np.array(ORACLE["i1_over_i0"]["y"])
    assert np.max(np.abs(i1 / i0 - ratio) / ratio) < 1e-13


def test_mills_matches_definition():
    """Check :func:`mills` against direct numerical integration."""
    from scipy.integrate import quad

    for a in [0.0, 0.5, 2.9, 3.0, 3.1, 10.0, 1e3]:
        m, u, w = (float(v) for v in sp.mills(np.array(a), np))
        for n, val in enumerate((m, u, w)):
            ref = quad(lambda t: t**n * np.exp(-a * t - t * t / 2), 0, np.inf, epsabs=0, epsrel=1e-13)[0]
            assert abs(val / ref - 1) < 1e-12, (a, n)


def test_ln_i0_series():
    """Check the ``ln I0`` series coefficients against ``log(i0)``."""
    from scipy.special import i0

    c = sp.ln_i0_series_coefficients(10)
    for x in [0.1, 0.5, 1.0]:
        approx = sum(ck * x ** (2 * (k + 1)) for k, ck in enumerate(c))
        assert abs(approx - np.log(i0(x))) < 1e-9
