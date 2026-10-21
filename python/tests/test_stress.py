"""Stress-grid accuracy against the dense reference quadrature (notes §7.2)."""
import numpy as np
import pytest

from distmarg._quadrature import reference_ln_likelihood_ratio
from conftest import make_dm

TARGET = 1e-3


@pytest.mark.parametrize("name", ["usf_100_5000", "usf_10_50000"])
@pytest.mark.parametrize("phase", [False, True], ids=["nophase", "phase"])
@pytest.mark.parametrize("backend", ["auto", "python"])
def test_stress_grid(name, phase, backend):
    """Check the stress grid against the reference quadrature (target 1e-3)."""
    dm = make_dm(name, phase, backend=backend)
    dmin, dmax = dm.prior.d_min, dm.prior.d_max
    rows = []
    for ds in np.geomspace(dmin / 5, dmax * 5, 17):
        for r in np.geomspace(1e-3, 1e4, 22):
            for sgn in ([1] if phase else [1, -1]):
                rows.append((sgn * r, r * ds / 1000.0, 1000.0))
    for ro in np.geomspace(1e-3, 1e3, 13):
        for r in np.geomspace(1e-3, 1e4, 22):
            for sgn in ([1] if phase else [1, -1]):
                rows.append((sgn * r, ro * dmin / 1000.0, 1000.0))
    rows = np.array(rows)
    got = dm.ln_likelihood_ratio(*rows.T)
    ref = np.array([reference_ln_likelihood_ratio(dm.prior, phase, *row) for row in rows])
    err = np.abs(got - ref) - 1e-15 * rows[:, 0] ** 2
    assert np.all(np.isfinite(got))
    assert err.max() < TARGET, rows[np.argmax(err)]
    assert err.max() < 1e-4  # calibrated defaults are ~4e-5 worst case
