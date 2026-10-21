import numpy as np
import pytest

from conftest import PRIOR_NAMES, load_json, make_dm

TOL = 1e-4


@pytest.mark.parametrize("name", PRIOR_NAMES)
@pytest.mark.parametrize("phase", [False, True], ids=["nophase", "phase"])
@pytest.mark.parametrize("backend", ["auto", "python"])
def test_against_mpmath_oracle(name, phase, backend):
    """Check ``ln L_D`` against the mpmath oracle fixtures to 1e-4."""
    o = load_json(f"oracle_{name}.json")
    dm = make_dm(name, phase, backend=backend)
    rm, ro, dr = (np.array(o[k]) for k in ("rho_mf", "rho_opt", "d_ref"))
    expect = np.array(o["ln_likelihood_ratio_phase" if phase else "ln_likelihood_ratio"])
    got = dm.ln_likelihood_ratio(rm, ro, dr)
    # absolute tolerance, relaxed by float64 rounding of rho^2/2 for huge rho
    tol = TOL + 1e-14 * np.abs(expect) + 1e-15 * rm**2
    err = np.abs(got - expect)
    assert np.all(err < tol), (err.max(), o["label"][int(np.argmax(err / tol))])


def test_known_values_d2_prior():
    """Spot values from peak_parameterization.md (d^2 prior on [50, 5000], d0 = 1 Gpc)."""
    from distmarg import DistanceMarginalization

    d = np.geomspace(50, 5000, 4096)
    dm = DistanceMarginalization(d, 2 * np.log(d))
    dmp = DistanceMarginalization(d, 2 * np.log(d), True)
    rows = [(4.74, 1333, 8.3704, 6.0867), (13.33, 2250, 85.9787, 82.4843), (2.00, 500, 0.5674, 0.0479)]
    for rho, dstar, lnl, lnlp in rows:
        ropt = rho * dstar / 1000.0
        # the table rounds rho and d*, so compare loosely
        assert abs(float(dm.ln_likelihood_ratio(rho, ropt, 1000.0)) - lnl) < 0.05 * abs(lnl) + 0.02
        assert abs(float(dmp.ln_likelihood_ratio(rho, ropt, 1000.0)) - lnlp) < 0.05 * abs(lnlp) + 0.02
