import numpy as np
import pytest

from conftest import make_dm

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402


def _points(dm):
    """Return points covering every regime, the prior edges and ``rho_mf = 0``.

    Parameters
    ----------
    dm : DistanceMarginalization

    Returns
    -------
    rho_mf, rho_opt, d_ref : numpy.ndarray
    """
    dmin, dmax = dm.prior.d_min, dm.prior.d_max
    rho = np.array([1e-3, 0.2, 1.0, 3.0, 8.0, -4.0, 25.0, 31.0, 100.0, 2000.0, 0.0])
    dstar = np.array([300.0, dmax * 2, dmin * 0.5, 900.0, dmax, 1200.0, dmin, 2000.0, 4000.0, 150.0, 500.0])
    ropt = np.abs(rho) * dstar / 1000.0
    ropt[-1] = 2.0  # rho_mf = 0 exactly
    return rho, ropt, np.full_like(rho, 1000.0)


def test_jax_matches_numpy(prior_name, phase):
    """Check JAX (where-path) values against numpy (masked path)."""
    dm = make_dm(prior_name, phase)
    rm, ro, dr = _points(dm)
    a = dm.ln_likelihood_ratio(rm, ro, dr)
    b = dm.ln_likelihood_ratio(jnp.asarray(rm), jnp.asarray(ro), jnp.asarray(dr))
    assert isinstance(b, jax.Array)
    np.testing.assert_allclose(np.asarray(b), a, rtol=1e-12, atol=1e-10)


def test_jit_vmap(phase):
    """Check ``jax.jit`` and ``jax.vmap`` give the eager result, including 0-d inputs."""
    dm = make_dm("usf_100_5000", phase)
    rm, ro, dr = (jnp.asarray(x) for x in _points(dm))
    f = jax.jit(dm.ln_likelihood_ratio)
    np.testing.assert_allclose(np.asarray(f(rm, ro, dr)), np.asarray(dm.ln_likelihood_ratio(rm, ro, dr)), rtol=1e-12, atol=1e-10)
    g = jax.vmap(dm.ln_likelihood_ratio, in_axes=(0, 0, None))
    np.testing.assert_allclose(np.asarray(g(rm, ro, 1000.0)), np.asarray(f(rm, ro, dr)), rtol=1e-12, atol=1e-10)
    s = f(jnp.asarray(3.0), jnp.asarray(2.0), jnp.asarray(1000.0))
    assert s.shape == ()


def test_grad_finite_and_matches_fd(prior_name, phase):
    """Check JAX gradients are finite and match central finite differences."""
    dm = make_dm(prior_name, phase)
    rm, ro, dr = _points(dm)
    f = jax.jit(lambda a, b: dm.ln_likelihood_ratio(a, b, 1000.0))
    ga = jax.vmap(jax.grad(f, argnums=(0, 1)))(jnp.asarray(rm), jnp.asarray(ro))
    for g in ga:
        assert np.all(np.isfinite(np.asarray(g)))
    h = 1e-6
    for i, (r, o) in enumerate(zip(rm, ro)):
        fd_r = (dm.ln_likelihood_ratio(r + h, o, 1000.0) - dm.ln_likelihood_ratio(r - h, o, 1000.0)) / (2 * h)
        fd_o = (dm.ln_likelihood_ratio(r, o * (1 + h), 1000.0) - dm.ln_likelihood_ratio(r, o * (1 - h), 1000.0)) / (2 * h * o)
        if phase and r == 0:
            continue  # |rho| has a kink at 0
        reg = dm.regime(np.array([r - h, r + h]), o, 1000.0)
        if reg[0] != reg[1]:
            continue  # straddles a regime switch
        scale = max(1.0, abs(fd_r), abs(fd_o))
        assert abs(float(ga[0][i]) - fd_r) < 1e-4 * scale, (i, r, o)
        assert abs(float(ga[1][i]) - fd_o) < 1e-4 * scale, (i, r, o)


def test_grad_at_edge_and_boundaries():
    """Check gradients stay finite with ``v*`` on a prior edge, at ``rho_high`` and at ``rho_mf = 0``."""
    dm = make_dm("usf_100_5000", False)
    f = jax.grad(lambda a, b: dm.ln_likelihood_ratio(a, b, 1000.0), argnums=(0, 1))
    vmax = dm.v_max
    # v* exactly on the upper prior edge, at several SNRs
    for rho in [0.5, 5.0, 29.999, 30.0, 300.0]:
        ropt = rho / (vmax * 1000.0)
        g = f(jnp.asarray(rho), jnp.asarray(ropt))
        assert all(np.isfinite(np.asarray(x)) for x in g)
    g = f(jnp.asarray(0.0), jnp.asarray(1.0))
    assert all(np.isfinite(np.asarray(x)) for x in g)


def test_torch_if_available():
    """Check torch values match numpy and autograd gradients are finite."""
    torch = pytest.importorskip("torch")
    dm = make_dm("usf_100_5000", True)
    rm, ro, dr = _points(dm)
    a = dm.ln_likelihood_ratio(rm, ro, dr)
    rt = torch.tensor(rm, dtype=torch.float64, requires_grad=True)
    out = dm.ln_likelihood_ratio(rt, torch.tensor(ro, dtype=torch.float64), torch.tensor(dr, dtype=torch.float64))
    np.testing.assert_allclose(out.detach().numpy(), a, rtol=1e-12, atol=1e-10)
    out.sum().backward()
    assert torch.all(torch.isfinite(rt.grad))
