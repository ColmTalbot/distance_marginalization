# distmarg: distance-marginalized GW likelihood in the (ρ_mf, v⋆) basis

This repository contains two independent implementations of the same specification:

- **`python/distmarg`**: an array-API implementation (numpy, JAX, torch, cupy) that works with `jit`, `vmap` and `grad`.
- **`rust/distmarg`**: a pure-Rust crate with its own special functions, Householder-QR fit, Gauss–Legendre nodes and adaptive Gauss–Kronrod reference. It has no Python bindings.

The two implementations share no code or tables. They are compared only through the data files in `fixtures/`.

These documents define the specification:

- `IMPLEMENTATION_NOTES.md` is the original specification.
- `SPEC_DECISIONS.md` records every detail the notes left open, and every place where the notes were changed. Read it first.

## Quick start

```python
import numpy as np
from distmarg import DistanceMarginalization

d = np.geomspace(100, 5000, 4096)               # dense, smooth tabulation
dm = DistanceMarginalization(d, ln_prior_of(d), phase_marginalization=True)
# or: DistanceMarginalization.from_bilby_prior(bilby_prior, phase_marginalization=True)

# from Bilby's quantities at the reference distance d_ref:
rho_opt = np.sqrt(optimal_snr_squared)
rho_mf = d_inner_h / rho_opt
ln_l = dm.ln_likelihood_ratio(rho_mf, rho_opt, d_ref)   # ln(L_D / Z_N)
```

```rust
let dm = distmarg::DistanceMarginalization::new(&d, &ln_prior, true, distmarg::Config::default())?;
let ln_l = dm.ln_likelihood_ratio(rho_mf, rho_opt, d_ref);
```

**Using JAX, torch or CuPy.** Set `SCIPY_ARRAY_API=1` before SciPy is imported. With JAX also enable `jax_enable_x64`. At ρ = 10⁴, ln L ≈ 5×10⁷, so float32 has an absolute error of about 4.

**Time marginalization.** Evaluate on arrays over time bins, then take a log-sum-exp over them.

## How it works (summary)

$\ln(\mathcal L_D/\mathcal Z_N)=\ln\int\pi_v(v)\,e^{\ln\ell(v)}\,dv$ with $v=1/d_L$ and $\ln\ell=Av-Bv^2/2$. With phase marginalization, $\ln\ell=\ln I_0(Av)-Bv^2/2$.

The prior is a Chebyshev fit of $\ln\pi_v$ in $s=\ln v$. Each evaluation uses one of three regimes, chosen by a rule that is identical in both implementations:

| regime | when | method | worst error on stress grid |
|---|---|---|---|
| high SNR | \|ρ\| ≥ 30 and relative TN width ≤ 0.01 | truncated-normal mass + Laplace curvature, including the Bessel weight's curvature | 4e-5 |
| prior dominated | κ₂ < 0.05 and range(ln ℓ) ≤ 1 | centred cumulant series to 4th order; for phase, an exact ln I₀ power series | 2e-5 |
| middle | otherwise | 6-panel × 20-node Gauss–Legendre in s, placed around the peak and edge tail, with ln π_v from a 64-knot Hermite grid | 6e-5 |

In the middle regime the 2-D Chebyshev table proposed in the notes is not needed. The fixed-node quadrature is accurate everywhere and table-free; see `SPEC_DECISIONS.md` §6.

## Rust backend for numpy inputs

The Rust crate has optional Python bindings (feature `python`, module `distmarg_rs`, built with pyo3 and rust-numpy). When they are installed, `DistanceMarginalization.ln_likelihood_ratio` sends numpy and plain-Python inputs to Rust:
- **scalars** take about 2 µs (4 µs with phase), against about 0.5 ms on the numpy path;
- **float64 arrays** are passed zero-copy and evaluated with the GIL released, in parallel with rayon for batches of 1024 or more.

JAX, torch, CuPy and float32 inputs always use the pure-Python path. If `distmarg_rs` is not installed, `backend="auto"` falls back to the Python path for everything. Use `backend="python"` to force that path, and `backend="rust"` to require the extension. Both backends are tested against the oracle, and the class pickles cleanly, so it works with multiprocessing.

```bash
cd rust/distmarg && ../../.venv/bin/maturin develop --release --uv
```

The bindings are off by default, so Rust users of the crate never see pyo3 or numpy.

## Layout

```
python/distmarg/      core.py, _prior.py, _special.py, _quadrature.py (reference), bilby.py
python/tests/         pytest suite (oracle, stress grid, continuity, JAX/torch grad, Bilby helper)
rust/distmarg/        src/{lib,prior,chebyshev,linalg,special,quadrature,regimes,python}.rs, src/bin/distmarg_eval.rs
                      tests/ (fixtures, accuracy, proptest), benches/ (criterion)
fixtures/             priors/*.json, oracle_*.json (mpmath), prior_fit_*.json, stress_queries_*.json
tools/                export_prior.py, make_fixtures.py, cross_check.py, benchmark.py
benchmarks/           generated reports
```

## Running things

```bash
uv venv --python 3.14 .venv && uv pip install --python .venv/bin/python -e "python[test]" jax torch bilby
```

```bash
.venv/bin/python -m pytest python/tests
```

```bash
cd rust && cargo test --release
```

```bash
.venv/bin/python tools/cross_check.py
```

```bash
.venv/bin/python tools/make_fixtures.py --all
```

```bash
.venv/bin/python tools/benchmark.py --bilby
```

```bash
cd rust && cargo bench --features parallel
```

What each command does:

- `pytest python/tests`: runs the Python test suite.
- `cargo test --release`: runs the Rust tests. Release mode is needed because the reference quadrature is slow in debug builds.
- `tools/cross_check.py`: compares Python and Rust through files, using the `distmarg-eval` binary.
- `tools/make_fixtures.py --all`: regenerates the fixtures. The mpmath oracle takes several minutes.
- `tools/benchmark.py --bilby`: writes `benchmarks/python_report.md`.
- `cargo bench --features parallel`: runs the Rust criterion benchmarks.

The Rust crate builds with rustc 1.86. `Cargo.lock` pins `proptest` 1.6, because newer versions need rustc 1.88.

## `distmarg-eval` (Rust CLI)

```bash
cargo run --release --features cli --bin distmarg-eval -- \
    --prior fixtures/priors/usf_100_5000.json [--phase] [--config cfg.json] \
    --queries fixtures/stress_queries_usf_100_5000.json --out results.json [--fit-info fit.json]
```

## License

MIT; see `LICENSE`.
