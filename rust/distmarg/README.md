# distmarg (Rust)

Distance-marginalized gravitational-wave likelihood, `ln(L_D / Z_N)`, in the
`(rho_mf, v*)` basis. This is a pure-Rust implementation with no runtime
dependencies beyond `libm` and `thiserror`.

```rust
use distmarg::{Config, DistanceMarginalization};

let d: Vec<f64> = (0..4096).map(|i| 100.0 * 50f64.powf(i as f64 / 4095.0)).collect();
let ln_prior: Vec<f64> = d.iter().map(|x| 2.0 * x.ln()).collect();
let dm = DistanceMarginalization::new(&d, &ln_prior, false, Config::default())?;
let ln_l = dm.ln_likelihood_ratio(12.0, 10.0, 1000.0);
# Ok::<(), distmarg::Error>(())
```

Optional features:

| feature | what it adds |
|---|---|
| `parallel` | rayon-parallel batch evaluation |
| `cli` | the `distmarg-eval` file-based evaluation tool |
| `python` | Python bindings (module `distmarg_rs`), built with maturin |
| `abi3` | stable-ABI Python bindings |

The bindings are published on PyPI as `distmarg-rs` and are used as the backend
of the Python package `distmarg`.

See the [repository](https://github.com/ColmTalbot/distance_marginalization)
for the method and benchmarks.
