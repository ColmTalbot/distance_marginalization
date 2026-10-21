#![allow(dead_code)]
use std::path::PathBuf;

use distmarg::{Config, DistanceMarginalization};
use serde_json::Value;

pub const PRIORS: [&str; 4] = ["usf_100_5000", "usf_10_50000", "pl2_50_5000", "plm1_50_5000"];

pub fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures")
}

pub fn load(name: &str) -> Value {
    let p = fixtures().join(name);
    serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{p:?}: {e}"))).unwrap()
}

pub fn arr(v: &Value, key: &str) -> Vec<f64> {
    v[key].as_array().unwrap_or_else(|| panic!("missing {key}")).iter().map(|x| x.as_f64().unwrap()).collect()
}

pub fn prior(name: &str) -> (Vec<f64>, Vec<f64>) {
    let v = load(&format!("priors/{name}.json"));
    (arr(&v, "distance"), arr(&v, "ln_prior"))
}

pub fn make(name: &str, phase: bool) -> DistanceMarginalization {
    let (d, lp) = prior(name);
    let dm = DistanceMarginalization::new(&d, &lp, phase, Config::default()).unwrap();
    assert!(dm.fit_info().warning.is_none(), "{name}: unexpected fit warning");
    dm
}

/// Stress grid: (rho_mf, rho_opt, d_ref) over d* in [dmin/5, 5 dmax], rho in [1e-3, 1e4]
/// (and negative without phase), plus rho_opt(dmin) in [1e-3, 1e3].
pub fn stress_grid(dm: &DistanceMarginalization, n: usize) -> Vec<(f64, f64, f64)> {
    let pr = dm.prior();
    let geom = |a: f64, b: f64, i: usize| a * (b / a).powf(i as f64 / (n - 1) as f64);
    let signs: &[f64] = if dm.phase_marginalization() { &[1.0] } else { &[1.0, -1.0] };
    let mut out = Vec::new();
    for i in 0..n {
        let ds = geom(pr.d_min / 5.0, pr.d_max * 5.0, i);
        for j in 0..n {
            let r = geom(1e-3, 1e4, j);
            for s in signs {
                out.push((s * r, r * ds / 1000.0, 1000.0));
            }
        }
    }
    for i in 0..n {
        let ro = geom(1e-3, 1e3, i);
        for j in 0..n {
            let r = geom(1e-3, 1e4, j);
            for s in signs {
                out.push((s * r, ro * pr.d_min / 1000.0, 1000.0));
            }
        }
    }
    out
}
