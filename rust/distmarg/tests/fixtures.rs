//! Special functions, prior fits and ln L_D against the shared data fixtures.
mod common;
use common::*;
use distmarg::special::{i0e, i1e, log_diff_ndtr, log_ndtr};
use distmarg::{Config, DistanceMarginalization};

#[test]
fn special_log_ndtr() {
    let o = load("oracle_special.json");
    let x = arr(&o["log_ndtr"], "x");
    let y = arr(&o["log_ndtr"], "y");
    for (xi, yi) in x.iter().zip(&y) {
        let got = log_ndtr(*xi);
        assert!((got - yi).abs() <= 1e-14 * yi.abs().max(1.0) + 1e-300, "x={xi}: {got} vs {yi}");
    }
}

#[test]
fn special_log_diff_ndtr() {
    let o = load("oracle_special.json");
    let a = arr(&o["log_diff_ndtr"], "a");
    let b = arr(&o["log_diff_ndtr"], "b");
    let y = arr(&o["log_diff_ndtr"], "y");
    for i in 0..a.len() {
        let got = log_diff_ndtr(a[i], b[i]);
        assert!(got.is_finite());
        // a difference of logs loses ~log10(1/width) digits for very narrow intervals
        let tol = if b[i] - a[i] > 1e-3 { 1e-12 } else { 1e-6 };
        assert!((got - y[i]).abs() <= tol * y[i].abs().max(1.0), "({}, {}): {got} vs {}", a[i], b[i], y[i]);
    }
}

#[test]
fn special_bessel() {
    let o = load("oracle_special.json");
    let x = arr(&o["ln_i0e"], "x");
    let ln_i0e = arr(&o["ln_i0e"], "y");
    let ratio = arr(&o["i1_over_i0"], "y");
    for i in 0..x.len() {
        let i0 = i0e(x[i]);
        assert!((i0.ln() - ln_i0e[i]).abs() < 1e-13, "x={}: {} vs {}", x[i], i0.ln(), ln_i0e[i]);
        assert!(((i1e(x[i]) / i0) / ratio[i] - 1.0).abs() < 1e-13, "x={}", x[i]);
    }
}

#[test]
fn prior_fits_match_python() {
    for name in PRIORS {
        let dm = make(name, false);
        let fi = dm.fit_info();
        let r = load(&format!("prior_fit_{name}.json"));
        assert_eq!(fi.degree as u64, r["degree"].as_u64().unwrap(), "{name}");
        let c = arr(&r, "coeffs");
        let cmax = c.iter().fold(0.0f64, |m, x| m.max(x.abs()));
        for (a, b) in fi.coeffs.iter().zip(&c) {
            assert!((a - b).abs() <= 1e-12 * cmax, "{name} coeff {a} vs {b}");
        }
        let ln_norm = r["ln_norm"].as_f64().unwrap();
        assert!((fi.ln_norm - ln_norm).abs() <= 1e-12 * ln_norm.abs().max(1.0), "{name} ln_norm");
        for (a, b) in fi.moments.iter().zip(arr(&r, "moments")) {
            assert!((a / b - 1.0).abs() <= 1e-12, "{name} moment {a} vs {b}");
        }
        assert!((fi.moments[0] - 1.0).abs() < 1e-13);
    }
}

#[test]
fn bad_priors_warn() {
    for name in ["kinked_100_5000", "noisy_50_5000"] {
        let (d, lp) = prior(name);
        let dm = DistanceMarginalization::new(&d, &lp, false, Config::default()).unwrap();
        assert!(dm.fit_info().warning.is_some(), "{name} should warn");
    }
}

#[test]
fn invalid_priors_rejected() {
    let d: Vec<f64> = (1..=100).map(|i| 10.0 * i as f64).collect();
    let lp: Vec<f64> = d.iter().map(|x| 2.0 * x.ln()).collect();
    let cfg = Config::default;
    let rev: Vec<f64> = d.iter().rev().cloned().collect();
    assert!(DistanceMarginalization::new(&rev, &lp, false, cfg()).is_err());
    let mut bad = lp.clone();
    bad[3] = f64::NEG_INFINITY;
    assert!(DistanceMarginalization::new(&d, &bad, false, cfg()).is_err());
    assert!(DistanceMarginalization::new(&d, &lp[..99], false, cfg()).is_err());
    let mut z = d.clone();
    z[0] = 0.0;
    assert!(DistanceMarginalization::new(&z, &lp, false, cfg()).is_err());
    assert!(DistanceMarginalization::new(&d, &lp, false, Config { fit_degree: Some(60), ..cfg() }).is_err());
}

#[test]
fn oracle_ln_likelihood() {
    for name in PRIORS {
        let o = load(&format!("oracle_{name}.json"));
        let rm = arr(&o, "rho_mf");
        let ro = arr(&o, "rho_opt");
        let dr = arr(&o, "d_ref");
        for (phase, key) in [(false, "ln_likelihood_ratio"), (true, "ln_likelihood_ratio_phase")] {
            let dm = make(name, phase);
            let expect = arr(&o, key);
            for i in 0..rm.len() {
                let got = dm.ln_likelihood_ratio(rm[i], ro[i], dr[i]);
                let tol = 1e-4 + 1e-14 * expect[i].abs() + 1e-15 * rm[i] * rm[i];
                assert!(
                    (got - expect[i]).abs() < tol,
                    "{name} phase={phase} {:?}: {got} vs {} ({:?})",
                    (rm[i], ro[i], dr[i]),
                    expect[i],
                    dm.regime(rm[i], ro[i], dr[i])
                );
            }
        }
    }
}
