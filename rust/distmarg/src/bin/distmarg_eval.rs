//! File-based evaluation tool for cross-checks (no Python involved).
//!
//! ```text
//! distmarg-eval --prior PRIOR.json [--phase] [--config CFG.json] \
//!               --queries QUERIES.json --out RESULTS.json [--fit-info FIT.json]
//! ```
//!
//! - prior:   `{"distance": [...], "ln_prior": [...]}`
//! - queries: `{"rho_mf": [...], "rho_opt": [...], "d_ref": [...]}`
//! - config:  any subset of the `Config` field names
//! - out:     `{"ln_likelihood_ratio": [...], "regime": [...]}`; NaN is written as null
//!   and floats use their shortest round-trip representation

use std::process::ExitCode;

use distmarg::{Config, DistanceMarginalization};
use serde_json::{json, Value};

fn usage() -> ExitCode {
    eprintln!(
        "usage: distmarg-eval --prior PRIOR.json [--phase] [--config CFG.json] \
         [--queries QUERIES.json --out RESULTS.json] [--fit-info FIT.json]"
    );
    ExitCode::from(2)
}

fn read_json(path: &str) -> Result<Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))
}

fn f64_array(v: &Value, key: &str) -> Result<Vec<f64>, String> {
    v.get(key)
        .and_then(Value::as_array)
        .ok_or_else(|| format!("missing array {key:?}"))?
        .iter()
        .map(|x| x.as_f64().ok_or_else(|| format!("non-numeric entry in {key:?}")))
        .collect()
}

fn float_json(x: f64) -> Value {
    if x.is_finite() {
        json!(x)
    } else {
        Value::Null
    }
}

fn parse_config(v: &Value) -> Result<Config, String> {
    let mut cfg = Config::default();
    let obj = v.as_object().ok_or("config must be a JSON object")?;
    for (k, val) in obj {
        let num = || val.as_f64().ok_or_else(|| format!("config {k:?} must be a number"));
        match k.as_str() {
            "fit_degree" => cfg.fit_degree = if val.is_null() { None } else { Some(num()? as usize) },
            "fit_tol" => cfg.fit_tol = num()?,
            "rho_high" => cfg.rho_high = num()?,
            "kappa2_low" => cfg.kappa2_low = num()?,
            "cumulant_range_max" => cfg.cumulant_range_max = num()?,
            "cumulant_order" => cfg.cumulant_order = num()? as usize,
            "quad_nodes" => cfg.quad_nodes = num()? as usize,
            "quad_grid" => cfg.quad_grid = num()? as usize,
            "high_max_rel_width" => cfg.high_max_rel_width = num()?,
            "phase_series_xmax" => cfg.phase_series_xmax = num()?,
            other => return Err(format!("unknown config key {other:?}")),
        }
    }
    Ok(cfg)
}

fn run(args: &[String]) -> Result<(), String> {
    let mut prior = None;
    let mut phase = false;
    let mut config = None;
    let mut queries = None;
    let mut out = None;
    let mut fit_info = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || it.next().cloned().ok_or_else(|| format!("{a} needs a value"));
        match a.as_str() {
            "--prior" => prior = Some(val()?),
            "--phase" => phase = true,
            "--config" => config = Some(val()?),
            "--queries" => queries = Some(val()?),
            "--out" => out = Some(val()?),
            "--fit-info" => fit_info = Some(val()?),
            other => return Err(format!("unknown argument {other}")),
        }
    }
    let prior = read_json(&prior.ok_or("--prior is required")?)?;
    let cfg = match config {
        Some(p) => parse_config(&read_json(&p)?)?,
        None => Config::default(),
    };
    let dm = DistanceMarginalization::new(&f64_array(&prior, "distance")?, &f64_array(&prior, "ln_prior")?, phase, cfg)
        .map_err(|e| e.to_string())?;
    if let Some(w) = &dm.fit_info().warning {
        eprintln!("warning: {w}");
    }
    if let Some(path) = fit_info {
        let fi = dm.fit_info();
        let v = json!({
            "degree": fi.degree,
            "max_residual": fi.max_residual,
            "coeffs": fi.coeffs,
            "ln_norm": fi.ln_norm,
            "moments": fi.moments,
            "warning": fi.warning,
        });
        std::fs::write(&path, serde_json::to_string(&v).unwrap()).map_err(|e| format!("{path}: {e}"))?;
    }
    match (queries, out) {
        (Some(q), Some(o)) => {
            let q = read_json(&q)?;
            let rm = f64_array(&q, "rho_mf")?;
            let ro = f64_array(&q, "rho_opt")?;
            let dr = f64_array(&q, "d_ref")?;
            let mut res = vec![0.0; rm.len()];
            dm.ln_likelihood_ratio_batch(&rm, &ro, &dr, &mut res).map_err(|e| e.to_string())?;
            let regime: Vec<i32> = (0..rm.len()).map(|i| dm.regime(rm[i], ro[i], dr[i]).code()).collect();
            let v = json!({
                "ln_likelihood_ratio": res.iter().map(|x| float_json(*x)).collect::<Vec<_>>(),
                "regime": regime,
            });
            std::fs::write(&o, serde_json::to_string(&v).unwrap()).map_err(|e| format!("{o}: {e}"))?;
        }
        (None, None) => {}
        _ => return Err("--queries and --out must be given together".into()),
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        return usage();
    }
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
