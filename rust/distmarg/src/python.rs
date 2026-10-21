//! Python bindings (feature `python`, module `distmarg_rs`), built with maturin.
//!
//! These exist for one purpose: a low-overhead scalar path for the Python package,
//! which otherwise pays ~0.5 ms of numpy dispatch per single-point call.  The Rust
//! API in the crate root is unchanged and does not depend on this module.

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};

use crate::{Config, DistanceMarginalization};

/// Rust implementation of the distance-marginalized likelihood ratio.
///
/// Parameters
/// ----------
/// distance, ln_prior : sequence of float
///     Prior tabulation (see ``distmarg.DistanceMarginalization``).
/// phase_marginalization : bool, optional
/// **config
///     Any ``Config`` field: ``fit_degree``, ``fit_tol``, ``rho_high``, ``kappa2_low``,
///     ``cumulant_range_max``, ``cumulant_order``, ``quad_nodes``, ``quad_grid``,
///     ``high_max_rel_width``, ``phase_series_xmax``.
#[pyclass(name = "DistanceMarginalization", module = "distmarg_rs", frozen)]
struct PyDistanceMarginalization {
    inner: DistanceMarginalization,
    // inputs, kept so the object can be pickled (rebuilt on unpickling)
    distance: Vec<f64>,
    ln_prior: Vec<f64>,
}

fn config_from_kwargs(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Config> {
    let mut cfg = Config::default();
    if let Some(kw) = kwargs {
        for (key, value) in kw.iter() {
            let key: String = key.extract()?;
            match key.as_str() {
                "fit_degree" => cfg.fit_degree = value.extract::<Option<usize>>()?,
                "fit_tol" => cfg.fit_tol = value.extract()?,
                "rho_high" => cfg.rho_high = value.extract()?,
                "kappa2_low" => cfg.kappa2_low = value.extract()?,
                "cumulant_range_max" => cfg.cumulant_range_max = value.extract()?,
                "cumulant_order" => cfg.cumulant_order = value.extract()?,
                "quad_nodes" => cfg.quad_nodes = value.extract()?,
                "quad_grid" => cfg.quad_grid = value.extract()?,
                "high_max_rel_width" => cfg.high_max_rel_width = value.extract()?,
                "phase_series_xmax" => cfg.phase_series_xmax = value.extract()?,
                other => return Err(PyValueError::new_err(format!("unknown config option {other:?}"))),
            }
        }
    }
    Ok(cfg)
}

#[pymethods]
impl PyDistanceMarginalization {
    #[new]
    #[pyo3(signature = (distance, ln_prior, phase_marginalization = false, **config))]
    fn new(
        distance: Vec<f64>,
        ln_prior: Vec<f64>,
        phase_marginalization: bool,
        config: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Self> {
        let cfg = config_from_kwargs(config)?;
        let inner = DistanceMarginalization::new(&distance, &ln_prior, phase_marginalization, cfg)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyDistanceMarginalization { inner, distance, ln_prior })
    }

    /// Pickle support: rebuild from the prior arrays, phase flag and configuration.
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyTuple>)> {
        let c = self.inner.config();
        let cfg = PyDict::new(py);
        cfg.set_item("fit_degree", c.fit_degree)?;
        cfg.set_item("fit_tol", c.fit_tol)?;
        cfg.set_item("rho_high", c.rho_high)?;
        cfg.set_item("kappa2_low", c.kappa2_low)?;
        cfg.set_item("cumulant_range_max", c.cumulant_range_max)?;
        cfg.set_item("cumulant_order", c.cumulant_order)?;
        cfg.set_item("quad_nodes", c.quad_nodes)?;
        cfg.set_item("quad_grid", c.quad_grid)?;
        cfg.set_item("high_max_rel_width", c.high_max_rel_width)?;
        cfg.set_item("phase_series_xmax", c.phase_series_xmax)?;
        let rebuild = py.import("distmarg_rs")?.getattr("_rebuild")?;
        let args = PyTuple::new(
            py,
            [
                self.distance.clone().into_pyobject(py)?.into_any(),
                self.ln_prior.clone().into_pyobject(py)?.into_any(),
                pyo3::types::PyBool::new(py, self.inner.phase_marginalization()).to_owned().into_any(),
                cfg.into_any(),
            ],
        )?;
        Ok((rebuild, args))
    }

    /// ln(L_D / Z_N) for scalar inputs (NaN for invalid input).
    fn ln_likelihood_ratio(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> f64 {
        self.inner.ln_likelihood_ratio(rho_mf, rho_opt, d_ref)
    }

    /// Element-wise ln(L_D / Z_N) for equal-length contiguous float64 numpy arrays.
    ///
    /// Zero-copy; the GIL is released while evaluating (rayon-parallel when built
    /// with the ``parallel`` feature).
    fn ln_likelihood_ratio_array<'py>(
        &self,
        py: Python<'py>,
        rho_mf: PyReadonlyArray1<'py, f64>,
        rho_opt: PyReadonlyArray1<'py, f64>,
        d_ref: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let (a, b, c) = (rho_mf.as_slice()?, rho_opt.as_slice()?, d_ref.as_slice()?);
        let mut out = vec![0.0; a.len()];
        py.allow_threads(|| self.inner.ln_likelihood_ratio_batch(a, b, c, &mut out))
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out.into_pyarray(py))
    }

    /// Element-wise ln(L_D / Z_N) for equal-length sequences.
    fn ln_likelihood_ratio_batch(&self, rho_mf: Vec<f64>, rho_opt: Vec<f64>, d_ref: Vec<f64>) -> PyResult<Vec<f64>> {
        let mut out = vec![0.0; rho_mf.len()];
        self.inner
            .ln_likelihood_ratio_batch(&rho_mf, &rho_opt, &d_ref, &mut out)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(out)
    }

    /// Regime code: 0 high SNR, 1 prior dominated, 2 quadrature, -1 invalid.
    fn regime(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> i32 {
        self.inner.regime(rho_mf, rho_opt, d_ref).code()
    }

    /// Fit summary: degree, max_residual, coeffs, ln_norm, moments, warning.
    #[getter]
    fn fit_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let fi = self.inner.fit_info();
        let d = PyDict::new(py);
        d.set_item("degree", fi.degree)?;
        d.set_item("max_residual", fi.max_residual)?;
        d.set_item("coeffs", fi.coeffs.clone())?;
        d.set_item("ln_norm", fi.ln_norm)?;
        d.set_item("moments", fi.moments.clone())?;
        d.set_item("warning", fi.warning.clone())?;
        Ok(d)
    }
}

/// Unpickling helper: ``DistanceMarginalization(distance, ln_prior, phase, **config)``.
#[pyfunction]
fn _rebuild(
    distance: Vec<f64>,
    ln_prior: Vec<f64>,
    phase_marginalization: bool,
    config: &Bound<'_, PyDict>,
) -> PyResult<PyDistanceMarginalization> {
    PyDistanceMarginalization::new(distance, ln_prior, phase_marginalization, Some(config))
}

#[pymodule]
fn distmarg_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDistanceMarginalization>()?;
    m.add_function(wrap_pyfunction!(_rebuild, m)?)?;
    Ok(())
}
