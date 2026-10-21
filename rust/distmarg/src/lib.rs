//! Distance-marginalized gravitational-wave likelihood ratio ln(L_D / Z_N) in the
//! (rho_mf, v*) basis.
//!
//! This is an independent implementation of `IMPLEMENTATION_NOTES.md` §2-§4 with
//! the decisions pinned down in `SPEC_DECISIONS.md`.  It shares no code or tables
//! with the Python package; the two are compared only through data files.
//!
//! Inputs to [`DistanceMarginalization::ln_likelihood_ratio`]:
//!
//! - `rho_opt` = sqrt(<h|h>) at `d_ref` (> 0);
//! - `rho_mf` = Re<d|h>/rho_opt (may be negative) without phase marginalization, or
//!   |<d|h>|/rho_opt with it (the absolute value is taken);
//! - `d_ref` the reference distance at which both were computed (> 0).
//!
//! The result excludes the noise log-likelihood ln Z_N.  It is NaN if any input is
//! NaN/infinite or if `rho_opt <= 0` or `d_ref <= 0`.

pub mod chebyshev;
pub mod linalg;
pub mod prior;
pub mod quadrature;
mod regimes;
pub mod special;
#[cfg(feature = "python")]
mod python;

pub use prior::PriorFit;

/// Tunable thresholds; defaults match the Python implementation.
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    /// Chebyshev degree for ln pi_v(s); `None` selects from 8, 16, ..., 128.
    pub fit_degree: Option<usize>,
    /// Fit residual at or above which [`FitInfo::warning`] is set.
    pub fit_tol: f64,
    /// |rho_mf| threshold of the high-SNR branch.
    pub rho_high: f64,
    /// kappa_2 threshold of the prior-dominated branch.
    pub kappa2_low: f64,
    /// Maximum range of ln ell over the support for the prior-dominated branch.
    pub cumulant_range_max: f64,
    /// Cumulant series order (1-4).
    pub cumulant_order: usize,
    /// Gauss–Legendre nodes per panel of the six-panel middle-regime quadrature.
    pub quad_nodes: usize,
    /// Knots of the uniform grid in s used for cubic-Hermite interpolation of ln pi_v
    /// inside the quadrature; 0 evaluates the Chebyshev series directly.
    pub quad_grid: usize,
    /// High-SNR branch also requires sqrt(Var_TN) <= high_max_rel_width * vbar.
    pub high_max_rel_width: f64,
    /// Phase case: cumulant branch requires |<d|h>| d_ref / d_min <= phase_series_xmax.
    pub phase_series_xmax: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            fit_degree: None,
            fit_tol: 1e-8,
            rho_high: 30.0,
            kappa2_low: 0.05,
            cumulant_range_max: 1.0,
            cumulant_order: 4,
            quad_nodes: 20,
            quad_grid: 64,
            high_max_rel_width: 0.01,
            phase_series_xmax: 1.0,
        }
    }
}

/// Evaluation regime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Regime {
    HighSnr = 0,
    PriorDominated = 1,
    Quadrature = 2,
    /// Invalid input (NaN, rho_opt <= 0 or d_ref <= 0).
    Invalid = -1,
}

impl Regime {
    pub fn code(self) -> i32 {
        self as i32
    }
    pub fn name(self) -> &'static str {
        match self {
            Regime::HighSnr => "high_snr",
            Regime::PriorDominated => "prior_dominated",
            Regime::Quadrature => "quadrature",
            Regime::Invalid => "invalid",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("invalid prior: {0}")]
    InvalidPrior(String),
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("input slices must have equal length")]
    LengthMismatch,
}

/// Summary of the prior fit (IMPLEMENTATION_NOTES §7.8).
#[derive(Debug, Clone)]
pub struct FitInfo {
    pub degree: usize,
    pub max_residual: f64,
    /// Least-squares Chebyshev coefficients of the unnormalized ln pi_v(t).
    pub coeffs: Vec<f64>,
    /// ln of the normalization constant subtracted from `coeffs[0]`.
    pub ln_norm: f64,
    /// E_pi[v^k], k = 0..=8.
    pub moments: Vec<f64>,
    /// Set when the fit residual reached `fit_tol`.
    pub warning: Option<String>,
}

/// Immutable after construction; `Send + Sync`.
#[derive(Debug, Clone)]
pub struct DistanceMarginalization {
    pub(crate) prior: PriorFit,
    pub(crate) phase: bool,
    pub(crate) cfg: Config,
    pub(crate) fit_info: FitInfo,
    pub(crate) gl_x: Vec<f64>,
    pub(crate) gl_lnw: Vec<f64>,
    pub(crate) ln_i0_c: Vec<f64>,
    pub(crate) zbar: f64,
    pub(crate) central: Vec<f64>,
    pub(crate) z0: f64,
    /// Hermite grid: spacing, ln pi_v and d ln pi_v / ds at the knots.
    pub(crate) grid_h: f64,
    pub(crate) grid_g: Vec<f64>,
    pub(crate) grid_dg: Vec<f64>,
}

impl DistanceMarginalization {
    /// Build from tabulated `distance` (finite, > 0, strictly increasing) and
    /// `ln_prior` (finite, need not be normalized).
    pub fn new(distance: &[f64], ln_prior: &[f64], phase_marginalization: bool, cfg: Config) -> Result<Self, Error> {
        if !(1..=4).contains(&cfg.cumulant_order) {
            return Err(Error::InvalidConfig("cumulant_order must be between 1 and 4".into()));
        }
        if cfg.quad_nodes < 2 {
            return Err(Error::InvalidConfig("quad_nodes must be >= 2".into()));
        }
        if cfg.quad_grid == 1 {
            return Err(Error::InvalidConfig("quad_grid must be 0 or >= 2".into()));
        }
        let (prior, warning) = PriorFit::new(distance, ln_prior, cfg.fit_degree, cfg.fit_tol)?;
        let (gl_x, gl_w) = quadrature::gauss_legendre(cfg.quad_nodes);
        let fit_info = FitInfo {
            degree: prior.degree,
            max_residual: prior.max_residual,
            coeffs: prior.raw_coeffs.clone(),
            ln_norm: prior.ln_norm,
            moments: prior.moments.clone(),
            warning,
        };
        let nu0 = prior.v_min / prior.v_max;
        let (zbar, central, z0) = if phase_marginalization {
            (prior.zbar_nu2, prior.central_nu2.clone(), nu0 * nu0)
        } else {
            (prior.zbar_nu, prior.central_nu.clone(), nu0)
        };
        let n = cfg.quad_grid;
        let (grid_h, grid_g, grid_dg) = if n >= 2 {
            let h = (prior.s_b - prior.s_a) / (n - 1) as f64;
            let s_at = |j: usize| if j == n - 1 { prior.s_b } else { prior.s_a + j as f64 * h };
            let g = (0..n).map(|j| chebyshev::clenshaw(&prior.coeffs, prior.t_of_s(s_at(j)))).collect();
            let dg = (0..n).map(|j| chebyshev::clenshaw(&prior.d1, prior.t_of_s(s_at(j)))).collect();
            (h, g, dg)
        } else {
            (0.0, Vec::new(), Vec::new())
        };
        Ok(DistanceMarginalization {
            grid_h,
            grid_g,
            grid_dg,
            prior,
            phase: phase_marginalization,
            cfg,
            fit_info,
            gl_x,
            gl_lnw: gl_w.iter().map(|w| w.ln()).collect(),
            ln_i0_c: special::ln_i0_series(regimes::MAX_DEG),
            zbar,
            central,
            z0,
        })
    }

    pub fn fit_info(&self) -> &FitInfo {
        &self.fit_info
    }

    pub fn prior(&self) -> &PriorFit {
        &self.prior
    }

    pub fn config(&self) -> &Config {
        &self.cfg
    }

    pub fn phase_marginalization(&self) -> bool {
        self.phase
    }

    /// (rho, k = rho_opt d_ref), or None for invalid input.
    #[inline]
    fn prepare(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> Option<(f64, f64)> {
        if !(rho_mf.is_finite() && rho_opt.is_finite() && d_ref.is_finite()) || rho_opt <= 0.0 || d_ref <= 0.0 {
            return None;
        }
        let rm = if self.phase { rho_mf.abs() } else { rho_mf };
        Some((rm, rho_opt * d_ref))
    }

    fn select(&self, rm: f64, k: f64) -> Regime {
        let vmax = self.prior.v_max;
        let a_p = rm * k * vmax;
        let b_p = (k * vmax) * (k * vmax);
        let (p, deg, ok) = if self.phase {
            let (p, d) = regimes::polynomial(self, a_p.min(self.cfg.phase_series_xmax), b_p);
            (p, d, a_p <= self.cfg.phase_series_xmax)
        } else {
            let (p, d) = regimes::polynomial(self, a_p, b_p);
            (p, d, true)
        };
        if ok
            && regimes::kappa2(self, &p, deg) < self.cfg.kappa2_low
            && regimes::ln_ell_range(self, &p, deg) <= self.cfg.cumulant_range_max
        {
            return Regime::PriorDominated;
        }
        if rm.abs() >= self.cfg.rho_high {
            let sigma = 1.0 / k;
            let tn = regimes::truncnorm(rm * sigma, sigma, self.prior.v_min, vmax);
            if tn.var.sqrt() <= self.cfg.high_max_rel_width * tn.vbar {
                return Regime::HighSnr;
            }
        }
        Regime::Quadrature
    }

    /// Regime that [`Self::ln_likelihood_ratio`] uses for these inputs.
    pub fn regime(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> Regime {
        match self.prepare(rho_mf, rho_opt, d_ref) {
            Some((rm, k)) => self.select(rm, k),
            None => Regime::Invalid,
        }
    }

    /// ln(L_D / Z_N).  No allocation; NaN for invalid input.
    pub fn ln_likelihood_ratio(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> f64 {
        match self.prepare(rho_mf, rho_opt, d_ref) {
            Some((rm, k)) => match self.select(rm, k) {
                Regime::PriorDominated => regimes::branch_cumulant(self, rm, k),
                Regime::HighSnr => regimes::branch_high(self, rm, k),
                _ => regimes::branch_quadrature(self, rm, k),
            },
            None => f64::NAN,
        }
    }

    /// Evaluate a specific branch regardless of the selection rule (tests, diagnostics).
    pub fn ln_likelihood_ratio_in(&self, regime: Regime, rho_mf: f64, rho_opt: f64, d_ref: f64) -> f64 {
        match self.prepare(rho_mf, rho_opt, d_ref) {
            Some((rm, k)) => match regime {
                Regime::PriorDominated => regimes::branch_cumulant(self, rm, k),
                Regime::HighSnr => regimes::branch_high(self, rm, k),
                Regime::Quadrature => regimes::branch_quadrature(self, rm, k),
                Regime::Invalid => f64::NAN,
            },
            None => f64::NAN,
        }
    }

    /// Element-wise evaluation into `out` (parallel with the `parallel` feature).
    pub fn ln_likelihood_ratio_batch(
        &self,
        rho_mf: &[f64],
        rho_opt: &[f64],
        d_ref: &[f64],
        out: &mut [f64],
    ) -> Result<(), Error> {
        let n = out.len();
        if rho_mf.len() != n || rho_opt.len() != n || d_ref.len() != n {
            return Err(Error::LengthMismatch);
        }
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            out.par_iter_mut()
                .with_min_len(1024)
                .enumerate()
                .for_each(|(i, o)| *o = self.ln_likelihood_ratio(rho_mf[i], rho_opt[i], d_ref[i]));
        }
        #[cfg(not(feature = "parallel"))]
        for i in 0..n {
            out[i] = self.ln_likelihood_ratio(rho_mf[i], rho_opt[i], d_ref[i]);
        }
        Ok(())
    }

    /// Serial batch evaluation, regardless of features (benchmarks).
    pub fn ln_likelihood_ratio_batch_serial(&self, rho_mf: &[f64], rho_opt: &[f64], d_ref: &[f64], out: &mut [f64]) -> Result<(), Error> {
        let n = out.len();
        if rho_mf.len() != n || rho_opt.len() != n || d_ref.len() != n {
            return Err(Error::LengthMismatch);
        }
        for i in 0..n {
            out[i] = self.ln_likelihood_ratio(rho_mf[i], rho_opt[i], d_ref[i]);
        }
        Ok(())
    }

    /// Reference value by adaptive Gauss–Kronrod quadrature with explicit break
    /// points (IMPLEMENTATION_NOTES §5), using the same fitted prior.  Slow; for tests.
    pub fn reference_ln_likelihood_ratio(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> f64 {
        let (rm, k) = match self.prepare(rho_mf, rho_opt, d_ref) {
            Some(x) => x,
            None => return f64::NAN,
        };
        let pr = &self.prior;
        let (vmin, vmax) = (pr.v_min, pr.v_max);
        let sigma = 1.0 / k;
        let vstar = rm * sigma;
        let amp = rm * k;
        let bb = k * k;
        let phase = self.phase;
        let ln_f = move |s: f64| {
            let v = s.exp();
            let mut f = pr.ln_pi_v(s) + s + amp * v - 0.5 * bb * v * v;
            if phase {
                f += special::i0e(amp * v).ln();
            }
            f
        };
        let centre = vstar.clamp(vmin, vmax);
        let mut pts: Vec<f64> = Vec::new();
        for i in 0..=64 {
            pts.push(pr.s_a + (pr.s_b - pr.s_a) * i as f64 / 64.0);
        }
        let mut add_v = |v: f64| {
            if v > vmin && v < vmax {
                pts.push(v.ln());
            }
        };
        add_v(centre);
        for m in [0.3, 1.0, 2.0, 3.5, 5.0, 7.0, 10.0, 14.0, 20.0, 30.0, 45.0, 70.0] {
            add_v(centre + m * sigma);
            add_v(centre - m * sigma);
        }
        let dist = (vstar - centre).abs();
        if dist > 0.0 {
            let lam = sigma * sigma / dist;
            let sgn = if centre == vmin { 1.0 } else { -1.0 };
            for m in [0.01, 0.1, 0.3, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0] {
                add_v(centre + sgn * m * lam);
            }
        }
        pts.push(pr.s_a);
        pts.push(pr.s_b);
        pts.sort_by(|a, b| a.total_cmp(b));
        pts.dedup();
        let shift = pts.iter().map(|&s| ln_f(s)).fold(f64::NEG_INFINITY, f64::max);
        let (val, _) = quadrature::integrate(|s| (ln_f(s) - shift).exp(), &pts, 1e-14, 20_000);
        shift + val.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn send_sync() {
        assert_send_sync::<DistanceMarginalization>();
    }
}
