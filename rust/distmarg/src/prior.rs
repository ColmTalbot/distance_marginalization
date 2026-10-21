//! Canonical interpretation of the tabulated prior (IMPLEMENTATION_NOTES §3.0).

use crate::chebyshev::{clenshaw, derivative, vandermonde_row};
use crate::linalg::lstsq;
use crate::quadrature::gauss_legendre;
use crate::Error;

pub(crate) const N_NORM_NODES: usize = 256;
pub(crate) const N_MOMENTS: usize = 8;
pub(crate) const N_CENTRAL: usize = 24;

/// Fitted prior; see [`crate::FitInfo`] for the public summary.
#[derive(Debug, Clone)]
pub struct PriorFit {
    pub d_min: f64,
    pub d_max: f64,
    pub s_a: f64,
    pub s_b: f64,
    pub v_min: f64,
    pub v_max: f64,
    pub degree: usize,
    pub max_residual: f64,
    /// least-squares coefficients of the unnormalized ln pi_v(t)
    pub raw_coeffs: Vec<f64>,
    pub ln_norm: f64,
    /// normalized ln pi_v(t)
    pub coeffs: Vec<f64>,
    /// d/ds and d²/ds² of ln pi_v as Chebyshev series in t
    pub d1: Vec<f64>,
    pub d2: Vec<f64>,
    /// E_pi[v^k], k = 0..=8
    pub moments: Vec<f64>,
    /// E[z] and central moments E[(z - E z)^k], k = 0..=24, for z = nu and z = nu^2
    pub zbar_nu: f64,
    pub central_nu: Vec<f64>,
    pub zbar_nu2: f64,
    pub central_nu2: Vec<f64>,
}

fn validate(distance: &[f64], ln_prior: &[f64]) -> Result<(), Error> {
    if distance.len() != ln_prior.len() {
        return Err(Error::InvalidPrior("distance and ln_prior must have equal length".into()));
    }
    if distance.len() < 2 {
        return Err(Error::InvalidPrior("at least two prior points are required".into()));
    }
    if distance.iter().any(|d| !d.is_finite() || *d <= 0.0) {
        return Err(Error::InvalidPrior("distance must be finite and > 0".into()));
    }
    if distance.windows(2).any(|w| w[1] <= w[0]) {
        return Err(Error::InvalidPrior("distance must be strictly increasing".into()));
    }
    if ln_prior.iter().any(|x| !x.is_finite()) {
        return Err(Error::InvalidPrior(
            "ln_prior must be finite everywhere (zero prior density is not allowed)".into(),
        ));
    }
    Ok(())
}

fn fit_degree(t: &[f64], g: &[f64], degree: usize) -> (Vec<f64>, f64) {
    let m = t.len();
    let n = degree + 1;
    let mut a = vec![0.0; m * n];
    let mut row = vec![0.0; n];
    for (i, &ti) in t.iter().enumerate() {
        vandermonde_row(ti, &mut row);
        for j in 0..n {
            a[j * m + i] = row[j];
        }
    }
    let c = lstsq(&a, m, n, g);
    let resid = t
        .iter()
        .zip(g)
        .map(|(&ti, &gi)| (clenshaw(&c, ti) - gi).abs())
        .fold(0.0f64, f64::max);
    (c, resid)
}

impl PriorFit {
    /// Returns the fit and an optional warning when the residual reached `fit_tol`.
    pub fn new(
        distance: &[f64],
        ln_prior: &[f64],
        degree: Option<usize>,
        fit_tol: f64,
    ) -> Result<(Self, Option<String>), Error> {
        validate(distance, ln_prior)?;
        let n = distance.len();
        let d_min = distance[0];
        let d_max = distance[n - 1];
        let s_a = -d_max.ln();
        let s_b = -d_min.ln();
        let t: Vec<f64> = distance.iter().map(|d| (2.0 * -d.ln() - s_a - s_b) / (s_b - s_a)).collect();
        let g: Vec<f64> = distance.iter().zip(ln_prior).map(|(d, l)| l + 2.0 * d.ln()).collect();

        let (degree, raw, resid) = match degree {
            Some(k) => {
                if n < 2 * (k + 1) {
                    return Err(Error::InvalidPrior(format!(
                        "need len(distance) >= 2(K+1) = {} for fit_degree={k}",
                        2 * (k + 1)
                    )));
                }
                let (c, r) = fit_degree(&t, &g, k);
                (k, c, r)
            }
            None => {
                let allowed: Vec<usize> = (8..=128).step_by(8).filter(|k| n >= 2 * (k + 1)).collect();
                if allowed.is_empty() {
                    return Err(Error::InvalidPrior(
                        "need at least 18 prior points for automatic degree selection".into(),
                    ));
                }
                let mut best = None;
                for &k in &allowed {
                    let (c, r) = fit_degree(&t, &g, k);
                    let done = r < fit_tol;
                    best = Some((k, c, r));
                    if done {
                        break;
                    }
                }
                best.expect("non-empty candidate list")
            }
        };
        let warning = if resid < fit_tol {
            None
        } else {
            Some(format!(
                "Chebyshev fit of ln pi_v has max residual {resid:.3e} >= fit_tol={fit_tol:.3e} (degree {degree}); \
                 the prior may be non-smooth or noisily tabulated and the high-SNR curvature model is unreliable"
            ))
        };

        // normalization: ln ∫ exp(g~(s) + s) ds with 256-node Gauss–Legendre
        let (xg, wg) = gauss_legendre(N_NORM_NODES);
        let half = 0.5 * (s_b - s_a);
        let mid = 0.5 * (s_a + s_b);
        let log_terms: Vec<f64> = xg
            .iter()
            .zip(&wg)
            .map(|(&x, &w)| (half * w).ln() + clenshaw(&raw, x) + mid + half * x)
            .collect();
        let mx = log_terms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let ln_norm = mx + log_terms.iter().map(|l| (l - mx).exp()).sum::<f64>().ln();

        let mut coeffs = raw.clone();
        coeffs[0] -= ln_norm;
        let scale = 2.0 / (s_b - s_a);
        let d1: Vec<f64> = derivative(&coeffs).into_iter().map(|c| c * scale).collect();
        let d2: Vec<f64> = derivative(&d1).into_iter().map(|c| c * scale).collect();

        let pw: Vec<f64> = log_terms.iter().map(|l| (l - ln_norm).exp()).collect();
        let v: Vec<f64> = xg.iter().map(|x| (mid + half * x).exp()).collect();
        let moments: Vec<f64> = (0..=N_MOMENTS)
            .map(|k| pw.iter().zip(&v).map(|(p, vi)| p * vi.powi(k as i32)).sum())
            .collect();

        let psum: f64 = pw.iter().sum();
        let pn: Vec<f64> = pw.iter().map(|p| p / psum).collect();
        let central = |z: &[f64]| {
            let zb: f64 = pn.iter().zip(z).map(|(p, zi)| p * zi).sum();
            let mut c: Vec<f64> = (0..=N_CENTRAL)
                .map(|k| pn.iter().zip(z).map(|(p, zi)| p * (zi - zb).powi(k as i32)).sum())
                .collect();
            c[0] = 1.0;
            c[1] = 0.0;
            (zb, c)
        };
        let nu: Vec<f64> = v.iter().map(|vi| vi * d_min).collect();
        let nu2: Vec<f64> = nu.iter().map(|x| x * x).collect();
        let (zbar_nu, central_nu) = central(&nu);
        let (zbar_nu2, central_nu2) = central(&nu2);

        Ok((
            PriorFit {
                d_min,
                d_max,
                s_a,
                s_b,
                v_min: 1.0 / d_max,
                v_max: 1.0 / d_min,
                degree,
                max_residual: resid,
                raw_coeffs: raw,
                ln_norm,
                coeffs,
                d1,
                d2,
                moments,
                zbar_nu,
                central_nu,
                zbar_nu2,
                central_nu2,
            },
            warning,
        ))
    }

    #[inline]
    pub fn t_of_s(&self, s: f64) -> f64 {
        (2.0 * s - self.s_a - self.s_b) / (self.s_b - self.s_a)
    }

    /// Normalized ln pi_v at s = ln v.
    #[inline]
    pub fn ln_pi_v(&self, s: f64) -> f64 {
        clenshaw(&self.coeffs, self.t_of_s(s))
    }
}
