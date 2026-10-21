//! Evaluation branches and the regime-selection rule (SPEC_DECISIONS.md §3-§7).
//! All functions are scalar, allocation-free and non-panicking.
#![allow(clippy::needless_range_loop)] // index loops mirror the formulas in SPEC_DECISIONS.md

use crate::chebyshev::clenshaw;
use crate::special::{i0e, i1e, ln_phi, log_diff_ndtr, mills, HALF_LN_2PI};
use crate::DistanceMarginalization;

/// Break points of the middle-regime quadrature, in units of the peak scale.
pub(crate) const WINDOW: [f64; 5] = [-40.0, -8.0, 0.0, 8.0, 40.0];
const L_CLIP: f64 = 60.0;
/// Maximum polynomial degree of ln ell in z (phase: 6 terms of the ln I0 series).
pub(crate) const MAX_DEG: usize = 6;
/// Maximum degree of (ln ell)^4.
const MAX_POW_DEG: usize = 4 * MAX_DEG;

pub(crate) struct TruncNorm {
    /// ln[sqrt(2 pi) sigma Z]
    pub ln_mass: f64,
    pub vbar: f64,
    pub var: f64,
}

/// Mean/variance/mass of N(v*, sigma) truncated to [vmin, vmax], stable in the tails.
pub(crate) fn truncnorm(vstar: f64, sigma: f64, vmin: f64, vmax: f64) -> TruncNorm {
    let alpha = (vmin - vstar) / sigma;
    let beta = (vmax - vstar) / sigma;
    let flip = alpha + beta < 0.0;
    let (a, b) = if flip { (-beta, -alpha) } else { (alpha, beta) };
    let (mean_x, var_x, lnz) = if a >= 0.0 {
        // shifted variable t = x - a with density ∝ exp(-a t - t²/2) on [0, L]
        let l = (b - a).min(L_CLIP);
        let (m_a, u_a, w_a) = mills(a);
        let (m_b, u_b, w_b) = mills(a + l);
        let q = (-a * l - 0.5 * l * l).exp();
        let i0 = (m_a - q * m_b).max(l * q);
        let i1 = u_a - q * (u_b + l * m_b);
        let i2 = w_a - q * (w_b + 2.0 * l * u_b + l * l * m_b);
        let mt = i1 / i0;
        (a + mt, i2 / i0 - mt * mt, -0.5 * a * a + i0.ln())
    } else {
        let lnz = log_diff_ndtr(a, b);
        let ra = (ln_phi(a) - lnz).exp();
        let rb = (ln_phi(b) - lnz).exp();
        let mean = ra - rb;
        (mean, 1.0 + a * ra - b * rb - mean * mean, lnz + HALF_LN_2PI)
    };
    let mean_x = if flip { -mean_x } else { mean_x };
    let vbar = (vstar + sigma * mean_x).clamp(vmin, vmax);
    let var = (var_x.max(0.0) * sigma * sigma).min(0.25 * (vmax - vmin) * (vmax - vmin));
    TruncNorm { ln_mass: lnz + sigma.ln(), vbar, var }
}

/// ln ell as a polynomial in z: returns (coefficients p_0..p_D, D).
pub(crate) fn polynomial(dm: &DistanceMarginalization, a_p: f64, b_p: f64) -> ([f64; MAX_DEG + 1], usize) {
    let mut p = [0.0; MAX_DEG + 1];
    if dm.phase {
        let a2 = a_p * a_p;
        p[1] = dm.ln_i0_c[0] * a2 - 0.5 * b_p;
        let mut power = a2;
        for k in 1..MAX_DEG {
            power *= a2;
            p[k + 1] = dm.ln_i0_c[k] * power;
        }
        (p, MAX_DEG)
    } else {
        p[1] = a_p;
        p[2] = -0.5 * b_p;
        (p, 2)
    }
}

fn binom(n: usize, k: usize) -> f64 {
    let mut r = 1.0;
    for i in 0..k {
        r = r * (n - i) as f64 / (i + 1) as f64;
    }
    r.round()
}

/// Re-expansion about E[z]: (kappa_1, coefficients of ln ell - kappa_1 in delta = z - E[z]).
pub(crate) fn centered(dm: &DistanceMarginalization, p: &[f64; MAX_DEG + 1], deg: usize) -> (f64, [f64; MAX_DEG + 1]) {
    let zb = dm.zbar;
    let c = &dm.central;
    let mut shifted = [0.0; MAX_DEG + 1];
    for j in 0..=deg {
        let mut acc = 0.0;
        for k in j.max(1)..=deg {
            acc += binom(k, j) * zb.powi((k - j) as i32) * p[k];
        }
        shifted[j] = acc;
    }
    let mut kappa1 = shifted[0];
    let mut y0 = 0.0;
    for j in 2..=deg {
        kappa1 += shifted[j] * c[j];
        y0 -= shifted[j] * c[j];
    }
    shifted[0] = y0;
    (kappa1, shifted)
}

pub(crate) fn kappa2(dm: &DistanceMarginalization, p: &[f64; MAX_DEG + 1], deg: usize) -> f64 {
    let (_, y) = centered(dm, p, deg);
    let c = &dm.central;
    let mut total = 0.0;
    for i in 0..=deg {
        for j in 0..=deg {
            if c[i + j] != 0.0 {
                total += y[i] * y[j] * c[i + j];
            }
        }
    }
    total
}

#[inline]
fn horner(p: &[f64; MAX_DEG + 1], deg: usize, z: f64) -> f64 {
    let mut out = p[deg];
    for k in (0..deg).rev() {
        out = out * z + p[k];
    }
    out
}

/// max - min of ln ell(z) over [z0, 1], using the vertex of its quadratic part.
pub(crate) fn ln_ell_range(dm: &DistanceMarginalization, p: &[f64; MAX_DEG + 1], deg: usize) -> f64 {
    let z0 = dm.z0;
    let z_c = (p[1] / (2.0 * (-p[2]).max(1e-300))).clamp(z0, 1.0);
    let f0 = horner(p, deg, z0);
    let f1 = horner(p, deg, 1.0);
    let fc = horner(p, deg, z_c);
    f0.max(f1).max(fc) - f0.min(f1)
}

pub(crate) fn branch_cumulant(dm: &DistanceMarginalization, rm: f64, k: f64) -> f64 {
    let a_p = rm * k * dm.prior.v_max;
    let b_p = (k * dm.prior.v_max) * (k * dm.prior.v_max);
    let (p, deg) = polynomial(dm, a_p, b_p);
    let (kappa1, y) = centered(dm, &p, deg);
    let c = &dm.central;
    let mut power = [0.0; MAX_POW_DEG + 1];
    let mut next = [0.0; MAX_POW_DEG + 1];
    power[..=deg].copy_from_slice(&y[..=deg]);
    let mut pdeg = deg;
    let mut mu = [0.0; 5];
    for n in 2..=dm.cfg.cumulant_order {
        let ndeg = pdeg + deg;
        for i in 0..=ndeg {
            let mut acc = 0.0;
            let lo = i.saturating_sub(deg);
            for j in lo..=i.min(pdeg) {
                acc += power[j] * y[i - j];
            }
            next[i] = acc;
        }
        power[..=ndeg].copy_from_slice(&next[..=ndeg]);
        pdeg = ndeg;
        let mut total = 0.0;
        for (i, coef) in power[..=pdeg].iter().enumerate() {
            if c[i] != 0.0 {
                total += coef * c[i];
            }
        }
        mu[n] = total;
    }
    let order = dm.cfg.cumulant_order;
    let mut out = kappa1;
    if order >= 2 {
        out += mu[2] / 2.0;
    }
    if order >= 3 {
        out += mu[3] / 6.0;
    }
    if order >= 4 {
        out += (mu[4] - 3.0 * mu[2] * mu[2]) / 24.0;
    }
    out
}

pub(crate) fn branch_high(dm: &DistanceMarginalization, rm: f64, k: f64) -> f64 {
    let pr = &dm.prior;
    let sigma = 1.0 / k;
    let tn = truncnorm(rm * sigma, sigma, pr.v_min, pr.v_max);
    let vbar = tn.vbar;
    let t = pr.t_of_s(vbar.ln());
    let g = clenshaw(&pr.coeffs, t);
    let g1 = clenshaw(&pr.d1, t);
    let g2 = clenshaw(&pr.d2, t);
    let mut h1 = g1 / vbar;
    let mut h2 = (g2 - g1) / (vbar * vbar);
    let mut out = 0.5 * rm * rm + tn.ln_mass + g;
    if dm.phase {
        let amp = rm * k;
        let x = amp * vbar;
        let i0 = i0e(x);
        let r = i1e(x) / i0;
        let r_over_x = if x < 1e-8 { 0.5 } else { r / x };
        h1 += amp * (r - 1.0);
        h2 += amp * amp * (1.0 - r_over_x - r * r);
        out += i0.ln();
    }
    out + 0.5 * tn.var * (h2 + h1 * h1)
}

/// Cubic-Hermite interpolation of ln pi_v on the uniform knot grid in s.
#[inline]
fn ln_pi_v_grid(dm: &DistanceMarginalization, s: f64) -> f64 {
    let n = dm.grid_g.len();
    let h = dm.grid_h;
    let u = (s - dm.prior.s_a) / h;
    let i = u.floor().clamp(0.0, (n - 2) as f64);
    let w = u - i;
    let i = i as usize;
    let (w2, w3) = (w * w, w * w * w);
    (2.0 * w3 - 3.0 * w2 + 1.0) * dm.grid_g[i]
        + (w3 - 2.0 * w2 + w) * h * dm.grid_dg[i]
        + (3.0 * w2 - 2.0 * w3) * dm.grid_g[i + 1]
        + (w3 - w2) * h * dm.grid_dg[i + 1]
}

pub(crate) fn branch_quadrature(dm: &DistanceMarginalization, rm: f64, k: f64) -> f64 {
    let pr = &dm.prior;
    let (vmin, vmax) = (pr.v_min, pr.v_max);
    let sigma = 1.0 / k;
    let vstar = rm * sigma;
    let amp = rm * k;
    let bb = k * k;
    let centre = vstar.clamp(vmin, vmax);
    let dist = (vstar - centre).abs();
    let scale = sigma * sigma / sigma.max(dist);
    let mut breaks = [0.0; WINDOW.len() + 2];
    breaks[0] = pr.s_a;
    for (i, w) in WINDOW.iter().enumerate() {
        breaks[i + 1] = (centre + w * scale).clamp(vmin, vmax).ln().clamp(pr.s_a, pr.s_b);
    }
    breaks[WINDOW.len() + 1] = pr.s_b;
    // streaming log-sum-exp: value = mx + ln(total)
    let mut mx = f64::NEG_INFINITY;
    let mut total = 0.0;
    for p in 0..breaks.len() - 1 {
        let (lo, hi) = (breaks[p], breaks[p + 1]);
        let half = 0.5 * (hi - lo);
        let mid = 0.5 * (hi + lo);
        for (&x, &lnw) in dm.gl_x.iter().zip(&dm.gl_lnw) {
            let s = mid + half * x;
            let v = s.exp();
            let av = amp * v;
            let ln_pi = if dm.grid_g.is_empty() { clenshaw(&pr.coeffs, pr.t_of_s(s)) } else { ln_pi_v_grid(dm, s) };
            let mut f = ln_pi + s + av - 0.5 * bb * v * v + lnw;
            if dm.phase {
                f += i0e(av).ln();
            }
            if f > mx {
                total *= (mx - f).exp();
                mx = f;
            }
            total += half * (f - mx).exp();
        }
    }
    mx + total.ln()
}
