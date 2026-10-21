//! Special functions: ln Phi, ln(Phi(b) - Phi(a)), log1mexp, scaled Bessel I0/I1,
//! truncated-exponential (Mills) moments and the ln I0 power series.
//!
//! `erfc` comes from `libm` (a port of musl, itself derived from FreeBSD/Sun);
//! `erfcx` uses `erfc` with an exactly-split square below 26 and Laplace's
//! continued fraction above.  The scaled Bessel functions use the power series
//! below |x| = 20 and the Hankel asymptotic expansion above.

use std::f64::consts::{FRAC_1_SQRT_2, LN_2, PI};

/// ln(sqrt(2 pi))
#[allow(clippy::excessive_precision)]
pub const HALF_LN_2PI: f64 = 0.918_938_533_204_672_8;
#[allow(clippy::excessive_precision)]
const FRAC_1_SQRT_PI: f64 = 0.564_189_583_547_756_3;
#[allow(clippy::excessive_precision)]
const SQRT_FRAC_PI_2: f64 = 1.253_314_137_315_500_3;

/// x^2 = hi + lo exactly.
#[inline]
fn two_square(x: f64) -> (f64, f64) {
    let hi = x * x;
    (hi, x.mul_add(x, -hi))
}

/// Scaled complementary error function exp(y^2) erfc(y) for y >= 0.
pub fn erfcx(y: f64) -> f64 {
    if y.is_nan() {
        return f64::NAN;
    }
    if y < 0.0 {
        let (hi, lo) = two_square(y);
        return 2.0 * hi.exp() * (1.0 + lo) - erfcx(-y);
    }
    if y < 26.0 {
        let (hi, lo) = two_square(y);
        hi.exp() * (1.0 + lo) * libm::erfc(y)
    } else {
        // erfc(y) = exp(-y^2)/sqrt(pi) * 1/(y + (1/2)/(y + 1/(y + (3/2)/(y + ...))))
        let mut t = 0.0;
        for k in (1..=40).rev() {
            t = (k as f64 * 0.5) / (y + t);
        }
        FRAC_1_SQRT_PI / (y + t)
    }
}

/// ln Phi(x) for the standard normal CDF.
pub fn log_ndtr(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let y = x.abs() * FRAC_1_SQRT_2;
    if x >= 0.0 {
        (-0.5 * libm::erfc(y)).ln_1p()
    } else if x > -1.0 {
        (0.5 * libm::erfc(y)).ln()
    } else {
        let (hi, lo) = two_square(x);
        -LN_2 + erfcx(y).ln() - 0.5 * hi - 0.5 * lo
    }
}

/// ln of the standard normal density.
#[inline]
pub fn ln_phi(x: f64) -> f64 {
    -0.5 * x * x - HALF_LN_2PI
}

/// ln(1 - exp(x)) for x <= 0.
pub fn log1mexp(x: f64) -> f64 {
    if x > -LN_2 {
        (-x.exp_m1()).ln()
    } else {
        (-x.exp()).ln_1p()
    }
}

/// ln(Phi(b) - Phi(a)) for a < b.
pub fn log_diff_ndtr(a: f64, b: f64) -> f64 {
    let (lo, hi) = if a > 0.0 { (-b, -a) } else { (a, b) };
    let la = log_ndtr(lo);
    let lb = log_ndtr(hi);
    lb + log1mexp((la - lb).min(0.0))
}

fn bessel_scaled(x: f64, order: u32) -> f64 {
    let ax = x.abs();
    let r = if ax < 20.0 {
        let q = 0.25 * ax * ax;
        let mut term = if order == 0 { 1.0 } else { 0.5 * ax };
        let mut sum = term;
        for k in 1..100u32 {
            term *= q / f64::from(k * (k + order));
            sum += term;
            if term <= sum * 1e-17 {
                break;
            }
        }
        sum * (-ax).exp()
    } else {
        let mu = f64::from(4 * order * order);
        let mut term = 1.0;
        let mut sum = 1.0;
        for k in 1..40u32 {
            let odd = f64::from(2 * k - 1);
            term *= -(mu - odd * odd) / (f64::from(k) * 8.0 * ax);
            sum += term;
            if term.abs() <= sum.abs() * 1e-17 {
                break;
            }
        }
        sum / (2.0 * PI * ax).sqrt()
    };
    if order == 1 && x < 0.0 {
        -r
    } else {
        r
    }
}

/// exp(-|x|) I_0(x)
pub fn i0e(x: f64) -> f64 {
    bessel_scaled(x, 0)
}

/// exp(-|x|) I_1(x)
pub fn i1e(x: f64) -> f64 {
    bessel_scaled(x, 1)
}

/// Truncated-exponential moments for a >= 0:
/// `m = ∫_0^∞ e^{-a t - t²/2} dt`, `u = ∫ t e^{..} = 1 - a m`, `w = ∫ t² e^{..} = m - a u`.
/// For a >= 3 the Laplace continued fraction `m = 1/(a + D_1)`, `D_k = k/(a + D_{k+1})`
/// gives `u = m D_1` and `w = m D_1 D_2` without cancellation.
pub fn mills(a: f64) -> (f64, f64, f64) {
    if a < 3.0 {
        let m = SQRT_FRAC_PI_2 * erfcx(a * FRAC_1_SQRT_2);
        let u = 1.0 - a * m;
        let w = m - a * u;
        (m, u, w)
    } else {
        let mut d = 0.0;
        let mut d2 = 0.0;
        for k in (1..=60u32).rev() {
            let dk = f64::from(k) / (a + d);
            if k == 2 {
                d2 = dk;
            }
            d = dk;
        }
        let m = 1.0 / (a + d);
        (m, m * d, m * d * d2)
    }
}

/// Coefficients c_k (k = 1..=n) of ln I0(x) = Σ c_k x^{2k}, from the power-series
/// logarithm of I0 = Σ (x²/4)^k / (k!)².
pub fn ln_i0_series(n: usize) -> Vec<f64> {
    let mut a = vec![1.0f64; n + 1];
    for k in 1..=n {
        a[k] = a[k - 1] / (4.0 * (k * k) as f64);
    }
    let mut f = vec![0.0f64; n + 1];
    for m in 1..=n {
        let mut acc = a[m];
        for k in 1..m {
            acc -= (k as f64 / m as f64) * f[k] * a[m - k];
        }
        f[m] = acc;
    }
    f[1..].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ln_i0_series_rationals() {
        let c = ln_i0_series(6);
        let exact = [
            0.25,
            -1.0 / 64.0,
            1.0 / 576.0,
            -11.0 / 49152.0,
            19.0 / 614_400.0,
            -473.0 / 106_168_320.0,
        ];
        for (a, b) in c.iter().zip(exact) {
            assert!((a - b).abs() <= 1e-15 * b.abs());
        }
    }

    #[test]
    fn erfcx_branches_continuous() {
        // d ln erfcx / dy ~ -1/y, so a 1e-12 step changes the value by ~4e-14
        let a = erfcx(26.0 - 1e-12);
        let b = erfcx(26.0);
        assert!((a / b - 1.0).abs() < 1e-13);
    }

    #[test]
    fn mills_branches_continuous() {
        let (m1, u1, w1) = mills(3.0 - 1e-13);
        let (m2, u2, w2) = mills(3.0);
        assert!((m1 / m2 - 1.0).abs() < 1e-13);
        assert!((u1 / u2 - 1.0).abs() < 1e-12);
        assert!((w1 / w2 - 1.0).abs() < 1e-12);
    }
}
