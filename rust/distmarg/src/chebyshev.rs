//! Chebyshev series: Clenshaw evaluation, derivative recurrence, Vandermonde rows.

/// Σ c_k T_k(t) by Clenshaw's recurrence.
#[inline]
pub fn clenshaw(c: &[f64], t: f64) -> f64 {
    match c.len() {
        0 => 0.0,
        1 => c[0],
        n => {
            let two_t = 2.0 * t;
            let (mut b1, mut b2) = (0.0, 0.0);
            for &ck in c[1..n].iter().rev() {
                let b0 = ck + two_t * b1 - b2;
                b2 = b1;
                b1 = b0;
            }
            c[0] + t * b1 - b2
        }
    }
}

/// Coefficients of d/dt Σ c_k T_k(t): c'_{k-1} = c'_{k+1} + 2 k c_k, c'_0 halved.
pub fn derivative(c: &[f64]) -> Vec<f64> {
    let n = c.len();
    if n <= 1 {
        return vec![0.0];
    }
    let mut d = vec![0.0; n + 1];
    for k in (1..n).rev() {
        d[k - 1] = d[k + 1] + 2.0 * k as f64 * c[k];
    }
    d[0] *= 0.5;
    d.truncate(n - 1);
    d
}

/// T_0(t) .. T_degree(t) written into `row`.
pub fn vandermonde_row(t: f64, row: &mut [f64]) {
    if row.is_empty() {
        return;
    }
    row[0] = 1.0;
    if row.len() > 1 {
        row[1] = t;
    }
    for k in 2..row.len() {
        row[k] = 2.0 * t * row[k - 1] - row[k - 2];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivative_matches_finite_difference() {
        let c = [0.3, -1.2, 0.7, 0.05, -0.02, 0.011, 0.004];
        let d1 = derivative(&c);
        let d2 = derivative(&d1);
        for i in 0..9 {
            let t = -0.9 + 0.2 * i as f64;
            let h = 1e-5;
            let fd1 = (clenshaw(&c, t + h) - clenshaw(&c, t - h)) / (2.0 * h);
            let hh = 1e-3;
            let fd2 = (clenshaw(&c, t + hh) - 2.0 * clenshaw(&c, t) + clenshaw(&c, t - hh)) / (hh * hh);
            assert!((clenshaw(&d1, t) - fd1).abs() < 1e-9);
            assert!((clenshaw(&d2, t) - fd2).abs() < 1e-5);
        }
    }

    #[test]
    fn clenshaw_matches_vandermonde() {
        let c = [1.0, 2.0, -3.0, 0.5];
        let mut row = [0.0; 4];
        vandermonde_row(0.37, &mut row);
        let direct: f64 = c.iter().zip(row).map(|(a, b)| a * b).sum();
        assert!((direct - clenshaw(&c, 0.37)).abs() < 1e-14);
    }
}
