//! Householder-QR linear least squares (column-major, written in-crate).

/// Minimize ||A x - b||_2 for an `m x n` column-major matrix `a` (m >= n, full column rank).
pub fn lstsq(a: &[f64], m: usize, n: usize, b: &[f64]) -> Vec<f64> {
    assert!(a.len() == m * n && b.len() == m && m >= n);
    let mut q = a.to_vec();
    let mut rhs = b.to_vec();
    let col = |j: usize| j * m;
    let mut v = vec![0.0; m];
    for k in 0..n {
        // Householder vector for column k, rows k..m
        let ck = col(k);
        let scale = q[ck + k..ck + m].iter().fold(0.0f64, |s, x| s.max(x.abs()));
        if scale == 0.0 {
            continue;
        }
        let norm = scale * q[ck + k..ck + m].iter().map(|x| (x / scale).powi(2)).sum::<f64>().sqrt();
        let alpha = if q[ck + k] > 0.0 { -norm } else { norm };
        for i in k..m {
            v[i] = q[ck + i];
        }
        v[k] -= alpha;
        let vnorm2: f64 = v[k..m].iter().map(|x| x * x).sum();
        if vnorm2 == 0.0 {
            continue;
        }
        for j in k..n {
            let cj = col(j);
            let s: f64 = (k..m).map(|i| v[i] * q[cj + i]).sum::<f64>() * 2.0 / vnorm2;
            for i in k..m {
                q[cj + i] -= s * v[i];
            }
        }
        let s: f64 = (k..m).map(|i| v[i] * rhs[i]).sum::<f64>() * 2.0 / vnorm2;
        for i in k..m {
            rhs[i] -= s * v[i];
        }
    }
    // back substitution R x = (Q^T b)[0..n]
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= q[col(j) + i] * x[j];
        }
        x[i] = s / q[col(i) + i];
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_exact_polynomial() {
        let m = 50;
        let n = 6;
        let coef = [1.5, -2.0, 0.25, 3.0, -0.75, 0.1];
        let mut a = vec![0.0; m * n];
        let mut b = vec![0.0; m];
        for i in 0..m {
            let t = -1.0 + 2.0 * i as f64 / (m - 1) as f64;
            for j in 0..n {
                a[j * m + i] = t.powi(j as i32);
                b[i] += coef[j] * t.powi(j as i32);
            }
        }
        let x = lstsq(&a, m, n, &b);
        for (xi, ci) in x.iter().zip(coef) {
            assert!((xi - ci).abs() < 1e-12, "{xi} vs {ci}");
        }
    }

    #[test]
    fn least_squares_residual_orthogonal() {
        let m = 7;
        let n = 2;
        let a: Vec<f64> = (0..m).map(|_| 1.0).chain((0..m).map(|i| i as f64)).collect();
        let b = [1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0];
        let x = lstsq(&a, m, n, &b);
        // normal equations hold at the solution
        for j in 0..n {
            let g: f64 = (0..m).map(|i| a[j * m + i] * (a[i] * x[0] + a[m + i] * x[1] - b[i])).sum();
            assert!(g.abs() < 1e-12);
        }
    }
}
