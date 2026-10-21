//! Gauss–Legendre nodes (Newton iteration on P_n) and adaptive Gauss–Kronrod (G7/K15)
//! with explicit break points.

use std::collections::BinaryHeap;
use std::f64::consts::PI;

/// Nodes (ascending) and weights of the n-point Gauss–Legendre rule on [-1, 1].
pub fn gauss_legendre(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    let legendre = |z: f64| {
        let (mut p1, mut p2) = (1.0, 0.0);
        for j in 0..n {
            let p3 = p2;
            p2 = p1;
            p1 = ((2 * j + 1) as f64 * z * p2 - j as f64 * p3) / (j + 1) as f64;
        }
        // P_n(z), P_n'(z)
        (p1, nf * (z * p1 - p2) / (z * z - 1.0))
    };
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        for _ in 0..100 {
            let (p, dp) = legendre(z);
            let dz = p / dp;
            z -= dz;
            if dz.abs() <= 1e-16 * z.abs().max(1e-300) {
                break;
            }
        }
        let (_, dp) = legendre(z);
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        x[i] = -z;
        x[n - 1 - i] = z;
        w[i] = wi;
        w[n - 1 - i] = wi;
    }
    if n % 2 == 1 {
        x[n / 2] = 0.0;
    }
    (x, w)
}

// QUADPACK dqk15 abscissae / weights (quoted at full published precision)
#[allow(clippy::excessive_precision)]
const XGK: [f64; 8] = [
    0.991_455_371_120_812_639_206_854_697_526_329,
    0.949_107_912_342_758_524_526_189_684_047_851,
    0.864_864_423_359_769_072_789_712_788_640_926,
    0.741_531_185_599_394_439_863_864_773_280_788,
    0.586_087_235_467_691_130_294_144_845_693_013,
    0.405_845_151_377_397_166_906_606_412_076_961,
    0.207_784_955_007_898_467_600_689_403_773_245,
    0.0,
];
#[allow(clippy::excessive_precision)]
const WGK: [f64; 8] = [
    0.022_935_322_010_529_224_963_732_008_058_970,
    0.063_092_092_629_978_553_290_700_663_189_204,
    0.104_790_010_322_250_183_839_876_322_541_518,
    0.140_653_259_715_525_918_745_189_590_510_238,
    0.169_004_726_639_267_902_826_583_426_598_550,
    0.190_350_578_064_785_409_913_256_402_421_014,
    0.204_432_940_075_298_892_414_161_999_234_649,
    0.209_482_141_084_727_828_012_999_174_891_714,
];
#[allow(clippy::excessive_precision)]
const WG: [f64; 4] = [
    0.129_484_966_168_869_693_270_611_432_679_082,
    0.279_705_391_489_276_667_901_467_771_423_780,
    0.381_830_050_505_118_944_950_369_775_488_975,
    0.417_959_183_673_469_387_755_102_040_816_327,
];

/// (Kronrod estimate, |Kronrod - Gauss|) on [a, b].
fn gk15<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> (f64, f64) {
    let c = 0.5 * (a + b);
    let h = 0.5 * (b - a);
    let fc = f(c);
    let mut rk = fc * WGK[7];
    let mut rg = fc * WG[3];
    for j in 0..7 {
        let dx = h * XGK[j];
        let s = f(c - dx) + f(c + dx);
        rk += WGK[j] * s;
        if j % 2 == 1 {
            rg += WG[j / 2] * s;
        }
    }
    (rk * h, ((rk - rg) * h).abs())
}

struct Seg {
    a: f64,
    b: f64,
    val: f64,
    err: f64,
}
impl PartialEq for Seg {
    fn eq(&self, o: &Self) -> bool {
        self.err == o.err
    }
}
impl Eq for Seg {}
impl PartialOrd for Seg {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Seg {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.err.total_cmp(&o.err)
    }
}

/// Adaptive G7/K15 integral of `f` over consecutive break points (sorted, deduplicated).
/// Bisects the segment with the largest error estimate until the total error is
/// below `rel_tol * |total|` or `max_segments` is reached.
pub fn integrate<F: Fn(f64) -> f64>(f: F, breaks: &[f64], rel_tol: f64, max_segments: usize) -> (f64, f64) {
    let mut heap = BinaryHeap::new();
    let (mut total, mut err) = (0.0, 0.0);
    for w in breaks.windows(2) {
        if w[1] > w[0] {
            let (v, e) = gk15(&f, w[0], w[1]);
            total += v;
            err += e;
            heap.push(Seg { a: w[0], b: w[1], val: v, err: e });
        }
    }
    while err > rel_tol * total.abs() && heap.len() < max_segments {
        let s = match heap.pop() {
            Some(s) => s,
            None => break,
        };
        let m = 0.5 * (s.a + s.b);
        if !(m > s.a && m < s.b) {
            heap.push(s);
            break;
        }
        let (v1, e1) = gk15(&f, s.a, m);
        let (v2, e2) = gk15(&f, m, s.b);
        total += v1 + v2 - s.val;
        err += e1 + e2 - s.err;
        heap.push(Seg { a: s.a, b: m, val: v1, err: e1 });
        heap.push(Seg { a: m, b: s.b, val: v2, err: e2 });
    }
    // re-sum for accuracy
    let total: f64 = heap.iter().map(|s| s.val).sum();
    let err: f64 = heap.iter().map(|s| s.err).sum();
    (total, err)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gauss_legendre_exact_for_degree_2n_minus_1() {
        for n in [1usize, 2, 5, 16, 32, 64, 256] {
            let (x, w) = gauss_legendre(n);
            assert!((w.iter().sum::<f64>() - 2.0).abs() < 1e-13);
            for deg in [2 * n - 2, 2 * n - 1] {
                let q: f64 = x.iter().zip(&w).map(|(xi, wi)| wi * xi.powi(deg as i32)).sum();
                let exact = if deg % 2 == 1 { 0.0 } else { 2.0 / (deg as f64 + 1.0) };
                assert!((q - exact).abs() < 1e-13, "n={n} deg={deg}: {q} vs {exact}");
            }
        }
    }

    #[test]
    fn kronrod_exact_for_degree_22() {
        let (v, _) = gk15(&|x: f64| x.powi(22) + x.powi(21), -1.0, 1.0);
        assert!((v - 2.0 / 23.0).abs() < 1e-15);
        let (v, _) = gk15(&|x: f64| x.powi(12), 0.0, 1.0);
        assert!((v - 1.0 / 13.0).abs() < 1e-15);
    }

    #[test]
    fn adaptive_peaked_integrand() {
        let s = 1e-4;
        let f = |x: f64| (-0.5 * ((x - 0.3) / s).powi(2)).exp();
        // GK15 has no end-point nodes: break points must bracket the peak scale
        let mut b = vec![0.0, 0.3, 1.0];
        for m in [1.0, 5.0, 10.0, 40.0] {
            b.push(0.3 - m * s);
            b.push(0.3 + m * s);
        }
        b.sort_by(f64::total_cmp);
        let (v, _) = integrate(f, &b, 1e-14, 2000);
        let exact = s * (2.0 * PI).sqrt();
        assert!((v / exact - 1.0).abs() < 1e-12, "{v} vs {exact}");
    }
}
