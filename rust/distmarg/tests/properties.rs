//! proptest properties (IMPLEMENTATION_NOTES §10.5).
mod common;
use common::*;
use distmarg::DistanceMarginalization;
use proptest::prelude::*;
use std::sync::OnceLock;

fn dms() -> &'static Vec<DistanceMarginalization> {
    static CELL: OnceLock<Vec<DistanceMarginalization>> = OnceLock::new();
    CELL.get_or_init(|| {
        let mut v = Vec::new();
        for name in ["usf_100_5000", "usf_10_50000"] {
            v.push(make(name, false));
            v.push(make(name, true));
        }
        v
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(400))]

    #[test]
    fn d_ref_scaling_invariance(which in 0usize..4, lr in -3.0f64..4.0, ld in -1.0f64..1.0, neg in any::<bool>(),
                                lk in -1.0f64..1.0) {
        let dm = &dms()[which];
        let pr = dm.prior();
        let rho = if neg { -10f64.powf(lr) } else { 10f64.powf(lr) };
        let dstar = (pr.d_min / 5.0) * (25.0 * pr.d_max / pr.d_min).powf(0.5 * (ld + 1.0));
        let ro = rho.abs() * dstar / 1000.0;
        let k = 10f64.powf(lk);
        let a = dm.ln_likelihood_ratio(rho, ro, 1000.0);
        let b = dm.ln_likelihood_ratio(rho, ro / k, 1000.0 * k);
        prop_assert!((a - b).abs() <= 1e-9 * a.abs().max(1.0), "{} vs {}", a, b);
    }

    #[test]
    fn finite_over_stress_domain(which in 0usize..4, lr in -3.0f64..4.0, ld in 0.0f64..1.0, neg in any::<bool>()) {
        let dm = &dms()[which];
        let pr = dm.prior();
        let rho = if neg { -10f64.powf(lr) } else { 10f64.powf(lr) };
        let dstar = (pr.d_min / 5.0) * (25.0 * pr.d_max / pr.d_min).powf(ld);
        let v = dm.ln_likelihood_ratio(rho, rho.abs() * dstar / 1000.0, 1000.0);
        prop_assert!(v.is_finite());
        // ln L_D <= rho^2/2 for a normalized prior
        prop_assert!(v <= 0.5 * rho * rho + 1e-9 * (1.0 + rho * rho));
    }

    #[test]
    fn continuity(which in 0usize..4, lr in -2.0f64..3.0, lo in -3.0f64..3.0) {
        let dm = &dms()[which];
        let rho = 10f64.powf(lr);
        let ro = 10f64.powf(lo) * dm.prior().d_min / 1000.0;
        let eps = 1e-7;
        let a = dm.ln_likelihood_ratio(rho, ro, 1000.0);
        let b = dm.ln_likelihood_ratio(rho * (1.0 + eps), ro * (1.0 + eps), 1000.0);
        // smooth part changes by O(eps (1 + rho^2)); regime switches add < 1e-4
        prop_assert!((a - b).abs() < 1e-4 + 1e-5 * (1.0 + rho * rho));
    }

    #[test]
    fn low_snr_limit(which in 0usize..4, le in -8.0f64..-3.0) {
        let dm = &dms()[which];
        let eps = 10f64.powf(le);
        let v = dm.ln_likelihood_ratio(eps, eps, 1000.0);
        prop_assert!(v.abs() < 10.0 * eps);
    }
}
