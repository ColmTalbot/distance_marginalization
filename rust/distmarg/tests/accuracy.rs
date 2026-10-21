//! Stress-grid accuracy against the adaptive Gauss–Kronrod reference and
//! regime-boundary continuity.
mod common;
use common::*;
use distmarg::Regime;

#[test]
fn stress_grid_vs_reference() {
    for name in ["usf_100_5000", "usf_10_50000"] {
        for phase in [false, true] {
            let dm = make(name, phase);
            let mut worst = (0.0f64, (0.0, 0.0, 0.0));
            for (rm, ro, dr) in stress_grid(&dm, 18) {
                let got = dm.ln_likelihood_ratio(rm, ro, dr);
                let refv = dm.reference_ln_likelihood_ratio(rm, ro, dr);
                assert!(got.is_finite() && refv.is_finite(), "{name} {:?}", (rm, ro, dr));
                let err = (got - refv).abs() - 1e-15 * rm * rm;
                if err > worst.0 {
                    worst = (err, (rm, ro, dr));
                }
            }
            assert!(worst.0 < 1e-4, "{name} phase={phase}: worst {worst:?}");
        }
    }
}

#[test]
fn quadrature_branch_vs_reference() {
    let dm = make("usf_10_50000", false);
    for (rm, ro, dr) in stress_grid(&dm, 14) {
        if rm.abs() > 100.0 {
            continue;
        }
        let q = dm.ln_likelihood_ratio_in(Regime::Quadrature, rm, ro, dr);
        let r = dm.reference_ln_likelihood_ratio(rm, ro, dr);
        assert!((q - r).abs() < 1e-4, "{:?}: {q} vs {r}", (rm, ro, dr));
    }
}

#[test]
fn regime_boundary_continuity() {
    for name in PRIORS {
        for phase in [false, true] {
            let dm = make(name, phase);
            let pr = dm.prior();
            let mut lines: Vec<Vec<(f64, f64)>> = Vec::new();
            let n = 3000;
            let geom = |a: f64, b: f64, i: usize| a * (b / a).powf(i as f64 / (n - 1) as f64);
            for ds in [pr.d_min / 3.0, pr.d_min * 1.5, (pr.d_min * pr.d_max).sqrt(), pr.d_max * 0.7, pr.d_max * 3.0] {
                lines.push((0..n).map(|i| geom(1e-2, 300.0, i)).map(|r| (r, r * ds / 1000.0)).collect());
            }
            for r0 in [0.3, 3.0, 40.0] {
                lines.push((0..n).map(|i| (r0, geom(1e-3, 1e3, i))).collect());
            }
            let mut switches = 0;
            for line in lines {
                for w in line.windows(2) {
                    let (r1, o1) = w[0];
                    let (r2, o2) = w[1];
                    let g1 = dm.regime(r1, o1, 1000.0);
                    let g2 = dm.regime(r2, o2, 1000.0);
                    if g1 != g2 {
                        switches += 1;
                        let jump = (dm.ln_likelihood_ratio(r2, o2, 1000.0) - dm.ln_likelihood_ratio(r1, o1, 1000.0))
                            - (dm.ln_likelihood_ratio_in(Regime::Quadrature, r2, o2, 1000.0)
                                - dm.ln_likelihood_ratio_in(Regime::Quadrature, r1, o1, 1000.0));
                        assert!(jump.abs() < 1e-4, "{name} phase={phase} {g1:?}->{g2:?} at {:?}: {jump}", w[0]);
                    }
                }
            }
            assert!(switches > 0);
        }
    }
}

#[test]
fn all_regimes_exercised_and_invalid_is_nan() {
    let dm = make("usf_100_5000", false);
    let mut seen = std::collections::HashSet::new();
    for (rm, ro, dr) in stress_grid(&dm, 20) {
        seen.insert(dm.regime(rm, ro, dr));
    }
    assert!(seen.contains(&Regime::HighSnr) && seen.contains(&Regime::PriorDominated) && seen.contains(&Regime::Quadrature));
    assert!(dm.ln_likelihood_ratio(1.0, 0.0, 1000.0).is_nan());
    assert!(dm.ln_likelihood_ratio(1.0, 1.0, -1.0).is_nan());
    assert!(dm.ln_likelihood_ratio(f64::NAN, 1.0, 1000.0).is_nan());
    assert_eq!(dm.regime(1.0, -1.0, 1000.0), Regime::Invalid);
}

#[test]
fn negative_rho_and_phase_abs() {
    let dm = make("usf_100_5000", false);
    for rho in [-2.0, -40.0, -500.0] {
        for ro in [0.5, 5.0, 50.0] {
            let a = dm.ln_likelihood_ratio(rho, ro, 1000.0);
            let r = dm.reference_ln_likelihood_ratio(rho, ro, 1000.0);
            assert!((a - r).abs() < 1e-4);
        }
    }
    let dmp = make("usf_100_5000", true);
    assert_eq!(dmp.ln_likelihood_ratio(-5.0, 3.0, 1000.0), dmp.ln_likelihood_ratio(5.0, 3.0, 1000.0));
}

#[test]
fn batch_matches_scalar() {
    let dm = make("usf_10_50000", true);
    let g = stress_grid(&dm, 10);
    let rm: Vec<f64> = g.iter().map(|x| x.0).collect();
    let ro: Vec<f64> = g.iter().map(|x| x.1).collect();
    let dr: Vec<f64> = g.iter().map(|x| x.2).collect();
    let mut out = vec![0.0; g.len()];
    dm.ln_likelihood_ratio_batch(&rm, &ro, &dr, &mut out).unwrap();
    for i in 0..g.len() {
        assert_eq!(out[i].to_bits(), dm.ln_likelihood_ratio(rm[i], ro[i], dr[i]).to_bits());
    }
    assert!(dm.ln_likelihood_ratio_batch(&rm[1..], &ro, &dr, &mut out).is_err());
}
