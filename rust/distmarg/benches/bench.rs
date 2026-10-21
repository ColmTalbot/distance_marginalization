use std::path::PathBuf;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use distmarg::{Config, DistanceMarginalization, Regime};

fn load_prior(name: &str) -> (Vec<f64>, Vec<f64>) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/priors/{name}.json"));
    let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let a = |k: &str| v[k].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect::<Vec<_>>();
    (a("distance"), a("ln_prior"))
}

fn sample_points(dm: &DistanceMarginalization, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // deterministic pseudo-random points over the stress domain
    let pr = dm.prior();
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut uni = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let (mut rm, mut ro, mut dr) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
    for _ in 0..n {
        let rho = 10f64.powf(-1.0 + 4.0 * uni());
        let ds = pr.d_min / 5.0 * (25.0 * pr.d_max / pr.d_min).powf(uni());
        rm.push(rho);
        ro.push(rho * ds / 1000.0);
        dr.push(1000.0);
    }
    (rm, ro, dr)
}

fn benches(c: &mut Criterion) {
    let (d, lp) = load_prior("usf_100_5000");
    for phase in [false, true] {
        let dm = DistanceMarginalization::new(&d, &lp, phase, Config::default()).unwrap();
        let tag = if phase { "phase" } else { "nophase" };
        // one representative point per regime
        let pts = [
            (Regime::HighSnr, 200.0, 200.0 * 800.0 / 1000.0),
            (Regime::PriorDominated, 0.5, 0.02),
            (Regime::Quadrature, 8.0, 8.0 * 900.0 / 1000.0),
        ];
        for (reg, r, o) in pts {
            assert_eq!(dm.regime(r, o, 1000.0), reg);
            c.bench_function(&format!("scalar/{tag}/{}", reg.name()), |b| {
                b.iter(|| dm.ln_likelihood_ratio(black_box(r), black_box(o), black_box(1000.0)))
            });
        }
        for n in [1_000usize, 1_000_000] {
            let (rm, ro, dr) = sample_points(&dm, n);
            let mut out = vec![0.0; n];
            let mut g = c.benchmark_group(format!("batch/{tag}"));
            if n >= 1_000_000 {
                g.sample_size(10);
            }
            g.bench_function(format!("serial/{n}"), |b| {
                b.iter(|| dm.ln_likelihood_ratio_batch_serial(&rm, &ro, &dr, &mut out).unwrap())
            });
            #[cfg(feature = "parallel")]
            g.bench_function(format!("rayon/{n}"), |b| {
                b.iter(|| dm.ln_likelihood_ratio_batch(&rm, &ro, &dr, &mut out).unwrap())
            });
            g.finish();
        }
    }
    c.bench_function("construct/usf_100_5000", |b| {
        b.iter_batched(|| (), |_| DistanceMarginalization::new(&d, &lp, false, Config::default()).unwrap(), BatchSize::SmallInput)
    });
}

criterion_group!(group, benches);
criterion_main!(group);
