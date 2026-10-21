# distmarg Python benchmark report

## Accuracy vs reference quadrature (stress grid)

| prior | phase | regime | N | max abs err | 99% | 50% |
|---|---|---|---|---|---|---|
| usf_100_5000 | False | high_snr | 915 | 3.2e-05 | 1.2e-05 | 1.9e-09 |
| usf_100_5000 | False | prior_dominated | 1195 | 4.5e-06 | 3.7e-07 | 7.9e-15 |
| usf_100_5000 | False | quadrature | 1490 | 1.0e-11 | 8.1e-13 | 4.7e-15 |
| usf_100_5000 | False | **all** | 3600 | **3.2e-05** | 2.8e-06 | 8.0e-15 |
| usf_100_5000 | True | high_snr | 491 | 4.3e-05 | 2.3e-05 | 1.9e-09 |
| usf_100_5000 | True | prior_dominated | 592 | 1.9e-07 | 6.5e-08 | 7.8e-15 |
| usf_100_5000 | True | quadrature | 717 | 4.1e-12 | 4.8e-13 | 4.2e-15 |
| usf_100_5000 | True | **all** | 1800 | **4.3e-05** | 5.6e-06 | 7.8e-15 |
| usf_10_50000 | False | high_snr | 758 | 3.4e-05 | 1.4e-05 | 9.3e-10 |
| usf_10_50000 | False | prior_dominated | 985 | 1.3e-09 | 5.0e-11 | 8.8e-16 |
| usf_10_50000 | False | quadrature | 1857 | 3.8e-08 | 3.4e-10 | 1.8e-15 |
| usf_10_50000 | False | **all** | 3600 | **3.4e-05** | 8.0e-07 | 1.8e-15 |
| usf_10_50000 | True | high_snr | 479 | 4.6e-05 | 2.5e-05 | 1.9e-09 |
| usf_10_50000 | True | prior_dominated | 489 | 1.8e-11 | 7.0e-12 | 7.4e-16 |
| usf_10_50000 | True | quadrature | 832 | 1.8e-08 | 2.6e-09 | 1.3e-15 |
| usf_10_50000 | True | **all** | 1800 | **4.6e-05** | 5.2e-06 | 1.3e-15 |

## Speed (Python, this machine)

Set-up (fit + normalization + moments, 4096-point prior): 6.5 ms

| backend | phase | batch | time/call | ns/eval |
|---|---|---|---|---|
| numpy | False | 1 | 0.545 ms | 545378 |
| numpy | False | 1000 | 3.629 ms | 3629 |
| numpy | False | 1000000 | 3030.173 ms | 3030 |
| jax-cpu jit | False | 1 | 0.031 ms | 31177 |
| jax-cpu jit | False | 1000 | 0.993 ms | 993 |
| jax-cpu jit | False | 100000 | 37.490 ms | 375 |
| numpy | True | 1 | 0.673 ms | 673425 |
| numpy | True | 1000 | 6.616 ms | 6616 |
| numpy | True | 1000000 | 5579.173 ms | 5579 |
| jax-cpu jit | True | 1 | 0.038 ms | 38009 |
| jax-cpu jit | True | 1000 | 1.709 ms | 1709 |
| jax-cpu jit | True | 100000 | 71.863 ms | 719 |

Per-regime numpy cost (batch 10⁴, no phase):

- high_snr: 447 ns/eval
- prior_dominated: 286 ns/eval
- quadrature: 4531 ns/eval

## Comparison with Bilby's lookup table

Posterior-like samples: true d ~ prior-ish in [200, 4000] Mpc, network SNR ρ₀ ∈ {8, 15, 30, 100}, ρ_mf ~ ρ₀ + N(0,1), ρ_opt at the true distance ~ ρ₀(1 + 0.1 N(0,1)).  Errors are against the dense reference quadrature with the same (Bilby-cosmology) prior.

| phase | ρ₀ | N | distmarg max | distmarg 99% | bilby max | bilby 99% | bilby median | distmarg ns/eval | bilby ns/eval |
|---|---|---|---|---|---|---|---|---|---|
| False | 8 | 400 | 1.4e-14 | 1.4e-14 | 2.5e-04 | 2.3e-04 | 1.2e-04 | 5874 | 326 |
| False | 15 | 400 | 2.8e-14 | 2.8e-14 | 1.9e-03 | 1.2e-03 | 3.1e-04 | 6119 | 345 |
| False | 30 | 400 | 1.1e-13 | 5.7e-14 | 3.2e-02 | 9.8e-03 | 9.8e-04 | 5917 | 351 |
| False | 100 | 400 | 4.2e-07 | 4.0e-07 | 3.3e-01 | 1.5e-01 | 1.0e-02 | 4661 | 377 |
| True | 8 | 400 | 1.4e-14 | 1.1e-14 | 5.9e-04 | 2.5e-04 | 1.3e-04 | 10002 | 272 |
| True | 15 | 400 | 2.8e-14 | 2.8e-14 | 1.4e-03 | 6.8e-04 | 2.9e-04 | 10508 | 293 |
| True | 30 | 400 | 1.1e-13 | 5.7e-14 | 2.8e-02 | 1.1e-02 | 9.5e-04 | 10501 | 310 |
| True | 100 | 400 | 1.0e-06 | 5.9e-07 | 7.9e-01 | 4.7e-02 | 1.0e-02 | 7017 | 335 |
