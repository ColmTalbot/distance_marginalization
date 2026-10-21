# Implementation notes: `DistanceMarginalization` class

These notes are the original specification for both implementations. `SPEC_DECISIONS.md` records the details this document left open and every place where the implementations depart from it. Numbers labelled "exploratory" come from an early numpy-only study that preceded the implementations.

---

## 0. Goal

Build **two completely independent implementations** of the same specification:

1. **Python (array API).** This is the free-standing class described below.
2. **Rust.** This is a separate, pure-Rust crate with its own quadrature, fitting, special functions and tests (see §10). It has **no Python bindings**.

They must share no code and no precomputed tables. Each one builds its own internal representation from the same input arrays, so that each acts as a check on the other.

Each implementation:

- **is built from** a pair of arrays, the luminosity distance and the natural-log prior density, plus a `phase_marginalization: bool` flag. It has no dependency on Bilby. A thin optional helper converts a Bilby prior into these arrays (see §3);
- **evaluates** $\ln(\mathcal L_D/\mathcal Z_N)$ from the matched-filter SNR, the optimal SNR and the reference distance at which they were computed;
- **supports the Python array API** (numpy, JAX, torch, cupy via `array_api_compat`);
- **uses JAX-compatible control flow**: no Python branching on array values, no boolean-mask indexing and no in-place mutation in the evaluation path, so `jit`, `vmap` and `grad` work.

The primary target prior is `bilby.gw.prior.UniformSourceFrame` on luminosity distance. Any smooth, strictly positive 1D density on a bounded interval should also work.

## 1. Interface (proposal)

```python
class DistanceMarginalization:
    def __init__(self, distance, ln_prior, phase_marginalization: bool = False, *,
                 fit_degree: int | None = None,  # Chebyshev degree for ln pi_v(s); None = auto (§3)
                 fit_tol: float = 1e-8,          # max allowed fit residual in ln pi before warning/raising
                 rho_high: float = 30.0,        # high-SNR analytic branch threshold on |rho_mf|
                 kappa2_low: float = 0.05,      # prior-dominated branch threshold (see §4.2)
                 cumulant_order: int = 3,
                 table_shape=(64, 64),          # middle-regime representation
                 ):
        ...  # all set-up is host-side numpy float64 (not traced)

    @classmethod
    def from_bilby_prior(cls, prior, n=4096, **kwargs):  # optional helper, imports bilby lazily
        ...

    def ln_likelihood_ratio(self, rho_mf, rho_opt, d_ref):
        """ln(L_D / Z_N). Arrays broadcast; namespace taken from inputs."""

    def distance_posterior(self, rho_mf, rho_opt, d_ref, n_grid=...):   # optional
    def sample_distance(self, rho_mf, rho_opt, d_ref, rng_or_key):       # optional, for reconstruction
```

**Prior arrays.**

- `distance` must be finite, strictly increasing and > 0. Its endpoints define the support, $[d_{\min},d_{\max}]=$ `[distance[0], distance[-1]]`.
- `ln_prior` gives $\ln\pi(d_L)$ at those points. It must be finite at every point, since a zero density is not allowed anywhere, including the edges. It need not be normalized; the class normalizes it internally.
- Raise `ValueError` if either condition is violated.
- Accept the arrays from any namespace, and convert them to numpy float64 for set-up.

**Input conventions.** Write these into the docstring.

- `rho_opt` $=\sqrt{\langle h|h\rangle}$ evaluated at `d_ref`, so it is > 0.
- `rho_mf` is evaluated at `d_ref`, but it is independent of distance:
  - without phase marginalization: $\mathrm{Re}\langle d|h\rangle/\rho_{\rm opt}$, which can be **negative**;
  - with phase marginalization: $|\langle d|h\rangle|/\rho_{\rm opt}\ge0$. The class should take `abs` defensively.
- The output excludes the noise log-likelihood $\ln\mathcal Z_N$. Bilby adds that separately.
- Bilby currently passes `d_inner_h` and `optimal_snr_squared` at its reference distance. The conversion is `rho_mf = d_inner_h / sqrt(optimal_snr_squared)` and `rho_opt = sqrt(optimal_snr_squared)`. Check the exact Bilby attribute names (for example `_ref_dist` and `_interp_dist_margd_loglikelihood`) in the installed version before writing comparison tests.

## 2. Mathematics (authoritative spec)

**Definitions.** Use $v=1/d_L$ and the prior density in $v$, $\pi_v(v)=\pi_d(1/v)/v^2$, on $[v_{\min},v_{\max}]=[1/d_{\max},1/d_{\min}]$.

$$
v_\star=\frac{\rho_{\rm mf}}{d_{\rm ref}\,\rho_{\rm opt}},\qquad
\sigma_v=\frac{1}{d_{\rm ref}\,\rho_{\rm opt}},\qquad(\;v_\star=\rho_{\rm mf}\sigma_v\;)
$$

**Exact identity.**

$$
\ln\frac{\mathcal L_D}{\mathcal Z_N}=\frac{\rho_{\rm mf}^2}{2}+\ln\mathcal I,\qquad
\mathcal I=\int_{v_{\min}}^{v_{\max}}\mathrm dv\,\pi_v(v)\,w(v)\,e^{-(v-v_\star)^2/2\sigma_v^2}
$$

The weight $w(v)$ is:

- without phase marginalization: $w=1$;
- with phase marginalization: $w(v)=\bar I_0(c\,v)$, where $c=\rho_{\rm mf}\rho_{\rm opt}d_{\rm ref}=|\langle d|h\rangle|\,d_{\rm ref}$ and $\bar I_0(x)=e^{-x}I_0(x)$. Equivalently $c\,v=\rho_{\rm mf}^2 v/v_\star$, but use the $c$ form because it is regular when $v_\star\to0$.

**Factorization**, used in the high-SNR and table branches.

$$
\ln\mathcal I=\underbrace{\tfrac12\ln2\pi+\ln\sigma_v+\ln[\Phi(\beta)-\Phi(\alpha)]}_{\ln M\ (\text{TN mass})}
+\underbrace{\ln\pi_v(\bar v)+\ln w(\bar v)}_{\text{prior·weight at TN mean}}+\ln R
$$

- $\alpha=(v_{\min}-v_\star)/\sigma_v$ and $\beta=(v_{\max}-v_\star)/\sigma_v$.
- $\bar v$ and $\mathrm{Var}_{\rm TN}$ are the mean and variance of $\mathcal N(v_\star,\sigma_v)$ truncated to $[v_{\min},v_{\max}]$, with $Z=\Phi(\beta)-\Phi(\alpha)$:
  - $\bar v=v_\star+\sigma_v(\phi(\alpha)-\phi(\beta))/Z$
  - $\mathrm{Var}=\sigma_v^2[1+(\alpha\phi(\alpha)-\beta\phi(\beta))/Z-((\phi(\alpha)-\phi(\beta))/Z)^2]$
  - Compute the ratios as $\exp(\ln\phi(\cdot)-\ln Z)$.
  - Clip $\bar v$ to $[v_{\min},v_{\max}]$ and $\mathrm{Var}$ to $\ge0$.

## 3. Prior handling (set-up, host-side numpy)

### 3.0 Canonical interpretation of the input arrays (shared spec, must match exactly in Python and Rust)

The two implementations are independent, so they must agree on **which continuous prior** a pair of arrays represents. Otherwise differences between their interpolants will look like bugs. The canonical definition is:

1. **Transform the points.** For each input point, compute $s_i=-\ln d_i$ and $g_i=\ell_i+2\ln d_i$, where $\ell_i$ is `ln_prior`. This gives $\ln\pi_v$, unnormalized.
2. **Map to $[-1,1]$:** $t=(2s-s_a-s_b)/(s_b-s_a)$, with $s_a=-\ln d_{\max}$ and $s_b=-\ln d_{\min}$.
3. **Fit.** $\tilde g(t)=\sum_{k=0}^{K}c_kT_k(t)$ is the **unweighted least-squares** Chebyshev fit to the points $(t_i,g_i)$.
   - Solve it via QR or SVD of the Vandermonde matrix, not via normal equations.
   - Require `len(distance) ≥ 2(K+1)`.
   - Degree selection when `fit_degree=None`: the smallest $K$ in $\{8,16,24,\dots,128\}$ whose max abs residual is below `fit_tol`. If none qualifies, use $K=128$ and **warn**. A large residual means the prior is not smooth, or the tabulation is noisy (for example, piecewise-linear interpolant artefacts). In that case the high-SNR curvature model is unreliable.
4. **Normalize.** Compute $\ln Z_\pi=\ln\int_{s_a}^{s_b}e^{\tilde g(s)}e^{s}\,\mathrm ds$ with Gauss–Legendre (≥ 256 nodes, or adaptive), since $\mathrm dv=e^s\mathrm ds$. Then subtract it: $g=\tilde g-\ln Z_\pi$.
5. **Precompute.** Derivative coefficients come from the Chebyshev derivative recurrence. Moments are $\mu_k=\int e^{g(s)}e^{(k+1)s}\mathrm ds$, also by Gauss–Legendre.

**Cross-implementation check:** the fitted $c_k$, $\ln Z_\pi$ and $\mu_k$ must agree to ~1e-12 relative between Python and Rust. Write them to a fixture file and test both against it.

**Choice of input grid.** Callers should pass dense, smooth tabulations, about 2–4k points log-spaced in $d$. Do not pass the knots of a coarse linear interpolant.

### 3.1 Optional Bilby helper (Python only; `from_bilby_prior`)

- For `UniformSourceFrame` and other `bilby.gw.prior.Cosmological` subclasses, **do not use `prior.prob`**. As far as I recall, Bilby builds these as interpolated priors on a finite grid, which would show up as fit residual. *Verify in the installed Bilby.* Instead, recompute the density on a log-spaced grid from `prior.cosmology` (astropy):
  - $\pi(z)\propto(1+z)^{-1}\,\mathrm dV_c/\mathrm dz$
  - $\pi_d=\pi(z)/(\mathrm dd_L/\mathrm dz)$, with $\mathrm dd_L/\mathrm dz=D_C+(1+z)D_H/E(z)$
  - The helper should use the prior's own cosmology rather than a hard-coded one.
- Handle the prior possibly being defined in `redshift` or `comoving_distance` rather than `luminosity_distance`. Convert the support and include the Jacobian, or raise.
- For other priors, evaluate `prior.ln_prob` on the grid.
- The helper returns arrays and passes them to the constructor; the core class never imports Bilby.
- Provide the same helper as a tiny script (`tools/export_prior.py`) that writes `distance`, `ln_prior` to JSON (or two-column CSV) for the Rust side. Use JSON or CSV for all fixtures, not `.npy`, so Rust needs only `serde_json`.

### 3.2 Using the fit

1. **Fit a Chebyshev series to $g(s)$** on $[\ln v_{\min},\ln v_{\max}]$, as specified in §3.0. This gives $g$, $g'$ and $g''$ analytically.
   - Store the coefficients only.
   - Evaluate with a Clenshaw recurrence written in array-API ops (a short loop over a fixed, Python-int number of coefficients is fine under `jit`).
   - In $v$: $\pi_v''/\pi_v=(g''-g'+g'^2)/v^2$ and $(\ln\pi_v)'=g'/v$.
3. **Prior moments** $\mu_k=\mathbb E_\pi[v^k]$ for $k\le2\times$`cumulant_order` (≤ 8). Use Gauss–Legendre in $s$ with ~200 nodes, or the Chebyshev form. Require $d_{\min}>0$.
4. **Handy diagnostics:** the local power-law index of $\pi(d_L)$ for USF. It is +1.89 / +1.44 / +0.33 at 100 / 707 / 5000 Mpc, and −1.32 at 50 Gpc.

## 4. Evaluation: three regimes, selected with `where`

Compute every branch with **safe inputs** and select between them using `xp.where`. Use the "double-where" trick so that NaN or inf in unselected branches does not poison gradients: replace the inputs to each branch with benign values where that branch is not selected, before computing it.

### 4.1 High SNR: $|\rho_{\rm mf}|\ge\rho_{\rm high}$ (default 30)

$$
\ln R\approx\tfrac12\,\mathrm{Var}_{\rm TN}\;\big[h''+h'^2\big]_{v=\bar v},\qquad h(v)=\ln\pi_v(v)+\ln w(v).
$$

For the phase case, $\ln w=\ln\bar I_0(cv)$ and $(\ln w)'=c\,[I_1/I_0(cx)-1]$; more simply, use the large-argument form $\ln\bar I_0(x)\approx-\tfrac12\ln(2\pi x)+\ldots$, since $cv\gtrsim\rho^2\ge900$ in this branch.

**Exploratory results** (USF; model without the $w$-curvature term), max |error| in $\ln\mathcal L_D$ for $\rho\ge30$ over $d_\star\in[d_{\min}/5,5d_{\max}]$:

| Prior | No phase | Phase |
|---|---|---|
| [100, 5000] Mpc | 3.3e-4 | 2.3e-3 |
| [10, 5e4] Mpc | 1.0e-3 | 3.1e-3 |

- I expect the phase errors to drop once the $w$-curvature is included. **This needs verifying.**
- The error scales as $\rho^{-2}$ in the interior, for example −1.8e-4 at ρ=30, −2e-5 at 100 and −2e-7 at 10³ for $d_\star=707$ Mpc. It is at the reference-quadrature floor (~1e-7) outside the support.
- The negative-$\rho_{\rm mf}$ case (no phase) is automatically "outside the support" and is handled exactly by the TN mass. Test it explicitly.

### 4.2 Prior-dominated: cumulant series

Write $\ln\ell(v)=Av-\tfrac12Bv^2$ with $A=\rho_{\rm mf}\rho_{\rm opt}d_{\rm ref}$ and $B=\rho_{\rm opt}^2d_{\rm ref}^2$ (no phase). Then

$$
\ln\frac{\mathcal L_D}{\mathcal Z_N}=\sum_{n\ge1}\frac{\kappa_n}{n!},\qquad
\mathbb E[(\ln\ell)^n]=\sum_{j=0}^{n}\binom nj A^{n-j}(-B/2)^j\,\mu_{n+j}.
$$

The cumulants $\kappa_n$ follow from these raw moments. This is a closed-form polynomial, so no table is needed.

- **Selection criterion:** use $\kappa_2=\mathrm{Var}_\pi[\ln\ell]<$ `kappa2_low`. It is computed from the same moments and is a better control variable than $\rho_{\rm opt}(d_{\min})$. Calibrate the threshold by benchmark.
- **Exploratory accuracy at O3** (max over $\rho_{\rm mf}\in[-3,5]$):

| $\rho_{\rm opt}(d_{\min})$ | [100, 5000] Mpc | [10, 5e4] Mpc |
|---|---|---|
| 0.01 | 2e-11 | 9e-13 |
| 0.1 | 3e-7 | 5e-11 |
| 0.5 | 2e-4 | 4e-8 |
| 1 | 4e-3 | 7e-7 |
| 2 | 0.1 | 2e-5 |
| 5 | fails | 4e-3 |

- The series is **asymptotic**: at large $\kappa_2$, going to higher order stops helping (O1→O4 error 2.7→1.8). Do not chase convergence.
- **Phase case:** $\ln\ell$ is no longer polynomial in $v$, because $\ln\bar I_0(cv)+cv=\ln I_0(cv)$. Use the small-argument expansion $\ln I_0(x)=x^2/4-x^4/64+\ldots$, which keeps it polynomial in $v$: replace $A v$ by the series in $(cv)^2$ and reuse the moments up to the required order. **Derive and verify this; the exploratory study did not.**

### 4.3 Middle regime: a compact 2D representation of the residual

This covers everything not in 4.1 or 4.2.

- **Represent** $T=\ln\mathcal I-\ln M-\ln\pi_v(\bar v)-\ln w(\bar v)-\ln R_{\rm high}$, i.e. the correction left after the high-SNR model, so that $T\to0$ as $\rho\to\infty$.
- **Suggested axes:**
  - $x=\mathrm{asinh}(v_\star/v_{\min})=\mathrm{asinh}(d_{\max}/d_\star)$. It is smooth through $v_\star=0$ (negative $\rho_{\rm mf}$) and log-like elsewhere.
  - $y=\ln|\rho_{\rm mf}|$ on $[\ln\rho_{\rm low},\ln\rho_{\rm high}]$.
- **Representation:** a tensor-product Chebyshev interpolant (start at 32×32 and increase).
  - Use first-kind Chebyshev nodes on the clamped $(x,y)$ domain, with values from **that implementation's own** reference quadrature (§5), to relative tolerance 1e-12.
  - Compute the coefficients by explicit discrete cosine sums.
  - Fix the domain, node placement and `table_shape` in the spec, so that Python and Rust tables can also be compared coefficient by coefficient.
  - A bicubic uniform grid is the fallback.
- **Points outside the domain:** clamp $x,y$ to the domain and rely on the regime selection. Then **check continuity at every regime boundary**, with the jump in $\ln\mathcal L_D$ below the target tolerance.
- **Expected size of $T$:** exploratory calculations show $\ln R\in[0,6]$ for [100, 5000] Mpc and $[0,15]$ for [10, 5e4] Mpc over the full stress grid. The large values occur at low $\rho$, where 4.2 should take over. Inside the middle band, $T$ should be of order 0.01–1.
- **Is the 2D fit needed?** Possibly not. If the boundary between 4.1 and 4.2 can be pushed until they overlap with acceptable error, the 2D representation becomes unnecessary. Gauss–Hermite quadrature in $v$ about $\bar v$ is another option: about 10–20 nodes in a fixed-size loop, with no table, which also supports non-separable priors. Benchmark both.

## 5. Reference quadrature (tests and table construction; numpy, float64)

Integrate in $v$ on a composite grid that contains:

- a global log-spaced grid;
- a dense grid within $\pm12\sigma_v$ of the clipped peak;
- when the peak lies outside the support, a dense grid over the tail scale $\lambda=\sigma_v^2/|v_\star-v_{\rm edge}|$ next to the edge.

The tail grid is **essential**. An earlier version without it reported spurious $\ln R\sim1$ at $\rho\ge10^3$. Cross-check a subset against `scipy.integrate.quad` or `mpmath.quad` with explicit breakpoints.

## 6. Array API and JAX constraints

- `xp = array_api_compat.array_namespace(rho_mf, rho_opt, d_ref)`. Convert the stored numpy coefficients with `xp.asarray(coeffs, dtype=...)`, cached per namespace and dtype.
- The standard has no `erfc`, `log_ndtr` or `i0e`. Add a `_special.py` dispatch:
  - JAX: `jax.scipy.special.log_ndtr`, `i0e`, `i1e`
  - numpy/cupy: `scipy.special` (array-API-aware in recent SciPy)
  - torch: `torch.special.log_ndtr`, `i0e`, `i1e`
  - a pure array-API fallback: rational erfc approximation plus asymptotic tails; for $\bar I_0$, polynomial approximations as in Numerical Recipes / Cephes.
- **Stable helpers:**
  - `log_diff_ndtr(a, b)` $=\ln(\Phi(b)-\Phi(a))$: use the reflection $(a,b)\to(-b,-a)$ when $a>0$, then `lb + log1mexp(la - lb)`.
  - `log1mexp(x)` $=$ `where(x > -ln2, log(-expm1(x)), log1p(-exp(x)))`.
  - These lose accuracy when $\beta-\alpha\ll1$ (very wide Gaussian), but the 4.2 branch covers that case. Still keep the outputs finite.
- **No Python control flow on array values.** Regime choice is `where`; loops have static length (Chebyshev degree, cumulant order).
- **dtype.** Require or recommend float64: $\rho^2/2$ at $\rho=10^4$ is 5e7, and float32 then has an absolute error of about 4 in $\ln\mathcal L$. With JAX, document that `jax_enable_x64` is needed, and warn if float32 is passed.
- **Gradients.** These must be finite everywhere, including at $\rho_{\rm mf}=0$, at regime boundaries, and at $v_\star$ exactly on a prior edge. Add `jax.grad` tests against finite differences.

## 7. Tests (pytest)

1. **Identity checks:**
   - $\ln\mathcal L_D$ against direct quadrature of $\int\pi(d)\exp(\ldots)\,\mathrm dd$ in the original variables, for both flags, including known values for a $d^2$ prior.
   - `d_ref` invariance: rescale `d_ref`→`k·d_ref` and `rho_opt`→`rho_opt/k`; the result must be unchanged.
2. **Stress-grid accuracy** for USF on [100, 5000] and [10, 5e4] Mpc, both flags:
   - $d_\star\in[d_{\min}/5,5d_{\max}]$, $\rho_{\rm mf}\in[10^{-3},10^4]$ including negative values (no phase), and $\rho_{\rm opt}(d_{\min})\in[10^{-3},10^3]$;
   - target max $|\Delta\ln\mathcal L|\le10^{-3}$ (configurable).
3. **Limits:** $\rho_{\rm mf}\to0$ with $\rho_{\rm opt}\to0$ gives $\ln\mathcal L_D\to0$; large $\rho$ matches Laplace.
4. **Regime-boundary continuity:** $|\Delta|$ < tolerance across each switch, for both value and gradient.
5. **Backends:** numpy and JAX at minimum, with `jit`, `vmap` and `grad`; torch if available. Inputs given as scalars, 0-d arrays and broadcast shapes.
6. **Comparison with Bilby's current lookup table** for the same prior, on posterior samples from a real or injected run. Report $\Delta\ln\mathcal L$ distributions and timing.
7. **Oracle fixtures,** shared data generated by a third, independent code path:
   - `tools/make_fixtures.py` uses **mpmath** at about 30 digits, with explicit breakpoints.
   - It writes `fixtures/oracle_*.json` containing input prior arrays, query points $(\rho_{\rm mf},\rho_{\rm opt},d_{\rm ref})$, the regime label, and $\ln\mathcal L_D$ for both flags.
   - It also writes the reference special-function values: $\ln\Phi$, $\ln(\Phi(b)-\Phi(a))$, $\ln\bar I_0$, $\bar I_1/\bar I_0$, with arguments spanning $[-40,40]$ and $[10^{-6},10^6]$.
   - Python and Rust both test against these fixtures. They are data, not code, so independence is preserved.
8. **Prior-fit fixtures:**
   - $c_k$, $\ln Z_\pi$ and $\mu_k$ for the USF priors, a $d^2$ prior and a $1/d$ prior, from both implementations; they must agree to ~1e-12.
   - A deliberately noisy or kinked prior that must trigger the fit-residual warning.
9. **Cross-implementation check** (`tools/cross_check.py`): evaluate both on the full stress grid. The comparison is file-based: Python writes `fixtures/stress_queries.json`, the Rust `distmarg-eval` binary (§10.4) writes its results to JSON, and the script compares them with the Python results.
   - Away from regime boundaries the two must agree to ~1e-9, since the formulas are identical.
   - In the table regime they must agree to the table tolerance.
   - Report the worst cases.

## 8. Benchmarks

- **Accuracy:** max and quantiles of $|\Delta\ln\mathcal L|$ on the stress grid and on realistic samples.
- **End-to-end checks:** reweight a run to the reference and report efficiency and $\Delta\ln\mathcal Z$; P–P tests at network SNR 8, 30, 100 and 1000; JS divergence of the reconstructed $d_L$.
- **Speed:**
  - time per call at batch sizes 1, 1e3 and 1e6, for numpy and for JAX CPU/GPU (after jit warm-up);
  - set-up time and memory;
  - compare against Bilby's table and against a single waveform plus inner-product evaluation.
- **Presentation:** a Pareto plot of error vs time for each method, split by regime.

## 9. Open items / risks

- High-SNR phase model: add the $w$-curvature term and verify (§4.1).
- Low-SNR phase branch: derive and verify the series in $(cv)^2$ (§4.2).
- Calibrate `rho_high`, `kappa2_low` and the middle-regime domain by benchmark. Check whether the middle 2D fit is needed at all.
- Check how Bilby's Cosmological priors evaluate `prob`, and the attribute names used for comparison tests.
- Distance-posterior reconstruction: inverse-CDF on a grid in $v$ concentrated around $\bar v\pm k\sqrt{\mathrm{Var}_{\rm TN}}$, plus the global grid. Low priority.
- Time marginalization: the class should accept arrays over time bins, and the caller does a log-sum-exp. Document this.

## 10. Rust implementation (independent)

### 10.1 Scope and independence rules

- It implements the same spec (§2–§4, with §3.0 canonical prior interpretation) from scratch.
- It does **not** call the Python code, does **not** load Python-built tables, and does **not** port the Python source line by line. Where a sensible alternative exists, it uses a *different* numerical method:
  - **Reference quadrature** (§5): adaptive Gauss–Kronrod (G7/K15 or G10/K21) with explicit breakpoints at $\hat v$, the support edges, $\hat v\pm k\sigma_v$, and the edge tail scale $\lambda$. Python uses composite trapezoid.
  - **Least-squares fit:** Householder QR written in-crate. Python uses numpy `lstsq`/QR.
  - **Gauss–Legendre nodes:** Newton iteration on $P_n$, or Golub–Welsch. Python uses `numpy.polynomial.legendre.leggauss`.
- The only shared artifacts are the spec (this file) and the data fixtures (§7.7–7.8).

### 10.2 Crate layout

```
rust/
  Cargo.toml                 # workspace
  distmarg/                  # core library, no Python deps
    src/lib.rs               # pub API, Config, Regime, Error
    src/prior.rs             # array validation, §3.0 transform/fit/normalize, moments
    src/chebyshev.rs         # Clenshaw (value, d1, d2), derivative recurrence, 2D tensor eval
    src/linalg.rs            # Householder QR least squares
    src/special.rs           # log_ndtr, log_diff_ndtr, log1mexp, i0e, i1e
    src/quadrature.rs        # Gauss–Legendre nodes; adaptive Gauss–Kronrod with breakpoints
    src/regimes.rs           # high-SNR, cumulant, table branches; selection
    src/table.rs             # middle-regime 2D Chebyshev build (uses quadrature.rs)
    tests/                   # integration tests vs fixtures/
    benches/                 # criterion benchmarks
    src/bin/distmarg_eval.rs # small CLI for file-based cross-checks (§10.4); no Python involved
```

**Dependencies.**

- Core: `thiserror`, and optionally `rayon` behind a `parallel` feature.
- Erfc: `libm` is acceptable as the base for erfc. Otherwise implement Cody's rational approximations.
- Bessel functions: port the Cephes Chebyshev coefficient tables for $\bar I_0$ and $\bar I_1$ (check the licence; Cephes is permissive).
- CLI: `serde`, `serde_json`, and optionally `clap`. The core library must not depend on these. Keep them behind a `cli` feature, or as `[[bin]]`-only dependencies.
- Dev: `serde`, `serde_json` (fixtures), `approx`, `proptest`, `criterion`.
- Avoid heavy numerical crates in the core, so that it stays small, auditable and independent.

### 10.3 API

```rust
pub struct Config {
    pub fit_degree: Option<usize>, pub fit_tol: f64,
    pub rho_high: f64, pub kappa2_low: f64, pub cumulant_order: usize,
    pub table_shape: (usize, usize),
}
impl Default for Config { /* same defaults as Python */ }

pub enum Regime { HighSnr, PriorDominated, Table }

pub struct DistanceMarginalization { /* coefficients, moments, table, flags; immutable after new() */ }

impl DistanceMarginalization {
    pub fn new(distance: &[f64], ln_prior: &[f64], phase_marginalization: bool, cfg: Config)
        -> Result<Self, Error>;                       // validation errors, fit warnings via FitInfo
    pub fn ln_likelihood_ratio(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> f64;
    pub fn ln_likelihood_ratio_batch(&self, rho_mf: &[f64], rho_opt: &[f64], d_ref: &[f64],
                                     out: &mut [f64]) -> Result<(), Error>; // equal lengths; rayon if enabled
    pub fn regime(&self, rho_mf: f64, rho_opt: f64, d_ref: f64) -> Regime; // diagnostics/tests
    pub fn fit_info(&self) -> &FitInfo;  // degree, max residual, coeffs, ln_norm, moments (for §7.8)
}
```

**Requirements.**

- `DistanceMarginalization` must be `Send + Sync`.
- `ln_likelihood_ratio` performs no allocation and never panics. NaN in gives NaN out. $\rho_{\rm opt}\le0$ or $d_{\rm ref}\le0$ returns NaN; document this.
- With phase marginalization, use $|\rho_{\rm mf}|$.
- The scalar path is branchy, which is fine in Rust. It must use **exactly the same regime-selection rule** as Python, including tie-breaking at equality, so that the cross-check is tight.

### 10.4 File-based evaluation tool (`distmarg-eval`)

There are no Python bindings. Cross-checks against Python go through files.

```
distmarg-eval --prior fixtures/priors/usf_100_5000.json [--phase] [--config cfg.json] \
              --queries fixtures/stress_queries.json --out results_rust.json [--fit-info fit_rust.json]
```

- **Prior file:** `{"distance": [...], "ln_prior": [...]}`.
- **Query file:** `{"rho_mf": [...], "rho_opt": [...], "d_ref": [...]}`.
- **Output:** `{"ln_likelihood_ratio": [...], "regime": [...]}`, plus optional `FitInfo` (coefficients, `ln_norm`, moments, degree, residual).
- Print floats with full round-trip precision (`{:e}` with 17 significant digits, or `ryu`).
- Optional stretch goal: `ln_likelihood_ratio_and_grad`, returning analytic $\partial/\partial\rho_{\rm mf}$ and $\partial/\partial\rho_{\rm opt}$, checked against finite differences.

### 10.5 Rust tests and benchmarks

- **Unit tests:**
  - special functions vs oracle fixtures (target rel. error ≤ 1e-14, with absolute error in log space ≤ 1e-13);
  - Clenshaw derivatives vs finite differences;
  - QR least squares vs exact polynomial data;
  - Gauss–Legendre exactness for degree $2n-1$.
- **Integration tests:** oracle $\ln\mathcal L_D$ fixtures for both flags, and prior-fit fixtures (§7.8).
- **`proptest` properties:**
  - `d_ref` scaling invariance;
  - finiteness over the whole stress domain;
  - continuity across regime boundaries;
  - $\ln\mathcal L_D\to0$ as $\rho_{\rm mf},\rho_{\rm opt}\to0$.
- **`criterion` benchmarks:** single scalar call per regime; batch sizes 1e3 and 1e6, serial and rayon; construction time with and without the table build.
- Report ns/eval next to the Python numpy and JAX numbers (§8). They are timed separately; no cross-language calls.

## 11. Suggested repository layout

```
distance_marginalization/
  IMPLEMENTATION_NOTES.md
  python/distmarg/{__init__,core,_special,_chebyshev,_prior,_quadrature,bilby}.py
  python/tests/
  rust/                                      # §10.2
  fixtures/{priors/*.json, oracle_*.json, prior_fit_*.json, stress_queries.json}
  tools/{export_prior.py, make_fixtures.py, cross_check.py, benchmark.py}
```

**Suggested build order:**

1. Write `tools/export_prior.py` and `tools/make_fixtures.py` (the mpmath oracle).
2. Build the Python core.
3. Build the Rust core.
4. Write `cross_check.py`.
5. Calibrate the thresholds and decide whether the 2D table is needed.
6. Run the benchmarks (Python and Rust separately, with numbers collected into one report).
