# Spec decisions shared by the Python and Rust implementations

This file pins down every detail that `IMPLEMENTATION_NOTES.md` left open, so
that the two independent implementations evaluate *the same formulas*. Python
lives in `python/distmarg/` and Rust in `rust/distmarg/`. Where this file
differs from the notes, it says why.

## 1. Prior (as in notes §3.0, unchanged)

- $s=-\ln d$ and $g=\ell+2\ln d$. Map to $t\in[-1,1]$, then take the
  unweighted least-squares Chebyshev fit.
  - Degree selection: auto from $\{8,\dots,128\}$, considering only degrees with $n\ge2(K+1)$. Warn when `max_residual >= fit_tol`.
- $\ln Z_\pi$ uses 256-node Gauss–Legendre (GL) on $[s_a,s_b]$, with weights $e^{\tilde g+s}$.
- Reported moments are `moments[k]` $=E_\pi[v^k]$ for $k=0..8$, computed on the same 256 nodes.
- **Cumulant constants:** $\nu=v/v_{\max}=v\,d_{\min}$ and $z=\nu$ (no phase) or $z=\nu^2$ (phase).
  - Take the prior weights $P_i$ at the 256 GL nodes, renormalized so that $\sum P_i=1$.
  - $\bar z=\sum P_i z_i$ and $C_k=\sum P_i(z_i-\bar z)^k$ for $k=0..24$, with $C_0=1$ and $C_1=0$ set exactly.

## 2. Evaluation variables

The output is NaN when `rho_opt <= 0`, `d_ref <= 0` or any input is non-finite.

| Quantity | Definition |
|---|---|
| $\rho$ | `rho_mf`; $\lvert$`rho_mf`$\rvert$ with phase marginalization |
| $k$ | `rho_opt * d_ref` |
| $\sigma$ | $1/k$ |
| $v_\star$ | $\rho\sigma$ |
| $A$ | $\rho k$ |
| $B$ | $k^2$ |
| $A'$ | $A\,v_{\max}$ |
| $B'$ | $B\,v_{\max}^2$ |

Then $\ln\ell(v)=Av-Bv^2/2$; with phase marginalization, $\ln\ell(v)=\ln I_0(Av)-Bv^2/2$.

## 3. Truncated-normal moments (stable)

Let $\alpha=(v_{\min}-v_\star)/\sigma$ and $\beta=(v_{\max}-v_\star)/\sigma$.

- **Orientation.** If $\alpha+\beta<0$, flip: $(a,b)=(-\beta,-\alpha)$. Otherwise $(a,b)=(\alpha,\beta)$. Let $L=b-a$.
- **Tail case, $a\ge0$.** Shift to $t=x-a$, so the density is $\propto e^{-at-t^2/2}$ on $[0,L_c]$ with $L_c=\min(L,60)$.
  - The functions $m,u,w$ are $\int_0^\infty t^{0,1,2}e^{-at-t^2/2}\,\mathrm dt$:
    - $a<3$: $m=\exp(\ln\Phi(-a)+a^2/2+\tfrac12\ln2\pi)$, $u=1-am$, $w=m-au$;
    - $a\ge3$: Laplace continued fraction $D_k=k/(a+D_{k+1})$, with $D_{61}=0$ and $k=60..1$. Then $m=1/(a+D_1)$, $u=mD_1$ and $w=mD_1D_2$.
  - With $q=e^{-aL_c-L_c^2/2}$ and $b_c=a+L_c$:
    - $I_0=\max(m_a-qm_{b_c},\,L_cq)$
    - $I_1=u_a-q(u_{b_c}+L_cm_{b_c})$
    - $I_2=w_a-q(w_{b_c}+2L_cu_{b_c}+L_c^2m_{b_c})$
  - Results: mean$_x=a+I_1/I_0$, var$_x=I_2/I_0-(I_1/I_0)^2$, and $\ln(\sqrt{2\pi}Z)=-a^2/2+\ln I_0$.
- **Straddling case, $a<0$.** Compute $\ln Z$ with `log_diff_ndtr`. Then, with $r_{a,b}=\phi(a,b)/Z$:
  - mean$_x=r_a-r_b$
  - var$_x=1+ar_a-br_b-\text{mean}_x^2$
- **Undo the flip:** negate the mean if flipped. Then
  - $\bar v=\mathrm{clip}(v_\star+\sigma\,\text{mean}_x,v_{\min},v_{\max})$
  - $\mathrm{Var}=\min(\max(\text{var}_x,0)\sigma^2,(v_{\max}-v_{\min})^2/4)$
  - $\ln M=\ln(\sqrt{2\pi}Z)+\ln\sigma$

The naive formula $1+(\alpha\phi_\alpha-\beta\phi_\beta)/Z-(\cdot)^2$ suffers catastrophic cancellation in the tails. For example, $\alpha=10^4$ returns pure noise for the variance $\approx\sigma^2/\alpha^2$.

## 4. Regime selection (identical rule in both implementations)

```
# ln ell as a polynomial in z (§5):
#   no phase: p = [0, A', -B'/2]
#   phase:    p = [0, c1 A''^2 - B'/2, c2 A''^4, ..., c6 A''^12],  A'' = min(A', phase_series_xmax)
prior_dominated = kappa2(p) < kappa2_low
               && range(p) <= cumulant_range_max
               && (no phase || A' <= phase_series_xmax)
high_snr        = !prior_dominated && |rho| >= rho_high && sqrt(Var_TN) <= high_max_rel_width * vbar
quadrature      = otherwise
```

- `range(p)` $=\max(P(z_0),P(1),P(z_c))-\min(P(z_0),P(1))$.
  - $z_0=\nu_{\min}$ ($\nu_{\min}^2$ with phase).
  - $z_c=\mathrm{clip}(p_1/(2\max(-p_2,10^{-300})),z_0,1)$.
- `kappa2(p)` $=\sum_{ij}y_iy_jC_{i+j}$, with $y$ from §5.
- Regime codes are: `0 = high_snr`, `1 = prior_dominated`, `2 = quadrature`.

**Why the range condition, which is not in the notes:** κ₂ alone is not a safe
criterion. For the wide USF prior most prior mass sits at large $d$. A point
with $\rho_{\rm mf}=21.5$ and $d_\star=2.7$ Mpc then has κ₂ = 0.033, yet
$\ln\ell$ spans 107 nats over the support, and the cumulant series is off by
85 nats.

**Calibrated defaults.** Calibration used the reference quadrature on stress
grids for four priors and both flags (see `tools/benchmark.py`).

| Parameter | Value | Worst-case error of that branch on the stress grid |
|---|---|---|
| `rho_high` | 30 | |
| `high_max_rel_width` | 0.01 | ≤ 4e-5 |
| `kappa2_low` | 0.05 | |
| `cumulant_range_max` | 1.0 | |
| `cumulant_order` | 4 | ≤ 2e-5 |
| `phase_series_xmax` | 1.0 | |
| `quad_nodes` | 20 | ≤ 6e-5 (32 nodes: ≤ 3e-9) |
| `quad_grid` | 64 | adds < 1e-6 |

The notes defaulted to a relative width of 0.1 with $\rho\ge30$, which gives
errors up to 6.7 nats near the edges.

## 5. Prior-dominated branch: centred cumulant series

1. Re-expand $P(z)=\sum p_kz^k$ about $\bar z$: $\tilde p_j=\sum_{k\ge\max(j,1)}\binom kj\bar z^{k-j}p_k$.
2. $\kappa_1=\tilde p_0+\sum_{j\ge2}\tilde p_jC_j$.
3. The coefficients of $Y=\ln\ell-\kappa_1$ in $\delta=z-\bar z$ are $y_0=-\sum_{j\ge2}\tilde p_jC_j$ and $y_j=\tilde p_j$.
4. $\mu_n=\sum_k(Y^n)_kC_k$, where $Y^n$ is formed by polynomial convolution.
5. $\kappa_2=\mu_2$, $\kappa_3=\mu_3$ and $\kappa_4=\mu_4-3\mu_2^2$.
6. $\ln(\mathcal L_D/\mathcal Z_N)=\kappa_1+\kappa_2/2+\kappa_3/6+\kappa_4/24$, truncated at `cumulant_order`.

**Phase case** (the derivation the notes asked for). With phase marginalization,
$\ln\ell=\ln I_0(Av)-Bv^2/2$ exactly: the $e^{Av}$ factor and $\bar I_0$ combine into $I_0$. Using

$$
\ln I_0(x)=\sum_{k\ge1}c_kx^{2k},\qquad
c=\left(\tfrac14,-\tfrac1{64},\tfrac1{576},-\tfrac{11}{49152},\tfrac{19}{614400},-\tfrac{473}{106168320},\dots\right),
$$

$\ln\ell$ becomes a polynomial in $z=\nu^2$ with $p_k=c_kA'^{2k}$, and $p_1$ gets an extra $-B'/2$.

- The $c_k$ come from the power-series logarithm of $I_0=\sum (x^2/4)^k/(k!)^2$.
- The series has radius $j_{0,1}=2.405$. Keeping six terms with $A'\le1$ truncates at $|c_7|\le8\times10^{-7}$.
- This was verified against the oracle: worst error 1e-5 on the stress grid.

## 6. Middle regime: six-panel Gauss–Legendre quadrature (replaces the 2-D table)

The notes left open whether the 2-D table is needed (§4.3, "Is the 2D fit
needed? Possibly not… Gauss–Hermite… Benchmark both"). A fixed-node quadrature
turned out to be accurate to ≤3e-9 over the whole middle regime. It needs no
table build, so there is nothing to compare between implementations. **No 2-D
table is built**, and `table_shape` does not exist.

**Peak location and scale:**

- $c=\mathrm{clip}(v_\star,v_{\min},v_{\max})$
- $\Delta=|v_\star-c|$
- scale $=\sigma^2/\max(\sigma,\Delta)$. This is $\sigma$ inside the support and the edge tail length $\lambda=\sigma^2/\Delta$ outside it.

**Breakpoints in $s$:**

- $s_a$, then $\mathrm{clip}(\ln\mathrm{clip}(c+w\cdot\text{scale},v_{\min},v_{\max}),s_a,s_b)$ for $w\in(-40,-8,0,8,40)$, then $s_b$.
- This gives six panels, each with `quad_nodes` GL nodes (default 20: worst error 5.8e-5 on the calibration grids, against 3e-9 at 32 nodes and 1.0e-3 at 16).
- **ln π_v at the nodes** comes from a cubic-Hermite interpolant, not the Chebyshev series, when `quad_grid` $=N>0$ (default 64):
  - knots $s_j=s_a+jh$, $h=(s_b-s_a)/(N-1)$, with the last knot set to exactly $s_b$;
  - stored $g_j=\ln\pi_v(s_j)$ and $g'_j=\mathrm d\ln\pi_v/\mathrm ds$ from the Chebyshev series;
  - $u=(s-s_a)/h$, $i=\mathrm{clip}(\lfloor u\rfloor,0,N-2)$, $w=u-i$;
  - $(2w^3-3w^2+1)g_i+(w^3-2w^2+w)hg'_i+(3w^2-2w^3)g_{i+1}+(w^3-w^2)hg'_{i+1}$.

  This is C¹, so JAX gradients stay smooth. Its error is below 1e-6. In Rust it removes most of the cost of each node; in numpy it is no faster than `chebval` at degree 16, but faster at degree 32. `quad_grid = 0` uses the series directly.

**Integrand and result:**

$$
\ln\mathcal L=\mathrm{LSE}_i\big[\ln(h_jw_i)+g(s_i)+s_i+Av_i-Bv_i^2/2\;(+\ln\bar I_0(Av_i))\big]
$$

- $h_j$ is the panel half-width.
- The panels are combined **linearly**: $M+\ln\sum_j h_j\sum_i w_ie^{f_{ij}-M}$, where $M$ is the maximum of $f$ over all nodes, including those of zero-width panels. Do not add $\ln h_j$ inside the log-sum-exp. A zero-width panel must keep its derivative with respect to its breakpoints; otherwise the JAX gradient is wrong when $v_\star$ lies exactly on an edge, because `clip` then splits its gradient.
- Writing $\ln\ell$ as $Av-Bv^2/2$ avoids the cancellation in $\rho^2/2-(v-v_\star)^2/2\sigma^2$.

## 7. High-SNR branch (notes §4.1, with the $w$-curvature included)

$$
\ln\mathcal L=\rho^2/2+\ln M+g(\bar s)+\ln w(\bar v)+\tfrac12\mathrm{Var}\,[h''+h'^2]_{\bar v}
$$

- $h'=g'/\bar v+(\ln w)'$ and $h''=(g''-g')/\bar v^2+(\ln w)''$.
- With phase: $x=A\bar v$ and $r=\bar I_1/\bar I_0(x)$. Then:
  - $\ln w=\ln\bar I_0(x)$
  - $(\ln w)'=A(r-1)$
  - $(\ln w)''=A^2(1-r/x-r^2)$, with $r/x\to\tfrac12$ for $x<10^{-8}$.

Including the $w$-curvature brings the phase errors down to the same level as
the no-phase case: ≤ 4e-5 at the default thresholds, against 2.3e-3 in the prototype.

## 8. Python evaluation strategy

- **Backend.** When the `distmarg_rs` extension (the Rust crate with feature `python`) is importable, numpy and plain-Python float64/integer inputs are evaluated by Rust. Scalars go through a direct call; arrays are broadcast in Python and passed as contiguous 1-D arrays. Everything else uses the Python path described below, as do all inputs when the extension is missing. The pure-Python numpy path stays tested (oracle and stress grid run on both backends) and is the implementation that `tools/cross_check.py` compares with Rust.

- **Special functions** come from `scipy.special` (`log_ndtr`, `i0e`, `i1e`, `erfcx`), so JAX, torch and CuPy arrays need `SCIPY_ARRAY_API=1` set before SciPy is imported. The class raises a clear error otherwise. Only `log1mexp`, `log_diff_ndtr`, the Mills moments $u,w$ and the $\ln I_0$ series are local, because SciPy has no equivalents.
- The truncated-normal moments are local too, because neither `stats.truncnorm` nor `stats.truncate(Normal())` is usable: the first gives a negative variance at $a=10^4$; the second is 20% off there and is not JAX-traceable.
- The middle-regime quadrature reduces each panel with a plain max-shifted log-sum-exp rather than `scipy.special.logsumexp`. In SciPy 1.18 the latter returns NaN JAX gradients when every term except the maximum underflows.
- Chebyshev series use `numpy.polynomial.chebyshev` (`chebval`, `chebder`, `chebvander`), which works on JAX and torch arrays.

- **numpy:** the regime is computed first, and each branch is evaluated only on its subset.
- **Other namespaces (JAX, torch, …):** every branch is computed for every element, and the result is picked with `where`. Unselected elements get benign inputs (the "double where"), so that gradients stay finite.

Both strategies call the same branch functions.
