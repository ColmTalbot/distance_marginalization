# distmarg

Distance-marginalized gravitational-wave likelihood, `ln(L_D / Z_N)`, in the
`(rho_mf, v*)` basis. It is accurate to about 1e-4 in `ln L` over the whole
parameter space, needs no lookup table, and evaluates in the namespace of its
inputs: numpy, JAX (with `jit`, `vmap` and `grad`), torch or CuPy.

```bash
pip install distmarg            # pure Python
pip install "distmarg[rust]"    # plus the compiled backend for numpy inputs
```

```python
import numpy as np
from distmarg import DistanceMarginalization

d = np.geomspace(100, 5000, 4096)
dm = DistanceMarginalization(d, 2 * np.log(d), phase_marginalization=True)
# or DistanceMarginalization.from_bilby_prior(bilby_prior, phase_marginalization=True)
dm.ln_likelihood_ratio(rho_mf=12.0, rho_opt=10.0, d_ref=1000.0)
```

For JAX, torch or CuPy inputs, set `SCIPY_ARRAY_API=1` before SciPy is first
imported. With JAX, also enable `jax_enable_x64`.

The method, accuracy and benchmarks are documented in the
[repository](https://github.com/ColmTalbot/distance_marginalization).
