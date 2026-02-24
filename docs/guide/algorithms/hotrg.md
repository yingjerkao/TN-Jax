# HOTRG

Higher-Order Tensor Renormalization Group (HOTRG) improves upon TRG by using
Higher-Order SVD (HOSVD) to compute optimal truncation isometries.

## Background

Instead of pairwise SVD splits (as in TRG), HOTRG constructs the truncation
isometry from the **environment tensor** -- formed by contracting two adjacent
tensors over their shared bonds -- and computing its HOSVD. This produces
a globally better approximation at each coarse-graining step.

Algorithm (horizontal step):

1. Form $M[u, u', d, d'] = \sum_{l,r} T[u,d,l,r] \cdot T[u',d',r,l]$
2. HOSVD of $M$: compute truncated isometries $U_u$, $U_d$
3. Compress $T$ using the isometries and contract to form $T_{\text{new}}$

The vertical step is analogous with left/right bonds.

Reference: Xie et al., PRB 86, 045139 (2012).

## Configuration

```python
from tnjax import HOTRGConfig

config = HOTRGConfig(
    max_bond_dim=16,               # maximum chi
    num_steps=20,                  # RG iterations
    direction_order="alternating", # "alternating" or "horizontal_first"
    svd_trunc_err=None,            # optional truncation error threshold
)
```

## Example -- 2D Ising model

```python
import math
from tnjax import HOTRGConfig, hotrg, compute_ising_tensor, ising_free_energy_exact

beta_c = math.log(1 + math.sqrt(2)) / 2
tensor = compute_ising_tensor(beta_c)

config = HOTRGConfig(max_bond_dim=16, num_steps=20)
log_Z_per_site = hotrg(tensor, config)

exact = ising_free_energy_exact(beta_c)
print(f"HOTRG log(Z)/N = {float(log_Z_per_site):.8f}")
print(f"Exact          = {exact:.8f}")
```

## TRG vs HOTRG

At the same bond dimension, HOTRG typically achieves better accuracy because
the HOSVD-based isometries account for the full tensor environment rather than
a single pairwise split.

| Method | `max_bond_dim=16` relative error |
|--------|----------------------------------|
| TRG | ~1e-5 |
| HOTRG | ~1e-7 |

The trade-off is that HOTRG is more expensive per step (additional SVDs for
the environment tensor).

## Direction order

- `"alternating"` (default): alternate horizontal and vertical coarse-graining
  steps. This preserves the square-lattice symmetry at each step.
- `"horizontal_first"`: perform both horizontal and vertical coarse-graining
  within each step. May converge faster for anisotropic systems.
