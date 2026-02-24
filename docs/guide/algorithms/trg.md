# TRG

The Tensor Renormalization Group (TRG) is a coarse-graining algorithm for
computing the partition function of 2D classical lattice models.

## Background

Starting from a single-site tensor $T_{udlr}$ (up, down, left, right)
placed on every site of an infinite 2D square lattice, TRG iteratively
reduces the lattice size by a factor of 2 via:

1. **SVD splitting**: decompose the tensor into half-tensors along
   horizontal and vertical directions.
2. **Plaquette contraction**: contract four half-tensors around a plaquette
   to form the coarse-grained tensor.

The partition function estimate is tracked via log normalisation:

$$\frac{\ln Z}{N} = \sum_k \frac{\ln \| T_k \|}{4^{k+1}}$$

Reference: Levin & Nave, PRL 99, 120601 (2007).

## Configuration

```python
from tnjax import TRGConfig

config = TRGConfig(
    max_bond_dim=16,      # maximum chi after each coarse-graining step
    num_steps=20,         # number of RG iterations
    svd_trunc_err=None,   # optional: truncation error threshold
)
```

## Example -- 2D Ising model

```python
import math
from tnjax import TRGConfig, trg, compute_ising_tensor, ising_free_energy_exact

# Critical temperature of the 2D Ising model
beta_c = math.log(1 + math.sqrt(2)) / 2

# Build the initial tensor for the partition function
tensor = compute_ising_tensor(beta_c, J=1.0)

# Run TRG
config = TRGConfig(max_bond_dim=16, num_steps=20)
log_Z_per_site = trg(tensor, config)

# Compare with the exact Onsager solution
exact = ising_free_energy_exact(beta_c)
print(f"TRG  log(Z)/N = {float(log_Z_per_site):.8f}")
print(f"Exact         = {exact:.8f}")
```

## Helper functions

### `compute_ising_tensor(beta, J=1.0)`

Builds the 4-leg transfer-matrix tensor for the 2D Ising model:

$$T_{udlr} = \sum_s \sqrt{Q_{u,s}} \sqrt{Q_{d,s}} \sqrt{Q_{l,s}} \sqrt{Q_{r,s}}$$

where $Q_{a,b} = \exp(\beta J \, \sigma_a \sigma_b)$.

Returns a `DenseTensor` with legs `("up", "down", "left", "right")`.

### `ising_free_energy_exact(beta, J=1.0)`

Computes the exact 2D Ising free energy per site via numerical integration of
the Onsager formula. Useful as a benchmark for TRG convergence.

## Convergence

TRG accuracy improves with `max_bond_dim`. Typical values:

| `max_bond_dim` | Relative error at $\beta_c$ |
|----------------|----------------------------|
| 4 | ~1e-2 |
| 8 | ~1e-3 |
| 16 | ~1e-5 |
| 32 | ~1e-7 |

For better accuracy at the same bond dimension, consider {doc}`hotrg`.
