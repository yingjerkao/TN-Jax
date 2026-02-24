# iPEPS

Infinite Projected Entangled Pair States (iPEPS) is a variational ansatz for
2D quantum lattice models. TN-Jax implements the **simple update** for
optimisation and the **Corner Transfer Matrix (CTM)** method for computing
observables.

## Background

An iPEPS represents a 2D quantum state as a tensor network where each site has
a local tensor $A[u,d,l,r,s]$ with four virtual bonds and one physical index.
For translationally invariant states, a single-site unit cell suffices.

### Simple update

Fast imaginary time evolution:

1. For each nearest-neighbour bond, apply $\exp(-\delta\tau\, H_{\text{bond}})$.
2. SVD to restore tensor-product form; truncate to bond dimension $D$.
3. Update diagonal $\lambda$ matrices that approximate the bond environment.

### CTM environment

The Corner Transfer Matrix method approximates the infinite environment of a
PEPS site using 8 tensors (4 corners + 4 edges):

```
C1 --- T1 --- C2
|             |
T4    [A]    T2
|             |
C4 --- T3 --- C3
```

CTM iteratively absorbs rows and columns until the corner singular values
converge.

## Configuration

```python
from tnjax import iPEPSConfig, CTMConfig

ctm_config = CTMConfig(
    chi=20,          # CTM environment bond dimension
    max_iter=100,    # maximum CTM iterations
    conv_tol=1e-8,   # convergence tolerance on corner singular values
    renormalize=True,
)

config = iPEPSConfig(
    max_bond_dim=2,            # PEPS virtual bond dimension D
    num_imaginary_steps=100,   # imaginary time evolution steps
    dt=0.01,                   # time step size
    ctm=ctm_config,
    gate_order="sequential",
)
```

## Example -- 2D Heisenberg model

```python
import jax.numpy as jnp
from tnjax import iPEPSConfig, CTMConfig, ipeps

# Heisenberg gate: H = Sz Sz + 0.5 (S+ S- + S- S+)
Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=jnp.float32)
Sp = jnp.array([[0, 1], [0, 0]], dtype=jnp.float32)
Sm = jnp.array([[0, 0], [1, 0]], dtype=jnp.float32)
I2 = jnp.eye(2, dtype=jnp.float32)

H_bond = (
    jnp.kron(Sz, Sz)
    + 0.5 * jnp.kron(Sp, Sm)
    + 0.5 * jnp.kron(Sm, Sp)
).reshape(2, 2, 2, 2)

config = iPEPSConfig(
    max_bond_dim=2,
    num_imaginary_steps=200,
    dt=0.01,
    ctm=CTMConfig(chi=10, max_iter=50),
)

energy, peps, env = ipeps(H_bond, initial_peps=None, config=config)
print(f"Energy per site: {energy:.6f}")
```

## Result

`ipeps()` returns a 3-tuple:

| Element | Type | Description |
|---------|------|-------------|
| `energy` | `float` | Energy per site |
| `peps` | `TensorNetwork` | Optimised PEPS (1x1 unit cell) |
| `env` | `CTMEnvironment` | Converged CTM environment tensors |

## CTMEnvironment

The `CTMEnvironment` named tuple contains the 8 environment tensors:

- **Corners** (`C1`, `C2`, `C3`, `C4`): shape `(chi, chi)`
- **Edges** (`T1`, `T2`, `T3`, `T4`): shape `(chi, D^2, chi)`

## Using CTM standalone

The `ctm()` function can be called independently to compute the
environment for an existing PEPS tensor:

```python
from tnjax import ctm, CTMConfig

env = ctm(A_tensor, CTMConfig(chi=20, max_iter=100))
```
