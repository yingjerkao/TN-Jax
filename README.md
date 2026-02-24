# TN-Jax

A JAX-based tensor network library with symmetry-aware block-sparse tensors and label-based contraction.

## Features

- **Block-sparse symmetric tensors** — only symmetry-allowed charge sectors stored (U(1), Z_n)
- **Label-based contraction** — legs are identified by string/integer labels; shared labels are automatically contracted (Cytnx-style)
- **opt_einsum integration** — optimal contraction path finding for multi-tensor contractions
- **Network class** — graph-based tensor network container with contraction caching
- **Algorithms** — DMRG, TRG, HOTRG, iPEPS (with CTM environment)
- **Extensible symmetry system** — non-Abelian symmetry interface for future SU(2) support

## Installation

```bash
# Using uv (recommended)
uv add tnjax

# Using pip
pip install tnjax
```

## Quick Start

```python
import jax
import jax.numpy as jnp
import numpy as np
from tnjax import (
    U1Symmetry, TensorIndex, FlowDirection,
    SymmetricTensor, TensorNetwork, contract
)

# Define U(1) symmetric tensor indices with named legs
u1 = U1Symmetry()
phys_charges = np.array([-1, 1], dtype=np.int32)
bond_charges = np.array([-1, 0, 1], dtype=np.int32)
key = jax.random.PRNGKey(0)

A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, FlowDirection.IN,  label="p0"),
        TensorIndex(u1, bond_charges, FlowDirection.IN,  label="left"),
        TensorIndex(u1, bond_charges, FlowDirection.OUT, label="bond"),
    ),
    key=key,
)
B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, FlowDirection.IN,  label="p1"),
        TensorIndex(u1, bond_charges, FlowDirection.IN,  label="bond"),  # shared label
        TensorIndex(u1, bond_charges, FlowDirection.OUT, label="right"),
    ),
    key=jax.random.PRNGKey(1),
)

# Contract by matching shared labels — "bond" is summed over automatically
result = contract(A, B)
print(result.labels())  # ('p0', 'left', 'p1', 'right')

# Build a tensor network and contract
tn = TensorNetwork()
tn.add_node("A", A)
tn.add_node("B", B)
tn.connect_by_shared_label("A", "B")
result = tn.contract()
```

## DMRG Example

```python
from tnjax.algorithms.dmrg import dmrg, build_mpo_heisenberg, DMRGConfig
from tnjax.network.network import build_mps

L = 10  # chain length
mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)

# Build random initial MPS
# ...

config = DMRGConfig(max_bond_dim=50, num_sweeps=10)
result = dmrg(mpo, initial_mps, config)
print(f"Ground state energy: {result.energy:.8f}")
```

## TRG Example

```python
from tnjax.algorithms.trg import trg, compute_ising_tensor, TRGConfig
import jax.numpy as jnp

beta = 0.44  # near critical temperature
T = compute_ising_tensor(beta)

config = TRGConfig(max_bond_dim=16, num_steps=20)
free_energy = trg(T, config)
print(f"Free energy per site: {free_energy:.8f}")
```

## Symmetry System

```python
from tnjax import U1Symmetry, ZnSymmetry
import numpy as np

# U(1): integer charges, fusion by addition
u1 = U1Symmetry()
charges = np.array([-1, 0, 1], dtype=np.int32)
print(u1.fuse(charges, charges))  # [-2, 0, 2]
print(u1.dual(charges))           # [1, 0, -1]

# Z_3: charges mod 3
z3 = ZnSymmetry(3)
print(z3.fuse(np.array([1, 2], dtype=np.int32),
              np.array([2, 2], dtype=np.int32)))  # [0, 1]
```

## Development

```bash
# Clone and install with dev dependencies
git clone https://github.com/yingjerkao/TN-Jax
cd TN-Jax
uv sync --all-extras --dev

# Run tests
uv run pytest tests/ -v

# Lint
uv run ruff check src/ tests/
```

## License

MIT
