# Core Concepts

TN-Jax is built on three layers: **symmetries**, **indices**, and **tensors**.
Understanding these building blocks is essential for using the library
effectively.

## Symmetries

A symmetry object defines how quantum numbers (charges) combine and what
constitutes a conserved quantity. TN-Jax ships two abelian symmetry types:

### U(1) Symmetry

`U1Symmetry` models particle number or magnetisation conservation. Charges are
arbitrary integers and fuse by addition.

```python
from tnjax import U1Symmetry

sym = U1Symmetry()

# Fusion: charges add
print(sym.fuse(1, 2))   # 3
print(sym.fuse(-1, 1))  # 0

# Dual: sign flip
print(sym.dual(3))  # -3

# Identity charge
print(sym.identity())  # 0
```

### Z_n Symmetry

`ZnSymmetry` models cyclic conservation laws (e.g. parity when n=2). Charges
fuse by addition modulo n.

```python
from tnjax import ZnSymmetry

z2 = ZnSymmetry(n=2)
print(z2.fuse(1, 1))  # 0  (mod 2)
print(z2.dual(1))     # 1  (mod 2)

z3 = ZnSymmetry(n=3)
print(z3.fuse(2, 2))  # 1  (mod 3)
```

### Conservation law

A set of charges satisfies the conservation law when:

```
sum_i  flow_i * charge_i  ==  identity
```

where `flow_i` is +1 for `IN` and -1 for `OUT`.

## Indices

A `TensorIndex` attaches metadata to one leg of a tensor:

- **symmetry** -- which symmetry group governs this leg
- **charges** -- integer array of quantum numbers for each basis state
- **flow** -- `FlowDirection.IN` or `FlowDirection.OUT`
- **label** -- string or integer identifier used for contraction matching

```python
import numpy as np
from tnjax import U1Symmetry, TensorIndex, FlowDirection

sym = U1Symmetry()

# A physical spin-1/2 leg: up = +1, down = -1
phys = TensorIndex(
    symmetry=sym,
    charges=np.array([1, -1], dtype=np.int32),
    flow=FlowDirection.IN,
    label="phys",
)

# A virtual bond with 3 sectors
bond = TensorIndex(
    symmetry=sym,
    charges=np.array([-1, 0, 1], dtype=np.int32),
    flow=FlowDirection.OUT,
    label="bond",
)

print(phys.dim)    # 2
print(bond.label)  # "bond"
```

`TensorIndex` is a frozen dataclass -- immutable after creation. Use
`idx.relabel(new_label)` to create a copy with a different label.

## Tensors

TN-Jax provides two concrete tensor types that share a common `Tensor`
protocol.

### DenseTensor

A full JAX array with index metadata. Every element is stored.

```python
import jax.numpy as jnp
from tnjax import DenseTensor

data = jnp.ones((2, 3))
tensor = DenseTensor(data, (phys, bond))

print(tensor.shape)    # (2, 3)
print(tensor.labels())  # ('phys', 'bond')
print(tensor.dtype)    # float32
```

### SymmetricTensor

A block-sparse tensor that only stores charge sectors satisfying the
conservation law. This can yield large memory and FLOP savings when the
symmetry has many sectors.

```python
import jax
from tnjax import SymmetricTensor

bond_in = TensorIndex(sym, np.array([-1, 0, 1], dtype=np.int32),
                       FlowDirection.IN, label="left")
bond_out = TensorIndex(sym, np.array([1, 0, -1], dtype=np.int32),
                        FlowDirection.OUT, label="right")

st = SymmetricTensor.random_normal(
    indices=(phys, bond_in, bond_out),
    key=jax.random.PRNGKey(42),
)

print(st.n_blocks)        # number of non-zero charge sectors
print(st.block_shapes())  # {BlockKey: shape, ...}
print(st.todense().shape) # full dense shape (2, 3, 3)
```

### JAX pytree integration

Both tensor types are registered as JAX pytrees. This means they work
transparently with `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
@jax.jit
def norm_squared(t):
    return t.norm() ** 2

print(norm_squared(tensor))
```

### Relabeling

Labels are the key to contraction. Use `relabel` and `relabels` to rename
legs:

```python
# Single relabel
t2 = tensor.relabel("bond", "right")
print(t2.labels())  # ('phys', 'right')

# Multiple relabels
t3 = tensor.relabels({"phys": "p0", "bond": "v0_1"})
print(t3.labels())  # ('p0', 'v0_1')
```
