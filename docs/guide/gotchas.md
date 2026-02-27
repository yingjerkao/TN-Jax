# Gotchas

Common pitfalls when working with TN-Jax and JAX.

## Float64 precision and `JAX_ENABLE_X64`

TN-Jax defaults to `float64` for all tensors and algorithms. Importing
`tnjax` automatically calls `jax.config.update("jax_enable_x64", True)`,
so 64-bit arithmetic is enabled out of the box.

If you use JAX directly *before* importing `tnjax`, you may need to enable
x64 mode yourself:

```python
import jax
jax.config.update("jax_enable_x64", True)

import tnjax  # also enables x64, but JAX was already imported above
```

Without x64 enabled:

- Computations run at `float32` precision (no errors raised, just warnings).
- DMRG/iDMRG energies may converge to fewer significant digits.
- Convergence tolerances below ~1e-7 may not be reachable.

## MPO index convention

The MPO W-tensor uses the convention `W[w_l, ket, bra, w_r]`:

- The two outer indices (`w_l`, `w_r`) are MPO bond dimensions.
- The two middle indices are physical: ket (top) and bra (bottom).

The DMRG effective-Hamiltonian matvec einsum is
`"abc,apqd,bpse,eqtf,dfg->cstg"` where `p,q` are ket and `s,t` are bra
physical indices.

## NumPy >= 2.0 dtype casting

Under NumPy 2.0+, adding a Python `complex` scalar (even `1+0j`) into a
`float64` array raises `UFuncOutputCastingError`. Extract the `.real` part
or use an explicit `complex128` dtype:

```python
# Bad — raises UFuncOutputCastingError with NumPy >= 2.0
arr = np.zeros(3, dtype=np.float64)
arr[0] = 1 + 0j  # fails

# Good
arr[0] = (1 + 0j).real
# or
arr = np.zeros(3, dtype=np.complex128)
arr[0] = 1 + 0j
```

## Odd-circumference cylinders

The `build_bulk_mpo_heisenberg_cylinder` function only accepts **even** `Ly`.
Wrapping a square lattice with odd circumference creates odd-length cycles in
the ring direction, breaking bipartiteness and frustrating antiferromagnetic
(Neel) order. This leads to poor iDMRG convergence and physically different
ground states.
