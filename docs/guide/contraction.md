# Contraction, SVD, and QR

The contraction engine translates label-based tensor operations into optimised
einsum calls executed by JAX.

## Label-based contraction

The core idea: **legs with the same label across different tensors are
automatically summed over**.

```python
from tenax import contract

# A has legs ("i", "bond"), B has legs ("bond", "j")
C = contract(A, B)
# "bond" appears in both -> contracted
# Result has legs ("i", "j")
```

### Multi-tensor contraction

`contract` accepts any number of tensors. Internally it uses `opt_einsum`
to find the optimal contraction order:

```python
D = contract(A, B, C)  # three-tensor contraction
```

### Controlling output label order

By default, free labels appear in the order they are encountered. Use
`output_labels` to specify an explicit ordering:

```python
C = contract(A, B, output_labels=("j", "i"))
```

### Optimiser selection

The `optimize` parameter selects the opt_einsum strategy:

```python
C = contract(A, B, optimize="auto")       # default
C = contract(A, B, optimize="greedy")     # faster for large networks
C = contract(A, B, optimize="optimal")    # brute-force optimal
```

## Truncated SVD

`truncated_svd` decomposes a tensor into U, s, V^dagger with truncation:

```python
from tenax import truncated_svd

# Split tensor T with legs ("left", "phys", "right") along the cut
# left_labels vs right_labels
U, s, Vh, s_full = truncated_svd(
    T,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
    max_singular_values=16,
)
# U has legs ("left", "phys", "bond")
# s is a 1D JAX array of truncated singular values
# Vh has legs ("bond", "right")
# s_full is the complete singular value spectrum before truncation
```

Parameters controlling truncation:

- `max_singular_values` -- hard cap on the bond dimension
- `max_truncation_err` -- discard smallest singular values until the
  relative truncation error exceeds this threshold

Both dense and symmetric tensors are supported. For `SymmetricTensor`,
the SVD is performed block-by-block within each charge sector.

## QR decomposition

`qr_decompose` splits a tensor into an orthogonal factor Q and an upper-
triangular factor R:

```python
from tenax import qr_decompose

Q, R = qr_decompose(
    T,
    left_labels=["left", "phys"],
    right_labels=["right"],
    new_bond_label="bond",
)
# Q has legs ("left", "phys", "bond")  -- isometric
# R has legs ("bond", "right")
```

QR is cheaper than SVD and is useful for canonicalising MPS tensors
during DMRG sweeps.

## Lower-level API

For full control, `contract_with_subscripts` accepts explicit einsum
subscript strings:

```python
from tenax import contract_with_subscripts

result = contract_with_subscripts(
    [A, B],
    subscripts="ij,jk->ik",
    output_indices=(...),  # TensorIndex tuple for the result
)
```

This is mainly used internally by the `TensorNetwork` graph container.
