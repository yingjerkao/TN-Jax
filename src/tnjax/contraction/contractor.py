r"""Tensor contraction engine with label-based API.

Primary API::

    contract(\*tensors, output_labels=None, optimize="auto") -> Tensor

Labels drive contraction: legs with the same label across different tensors
are contracted (summed over). Free labels (unique to one tensor) become
output legs. This is the Cytnx-style label-based contraction model.

Under the hood, labels are translated to einsum subscript strings which
are fed to opt_einsum for optimal contraction path finding, then executed
with the JAX backend.

Lower-level API::

    contract_with_subscripts(tensors, subscripts, output_indices, optimize) -> Tensor
    truncated_svd(tensor, left_labels, right_labels, ...) -> (U, s, Vh)
    qr_decompose(tensor, left_labels, right_labels, ...) -> (Q, R)
"""

from __future__ import annotations

import string
from collections import Counter
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum

from tnjax.core.index import FlowDirection, Label, TensorIndex
from tnjax.core.tensor import (
    BlockKey,
    DenseTensor,
    SymmetricTensor,
    Tensor,
    _compute_valid_blocks,
)

# ---------- Label → Subscript Translation ----------

def _labels_to_subscripts(
    tensors: Sequence[Tensor],
    output_labels: Sequence[Label] | None = None,
) -> tuple[str, tuple[TensorIndex, ...]]:
    """Build an einsum subscript string from tensor labels.

    Algorithm:
    1. Count how many times each label appears across all tensors.
    2. Labels appearing >= 2 times are contracted (summed over).
    3. Labels appearing exactly once are free (output) legs.
    4. Assign a unique letter from the alphabet to each unique label.
    5. Build the subscript string "legs_t0,legs_t1,...->output_legs".

    Args:
        tensors:       Sequence of Tensor objects.
        output_labels: Explicit ordering of free labels in the output.
                       If None, uses the order: free labels of t0, t1, ...

    Returns:
        (subscripts, output_indices) where output_indices are TensorIndex
        objects for the output legs in output_labels order.

    Raises:
        ValueError: If a label appears more than 2 times (ambiguous).
        ValueError: If output_labels contains a label not present as a free label.
    """
    # Count label occurrences across all tensors
    label_counts: Counter[Label] = Counter()
    label_to_index: dict[Label, TensorIndex] = {}

    for tensor in tensors:
        for idx in tensor.indices:
            label_counts[idx.label] += 1
            # Keep the first-seen index metadata for each label
            if idx.label not in label_to_index:
                label_to_index[idx.label] = idx

    # Validate: no label appears more than 2 times
    for label, count in label_counts.items():
        if count > 2:
            raise ValueError(
                f"Label {label!r} appears {count} times across tensors. "
                f"Labels must appear at most 2 times (one per tensor to contract)."
            )

    # Identify free labels (appear exactly once) and contracted labels (appear twice)
    free_labels = [lbl for lbl, cnt in label_counts.items() if cnt == 1]
    # contracted_labels = [lbl for lbl, cnt in label_counts.items() if cnt == 2]

    # Assign letters to labels (need at most 52 unique labels for a-zA-Z)
    # For larger networks use a different encoding (multi-char not supported by einsum)
    all_labels = sorted(label_counts.keys(), key=str)
    if len(all_labels) > 52:
        raise ValueError(
            f"Too many unique labels ({len(all_labels)}) for einsum encoding. "
            f"Maximum supported is 52 (a-z + A-Z)."
        )

    available_chars = string.ascii_lowercase + string.ascii_uppercase
    label_to_char: dict[Label, str] = {
        lbl: available_chars[i] for i, lbl in enumerate(all_labels)
    }

    # Build subscript strings per tensor
    tensor_subscripts = []
    for tensor in tensors:
        subs = "".join(label_to_char[idx.label] for idx in tensor.indices)
        tensor_subscripts.append(subs)

    # Determine output label ordering
    if output_labels is None:
        # Default: free labels in the order they appear across tensors
        seen: set[Label] = set()
        ordered_free: list[Label] = []
        for tensor in tensors:
            for idx in tensor.indices:
                if idx.label in free_labels and idx.label not in seen:
                    ordered_free.append(idx.label)
                    seen.add(idx.label)
        output_labels = ordered_free
    else:
        # Validate user-specified output labels
        free_set = set(free_labels)
        for lbl in output_labels:
            if lbl not in free_set:
                raise ValueError(
                    f"output_labels contains {lbl!r} which is not a free label. "
                    f"Free labels are: {free_labels}"
                )

    output_subs = "".join(label_to_char[lbl] for lbl in output_labels)
    subscripts = ",".join(tensor_subscripts) + "->" + output_subs

    # Build output TensorIndex objects (use first-seen index for each free label)
    output_indices = tuple(label_to_index[lbl] for lbl in output_labels)

    return subscripts, output_indices


# ---------- Dense contraction ----------

def _contract_dense(
    tensors: Sequence[DenseTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> DenseTensor:
    """Contract dense tensors using opt_einsum with JAX backend.

    Calls opt_einsum.contract_path first (Python-level, no JAX tracing)
    then executes the contraction with backend='jax'.

    Args:
        tensors:        Sequence of DenseTensor.
        subscripts:     Einsum subscript string (e.g., "ij,jk->ik").
        output_indices: TensorIndex metadata for the output legs.
        optimize:       opt_einsum optimizer ('auto', 'greedy', 'dp', etc.).

    Returns:
        Contracted DenseTensor.
    """
    arrays = [t.todense() for t in tensors]

    # Find optimal contraction path (Python-level, pure Python overhead)
    _, path_info = opt_einsum.contract_path(subscripts, *arrays, optimize=optimize)

    # Execute contraction with JAX backend (GPU-compatible)
    result = opt_einsum.contract(
        subscripts, *arrays, optimize=path_info.path, backend="jax"
    )

    return DenseTensor(result, output_indices)


# ---------- Symmetric (block-sparse) contraction ----------

def _contract_symmetric(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> SymmetricTensor:
    """Contract block-sparse symmetric tensors.

    Algorithm:
    1. Parse the subscript string to identify contracted and free legs per tensor.
    2. Find all valid output blocks (charge combinations for free legs that
       satisfy conservation).
    3. For each output block, find all input block combinations where:
       - Contracted leg charges match between the two (or more) tensors
       - Free leg charges match the output block
    4. Contract those sub-arrays and accumulate into the output block.

    This preserves the block structure throughout: the output tensor is also
    a SymmetricTensor with only symmetry-allowed sectors stored.

    Args:
        tensors:        Sequence of SymmetricTensor with the same symmetry group.
        subscripts:     Einsum subscript string.
        output_indices: TensorIndex metadata for output legs.
        optimize:       opt_einsum optimizer for within-block contractions.

    Returns:
        Contracted SymmetricTensor.
    """
    # Parse subscripts: e.g., "ij,jk->ik" → inputs=["ij","jk"], output="ik"
    input_part, output_part = subscripts.split("->")
    input_subs = input_part.split(",")

    # Map each character to the corresponding TensorIndex
    char_to_index: dict[str, TensorIndex] = {}
    for tensor, subs in zip(tensors, input_subs):
        for char, idx in zip(subs, tensor.indices):
            char_to_index[char] = idx

    # Build output_indices list in output_part order
    out_indices_ordered = tuple(char_to_index[c] for c in output_part)

    # Find valid output blocks
    valid_output_keys = _compute_valid_blocks(out_indices_ordered)

    # For each tensor, build a map: (free leg charges) -> list of blocks
    # that can contribute to each output block.
    # We'll compute output blocks by iterating over all input block combinations.

    output_blocks: dict[BlockKey, Any] = {}

    # Iterate over all combinations of blocks from all input tensors
    tensor_block_lists = [list(t.blocks.items()) for t in tensors]

    for block_combo in _cartesian_product(tensor_block_lists):
        # block_combo: list of (key, array) pairs, one per tensor
        keys = [bc[0] for bc in block_combo]
        arrays = [bc[1] for bc in block_combo]

        # Check that contracted legs have matching charges
        # Build char -> charge mapping for this combination
        char_to_charge: dict[str, int] = {}
        compatible = True
        for tensor_idx, (key, _) in enumerate(zip(keys, tensors)):
            subs = input_subs[tensor_idx]
            for char, charge in zip(subs, block_combo[tensor_idx][0]):
                if char in char_to_charge:
                    if char_to_charge[char] != int(charge):
                        compatible = False
                        break
                else:
                    char_to_charge[char] = int(charge)
            if not compatible:
                break

        if not compatible:
            continue

        # Determine output block key from free leg charges
        output_key = tuple(char_to_charge.get(c, 0) for c in output_part)

        # Check this output key is valid (should be, but verify)
        if output_key not in set(valid_output_keys):
            continue

        # Contract this block combination using opt_einsum on small arrays
        try:
            result_array = opt_einsum.contract(
                subscripts, *arrays, optimize=optimize, backend="jax"
            )
        except Exception:
            continue

        # Accumulate into output block
        if output_key in output_blocks:
            output_blocks[output_key] = output_blocks[output_key] + result_array
        else:
            output_blocks[output_key] = result_array

    return SymmetricTensor(output_blocks, out_indices_ordered)


def _cartesian_product(lists: list[list]) -> list[list]:
    """Cartesian product of lists of (key, array) pairs."""
    if not lists:
        return [[]]
    result = []
    for item in lists[0]:
        for rest in _cartesian_product(lists[1:]):
            result.append([item] + rest)
    return result


# ---------- Public API ----------

def contract(
    *tensors: Tensor,
    output_labels: Sequence[Label] | None = None,
    optimize: str = "auto",
) -> Tensor:
    """Contract tensors by matching shared labels (Cytnx-style).

    Legs with the same label across different tensors are automatically
    contracted (summed over). Legs with unique labels become output legs.

    Args:
        *tensors:       Two or more Tensor objects to contract.
        output_labels:  Explicit ordering of output legs by label.
                        If None, uses the natural order (labels of first tensor
                        that is free, then second, etc.).
        optimize:       opt_einsum path optimizer strategy.

    Returns:
        Contracted Tensor with indices corresponding to free labels.

    Raises:
        ValueError: If a label appears more than 2 times (ambiguous contraction).
        TypeError:  If tensors have mixed DenseTensor/SymmetricTensor types.

    Example:
        >>> # A has labels ('i', 'j', 'k'), B has labels ('k', 'l', 'm')
        >>> result = contract(A, B)
        >>> result.labels()
        ('i', 'j', 'l', 'm')
    """
    if not tensors:
        raise ValueError("contract() requires at least one tensor")

    subscripts, output_indices = _labels_to_subscripts(tensors, output_labels)

    # If a single tensor with no contractions needed, return it as-is
    if len(tensors) == 1 and "->" in subscripts:
        lhs, rhs = subscripts.split("->")
        if lhs == rhs:
            return tensors[0]

    return contract_with_subscripts(tensors, subscripts, output_indices, optimize)


def contract_with_subscripts(
    tensors: Sequence[Tensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> Tensor:
    """Contract tensors using an explicit einsum subscript string.

    Lower-level API for power users who prefer subscript notation.
    The output_indices must provide TensorIndex metadata for each output leg.

    Args:
        tensors:        Sequence of Tensor objects.
        subscripts:     Einsum subscript string (e.g., "ij,jk->ik").
        output_indices: TensorIndex metadata for output legs in subscript order.
        optimize:       opt_einsum optimizer.

    Returns:
        Contracted Tensor.

    Raises:
        TypeError: If tensors have mixed DenseTensor/SymmetricTensor types.
    """
    all_dense = all(isinstance(t, DenseTensor) for t in tensors)
    all_sym = all(isinstance(t, SymmetricTensor) for t in tensors)

    if all_dense:
        return _contract_dense(list(tensors), subscripts, output_indices, optimize)  # type: ignore[arg-type]
    elif all_sym:
        return _contract_symmetric(list(tensors), subscripts, output_indices, optimize)  # type: ignore[arg-type]
    else:
        types = [type(t).__name__ for t in tensors]
        raise TypeError(
            f"Cannot mix DenseTensor and SymmetricTensor in a single contraction. "
            f"Got types: {types}. Convert all tensors to the same type first."
        )


# ---------- Truncated SVD ----------

def truncated_svd(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
) -> tuple[Tensor, jax.Array, Tensor]:
    """Reshape tensor into matrix, compute SVD, truncate, reshape back.

    The tensor is first reshaped into a matrix by grouping left_labels as
    rows and right_labels as columns. After SVD and truncation, the result
    is reshaped back.

    The new bond leg (connecting U and Vh factors) is given label
    new_bond_label, making it immediately usable in label-based contractions.

    Output labels::

        U:  (left_labels..., new_bond_label)
        Vh: (new_bond_label, right_labels...)

    Note:
        This function is not JIT-able as a whole because the truncation
        cutoff is determined dynamically from singular values (dynamic shape).
        Apply ``@jax.jit`` to the inner SVD step only; call this at Python level.

    Args:
        tensor:               Tensor to decompose.
        left_labels:          Labels forming the "left" (U) factor.
        right_labels:         Labels forming the "right" (Vh) factor.
        new_bond_label:       Label for the new virtual bond.
        max_singular_values:  Hard cap on bond dimension after truncation.
        max_truncation_err:   Truncate until relative truncation error <= this.
        normalize:            Normalize singular values to sum to 1.

    Returns:
        ``(U_tensor, singular_values, Vh_tensor)``
        -- U has labels ``(left_labels..., new_bond_label)``.
        Vh has labels ``(new_bond_label, right_labels...)``.
        singular_values is a 1-D JAX float array.

    Raises:
        ValueError: If left_labels + right_labels don't cover all tensor labels.
    """
    all_labels = tensor.labels()
    all_labels_set = set(all_labels)
    left_set = set(left_labels)
    right_set = set(right_labels)

    if left_set | right_set != all_labels_set:
        raise ValueError(
            f"left_labels {list(left_labels)} + right_labels {list(right_labels)} "
            f"must cover all tensor labels {list(all_labels)}"
        )
    if left_set & right_set:
        raise ValueError(
            f"left_labels and right_labels must be disjoint, "
            f"got overlap: {left_set & right_set}"
        )

    # Build axis ordering: left labels first, then right labels
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    # Get dense representation and reshape
    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)

    # SVD (not JIT-able at this level due to dynamic truncation)
    U, s, Vh = jnp.linalg.svd(matrix, full_matrices=False)

    # Determine truncation cutoff
    s_np = np.array(s)
    n_keep = len(s_np)

    if max_truncation_err is not None:
        # Keep singular values until truncation error <= max_truncation_err
        total_sq = float(np.sum(s_np**2))
        trunc_sq = 0.0
        for i in range(len(s_np) - 1, -1, -1):
            trunc_sq += float(s_np[i] ** 2)
            if trunc_sq / total_sq > max_truncation_err**2:
                n_keep = i + 2  # keep up to i+1 (1-indexed)
                break
        else:
            n_keep = len(s_np)

    if max_singular_values is not None:
        n_keep = min(n_keep, max_singular_values)

    n_keep = max(1, n_keep)  # always keep at least one

    # Truncate
    U = U[:, :n_keep]
    s = s[:n_keep]
    Vh = Vh[:n_keep, :]

    if normalize:
        s = s / jnp.sum(s)

    # Reshape back and build output tensors
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    U_dense = U.reshape(left_shape + (n_keep,))
    Vh_dense = Vh.reshape((n_keep,) + right_shape)

    # Build new bond index
    # Convention: bond on U is OUT (outgoing from left side)
    #             bond on Vh is IN (incoming to right side)
    # The charges on the bond index are 0..n_keep-1 (no symmetry on singular values)
    # For a dense SVD we use a trivial bond with all charges = 0
    # (symmetric SVD with charge-preserving structure is handled separately)
    bond_charges_out = np.zeros(n_keep, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    elif right_indices:
        sym = right_indices[0].symmetry
    else:
        from tnjax.core.symmetry import U1Symmetry
        sym = U1Symmetry()

    bond_index_out = TensorIndex(sym, bond_charges_out, FlowDirection.OUT, label=new_bond_label)
    bond_index_in = TensorIndex(sym, bond_charges_out, FlowDirection.IN, label=new_bond_label)

    U_indices = left_indices + (bond_index_out,)
    Vh_indices = (bond_index_in,) + right_indices

    U_tensor: Tensor
    Vh_tensor: Tensor
    if isinstance(tensor, SymmetricTensor):
        # For symmetric tensors, extract block structure from the dense result
        # This is a simplified version; a full symmetric SVD would preserve blocks
        try:
            U_tensor = SymmetricTensor.from_dense(U_dense, U_indices)
            Vh_tensor = SymmetricTensor.from_dense(Vh_dense, Vh_indices)
        except ValueError:
            # Fallback to dense if block extraction fails
            U_tensor = DenseTensor(U_dense, U_indices)
            Vh_tensor = DenseTensor(Vh_dense, Vh_indices)
    else:
        U_tensor = DenseTensor(U_dense, U_indices)
        Vh_tensor = DenseTensor(Vh_dense, Vh_indices)

    return U_tensor, s, Vh_tensor


# ---------- QR Decomposition ----------

def qr_decompose(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
) -> tuple[Tensor, Tensor]:
    """QR decomposition of a tensor for canonical form in DMRG.

    Reshapes tensor into a matrix, performs QR, then reshapes back.

    Output labels::

        Q: (left_labels..., new_bond_label)
        R: (new_bond_label, right_labels...)

    Args:
        tensor:          Tensor to decompose.
        left_labels:     Labels forming the Q (isometric) factor.
        right_labels:    Labels forming the R (upper triangular) factor.
        new_bond_label:  Label for the new virtual bond.

    Returns:
        (Q_tensor, R_tensor) where Q is isometric (Q^dag Q = I).
    """
    all_labels = tensor.labels()
    label_to_axis = {lbl: i for i, lbl in enumerate(all_labels)}
    left_axes = [label_to_axis[lbl] for lbl in left_labels]
    right_axes = [label_to_axis[lbl] for lbl in right_labels]

    dense = tensor.todense()
    perm = left_axes + right_axes
    dense_perm = jnp.transpose(dense, perm)

    left_indices = tuple(tensor.indices[i] for i in left_axes)
    right_indices = tuple(tensor.indices[i] for i in right_axes)
    left_dim = int(np.prod([idx.dim for idx in left_indices]))
    right_dim = int(np.prod([idx.dim for idx in right_indices]))

    matrix = dense_perm.reshape(left_dim, right_dim)
    Q, R = jnp.linalg.qr(matrix)

    bond_dim = Q.shape[1]
    left_shape = tuple(idx.dim for idx in left_indices)
    right_shape = tuple(idx.dim for idx in right_indices)

    Q_dense = Q.reshape(left_shape + (bond_dim,))
    R_dense = R.reshape((bond_dim,) + right_shape)

    bond_charges = np.zeros(bond_dim, dtype=np.int32)
    if left_indices:
        sym = left_indices[0].symmetry
    else:
        from tnjax.core.symmetry import U1Symmetry
        sym = U1Symmetry()

    bond_index_out = TensorIndex(sym, bond_charges, FlowDirection.OUT, label=new_bond_label)
    bond_index_in = TensorIndex(sym, bond_charges, FlowDirection.IN, label=new_bond_label)

    Q_indices = left_indices + (bond_index_out,)
    R_indices = (bond_index_in,) + right_indices

    Q_tensor = DenseTensor(Q_dense, Q_indices)
    R_tensor = DenseTensor(R_dense, R_indices)

    return Q_tensor, R_tensor
