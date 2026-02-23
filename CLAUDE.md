# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync --all-extras --dev

# Run all tests with coverage
uv run pytest tests/ -v --cov=tnjax --cov-report=term-missing

# Run a single test file
uv run pytest tests/test_contraction.py -v

# Run a single test by name
uv run pytest tests/test_dmrg.py::test_dmrg_energy -v

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/
```

## Architecture

**TN-Jax** is a JAX-based tensor network library with label-based contraction (Cytnx-style) and symmetry-aware block-sparse tensors.

### Core Layer (`src/tnjax/core/`)

The foundational type hierarchy:

- **`symmetry.py`**: `U1Symmetry` and `ZnSymmetry` implement the `BaseSymmetry` interface (fuse, dual, identity, n_values). Charges are integer numpy arrays.
- **`index.py`**: `TensorIndex` is a frozen dataclass carrying a `symmetry`, `charges` array, `flow` (IN/OUT), and string/integer `label`. Immutable and slot-based for memory efficiency.
- **`tensor.py`**: Two tensor types sharing a `Tensor` protocol:
  - `DenseTensor`: Full JAX array with index metadata
  - `SymmetricTensor`: Block-sparse dict mapping `BlockKey` (tuple of one charge per leg) → JAX array; only charge sectors satisfying conservation law are stored. Registered as JAX pytrees for jit/grad/vmap compatibility.

### Contraction Engine (`src/tnjax/contraction/contractor.py`)

- `contract(*tensors, output_labels, optimize)`: Translates shared string labels to einsum subscripts, uses opt_einsum for optimal contraction paths, executes via JAX.
- `truncated_svd(tensor, left_labels, right_labels, max_bond_dim, ...)`: Symmetry-aware SVD with truncation.
- `qr_decompose(...)`: QR decomposition with label routing.

### Network Layer (`src/tnjax/network/network.py`)

`TensorNetwork` is a networkx-based graph container (nodes = tensors, edges = shared labels). It caches contraction results keyed by `frozenset[NodeId]` and invalidates on structural changes. `build_mps()` and `build_peps()` are factory helpers.

### Algorithms (`src/tnjax/algorithms/`)

All algorithms use **Python for-loops** (not `jax.lax.scan`) as outer iteration because bond dimensions change dynamically after SVD truncation.

- **`dmrg.py`**: DMRG for 1D Hamiltonians. Lanczos eigensolver runs inside `jax.lax.while_loop` (JIT-able). Leg label conventions: virtual bonds `"v{i-1}_{i}"`, physical legs `"p{i}"`, MPO bonds `"w{i-1}_{i}"`.
- **`trg.py`**: Tensor Renormalization Group for 2D classical partition functions (Levin & Nave). Tracks free energy via log normalization.
- **`hotrg.py`**: Higher-order TRG (Evenbly & Vidal) for improved truncation.
- **`ipeps.py`**: Infinite PEPS with simple update (imaginary time evolution) and Corner Transfer Matrix (CTM) environment—8-tensor environment (4 corners + 4 edge tensors) for computing observables.

### Public API

`src/tnjax/__init__.py` exports: `contract`, `U1Symmetry`, `ZnSymmetry`, `TensorIndex`, `FlowDirection`, `DenseTensor`, `SymmetricTensor`, `TensorNetwork`, `build_mps`, `build_peps`, `DMRG`, `TRG`, `HOTRG`, `iPEPS`.

## Key Design Decisions

- **SymmetricTensor** stores only symmetry-allowed blocks; block structure is derived from index charges at construction time.
- **Label-based contraction**: shared labels between tensors are automatically contracted (no explicit einsum subscript management needed by users).
- **opt_einsum** finds the optimal contraction path for multi-tensor networks before JAX executes it.
- **Python-level DMRG/iPEPS loops**: necessary because bond dimensions are data-dependent and cannot be statically traced by JAX.
- **Hypothesis** is used for property-based tests of symmetry axioms (associativity, dual involution, conservation law).
