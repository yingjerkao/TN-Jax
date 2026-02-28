# Cytnx vs TN-Jax API Comparison

## Core Abstractions

| Concept | Cytnx (C++/Python) | TN-Jax (JAX/Python) |
|---------|-------------------|---------------------|
| **Tensor** | `UniTensor` — labeled tensor with row/column rank | `DenseTensor` / `SymmetricTensor` — labeled tensor, no row/col distinction |
| **Index** | `Bond` — dimension, symmetry, quantum numbers, direction | `TensorIndex` — charges, symmetry, `FlowDirection` (IN/OUT), label |
| **Symmetry** | `Symmetry` — U1, Zn, etc. attached to Bond | `BaseSymmetry` — `U1Symmetry`, `ZnSymmetry`, same idea |
| **Network** | `Network` — graph container, `.net` file driven | `TensorNetwork` (graph) + `NetworkBlueprint` (.net files) |
| **Contraction** | `Network.Launch()` or `UniTensor.contract()` | `contract(*tensors)` or `blueprint.launch()` |

## Key Design Differences

### 1. Backend & Differentiability

- **Cytnx**: C++ core with Python bindings, GPU via CUDA. No autodiff.
- **TN-Jax**: Pure JAX. Both tensor types are JAX pytrees — `jit`, `vmap`, `grad` work natively. Enables AD-based iPEPS optimization (`optimize_gs_ad`).

### 2. Row/Column Rank

- **Cytnx**: `UniTensor` tracks row-rank vs column-rank (semicolon in `.net` files separates them). This matters for matrix operations like SVD.
- **TN-Jax**: No row/column rank concept. SVD/QR take explicit `left_labels`/`right_labels` arguments instead.

### 3. Block-Sparse Tensors

- **Cytnx**: `UniTensor` with symmetry Bonds stores blocks internally; block structure tied to Bond.
- **TN-Jax**: `SymmetricTensor` stores `dict[BlockKey, jax.Array]` blocks. Block structure is static (JIT-friendly — recompilation only on structure change, not value change).

### 4. `.net` File Format

- **Cytnx**: `TensorA: i,j ; k,l` (semicolon separates row/col rank), `TOUT:`, `ORDER:`
- **TN-Jax**: `TensorA: i, j, k, l` (no semicolon needed), same `TOUT:` and `ORDER:` keywords. Simplified since there's no row/column rank.

## API Surface Comparison

| Feature | Cytnx | TN-Jax |
|---------|-------|--------|
| Label-based contraction | `Contract(A, B)` | `contract(A, B)` |
| SVD | `Svd(T)` (uses row/col rank) | `truncated_svd(T, left_labels, right_labels)` |
| QR | `Qr(T)` | `qr_decompose(T, left_labels, right_labels)` |
| .net files | `Network("file.net")` → `.PutUniTensor()` → `.Launch()` | `NetworkBlueprint("file.net")` → `.put_tensor()` → `.launch()` |
| MPO construction | Manual | `AutoMPO` class + `build_auto_mpo()` |
| DMRG | Not built-in | `dmrg()`, `idmrg()` with config dataclasses |
| TRG/HOTRG | Not built-in | `trg()`, `hotrg()` |
| iPEPS + CTM | Not built-in | `ipeps()`, `ctm()`, `optimize_gs_ad()`, `compute_excitations()` |
| Spin operators | Manual | `spin_half_ops()`, `spin_one_ops()` |

## What TN-Jax Borrows from Cytnx

- **Label-based contraction philosophy** — legs carry string labels; matching labels auto-contract
- **`.net` file format** — declarative topology, template pattern (parse once, reuse)
- **Bond/Index with flow direction** — `FlowDirection.IN`/`OUT` mirrors cytnx's `BD_IN`/`BD_OUT`
- **Symmetry-aware tensors** — charge conservation across legs

## What TN-Jax Adds Beyond Cytnx

- **JAX autodiff** — gradient-based iPEPS optimization, differentiable CTM
- **Built-in algorithms** — DMRG, iDMRG, TRG, HOTRG, iPEPS, excitation spectra
- **AutoMPO** — ITensor-style Hamiltonian builder (not in cytnx)
- **JIT compilation** — static block structure enables efficient `jax.jit`

## What Cytnx Has That TN-Jax Doesn't

- **C++ performance** — compiled C++ backend, explicit CUDA GPU kernels
- **Row/column rank** — natural matrix semantics for linear algebra
- **Richer linear algebra** — `Eig`, `Inv`, `Det`, `Exp`, etc. as first-class operations on `UniTensor`
- **Non-Abelian symmetry** — SU(2) support (TN-Jax has only a stub `BaseNonAbelianSymmetry`)
- **Broader language support** — C++ API alongside Python

## Should TN-Jax Have a `UniTensor` Class?

Cytnx's `UniTensor` bundles a labeled tensor with a **row/column rank** — a persistent split of legs into "row" and "column" groups that gives every tensor implicit matrix semantics. This makes `Svd(T)` and `Qr(T)` zero-argument: the decomposition always splits along the row/column boundary.

### Arguments For

- **Simpler decomposition calls.** `Svd(T)` reads more cleanly than `truncated_svd(T, left_labels=["i","j"], right_labels=["k","l"])`. The row/column rank removes one decision point at call sites.
- **Cytnx migration.** Users moving code from cytnx would find the API familiar.
- **Self-documenting tensors.** A rank-(2,3) `UniTensor` immediately communicates "this is a map from a 2-index space to a 3-index space."

### Arguments Against

- **JAX pytree friction.** Row/column rank must be static (pytree aux data) for `jax.jit`, but it *looks* like mutable state. Users would expect `set_row_rank()` to work inside JIT — it can't without triggering recompilation. This is a constant source of confusion in any JAX-based design.
- **Rank is context-dependent.** The same MPS tensor `A[v_L, p, v_R]` is split as `(v_L, p | v_R)` for right-canonical SVD and `(v_L | p, v_R)` for left-canonical SVD. A single row/column rank baked into the tensor forces the user to reset it between operations, which is busywork that `left_labels`/`right_labels` avoids.
- **Redundant with labels.** TN-Jax's label system already disambiguates legs. The `left_labels`/`right_labels` API is more explicit and equally concise once you're used to it.
- **Two tensor classes already exist.** Adding `UniTensor` as a wrapper around `DenseTensor`/`SymmetricTensor` creates a third class (or replaces both), increasing API surface for little gain.

### Verdict

**Not recommended.** The `left_labels`/`right_labels` pattern is a better fit for JAX's functional/immutable model. Row/column rank solves a real UX problem in C++ (fewer function arguments), but in Python with keyword arguments and good label conventions, the benefit is marginal and the pytree complications are real.

If decomposition call sites feel verbose, a lighter-weight solution is a helper:

```python
def svd_at(tensor, bond_label):
    """SVD splitting left of the named bond."""
    idx = tensor.labels().index(bond_label)
    return truncated_svd(tensor,
        left_labels=tensor.labels()[:idx],
        right_labels=tensor.labels()[idx:])
```

This gives the same one-argument ergonomics without baking rank into the tensor.

---

## Should TN-Jax Have a `Bond` Class?

Cytnx's `Bond` is a standalone object — dimension, symmetry charges, direction — that can be created independently and passed to `UniTensor` constructors. TN-Jax's `TensorIndex` carries the same information (symmetry, charges, flow, label) but is always embedded inside a tensor.

### Arguments For

- **Explicit shared bonds.** When two tensors share a virtual bond, a `Bond` object could enforce that they have compatible dimensions and charges by construction, rather than catching mismatches at contraction time.
- **Construction ergonomics.** `Bond(dim=4, symmetry=U1, charges=[0,1,-1,0])` as a reusable building block reads well when constructing many tensors with the same bond structure (e.g., an MPS chain where every tensor shares the same virtual bond type).
- **Closer to the physics.** In tensor network diagrams, bonds (edges) are first-class objects connecting tensors. A `Bond` class mirrors this.

### Arguments Against

- **`TensorIndex` already is the Bond class.** It carries symmetry, charges, flow, dimension, and label — all the same data. Renaming it to `Bond` would be cosmetic; wrapping it adds indirection for no functional gain.
- **Bonds diverge in practice.** Even when two tensors "share" a bond, truncation (SVD) changes one side's dimension. A shared `Bond` object would either go stale or require synchronization machinery, adding complexity.
- **Label matching suffices.** TN-Jax identifies connected legs by label equality. There's no need for an object-identity check ("same Bond instance") when string equality ("same label") works and is simpler.
- **JAX immutability.** Mutable shared state (a `Bond` referenced by multiple tensors) conflicts with JAX's functional paradigm. Every truncation or reshape would need to produce new Bond objects anyway, eliminating the sharing benefit.

### Verdict

**Not recommended as a separate class.** `TensorIndex` already serves the Bond role. The "shared bond" use case is better handled by factory functions:

```python
def make_virtual_bond(dim, label, symmetry=None, charges=None):
    """Create a matched pair of TensorIndex for a virtual bond."""
    idx = TensorIndex(symmetry=symmetry, charges=charges,
                      flow=FlowDirection.OUT, label=label)
    return idx, idx.dual()
```

This gives explicit bond-pair construction without introducing a new class or shared mutable state.

### Possible Middle Ground

If user feedback shows that `TensorIndex` is too low-level for common workflows, consider:

1. **Aliasing**: `Bond = TensorIndex` — zero-cost rename for readability.
2. **Bond factory module**: a collection of helpers like `make_virtual_bond()`, `make_physical_index()`, `trivial_bond(dim)` that return `TensorIndex` objects with sensible defaults.
3. **Bond dataclass** (metadata only): a lightweight spec `Bond(dim, symmetry, charges)` that generates `TensorIndex` pairs but is never stored on tensors.

Option 2 is the most pragmatic — it improves ergonomics without touching the core data model.

---

## Summary

Cytnx is a **lower-level tensor library** (think NumPy-for-tensor-networks) with a polished C++/Python interface and symmetry support. TN-Jax is a **higher-level algorithm framework** built on JAX that borrows cytnx's labeling/network conventions but layers on autodiff, JIT, and ready-to-use algorithms (DMRG, iPEPS, TRG). If you need raw tensor manipulation with GPU and non-Abelian symmetries, cytnx is more mature. If you need differentiable tensor network algorithms with minimal boilerplate, TN-Jax is the better fit.

The design trade-offs around `UniTensor` and `Bond` boil down to **JAX's functional model vs. object-oriented convenience**. Cytnx's approach (mutable, stateful tensors with embedded rank and shared bonds) works well in C++ but clashes with JAX's immutable pytrees and JIT compilation. TN-Jax's current approach (stateless tensors, explicit label arguments, `TensorIndex` as the sole leg descriptor) is the right fit for a JAX-based framework, and can be made more ergonomic through helpers and factories rather than new classes.
