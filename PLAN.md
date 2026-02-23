# TN-Jax: JAX Tensor Network Library — Implementation Plan

## Context

Build a JAX-based tensor network library from scratch at `/Users/yjkao/TN-Jax`. The library provides block-sparse symmetric tensors, a graph-based network abstraction with **Cytnx-style label-based contraction**, opt_einsum-based contraction, and full implementations of DMRG, TRG, HOTRG, and iPEPS algorithms. The package uses uv for dependency management and GitHub Actions for CI/CD with PyPI deployment.

**Key decisions (from user):**
- Python 3.11+
- Block-sparse tensor storage (not dense-with-mask)
- U(1) + Zn symmetries fully implemented; non-Abelian as stub/interface only
- All four algorithms (DMRG, TRG, HOTRG, iPEPS) fully implemented in v0.1
- **Label-based contraction (Cytnx-style):** tensor legs carry string/integer labels; contractions are specified by matching labels, not by positional index pairs

---

## Directory Structure

```
TN-Jax/
├── pyproject.toml
├── uv.lock
├── .python-version          # "3.11"
├── .gitignore
├── README.md
├── .github/
│   └── workflows/
│       ├── ci.yml           # test + lint on push/PR
│       └── publish.yml      # PyPI on tag v*.*.*
├── src/
│   └── tnjax/
│       ├── __init__.py
│       ├── core/
│       │   ├── __init__.py
│       │   ├── symmetry.py  # BaseSymmetry, U1Symmetry, ZnSymmetry, BaseNonAbelianSymmetry
│       │   ├── index.py     # TensorIndex, FlowDirection
│       │   └── tensor.py    # Tensor protocol, DenseTensor, SymmetricTensor
│       ├── contraction/
│       │   ├── __init__.py
│       │   └── contractor.py  # contract_tensors, truncated_svd, qr_decompose
│       ├── network/
│       │   ├── __init__.py
│       │   └── network.py     # TensorNetwork, build_mps, build_peps
│       └── algorithms/
│           ├── __init__.py
│           ├── dmrg.py
│           ├── trg.py
│           ├── hotrg.py
│           └── ipeps.py
└── tests/
    ├── conftest.py
    ├── test_symmetry.py
    ├── test_index.py
    ├── test_tensor.py
    ├── test_contraction.py
    ├── test_network.py
    ├── test_dmrg.py
    ├── test_trg.py
    ├── test_hotrg.py
    └── test_ipeps.py
```

---

## Phase 0: Project Scaffolding

**Files to create:**

### `pyproject.toml`
```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "tnjax"
version = "0.1.0"
description = "JAX-based tensor network library with symmetry-aware block-sparse tensors"
readme = "README.md"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "jax>=0.4.30",
    "jaxlib>=0.4.30",
    "opt-einsum>=3.3.0",
    "numpy>=1.26",
    "networkx>=3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "hypothesis>=6.100",
    "ruff>=0.4",
    "mypy>=1.10",
    "twine>=5.0",
]

[tool.uv]
dev-dependencies = ["tnjax[dev]"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--cov=tnjax --cov-report=term-missing --cov-report=xml"

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
```

---

## Phase 1: Core Layer (`src/tnjax/core/`)

### `symmetry.py` — No JAX dependency; pure Python arithmetic

**Key classes:**
```python
class BaseSymmetry(ABC):
    @abstractmethod
    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def dual(self, charges: np.ndarray) -> np.ndarray: ...
    @abstractmethod
    def identity(self) -> int: ...
    @abstractmethod
    def n_values(self) -> int | None: ...
    def fuse_many(self, charge_list: list[np.ndarray]) -> np.ndarray: ...

class U1Symmetry(BaseSymmetry):   # fuse = add, dual = negate
class ZnSymmetry(BaseSymmetry):   # fuse = add mod n, dual = negate mod n
    def __init__(self, n: int):   # raises ValueError if n < 2

class BaseNonAbelianSymmetry(BaseSymmetry):
    @abstractmethod
    def recoupling_coefficients(self, j1, j2, j3) -> np.ndarray: ...
    @abstractmethod
    def allowed_fusions(self, j1, j2) -> list[int]: ...
```

All concrete classes implement `__eq__` and `__hash__` for use as dict keys.

### `index.py` — Depends on `symmetry.py`

```python
class FlowDirection(IntEnum):
    IN  =  1   # incoming / ket-type
    OUT = -1   # outgoing / bra-type

# Labels are strings or integers; unique within a tensor, shared labels = contracted
Label = str | int

@dataclass(frozen=True, slots=True)
class TensorIndex:
    symmetry: BaseSymmetry
    charges:  np.ndarray    # shape (D,), dtype int32 — coerced in __post_init__
    flow:     FlowDirection
    label:    Label = ""    # The canonical identity of this leg

    def dual(self) -> "TensorIndex":             # flips flow + applies sym.dual()
    def relabel(self, new_label: Label) -> "TensorIndex":  # new frozen copy with new label
    def is_dual_of(self, other) -> bool:         # strict: opposite flow + matching charges
    def compatible_with(self, other) -> bool:    # loose: same sym type + dim + opposite flow
```

**Label semantics:** Labels are the primary user-facing identity for each leg. Two legs with the same label on different tensors will be automatically contracted when `Contract()` or `TensorNetwork.contract()` is called. Labels survive non-contracted legs and propagate to the result tensor. Users must rename labels explicitly to avoid unintended contractions.

`frozen=True, slots=True` for memory efficiency (millions created in large networks).

### `tensor.py` — Depends on `symmetry.py`, `index.py`, JAX

```python
@runtime_checkable
class Tensor(Protocol):
    indices: tuple[TensorIndex, ...]
    ndim: int
    dtype: Any
    def todense(self) -> jax.Array: ...
    def conj(self) -> "Tensor": ...
    def transpose(self, axes: tuple[int, ...]) -> "Tensor": ...
    def norm(self) -> jax.Array: ...
    def labels(self) -> tuple[Label, ...]: ...           # convenience: extract labels from indices
    def relabel(self, old: Label, new: Label) -> "Tensor": ...  # return new tensor with one label changed
    def relabels(self, mapping: dict[Label, Label]) -> "Tensor": ...  # batch relabeling

@jax.tree_util.register_pytree_node_class
class DenseTensor:
    # Pytree: data array is leaf, indices are aux
    def tree_flatten(self): return (self._data,), self._indices
    @classmethod
    def tree_unflatten(cls, indices, children): ...

    def labels(self) -> tuple[Label, ...]: ...
    def relabel(self, old, new) -> "DenseTensor": ...
    def relabels(self, mapping) -> "DenseTensor": ...

BlockKey = tuple[int, ...]  # one charge per leg

@jax.tree_util.register_pytree_node_class
class SymmetricTensor:
    # Storage: dict[BlockKey, jax.Array] — only symmetry-allowed sectors
    # Pytree: block arrays are leaves, (keys, indices) are aux
    def tree_flatten(self): return list(blocks.values()), (list(blocks.keys()), indices)

    @classmethod
    def zeros(cls, indices, dtype=jnp.float32) -> "SymmetricTensor": ...
    @classmethod
    def random_normal(cls, indices, key, dtype=jnp.float32) -> "SymmetricTensor": ...
    @classmethod
    def from_dense(cls, data, indices, tol=1e-12) -> "SymmetricTensor": ...

    def _validate(self):  # checks all block keys satisfy conservation law

    def labels(self) -> tuple[Label, ...]: ...
    def relabel(self, old, new) -> "SymmetricTensor": ...
    def relabels(self, mapping) -> "SymmetricTensor": ...
```

**Critical design note:** Block structure (dict keys) is static Python-level data; JAX traces only through block array values. This means `jax.jit` can compile over `SymmetricTensor` when block keys are fixed (they are within a single sweep in DMRG). Recompilation only occurs when bond dimension changes after SVD truncation — acceptable since bond dim stabilizes quickly.

Helper function `_compute_valid_blocks(indices)` returns all charge-sector tuples satisfying `sum_i(flow_i * charge_i) == identity`.

---

## Phase 2: Contraction Layer (`src/tnjax/contraction/contractor.py`)

Depends on `tensor.py`, `index.py`, `opt_einsum`.

**Label-based contraction model (Cytnx-style):** The primary contraction API takes tensors and contracts all legs whose labels match across tensors. No subscript string is required from the user — the label system automatically determines which legs to contract and which are free (output) legs.

```python
def contract(
    *tensors: Tensor,
    output_labels: Sequence[Label] | None = None,
    optimize: str = "auto",
) -> Tensor:
    """Contract tensors by matching shared labels.

    Shared labels (appearing in more than one tensor) are summed over.
    Free labels (appearing in exactly one tensor) become output legs.

    Args:
        *tensors:       Two or more tensors to contract.
        output_labels:  Explicit ordering of output labels. If None, order
                        is: free labels of tensors[0], then tensors[1], etc.
        optimize:       opt_einsum path optimizer.
    Returns:
        Contracted tensor whose indices are the free-label legs.

    Raises:
        ValueError if a label appears > 2 times across all tensors
                   (ambiguous contraction).

    Example:
        A has labels ('i', 'j', 'k')
        B has labels ('k', 'l', 'm')
        contract(A, B) → tensor with labels ('i', 'j', 'l', 'm')
        contract(A, B, output_labels=('j', 'l', 'i', 'm')) → reordered
    """
    # Implementation:
    # 1. _labels_to_subscripts(*tensors, output_labels) → (subscripts_str, output_indices)
    # 2. Dispatch to _contract_dense or _contract_symmetric

def _labels_to_subscripts(
    tensors: Sequence[Tensor],
    output_labels: Sequence[Label] | None,
) -> tuple[str, tuple[TensorIndex, ...]]:
    """Build einsum subscript string from tensor labels.

    Assigns a letter to each unique label. Shared labels → same letter on
    both sides. Output section contains only free-label letters, in
    output_labels order (or default order if None).
    """

# Lower-level subscript-based API (used internally and for power users)
def contract_with_subscripts(
    tensors: Sequence[Tensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> Tensor:
    # Dispatches to _contract_dense or _contract_symmetric

def _contract_dense(
    tensors: Sequence[DenseTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> DenseTensor:
    # opt_einsum.contract_path() called first (Python-side, no JIT)
    # then opt_einsum.contract(..., backend="jax")

def _contract_symmetric(
    tensors: Sequence[SymmetricTensor],
    subscripts: str,
    output_indices: tuple[TensorIndex, ...],
    optimize: str = "auto",
) -> SymmetricTensor:
    # For each valid output block, find compatible input blocks, contract, accumulate

def truncated_svd(
    tensor: Tensor,
    left_labels: Sequence[Label],   # which labels go to U factor
    right_labels: Sequence[Label],  # which labels go to Vh factor
    new_bond_label: Label = "bond", # label for the new singular-value bond
    max_singular_values: int | None = None,
    max_truncation_err: float | None = None,
    normalize: bool = False,
) -> tuple[Tensor, jax.Array, Tensor]:
    # Left tensor has labels (*left_labels, new_bond_label)
    # Right tensor has labels (new_bond_label, *right_labels)
    # Python-level (not JIT-able across truncation)

def qr_decompose(
    tensor: Tensor,
    left_labels: Sequence[Label],
    right_labels: Sequence[Label],
    new_bond_label: Label = "bond",
) -> tuple[Tensor, Tensor]:
    # Q has labels (*left_labels, new_bond_label)
    # R has labels (new_bond_label, *right_labels)
```

**Key insight:** By using labels in `truncated_svd` and `qr_decompose`, the new bond leg gets a user-specified label. This label can then be immediately used to connect the result to other tensors in a network — purely by label matching, no index position tracking needed.

---

## Phase 3: Network Layer (`src/tnjax/network/network.py`)

Depends on `tensor.py`, `index.py`, `contractor.py`, `networkx`.

**Label-based edge model:** In the network, edges between nodes are identified by the shared tensor leg label. Connecting two tensor legs means both `TensorIndex` objects share the same `label`. The graph stores (node_id, label) pairs as edge endpoints — no positional leg index needed for graph traversal or contraction specification.

```python
class TensorNetwork:
    # Internal: nx.MultiGraph
    #   Nodes carry Tensor objects
    #   Edges identified by the shared label (not leg position)
    # Cache: dict[frozenset[NodeId], Tensor] — invalidated on graph change

    def add_node(self, node_id: NodeId, tensor: Tensor) -> None:
        # Validates all labels within tensor are unique
        # Registers node in graph

    def remove_node(self, node_id: NodeId) -> Tensor: ...
    def replace_tensor(self, node_id: NodeId, tensor: Tensor) -> None:
        # Validates labels are preserved (same set of labels as old tensor)

    def get_tensor(self, node_id: NodeId) -> Tensor: ...

    def connect(self, node_a: NodeId, label_a: Label,
                node_b: NodeId, label_b: Label) -> None:
        """Connect two legs by their labels.

        After connection, leg label_a on node_a and leg label_b on node_b
        are marked as contracted. Internally, a shared label is chosen
        (user may also call relabel_bond to rename both to the same label).

        Validates:
        - Both labels exist in their respective tensors
        - The two TensorIndex objects are compatible_with() each other
        """

    def connect_by_shared_label(self, node_a: NodeId, node_b: NodeId) -> int:
        """Auto-connect all legs sharing the same label between two nodes.

        Returns the number of connections made.
        Raises if no shared labels exist or shared labels are incompatible.
        """

    def disconnect(self, node_a: NodeId, label_a: Label,
                   node_b: NodeId, label_b: Label) -> None: ...

    def relabel_bond(self, node_id: NodeId, old_label: Label, new_label: Label) -> None:
        """Rename a leg's label on a node (and update edge registry)."""

    def open_legs(self, node_id: NodeId) -> list[Label]:
        """Return labels of legs not currently connected to any other node."""

    def contract(
        self,
        nodes: list[NodeId] | None = None,
        output_labels: Sequence[Label] | None = None,
        optimize: str = "auto",
        cache: bool = True,
    ) -> Tensor:
        """Contract a subset of nodes.

        Internally:
        1. Check cache (frozenset of node IDs)
        2. Build einsum subscripts using _labels_to_subscripts() —
           shared labels between nodes in the subset are contracted,
           legs connected to nodes outside the subset are kept free
        3. Call contract() from contractor.py
        4. Store in cache and return

        The output tensor's leg labels are the free labels, ordered by
        output_labels if given, or default ordering otherwise.
        """

    def _build_subscripts_for_nodes(
        self, nodes: list[NodeId]
    ) -> tuple[str, tuple[TensorIndex, ...]]:
        # Uses graph edges to identify shared labels within the subset
        # Uses label matching (not positional) to build the einsum string

def build_mps(
    tensors: list[Tensor],
    open_boundary: bool = True,
) -> TensorNetwork:
    """Build MPS TensorNetwork from a list of tensors.

    Tensors must already have meaningful labels:
    - Each site tensor should have a 'phys' label for the physical leg
    - Virtual bond legs should share labels with neighbors (e.g., site i's
      right bond labeled 'bond_i' matches site i+1's left bond 'bond_i')
    Alternatively, default virtual bond labels are assigned: 'v{i}_{i+1}'.
    """

def build_peps(
    tensors: list[list[Tensor]],
    Lx: int,
    Ly: int,
    open_boundary: bool = True,
) -> TensorNetwork:
    """Build PEPS TensorNetwork from a 2D list of tensors.

    Virtual bond labels follow convention: 'h{i}_{j}_{j+1}' (horizontal)
    and 'v{i}_{i+1}_{j}' (vertical). Physical legs labeled 'p{i}_{j}'.
    """
```

**API example showing label-based workflow:**
```python
# Create tensors with named legs
A = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, IN,  label="p0"),
        TensorIndex(u1, bond_charges, IN,  label="left"),
        TensorIndex(u1, bond_charges, OUT, label="bond_01"),  # shared with B
    ),
    key=key,
)
B = SymmetricTensor.random_normal(
    indices=(
        TensorIndex(u1, phys_charges, IN,  label="p1"),
        TensorIndex(u1, bond_charges, IN,  label="bond_01"),  # same label as A's right
        TensorIndex(u1, bond_charges, OUT, label="right"),
    ),
    key=key2,
)
tn = TensorNetwork()
tn.add_node("A", A)
tn.add_node("B", B)
tn.connect_by_shared_label("A", "B")  # auto-connects "bond_01"
result = tn.contract()  # result has labels ("p0", "left", "p1", "right")

# Or direct two-tensor contract without a network:
from tnjax import contract
result = contract(A, B)  # same result by label matching
```

---

## Phase 4: Algorithm Modules

All algorithms use label-based operations throughout — no positional leg indexing in algorithm code.

### Label conventions across algorithms

| Label pattern | Meaning |
|---------------|---------|
| `"phys"` / `"p{i}"` | Physical / site index |
| `"left"` / `"right"` | Open boundary virtual legs |
| `"v{i}_{i+1}"` | Virtual bond between sites i and i+1 |
| `"mpo_top"` / `"mpo_bot"` | MPO physical in/out legs |
| `"env_l"` / `"env_r"` | Environment contraction legs |
| `"svd_bond"` | New bond created by `truncated_svd` |
| `"ctm_{dir}"` | CTM corner/edge connection legs |

### `algorithms/dmrg.py`

**Architecture decision:** Python `for`-loop over sites (not `jax.lax.scan`) because bond dimensions change after SVD truncation. `@jax.jit` applied to the inner effective Hamiltonian matvec.

```python
@dataclass
class DMRGConfig:
    max_bond_dim: int = 100
    num_sweeps: int = 10
    convergence_tol: float = 1e-10
    two_site: bool = True           # 2-site DMRG (allows bond dim growth)
    lanczos_max_iter: int = 50
    noise: float = 0.0
    svd_trunc_err: float | None = None

class DMRGResult(NamedTuple):
    energy: float
    energies_per_sweep: list[float]
    mps: TensorNetwork
    truncation_errors: list[float]
    converged: bool

def dmrg(hamiltonian: TensorNetwork, initial_mps: TensorNetwork, config: DMRGConfig) -> DMRGResult:
    # 1. Build full left/right environment lists (L[i], R[i])
    # 2. Sweep loop (Python for): left→right then right→left
    # 3. At each site i: contract theta using label-based contract():
    #      theta = contract(mps[i], mps[i+1])
    #    shared virtual bond label "v{i}_{i+1}" is contracted automatically
    # 4. Eigenvalue solve, then:
    #      U, s, Vh = truncated_svd(theta, left_labels=["v{i-1}_i","p{i}"],
    #                               right_labels=["p{i+1}","v{i+1}_{i+2}"],
    #                               new_bond_label="v{i}_{i+1}")
    # 5. Update environments; early exit if energy converged

def _effective_hamiltonian_matvec(theta, left_env, mpo_l, mpo_r, right_env) -> Tensor:
    # Uses label-based contract(): L, W_l, W_r, R all share appropriate labels
    # H_eff = L * W_l * W_r * R  (labels guide which legs contract)
_matvec_jit = jax.jit(_effective_hamiltonian_matvec)

def _lanczos_solve(matvec, initial_vector, num_steps, tol) -> tuple[float, Tensor]:
    # jax.lax.while_loop for static shapes; tridiagonal solve via jnp.linalg.eigh

def build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0, hz=0.0, symmetry=None) -> TensorNetwork:
    # Reference implementation for testing; supports optional U(1) symmetry
```

### `algorithms/trg.py`

```python
@dataclass
class TRGConfig:
    max_bond_dim: int = 16
    num_steps: int = 10
    svd_trunc_err: float | None = None

def trg(tensor: Tensor, config: TRGConfig) -> jax.Array:
    # Returns log(Z)/N (free energy per site)
    # Python for-loop over steps; each step calls _trg_step

def _trg_step(tensor, max_bond_dim, svd_trunc_err) -> tuple[Tensor, jax.Array]:
    # SVD split: T → Sl * Sr (horiz) and Su * Sd (vert)
    # Contract 4 half-tensors around plaquette → new coarse-grained tensor

@jax.jit  # static chi via functools.partial or static_argnums
def _trg_svd_step(T, chi) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    # Returns (Sl, Sr, Su, Sd) — four half-tensors

def compute_ising_tensor(beta, J=1.0) -> jax.Array:
    # 2D Ising transfer matrix tensor of shape (2,2,2,2)
    # Used as reference for testing TRG convergence vs. Onsager solution
```

### `algorithms/hotrg.py`

```python
@dataclass
class HOTRGConfig:
    max_bond_dim: int = 16
    num_steps: int = 10
    direction_order: str = "alternating"

def hotrg(tensor: Tensor, config: HOTRGConfig) -> jax.Array:
    # Python for-loop; alternates horizontal/vertical HOTRG steps

def _hotrg_step_horizontal(tensor, max_bond_dim) -> tuple[Tensor, jax.Array]:
    # 1. Form M = T ⊗ T contracted over shared bonds
    # 2. HOSVD of M: compute isometry U via SVD of matricization
    # 3. Compress: T' = U† * T * U on both up/down legs

def _compute_hosvd_isometry(M, axis, chi_target) -> jax.Array:
    # Matricize M along axis → SVD → take top chi_target left singular vectors
    # @jax.jit compatible (all shapes static)
```

### `algorithms/ipeps.py`

```python
@dataclass
class CTMConfig:
    chi: int = 20
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True

@dataclass
class iPEPSConfig:
    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = None

class CTMEnvironment(NamedTuple):
    # 4 corners (chi×chi) + 4 edge tensors (chi×d²×chi)
    C1, C2, C3, C4: Tensor
    T1, T2, T3, T4: Tensor

def ipeps(hamiltonian_gate, initial_peps, config) -> tuple[float, TensorNetwork, CTMEnvironment]:
    # 1. Simple update (imaginary time): apply gate, SVD+truncate, update lambda matrices
    # 2. CTM: iterate absorption until convergence
    # 3. Compute energy via CTM environment

def ctm(peps, config, initial_env=None) -> CTMEnvironment:
    # Python while-loop (convergence check); each absorption step JIT-compiled

def simple_update_step(peps, site_a, site_b, gate, lambdas, max_bond_dim, ...) -> ...:
    # @jax.jit with static max_bond_dim

def compute_energy_ctm(peps, env, hamiltonian_gate) -> jax.Array:
    # Contracts PEPS + conjugate + gate + CTM environment around one bond
```

---

## Phase 5: CI/CD Workflows

### `.github/workflows/ci.yml`
- Trigger: push to `main`/`dev`, PR to `main`
- Matrix: Python 3.11 and 3.12
- Steps: `uv sync`, `ruff check`, `mypy`, `pytest --cov`, upload to Codecov
- Build job (after tests): `uv build`, `twine check dist/*`, upload artifacts

### `.github/workflows/publish.yml`
- Trigger: tag push matching `v*.*.*`
- Permissions: `id-token: write` (OIDC trusted publishing — no API token secrets needed)
- Jobs: build → publish TestPyPI → publish PyPI
- Uses `pypa/gh-action-pypi-publish@release/v1` with GitHub Environments `testpypi` and `pypi`
- Setup: configure "pending publisher" in PyPI/TestPyPI web UIs pointing to `publish.yml`

---

## File Creation Order (strict dependency order)

| # | File |
|---|------|
| 1 | `pyproject.toml`, `.python-version`, `.gitignore`, `README.md` |
| 2 | `.github/workflows/ci.yml`, `.github/workflows/publish.yml` |
| 3 | `src/tnjax/core/symmetry.py` |
| 4 | `src/tnjax/core/index.py` |
| 5 | `src/tnjax/core/tensor.py` |
| 6 | `src/tnjax/contraction/contractor.py` |
| 7 | `src/tnjax/network/network.py` |
| 8 | `src/tnjax/algorithms/dmrg.py` |
| 9 | `src/tnjax/algorithms/trg.py` |
| 10 | `src/tnjax/algorithms/hotrg.py` |
| 11 | `src/tnjax/algorithms/ipeps.py` |
| 12 | All `__init__.py` files |
| 13 | `tests/conftest.py` |
| 14–22 | All test files |

---

## Test Strategy

| Module | Test Type | Correctness Reference |
|--------|-----------|-----------------------|
| `symmetry.py` | Unit + hypothesis (group axioms) | Associativity, identity, inverse |
| `index.py` | Unit | Charge arithmetic, duality |
| `tensor.py` | Unit + JIT compat | Dense vs. symmetric norm/todense |
| `contractor.py` | Unit + monkeypatch | Dense einsum reference |
| `network.py` | Integration | Manual graph inspection |
| `dmrg.py` | Integration | Exact diagonalization, L=4 Heisenberg |
| `trg.py` | Integration | Onsager exact free energy for 2D Ising |
| `hotrg.py` | Integration | TRG at same chi + Onsager |
| `ipeps.py` | Integration | CTM convergence, energy sign |

**Coverage target:** 90%+ line coverage. Hypothesis tests use `@settings(max_examples=200)` in CI.

---

## Key Architectural Trade-offs

1. **Label-based contraction (Cytnx-style) over positional indexing:** Users assign meaningful string/integer labels to tensor legs. `contract(A, B)` automatically identifies shared labels and contracts them — no einsum strings or `(tensor, leg_pos)` pairs. Labels propagate to output tensors and to the new bonds created by `truncated_svd` / `qr_decompose`. Users must explicitly relabel when a bond should not be contracted.

2. **`dict[BlockKey, jax.Array]` over JAX BCOO:** Block-sparse (not element-sparse) structure is natural for symmetric tensors; BCOO is optimized for element sparsity. Dict with pytree registration gives per-block JIT and no memory padding waste.

3. **Python loops in DMRG, JIT inside:** Bond dimensions change after SVD truncation, preventing `jax.lax.scan` for the sweep. JIT on matvec only; recompilation occurs only on bond dimension change (which stabilizes quickly).

4. **`jax.lax.while_loop` in Lanczos:** Pre-allocates Krylov basis of shape `(max_iter, hilbert_dim)`; convergence masking avoids dynamic shapes inside JIT.

5. **networkx for graph + label-based edges:** Graph edges are keyed by `(node_a, label_a, node_b, label_b)` — no positional leg index in the graph. `connect_by_shared_label()` auto-connects when two tensors share a label, mimicking Cytnx's natural workflow. Cache key is `frozenset[NodeId]` for O(1) lookup.

6. **`contract_path` separate from execution:** opt_einsum's `contract_path` runs at Python level (no JIT overhead), then actual contraction uses the precomputed path with `backend="jax"`. The label→subscript translation (`_labels_to_subscripts`) feeds into this pipeline transparently.

---

## Verification

After implementation, verify end-to-end:

```bash
# Install in dev mode
uv sync --all-extras --dev
uv run python -c "import tnjax; print(tnjax.__version__)"

# Run full test suite with coverage
uv run pytest tests/ -v --cov=tnjax --cov-report=term-missing

# Lint
uv run ruff check src/ tests/

# Build and check distribution
uv build
uv run twine check dist/*

# Quick integration test (DMRG)
uv run python -c "
from tnjax import U1Symmetry, DMRGConfig
from tnjax.algorithms.dmrg import dmrg, build_mpo_heisenberg, ...
# Run L=4 Heisenberg DMRG, check energy vs exact
"
```
