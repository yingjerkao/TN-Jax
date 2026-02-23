# TN-Jax Revised Implementation Plan (Phased, Risk-Gated)

## 1. Objective

Build a reliable, extensible JAX tensor-network library with:
- Label-based contraction API for usability.
- Strong correctness guarantees for symmetry-aware block-sparse tensors.
- Incremental algorithm rollout with measurable acceptance gates.

This plan prioritizes correctness and maintainability over maximum feature count in `v0.1`.

## 2. Scope by Release

### `v0.1` (Production baseline)
- Core symmetry/index/tensor abstractions.
- Dense tensor contraction by labels.
- TensorNetwork graph abstraction with explicit edge semantics.
- Dense DMRG (small-system validated).
- CI/CD, packaging, docs, and test harness.

### `v0.2` (Symmetry + block-sparse core)
- U(1) and Zn symmetry-complete block-sparse tensors.
- Symmetry-aware contraction (dense parity tests + performance sanity checks).
- Symmetric DMRG.

### `v0.3` (2D coarse graining)
- TRG + HOTRG (dense first, optional symmetric extension).

### `v0.4` (iPEPS stack)
- iPEPS + CTM with quantitative validation criteria.

## 3. Non-Goals (until later release)

- Full non-Abelian implementation (interface only).
- Maximum-performance kernel fusion on day one.
- Large-scale benchmark leadership claims before numerical validation suite is complete.

## 4. Core Design Decisions (Revised)

1. **Contraction semantics are formalized early**
- Direct `contract(*tensors)` uses label matching.
- Network contraction uses graph edges as source of truth.
- Label matching in network mode is only a helper for constructing edges.

2. **Label multiplicity rules**
- Default: a label may appear any number of times.
- Pairwise-only mode available as strict validator for debugging.
- Multi-occurrence labels require explicit policy (`trace`, `sum_all`, or `error`) to avoid ambiguity.

3. **Stable identity and hashing for index metadata**
- `TensorIndex.charges` stored in immutable hash-stable form (e.g., tuple-backed internal representation).
- Dataclass equality/hash guaranteed deterministic across runs.

4. **Symmetry conservation uses symmetry operations, not raw arithmetic**
- No hardcoded `sum(flow * charge) == identity`.
- Validation uses each symmetry’s `fuse`/`dual` behavior (Zn modulo behavior included by construction).

5. **Block enumeration is lazy/reachable-sector based**
- Avoid full cartesian sector generation.
- Construct blocks from reachable fused sectors and active support.

6. **Contraction backend avoids subscript-symbol bottlenecks**
- Use expression/index-form APIs where possible (not single-string symbol mapping only).
- Keep label order explicit and reproducible.

7. **Cache keys include full state dependencies**
- Include graph revision, tensor revision/version, selected nodes, output label order, and optimize mode.

## 5. Implementation Phases

## Phase A: Foundations and Tooling

### Deliverables
- Project scaffolding (`pyproject.toml`, CI workflows, package layout).
- Lint/type/test setup (`ruff`, `mypy`, `pytest`, coverage).
- Minimal docs and API conventions.

### Acceptance Criteria
- `uv sync --all-extras --dev` succeeds.
- CI green on Python 3.11 and 3.12.
- Import smoke test passes.

### Risks and Gates
- Risk: unstable dependency matrix.
- Gate: lock dependencies and require reproducible install in CI.

## Phase B: Core Types + Dense Contraction

### Deliverables
- `symmetry.py`, `index.py`, dense portion of `tensor.py`.
- `contractor.py` dense path with label-based API.
- Formal contraction semantics document in repo (`docs/contraction_semantics.md`).

### Acceptance Criteria
- Property tests for group axioms and dual/fuse consistency.
- Dense contraction equals reference `jax.numpy.einsum` results.
- Deterministic output label ordering behavior tested.

### Risks and Gates
- Risk: ambiguous label behavior creates silent wrong results.
- Gate: strict validation mode enabled in tests; ambiguity must throw with actionable error.

## Phase C: Network Layer

### Deliverables
- `TensorNetwork` with explicit edge registry and graph revisioning.
- Clear contract: edge-defined contraction in network mode.
- Edge-label synchronization utilities (`connect`, `disconnect`, `relabel_bond`).

### Acceptance Criteria
- Network contraction equals direct contraction for equivalent topology.
- Cache correctness tests prove no stale retrieval after mutation.
- Topology mutation invalidates dependent cache entries.

### Risks and Gates
- Risk: cache key collisions or stale values.
- Gate: mutation-fuzz tests over add/remove/relabel/connect/disconnect.

## Phase D: Dense DMRG (Reference Algorithm)

### Deliverables
- Two-site dense DMRG with Lanczos inner solve.
- MPO builder for Heisenberg chain.
- Sweep diagnostics (`energies_per_sweep`, truncation errors, convergence flags).

### Acceptance Criteria
- Matches exact diagonalization on small chains (e.g., `L<=8`) within tolerance.
- Energy monotonicity/expected descent behavior documented and tested.
- Reproducible results with fixed PRNG seed.

### Risks and Gates
- Risk: incorrect environment updates causing plausible but wrong energies.
- Gate: per-sweep invariant checks and small-system cross-validation.

## Phase E: Symmetric Block-Sparse Core + Symmetric DMRG

### Deliverables
- `SymmetricTensor` with lazy block support.
- Symmetry-aware contraction path (U(1), Zn).
- Symmetric DMRG path parity with dense baseline where comparable.

### Acceptance Criteria
- `from_dense(...).todense()` roundtrip within tolerance.
- Symmetric vs dense parity for small tractable problems.
- Runtime/memory sanity check shows benefit over dense at representative settings.

### Risks and Gates
- Risk: incorrect sector selection or dropped blocks.
- Gate: exhaustive small-sector enumeration tests against dense truth tables.

## Phase F: TRG + HOTRG

### Deliverables
- Dense TRG and HOTRG reference implementations.
- Optional symmetric extension behind explicit feature flag.

### Acceptance Criteria
- 2D Ising free-energy estimates converge toward analytic benchmark over steps.
- Regression tests for stability versus `chi` and step count.

### Risks and Gates
- Risk: numerically stable but physically wrong fixed points.
- Gate: benchmark curves checked against tolerance bands, not sign-only checks.

## Phase G: iPEPS + CTM

### Deliverables
- Simple-update iPEPS loop.
- CTM environment iteration with convergence diagnostics.
- Energy evaluation routines.

### Acceptance Criteria
- Convergence measured by environment/tensor fixed-point residuals.
- Energy compared against trusted baselines in small benchmark cases.
- Deterministic run mode for CI-compatible tests.

### Risks and Gates
- Risk: weak correctness oracle.
- Gate: require quantitative benchmark thresholds before release labeling.

## 6. Testing Strategy (Upgraded)

1. **Unit tests**
- Symmetry operations, index duality, label operations, cache key behavior.

2. **Property tests**
- Group-like laws where applicable, contraction invariants, relabel invariants.

3. **Parity tests**
- Dense vs symmetric parity on small systems.

4. **Algorithmic correctness tests**
- DMRG vs exact diagonalization.
- TRG/HOTRG vs known Ising free-energy reference values.
- iPEPS/CTM convergence metrics and benchmark windows.

5. **Mutation and fuzz-style graph tests**
- Random network edits with cache correctness assertions.

## 7. CI/CD and Quality Gates

- CI matrix: Python 3.11, 3.12.
- Required checks: `ruff`, `mypy`, `pytest`, coverage threshold.
- Coverage target:
- `v0.1`: >= 85% overall.
- `v0.2+`: >= 90% overall with no critical-module drop below 85%.
- Build verification: `uv build` and `twine check`.
- Publish only from signed/tagged release workflow after all gates pass.

## 8. File Creation / Execution Order (Revised)

1. Project scaffolding and CI files.
2. `core/symmetry.py`, `core/index.py`, dense `core/tensor.py`.
3. Dense `contraction/contractor.py` and semantics doc.
4. `network/network.py` with revisioned cache.
5. `algorithms/dmrg.py` dense baseline + tests.
6. Symmetric tensor + symmetric contraction.
7. Symmetric DMRG.
8. TRG/HOTRG.
9. iPEPS/CTM.

## 9. Release Readiness Checklist

- API docs include explicit contraction semantics and ambiguity policy.
- Numerical benchmarks recorded and versioned.
- Reproducibility instructions include seed control and hardware notes.
- Known limitations listed per release.

## 10. Immediate Next Actions

1. Lock `v0.1` scope to Phases A-D only.
2. Add `docs/contraction_semantics.md` before implementing network logic.
3. Implement deterministic hash/equality behavior for `TensorIndex`.
4. Build dense DMRG validation suite before starting block-sparse implementation.
