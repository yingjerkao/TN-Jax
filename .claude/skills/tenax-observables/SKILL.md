---
name: tenax-observables
description: >
  Guide users through computing physical observables from Tenax ground states:
  local expectation values, correlation functions, entanglement entropy, and
  order parameters from MPS (DMRG/iDMRG) and iPEPS states. Use this skill
  when the user asks "how do I compute observables", "measure correlation
  function", "entanglement entropy", "expectation value", "order parameter",
  "magnetization", "structure factor", or "what can I measure from my ground
  state". Also trigger for "post-DMRG analysis", "reduced density matrix",
  or "how to extract physics from MPS".
---

# Observables and Measurements in Tenax

Guide users through extracting physical quantities from optimized tensor
network states. Tenax provides the ground state as a TensorNetwork (MPS)
or raw tensor (iPEPS); measuring observables requires contracting the
state with operator insertions.

## What's Available from Algorithm Results

### DMRG → DMRGResult

```python
result = dmrg(mpo, mps, config)
result.energy              # Total ground state energy (float)
result.energies_per_sweep  # Energy history (list[float])
result.mps                 # Optimized MPS (TensorNetwork)
result.truncation_errors   # Truncation errors per bond update
result.converged           # Whether DMRG converged (bool)
```

### iDMRG → iDMRGResult

```python
result = idmrg(W, config)
result.energy_per_site     # Energy per site (float)
result.energies_per_step   # Energy history
result.mps_tensors         # [A_L, A_R] unit cell tensors
result.singular_values     # Centre bond singular values (jax.Array)
result.converged           # bool
```

### iPEPS → (energy, peps, env)

```python
energy, peps, env = ipeps(gate, None, config)
# or
A_opt, env, E_gs = optimize_gs_ad(gate, None, config)
```

The CTM environment `env` enables observable computation via reduced
density matrices.

---

## Entanglement Entropy from iDMRG

The singular values on the centre bond give the entanglement entropy
directly — no extra contraction needed.

```python
import jax.numpy as jnp

S = result.singular_values
# Normalize
S_norm = S / jnp.linalg.norm(S)
# Von Neumann entropy
p = S_norm**2
S_ent = -jnp.sum(p * jnp.log(p))
print(f"Entanglement entropy: {S_ent:.6f}")

# For critical systems: S = (c/3) ln(xi) where c is the central charge
# and xi is the correlation length (related to bond dimension)
```

---

## Local Expectation Values from MPS

To compute ⟨ψ|O_i|ψ⟩ for a local operator at site i, insert the operator
into the MPS contraction using `NetworkBlueprint`:

```python
import numpy as np
from tenax import NetworkBlueprint, contract

# Example: ⟨Sz_i⟩ for an MPS with L sites
Sz = np.array([[0.5, 0.0], [0.0, -0.5]])

def measure_local(mps, op, site):
    """Compute ⟨ψ|op_site|ψ⟩ from a DMRG MPS TensorNetwork."""
    # Strategy: contract the MPS with its conjugate, inserting op at the
    # target site. For efficiency, build left and right environments
    # iteratively.
    L = mps.n_nodes()

    # Start with trivial left environment
    left_env = None
    for i in range(L):
        A = mps.get_tensor(f"site_{i}")
        A_conj = A.conj()

        if i == site:
            # Insert operator: contract op with ket physical leg
            # This requires manual contraction with the operator matrix
            pass  # See full implementation below

        # Contract into running environment
        # ... (site-by-site left-to-right sweep)

    return expectation_value
```

### Practical approach using contract()

For small systems, the simplest approach is to contract the full MPS
into a state vector and use standard linear algebra:

```python
import jax.numpy as jnp

def mps_to_vector(mps):
    """Contract an MPS TensorNetwork into a full state vector.
    Only practical for L <= ~20 (2^L memory).
    """
    result = mps.contract()
    return result.todense().flatten()

psi = mps_to_vector(result.mps)
# ⟨Sz_i⟩
Sz_full = build_full_operator(Sz, site=i, L=L)  # Embed in full Hilbert space
expectation = jnp.real(psi.conj() @ Sz_full @ psi)
```

This is only feasible for small systems (L ≤ ~18 for spin-1/2). For
larger systems, use the environment-based approach above.

---

## Two-Point Correlation Functions

⟨S^z_i S^z_j⟩ measures spin-spin correlations and reveals the nature of
order in the ground state.

### From a full state vector (small systems)

```python
def correlation_function(psi, op1, op2, site_i, site_j, L, d=2):
    """Compute ⟨ψ|op1_i op2_j|ψ⟩."""
    from scipy.sparse import kron, eye, csr_matrix

    I = eye(d)
    ops = [I] * L
    ops[site_i] = csr_matrix(op1)
    ops[site_j] = csr_matrix(op2)

    O = ops[0]
    for k in range(1, L):
        O = kron(O, ops[k])

    return float(jnp.real(psi.conj() @ O @ psi))

# Example: ⟨Sz_0 Sz_r⟩ for all r
Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
correlations = [correlation_function(psi, Sz, Sz, 0, r, L) for r in range(L)]
```

### Expected behavior

| Phase | ⟨S^z_0 S^z_r⟩ | Signature |
|-------|---------------|-----------|
| Antiferromagnetic (gapless, 1D) | ∼ (-1)^r / r | Algebraic decay |
| Gapped (e.g., Haldane) | ∼ e^{-r/ξ} | Exponential decay |
| Ferromagnetic | → const > 0 | Long-range order |
| Néel ordered (2D) | → (-1)^r m² | Staggered long-range order |

---

## iPEPS Observables via CTM

For iPEPS, observables are computed using the CTM environment and reduced
density matrices.

### Energy (built-in)

```python
from tenax import compute_energy_ctm_2site

# For 2-site unit cell
energy = compute_energy_ctm_2site(peps_a, peps_b, gate, env_a, env_b)
```

### Custom observables from RDMs

The CTM environment enables computing 2-site reduced density matrices
(RDMs). From a 2-site RDM ρ, any 2-site observable is:

    ⟨O⟩ = Tr(ρ · O)

```python
# Staggered magnetization for Néel order
Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]])
# m_s = (1/2)|⟨Sz_A⟩ - ⟨Sz_B⟩| for A-B sublattice
```

---

## Order Parameters

### Staggered magnetization (Néel order)

```python
# m_stag = (1/L) Σ (-1)^i ⟨Sz_i⟩
m_stag = sum((-1)**i * expect_Sz[i] for i in range(L)) / L
```

For the 1D Heisenberg chain, m_stag → 0 as L → ∞ (no Néel order in 1D,
Mermin-Wagner theorem). For 2D (iPEPS with 2-site unit cell), m_stag ≈ 0.307.

### String order parameter (Haldane phase)

For spin-1 chains, the Haldane phase has hidden topological order detected
by the string order parameter:

    O_string = ⟨S^z_i exp(iπ Σ_{k=i+1}^{j-1} S^z_k) S^z_j⟩

This requires inserting a non-local string of operators into the MPS
contraction — a good exercise for the NetworkBlueprint.

---

## Structure Factor

The static structure factor S(q) is the Fourier transform of the
real-space correlation function:

```python
import numpy as np

# From real-space correlations C(r) = ⟨Sz_0 Sz_r⟩
def structure_factor(correlations, q_values, L):
    """S(q) = Σ_r exp(iqr) C(r)"""
    S_q = []
    for q in q_values:
        s = sum(np.exp(1j * q * r) * correlations[r] for r in range(L))
        S_q.append(float(np.real(s)))
    return S_q

q_values = np.linspace(0, 2 * np.pi, 100)
S_q = structure_factor(correlations, q_values, L)
# Peak at q=π → antiferromagnetic correlations
```

---

## Practical Workflow

1. **Run DMRG/iDMRG/iPEPS** to get the ground state.
2. **Check energy** — compare against known exact results or extrapolate
   in bond dimension.
3. **Compute entanglement entropy** — from singular values (iDMRG) or
   by SVD of the MPS at a bond (finite DMRG).
4. **Measure correlations** — ⟨S^z_0 S^z_r⟩ to identify the phase.
5. **Compute order parameters** — staggered magnetization, string order, etc.
6. **Structure factor** — Fourier transform of correlations for momentum-space
   picture.

## Pedagogical Notes

- Observables are where the physics lives. The energy alone doesn't tell you
  what phase you're in — you need correlations and order parameters.
- For 1D systems, the entanglement entropy scaling (constant vs logarithmic)
  immediately tells you if the system is gapped or gapless.
- Always compare observables across multiple bond dimensions to assess
  convergence. If ⟨Sz_0 Sz_r⟩ changes significantly when you double χ,
  you need more bond dimension.
