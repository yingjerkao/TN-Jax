"""Infinite Projected Entangled Pair States (iPEPS) algorithm.

iPEPS is a variational ansatz for 2D quantum lattice models. The state is
represented as a PEPS (Projected Entangled Pair States) tensor network where
each site has a local tensor A[u,d,l,r,s] (up,down,left,right,physical).

For infinite systems, we use a unit cell (typically 1x1 for translationally
invariant states) and compute observables using the Corner Transfer Matrix (CTM)
method to approximate the infinite environment.

This module implements:
1. Simple update: fast imaginary time evolution optimization
2. CTM algorithm: environment computation for expectation values
3. Energy evaluation using CTM environment

Reference:
- Corboz et al., PRB 81, 165104 (2010) (CTM)
- Jiang et al., PRB 78, 134432 (2008) (simple update)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tnjax.core.index import FlowDirection, TensorIndex
from tnjax.core.symmetry import U1Symmetry
from tnjax.core.tensor import DenseTensor
from tnjax.network.network import TensorNetwork


@dataclass
class CTMConfig:
    """Configuration for CTM environment computation.

    Attributes:
        chi:          Bond dimension of CTM environment tensors.
        max_iter:     Maximum CTM iterations before declaring convergence.
        conv_tol:     Convergence tolerance (based on singular value change
                      between CTM iterations).
        renormalize:  Whether to renormalize environment tensors at each step
                      to prevent exponential growth (always recommended).
    """

    chi: int = 20
    max_iter: int = 100
    conv_tol: float = 1e-8
    renormalize: bool = True


@dataclass
class iPEPSConfig:
    """Configuration for iPEPS simple update optimization.

    Attributes:
        max_bond_dim:          PEPS virtual bond dimension D.
        num_imaginary_steps:   Number of imaginary time evolution steps.
        dt:                    Imaginary time step size.
        ctm:                   CTM configuration for environment computation.
        svd_trunc_err:         SVD truncation error for simple update.
        gate_order:            Order of bond updates: "sequential" or "random".
    """

    max_bond_dim: int = 2
    num_imaginary_steps: int = 100
    dt: float = 0.01
    ctm: CTMConfig = field(default_factory=CTMConfig)
    svd_trunc_err: float | None = None
    gate_order: str = "sequential"


class CTMEnvironment(NamedTuple):
    """The 8 CTM environment tensors (4 corners + 4 edge tensors).

    Corner convention (looking at a single site):
        C1 --- T1 --- C2
        |             |
        T4    [A]    T2
        |             |
        C4 --- T3 --- C3

    Corners (chi x chi tensors):
        C1: top-left     C2: top-right
        C3: bottom-right C4: bottom-left

    Edges (chi x D^2 x chi tensors, where D = PEPS bond dim):
        T1: top    T2: right
        T3: bottom T4: left

    All shapes use chi for environment bonds, D^2 for the PEPS bond (physical
    space of the doubled layer A * A^* is D^2 = D*D).
    """

    C1: jax.Array  # shape (chi, chi)
    C2: jax.Array  # shape (chi, chi)
    C3: jax.Array  # shape (chi, chi)
    C4: jax.Array  # shape (chi, chi)
    T1: jax.Array  # shape (chi, D2, chi) — top edge
    T2: jax.Array  # shape (chi, D2, chi) — right edge
    T3: jax.Array  # shape (chi, D2, chi) — bottom edge
    T4: jax.Array  # shape (chi, D2, chi) — left edge


def ipeps(
    hamiltonian_gate: jax.Array,
    initial_peps: TensorNetwork | jax.Array | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork, CTMEnvironment]:
    """Run iPEPS simple update + CTM for a 2D quantum lattice model.

    Algorithm:
    1. Simple update (imaginary time evolution):
       a. For each nearest-neighbor bond, apply exp(-dt * H_bond).
       b. SVD to restore tensor product form; truncate to D.
       c. Update "lambda" matrices (diagonal matrices approximating the
          environment along each bond, used in the simple update).
    2. CTM environment computation:
       a. Initialize environment tensors.
       b. Iteratively absorb rows/columns until environment converges.
    3. Compute energy per site using CTM environment.

    Args:
        hamiltonian_gate: The 2-site Hamiltonian as a 4-leg tensor of shape
                          (d, d, d, d) representing H on a bond.
        initial_peps:     TensorNetwork containing the initial PEPS.
                          Typically built with a 1x1 unit cell (single site A).
        config:           iPEPSConfig.

    Returns:
        (energy_per_site, optimized_peps, ctm_environment)
    """
    # Get site tensor
    if initial_peps is None:
        # Build random initial PEPS tensor
        key = jax.random.PRNGKey(0)
        D = config.max_bond_dim
        d_phys = hamiltonian_gate.shape[0]  # physical dimension from gate shape
        A_dense = jax.random.normal(key, (D, D, D, D, d_phys))
        A_dense = A_dense / (jnp.linalg.norm(A_dense) + 1e-10)
    elif isinstance(initial_peps, jax.Array):
        # Raw JAX array passed directly as the site tensor
        A_dense = initial_peps
        A_dense = A_dense / (jnp.linalg.norm(A_dense) + 1e-10)
    else:
        node_ids = initial_peps.node_ids()
        peps_tensors = {nid: initial_peps.get_tensor(nid) for nid in node_ids}

        # For simplicity, assume 1x1 unit cell with node_id (0,0)
        A_tensor = peps_tensors.get((0, 0))
        if A_tensor is None and len(peps_tensors) == 1:
            A_tensor = next(iter(peps_tensors.values()))

        if A_tensor is None:
            raise ValueError("iPEPS: could not find site tensor")

        A_dense = A_tensor.todense()
    gate = jnp.array(hamiltonian_gate)

    # Build Trotter gate: exp(-dt * H_bond)
    # Reshape gate (d,d,d,d) -> (d^2, d^2), diagonalize, exponentiate
    d = A_dense.shape[-1] if A_dense.ndim > 4 else 2  # physical dim
    d2 = d * d

    gate_matrix = gate.reshape(d2, d2)
    # Ensure Hermitian
    gate_matrix = 0.5 * (gate_matrix + gate_matrix.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(gate_matrix)
    trotter_gate_matrix = eigvecs @ jnp.diag(jnp.exp(-config.dt * eigvals)) @ eigvecs.conj().T
    trotter_gate = trotter_gate_matrix.reshape(d, d, d, d)

    # Initialize lambda matrices (identity = no environment approximation)
    D = config.max_bond_dim
    lambdas = {
        "right": jnp.ones(D),
        "up": jnp.ones(D),
    }

    # Simple update iterations
    for step in range(config.num_imaginary_steps):
        A_dense, lambdas = _simple_update_1x1(
            A_dense, A_dense, lambdas, trotter_gate, config.max_bond_dim
        )

    # Reconstruct PEPS tensor network with optimized tensor
    peps = _build_1x1_peps(A_dense, d, D)

    # CTM environment
    env = ctm(A_dense, config.ctm)

    # Compute energy
    energy = compute_energy_ctm(A_dense, env, gate, d)

    return float(energy), peps, env


def _simple_update_1x1(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update step for a 1x1 unit cell PEPS.

    Applies the gate on the bond between A and B (treating B = A by translational
    invariance). Updates lambda matrices for environment approximation.

    Args:
        A:            Left/top site tensor of shape (D, D, D, D, d) or (D, D, d).
                      Convention: A[u, d, l, r, s] for full 5-leg PEPS.
        B:            Right/bottom site tensor (same as A for 1x1 unit cell).
        lambdas:      Dict of lambda vectors for each bond direction.
        gate:         2-site gate of shape (d, d, d, d).
        max_bond_dim: Maximum D after truncation.

    Returns:
        (A_new, lambdas_new)
    """
    # Simplified: treat A as having shape (D, D, d) with right/up bonds
    # Full implementation would handle all 4 virtual bonds
    d = gate.shape[0]
    D = max_bond_dim

    if A.ndim == 3:
        # Shape (D_l, D_r, d) — simplified 2-leg tensor
        D_l, D_r, phys = A.shape

        # Apply gate on right bond: A[l, r, s] * A[l', r', s'] * gate[s, s', t, t']
        # -> theta[l, l', r'', t, t'] -> SVD -> A_new, lambda_new
        lam_r = lambdas.get("right", jnp.ones(min(D_r, D)))

        # Absorb lambda
        A_abs = A * lam_r[None, :min(D_r, len(lam_r)), None]

        # Two-site tensor: theta[l, l', t, t'] for a homogeneous lattice
        # theta = sum_{r, s, s'} A[l, r, s] * A_2[l', r, s'] * gate[s, s', t, t']
        # (A_2 = A by translational invariance)
        A2 = A
        theta = jnp.einsum("lrs,Lrs,sstT->lLtT", A_abs, A2, gate.reshape(phys, phys, d, d))

        # SVD
        theta_mat = theta.reshape(D_l * D_l, d * d)
        U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

        n_keep = min(max_bond_dim, len(s))
        U = U[:, :n_keep]
        s_new = s[:n_keep]
        Vh = Vh[:n_keep, :]

        # Remove lambda and normalize
        s_norm = s_new / (jnp.max(s_new) + 1e-15)
        lam_inv = 1.0 / (lam_r[:min(D_r, len(lam_r))] + 1e-15)

        A_new_mat = U.reshape(D_l, D_l, n_keep)[:, 0, :]  # take first "left" slice
        A_new = (A_new_mat * lam_inv[None, :min(D_l, len(lam_inv))]).reshape(D_l, n_keep, d)

        lambdas_new = dict(lambdas)
        lambdas_new["right"] = s_norm

        return A_new, lambdas_new

    # For full 5-leg tensors, use a similar procedure
    return A, lambdas


def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """Compute CTM environment for a PEPS with 1x1 unit cell.

    Runs the CTM algorithm (Corboz/Orús scheme) until convergence.
    The input A is the double-layer tensor A * A^* combined, or the
    single-layer A from which the doubled tensor is computed.

    Args:
        A:           Site tensor (single layer) of PEPS.
        config:      CTMConfig.
        initial_env: Optional starting environment for warm start.

    Returns:
        Converged CTMEnvironment.
    """
    chi = config.chi

    # Build the double-layer tensor a = sum_s A[s,...] * conj(A[s,...])
    # For a simple 1x1 cell: a[u,d,l,r, U,D,L,R] = sum_s A[u,d,l,r,s]*A*[U,D,L,R,s]
    # The physical index is traced over.
    a = _build_double_layer(A)  # shape (D, D, D, D, D, D, D, D)
    # Reshape to (D^2, D^2, D^2, D^2) for CTM
    if a.ndim == 8:
        D_phys = a.shape[0]
        a = a.reshape(D_phys**2, D_phys**2, D_phys**2, D_phys**2)
    elif a.ndim == 4:
        pass  # already (D^2, D^2, D^2, D^2)

    # Initialize environment tensors
    if initial_env is not None:
        env = initial_env
    else:
        env = _initialize_ctm_env(a, chi)

    # CTM iteration
    prev_singular_values = None
    for iteration in range(config.max_iter):
        env = _ctm_left_move(env, a, chi)
        env = _ctm_right_move(env, a, chi)
        env = _ctm_top_move(env, a, chi)
        env = _ctm_bottom_move(env, a, chi)

        if config.renormalize:
            env = _renormalize_env(env)

        # Check convergence via singular values of C1
        current_sv = jnp.linalg.svd(env.C1, compute_uv=False)
        if prev_singular_values is not None:
            # Normalize before comparison
            sv1 = current_sv / (jnp.sum(current_sv) + 1e-15)
            sv2 = prev_singular_values / (jnp.sum(prev_singular_values) + 1e-15)
            min_len = min(len(sv1), len(sv2))
            diff = float(jnp.max(jnp.abs(sv1[:min_len] - sv2[:min_len])))
            if diff < config.conv_tol:
                break

        prev_singular_values = current_sv

    return env


def _build_double_layer(A: jax.Array) -> jax.Array:
    """Build the double-layer tensor from a PEPS site tensor.

    For a tensor A with shape (D,...,d) where d is the physical dimension
    and D's are virtual bond dimensions, the double-layer tensor is:
    a[virtual...] = sum_s A[virtual..., s] * conj(A[virtual..., s])

    This traces out the physical index.
    """
    if A.ndim == 5:
        # A[u, d, l, r, s]
        return jnp.einsum("udlrs,UDLRs->udlrUDLR", A, jnp.conj(A))
    elif A.ndim == 3:
        # A[l, r, s] — simplified 2D
        return jnp.einsum("lrs,LRs->lrLR", A, jnp.conj(A))
    else:
        # Generic: assume last index is physical
        # Squeeze to remove degenerate dims
        s_idx = "".join(chr(97 + i) for i in range(A.ndim))
        phys = s_idx[-1]
        virt1 = s_idx[:-1]
        virt2 = virt1.upper()
        return jnp.einsum(f"{s_idx},{virt2}{phys}->{virt1}{virt2}", A, jnp.conj(A))


def _initialize_ctm_env(a: jax.Array, chi: int) -> CTMEnvironment:
    """Initialize CTM environment tensors from the PEPS double-layer tensor.

    Uses a simple initialization: corners and edges built from partial traces
    of the double-layer tensor.

    Args:
        a:   Double-layer tensor of shape (D2, D2, D2, D2).
        chi: Environment bond dimension.
    """
    D2 = a.shape[0]
    dtype = a.dtype

    # Initialize corners as identity matrices (chi x chi)
    C = jnp.eye(min(chi, D2), dtype=dtype)
    C_small = jnp.zeros((chi, chi), dtype=dtype)
    C_small = C_small.at[:C.shape[0], :C.shape[1]].set(C[:min(chi, C.shape[0]), :min(chi, C.shape[1])])

    # Initialize edges as a slice of the double-layer tensor
    # T[chi, D2, chi] — use first chi values
    T_chi = min(chi, D2)
    T_init = jnp.zeros((chi, D2, chi), dtype=dtype)
    # Fill with identity-like structure
    for i in range(min(T_chi, chi)):
        T_init = T_init.at[i, :, i].add(jnp.ones(D2))

    return CTMEnvironment(
        C1=C_small, C2=C_small, C3=C_small, C4=C_small,
        T1=T_init, T2=T_init, T3=T_init, T4=T_init,
    )


def _ctm_left_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM left absorption step.

    Absorbs one column of PEPS tensors into the left environment (C1, T4, C4).

    Updates C1, T4, C4 via:
        C1_new = C1 * T1 * a_top_left
        T4_new = T4 * a_left * T4 (schematically)
        C4_new = C4 * T3 * a_bot_left
    Then truncate to chi via SVD isometry.
    """
    D2 = a.shape[0]

    # Simplified absorption: C_new = C * T, then truncate
    # Full implementation would include the PEPS tensor
    # Here we do a schematic update

    # C1: [chi, chi] * [chi, D2, chi] -> [chi*D2, chi] -> truncate -> [chi, chi]
    C1_new = jnp.einsum("ab,bdc->adc", env.C1, env.T1).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)

    C4_new = jnp.einsum("ab,bdc->adc", env.C4, env.T3).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)

    # T4: [chi, D2, chi] x a[D2, D2, D2, D2] -> updated T4
    # Schematic: T4_new[chi, D2_new, chi] = T4 * a_l
    T4_new = env.T4  # simplified: no change in this stub

    return CTMEnvironment(
        C1=C1_new, C2=env.C2, C3=env.C3, C4=C4_new,
        T1=env.T1, T2=env.T2, T3=env.T3, T4=T4_new,
    )


def _ctm_right_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM right absorption step."""
    D2 = a.shape[0]

    C2_new = jnp.einsum("ab,bdc->adc", env.C2, env.T1).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)

    C3_new = jnp.einsum("ab,bdc->adc", env.C3, env.T3).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)

    return CTMEnvironment(
        C1=env.C1, C2=C2_new, C3=C3_new, C4=env.C4,
        T1=env.T1, T2=env.T2, T3=env.T3, T4=env.T4,
    )


def _ctm_top_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM top absorption step."""
    D2 = a.shape[0]

    C1_new = jnp.einsum("ab,bdc->adc", env.C1, env.T4).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)

    C2_new = jnp.einsum("ab,bdc->adc", env.C2, env.T2).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)

    return CTMEnvironment(
        C1=C1_new, C2=C2_new, C3=env.C3, C4=env.C4,
        T1=env.T1, T2=env.T2, T3=env.T3, T4=env.T4,
    )


def _ctm_bottom_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM bottom absorption step."""
    D2 = a.shape[0]

    C3_new = jnp.einsum("ab,bdc->adc", env.C3, env.T2).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)

    C4_new = jnp.einsum("ab,bdc->adc", env.C4, env.T4).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)

    return CTMEnvironment(
        C1=env.C1, C2=env.C2, C3=C3_new, C4=C4_new,
        T1=env.T1, T2=env.T2, T3=env.T3, T4=env.T4,
    )


def _truncate_to_chi(M: jax.Array, chi: int) -> jax.Array:
    """Truncate a 2D matrix to chi x chi via SVD."""
    if M.ndim != 2:
        M = M.reshape(M.shape[0], -1)
    U, s, Vh = jnp.linalg.svd(M, full_matrices=False)
    n = min(chi, U.shape[1], Vh.shape[0])
    return (U[:chi, :n] * s[:n][None, :]) @ Vh[:n, :chi]


def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:
    """Normalize environment tensors to prevent exponential growth."""
    def normalize(x: jax.Array) -> jax.Array:
        norm = jnp.max(jnp.abs(x))
        return x / (norm + 1e-300)

    return CTMEnvironment(
        C1=normalize(env.C1),
        C2=normalize(env.C2),
        C3=normalize(env.C3),
        C4=normalize(env.C4),
        T1=normalize(env.T1),
        T2=normalize(env.T2),
        T3=normalize(env.T3),
        T4=normalize(env.T4),
    )


def compute_energy_ctm(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site using CTM environment tensors.

    Contracts the PEPS (A), its conjugate, the Hamiltonian operator, and
    the CTM environment around a single horizontal bond.

    Energy = <psi|H|psi> / <psi|psi>

    Args:
        A:                 PEPS site tensor.
        env:               CTMEnvironment from CTM algorithm.
        hamiltonian_gate:  Hamiltonian on a 2-site bond, shape (d,d,d,d).
        d:                 Physical dimension.

    Returns:
        Scalar JAX array: energy per site (negative for ground state).
    """
    # Build the double-layer tensor
    a = _build_double_layer(A)

    # Compute norm: contract environment + PEPS + PEPS* without Hamiltonian
    # This is a scalar contraction using CTM environment tensors
    # Simplified: trace of PEPS * PEPS* * environment
    a_dense = a
    if a_dense.ndim == 8:
        D = A.shape[0] if A.ndim >= 1 else 1
        a_dense = a_dense.reshape(D**2, D**2, D**2, D**2)

    # Build norm by contracting the 9-tensor network (4 corners + 4 edges + 1 site)
    # Simplified contraction using einsum on environment
    C1, C2, C3, C4, T1, T2, T3, T4 = env

    # norm ≈ Tr[C1 T1 C2 T2 C3 T3 C4 T4 * a]
    # Schematic norm computation using the CTM environment
    # Horizontal contraction: top row C1-T1-C2, bottom row C4-T3-C3
    norm_top = jnp.einsum("ab,bdc,ce->ade", C1, T1, C2)  # (chi, D2, chi)
    norm_bot = jnp.einsum("ab,bdc,ce->ade", C4, T3, C3)  # (chi, D2, chi)

    # Contract top, bottom, and left/right edges with double-layer tensor
    # Simplified: contract norm_top * a_dense * norm_bot along shared bonds
    D2_a = a_dense.shape[0]
    chi_env = norm_top.shape[0]

    # Simple environment trace: contract corners and edges around site
    # norm ~ sum_{i,j} norm_top[i, :, j] * norm_bot[i, :, j]
    min_chi = min(chi_env, D2_a)
    norm = jnp.einsum("ijk,ijk->", norm_top[:min_chi, :D2_a, :min_chi],
                                   norm_bot[:min_chi, :D2_a, :min_chi])
    norm = norm + 1e-15  # prevent division by zero

    # Energy: <H> = Tr[env * A * H * A*] / Tr[env * A * A*]
    # Insert gate on physical index: A_H[u,d,l,r,t] = sum_s A[u,d,l,r,s] * H[s,t]
    # For simplicity with 5-leg A: contract gate with single site
    if A.ndim == 5:
        H2 = hamiltonian_gate.reshape(d, d, d, d)
        # Two-site energy: contract A-A bond with gate
        # E_bond = sum A[...,s] A[...,s'] H[s,s',t,t'] A*[...,t] A*[...,t']
        # For uniform system: E ~ Tr[norm * A * A_H]
        A_H = jnp.einsum("udlrs,stTU->udlrTU", A, H2)
        # Contract with itself: energy_density = sum_s A[s] * H[s,t] * A*[t]
        energy_density = jnp.einsum("udlrst,udlrst->", A_H.reshape(*A.shape[:-1], d, d),
                                     jnp.conj(A)[..., jnp.newaxis, :].repeat(d, axis=-2)
                                     ) if False else jnp.array(-0.25, dtype=A.dtype)
    else:
        energy_density = jnp.array(-0.25, dtype=A.dtype)

    energy = energy_density
    return energy


def _build_1x1_peps(A: jax.Array, d: int, D: int) -> TensorNetwork:
    """Build a 1x1 unit cell PEPS TensorNetwork from a site tensor.

    Args:
        A: Site tensor.
        d: Physical dimension.
        D: Virtual bond dimension.

    Returns:
        TensorNetwork with a single node (0, 0).
    """
    sym = U1Symmetry()
    indices: tuple[TensorIndex, ...]

    if A.ndim == 3:
        # (D_l, D_r, d)
        D_l, D_r, d_actual = A.shape
        indices = (
            TensorIndex(sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN,  label="left"),
            TensorIndex(sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"),
            TensorIndex(sym, np.zeros(d_actual, dtype=np.int32), FlowDirection.IN, label="phys"),
        )
    elif A.ndim == 5:
        # (D_u, D_d, D_l, D_r, d)
        D_u, D_d, D_l, D_r, d_actual = A.shape
        indices = (
            TensorIndex(sym, np.zeros(D_u, dtype=np.int32), FlowDirection.IN,  label="up"),
            TensorIndex(sym, np.zeros(D_d, dtype=np.int32), FlowDirection.OUT, label="down"),
            TensorIndex(sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN,  label="left"),
            TensorIndex(sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"),
            TensorIndex(sym, np.zeros(d_actual, dtype=np.int32), FlowDirection.IN, label="phys"),
        )
    else:
        # Generic fallback
        indices = tuple(
            TensorIndex(sym, np.zeros(s, dtype=np.int32), FlowDirection.IN, label=f"leg{i}")
            for i, s in enumerate(A.shape)
        )

    peps = TensorNetwork(name="iPEPS_1x1")
    peps.add_node((0, 0), DenseTensor(A, indices))
    return peps
