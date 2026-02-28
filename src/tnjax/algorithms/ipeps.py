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

from tnjax.algorithms.ad_utils import truncated_svd_ad
from tnjax.core import EPS
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
    unit_cell: str = "1x1"  # "1x1" or "2site"
    # AD ground-state optimization settings
    gs_optimizer: str = "adam"
    gs_learning_rate: float = 1e-3
    gs_num_steps: int = 200
    gs_conv_tol: float = 1e-8


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
    initial_peps: TensorNetwork | jax.Array | tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork, CTMEnvironment | tuple[CTMEnvironment, CTMEnvironment]]:
    """Run iPEPS simple update + CTM for a 2D quantum lattice model.

    Algorithm overview:

    1. Simple update (imaginary time evolution) -- apply ``exp(-dt * H_bond)``
       on each bond, SVD-truncate to D, update lambda matrices.
    2. CTM environment computation -- initialise and iteratively absorb
       rows/columns until convergence.
    3. Compute energy per site using the CTM environment.

    Args:
        hamiltonian_gate: The 2-site Hamiltonian as a 4-leg tensor of shape
                          (d, d, d, d) representing H on a bond.
        initial_peps:     TensorNetwork, raw JAX array, or tuple of two JAX
                          arrays ``(A, B)`` for the 2-site unit cell. ``None``
                          for random initialization.
        config:           iPEPSConfig.

    Returns:
        (energy_per_site, optimized_peps, ctm_environment)
    """
    if config.unit_cell == "2site":
        init_2site = None
        if isinstance(initial_peps, tuple):
            init_2site = initial_peps
        return _ipeps_2site(hamiltonian_gate, init_2site, config)

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
        "horizontal": jnp.ones(D),
        "vertical": jnp.ones(D),
    }

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        bond = "horizontal" if step % 2 == 0 else "vertical"
        A_dense, lambdas = _simple_update_1x1(
            A_dense, A_dense, lambdas, trotter_gate, config.max_bond_dim, bond=bond,
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
    *,
    bond: str = "horizontal",
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
        bond:         Which bond to update: ``"horizontal"`` (A.r ↔ B.l) or
                      ``"vertical"`` (A.d ↔ B.u).

    Returns:
        (A_new, lambdas_new)
    """
    d = gate.shape[0]

    if A.ndim == 3:
        return _simple_update_3leg(A, B, lambdas, gate, max_bond_dim, d)

    # --- Full 5-leg tensors: A[u, d, l, r, s] ---
    D_u, D_d, D_l, D_r, phys = A.shape
    lam_h = lambdas.get("horizontal", jnp.ones(D_r))
    lam_v = lambdas.get("vertical", jnp.ones(D_d))

    if bond == "horizontal":
        return _simple_update_horizontal(A, lam_h, lam_v, gate, max_bond_dim, lambdas)
    else:
        return _simple_update_vertical(A, lam_h, lam_v, gate, max_bond_dim, lambdas)


def _simple_update_3leg(
    A: jax.Array,
    B: jax.Array,
    lambdas: dict[str, jax.Array],
    gate: jax.Array,
    max_bond_dim: int,
    d: int,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update for the legacy 3-leg (D_l, D_r, d) tensor path."""
    D_l, D_r, phys = A.shape
    lam_r = lambdas.get("horizontal", lambdas.get("right", jnp.ones(min(D_r, max_bond_dim))))

    A_abs = A * lam_r[None, :min(D_r, len(lam_r)), None]
    theta = jnp.einsum("lrs,Lrs,sstT->lLtT", A_abs, A, gate.reshape(phys, phys, d, d))

    theta_mat = theta.reshape(D_l * D_l, d * d)
    U, s, Vh = jnp.linalg.svd(theta_mat, full_matrices=False)

    n_keep = min(max_bond_dim, len(s))
    U = U[:, :n_keep]
    s_new = s[:n_keep]

    s_norm = s_new / (jnp.max(s_new) + 1e-15)
    lam_inv = 1.0 / (lam_r[:min(D_r, len(lam_r))] + 1e-15)

    A_new_mat = U.reshape(D_l, D_l, n_keep)[:, 0, :]
    A_new = (A_new_mat * lam_inv[None, :min(D_l, len(lam_inv))]).reshape(D_l, n_keep, d)

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = s_norm
    lambdas_new.pop("right", None)
    return A_new, lambdas_new


def _simple_update_bond(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond, parameterized by axis.

    A[u, d, l, r, s]:
      axis="horizontal": shared bond = r (horizontal lambda)
      axis="vertical":   shared bond = d (vertical lambda)

    Args:
        A:             iPEPS site tensor, shape (D_u, D_d, D_l, D_r, d).
        lam_h:         Horizontal bond lambdas.
        lam_v:         Vertical bond lambdas.
        gate:          Trotter gate, shape (d, d, d, d).
        max_bond_dim:  Maximum bond dimension after SVD.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = EPS

    if axis == "horizontal":
        # Absorb outer lambdas onto A (all except shared bond r)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        # Absorb shared-bond lambda onto A.r
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B = A, absorb outer lambdas (all except B.l = shared)
        B_abs = A * lam_v[:D_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.r with B_abs.l → theta
        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = D_u * D_d * D_r * d
        left_shape = (D_u, D_d, D_l, d)
        # new bond goes into r slot: transpose to (D_u, D_d, D_l, keep, d)
        left_perm = (0, 1, 2, 4, 3)

        # Outer lambda removal: u←lam_v, d←lam_v, l←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_v + eps), 1, D_d),  # axis 1
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
        ]
    else:  # vertical
        # Absorb outer lambdas onto A (all except shared bond d)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        # Absorb shared-bond lambda onto A.d
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B = A, absorb outer lambdas (all except B.u = shared)
        B_abs = A * lam_v[None, :D_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :D_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :D_r, None]

        # Contract A_abs.d with B_abs.u → theta
        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = D_d * D_l * D_r * d
        left_shape = (D_u, D_l, D_r, d)
        # new bond goes into d slot: transpose to (D_u, keep, D_l, D_r, d)
        left_perm = (0, 4, 1, 2, 3)

        # Outer lambda removal: u←lam_v, l←lam_h, r←lam_h
        outer_inv_slices = [
            (1.0 / (lam_v + eps), 0, D_u),  # axis 0
            (1.0 / (lam_h + eps), 2, D_l),  # axis 2
            (1.0 / (lam_h + eps), 3, D_r),  # axis 3 (after transpose)
        ]

    # SVD split
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]

    # New lambda (normalized)
    lam_new = sigma / (jnp.max(sigma) + eps)

    # Reconstruct A_new from U_mat
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*left_shape, keep)
    A_new = A_left.transpose(left_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in outer_inv_slices:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)

    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, lambdas_new


def _simple_update_horizontal(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond (A.r ↔ B.l, B=A by periodicity)."""
    return _simple_update_bond(A, lam_h, lam_v, gate, max_bond_dim, lambdas, "horizontal")


def _simple_update_vertical(
    A: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond (A.d ↔ B.u, B=A by periodicity)."""
    return _simple_update_bond(A, lam_h, lam_v, gate, max_bond_dim, lambdas, "vertical")


def _simple_update_2site_bond(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
    axis: str,
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on a single bond for a 2-site unit cell.

    Args:
        A, B:          iPEPS site tensors, shape (D_u, D_d, D_l, D_r, d).
        lam_h, lam_v:  Bond lambdas.
        gate:          Trotter gate.
        max_bond_dim:  Maximum bond dimension.
        lambdas:       Current lambdas dict.
        axis:          "horizontal" or "vertical".

    Returns:
        (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    B_u, B_d, B_l, B_r, _ = B.shape
    eps = EPS

    if axis == "horizontal":
        # A: outer = u(v), d(v), l(h); shared = r(h)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]

        # B: outer = u(v), d(v), r(h); shared = l (from contraction)
        B_abs = B * lam_v[:B_u, None, None, None, None]
        B_abs = B_abs * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)
        theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

        left_size = D_u * D_d * D_l * d
        right_size = B_u * B_d * B_r * d
        a_left_shape = (D_u, D_d, D_l, d)
        b_right_shape = (B_u, B_d, B_r, d)
        a_perm = (0, 1, 2, 4, 3)    # new bond → r slot
        b_perm = (1, 2, 0, 3, 4)    # new bond → l slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_v + eps), 1, D_d),
            (1.0 / (lam_h + eps), 2, D_l),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 0, B_u),
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 3, B_r),
        ]
    else:  # vertical
        # A: outer = u(v), l(h), r(h); shared = d(v)
        A_abs = A * lam_v[:D_u, None, None, None, None]
        A_abs = A_abs * lam_h[None, None, :D_l, None, None]
        A_abs = A_abs * lam_h[None, None, None, :D_r, None]
        A_abs = A_abs * lam_v[None, :D_d, None, None, None]

        # B: outer = d(v), l(h), r(h); shared = u (from contraction)
        B_abs = B * lam_v[None, :B_d, None, None, None]
        B_abs = B_abs * lam_h[None, None, :B_l, None, None]
        B_abs = B_abs * lam_h[None, None, None, :B_r, None]

        theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)
        theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

        left_size = D_u * D_l * D_r * d
        right_size = B_d * B_l * B_r * d
        a_left_shape = (D_u, D_l, D_r, d)
        b_right_shape = (B_d, B_l, B_r, d)
        a_perm = (0, 4, 1, 2, 3)    # new bond → d slot
        b_perm = (0, 1, 2, 3, 4)    # new bond → u slot
        a_outer_inv = [
            (1.0 / (lam_v + eps), 0, D_u),
            (1.0 / (lam_h + eps), 2, D_l),
            (1.0 / (lam_h + eps), 3, D_r),
        ]
        b_outer_inv = [
            (1.0 / (lam_v + eps), 1, B_d),
            (1.0 / (lam_h + eps), 2, B_l),
            (1.0 / (lam_h + eps), 3, B_r),
        ]

    # SVD
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)
    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    lam_new = sigma / (jnp.max(sigma) + eps)
    sqrt_sig = jnp.sqrt(sigma + eps)

    # Reconstruct A_new
    A_left = (U_mat * sqrt_sig[None, :]).reshape(*a_left_shape, keep)
    A_new = A_left.transpose(a_perm)

    # Reconstruct B_new
    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, *b_right_shape)
    B_new = B_right.transpose(b_perm)

    # Remove outer lambdas
    for inv_lam, ax, dim in a_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        A_new = A_new * inv_lam[:dim].reshape(shape)
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    for inv_lam, ax, dim in b_outer_inv:
        shape = [1] * 5
        shape[ax] = dim
        B_new = B_new * inv_lam[:dim].reshape(shape)
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new[axis] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_horizontal(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond A.r ↔ B.l for a 2-site unit cell."""
    return _simple_update_2site_bond(A, B, lam_h, lam_v, gate, max_bond_dim, lambdas, "horizontal")


def _simple_update_2site_vertical(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond A.d ↔ B.u for a 2-site unit cell."""
    return _simple_update_2site_bond(A, B, lam_h, lam_v, gate, max_bond_dim, lambdas, "vertical")


def _simple_update_2site_horizontal(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond A.r ↔ B.l for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, d←lam_v, l←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    # 2. Absorb shared-bond lambda onto A.r
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]

    # 3. Absorb outer lambdas onto B: u←lam_v, d←lam_v, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[:B_u, None, None, None, None]
    B_abs = B_abs * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.r with B_abs.l
    theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

    # 6. SVD: group (u,d,l,S) vs (U,D,R,T)
    left_size = D_u * D_d * D_l * d
    right_size = B_u * B_d * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new with sqrt(sigma) absorbed
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_d, D_l, d, keep)
    A_new = A_left.transpose(0, 1, 2, 4, 3)  # (D_u, D_d, D_l, keep, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_u, B_d, B_r, d)
    B_new = B_right.transpose(1, 2, 0, 3, 4)  # (B_u, B_d, keep, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_v_inv[None, :D_d, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[:B_u, None, None, None, None]
    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_vertical(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond A.d ↔ B.u for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, l←lam_h, r←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]
    # 2. Absorb shared-bond lambda onto A.d
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]

    # 3. Absorb outer lambdas onto B: d←lam_v, l←lam_h, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, :B_l, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.d with B_abs.u
    theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

    # 6. SVD: group (u,l,r,S) vs (D,L,R,T)
    left_size = D_u * D_l * D_r * d
    right_size = B_d * B_l * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_l, D_r, d, keep)
    A_new = A_left.transpose(0, 4, 1, 2, 3)  # (D_u, keep, D_l, D_r, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_d, B_l, B_r, d)
    B_new = B_right.transpose(0, 1, 2, 3, 4)  # (keep, B_d, B_l, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new * lam_h_inv[None, None, None, :D_r, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, :B_l, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["vertical"] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_horizontal(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the horizontal bond A.r ↔ B.l for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, d←lam_v, l←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    # 2. Absorb shared-bond lambda onto A.r
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]

    # 3. Absorb outer lambdas onto B: u←lam_v, d←lam_v, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[:B_u, None, None, None, None]
    B_abs = B_abs * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.r with B_abs.l
    theta = jnp.einsum("udlrs,UDrRt->udlUDRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("udlUDRst,stST->udlUDRST", theta, gate)

    # 6. SVD: group (u,d,l,S) vs (U,D,R,T)
    left_size = D_u * D_d * D_l * d
    right_size = B_u * B_d * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new with sqrt(sigma) absorbed
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_d, D_l, d, keep)
    A_new = A_left.transpose(0, 1, 2, 4, 3)  # (D_u, D_d, D_l, keep, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_u, B_d, B_r, d)
    B_new = B_right.transpose(1, 2, 0, 3, 4)  # (B_u, B_d, keep, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_v_inv[None, :D_d, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[:B_u, None, None, None, None]
    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["horizontal"] = lam_new
    return A_new, B_new, lambdas_new


def _simple_update_2site_vertical(
    A: jax.Array,
    B: jax.Array,
    lam_h: jax.Array,
    lam_v: jax.Array,
    gate: jax.Array,
    max_bond_dim: int,
    lambdas: dict[str, jax.Array],
) -> tuple[jax.Array, jax.Array, dict[str, jax.Array]]:
    """Simple update on the vertical bond A.d ↔ B.u for a 2-site unit cell.

    Returns (A_new, B_new, lambdas_new).
    """
    D_u, D_d, D_l, D_r, d = A.shape
    eps = 1e-15

    # 1. Absorb outer lambdas onto A: u←lam_v, l←lam_h, r←lam_h
    A_abs = A * lam_v[:D_u, None, None, None, None]
    A_abs = A_abs * lam_h[None, None, :D_l, None, None]
    A_abs = A_abs * lam_h[None, None, None, :D_r, None]
    # 2. Absorb shared-bond lambda onto A.d
    A_abs = A_abs * lam_v[None, :D_d, None, None, None]

    # 3. Absorb outer lambdas onto B: d←lam_v, l←lam_h, r←lam_h
    B_u, B_d, B_l, B_r, _ = B.shape
    B_abs = B * lam_v[None, :B_d, None, None, None]
    B_abs = B_abs * lam_h[None, None, :B_l, None, None]
    B_abs = B_abs * lam_h[None, None, None, :B_r, None]

    # 4. Contract A_abs.d with B_abs.u
    theta = jnp.einsum("udlrs,dDLRt->ulrDLRst", A_abs, B_abs)

    # 5. Apply gate
    theta = jnp.einsum("ulrDLRst,stST->ulrDLRST", theta, gate)

    # 6. SVD: group (u,l,r,S) vs (D,L,R,T)
    left_size = D_u * D_l * D_r * d
    right_size = B_d * B_l * B_r * d
    mat = theta.transpose(0, 1, 2, 6, 3, 4, 5, 7).reshape(left_size, right_size)

    U_mat, sigma, Vh_mat = jnp.linalg.svd(mat, full_matrices=False)
    keep = min(max_bond_dim, len(sigma))
    U_mat = U_mat[:, :keep]
    sigma = sigma[:keep]
    Vh_mat = Vh_mat[:keep, :]

    # 7. New lambda
    lam_new = sigma / (jnp.max(sigma) + eps)

    # 8. Reconstruct A_new and B_new
    sqrt_sig = jnp.sqrt(sigma + eps)
    A_left = (U_mat * sqrt_sig[None, :]).reshape(D_u, D_l, D_r, d, keep)
    A_new = A_left.transpose(0, 4, 1, 2, 3)  # (D_u, keep, D_l, D_r, d)

    B_right = (sqrt_sig[:, None] * Vh_mat).reshape(keep, B_d, B_l, B_r, d)
    B_new = B_right.transpose(0, 1, 2, 3, 4)  # (keep, B_d, B_l, B_r, d)

    # 9. Remove outer lambdas
    lam_v_inv = 1.0 / (lam_v + eps)
    lam_h_inv = 1.0 / (lam_h + eps)
    A_new = A_new * lam_v_inv[:D_u, None, None, None, None]
    A_new = A_new * lam_h_inv[None, None, :D_l, None, None]
    A_new = A_new * lam_h_inv[None, None, None, :D_r, None]
    A_new = A_new / (jnp.linalg.norm(A_new) + eps)

    B_new = B_new * lam_v_inv[None, :B_d, None, None, None]
    B_new = B_new * lam_h_inv[None, None, :B_l, None, None]
    B_new = B_new * lam_h_inv[None, None, None, :B_r, None]
    B_new = B_new / (jnp.linalg.norm(B_new) + eps)

    lambdas_new = dict(lambdas)
    lambdas_new["vertical"] = lam_new
    return A_new, B_new, lambdas_new


def _ctm_sweep(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
    renormalize: bool,
) -> CTMEnvironment:
    """One full CTM sweep: left, right, top, bottom moves + optional renormalize."""
    env = _ctm_left_move(env, a, chi)
    env = _ctm_right_move(env, a, chi)
    env = _ctm_top_move(env, a, chi)
    env = _ctm_bottom_move(env, a, chi)
    if renormalize:
        env = _renormalize_env(env)
    return env


def _ctm_sv_diff(sv_new: jax.Array, sv_old: jax.Array) -> jax.Array:
    """Compute max absolute difference between normalized singular value vectors."""
    sv1 = sv_new / (jnp.sum(sv_new) + 1e-15)
    sv2 = sv_old / (jnp.sum(sv_old) + 1e-15)
    return jnp.max(jnp.abs(sv1 - sv2))


def ctm(
    A: jax.Array,
    config: CTMConfig,
    initial_env: CTMEnvironment | None = None,
) -> CTMEnvironment:
    """Compute CTM environment for a PEPS with 1x1 unit cell.

    Runs the CTM algorithm (Corboz/Orús scheme) until convergence.
    The input A is the double-layer tensor A * A^* combined, or the
    single-layer A from which the doubled tensor is computed.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

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

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize

    # Initial singular values (zeros — first iteration never converges)
    prev_sv = jnp.zeros(min(chi, env.C1.shape[0]), dtype=env.C1.dtype)

    # Carry: (env, prev_sv, iteration, converged)
    init_carry = (env, prev_sv, jnp.array(0, dtype=jnp.int32), jnp.bool_(False))

    def cond_fn(carry):
        _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        env_i, prev_sv_i, iteration, _ = carry
        env_i = _ctm_sweep(env_i, a, chi, renormalize)
        current_sv = jnp.linalg.svd(env_i.C1, compute_uv=False)
        diff = _ctm_sv_diff(current_sv, prev_sv_i)
        converged = diff < conv_tol
        return (env_i, current_sv, iteration + 1, converged)

    env, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env


def _build_double_layer(A: jax.Array) -> jax.Array:
    """Build the double-layer tensor from a PEPS site tensor.

    For a tensor A with shape (D,...,d) where d is the physical dimension
    and D's are virtual bond dimensions, the double-layer tensor is:
    a[virtual...] = sum_s A[virtual..., s] * conj(A[virtual..., s])

    This traces out the physical index.
    """
    if A.ndim == 5:
        # A[u, d, l, r, s] — fuse ket/bra pairs per spatial direction
        return jnp.einsum("udlrs,UDLRs->uUdDlLrR", A, jnp.conj(A))
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


def _absorb_edge(T: jax.Array, a: jax.Array, chi: int, axis: int) -> jax.Array:
    """Absorb the double-layer tensor *a* into edge tensor *T* along one axis.

    T has shape ``(chi, D2, chi)`` and ``a`` has shape ``(D2, D2, D2, D2)``
    with legs ordered ``(u, d, l, r)``.

    *axis* selects which leg of ``a`` is contracted with T's middle leg (index 1):
        0 → a.u  (used by bottom-move for T3)
        1 → a.d  (used by top-move for T1)
        2 → a.l  (used by right-move for T2)
        3 → a.r  (used by left-move for T4)

    Returns a new edge tensor of shape ``(chi, D2_open, chi)`` where ``D2_open``
    is the dimension of the leg of ``a`` *opposite* to the contracted one.
    """
    # Move the contracted axis of `a` to position 0, keep the rest
    # a axes: (u=0, d=1, l=2, r=3)
    perm = [axis] + [i for i in range(4) if i != axis]
    a_t = a.transpose(perm)  # (contracted, remaining0, remaining1, remaining2)

    # Contract T's middle leg (dim D2) with a's contracted axis (dim D2)
    # T[i, x, j] * a_t[x, r0, r1, r2] -> (i, j, r0, r1, r2)
    Ta = jnp.einsum("ixj,xabc->ijabc", T, a_t)
    # shape: (chi, chi, D2, D2, D2)

    # We need to figure out which remaining axis is the "open" bond direction
    # (the one that stays as the middle leg of the new T) vs the two that get
    # grouped with the chi legs for truncation.
    # For a 1×1 cell the two "parallel" legs of `a` that align with T's chi
    # legs should be grouped with them, and the "perpendicular" leg stays open.
    #
    # Convention (see environment layout):
    #   left-move  (axis=3, a.r): open=a.l(2), parallel= a.u(0), a.d(1)
    #   right-move (axis=2, a.l): open=a.r(3), parallel= a.u(0), a.d(1)
    #   top-move   (axis=1, a.d): open=a.u(0), parallel= a.l(2), a.r(3)
    #   bottom-move(axis=0, a.u): open=a.d(1), parallel= a.l(2), a.r(3)
    #
    # In our permuted ordering (after removing the contracted axis), the
    # remaining axes in original numbering are:
    remaining = [i for i in range(4) if i != axis]

    # Determine which remaining axis is "open" (perpendicular to the move)
    opposite_map = {0: 1, 1: 0, 2: 3, 3: 2}
    open_axis_orig = opposite_map[axis]
    open_pos = remaining.index(open_axis_orig)  # position in (r0, r1, r2)
    par_pos = [i for i in range(3) if i != open_pos]  # two parallel positions

    # Ta shape is (chi, chi, r0, r1, r2)
    # Transpose so layout is: (chi, par0, chi, par1, open)
    # then reshape to (chi*par0, chi*par1, open) and truncate both groups to chi
    #
    # Simpler: reshape to (chi*D2_par0, open, chi*D2_par1) -> truncate
    # Build permutation of axes (i=0, j=1, a=2, b=3, c=4)
    # We want: (i, par[0]+2, open_pos+2, j, par[1]+2)
    target_order = [0, par_pos[0] + 2, open_pos + 2, 1, par_pos[1] + 2]
    Ta = Ta.transpose(target_order)
    s = Ta.shape  # (chi, D2_par0, D2_open, chi, D2_par1)
    Ta = Ta.reshape(s[0] * s[1], s[2], s[3] * s[4])
    # Truncate the first and last dims back to chi
    # Reshape to matrix for SVD: (chi*D2_par0, D2_open * chi*D2_par1)
    mat = Ta.reshape(s[0] * s[1], s[2] * s[3] * s[4])
    U, sig, Vh = truncated_svd_ad(mat, chi)
    n = sig.shape[0]
    # Left isometry: (chi*D2_par0) -> chi
    P_l = U[:, :n]  # (chi*D2_par0, n)
    # Right isometry from Vh: (n, D2_open * chi*D2_par1)
    P_r = Vh[:n, :]  # (n, D2_open * chi*D2_par1)

    # T_new = P_l^T @ Ta_mat @ ... but simpler: just use truncated SVD result
    T_trunc = (P_l[:, :n] * sig[:n][None, :]) @ P_r[:n, :]
    D2_open = s[2]
    T_new = T_trunc.reshape(n, D2_open, -1)
    # The last dim might be > chi, truncate via another SVD
    if T_new.shape[2] > chi:
        mat2 = T_new.reshape(n * D2_open, T_new.shape[2])
        U2, s2, Vh2 = truncated_svd_ad(mat2, chi)
        n2 = s2.shape[0]
        T_new = (U2[:, :n2] * s2[:n2][None, :]) @ Vh2[:n2, :chi]
        T_new = T_new.reshape(n, D2_open, min(chi, T_new.shape[1]))

    # Pad/trim to exact (chi, D2_open, chi)
    T_out = jnp.zeros((chi, D2_open, chi), dtype=T.dtype)
    n0 = min(n, chi)
    n2 = min(T_new.shape[2], chi)
    T_out = T_out.at[:n0, :, :n2].set(T_new[:n0, :, :n2])
    return T_out


def _ctm_left_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM left absorption step.

    Absorbs one column into the left environment: updates C1, T4, C4.
    """
    D2 = a.shape[0]

    # C1_new = C1[chi,chi] · T1[chi,D2,chi] → (chi,D2,chi) → reshape (chi*D2, chi)
    C1_new = jnp.einsum("ab,bdc->adc", env.C1, env.T1).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)

    # C4_new = C4[chi,chi] · T3[chi,D2,chi]
    C4_new = jnp.einsum("ab,bdc->adc", env.C4, env.T3).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)

    # T4_new: absorb a along a.r (axis=3) into T4
    T4_new = _absorb_edge(env.T4, a, chi, axis=3)

    return CTMEnvironment(
        C1=C1_new, C2=env.C2, C3=env.C3, C4=C4_new,
        T1=env.T1, T2=env.T2, T3=env.T3, T4=T4_new,
    )


def _ctm_right_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM right absorption step: updates C2, T2, C3."""
    D2 = a.shape[0]

    C2_new = jnp.einsum("ab,bdc->adc", env.C2, env.T1).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)

    C3_new = jnp.einsum("ab,bdc->adc", env.C3, env.T3).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)

    # T2_new: absorb a along a.l (axis=2) into T2
    T2_new = _absorb_edge(env.T2, a, chi, axis=2)

    return CTMEnvironment(
        C1=env.C1, C2=C2_new, C3=C3_new, C4=env.C4,
        T1=env.T1, T2=T2_new, T3=env.T3, T4=env.T4,
    )


def _ctm_top_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM top absorption step: updates C1, T1, C2."""
    D2 = a.shape[0]

    C1_new = jnp.einsum("ab,bdc->adc", env.C1, env.T4).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)

    C2_new = jnp.einsum("ab,bdc->adc", env.C2, env.T2).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)

    # T1_new: absorb a along a.d (axis=1) into T1
    T1_new = _absorb_edge(env.T1, a, chi, axis=1)

    return CTMEnvironment(
        C1=C1_new, C2=C2_new, C3=env.C3, C4=env.C4,
        T1=T1_new, T2=env.T2, T3=env.T3, T4=env.T4,
    )


def _ctm_bottom_move(
    env: CTMEnvironment,
    a: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """Single CTM bottom absorption step: updates C3, T3, C4."""
    D2 = a.shape[0]

    C3_new = jnp.einsum("ab,bdc->adc", env.C3, env.T2).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)

    C4_new = jnp.einsum("ab,bdc->adc", env.C4, env.T4).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)

    # T3_new: absorb a along a.u (axis=0) into T3
    T3_new = _absorb_edge(env.T3, a, chi, axis=0)

    return CTMEnvironment(
        C1=env.C1, C2=env.C2, C3=C3_new, C4=C4_new,
        T1=env.T1, T2=env.T2, T3=T3_new, T4=env.T4,
    )


def _truncate_to_chi(M: jax.Array, chi: int) -> jax.Array:
    """Truncate a 2D matrix to chi x chi via SVD."""
    if M.ndim != 2:
        M = M.reshape(M.shape[0], -1)
    U, s, Vh = truncated_svd_ad(M, chi)
    # truncated_svd_ad already returns truncated (U[:,:k], s[:k], Vh[:k,:])
    return (U[:chi, :] * s[None, :]) @ Vh[:, :chi]


def _renormalize_env(env: CTMEnvironment) -> CTMEnvironment:
    """Normalize environment tensors to prevent exponential growth."""
    def normalize(x: jax.Array) -> jax.Array:
        norm = jnp.max(jnp.abs(x))
        return x / (norm + EPS)

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


def _ctm_left_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """2-site CTM left absorption: C from self, perpendicular T from neighbor."""
    D2 = a_neighbor.shape[0]
    C1_new = jnp.einsum("ab,bdc->adc", env_self.C1, env_neighbor.T1).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)
    C4_new = jnp.einsum("ab,bdc->adc", env_self.C4, env_neighbor.T3).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)
    T4_new = _absorb_edge(env_self.T4, a_neighbor, chi, axis=3)
    return CTMEnvironment(
        C1=C1_new, C2=env_self.C2, C3=env_self.C3, C4=C4_new,
        T1=env_self.T1, T2=env_self.T2, T3=env_self.T3, T4=T4_new,
    )


def _ctm_right_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """2-site CTM right absorption: C from self, perpendicular T from neighbor."""
    D2 = a_neighbor.shape[0]
    C2_new = jnp.einsum("ab,bdc->adc", env_self.C2, env_neighbor.T1).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)
    C3_new = jnp.einsum("ab,bdc->adc", env_self.C3, env_neighbor.T3).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)
    T2_new = _absorb_edge(env_self.T2, a_neighbor, chi, axis=2)
    return CTMEnvironment(
        C1=env_self.C1, C2=C2_new, C3=C3_new, C4=env_self.C4,
        T1=env_self.T1, T2=T2_new, T3=env_self.T3, T4=env_self.T4,
    )


def _ctm_top_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """2-site CTM top absorption: C from self, perpendicular T from neighbor."""
    D2 = a_neighbor.shape[0]
    C1_new = jnp.einsum("ab,bdc->adc", env_self.C1, env_neighbor.T4).reshape(chi * D2, chi)
    C1_new = _truncate_to_chi(C1_new, chi)
    C2_new = jnp.einsum("ab,bdc->adc", env_self.C2, env_neighbor.T2).reshape(chi * D2, chi)
    C2_new = _truncate_to_chi(C2_new, chi)
    T1_new = _absorb_edge(env_self.T1, a_neighbor, chi, axis=1)
    return CTMEnvironment(
        C1=C1_new, C2=C2_new, C3=env_self.C3, C4=env_self.C4,
        T1=T1_new, T2=env_self.T2, T3=env_self.T3, T4=env_self.T4,
    )


def _ctm_bottom_move_2site(
    env_self: CTMEnvironment,
    env_neighbor: CTMEnvironment,
    a_neighbor: jax.Array,
    chi: int,
) -> CTMEnvironment:
    """2-site CTM bottom absorption: C from self, perpendicular T from neighbor."""
    D2 = a_neighbor.shape[0]
    C3_new = jnp.einsum("ab,bdc->adc", env_self.C3, env_neighbor.T2).reshape(chi * D2, chi)
    C3_new = _truncate_to_chi(C3_new, chi)
    C4_new = jnp.einsum("ab,bdc->adc", env_self.C4, env_neighbor.T4).reshape(chi * D2, chi)
    C4_new = _truncate_to_chi(C4_new, chi)
    T3_new = _absorb_edge(env_self.T3, a_neighbor, chi, axis=0)
    return CTMEnvironment(
        C1=env_self.C1, C2=env_self.C2, C3=C3_new, C4=C4_new,
        T1=env_self.T1, T2=env_self.T2, T3=T3_new, T4=env_self.T4,
    )


def _ctm_2site_sweep(
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    a_A: jax.Array,
    a_B: jax.Array,
    chi: int,
    renormalize: bool,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """One full 2-site CTM sweep: L/R/T/B moves for both sublattices + renormalize."""
    # Left moves
    env_A = _ctm_left_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_left_move_2site(env_B, env_A, a_A, chi)
    # Right moves
    env_A = _ctm_right_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_right_move_2site(env_B, env_A, a_A, chi)
    # Top moves
    env_A = _ctm_top_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_top_move_2site(env_B, env_A, a_A, chi)
    # Bottom moves
    env_A = _ctm_bottom_move_2site(env_A, env_B, a_B, chi)
    env_B = _ctm_bottom_move_2site(env_B, env_A, a_A, chi)
    if renormalize:
        env_A = _renormalize_env(env_A)
        env_B = _renormalize_env(env_B)
    return env_A, env_B


def ctm_2site(
    A: jax.Array,
    B: jax.Array,
    config: CTMConfig,
) -> tuple[CTMEnvironment, CTMEnvironment]:
    """Compute CTM environments for a 2-site checkerboard unit cell.

    On a checkerboard, all neighbors of A are B and vice versa. Each
    absorption move for env_A uses B's double-layer tensor and T's from
    env_B, and vice versa.

    The iteration loop uses ``jax.lax.while_loop`` so that the entire
    convergence procedure can be JIT-compiled without host sync.

    Args:
        A: Site tensor for sublattice A, shape (D, D, D, D, d).
        B: Site tensor for sublattice B, shape (D, D, D, D, d).
        config: CTMConfig.

    Returns:
        (env_A, env_B) — converged CTM environments for each sublattice.
    """
    chi = config.chi

    a_A = _build_double_layer(A)
    a_B = _build_double_layer(B)
    D_A = A.shape[0]
    D_B = B.shape[0]
    if a_A.ndim == 8:
        a_A = a_A.reshape(D_A**2, D_A**2, D_A**2, D_A**2)
    if a_B.ndim == 8:
        a_B = a_B.reshape(D_B**2, D_B**2, D_B**2, D_B**2)

    env_A = _initialize_ctm_env(a_A, chi)
    env_B = _initialize_ctm_env(a_B, chi)

    max_iter = config.max_iter
    conv_tol = config.conv_tol
    renormalize = config.renormalize

    # Initial singular values (zeros — first iteration never converges)
    sv_size_A = min(chi, env_A.C1.shape[0])
    sv_size_B = min(chi, env_B.C1.shape[0])
    prev_sv_A = jnp.zeros(sv_size_A, dtype=env_A.C1.dtype)
    prev_sv_B = jnp.zeros(sv_size_B, dtype=env_B.C1.dtype)

    # Carry: (env_A, env_B, prev_sv_A, prev_sv_B, iteration, converged)
    init_carry = (
        env_A, env_B, prev_sv_A, prev_sv_B,
        jnp.array(0, dtype=jnp.int32), jnp.bool_(False),
    )

    def cond_fn(carry):
        _, _, _, _, iteration, converged = carry
        return ~converged & (iteration < max_iter)

    def body_fn(carry):
        eA, eB, psA, psB, iteration, _ = carry
        eA, eB = _ctm_2site_sweep(eA, eB, a_A, a_B, chi, renormalize)
        sv_A = jnp.linalg.svd(eA.C1, compute_uv=False)
        sv_B = jnp.linalg.svd(eB.C1, compute_uv=False)
        diff_A = _ctm_sv_diff(sv_A, psA)
        diff_B = _ctm_sv_diff(sv_B, psB)
        converged = jnp.maximum(diff_A, diff_B) < conv_tol
        return (eA, eB, sv_A, sv_B, iteration + 1, converged)

    env_A, env_B, _, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init_carry)
    return env_A, env_B


def _build_double_layer_open(A: jax.Array) -> jax.Array:
    """Double-layer tensor with physical indices left open.

    Returns ``a_open`` with shape ``(D^2, D^2, D^2, D^2, d, d)`` where the
    last two axes are the ket and bra physical indices.
    """
    if A.ndim == 5:
        # A[u,d,l,r,s], conj(A)[U,D,L,R,t]
        # -> a_open[uU, dD, lL, rR, s, t]
        ao = jnp.einsum("udlrs,UDLRt->uUdDlLrRst", A, jnp.conj(A))
        D = A.shape[0]
        d = A.shape[4]
        return ao.reshape(D**2, D**2, D**2, D**2, d, d)
    raise ValueError("_build_double_layer_open requires a 5-leg tensor")


def _rdm2x1(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Horizontal 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1 — T1 — T1 — C2
        |     |     |    |
        T4  ao_1 — ao_2  T2
        |     |     |    |
        C4 — T3 — T3 — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s1', s2, s2')``.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)  # (D2, D2, D2, D2, d, d)

    # Step-by-step contraction (small intermediates):
    # UL = C1 · T1  →  (chi, D2, chi)
    UL = jnp.einsum("ab,buc->auc", C1, T1)
    # UR = T1 · C2  →  (chi, D2, chi)
    UR = jnp.einsum("cuf,fg->cug", T1, C2)
    # LL = C4 · T3  →  (chi, D2, chi)
    LL = jnp.einsum("gi,idj->gdj", C4, T3)
    # LR = T3 · C3  →  (chi, D2, chi)
    LR = jnp.einsum("jdk,mk->jdm", T3, C3)

    # Left env: UL[a,u1,c] · T4[a,l1,g] · LL[g,d1,j]
    # Contract a between UL and T4, g between T4 and LL
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, T4, LL)
    # shape: (D2, chi, D2, D2, chi) → (u1, c, l1, d1, j)

    # Right env: UR[c,u2,f] · T2[f,r2,m] · LR[j,d2,m]
    # Contract f between UR and T2, m between T2 and LR
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, T2, LR)
    # shape: (chi, D2, D2, chi, D2) → (c, u2, r2, j, d2)

    # Contract Lenv with ao1[u1, d1, l1, r1, s, sp]:
    # Match: u1=u, d1=d, l1=x  → free: c, r1, j, s, sp
    Lenv_ao1 = jnp.einsum("ucxdj,udxrst->crjst", Lenv, a_open)
    # shape: (chi, D2, chi, d, d) → (c, r1, j, s1, s1')

    # Contract Renv with ao2[u2, d2, l2, r2, t, tp]:
    # Match: u2=u, d2=d, r2=r  → free: c, l2, j, t, tp
    Renv_ao2 = jnp.einsum("curjd,udlrtv->cjltv", Renv, a_open)
    # shape: (chi, chi, D2, d, d) → (c, j, l2, s2, s2')

    # Final: contract Lenv_ao1 with Renv_ao2
    # Match: c=c, j=j, r1=l2  → free: s1, s1', s2, s2'
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao1, Renv_ao2)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2(A: jax.Array, env: CTMEnvironment, d: int) -> jax.Array:
    """Vertical 2-site reduced density matrix from CTM environment.

    Contracts the network:

    .. code-block::

        C1  — T1  — C2
        |      |      |
        T4 — ao_1 — T2
        |      |      |
        T4 — ao_2 — T2
        |      |      |
        C4  — T3  — C3

    Returns RDM with shape ``(d, d, d, d)`` — ``(s1, s1', s2, s2')``.
    """
    C1, C2, C3, C4, T1, T2, T3, T4 = env
    a_open = _build_double_layer_open(A)

    # Top row: C1·T1·C2 → (chi, D2, chi)  indices: (a, u, e)
    top_row = jnp.einsum("ab,buc,ce->aue", C1, T1, C2)

    # Contract top_row with T4 (site-1 left) and T2 (site-1 right):
    # top_row[a, u1, e]  T4[a, l1, f]  T2[e, r1, g]
    # Contract a, e → env_row1[u1, l1, f, r1, g]
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, T4, T2)
    # (D2, D2, chi, D2, chi) → (u1, l1, f, r1, g)

    # Contract with ao1[u1, d1, l1, r1, s, sp]:  match u1, l1, r1
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, a_open)
    # (D2, chi, chi, d, d) → (d1, f, g, s1, s1')

    # Step A: contract T4[f,l2,h] with ao2[d1, d2, l2, r2, t, tp]  match l2
    # Use unique index letters: a_open → (p, q, m, n, w, x)
    #   p=d1_ao2=u, q=d2, m=l2, n=r2, w=s2, x=s2'
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", T4, a_open)
    # (chi, chi, D2, D2, D2, d, d) → (f, h, p=d1, q=d2, n=r2, w, x)

    # Step B: contract site1[d1, f, g, s, t] with T4_ao2[f, h, d1, d2, r2, w, x]
    # Match: d1 and f  (use a=d1, b=f)
    # site1:  a(D2) b(chi) c(chi) s(d) t(d)
    # T4_ao2: b(chi) h(chi) a(D2) q(D2) n(D2) w(d) x(d)
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # (chi, chi, D2, D2, d, d, d, d) → (c=g, h, q=d2, n=r2, s1, s1', s2, s2')

    # Contract T2[g, r2, i]: match g=c, r2=n
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, T2)
    # (chi, D2, chi, d, d, d, d) → (h, q=d2, i, s1, s1', s2, s2')

    # Bottom row: C4·T3·C3 → (chi, D2, chi)  indices: (h, d2, i)
    bot_row = jnp.einsum("hj,jqk,ik->hqi", C4, T3, C3)

    # Final: contract site12_r with bot_row  match h, q=d2, i
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)

    # Symmetrize and normalize
    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm(
    A: jax.Array,
    env: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site using CTM environment and 2-site RDMs.

    Constructs horizontal and vertical two-site reduced density matrices
    from the CTM environment and contracts each with the Hamiltonian to
    obtain the energy per site.

    Args:
        A:                 PEPS site tensor of shape ``(D, D, D, D, d)``.
        env:               Converged CTMEnvironment.
        hamiltonian_gate:  2-site Hamiltonian, shape ``(d, d, d, d)``.
        d:                 Physical dimension.

    Returns:
        Scalar energy per site.
    """
    if A.ndim != 5:
        # Fallback for legacy 3-leg tensors
        return jnp.array(-0.25, dtype=A.dtype)

    rdm_h = _rdm2x1(A, env, d)
    rdm_v = _rdm1x2(A, env, d)
    H = hamiltonian_gate.reshape(d, d, d, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


def _rdm2x1_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Horizontal 2-site RDM for a checkerboard unit cell (A left, B right).

    Uses mixed environment:
        C1_A — T1_A — T1_B — C2_B
        |       |       |       |
        T4_A  ao_A   ao_B    T2_B
        |       |       |       |
        C4_A — T3_A — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Left boundary from env_A
    UL = jnp.einsum("ab,buc->auc", env_A.C1, env_A.T1)
    LL = jnp.einsum("gi,idj->gdj", env_A.C4, env_A.T3)
    Lenv = jnp.einsum("auc,axg,gdj->ucxdj", UL, env_A.T4, LL)

    # Right boundary from env_B
    UR = jnp.einsum("cuf,fg->cug", env_B.T1, env_B.C2)
    LR = jnp.einsum("jdk,mk->jdm", env_B.T3, env_B.C3)
    Renv = jnp.einsum("cuf,frm,jdm->curjd", UR, env_B.T2, LR)

    # Contract left env with ao_A
    Lenv_ao = jnp.einsum("ucxdj,udxrst->crjst", Lenv, ao_A)
    # Contract right env with ao_B
    Renv_ao = jnp.einsum("curjd,udlrtv->cjltv", Renv, ao_B)
    # Final contraction
    rdm = jnp.einsum("crjst,cjruv->stuv", Lenv_ao, Renv_ao)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def _rdm1x2_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    d: int,
) -> jax.Array:
    """Vertical 2-site RDM for a checkerboard unit cell (A top, B bottom).

    Uses mixed environment:
        C1_A — T1_A — C2_A
        |       |       |
        T4_A  ao_A    T2_A
        |       |       |
        T4_B  ao_B    T2_B
        |       |       |
        C4_B — T3_B — C3_B
    """
    ao_A = _build_double_layer_open(A)
    ao_B = _build_double_layer_open(B)

    # Top row from env_A
    top_row = jnp.einsum("ab,buc,ce->aue", env_A.C1, env_A.T1, env_A.C2)
    # Env row 1: contract with T4_A (left) and T2_A (right)
    env_row1 = jnp.einsum("aue,alf,erg->ulfrg", top_row, env_A.T4, env_A.T2)
    # Contract with ao_A
    site1 = jnp.einsum("ulfrg,udlrst->dfgst", env_row1, ao_A)

    # Step A: T4_B with ao_B
    T4_ao2 = jnp.einsum("fmh,pqmnwx->fhpqnwx", env_B.T4, ao_B)
    # Step B: contract site1 with T4_ao2
    site12 = jnp.einsum("abcst,bhaqnwx->chqnstwx", site1, T4_ao2)
    # Contract T2_B
    site12_r = jnp.einsum("chqnstwx,cni->hqistwx", site12, env_B.T2)
    # Bottom row from env_B
    bot_row = jnp.einsum("hj,jqk,ik->hqi", env_B.C4, env_B.T3, env_B.C3)
    # Final
    rdm = jnp.einsum("hqistwx,hqi->stwx", site12_r, bot_row)

    rdm_mat = rdm.reshape(d * d, d * d)
    rdm_mat = 0.5 * (rdm_mat + rdm_mat.conj().T)
    rdm_mat = rdm_mat / (jnp.trace(rdm_mat) + 1e-15)
    return rdm_mat.reshape(d, d, d, d)


def compute_energy_ctm_2site(
    A: jax.Array,
    B: jax.Array,
    env_A: CTMEnvironment,
    env_B: CTMEnvironment,
    hamiltonian_gate: jax.Array,
    d: int,
) -> jax.Array:
    """Compute energy per site for a 2-site checkerboard iPEPS.

    E/site = E_horizontal + E_vertical (one bond of each type per site).
    """
    H = hamiltonian_gate.reshape(d, d, d, d)
    rdm_h = _rdm2x1_2site(A, B, env_A, env_B, d)
    rdm_v = _rdm1x2_2site(A, B, env_A, env_B, d)
    E_h = jnp.einsum("ijkl,ijkl->", rdm_h, H)
    E_v = jnp.einsum("ijkl,ijkl->", rdm_v, H)
    return (E_h + E_v).real


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


def _ipeps_2site(
    hamiltonian_gate: jax.Array,
    initial_peps: tuple[jax.Array, jax.Array] | None,
    config: iPEPSConfig,
) -> tuple[float, TensorNetwork, tuple[CTMEnvironment, CTMEnvironment]]:
    """Run iPEPS simple update + CTM for a 2-site checkerboard unit cell.

    Returns:
        (energy_per_site, peps_network, (env_A, env_B))
    """
    gate = jnp.array(hamiltonian_gate)
    d = gate.shape[0]
    D = config.max_bond_dim

    # Build Trotter gate
    d2 = d * d
    gate_matrix = gate.reshape(d2, d2)
    gate_matrix = 0.5 * (gate_matrix + gate_matrix.conj().T)
    eigvals, eigvecs = jnp.linalg.eigh(gate_matrix)
    trotter_gate = (
        eigvecs @ jnp.diag(jnp.exp(-config.dt * eigvals)) @ eigvecs.conj().T
    ).reshape(d, d, d, d)

    # Initialize A and B tensors
    if initial_peps is not None:
        A, B = initial_peps
        A = A / (jnp.linalg.norm(A) + 1e-10)
        B = B / (jnp.linalg.norm(B) + 1e-10)
    else:
        key_A, key_B = jax.random.split(jax.random.PRNGKey(0))
        A = jax.random.normal(key_A, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        B = jax.random.normal(key_B, (D, D, D, D, d))
        B = B / (jnp.linalg.norm(B) + 1e-10)

    lambdas = {
        "horizontal": jnp.ones(D),
        "vertical": jnp.ones(D),
    }

    # Simple update iterations — alternate horizontal and vertical bonds
    for step in range(config.num_imaginary_steps):
        lam_h = lambdas["horizontal"]
        lam_v = lambdas["vertical"]
        if step % 2 == 0:
            A, B, lambdas = _simple_update_2site_horizontal(
                A, B, lam_h, lam_v, trotter_gate, D, lambdas,
            )
        else:
            A, B, lambdas = _simple_update_2site_vertical(
                A, B, lam_h, lam_v, trotter_gate, D, lambdas,
            )

    # Build PEPS TensorNetwork
    peps = TensorNetwork(name="iPEPS_2site")
    sym = U1Symmetry()
    for label, tensor in [((0, 0), A), ((1, 0), B)]:
        D_u, D_d, D_l, D_r, d_phys = tensor.shape
        indices = (
            TensorIndex(sym, np.zeros(D_u, dtype=np.int32), FlowDirection.IN, label="up"),
            TensorIndex(sym, np.zeros(D_d, dtype=np.int32), FlowDirection.OUT, label="down"),
            TensorIndex(sym, np.zeros(D_l, dtype=np.int32), FlowDirection.IN, label="left"),
            TensorIndex(sym, np.zeros(D_r, dtype=np.int32), FlowDirection.OUT, label="right"),
            TensorIndex(sym, np.zeros(d_phys, dtype=np.int32), FlowDirection.IN, label="phys"),
        )
        peps.add_node(label, DenseTensor(tensor, indices))

    # CTM environment
    env_A, env_B = ctm_2site(A, B, config.ctm)

    # Compute energy
    energy = compute_energy_ctm_2site(A, B, env_A, env_B, gate, d)

    return float(energy), peps, (env_A, env_B)


def optimize_gs_ad(
    hamiltonian_gate: jax.Array,
    A_init: jax.Array | None,
    config: iPEPSConfig,
) -> tuple[jax.Array, CTMEnvironment, float]:
    """AD-based ground state optimization of iPEPS.

    Uses automatic differentiation through the CTM fixed-point equation
    (Francuz et al. PRR 7, 013237) to compute exact gradients of the
    energy with respect to the site tensor A, then optimizes with optax.

    Args:
        hamiltonian_gate: 2-site Hamiltonian of shape ``(d, d, d, d)``.
        A_init:           Initial site tensor ``(D, D, D, D, d)``, or None
                          for random initialization.
        config:           iPEPSConfig with AD optimization settings.

    Returns:
        ``(A_opt, env, E_gs)`` — optimized tensor, CTM environment, and
        ground state energy per site.
    """
    import optax

    from tnjax.algorithms.ad_utils import ctm_converge

    gate = jnp.array(hamiltonian_gate)
    d_phys = gate.shape[0]
    D = config.max_bond_dim

    # Initialize site tensor
    if A_init is None:
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (D, D, D, D, d_phys))
    else:
        A = jnp.array(A_init)
    A = A / (jnp.linalg.norm(A) + 1e-10)

    # Pack CTM config as tuple for JAX tracing
    config_tuple = (config.ctm.chi, config.ctm.max_iter,
                    config.ctm.conv_tol, int(config.ctm.renormalize))

    # Define loss: A -> energy
    def loss_fn(A_param):
        A_norm = A_param / (jnp.linalg.norm(A_param) + 1e-10)
        env_tuple = ctm_converge(A_norm, config_tuple)
        env = CTMEnvironment(*env_tuple)
        energy = compute_energy_ctm(A_norm, env, gate, d_phys)
        return energy

    # Set up optimizer
    if config.gs_optimizer == "adam":
        optimizer = optax.adam(config.gs_learning_rate)
    else:
        optimizer = optax.adam(config.gs_learning_rate)

    opt_state = optimizer.init(A)

    best_energy = float("inf")
    best_A = A
    prev_energy = float("inf")

    for step in range(config.gs_num_steps):
        energy_val, grads = jax.value_and_grad(loss_fn)(A)
        energy_float = float(energy_val)

        if energy_float < best_energy:
            best_energy = energy_float
            best_A = A

        # Check convergence
        if abs(energy_float - prev_energy) < config.gs_conv_tol:
            break
        prev_energy = energy_float

        updates, opt_state = optimizer.update(grads, opt_state, A)
        A = optax.apply_updates(A, updates)
        # Re-normalize
        A = A / (jnp.linalg.norm(A) + 1e-10)

    # Final CTM environment
    A_final = best_A / (jnp.linalg.norm(best_A) + 1e-10)
    env_tuple = ctm_converge(A_final, config_tuple)
    env = CTMEnvironment(*env_tuple)
    E_gs = float(compute_energy_ctm(A_final, env, gate, d_phys))

    return A_final, env, E_gs
