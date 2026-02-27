"""Higher-Order Tensor Renormalization Group (HOTRG) algorithm.

HOTRG improves upon TRG by using Higher-Order Singular Value Decomposition
(HOSVD) to compute truncation isometries. Instead of pairwise SVD splits,
HOTRG constructs an optimal projector by computing the truncated SVD of the
"environment tensor" M obtained by contracting two tensors over shared bonds.

Reference: Xie et al., PRB 86, 045139 (2012).

Algorithm (horizontal coarse-graining step):
  1. Form M[u, u', d, d'] = sum_{l,r} T[u, d, l, r] * T[u', d', r, l]
     (contract two adjacent tensors over their shared left-right bonds)
  2. HOSVD of M: matricize M along u-axis -> M_mat[u, u'd'd]
     SVD M_mat -> take top chi_target left singular vectors -> isometry U_u
     Similarly compute U_d for the d-axis.
  3. Compress: T_new[u'', d'', l, r] = sum_{u,d} U_u[u, u''] * T[u, d, l, r] * U_d[d, d'']
     (applied to both tensors, then contracted)

The vertical step is analogous with l/r bonds.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from tnjax.core import LOG_EPS
from tnjax.core.tensor import DenseTensor, Tensor


@dataclass
class HOTRGConfig:
    """Configuration for HOTRG coarse-graining.

    Attributes:
        max_bond_dim:    Maximum bond dimension chi after each coarse-graining step.
        num_steps:       Number of coarse-graining iterations.
        direction_order: Order of coarse-graining directions.
                         "alternating": alternate horizontal/vertical (default).
                         "horizontal_first": all horizontal then all vertical.
        svd_trunc_err:   Optional maximum truncation error per HOSVD.
    """

    max_bond_dim: int = 16
    num_steps: int = 10
    direction_order: str = "alternating"
    svd_trunc_err: float | None = None


def hotrg(
    tensor: Tensor,
    config: HOTRGConfig,
) -> jax.Array:
    """HOTRG coarse-graining for a 2D square lattice partition function.

    Uses Higher-Order SVD (HOSVD) for computing truncation isometries,
    providing better accuracy than TRG at the same bond dimension.

    Args:
        tensor: Initial site tensor with 4 legs (up, down, left, right).
                Can be a DenseTensor or raw tensor-like with shape (d,d,d,d).
        config: HOTRGConfig parameters.

    Returns:
        Scalar JAX array: estimated log(Z)/N (free energy per site).
    """
    if isinstance(tensor, DenseTensor):
        T = tensor.todense()
    else:
        T = tensor.todense()

    log_norm_total = jnp.zeros((), dtype=T.dtype)

    for step in range(config.num_steps):
        if config.direction_order == "alternating":
            # Alternate horizontal and vertical
            if step % 2 == 0:
                T, log_norm = _hotrg_step_horizontal(T, config.max_bond_dim)
            else:
                T, log_norm = _hotrg_step_vertical(T, config.max_bond_dim)
        else:
            # horizontal_first: do all horizontal first
            T, log_norm_h = _hotrg_step_horizontal(T, config.max_bond_dim)
            T, log_norm_v = _hotrg_step_vertical(T, config.max_bond_dim)
            log_norm = log_norm_h + log_norm_v

        log_norm_total = log_norm_total + log_norm / (4.0 ** (step + 1))

    return log_norm_total


def _hotrg_step_horizontal(
    T: jax.Array,
    max_bond_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Single horizontal HOTRG coarse-graining step.

    Contracts two adjacent tensors horizontally and uses HOSVD to find
    the optimal truncation isometries for the up and down bonds.

    Args:
        T:            Site tensor of shape (d_u, d_d, d_l, d_r).
        max_bond_dim: Maximum chi after truncation.

    Returns:
        (T_new, log_norm) where T_new has compressed up/down bonds.
    """
    d_u, d_d, d_l, d_r = T.shape

    # Step 1: Form environment tensor M by contracting two T tensors over horizontal bonds
    # M[u, U, d, D] = sum_{l,r} T[u,d,l,r] * T[U,D,r,l]
    # (contract left-right bonds between adjacent horizontal sites)
    M = jnp.einsum("udlr,UDrl->uUdD", T, T)
    # Shape M: (d_u, d_u, d_d, d_d)

    # Step 2: HOSVD — compute isometry for u and d dimensions
    U_u = _compute_hosvd_isometry(M, axis=0, chi_target=min(max_bond_dim, d_u))  # (d_u, chi_u)
    U_d = _compute_hosvd_isometry(M, axis=2, chi_target=min(max_bond_dim, d_d))  # (d_d, chi_d)
    chi_u = U_u.shape[1]
    chi_d = U_d.shape[1]

    # Step 3: Compress T using isometries
    # T_comp[a,b,l,r] = U_u[u,a] * T[u,d,l,r] * U_d[d,b]
    T_comp = jnp.einsum("ua,udlr,db->ablr", U_u, T, U_d)

    # Step 4: Contract two compressed tensors horizontally to form new coarse tensor
    # T_new[a,b,l,r] where the shared bond r of left and l of right is contracted
    # Left T_comp: T_comp[a, b, l, k]
    # Right T_comp: T_comp[A, B, k, r]
    # Apply U_u and U_d again to compress the doubled bonds (a,A) -> new_u, (b,B) -> new_d
    # First contract the two tensors
    T_merged = jnp.einsum("ablk,ABkr->aAbBlr", T_comp, T_comp)
    # Shape: (chi_u, chi_u, chi_d, chi_d, d_l, d_r)
    # Apply isometry to merged up and down indices using the same U (HOSVD environment)
    # Reshape (a,A) -> up_pair and compress with U_u^T U_u (effectively another isometry step)
    # For simplicity: reshape merged axes and truncate
    T_new = T_merged.reshape(chi_u * chi_u, chi_d * chi_d, d_l, d_r)

    # Apply second-level HOSVD to compress back to max_bond_dim
    if T_new.shape[0] > max_bond_dim:
        M2 = jnp.einsum("udlr,UDrl->uUdD", T_new, T_new)
        U_u2 = _compute_hosvd_isometry(M2, axis=0, chi_target=max_bond_dim)
        U_d2 = _compute_hosvd_isometry(M2, axis=2, chi_target=max_bond_dim)
        T_new = jnp.einsum("ua,udlr,db->ablr", U_u2, T_new, U_d2)

    # Normalize
    norm = jnp.max(jnp.abs(T_new))
    log_norm = jnp.log(norm + LOG_EPS)
    T_new = T_new / (norm + LOG_EPS)

    return T_new, log_norm


def _hotrg_step_vertical(
    T: jax.Array,
    max_bond_dim: int,
) -> tuple[jax.Array, jax.Array]:
    """Single vertical HOTRG coarse-graining step.

    Analogous to horizontal step but contracts along the up-down direction.

    Args:
        T:            Site tensor of shape (d_u, d_d, d_l, d_r).
        max_bond_dim: Maximum chi after truncation.

    Returns:
        (T_new, log_norm) where T_new has compressed left/right bonds.
    """
    d_u, d_d, d_l, d_r = T.shape

    # Form environment: M[l,L,r,R] = sum_{u,d} T[u,d,l,r] * T[d,u,L,R]
    # Vertical: contract two tensors sharing up/down bonds
    # T1[u,d,l,r] * T2[d,u,L,R] -> M[l,L,r,R] (sum over u,d between rows)
    M = jnp.einsum("udlr,duLR->lLrR", T, T)
    # Shape: (d_l, d_l, d_r, d_r)

    U_l = _compute_hosvd_isometry(M, axis=0, chi_target=min(max_bond_dim, d_l))  # (d_l, chi_l)
    U_r = _compute_hosvd_isometry(M, axis=2, chi_target=min(max_bond_dim, d_r))  # (d_r, chi_r)
    chi_l = U_l.shape[1]
    chi_r = U_r.shape[1]

    # Compress
    T_comp = jnp.einsum("la,udlr,rb->udab", U_l, T, U_r)

    # Contract two compressed tensors vertically
    # Top T_comp: T_comp[u, k, a, b]  (k = internal bond going down)
    # Bot T_comp: T_comp[k, d, A, B]
    # Apply isometry to compress (a,A) -> new_l and (b,B) -> new_r
    T_merged = jnp.einsum("ukab,kdAB->udaAbB", T_comp, T_comp)
    # Shape: (d_u, d_d, chi_l, chi_l, chi_r, chi_r)
    T_new = T_merged.reshape(d_u, d_d, chi_l * chi_l, chi_r * chi_r)

    # Apply second-level HOSVD to compress back to max_bond_dim
    if T_new.shape[2] > max_bond_dim:
        M2 = jnp.einsum("udlr,duLR->lLrR", T_new, T_new)
        U_l2 = _compute_hosvd_isometry(M2, axis=0, chi_target=max_bond_dim)
        U_r2 = _compute_hosvd_isometry(M2, axis=2, chi_target=max_bond_dim)
        T_new = jnp.einsum("la,udlr,rb->udab", U_l2, T_new, U_r2)

    norm = jnp.max(jnp.abs(T_new))
    log_norm = jnp.log(norm + LOG_EPS)
    T_new = T_new / (norm + LOG_EPS)

    return T_new, log_norm


def _compute_hosvd_isometry(
    M: jax.Array,
    axis: int,
    chi_target: int,
) -> jax.Array:
    """Compute the HOSVD truncation isometry for one axis of tensor M.

    The isometry is computed by:
    1. Matricizing M along `axis`: reshape to (dim_axis, rest).
    2. SVD of the matricized tensor.
    3. Take the top chi_target left singular vectors as the isometry.

    Args:
        M:            4-dimensional environment tensor.
        axis:         Which axis to compute the isometry for.
        chi_target:   Target bond dimension after truncation.

    Returns:
        Isometry matrix U of shape (dim_axis, chi_target).
        Satisfies U^T U ≈ I (isometric).
    """
    shape = M.shape
    dim = shape[axis]
    rest = int(np.prod([shape[i] for i in range(len(shape)) if i != axis]))

    # Move target axis to front, then reshape to matrix
    axes = [axis] + [i for i in range(len(shape)) if i != axis]
    M_perm = jnp.transpose(M, axes)
    M_mat = M_perm.reshape(dim, rest)

    # SVD and truncate
    U, _, _ = jnp.linalg.svd(M_mat, full_matrices=False)
    chi_actual = min(chi_target, U.shape[1])
    return U[:, :chi_actual]
