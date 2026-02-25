"""Infinite Density Matrix Renormalization Group (iDMRG) algorithm.

Finds the ground-state energy per site of a translationally invariant 1D
Hamiltonian in the thermodynamic limit.  The Hamiltonian is specified by a
single bulk MPO tensor (the W-matrix repeated at every site).

The algorithm works on a 2-site unit cell, optimising a two-site wavefunction
at each step and growing the chain by two sites per iteration.  Left- and
right-canonical MPS tensors are obtained from the SVD of the optimised
wavefunction, and the environments are updated incrementally.

Architecture decisions mirror ``dmrg.py``:

- The outer loop is a Python for-loop (bond dimensions change after SVD).
- The effective-Hamiltonian matvec is JIT-compiled via the same helper used
  in finite DMRG.
- Environments are dense JAX arrays wrapped in ``DenseTensor``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tnjax.algorithms.dmrg import (
    _lanczos_solve,
)
from tnjax.core.index import FlowDirection, TensorIndex
from tnjax.core.symmetry import U1Symmetry
from tnjax.core.tensor import DenseTensor, Tensor

# ---------------------------------------------------------------------------
# Config & Result
# ---------------------------------------------------------------------------


@dataclass
class iDMRGConfig:
    """Configuration for an iDMRG run.

    Attributes:
        max_bond_dim:    Maximum allowed bond dimension (chi).
        max_iterations:  Maximum number of 2-site growth steps.
        convergence_tol: Convergence threshold on energy per site.
        lanczos_max_iter: Maximum Lanczos iterations.
        lanczos_tol:     Lanczos convergence tolerance.
        svd_trunc_err:   Maximum SVD truncation error (None = use max_bond_dim).
        verbose:         Print per-step diagnostics.
    """

    max_bond_dim: int = 100
    max_iterations: int = 200
    convergence_tol: float = 1e-8
    lanczos_max_iter: int = 50
    lanczos_tol: float = 1e-12
    svd_trunc_err: float | None = None
    verbose: bool = False


class iDMRGResult(NamedTuple):
    """Result of an iDMRG run.

    Attributes:
        energy_per_site:    Converged energy per site.
        energies_per_step:  Energy-per-site estimate at each iteration.
        mps_tensors:        2-site unit cell ``[A_L, A_R]`` as ``Tensor``.
        singular_values:    Singular values on the centre bond.
        converged:          True if the run converged within tolerance.
    """

    energy_per_site: float
    energies_per_step: list[float]
    mps_tensors: list[Tensor]
    singular_values: jax.Array
    converged: bool


# ---------------------------------------------------------------------------
# Bulk MPO builder
# ---------------------------------------------------------------------------


def build_bulk_mpo_heisenberg(
    Jz: float = 1.0,
    Jxy: float = 1.0,
    hz: float = 0.0,
    d: int = 2,
    dtype: Any = jnp.float32,
) -> DenseTensor:
    """Build a single bulk W-matrix for the spin-1/2 XXZ Heisenberg model.

    The returned tensor is the 5×d×d×5 MPO site tensor that is repeated at
    every site of an infinite chain.

    Args:
        Jz:    Ising coupling strength.
        Jxy:   XY coupling strength.
        hz:    Longitudinal magnetic field.
        d:     Physical dimension (must be 2).
        dtype: JAX dtype for the tensor data.

    Returns:
        ``DenseTensor`` with legs ``("w_l", "mpo_top", "mpo_bot", "w_r")``.
    """
    if d != 2:
        raise ValueError(f"build_bulk_mpo_heisenberg only supports d=2, got {d}")

    Sp = jnp.array([[0, 1], [0, 0]], dtype=dtype)
    Sm = jnp.array([[0, 0], [1, 0]], dtype=dtype)
    Sz = 0.5 * jnp.array([[1, 0], [0, -1]], dtype=dtype)
    I2 = jnp.eye(d, dtype=dtype)

    D_w = 5
    W = jnp.zeros((D_w, d, d, D_w), dtype=dtype)
    W = W.at[0, :, :, 0].set(I2)
    W = W.at[1, :, :, 0].set(Sp)
    W = W.at[2, :, :, 0].set(Sm)
    W = W.at[3, :, :, 0].set(Sz)
    W = W.at[4, :, :, 0].set(hz * Sz)
    W = W.at[4, :, :, 1].set((Jxy / 2) * Sm)
    W = W.at[4, :, :, 2].set((Jxy / 2) * Sp)
    W = W.at[4, :, :, 3].set(Jz * Sz)
    W = W.at[4, :, :, 4].set(I2)

    sym = U1Symmetry()
    bond_dw = np.zeros(D_w, dtype=np.int32)
    bond_d = np.zeros(d, dtype=np.int32)

    indices = (
        TensorIndex(sym, bond_dw, FlowDirection.IN,  label="w_l"),
        TensorIndex(sym, bond_d,  FlowDirection.IN,  label="mpo_top"),
        TensorIndex(sym, bond_d,  FlowDirection.OUT, label="mpo_bot"),
        TensorIndex(sym, bond_dw, FlowDirection.OUT, label="w_r"),
    )
    return DenseTensor(W, indices)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _trivial_left_env(D_w: int, dtype: Any = jnp.float32) -> DenseTensor:
    """Trivial (1, D_w, 1) left environment for iDMRG."""
    sym = U1Symmetry()
    bond_mps = np.zeros(1, dtype=np.int32)
    bond_mpo = np.zeros(D_w, dtype=np.int32)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    # Initialise: only the "vacuum" row (last index) is 1.
    # In the standard MPO convention, the vacuum state is the last row.
    data = data.at[0, D_w - 1, 0].set(1.0)
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.IN,  label="env_mps_l"),
        TensorIndex(sym, bond_mpo, FlowDirection.IN,  label="env_mpo_l"),
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_conj_l"),
    )
    return DenseTensor(data, indices)


def _trivial_right_env(D_w: int, dtype: Any = jnp.float32) -> DenseTensor:
    """Trivial (1, D_w, 1) right environment for iDMRG."""
    sym = U1Symmetry()
    bond_mps = np.zeros(1, dtype=np.int32)
    bond_mpo = np.zeros(D_w, dtype=np.int32)
    data = jnp.zeros((1, D_w, 1), dtype=dtype)
    # Only the "done" row (index 0) is 1.
    data = data.at[0, 0, 0].set(1.0)
    indices = (
        TensorIndex(sym, bond_mps, FlowDirection.OUT, label="env_mps_r"),
        TensorIndex(sym, bond_mpo, FlowDirection.OUT, label="env_mpo_r"),
        TensorIndex(sym, bond_mps, FlowDirection.IN,  label="env_mps_conj_r"),
    )
    return DenseTensor(data, indices)


def _idmrg_matvec(
    theta_flat: jax.Array,
    theta_shape: tuple[int, ...],
    L_env: jax.Array,
    W_l: jax.Array,
    W_r: jax.Array,
    R_env: jax.Array,
) -> jax.Array:
    """Apply effective Hamiltonian to 2-site wavefunction (iDMRG version)."""
    theta = theta_flat.reshape(theta_shape)
    result = jnp.einsum(
        "abc,apqd,bpse,eqtf,dfg->cstg",
        L_env, theta, W_l, W_r, R_env,
    )
    return result.ravel()


_idmrg_matvec_jit = jax.jit(_idmrg_matvec, static_argnums=(1,))


def _compute_local_energy(
    theta: jax.Array,
    W_bulk: jax.Array,
    d: int,
) -> float:
    """Compute the energy per site from the 2-site wavefunction.

    Evaluates ``<theta|H_bond|theta>`` where ``H_bond`` is the nearest-
    neighbour Hamiltonian extracted from the bulk MPO's vacuum→done
    transition.  For translationally invariant nearest-neighbour models,
    this equals the energy per bond = energy per site.

    Args:
        theta: Optimised 2-site wavefunction, shape (chi_l, d, d, chi_r).
        W_bulk: Bulk MPO tensor, shape (D_w, d, d, D_w).
        d: Physical dimension.

    Returns:
        Energy per site (float).
    """
    # Build 2-site Hamiltonian from the MPO: H[p,q,p',q'] = sum_e W[D-1,p,p',e] * W[e,q,q',0]
    # (vacuum row of left site → done column of right site).
    D_w = W_bulk.shape[0]
    W_left = W_bulk[D_w - 1, :, :, :]   # (d, d, D_w) — vacuum row
    W_right = W_bulk[:, :, :, 0]         # (D_w, d, d) — done column
    # H_2site[p, p', q, q'] = sum_e W_left[p, p', e] * W_right[e, q, q']
    H_2site = jnp.einsum("abe,ecd->abcd", W_left, W_right)
    # Contract indices: H_2site[p_top, p_bot, q_top, q_bot]

    # <theta|H_2site|theta> with theta[a, p, q, b]:
    # = sum_{a,b} sum_{p,q,p',q'} conj(theta[a,p',q',b]) * H[p',p,q',q] * theta[a,p,q,b]
    # Wait, let me be careful with bra vs ket indices.
    # H acts as: H|p,q> = sum_{p',q'} H[p',q',p,q] |p',q'>
    # <theta|H|theta> = sum_{a,b,p,q,p',q'} conj(theta[a,p',q',b]) * H[p',q',p,q] * theta[a,p,q,b]
    # H_2site[p_top, p_bot, q_top, q_bot]:
    #   p_top, q_top = bra (output) physical indices
    #   p_bot, q_bot = ket (input) physical indices
    energy = jnp.einsum(
        "asrb,PsQr,aPQb->",
        jnp.conj(theta), H_2site, theta,
    )
    norm = jnp.einsum("apqb,apqb->", jnp.conj(theta), theta)
    return float(energy / norm)


def _update_left_env_dense(
    L_env: jax.Array,
    A: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Update left environment (raw arrays, always 3-leg).

    L_env: (chi_l, D_w, chi_l)
    A:     (chi_l, d, chi_r)
    W:     (D_w_l, d, d, D_w_r)
    returns: (chi_r, D_w_r, chi_r)
    """
    return jnp.einsum("abc,apd,bpxe,cxf->def", L_env, A, W, jnp.conj(A))


def _update_right_env_dense(
    R_env: jax.Array,
    B: jax.Array,
    W: jax.Array,
) -> jax.Array:
    """Update right environment (raw arrays, always 3-leg).

    R_env: (chi_r, D_w, chi_r)
    B:     (chi_l, d, chi_r)
    W:     (D_w_l, d, d, D_w_r)
    returns: (chi_l, D_w_l, chi_l)
    """
    return jnp.einsum("abc,dpa,epxb,fxc->def", R_env, B, W, jnp.conj(B))


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def idmrg(
    bulk_mpo: DenseTensor,
    config: iDMRGConfig | None = None,
    d: int = 2,
    dtype: Any = jnp.float32,
) -> iDMRGResult:
    """Run infinite DMRG to find the ground-state energy per site.

    Args:
        bulk_mpo: Bulk MPO tensor (D_w, d, d, D_w) as a ``DenseTensor``.
        config:   iDMRG configuration. Uses defaults if *None*.
        d:        Physical dimension.
        dtype:    JAX dtype for computation.

    Returns:
        ``iDMRGResult`` with energy per site and diagnostic information.
    """
    if config is None:
        config = iDMRGConfig()

    W = bulk_mpo.todense()  # (D_w, d, d, D_w)
    D_w = W.shape[0]

    # ---- Initialise environments ----
    L_env = _trivial_left_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)
    R_env = _trivial_right_env(D_w, dtype=dtype).todense()  # (1, D_w, 1)

    energies_per_step: list[float] = []
    e_per_site = 0.0
    converged = False
    chi_env = 1
    key = jax.random.PRNGKey(0)
    s_vals = jnp.ones(1, dtype=dtype)
    theta_prev: jax.Array | None = None
    E_prev: float | None = None  # previous Lanczos eigenvalue

    for step in range(config.max_iterations):
        # ---- Form initial two-site wavefunction theta ----
        if theta_prev is not None and theta_prev.shape == (chi_env, d, d, chi_env):
            theta = theta_prev
        elif theta_prev is not None:
            old_chi = theta_prev.shape[0]
            theta = jnp.zeros((chi_env, d, d, chi_env), dtype=dtype)
            theta = theta.at[:old_chi, :, :, :old_chi].set(theta_prev)
            key, subkey = jax.random.split(key)
            noise = 1e-3 * jax.random.normal(
                subkey, (chi_env, d, d, chi_env), dtype=dtype
            )
            theta = theta + noise
        else:
            key, subkey = jax.random.split(key)
            theta = jax.random.normal(
                subkey, (chi_env, d, d, chi_env), dtype=dtype
            )
        theta = theta / jnp.linalg.norm(theta)

        theta_shape = theta.shape
        theta_flat = theta.ravel()

        # ---- Solve eigenvalue problem via Lanczos ----
        _ts = theta_shape
        _le = L_env
        _re = R_env

        def matvec(v: jax.Array) -> jax.Array:
            return _idmrg_matvec_jit(v, _ts, _le, W, W, _re)

        E_total, theta_opt_flat = _lanczos_solve(
            matvec, theta_flat, config.lanczos_max_iter, config.lanczos_tol
        )
        E_total = float(E_total)

        theta_opt = theta_opt_flat.reshape(theta_shape)

        # ---- SVD and truncate ----
        chi_l, d_l, d_r, chi_r = theta_shape
        matrix = theta_opt.reshape(chi_l * d_l, d_r * chi_r)
        U, s_full, Vt = jnp.linalg.svd(matrix, full_matrices=False)

        n_keep = min(config.max_bond_dim, len(s_full))
        if config.svd_trunc_err is not None:
            total_sq = float(jnp.sum(s_full**2))
            cumul_sq = jnp.cumsum(s_full[::-1] ** 2)[::-1]
            mask = cumul_sq > (config.svd_trunc_err**2 * total_sq)
            n_by_err = max(int(jnp.sum(mask)), 1)
            n_keep = min(n_keep, n_by_err)

        U = U[:, :n_keep]
        s_vals = s_full[:n_keep]
        Vt = Vt[:n_keep, :]

        # Normalise singular values
        s_norm = jnp.linalg.norm(s_vals)
        if s_norm > 1e-15:
            s_vals = s_vals / s_norm

        # A_L: left-isometric (from U columns)
        A_L = U.reshape(chi_l, d_l, n_keep)
        # A_R_iso: right-isometric (from Vt rows, no singular values)
        A_R_iso = Vt.reshape(n_keep, d_r, chi_r)

        # ---- Update environments with isometric tensors ----
        L_env_new = _update_left_env_dense(L_env, A_L, W)
        R_env_new = _update_right_env_dense(R_env, A_R_iso, W)

        # ---- Compute energy per site via energy difference ----
        if E_prev is not None:
            e_per_site = (E_total - E_prev) / 2.0
        else:
            e_per_site = E_total / 2.0
        energies_per_step.append(e_per_site)

        if config.verbose:
            print(
                f"iDMRG step {step + 1}: E_total={E_total:.10f}, "
                f"e/site={e_per_site:.10f}, chi={n_keep}"
            )

        # ---- Check convergence (rolling average to handle oscillation) ----
        n_e = len(energies_per_step)
        if n_e >= 4:
            n_half = min(n_e // 2, 5)
            avg_recent = sum(energies_per_step[-n_half:]) / n_half
            avg_prev = sum(energies_per_step[-2 * n_half:-n_half]) / n_half
            if abs(avg_recent - avg_prev) < config.convergence_tol:
                converged = True
                if config.verbose:
                    print(f"Converged at step {step + 1}")
                break

        # ---- Prepare for next iteration ----
        E_prev = E_total
        theta_prev = theta_opt
        chi_env = n_keep
        L_env = L_env_new
        R_env = R_env_new

    # ---- Wrap final MPS tensors ----
    sym = U1Symmetry()

    def _wrap_mps(data: jax.Array, labels: tuple[str, ...]) -> DenseTensor:
        indices = tuple(
            TensorIndex(
                sym,
                np.zeros(data.shape[k], dtype=np.int32),
                FlowDirection.IN if k < data.ndim - 1 else FlowDirection.OUT,
                label=labels[k],
            )
            for k in range(data.ndim)
        )
        return DenseTensor(data, indices)

    A_L_tensor = _wrap_mps(A_L, ("v_l", "p_l", "v_c"))
    # Return A_R with singular values absorbed for a complete MPS
    A_R_sv = (jnp.diag(s_vals) @ Vt).reshape(n_keep, d, chi_r)
    A_R_tensor = _wrap_mps(A_R_sv, ("v_c", "p_r", "v_r"))

    # Report energy as average of last few steps to smooth oscillation
    n_avg = min(10, len(energies_per_step))
    e_per_site_avg = sum(energies_per_step[-n_avg:]) / n_avg

    return iDMRGResult(
        energy_per_site=e_per_site_avg,
        energies_per_step=energies_per_step,
        mps_tensors=[A_L_tensor, A_R_tensor],
        singular_values=s_vals,
        converged=converged,
    )
