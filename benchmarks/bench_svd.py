#!/usr/bin/env python
"""Benchmark: dense vs block-sparse SVD at DMRG-relevant sizes.

Part 1 — Standalone SVD benchmark:
    Build a 4-leg SymmetricTensor mimicking a 2-site DMRG theta tensor
    (chi, d, d, chi) with U(1) charges, and compare truncated_svd()
    timings for dense vs block-sparse representations.

Part 2 — End-to-end DMRG comparison:
    Run DMRG on a small Heisenberg chain with dense and symmetric
    initial MPS, and compare wall times and final energies.

Usage:
    uv run python benchmarks/bench_svd.py
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from tnjax.contraction.contractor import truncated_svd
from tnjax.core.index import FlowDirection, TensorIndex
from tnjax.core.symmetry import U1Symmetry
from tnjax.core.tensor import DenseTensor, SymmetricTensor

# ---------------------------------------------------------------------------
# Part 1: Standalone SVD benchmark
# ---------------------------------------------------------------------------


def _build_theta_indices(
    chi: int,
    d: int = 2,
) -> tuple[tuple[TensorIndex, ...], np.ndarray, np.ndarray]:
    """Build TensorIndex metadata for a (chi, d, d, chi) theta tensor."""
    sym = U1Symmetry()
    phys_charges = np.array([1, -1], dtype=np.int32)

    # Virtual bond: distribute chi states across Sz = {-1, 0, +1}
    q_each = max(1, chi // 4)
    q_zero = max(1, chi - 2 * q_each)
    virt_charges = np.concatenate([
        np.full(q_each, -1, dtype=np.int32),
        np.full(q_zero, 0, dtype=np.int32),
        np.full(q_each, 1, dtype=np.int32),
    ])[:chi]

    indices = (
        TensorIndex(sym, virt_charges, FlowDirection.IN, label="v_left"),
        TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_left"),
        TensorIndex(sym, phys_charges, FlowDirection.IN, label="p_right"),
        TensorIndex(sym, virt_charges, FlowDirection.OUT, label="v_right"),
    )
    return indices, virt_charges, phys_charges


def bench_svd_standalone() -> None:
    """Benchmark truncated_svd for dense vs block-sparse at various chi."""
    print("=" * 70)
    print("Part 1: Standalone truncated_svd benchmark")
    print("=" * 70)
    print(f"{'chi':>6} {'d':>4} {'dense(ms)':>10} {'sparse(ms)':>11} "
          f"{'speedup':>8} {'fill%':>7} {'n_blocks':>9}")
    print("-" * 70)

    d = 2
    n_warmup = 2
    n_iter = 10

    for chi in [16, 32, 64, 128, 256, 512, 1024]:
        indices, virt_charges, phys_charges = _build_theta_indices(chi, d)

        # Build block-sparse theta
        key = jax.random.PRNGKey(42)
        sym_tensor = SymmetricTensor.random_normal(indices, key=key, dtype=jnp.float32)

        # Build equivalent dense theta
        dense_data = sym_tensor.todense()
        dense_tensor = DenseTensor(dense_data, indices)

        left_labels = ["v_left", "p_left"]
        right_labels = ["p_right", "v_right"]
        max_sv = chi

        # Fill ratio
        total_elements = int(np.prod(dense_data.shape))
        nnz = sum(v.size for v in sym_tensor.blocks.values())
        fill_pct = 100.0 * nnz / total_elements if total_elements > 0 else 0.0

        # --- Dense SVD timing ---
        for _ in range(n_warmup):
            truncated_svd(dense_tensor, left_labels, right_labels,
                          new_bond_label="bond", max_singular_values=max_sv)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            truncated_svd(dense_tensor, left_labels, right_labels,
                          new_bond_label="bond", max_singular_values=max_sv)
        dense_ms = 1000.0 * (time.perf_counter() - t0) / n_iter

        # --- Block-sparse SVD timing ---
        for _ in range(n_warmup):
            truncated_svd(sym_tensor, left_labels, right_labels,
                          new_bond_label="bond", max_singular_values=max_sv)

        t0 = time.perf_counter()
        for _ in range(n_iter):
            truncated_svd(sym_tensor, left_labels, right_labels,
                          new_bond_label="bond", max_singular_values=max_sv)
        sparse_ms = 1000.0 * (time.perf_counter() - t0) / n_iter

        speedup = dense_ms / sparse_ms if sparse_ms > 0 else float("inf")

        print(f"{chi:>6} {d:>4} {dense_ms:>10.2f} {sparse_ms:>11.2f} "
              f"{speedup:>7.2f}x {fill_pct:>6.1f}% {sym_tensor.n_blocks:>9}")

    print()


# ---------------------------------------------------------------------------
# Part 2: End-to-end DMRG comparison
# ---------------------------------------------------------------------------


def bench_dmrg_comparison() -> None:
    """Compare wall time and energy for dense vs symmetric initial MPS."""
    from tnjax.algorithms.dmrg import (
        DMRGConfig,
        build_mpo_heisenberg,
        build_random_mps,
        build_random_symmetric_mps,
        dmrg,
    )

    print("=" * 70)
    print("Part 2: End-to-end DMRG comparison (Heisenberg chain)")
    print("=" * 70)

    L = 6
    chi = 8
    n_sweeps = 6
    lanczos_iter = 20

    mpo = build_mpo_heisenberg(L, Jz=1.0, Jxy=1.0)
    config = DMRGConfig(
        max_bond_dim=chi,
        num_sweeps=n_sweeps,
        lanczos_max_iter=lanczos_iter,
        convergence_tol=1e-8,
        verbose=False,
    )

    # --- Dense MPS ---
    mps_dense = build_random_mps(L, physical_dim=2, bond_dim=chi, seed=7)
    t0 = time.perf_counter()
    result_dense = dmrg(mpo, mps_dense, config)
    t_dense = time.perf_counter() - t0

    # --- Symmetric MPS ---
    mps_sym = build_random_symmetric_mps(L, bond_dim=chi, seed=7)
    t0 = time.perf_counter()
    result_sym = dmrg(mpo, mps_sym, config)
    t_sym = time.perf_counter() - t0

    print(f"  Chain length L = {L}, bond dim chi = {chi}, sweeps = {n_sweeps}")
    print()
    print(f"  {'':>20} {'Energy':>14} {'Wall time':>12} {'Converged':>10}")
    print(f"  {'-' * 56}")
    print(f"  {'Dense MPS':>20} {result_dense.energy:>14.8f} {t_dense:>10.3f} s "
          f"{'Yes' if result_dense.converged else 'No':>10}")
    print(f"  {'Symmetric MPS':>20} {result_sym.energy:>14.8f} {t_sym:>10.3f} s "
          f"{'Yes' if result_sym.converged else 'No':>10}")

    delta_e = abs(result_dense.energy - result_sym.energy)
    speedup = t_dense / t_sym if t_sym > 0 else float("inf")
    print()
    print(f"  Energy difference:  {delta_e:.2e}")
    print(f"  Symmetric / Dense:  {speedup:.2f}x")
    print()


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    bench_svd_standalone()
    bench_dmrg_comparison()
