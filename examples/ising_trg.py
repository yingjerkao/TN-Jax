#!/usr/bin/env python3
"""2D classical Ising model partition function via TRG.

Computes the free energy per site of the 2D classical Ising model on an
infinite square lattice using the Tensor Renormalization Group (TRG)
algorithm (Levin & Nave, PRL 99, 120601, 2007).

Results are compared against the exact Onsager solution at several
temperatures: above, near, and below the critical temperature
Tc = 2 / ln(1 + sqrt(2)) ~ 2.269 (i.e. beta_c ~ 0.4407).

Usage::

    uv run python examples/ising_trg.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np

from tnjax import TRGConfig, compute_ising_tensor, ising_free_energy_exact, trg


def run_trg_ising(
    beta: float,
    max_bond_dim: int = 24,
    num_steps: int = 20,
) -> tuple[float, float, float]:
    """Run TRG for the 2D Ising model at inverse temperature beta.

    Args:
        beta:          Inverse temperature 1/(k_B T).
        max_bond_dim:  Maximum bond dimension (chi) for TRG truncation.
        num_steps:     Number of coarse-graining iterations.

    Returns:
        Tuple of (trg_free_energy, exact_free_energy, relative_error).
    """
    tensor = compute_ising_tensor(beta=beta)
    config = TRGConfig(max_bond_dim=max_bond_dim, num_steps=num_steps)

    log_z_per_n = trg(tensor, config)
    trg_free_energy = float(-log_z_per_n / beta)
    exact_free_energy = ising_free_energy_exact(beta)
    relative_error = abs(trg_free_energy - exact_free_energy) / abs(exact_free_energy)

    return trg_free_energy, exact_free_energy, relative_error


def main():
    beta_c = np.log(1 + np.sqrt(2)) / 2  # ~ 0.4407

    temperatures = [
        ("High T  (T = 4.0)", 1.0 / 4.0),
        ("Above Tc (T = 2.5)", 1.0 / 2.5),
        ("Near Tc  (T ~ 2.27)", beta_c),
        ("Below Tc (T = 2.0)", 1.0 / 2.0),
        ("Low T   (T = 1.5)", 1.0 / 1.5),
    ]

    chi = 24
    num_steps = 20

    print("2D Classical Ising Model: TRG vs Onsager Exact Solution")
    print(f"TRG parameters: chi = {chi}, num_steps = {num_steps}")
    print(f"Critical temperature: Tc = {1.0 / beta_c:.4f}, beta_c = {beta_c:.4f}")
    print()
    print(
        f"{'Label':<20s} {'beta':>8s} {'f_TRG':>14s} {'f_exact':>14s} {'rel error':>12s}"
    )
    print("-" * 72)

    for label, beta in temperatures:
        f_trg, f_exact, rel_err = run_trg_ising(
            beta=beta,
            max_bond_dim=chi,
            num_steps=num_steps,
        )
        print(f"{label:<20s} {beta:8.4f} {f_trg:14.8f} {f_exact:14.8f} {rel_err:12.2e}")

    print()
    print("Note: TRG accuracy degrades near the critical point due to the")
    print("divergent correlation length. Increasing chi improves accuracy.")


if __name__ == "__main__":
    main()
