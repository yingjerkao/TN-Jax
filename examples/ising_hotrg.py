#!/usr/bin/env python3
"""2D classical Ising partition function via HOTRG.

Computes the free energy per site of the 2D square-lattice Ising model at
several temperatures using Higher-Order Tensor Renormalization Group (HOTRG),
and compares against Onsager's exact solution.

The Ising Hamiltonian is H = -J * sum_{<i,j>} s_i s_j with J=1 (ferromagnet).
The critical temperature is Tc = 2 / ln(1 + sqrt(2)) ~ 2.269, corresponding
to an inverse temperature beta_c ~ 0.4407.

Reference: Xie et al., PRB 86, 045139 (2012).

Usage::

    uv run python examples/ising_hotrg.py
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

from tnjax import HOTRGConfig, compute_ising_tensor, hotrg, ising_free_energy_exact

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

BETA_C = 0.4407  # approximate inverse critical temperature
TEMPERATURES = [3.0, 2.5, 2.269, 2.0, 1.5]  # above, near, below Tc
MAX_BOND_DIM = 16
NUM_STEPS = 20


# ---------------------------------------------------------------------------
# Run HOTRG at several temperatures
# ---------------------------------------------------------------------------


def main():
    print("2D Ising model: free energy via HOTRG")
    print(f"  chi = {MAX_BOND_DIM}, HOTRG steps = {NUM_STEPS}")
    print(f"  Tc ~ {1.0 / BETA_C:.4f}  (beta_c ~ {BETA_C})")
    print()
    print(
        f"  {'T':>8s}  {'beta':>8s}  {'f_HOTRG':>14s}  {'f_exact':>14s}  {'rel_err':>12s}"
    )
    print(f"  {'-' * 8}  {'-' * 8}  {'-' * 14}  {'-' * 14}  {'-' * 12}")

    for T in TEMPERATURES:
        beta = 1.0 / T

        # Build the initial 2x2x2x2 Boltzmann weight tensor
        tensor = compute_ising_tensor(beta=beta)

        # Run HOTRG coarse-graining
        config = HOTRGConfig(
            max_bond_dim=MAX_BOND_DIM,
            num_steps=NUM_STEPS,
            direction_order="alternating",
        )
        log_z_per_n = hotrg(tensor, config)

        # Free energy per site: f = -T * ln(Z) / N = -ln(Z) / (N * beta)
        f_hotrg = float(-log_z_per_n / beta)
        f_exact = ising_free_energy_exact(beta)

        rel_err = abs(f_hotrg - f_exact) / abs(f_exact)
        print(
            f"  {T:8.3f}  {beta:8.4f}  {f_hotrg:14.8f}  {f_exact:14.8f}  {rel_err:12.2e}"
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
