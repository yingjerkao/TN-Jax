#!/usr/bin/env python3
"""Infinite cylinder Heisenberg model via iDMRG with super-site MPO.

Each "super-site" bundles an entire ring of Ly spins (physical dimension
d = 2^Ly).  Within-ring Heisenberg bonds (periodic in y) become an on-site
MPO term, while between-ring bonds are nearest-neighbour MPO interactions.
This lets us use the standard ``idmrg()`` function without modification.

Usage::

    uv run python examples/heisenberg_infinite_cylinder.py
"""

from __future__ import annotations

import time

import jax
import numpy as np

jax.config.update("jax_enable_x64", True)

from tnjax import build_bulk_mpo_heisenberg_cylinder, iDMRGConfig, idmrg


# ---------------------------------------------------------------------------
# Exact diagonalisation for a 2-ring cylinder (cross-check for Ly=2)
# ---------------------------------------------------------------------------


def ed_two_ring_cylinder(Ly: int, J: float = 1.0) -> float:
    """Exact ground-state energy of a 2-ring Heisenberg cylinder.

    Two rings of Ly spins each → N = 2*Ly sites, Hilbert space = 2^N.
    Bonds: within-ring (periodic y) for each ring + between-ring for each y.
    """
    N = 2 * Ly
    dim = 2**N
    Sz = np.array([[0.5, 0.0], [0.0, -0.5]])
    Sp = np.array([[0.0, 1.0], [0.0, 0.0]])
    Sm = np.array([[0.0, 0.0], [1.0, 0.0]])
    I2 = np.eye(2)

    def embed(op: np.ndarray, site: int) -> np.ndarray:
        parts = [I2] * N
        parts[site] = op
        result = parts[0]
        for p in parts[1:]:
            result = np.kron(result, p)
        return result

    H = np.zeros((dim, dim))
    for ring in range(2):
        seen_bonds: set[tuple[int, int]] = set()
        for y in range(Ly):
            y_next = (y + 1) % Ly
            if Ly == 1:
                break  # no within-ring bond for Ly=1
            bond = (min(y, y_next), max(y, y_next))
            if bond in seen_bonds:
                continue
            seen_bonds.add(bond)
            i = ring * Ly + y
            j = ring * Ly + y_next
            H += J * (embed(Sz, i) @ embed(Sz, j)
                      + 0.5 * (embed(Sp, i) @ embed(Sm, j)
                               + embed(Sm, i) @ embed(Sp, j)))
    for y in range(Ly):
        i = y
        j = Ly + y
        H += J * (embed(Sz, i) @ embed(Sz, j)
                  + 0.5 * (embed(Sp, i) @ embed(Sm, j)
                           + embed(Sm, i) @ embed(Sp, j)))

    return float(np.linalg.eigvalsh(H)[0])


# ---------------------------------------------------------------------------
# Run iDMRG for a given Ly
# ---------------------------------------------------------------------------


def run_cylinder_idmrg(
    Ly: int,
    chi: int,
    max_iterations: int = 200,
    convergence_tol: float = 1e-8,
    verbose: bool = True,
    ed_check: bool = False,
):
    """Run iDMRG for an infinite cylinder of circumference Ly."""
    d = 2**Ly
    D_w = 3 * Ly + 2
    n_bonds = 2 * Ly  # Ly within-ring + Ly between-ring per unit cell

    print(f"\n{'='*60}")
    print(f"  Infinite Heisenberg cylinder  Ly={Ly}")
    print(f"  d={d}, D_w={D_w}, chi={chi}")
    print(f"{'='*60}")

    W = build_bulk_mpo_heisenberg_cylinder(Ly)
    config = iDMRGConfig(
        max_bond_dim=chi,
        max_iterations=max_iterations,
        convergence_tol=convergence_tol,
        lanczos_max_iter=50,
        lanczos_tol=1e-14,
        verbose=verbose,
    )

    t0 = time.perf_counter()
    result = idmrg(W, config, d=d)
    dt = time.perf_counter() - t0

    e_per_spin = result.energy_per_site / Ly
    print(f"\n  iDMRG finished in {dt:.1f}s")
    print(f"  Energy per super-site: {result.energy_per_site:.10f}")
    print(f"  Energy per spin:       {e_per_spin:.10f}")
    print(f"  Bonds per unit cell:   {n_bonds}")
    print(f"  Converged: {result.converged}")

    if ed_check:
        e_ed = ed_two_ring_cylinder(Ly)
        e_ed_per_spin = e_ed / (2 * Ly)
        print(f"\n  ED (2 rings): E_exact = {e_ed:.10f}")
        print(f"  ED E/spin:    {e_ed_per_spin:.10f}")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    print("Infinite Heisenberg cylinder via iDMRG (super-site MPO)")
    print("H = J * sum_{<i,j>} S_i . S_j   (J=1, antiferromagnetic)")

    # Only even Ly is meaningful: odd circumference creates frustrated
    # odd-length cycles incompatible with AFM (Néel) order.

    # Ly=2: small cylinder, fast convergence, ED cross-check
    run_cylinder_idmrg(Ly=2, chi=64, ed_check=True)

    # Ly=4: moderate cylinder (d=16, D_w=14)
    run_cylinder_idmrg(Ly=4, chi=200, max_iterations=300, convergence_tol=1e-4)


if __name__ == "__main__":
    main()
