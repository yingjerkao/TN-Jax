"""iPEPS AD optimization benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tnjax import CTMConfig, iPEPSConfig, optimize_gs_ad

_SIZES = {
    "small": {"D": 2, "chi_ctm": 8, "gs_steps": 20, "lr": 1e-3},
    "medium": {"D": 2, "chi_ctm": 16, "gs_steps": 30, "lr": 1e-3},
    "large": {"D": 3, "chi_ctm": 16, "gs_steps": 20, "lr": 5e-4},
}


def _heisenberg_gate(dtype=jnp.float64) -> jnp.ndarray:
    """Build H = Sz*Sz + 0.5*(S+*S- + S-*S+) as (2,2,2,2) tensor."""
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = (
        jnp.kron(Sz, Sz)
        + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    )
    return H.reshape(2, 2, 2, 2)


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p, dtype=dtype):
            def setup():
                gate = _heisenberg_gate(dtype=dtype)
                config = iPEPSConfig(
                    max_bond_dim=p["D"],
                    ctm=CTMConfig(chi=p["chi_ctm"]),
                    gs_learning_rate=p["lr"],
                    gs_num_steps=p["gs_steps"],
                )
                return (gate, None, config)

            def run(gate, A_init, config):
                return optimize_gs_ad(gate, A_init, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "ipeps_ad",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
