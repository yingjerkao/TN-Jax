"""iPEPS simple-update benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import CTMConfig, ipeps, iPEPSConfig

_SIZES = {
    "small": {"D": 2, "chi_ctm": 8, "steps": 50},
    "medium": {"D": 3, "chi_ctm": 16, "steps": 50},
    "large": {"D": 4, "chi_ctm": 24, "steps": 30},
}


def _heisenberg_gate(dtype=jnp.float64) -> jnp.ndarray:
    """Build H = Sz*Sz + 0.5*(S+*S- + S-*S+) as (2,2,2,2) tensor."""
    Sz = jnp.array([[0.5, 0.0], [0.0, -0.5]], dtype=dtype)
    Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]], dtype=dtype)
    Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]], dtype=dtype)
    H = jnp.kron(Sz, Sz) + 0.5 * (jnp.kron(Sp, Sm) + jnp.kron(Sm, Sp))
    return H.reshape(2, 2, 2, 2)


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p, dtype=dtype):
            def setup():
                gate = _heisenberg_gate(dtype=dtype)
                config = iPEPSConfig(
                    max_bond_dim=p["D"],
                    num_imaginary_steps=p["steps"],
                    dt=0.01,
                    ctm=CTMConfig(chi=p["chi_ctm"]),
                )
                return (gate, None, config)

            def run(gate, initial_peps, config):
                return ipeps(gate, initial_peps, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "ipeps",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
