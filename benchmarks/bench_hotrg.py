"""HOTRG benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import HOTRGConfig, compute_ising_tensor, hotrg

_BETA_C = 0.4407

_SIZES = {
    "small": {"chi": 8, "steps": 10},
    "medium": {"chi": 20, "steps": 15},
    "large": {"chi": 40, "steps": 15},
}


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p):
            def setup():
                tensor = compute_ising_tensor(_BETA_C)
                config = HOTRGConfig(
                    max_bond_dim=p["chi"],
                    num_steps=p["steps"],
                )
                return (tensor, config)

            def run(tensor, config):
                return hotrg(tensor, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "hotrg",
                "size_label": size_label,
                "params": {"beta": _BETA_C, **p},
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
