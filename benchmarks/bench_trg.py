"""TRG benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import TRGConfig, compute_ising_tensor, trg

_BETA_C = 0.4407

_SIZES = {
    "small": {"chi": 8, "steps": 15},
    "medium": {"chi": 24, "steps": 20},
    "large": {"chi": 48, "steps": 20},
}


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p):
            def setup():
                tensor = compute_ising_tensor(_BETA_C)
                config = TRGConfig(
                    max_bond_dim=p["chi"],
                    num_steps=p["steps"],
                )
                return (tensor, config)

            def run(tensor, config):
                return trg(tensor, config)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "trg",
                "size_label": size_label,
                "params": {"beta": _BETA_C, **p},
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
