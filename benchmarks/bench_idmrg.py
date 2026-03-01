"""iDMRG benchmark cases."""

from __future__ import annotations

import jax.numpy as jnp

from tenax import build_bulk_mpo_heisenberg, idmrg, iDMRGConfig

_SIZES = {
    "small": {"chi": 32, "iterations": 50},
    "medium": {"chi": 64, "iterations": 100},
    "large": {"chi": 128, "iterations": 50},
}


def get_benchmarks(dtype=jnp.float64) -> list[dict]:
    benchmarks = []
    for size_label, p in _SIZES.items():

        def _make_fns(p=p, dtype=dtype):
            def setup():
                bulk_mpo = build_bulk_mpo_heisenberg(dtype=dtype)
                config = iDMRGConfig(
                    max_bond_dim=p["chi"],
                    max_iterations=p["iterations"],
                    convergence_tol=1e-10,
                    verbose=False,
                )
                return (bulk_mpo, config)

            def run(bulk_mpo, config):
                return idmrg(bulk_mpo, config=config, dtype=dtype)

            return setup, run

        setup_fn, run_fn = _make_fns()
        benchmarks.append(
            {
                "name": "idmrg",
                "size_label": size_label,
                "params": p,
                "setup_fn": setup_fn,
                "run_fn": run_fn,
            }
        )
    return benchmarks
