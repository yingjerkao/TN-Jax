"""Backend selection and configuration for TN-Jax benchmarks.

Must call ``configure_backend`` **before** any ``import jax``.
"""

from __future__ import annotations

import os

_BACKEND_MAP = {
    "cpu": "cpu",
    "cuda": "cuda",
    "gpu": "cuda",
    "tpu": "tpu",
    "metal": "METAL",
}


def configure_backend(backend: str) -> None:
    """Set JAX platform env vars. Call before importing JAX."""
    backend = backend.lower()
    if backend == "auto":
        # Let JAX pick the best available backend
        os.environ.setdefault("JAX_ENABLE_X64", "1")
        return
    if backend not in _BACKEND_MAP:
        raise ValueError(
            f"Unknown backend {backend!r}. "
            f"Choose from: {', '.join(sorted(_BACKEND_MAP))} or 'auto'."
        )
    os.environ["JAX_PLATFORMS"] = _BACKEND_MAP[backend]
    # Metal has limited float64 support
    if backend != "metal":
        os.environ["JAX_ENABLE_X64"] = "1"


def get_backend_info() -> dict:
    """Return runtime backend info (call after JAX is imported)."""
    import jax

    devices = jax.devices()
    return {
        "backend": devices[0].platform if devices else "unknown",
        "device_kind": getattr(devices[0], "device_kind", "unknown") if devices else "unknown",
        "device_count": len(devices),
        "x64_enabled": jax.config.x64_enabled,
    }


def default_dtype(backend: str):
    """Return the default dtype for the given backend."""
    import jax
    import jax.numpy as jnp

    if backend.lower() == "metal":
        return jnp.float32
    return jnp.float64 if jax.config.x64_enabled else jnp.float32
