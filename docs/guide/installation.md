# Installation

## Prerequisites

- Python 3.11 or 3.12
- A working JAX installation (CPU or GPU)

## Install with pip

```bash
pip install tnjax
```

## Install with uv (recommended for development)

```bash
# Clone the repository
git clone https://github.com/yingjerkao/TN-Jax.git
cd TN-Jax

# Install in development mode with all extras
uv sync --all-extras --dev
```

## GPU support

TN-Jax uses JAX as its backend. To run on GPU, install the appropriate
`jaxlib` build *before* installing TN-Jax:

```bash
# CUDA 12
pip install jax[cuda12]

# Then install TN-Jax
pip install tnjax
```

See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for other accelerator options.

## Building the documentation

```bash
uv sync --extra docs
cd docs && uv run make html
```

The built site will be in `docs/_build/html/`.

## Verifying the installation

```python
import tnjax
print(tnjax.__version__)
```
