# Installation

## Prerequisites

- Python 3.11 or 3.12
- A working JAX installation (CPU or GPU)

## Install with pip

```bash
pip install tenax-tn
```

## Install with uv (recommended for development)

```bash
# Clone the repository
git clone https://github.com/tenax-lab/tenax.git
cd Tenax

# Install in development mode with all extras
uv sync --all-extras --dev
```

## Hardware acceleration

Tenax uses JAX as its backend. Install with a hardware-specific extra to
enable GPU or TPU acceleration:

```bash
# NVIDIA GPU (CUDA 13, recommended)
pip install tenax-tn[cuda13]

# NVIDIA GPU (CUDA 12)
pip install tenax-tn[cuda12]

# NVIDIA GPU with locally installed CUDA
pip install tenax-tn[cuda12-local]
pip install tenax-tn[cuda13-local]

# Google Cloud TPU
pip install tenax-tn[tpu]

# Apple Silicon GPU (macOS only, experimental)
pip install tenax-tn[metal]
```

For AMD ROCm GPUs, install JAX with ROCm support separately following
[AMD's installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html),
then install Tenax on top:

```bash
# After installing jax+jaxlib with ROCm
pip install tenax-tn
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the latest accelerator options.

## Building the documentation

```bash
uv sync --extra docs
cd docs && uv run make html
```

The built site will be in `docs/_build/html/`.

## Float64 precision

Tenax defaults to `float64` for all tensors and algorithms. Importing
`tenax` automatically enables JAX 64-bit mode via
`jax.config.update("jax_enable_x64", True)`.

If you import JAX *before* `tenax` and create arrays in that window, they
will still be `float32`. To avoid surprises, either import `tenax` first or
enable x64 manually:

```python
import jax
jax.config.update("jax_enable_x64", True)  # before any array creation

import tenax  # also calls the same update
```

See {doc}`gotchas` for more details on float64 behaviour.

## Verifying the installation

```python
import tenax
print(tenax.__version__)
```
