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

## Hardware acceleration

TN-Jax uses JAX as its backend. Install with a hardware-specific extra to
enable GPU or TPU acceleration:

```bash
# NVIDIA GPU (CUDA 13, recommended)
pip install tnjax[cuda13]

# NVIDIA GPU (CUDA 12)
pip install tnjax[cuda12]

# NVIDIA GPU with locally installed CUDA
pip install tnjax[cuda12-local]
pip install tnjax[cuda13-local]

# Google Cloud TPU
pip install tnjax[tpu]

# Apple Silicon GPU (macOS only, experimental)
pip install tnjax[metal]
```

For AMD ROCm GPUs, install JAX with ROCm support separately following
[AMD's installation guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/jax-install.html),
then install TN-Jax on top:

```bash
# After installing jax+jaxlib with ROCm
pip install tnjax
```

See the [JAX installation guide](https://docs.jax.dev/en/latest/installation.html) for the latest accelerator options.

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
