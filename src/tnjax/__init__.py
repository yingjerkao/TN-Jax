"""TN-Jax: JAX-based tensor network library with symmetry-aware block-sparse tensors.

Label-based contraction (Cytnx-style):
    Tensor legs carry string or integer labels. Two legs with the same label
    across different tensors are automatically contracted when contract() is called.

Quick start::

    import jax
    import numpy as np
    from tnjax import (
        U1Symmetry, TensorIndex, FlowDirection,
        SymmetricTensor, TensorNetwork, contract
    )

    u1 = U1Symmetry()
    key = jax.random.PRNGKey(0)

    A = SymmetricTensor.random_normal(
        indices=(
            TensorIndex(u1, np.array([-1, 1], dtype=np.int32), FlowDirection.IN,  label="phys"),
            TensorIndex(u1, np.array([-1, 0, 1], dtype=np.int32), FlowDirection.IN,  label="left"),
            TensorIndex(u1, np.array([1, 0, -1], dtype=np.int32), FlowDirection.OUT, label="bond"),
        ),
        key=key,
    )
    B = A.relabel("bond", "right").relabel("left", "bond")
    result = contract(A, B)   # contracts over shared label "bond"
    print(result.labels())    # ('phys', 'left', 'phys', 'right')
"""

from tnjax.contraction.contractor import (
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)
from tnjax.core.index import FlowDirection, Label, TensorIndex
from tnjax.core.symmetry import (
    BaseNonAbelianSymmetry,
    BaseSymmetry,
    U1Symmetry,
    ZnSymmetry,
)
from tnjax.core.tensor import BlockKey, DenseTensor, SymmetricTensor, Tensor
from tnjax.network.network import TensorNetwork, build_mps, build_peps

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Symmetries
    "BaseSymmetry",
    "U1Symmetry",
    "ZnSymmetry",
    "BaseNonAbelianSymmetry",
    # Index
    "FlowDirection",
    "Label",
    "TensorIndex",
    # Tensors
    "Tensor",
    "DenseTensor",
    "SymmetricTensor",
    "BlockKey",
    # Contraction
    "contract",
    "contract_with_subscripts",
    "truncated_svd",
    "qr_decompose",
    # Network
    "TensorNetwork",
    "build_mps",
    "build_peps",
]
