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

from tnjax.algorithms.auto_mpo import (
    AutoMPO,
    HamiltonianTerm,
    build_auto_mpo,
    spin_half_ops,
    spin_one_ops,
)
from tnjax.algorithms.dmrg import (
    DMRGConfig,
    DMRGResult,
    build_mpo_heisenberg,
    build_random_mps,
    dmrg,
)
from tnjax.algorithms.hotrg import HOTRGConfig, hotrg
from tnjax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    ctm,
    ipeps,
    iPEPSConfig,
    optimize_gs_ad,
)
from tnjax.algorithms.ipeps_excitations import (
    ExcitationConfig,
    ExcitationResult,
    compute_excitations,
    make_momentum_path,
)
from tnjax.algorithms.trg import (
    TRGConfig,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)
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
    # AutoMPO
    "AutoMPO",
    "HamiltonianTerm",
    "build_auto_mpo",
    "spin_half_ops",
    "spin_one_ops",
    # DMRG
    "DMRGConfig",
    "DMRGResult",
    "dmrg",
    "build_mpo_heisenberg",
    "build_random_mps",
    # TRG
    "TRGConfig",
    "trg",
    "compute_ising_tensor",
    "ising_free_energy_exact",
    # HOTRG
    "HOTRGConfig",
    "hotrg",
    # iPEPS
    "iPEPSConfig",
    "CTMConfig",
    "CTMEnvironment",
    "ipeps",
    "ctm",
    "optimize_gs_ad",
    # iPEPS Excitations
    "ExcitationConfig",
    "ExcitationResult",
    "compute_excitations",
    "make_momentum_path",
    # Network
    "TensorNetwork",
    "build_mps",
    "build_peps",
]
