"""Tensor network algorithms: DMRG, TRG, HOTRG, iPEPS."""

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
from tnjax.algorithms.ipeps import CTMConfig, CTMEnvironment, ctm, ipeps, iPEPSConfig
from tnjax.algorithms.trg import (
    TRGConfig,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)

__all__ = [
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
]
