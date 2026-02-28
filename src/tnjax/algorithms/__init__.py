"""Tensor network algorithms: DMRG, iDMRG, TRG, HOTRG, iPEPS."""

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
from tnjax.algorithms.idmrg import (
    build_bulk_mpo_heisenberg,
    build_bulk_mpo_heisenberg_cylinder,
    iDMRGConfig,
    iDMRGResult,
    idmrg,
)
from tnjax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    compute_energy_ctm_2site,
    ctm,
    ctm_2site,
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
    # iDMRG
    "iDMRGConfig",
    "iDMRGResult",
    "idmrg",
    "build_bulk_mpo_heisenberg",
    "build_bulk_mpo_heisenberg_cylinder",
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
    "ctm_2site",
    "compute_energy_ctm_2site",
    "optimize_gs_ad",
    # iPEPS Excitations
    "ExcitationConfig",
    "ExcitationResult",
    "compute_excitations",
    "make_momentum_path",
]
