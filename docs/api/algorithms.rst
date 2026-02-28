Algorithms
==========

DMRG
----

.. autoclass:: tnjax.algorithms.dmrg.DMRGConfig
   :members:
   :no-index:

.. autoclass:: tnjax.algorithms.dmrg.DMRGResult
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.dmrg.dmrg

.. autofunction:: tnjax.algorithms.dmrg.build_mpo_heisenberg

.. autofunction:: tnjax.algorithms.dmrg.build_random_mps

iDMRG
-----

.. autoclass:: tnjax.algorithms.idmrg.iDMRGConfig
   :members:
   :no-index:

.. autoclass:: tnjax.algorithms.idmrg.iDMRGResult
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.idmrg.idmrg

.. autofunction:: tnjax.algorithms.idmrg.build_bulk_mpo_heisenberg

.. autofunction:: tnjax.algorithms.idmrg.build_bulk_mpo_heisenberg_cylinder

TRG
---

.. autoclass:: tnjax.algorithms.trg.TRGConfig
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.trg.trg

.. autofunction:: tnjax.algorithms.trg.compute_ising_tensor

.. autofunction:: tnjax.algorithms.trg.ising_free_energy_exact

HOTRG
-----

.. autoclass:: tnjax.algorithms.hotrg.HOTRGConfig
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.hotrg.hotrg

iPEPS
-----

.. autoclass:: tnjax.algorithms.ipeps.iPEPSConfig
   :members:
   :no-index:

.. autoclass:: tnjax.algorithms.ipeps.CTMConfig
   :members:
   :no-index:

.. autoclass:: tnjax.algorithms.ipeps.CTMEnvironment
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.ipeps.ipeps

.. autofunction:: tnjax.algorithms.ipeps.ctm

.. autofunction:: tnjax.algorithms.ipeps.ctm_2site

.. autofunction:: tnjax.algorithms.ipeps.compute_energy_ctm_2site

.. autofunction:: tnjax.algorithms.ipeps.optimize_gs_ad

AD Utilities
------------

.. autofunction:: tnjax.algorithms.ad_utils.truncated_svd_ad

.. autofunction:: tnjax.algorithms.ad_utils.ctm_converge

iPEPS Excitations
-----------------

.. autoclass:: tnjax.algorithms.ipeps_excitations.ExcitationConfig
   :members:
   :no-index:

.. autoclass:: tnjax.algorithms.ipeps_excitations.ExcitationResult
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.ipeps_excitations.compute_excitations

.. autofunction:: tnjax.algorithms.ipeps_excitations.make_momentum_path

AutoMPO
-------

.. autoclass:: tnjax.algorithms.auto_mpo.AutoMPO
   :members:

.. autoclass:: tnjax.algorithms.auto_mpo.HamiltonianTerm
   :members:
   :no-index:

.. autofunction:: tnjax.algorithms.auto_mpo.build_auto_mpo

.. autofunction:: tnjax.algorithms.auto_mpo.spin_half_ops

.. autofunction:: tnjax.algorithms.auto_mpo.spin_one_ops
