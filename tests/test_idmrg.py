"""Tests for the iDMRG algorithm."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

from tnjax.algorithms.idmrg import (
    build_bulk_mpo_heisenberg,
    idmrg,
    iDMRGConfig,
    iDMRGResult,
)

# ---------------------------------------------------------------------------
# TestiDMRGConfig
# ---------------------------------------------------------------------------


class TestiDMRGConfig:
    def test_default_values(self):
        cfg = iDMRGConfig()
        assert cfg.max_bond_dim == 100
        assert cfg.max_iterations == 200
        assert cfg.convergence_tol == 1e-8
        assert cfg.lanczos_max_iter == 50
        assert cfg.lanczos_tol == 1e-12
        assert cfg.svd_trunc_err is None
        assert cfg.verbose is False

    def test_custom_values(self):
        cfg = iDMRGConfig(
            max_bond_dim=64,
            max_iterations=50,
            convergence_tol=1e-6,
            lanczos_max_iter=30,
            lanczos_tol=1e-10,
            svd_trunc_err=1e-8,
            verbose=True,
        )
        assert cfg.max_bond_dim == 64
        assert cfg.max_iterations == 50
        assert cfg.convergence_tol == 1e-6
        assert cfg.verbose is True


# ---------------------------------------------------------------------------
# TestBuildBulkMPO
# ---------------------------------------------------------------------------


class TestBuildBulkMPO:
    def test_shape(self):
        W = build_bulk_mpo_heisenberg()
        dense = W.todense()
        assert dense.shape == (5, 2, 2, 5)

    def test_labels(self):
        W = build_bulk_mpo_heisenberg()
        labels = W.labels()
        assert "w_l" in labels
        assert "w_r" in labels
        assert "mpo_top" in labels
        assert "mpo_bot" in labels

    def test_dtype_default_float32(self):
        W = build_bulk_mpo_heisenberg(dtype=jnp.float32)
        assert W.todense().dtype == jnp.float32

    def test_matches_build_mpo_heisenberg_bulk(self):
        """The bulk W-matrix should match the internal bulk W from build_mpo_heisenberg."""
        from tnjax.algorithms.dmrg import build_mpo_heisenberg

        # For L=3 the middle site (i=1) is a bulk tensor: (5, 2, 2, 5)
        mpo = build_mpo_heisenberg(L=3, Jz=1.0, Jxy=1.0, hz=0.0)
        W_ref = mpo.get_tensor(1).todense()  # middle site

        W_bulk = build_bulk_mpo_heisenberg(Jz=1.0, Jxy=1.0, hz=0.0).todense()

        np.testing.assert_allclose(
            np.array(W_bulk), np.array(W_ref), atol=1e-6,
            err_msg="Bulk MPO does not match build_mpo_heisenberg middle site"
        )

    def test_invalid_d_raises(self):
        with pytest.raises(ValueError, match="only supports d=2"):
            build_bulk_mpo_heisenberg(d=3)


# ---------------------------------------------------------------------------
# TestiDMRGRun
# ---------------------------------------------------------------------------


class TestiDMRGRun:
    def test_runs_without_error(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=5, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert isinstance(result, iDMRGResult)

    def test_energy_is_finite(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert np.isfinite(result.energy_per_site), (
            f"Energy per site is not finite: {result.energy_per_site}"
        )

    def test_energy_per_site_negative(self):
        """The Heisenberg ground state energy per site should be negative."""
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=16, max_iterations=30, lanczos_max_iter=20)
        result = idmrg(W, cfg)
        assert result.energy_per_site < 0, (
            f"Energy per site should be negative, got {result.energy_per_site}"
        )

    def test_energy_converges_toward_bethe_ansatz(self):
        """With moderate chi, e_0 should approach 1/4 - ln(2) ~ -0.4431."""
        e_exact = 0.25 - math.log(2)  # ~ -0.4431
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)
        cfg = iDMRGConfig(
            max_bond_dim=32,
            max_iterations=100,
            convergence_tol=1e-8,
            lanczos_max_iter=30,
            lanczos_tol=1e-14,
        )
        result = idmrg(W, cfg, dtype=jnp.float64)
        assert abs(result.energy_per_site - e_exact) < 0.01, (
            f"e/site = {result.energy_per_site:.6f} far from "
            f"Bethe ansatz {e_exact:.6f}"
        )

    def test_energy_improves_with_bond_dim(self):
        """Larger bond dimension should give a lower (better) energy."""
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)

        cfg_small = iDMRGConfig(
            max_bond_dim=8, max_iterations=40, lanczos_max_iter=20,
        )
        cfg_large = iDMRGConfig(
            max_bond_dim=24, max_iterations=60, lanczos_max_iter=20,
        )
        res_small = idmrg(W, cfg_small, dtype=jnp.float64)
        res_large = idmrg(W, cfg_large, dtype=jnp.float64)

        assert res_large.energy_per_site <= res_small.energy_per_site + 1e-6, (
            f"chi=24 energy {res_large.energy_per_site:.6f} should be <= "
            f"chi=8 energy {res_small.energy_per_site:.6f}"
        )

    def test_singular_values_returned(self):
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert result.singular_values is not None
        assert len(result.singular_values) > 0
        assert jnp.all(result.singular_values >= 0)

    def test_convergence_flag(self):
        """With enough iterations and a generous tolerance, convergence should be True."""
        W = build_bulk_mpo_heisenberg(dtype=jnp.float64)
        cfg = iDMRGConfig(
            max_bond_dim=16,
            max_iterations=200,
            convergence_tol=1e-4,
            lanczos_max_iter=30,
        )
        result = idmrg(W, cfg, dtype=jnp.float64)
        assert result.converged, (
            f"Expected convergence with tol=1e-4, "
            f"last energies: {result.energies_per_step[-5:]}"
        )

    def test_mps_tensors_shapes(self):
        """The returned MPS tensors should have valid shapes."""
        W = build_bulk_mpo_heisenberg()
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=10, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        A_L, A_R = result.mps_tensors
        # A_L: (chi_l, d, chi_c)  — 3D
        assert A_L.todense().ndim == 3
        # A_R: (chi_c, d, chi_r)  — 3D
        assert A_R.todense().ndim == 3
        # Centre bond should match
        assert A_L.todense().shape[2] == A_R.todense().shape[0]

    def test_energies_per_step_length(self):
        W = build_bulk_mpo_heisenberg()
        n_iter = 7
        cfg = iDMRGConfig(max_bond_dim=8, max_iterations=n_iter, lanczos_max_iter=10)
        result = idmrg(W, cfg)
        assert len(result.energies_per_step) <= n_iter
        assert len(result.energies_per_step) > 0
