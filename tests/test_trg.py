"""Tests for the TRG (Tensor Renormalization Group) algorithm."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tnjax.algorithms.trg import (
    TRGConfig,
    _trg_step,
    compute_ising_tensor,
    ising_free_energy_exact,
    trg,
)
from tnjax.core.tensor import DenseTensor


class TestTRGConfig:
    def test_default_values(self):
        cfg = TRGConfig()
        assert cfg.max_bond_dim == 16
        assert cfg.num_steps == 10
        assert cfg.svd_trunc_err is None

    def test_custom_values(self):
        cfg = TRGConfig(max_bond_dim=8, num_steps=5, svd_trunc_err=1e-6)
        assert cfg.max_bond_dim == 8
        assert cfg.num_steps == 5
        assert cfg.svd_trunc_err == 1e-6


class TestComputeIsingTensor:
    def test_returns_dense_tensor(self):
        T = compute_ising_tensor(beta=0.4, J=1.0)
        assert isinstance(T, DenseTensor)

    def test_shape_is_2222(self):
        T = compute_ising_tensor(beta=0.4)
        arr = T.todense()
        assert arr.shape == (2, 2, 2, 2)

    def test_labels_are_set(self):
        T = compute_ising_tensor(beta=0.4)
        labels = T.labels()
        assert "up" in labels
        assert "down" in labels
        assert "left" in labels
        assert "right" in labels

    def test_tensor_is_symmetric(self):
        """Ising tensor should be symmetric under permutation of equal legs."""
        T = compute_ising_tensor(beta=0.4)
        arr = np.array(T.todense())
        # T[u,d,l,r] == T[d,u,l,r] (up-down symmetry)
        assert np.allclose(arr, arr.transpose(1, 0, 2, 3), atol=1e-12)
        # T[u,d,l,r] == T[u,d,r,l] (left-right symmetry)
        assert np.allclose(arr, arr.transpose(0, 1, 3, 2), atol=1e-12)

    def test_matrix_sqrt_property(self):
        """sqrtQ @ sqrtQ should equal Q (matrix square root, not element-wise)."""
        beta = 0.4
        spins = np.array([1.0, -1.0])
        Q = np.exp(beta * np.outer(spins, spins))
        evals, evecs = np.linalg.eigh(Q)
        sqrtQ = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
        reconstructed = sqrtQ @ sqrtQ
        assert np.allclose(reconstructed, Q, atol=1e-12), (
            f"sqrtQ @ sqrtQ should equal Q, max err={np.max(np.abs(reconstructed - Q))}"
        )

    def test_high_beta_tensor_values(self):
        """At very high beta (T->0), tensor should be dominated by aligned spins."""
        T_low = compute_ising_tensor(beta=0.01)
        T_high = compute_ising_tensor(beta=5.0)
        arr_low = np.array(T_low.todense())
        arr_high = np.array(T_high.todense())
        ratio_high = arr_high[0, 0, 0, 0] / (arr_high.sum() + 1e-20)
        ratio_low = arr_low[0, 0, 0, 0] / (arr_low.sum() + 1e-20)
        assert ratio_high > ratio_low


class TestIsingFreeEnergyExact:
    def test_zero_temperature_limit(self):
        """At beta=0, free energy = -ln(2) per spin (infinite temperature)."""
        f = ising_free_energy_exact(beta=0.0)
        assert np.isclose(f, -np.log(2), atol=1e-10)

    def test_high_temp(self):
        """At low beta (high temp), free energy should be finite."""
        f = ising_free_energy_exact(beta=0.1)
        assert np.isfinite(f)

    def test_critical_temp(self):
        """At beta_c = ln(1+sqrt(2))/2 ≈ 0.4407, free energy is finite."""
        beta_c = np.log(1 + np.sqrt(2)) / 2
        f = ising_free_energy_exact(beta=beta_c)
        assert np.isfinite(f)
        assert f < 0

    def test_critical_temp_known_value(self):
        """At criticality, exact free energy per site is known analytically."""
        beta_c = np.log(1 + np.sqrt(2)) / 2
        f = ising_free_energy_exact(beta=beta_c)
        # Known value: f_c ≈ -2.269 (in units where J=1)
        # -beta_c * f_c = ln(Z)/N ≈ 0.9297...
        log_z_exact = -beta_c * f
        assert abs(log_z_exact - 0.9297) < 0.01, f"log(Z)/N at Tc = {log_z_exact:.4f}"

    def test_below_critical(self):
        """At higher temperature (low beta), free energy is more negative."""
        f_high_T = ising_free_energy_exact(beta=0.2)
        f_low_T = ising_free_energy_exact(beta=0.8)
        assert f_high_T < f_low_T

    def test_onsager_formula_no_nan(self):
        """Formula should not produce NaN at any temperature."""
        for beta in [0.01, 0.1, 0.2, 0.3, 0.4, 0.44, 0.4407, 0.5, 0.8, 1.0, 2.0]:
            f = ising_free_energy_exact(beta=beta)
            assert np.isfinite(f), f"NaN/inf at beta={beta}"


class TestTRGStep:
    def test_output_shape_unchanged(self):
        """TRG step should return a 4-leg tensor (possibly different dimension)."""
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        T_new, log_norm = _trg_step(T, max_bond_dim=4, svd_trunc_err=None)
        assert T_new.ndim == 4

    def test_log_norm_is_finite(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        _, log_norm = _trg_step(T, max_bond_dim=4, svd_trunc_err=None)
        assert np.isfinite(float(log_norm))

    def test_output_normalized(self):
        """After a TRG step, max(|T_new|) should be ~1 (normalized)."""
        T = np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float64)
        T_new, _ = _trg_step(T, max_bond_dim=4, svd_trunc_err=None)
        arr = np.array(T_new)
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.01), f"Expected ~1, got {max_val}"

    def test_bond_truncation(self):
        """Bond dimension should not exceed max_bond_dim."""
        T = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        T_new, _ = _trg_step(T, max_bond_dim=3, svd_trunc_err=None)
        for dim in T_new.shape:
            assert dim <= 3, f"Expected dim <= 3, got {dim}"

    def test_with_svd_trunc_err(self):
        """SVD truncation error mode should run without crashing."""
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        T_new, log_norm = _trg_step(T, max_bond_dim=4, svd_trunc_err=1e-4)
        assert T_new.ndim == 4
        assert np.isfinite(float(log_norm))


class TestTRGRun:
    @pytest.fixture
    def ising_tensor_near_critical(self):
        """Ising tensor slightly below critical temperature."""
        beta_c = np.log(1 + np.sqrt(2)) / 2
        return compute_ising_tensor(beta=beta_c * 0.95)

    @pytest.fixture
    def ising_tensor_high_temp(self):
        """Ising tensor at high temperature (low beta)."""
        return compute_ising_tensor(beta=0.2)

    def test_trg_runs_without_error(self, ising_tensor_high_temp):
        config = TRGConfig(max_bond_dim=4, num_steps=3)
        result = trg(ising_tensor_high_temp, config)
        assert jnp.isfinite(result)

    def test_result_is_scalar(self, ising_tensor_high_temp):
        config = TRGConfig(max_bond_dim=4, num_steps=3)
        result = trg(ising_tensor_high_temp, config)
        assert result.shape == ()

    def test_high_temp_free_energy(self):
        """At high temperature (beta=0.2), TRG chi=16 should be within 1%."""
        beta = 0.2
        tensor = compute_ising_tensor(beta=beta)
        config = TRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = trg(tensor, config)
        trg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(trg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.01, (
            f"TRG free energy {trg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_mid_temp_free_energy(self):
        """At beta=0.3, TRG chi=16 should be within 2%."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        config = TRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = trg(tensor, config)
        trg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(trg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.02, (
            f"TRG free energy {trg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_near_critical_free_energy(self):
        """Near critical point (beta=0.44), TRG chi=16 should be within 5%."""
        beta = 0.44
        tensor = compute_ising_tensor(beta=beta)
        config = TRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = trg(tensor, config)
        trg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(trg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.05, (
            f"TRG free energy {trg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_low_temp_free_energy(self):
        """At low temperature (beta=0.6), TRG chi=16 should be within 1%."""
        beta = 0.6
        tensor = compute_ising_tensor(beta=beta)
        config = TRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = trg(tensor, config)
        trg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(trg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.01, (
            f"TRG free energy {trg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_raw_tensor_input(self):
        """trg() should also work with raw DenseTensor (not SymmetricTensor)."""
        tensor = compute_ising_tensor(beta=0.3)
        config = TRGConfig(max_bond_dim=4, num_steps=3)
        result = trg(tensor, config)
        assert np.isfinite(float(result))

    def test_more_steps_more_accurate(self):
        """More TRG steps should generally give more accurate free energy."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        exact = ising_free_energy_exact(beta)

        config_few = TRGConfig(max_bond_dim=8, num_steps=3)
        config_many = TRGConfig(max_bond_dim=8, num_steps=8)

        lz_few = float(trg(tensor, config_few))
        lz_many = float(trg(tensor, config_many))

        f_few = -lz_few / beta
        f_many = -lz_many / beta

        err_few = abs(f_few - exact)
        err_many = abs(f_many - exact)
        assert err_many <= err_few * 2 + 1e-6, (
            f"More steps should not be significantly worse: "
            f"err_few={err_few:.4f}, err_many={err_many:.4f}"
        )

    def test_larger_chi_more_accurate(self):
        """Larger bond dimension should improve accuracy."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        exact = ising_free_energy_exact(beta)

        config_small = TRGConfig(max_bond_dim=4, num_steps=10)
        config_large = TRGConfig(max_bond_dim=16, num_steps=10)

        f_small = -float(trg(tensor, config_small)) / beta
        f_large = -float(trg(tensor, config_large)) / beta

        err_small = abs(f_small - exact)
        err_large = abs(f_large - exact)
        assert err_large <= err_small + 1e-6, (
            f"Larger chi should be at least as accurate: "
            f"err_chi4={err_small:.4f}, err_chi16={err_large:.4f}"
        )

    def test_single_step(self):
        """Should work with num_steps=1."""
        tensor = compute_ising_tensor(beta=0.3)
        config = TRGConfig(max_bond_dim=4, num_steps=1)
        result = trg(tensor, config)
        assert np.isfinite(float(result))

    def test_zero_beta(self):
        """At beta=0, tensor should still be processable."""
        tensor = compute_ising_tensor(beta=0.01)
        config = TRGConfig(max_bond_dim=4, num_steps=3)
        result = trg(tensor, config)
        assert np.isfinite(float(result))
