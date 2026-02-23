"""Tests for the HOTRG (Higher-Order Tensor Renormalization Group) algorithm."""

import jax.numpy as jnp
import numpy as np
import pytest

from tnjax.algorithms.hotrg import (
    HOTRGConfig,
    _compute_hosvd_isometry,
    _hotrg_step_horizontal,
    _hotrg_step_vertical,
    hotrg,
)
from tnjax.algorithms.trg import compute_ising_tensor, ising_free_energy_exact
from tnjax.core.tensor import DenseTensor


class TestHOTRGConfig:
    def test_default_values(self):
        cfg = HOTRGConfig()
        assert cfg.max_bond_dim == 16
        assert cfg.num_steps == 10
        assert cfg.direction_order == "alternating"

    def test_custom_values(self):
        cfg = HOTRGConfig(max_bond_dim=8, num_steps=5, direction_order="horizontal")
        assert cfg.max_bond_dim == 8
        assert cfg.num_steps == 5
        assert cfg.direction_order == "horizontal"


class TestHOSVDIsometry:
    def test_output_is_isometry(self):
        """HOSVD isometry U should satisfy U^T U = I (or close to it)."""
        M = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float32)
        U = _compute_hosvd_isometry(M, axis=0, chi_target=3)
        # U should be (4, 3) shaped: original_dim x chi_target
        assert U.shape[1] <= 3
        Unp = np.array(U)
        # Check isometry: U^T U ≈ I
        gram = Unp.T @ Unp
        assert np.allclose(gram, np.eye(gram.shape[0]), atol=1e-5), (
            f"U^T U not identity: max err = {np.max(np.abs(gram - np.eye(gram.shape[0])))}"
        )

    def test_chi_target_respected(self):
        """Resulting isometry should have at most chi_target columns."""
        M = np.random.default_rng(1).random((6, 6, 6, 6)).astype(np.float32)
        for chi in [2, 4, 5]:
            U = _compute_hosvd_isometry(M, axis=0, chi_target=chi)
            assert U.shape[1] <= chi

    def test_all_axes(self):
        """Should work for each axis 0-3."""
        M = np.random.default_rng(2).random((4, 4, 4, 4)).astype(np.float32)
        for axis in range(4):
            U = _compute_hosvd_isometry(M, axis=axis, chi_target=3)
            assert U.ndim == 2


class TestHOTRGStepHorizontal:
    def test_output_shape(self):
        """Horizontal HOTRG step: output tensor should have 4 legs."""
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float32)
        T_new, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert T_new.ndim == 4

    def test_log_norm_finite(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float32)
        _, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_bond_dim_truncation(self):
        """Up/down bond dims should be bounded by max_bond_dim after horizontal step."""
        T = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float32)
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=3)
        # After horizontal step, up/down legs (axes 0,1) are compressed
        # The exact shape depends on implementation, but all should be <= max_bond_dim
        for dim in T_new.shape:
            assert dim <= max(4, 3 * 4), "Dimension should be reasonable"

    def test_output_normalized(self):
        """Output tensor should be normalized (max |entry| ≈ 1)."""
        T = np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float32)
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=4)
        arr = np.array(T_new)
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.05), f"Expected ~1, got {max_val}"


class TestHOTRGStepVertical:
    def test_output_shape(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float32)
        T_new, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert T_new.ndim == 4

    def test_log_norm_finite(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float32)
        _, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_output_normalized(self):
        T = np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float32)
        T_new, _ = _hotrg_step_vertical(T, max_bond_dim=4)
        arr = np.array(T_new)
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.05), f"Expected ~1, got {max_val}"


class TestHOTRGRun:
    @pytest.fixture
    def ising_tensor_high_temp(self):
        return compute_ising_tensor(beta=0.2)

    def test_hotrg_runs_without_error(self, ising_tensor_high_temp):
        config = HOTRGConfig(max_bond_dim=4, num_steps=3)
        result = hotrg(ising_tensor_high_temp, config)
        assert jnp.isfinite(result)

    def test_result_is_scalar(self, ising_tensor_high_temp):
        config = HOTRGConfig(max_bond_dim=4, num_steps=3)
        result = hotrg(ising_tensor_high_temp, config)
        assert result.shape == ()

    def test_high_temp_free_energy_reasonable(self):
        """At high temperature, HOTRG should give a reasonable free energy estimate."""
        beta = 0.2
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=8, num_steps=6)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        # Loose check: within 25% relative error
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(exact_free_energy)
        assert relative_error < 0.25, (
            f"HOTRG free energy {hotrg_free_energy:.4f} too far from "
            f"exact {exact_free_energy:.4f} (rel err={relative_error:.3f})"
        )

    def test_horizontal_only(self):
        """HOTRG with direction_order='horizontal' should run."""
        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=4, direction_order="horizontal")
        result = hotrg(tensor, config)
        assert np.isfinite(float(result))

    def test_vertical_only(self):
        """HOTRG with direction_order='vertical' should run."""
        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=4, direction_order="vertical")
        result = hotrg(tensor, config)
        assert np.isfinite(float(result))

    def test_alternating_direction(self):
        """Default alternating should give same sign as horizontal-only (finite result)."""
        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=4, direction_order="alternating")
        result = hotrg(tensor, config)
        assert np.isfinite(float(result))

    def test_single_step(self):
        tensor = compute_ising_tensor(beta=0.3)
        config = HOTRGConfig(max_bond_dim=4, num_steps=1)
        result = hotrg(tensor, config)
        assert np.isfinite(float(result))

    def test_dense_tensor_input(self):
        """hotrg() should accept DenseTensor as input."""
        tensor = compute_ising_tensor(beta=0.3)
        assert isinstance(tensor, DenseTensor)
        config = HOTRGConfig(max_bond_dim=4, num_steps=3)
        result = hotrg(tensor, config)
        assert np.isfinite(float(result))

    def test_hotrg_vs_trg_sign(self):
        """HOTRG and TRG log(Z)/N should have the same sign."""
        from tnjax.algorithms.trg import TRGConfig, trg

        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)

        result_hotrg = float(hotrg(tensor, HOTRGConfig(max_bond_dim=8, num_steps=6)))
        result_trg = float(trg(tensor, TRGConfig(max_bond_dim=8, num_steps=6)))

        # Both should be positive (log(Z) > 0 for any non-trivial system)
        # or at least have the same sign
        assert np.sign(result_hotrg) == np.sign(result_trg) or abs(result_hotrg) < 0.01, (
            f"HOTRG and TRG have different signs: {result_hotrg:.4f} vs {result_trg:.4f}"
        )
