"""Tests for the HOTRG (Higher-Order Tensor Renormalization Group) algorithm."""

import jax.numpy as jnp
import numpy as np
import pytest

from tenax.algorithms.hotrg import (
    HOTRGConfig,
    _hotrg_step_horizontal,
    _hotrg_step_vertical,
    hotrg,
)
from tenax.algorithms.trg import compute_ising_tensor, ising_free_energy_exact
from tenax.core.tensor import DenseTensor


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


class TestHOTRGStepHorizontal:
    def test_output_shape(self):
        """Horizontal HOTRG step: output tensor should have 4 legs."""
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        T_new, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert T_new.ndim == 4

    def test_log_norm_finite(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        _, log_norm = _hotrg_step_horizontal(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_bond_dim_truncation(self):
        """Up/down bond dims should be bounded by max_bond_dim after step."""
        T = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=3)
        # After horizontal step, up/down legs (axes 0,1) are compressed
        assert T_new.shape[0] <= 3
        assert T_new.shape[1] <= 3

    def test_output_normalized(self):
        """Output tensor should be normalized (max |entry| ≈ 1)."""
        T = np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float64)
        T_new, _ = _hotrg_step_horizontal(T, max_bond_dim=4)
        arr = np.array(T_new)
        max_val = np.max(np.abs(arr))
        assert np.isclose(max_val, 1.0, atol=0.05), f"Expected ~1, got {max_val}"


class TestHOTRGStepVertical:
    def test_output_shape(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        T_new, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert T_new.ndim == 4

    def test_log_norm_finite(self):
        T = np.random.default_rng(0).random((2, 2, 2, 2)).astype(np.float64)
        _, log_norm = _hotrg_step_vertical(T, max_bond_dim=3)
        assert np.isfinite(float(log_norm))

    def test_bond_dim_truncation(self):
        """Left/right bond dims should be bounded by max_bond_dim."""
        T = np.random.default_rng(0).random((4, 4, 4, 4)).astype(np.float64)
        T_new, _ = _hotrg_step_vertical(T, max_bond_dim=3)
        assert T_new.shape[2] <= 3
        assert T_new.shape[3] <= 3

    def test_output_normalized(self):
        T = np.random.default_rng(42).random((2, 2, 2, 2)).astype(np.float64)
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

    def test_high_temp_free_energy(self):
        """At high temperature (beta=0.2), HOTRG chi=16 should be within 0.5%."""
        beta = 0.2
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.005, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_mid_temp_free_energy(self):
        """At beta=0.3, HOTRG chi=16 should be within 1%."""
        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.01, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_near_critical_free_energy(self):
        """Near critical point (beta=0.44), HOTRG chi=16 should be within 2%."""
        beta = 0.44
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.02, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
        )

    def test_low_temp_free_energy(self):
        """At low temperature (beta=0.6), HOTRG chi=16 should be within 0.5%."""
        beta = 0.6
        tensor = compute_ising_tensor(beta=beta)
        config = HOTRGConfig(max_bond_dim=16, num_steps=20)
        log_z_per_n = hotrg(tensor, config)
        hotrg_free_energy = float(-log_z_per_n / beta)
        exact_free_energy = ising_free_energy_exact(beta)
        relative_error = abs(hotrg_free_energy - exact_free_energy) / abs(
            exact_free_energy
        )
        assert relative_error < 0.005, (
            f"HOTRG free energy {hotrg_free_energy:.6f} too far from "
            f"exact {exact_free_energy:.6f} (rel err={relative_error:.4f})"
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
        from tenax.algorithms.trg import TRGConfig, trg

        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)

        result_hotrg = float(hotrg(tensor, HOTRGConfig(max_bond_dim=8, num_steps=6)))
        result_trg = float(trg(tensor, TRGConfig(max_bond_dim=8, num_steps=6)))

        assert (
            np.sign(result_hotrg) == np.sign(result_trg) or abs(result_hotrg) < 0.01
        ), f"HOTRG and TRG have different signs: {result_hotrg:.4f} vs {result_trg:.4f}"

    def test_hotrg_more_accurate_than_trg(self):
        """HOTRG should achieve better accuracy than TRG at the same chi."""
        from tenax.algorithms.trg import TRGConfig, trg

        beta = 0.3
        tensor = compute_ising_tensor(beta=beta)
        exact = ising_free_energy_exact(beta)

        f_trg = -float(trg(tensor, TRGConfig(max_bond_dim=8, num_steps=10))) / beta
        f_hotrg = (
            -float(hotrg(tensor, HOTRGConfig(max_bond_dim=8, num_steps=10))) / beta
        )

        err_trg = abs(f_trg - exact)
        err_hotrg = abs(f_hotrg - exact)
        assert err_hotrg <= err_trg + 1e-6, (
            f"HOTRG should be at least as accurate as TRG: "
            f"err_trg={err_trg:.6f}, err_hotrg={err_hotrg:.6f}"
        )
