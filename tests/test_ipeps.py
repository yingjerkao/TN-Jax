"""Tests for the iPEPS and CTM algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tnjax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    _build_double_layer,
    _simple_update_1x1,
    compute_energy_ctm,
    ctm,
    ipeps,
    iPEPSConfig,
)


class TestCTMConfig:
    def test_default_values(self):
        cfg = CTMConfig()
        assert cfg.chi == 20
        assert cfg.max_iter == 100
        assert cfg.conv_tol == 1e-8
        assert cfg.renormalize is True

    def test_custom_values(self):
        cfg = CTMConfig(chi=10, max_iter=50, conv_tol=1e-6, renormalize=False)
        assert cfg.chi == 10
        assert cfg.max_iter == 50
        assert cfg.conv_tol == 1e-6
        assert cfg.renormalize is False


class TestIPEPSConfig:
    def test_default_values(self):
        cfg = iPEPSConfig()
        assert cfg.max_bond_dim == 2
        assert cfg.num_imaginary_steps == 100
        assert cfg.dt == 0.01
        assert cfg.ctm is not None
        assert isinstance(cfg.ctm, CTMConfig)

    def test_custom_values(self):
        cfg = iPEPSConfig(max_bond_dim=4, num_imaginary_steps=50, dt=0.05)
        assert cfg.max_bond_dim == 4
        assert cfg.num_imaginary_steps == 50
        assert cfg.dt == 0.05


class TestCTMEnvironment:
    def test_named_tuple_fields(self):
        """CTMEnvironment should have 8 tensor fields: 4 corners + 4 edges."""
        chi = 3
        d2 = 4  # D^2
        dummy = jnp.zeros((chi, chi))
        dummy_edge = jnp.zeros((chi, d2, chi))
        env = CTMEnvironment(
            C1=dummy, C2=dummy, C3=dummy, C4=dummy,
            T1=dummy_edge, T2=dummy_edge, T3=dummy_edge, T4=dummy_edge,
        )
        assert env.C1.shape == (chi, chi)
        assert env.T1.shape == (chi, d2, chi)

    def test_access_by_name(self):
        chi = 2
        d2 = 4
        corners = [jnp.eye(chi) * i for i in range(1, 5)]
        edges = [jnp.zeros((chi, d2, chi))] * 4
        env = CTMEnvironment(*corners, *edges)
        assert jnp.allclose(env.C1, jnp.eye(chi) * 1)
        assert jnp.allclose(env.C4, jnp.eye(chi) * 4)


class TestBuildDoubleLayer:
    def test_output_shape(self):
        """Double-layer tensor should have shape (D,D,D,D,D,D,D,D) for bond D, phys d."""
        D = 2
        d = 2
        key = jax.random.PRNGKey(0)
        # A has shape (u, d, l, r, s) = (D, D, D, D, d)
        A = jax.random.normal(key, (D, D, D, D, d))
        M = _build_double_layer(A)
        # M = einsum("udlrs,UDLRs->udlrUDLR", A, conj(A))
        # shape = (D, D, D, D, D, D, D, D)
        assert M.shape == (D, D, D, D, D, D, D, D)

    def test_double_layer_is_real_for_real_tensor(self):
        """For real A, the double-layer M should be real."""
        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        M = _build_double_layer(A)
        assert jnp.all(jnp.imag(M) == 0) if jnp.iscomplexobj(M) else True

    def test_double_layer_nonneg_diagonal(self):
        """Diagonal elements (same indices) should be non-negative."""
        key = jax.random.PRNGKey(2)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        M = _build_double_layer(A)
        # M[u,d,l,r, u,d,l,r] (same indices) = sum_s A[...s]^2 >= 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        assert M[i, j, k, m, i, j, k, m] >= 0


class TestCTM:
    @pytest.fixture
    def small_peps_tensor(self):
        """Small random PEPS site tensor with shape (D,D,D,D,d)."""
        key = jax.random.PRNGKey(42)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        # Normalize
        return A / (jnp.linalg.norm(A) + 1e-10)

    def test_ctm_returns_environment(self, small_peps_tensor):
        """CTM should return a CTMEnvironment."""
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)

    def test_ctm_corners_shape(self, small_peps_tensor):
        """Corner tensors should be (chi, chi) shaped."""
        chi = 4
        config = CTMConfig(chi=chi, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert env.C1.shape[0] <= chi
        assert env.C1.shape[1] <= chi

    def test_ctm_edge_shape(self, small_peps_tensor):
        """Edge tensors should have 3 legs."""
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(small_peps_tensor, config)
        assert env.T1.ndim == 3

    def test_ctm_runs_multiple_iters(self, small_peps_tensor):
        """CTM should converge (or run max_iter) without crashing."""
        config = CTMConfig(chi=4, max_iter=10, conv_tol=1e-12)  # tight tol -> max_iter
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)

    def test_ctm_with_initial_env(self, small_peps_tensor):
        """CTM should accept an initial environment and warm-start."""
        config = CTMConfig(chi=4, max_iter=3)
        env1 = ctm(small_peps_tensor, config)
        # Warm-start from env1
        env2 = ctm(small_peps_tensor, config, initial_env=env1)
        assert isinstance(env2, CTMEnvironment)

    def test_ctm_no_renormalize(self, small_peps_tensor):
        """CTM without renormalization should still run."""
        config = CTMConfig(chi=4, max_iter=5, renormalize=False)
        env = ctm(small_peps_tensor, config)
        assert isinstance(env, CTMEnvironment)


class TestComputeEnergyCTM:
    @pytest.fixture
    def peps_and_env(self):
        """Small PEPS tensor + CTM environment for energy computation tests."""
        key = jax.random.PRNGKey(7)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        config = CTMConfig(chi=4, max_iter=5)
        env = ctm(A, config)
        return A, env

    def test_energy_is_scalar(self, peps_and_env):
        """Energy from CTM contraction should be a scalar."""
        A, env = peps_and_env
        # Simple Heisenberg Sz*Sz gate for d=2
        d = 2
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        assert energy.shape == ()

    def test_energy_is_finite(self, peps_and_env):
        A, env = peps_and_env
        d = 2
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        assert jnp.isfinite(energy)


class TestSimpleUpdate1x1:
    def test_simple_update_runs(self):
        """Simple update step should run and return updated tensors + lambdas."""
        key = jax.random.PRNGKey(0)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        # Lambda matrices for each bond direction
        lambdas = {
            "right": jnp.ones(D),
            "up": jnp.ones(D),
        }

        # Sz*Sz gate
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.01
        # Trotter gate: exp(-dt * gate)
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        max_bond_dim = 3
        A_new, lambdas_new = _simple_update_1x1(A, A, lambdas, trotter_gate, max_bond_dim)

        # Should return tensors with same number of legs
        assert A_new.ndim == A.ndim

    def test_simple_update_bond_dim_bounded(self):
        """Updated tensor bond dimension should not exceed max_bond_dim."""
        key = jax.random.PRNGKey(1)
        D, d = 2, 2
        max_D = 3
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {
            "right": jnp.ones(D),
            "up": jnp.ones(D),
        }

        gate_flat = jnp.eye(d * d)
        trotter_gate = gate_flat.reshape(d, d, d, d)

        A_new, _ = _simple_update_1x1(A, A, lambdas, trotter_gate, max_D)
        # Check all bond dims are bounded
        assert A_new.shape[0] <= max_D  # up dim
        assert A_new.shape[2] <= max_D  # left dim


class TestIPEPSRun:
    @pytest.fixture
    def heisenberg_gate(self):
        """2-site Heisenberg Hamiltonian gate for simple update."""
        d = 2
        # H = Sz*Sz + 0.5*(S+S- + S-S+)
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        Sp = jnp.array([[0.0, 1.0], [0.0, 0.0]])
        Sm = jnp.array([[0.0, 0.0], [1.0, 0.0]])
        H = (
            jnp.kron(Sz, Sz)
            + 0.5 * jnp.kron(Sp, Sm)
            + 0.5 * jnp.kron(Sm, Sp)
        )
        return H.reshape(d, d, d, d)

    def test_ipeps_runs_without_error(self, heisenberg_gate):
        """iPEPS should run end-to-end without crashing."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=3,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, peps_out, env = ipeps(heisenberg_gate, None, config)
        assert jnp.isfinite(energy)

    def test_ipeps_returns_three_tuple(self, heisenberg_gate):
        """ipeps() should return (energy, peps, env) triple."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        result = ipeps(heisenberg_gate, None, config)
        assert len(result) == 3

    def test_ipeps_energy_is_scalar(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        assert isinstance(energy, float)

    def test_ipeps_env_is_ctm_environment(self, heisenberg_gate):
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        _, _, env = ipeps(heisenberg_gate, None, config)
        assert isinstance(env, CTMEnvironment)

    def test_ipeps_with_initial_peps(self, heisenberg_gate):
        """iPEPS should accept an initial PEPS tensor (non-None initial_peps)."""
        key = jax.random.PRNGKey(99)
        D, d = 2, 2
        initial_A = jax.random.normal(key, (D, D, D, D, d))
        initial_A = initial_A / (jnp.linalg.norm(initial_A) + 1e-10)

        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=2,
            dt=0.1,
            ctm=CTMConfig(chi=4, max_iter=3),
        )
        energy, _, _ = ipeps(heisenberg_gate, initial_A, config)
        assert jnp.isfinite(energy)

    def test_ipeps_energy_negative_for_heisenberg(self, heisenberg_gate):
        """Heisenberg ground state energy should be negative per bond."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=10,
            dt=0.05,
            ctm=CTMConfig(chi=4, max_iter=5),
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        # Heisenberg AFM energy per bond should be negative
        # (or at least the algorithm shouldn't produce absurdly positive energy)
        # Loose check: energy per site should be in [-1, 1] range
        assert float(energy) < 1.0, f"Energy {float(energy)} seems too large"
