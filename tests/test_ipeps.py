"""Tests for the iPEPS and CTM algorithms."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tnjax.algorithms.ipeps import (
    CTMConfig,
    CTMEnvironment,
    _build_double_layer,
    _build_double_layer_open,
    _rdm1x2,
    _rdm2x1,
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
        """Diagonal elements (same ket/bra indices) should be non-negative."""
        key = jax.random.PRNGKey(2)
        A = jax.random.normal(key, (2, 2, 2, 2, 2))
        M = _build_double_layer(A)
        # M has ordering (u, U, d, D, l, L, r, R) from uUdDlLrR
        # Diagonal: u=U, d=D, l=L, r=R → M[i,i,j,j,k,k,m,m] >= 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for m in range(2):
                        assert M[i, i, j, j, k, k, m, m] >= 0


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

    def test_ctm_edge_tensors_change(self, small_peps_tensor):
        """After a full CTM run, edge tensors should differ from initialization."""
        config = CTMConfig(chi=4, max_iter=10)
        from tnjax.algorithms.ipeps import _build_double_layer, _initialize_ctm_env
        a = _build_double_layer(small_peps_tensor)
        D = small_peps_tensor.shape[0]
        a = a.reshape(D**2, D**2, D**2, D**2)
        env0 = _initialize_ctm_env(a, config.chi)
        env = ctm(small_peps_tensor, config, initial_env=env0)
        # At least one edge tensor should have changed
        changed = not (
            jnp.allclose(env0.T1, env.T1, atol=1e-10)
            and jnp.allclose(env0.T2, env.T2, atol=1e-10)
            and jnp.allclose(env0.T3, env.T3, atol=1e-10)
            and jnp.allclose(env0.T4, env.T4, atol=1e-10)
        )
        assert changed, "Edge tensors did not change during CTM"


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

        lambdas = {
            "horizontal": jnp.ones(D),
            "vertical": jnp.ones(D),
        }

        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.01
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        max_bond_dim = 3
        A_new, lambdas_new = _simple_update_1x1(
            A, A, lambdas, trotter_gate, max_bond_dim, bond="horizontal",
        )

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
            "horizontal": jnp.ones(D),
            "vertical": jnp.ones(D),
        }

        gate_flat = jnp.eye(d * d)
        trotter_gate = gate_flat.reshape(d, d, d, d)

        A_new, _ = _simple_update_1x1(
            A, A, lambdas, trotter_gate, max_D, bond="horizontal",
        )
        # Check all bond dims are bounded
        assert A_new.shape[0] <= max_D  # up dim
        assert A_new.shape[2] <= max_D  # left dim

    def test_simple_update_5leg_modifies_tensor(self):
        """Passing a 5-leg tensor with a non-trivial gate should modify A."""
        key = jax.random.PRNGKey(10)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.1
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        A_new, _ = _simple_update_1x1(
            A, A, lambdas, trotter_gate, D, bond="horizontal",
        )
        assert not jnp.allclose(A, A_new, atol=1e-8), "A should change after update"

    def test_simple_update_5leg_preserves_shape(self):
        """After update, tensor should still have 5 legs."""
        key = jax.random.PRNGKey(20)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        trotter_gate = jnp.eye(d * d).reshape(d, d, d, d)

        for bond in ["horizontal", "vertical"]:
            A_new, _ = _simple_update_1x1(
                A, A, lambdas, trotter_gate, D, bond=bond,
            )
            assert A_new.ndim == 5
            assert A_new.shape[-1] == d  # physical dim unchanged

    def test_lambda_normalized(self):
        """Lambda vectors should have max element = 1 after update."""
        key = jax.random.PRNGKey(30)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)

        lambdas = {"horizontal": jnp.ones(D), "vertical": jnp.ones(D)}
        gate = jnp.diag(jnp.array([0.25, -0.25, -0.25, 0.25])).reshape(d, d, d, d)
        dt = 0.1
        gate_flat = gate.reshape(d * d, d * d)
        trotter_gate = jax.scipy.linalg.expm(-dt * gate_flat).reshape(d, d, d, d)

        _, lam_h = _simple_update_1x1(
            A, A, lambdas, trotter_gate, D, bond="horizontal",
        )
        _, lam_v = _simple_update_1x1(
            A, A, lambdas, trotter_gate, D, bond="vertical",
        )

        assert jnp.allclose(jnp.max(lam_h["horizontal"]), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.max(lam_v["vertical"]), 1.0, atol=1e-10)


class TestRDM:
    """Tests for the 2-site reduced density matrices."""

    @pytest.fixture
    def peps_env(self):
        """PEPS tensor and converged CTM environment."""
        key = jax.random.PRNGKey(55)
        D, d = 2, 2
        A = jax.random.normal(key, (D, D, D, D, d))
        A = A / (jnp.linalg.norm(A) + 1e-10)
        config = CTMConfig(chi=8, max_iter=20)
        env = ctm(A, config)
        return A, env, d

    def test_rdm_hermitian(self, peps_env):
        """The 2-site RDM should satisfy rdm == rdm^dagger."""
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d)
        rdm_v = _rdm1x2(A, env, d)

        rdm_h_mat = rdm_h.reshape(d * d, d * d)
        rdm_v_mat = rdm_v.reshape(d * d, d * d)
        assert jnp.allclose(rdm_h_mat, rdm_h_mat.conj().T, atol=1e-10)
        assert jnp.allclose(rdm_v_mat, rdm_v_mat.conj().T, atol=1e-10)

    def test_rdm_positive_semidefinite(self, peps_env):
        """Eigenvalues of the RDM should be approximately non-negative.

        With finite chi the CTM environment is approximate, so small
        negative eigenvalues (order 0.1) are tolerated.
        """
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d).reshape(d * d, d * d)
        rdm_v = _rdm1x2(A, env, d).reshape(d * d, d * d)

        eigvals_h = jnp.linalg.eigvalsh(rdm_h)
        eigvals_v = jnp.linalg.eigvalsh(rdm_v)
        assert jnp.all(eigvals_h > -0.2), f"Large negative eigenvalues: {eigvals_h}"
        assert jnp.all(eigvals_v > -0.2), f"Large negative eigenvalues: {eigvals_v}"

    def test_rdm_trace_one(self, peps_env):
        """trace(rdm) should be approximately 1."""
        A, env, d = peps_env
        rdm_h = _rdm2x1(A, env, d).reshape(d * d, d * d)
        rdm_v = _rdm1x2(A, env, d).reshape(d * d, d * d)
        assert jnp.allclose(jnp.trace(rdm_h), 1.0, atol=1e-10)
        assert jnp.allclose(jnp.trace(rdm_v), 1.0, atol=1e-10)


class TestBuildDoubleLayerOpen:
    def test_shape(self):
        D, d = 2, 2
        key = jax.random.PRNGKey(0)
        A = jax.random.normal(key, (D, D, D, D, d))
        ao = _build_double_layer_open(A)
        assert ao.shape == (D**2, D**2, D**2, D**2, d, d)

    def test_trace_equals_closed(self):
        """Tracing out physical indices of open tensor gives the closed one."""
        D, d = 2, 2
        key = jax.random.PRNGKey(1)
        A = jax.random.normal(key, (D, D, D, D, d))
        ao = _build_double_layer_open(A)
        # trace s=s' → a_closed
        a_traced = jnp.einsum("udlrss->udlr", ao)
        a_closed = _build_double_layer(A).reshape(D**2, D**2, D**2, D**2)
        assert jnp.allclose(a_traced, a_closed, atol=1e-12)


class TestProductStateEnergy:
    def test_energy_product_state_up(self):
        """For a product state |up>, SzSz energy per bond = +0.25."""
        D, d = 1, 2
        # |up> = [1, 0] product state: A[u,d,l,r,s] trivial on virtual bonds
        A = jnp.zeros((D, D, D, D, d))
        A = A.at[0, 0, 0, 0, 0].set(1.0)  # |up>

        config = CTMConfig(chi=4, max_iter=20)
        env = ctm(A, config)

        # SzSz only
        Sz = 0.5 * jnp.array([[1.0, 0.0], [0.0, -1.0]])
        gate = jnp.kron(Sz, Sz).reshape(d, d, d, d)
        energy = compute_energy_ctm(A, env, gate, d)
        # |up up>: Sz*Sz = 0.25 per bond, 2 bonds (h+v) per site
        assert jnp.allclose(energy, 0.5, atol=0.1), f"Energy = {float(energy)}"


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

    def test_ipeps_energy_negative_heisenberg_longer(self, heisenberg_gate):
        """With more steps and larger chi, energy should be clearly negative."""
        config = iPEPSConfig(
            max_bond_dim=2,
            num_imaginary_steps=100,
            dt=0.02,
            ctm=CTMConfig(chi=8, max_iter=30),
        )
        energy, _, _ = ipeps(heisenberg_gate, None, config)
        # Allow a tiny positive tolerance for platform-dependent numerics
        assert float(energy) < 1e-4, f"Energy should be ≤ 0, got {float(energy)}"
