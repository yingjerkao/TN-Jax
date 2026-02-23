"""Tests for the symmetry module."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from tnjax.core.symmetry import (
    BaseNonAbelianSymmetry,
    BaseSymmetry,
    U1Symmetry,
    ZnSymmetry,
)


class TestU1Symmetry:
    def test_fuse_addition(self):
        sym = U1Symmetry()
        a = np.array([0, 1, -1], dtype=np.int32)
        b = np.array([1, -1, 2], dtype=np.int32)
        np.testing.assert_array_equal(sym.fuse(a, b), a + b)

    def test_dual_negation(self):
        sym = U1Symmetry()
        a = np.array([0, 1, -2, 5], dtype=np.int32)
        np.testing.assert_array_equal(sym.dual(a), -a)

    def test_identity_is_zero(self):
        assert U1Symmetry().identity() == 0

    def test_n_values_is_none(self):
        assert U1Symmetry().n_values() is None

    def test_fuse_associativity(self):
        sym = U1Symmetry()
        a = np.array([1, -2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        c = np.array([-1, 2], dtype=np.int32)
        left = sym.fuse(sym.fuse(a, b), c)
        right = sym.fuse(a, sym.fuse(b, c))
        np.testing.assert_array_equal(left, right)

    def test_fuse_identity(self):
        sym = U1Symmetry()
        a = np.array([-2, 0, 3], dtype=np.int32)
        zero = np.array([0, 0, 0], dtype=np.int32)
        np.testing.assert_array_equal(sym.fuse(a, zero), a)

    def test_dual_is_inverse(self):
        sym = U1Symmetry()
        a = np.array([1, -2, 3], dtype=np.int32)
        result = sym.fuse(a, sym.dual(a))
        np.testing.assert_array_equal(result, np.zeros_like(a))

    def test_fuse_many(self):
        sym = U1Symmetry()
        a = np.array([1, 2], dtype=np.int32)
        b = np.array([3, 4], dtype=np.int32)
        c = np.array([-4, -6], dtype=np.int32)
        result = sym.fuse_many([a, b, c])
        np.testing.assert_array_equal(result, a + b + c)

    def test_fuse_many_single(self):
        sym = U1Symmetry()
        a = np.array([1, 2, 3], dtype=np.int32)
        np.testing.assert_array_equal(sym.fuse_many([a]), a)

    def test_fuse_many_empty_raises(self):
        sym = U1Symmetry()
        with pytest.raises(ValueError):
            sym.fuse_many([])

    def test_equality(self):
        assert U1Symmetry() == U1Symmetry()
        assert not (U1Symmetry() == ZnSymmetry(2))

    def test_hashable(self):
        sym1 = U1Symmetry()
        sym2 = U1Symmetry()
        d = {sym1: "test"}
        assert d[sym2] == "test"

    def test_repr(self):
        assert repr(U1Symmetry()) == "U1Symmetry()"

    def test_is_conserved_simple(self):
        sym = U1Symmetry()
        # net = (+1)*1 + (-1)*1 = 0 = identity
        assert sym.is_conserved([np.int32(1), np.int32(1)], [1, -1])

    def test_is_conserved_fails(self):
        sym = U1Symmetry()
        assert not sym.is_conserved([np.int32(1), np.int32(2)], [1, 1])

    @given(
        hnp.arrays(np.int32, st.integers(1, 10)),
    )
    @settings(max_examples=100)
    def test_dual_inverse_property(self, a):
        """Property: fuse(a, dual(a)) == 0 for all a."""
        sym = U1Symmetry()
        result = sym.fuse(a, sym.dual(a))
        np.testing.assert_array_equal(result, np.zeros_like(result))

    @given(
        st.integers(1, 8).flatmap(
            lambda n: st.tuples(
                hnp.arrays(np.int32, n),
                hnp.arrays(np.int32, n),
            )
        )
    )
    @settings(max_examples=100)
    def test_commutativity(self, ab):
        """Property: fuse(a, b) == fuse(b, a) for abelian U(1)."""
        a, b = ab
        sym = U1Symmetry()
        np.testing.assert_array_equal(sym.fuse(a, b), sym.fuse(b, a))


class TestZnSymmetry:
    @pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7])
    def test_fuse_modular(self, n):
        sym = ZnSymmetry(n)
        a = np.arange(n, dtype=np.int32)
        b = np.arange(n, dtype=np.int32)
        result = sym.fuse(a, b)
        assert np.all(result >= 0)
        assert np.all(result < n)

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_dual_is_inverse(self, n):
        sym = ZnSymmetry(n)
        a = np.arange(n, dtype=np.int32)
        result = sym.fuse(a, sym.dual(a))
        np.testing.assert_array_equal(result, np.zeros(n, dtype=np.int32))

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_identity_is_zero(self, n):
        assert ZnSymmetry(n).identity() == 0

    @pytest.mark.parametrize("n", [2, 3, 4, 5])
    def test_n_values(self, n):
        assert ZnSymmetry(n).n_values() == n

    def test_invalid_n_one(self):
        with pytest.raises(ValueError, match="n must be >= 2"):
            ZnSymmetry(1)

    def test_invalid_n_zero(self):
        with pytest.raises(ValueError):
            ZnSymmetry(0)

    def test_invalid_n_negative(self):
        with pytest.raises(ValueError):
            ZnSymmetry(-3)

    @pytest.mark.parametrize("n", [2, 3])
    def test_equality(self, n):
        assert ZnSymmetry(n) == ZnSymmetry(n)
        assert ZnSymmetry(n) != ZnSymmetry(n + 1)
        assert ZnSymmetry(n) != U1Symmetry()

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_hashable(self, n):
        sym1 = ZnSymmetry(n)
        sym2 = ZnSymmetry(n)
        d = {sym1: n}
        assert d[sym2] == n

    def test_repr(self):
        assert repr(ZnSymmetry(3)) == "ZnSymmetry(3)"

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_associativity(self, n):
        sym = ZnSymmetry(n)
        a = np.array([1, 0, n - 1], dtype=np.int32)
        b = np.array([1, 1, 1], dtype=np.int32)
        c = np.array([0, n - 1, 1], dtype=np.int32)
        left = sym.fuse(sym.fuse(a, b), c)
        right = sym.fuse(a, sym.fuse(b, c))
        np.testing.assert_array_equal(left, right)

    def test_z2_parity(self):
        """Z2: only even/odd parity."""
        sym = ZnSymmetry(2)
        a = np.array([0, 1, 0, 1], dtype=np.int32)
        b = np.array([0, 0, 1, 1], dtype=np.int32)
        expected = np.array([0, 1, 1, 0], dtype=np.int32)  # XOR
        np.testing.assert_array_equal(sym.fuse(a, b), expected)

    @given(st.integers(min_value=2, max_value=10))
    @settings(max_examples=50)
    def test_closure(self, n):
        """Property: fuse result is always in [0, n)."""
        sym = ZnSymmetry(n)
        a = np.array([n - 1, 0, n // 2], dtype=np.int32)
        b = np.array([1, n - 1, n // 2], dtype=np.int32)
        result = sym.fuse(a, b)
        assert np.all(result >= 0)
        assert np.all(result < n)


class TestBaseNonAbelianSymmetry:
    def test_is_abstract(self):
        """BaseNonAbelianSymmetry should not be instantiable directly."""
        with pytest.raises(TypeError):
            BaseNonAbelianSymmetry()

    def test_stub_methods_required(self):
        """Subclass must implement all abstract methods."""
        class Incomplete(BaseNonAbelianSymmetry):
            def fuse(self, a, b): return a + b
            def dual(self, a): return -a
            def identity(self): return 0
            def n_values(self): return None
            # Missing: recoupling_coefficients, allowed_fusions

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass(self):
        """A fully implemented subclass should instantiate."""
        class ConcreteNonAbelian(BaseNonAbelianSymmetry):
            def fuse(self, a, b): return a + b
            def dual(self, a): return -a
            def identity(self): return 0
            def n_values(self): return None

            def recoupling_coefficients(self, j1, j2, j3):
                return np.ones((1,))

            def allowed_fusions(self, j1, j2):
                return list(range(abs(j1 - j2), j1 + j2 + 1, 2))

        sym = ConcreteNonAbelian()
        assert sym.identity() == 0
        assert sym.allowed_fusions(1, 1) == [0, 2]
