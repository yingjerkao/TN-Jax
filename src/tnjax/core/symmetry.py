"""Symmetry group definitions for symmetric tensor networks.

This module provides the mathematical foundation for symmetric tensors.
All symmetry classes operate on numpy integer arrays of charge values.
No JAX dependency — pure Python/numpy arithmetic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseSymmetry(ABC):
    """Abstract base for symmetry groups governing tensor index charges.

    A symmetry group defines how charges combine (fuse) when tensor legs
    are merged, and how charges transform when a leg's flow is reversed (dual).

    Concrete subclasses must implement fuse, dual, identity, and n_values.
    All concrete classes should implement __eq__ and __hash__ so they can
    serve as dict keys and be compared for compatibility checks.
    """

    @abstractmethod
    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        """Fuse two charge arrays element-wise under group multiplication.

        Args:
            charges_a: Integer charge array of shape (D,).
            charges_b: Integer charge array of shape (D,).

        Returns:
            Fused charge array of shape (D,).
        """

    @abstractmethod
    def dual(self, charges: np.ndarray) -> np.ndarray:
        """Return the group inverse (dual) of each charge.

        Used to flip a leg's flow direction while preserving compatibility.

        Args:
            charges: Integer charge array of shape (D,).

        Returns:
            Dual charge array of shape (D,).
        """

    @abstractmethod
    def identity(self) -> int:
        """Return the identity element (neutral charge, typically 0)."""

    @abstractmethod
    def n_values(self) -> int | None:
        """Return the number of distinct charge values, or None if infinite (U(1))."""

    def fuse_many(self, charge_list: list[np.ndarray]) -> np.ndarray:
        """Fuse a list of charge arrays left-to-right via repeated fuse().

        Args:
            charge_list: Non-empty list of integer charge arrays, all shape (D,).

        Returns:
            Fully fused charge array of shape (D,).
        """
        if not charge_list:
            raise ValueError("charge_list must be non-empty")
        result = charge_list[0]
        for c in charge_list[1:]:
            result = self.fuse(result, c)
        return result

    def is_conserved(
        self,
        charges_per_leg: list[np.ndarray],
        flows: list[int],
        target: int | None = None,
    ) -> bool:
        """Check if a single charge combination satisfies conservation.

        Args:
            charges_per_leg: List of scalar-or-array charge values per leg.
            flows: List of +1 (IN) or -1 (OUT) per leg.
            target: Required net charge; defaults to identity().

        Returns:
            True if the net charge equals target.
        """
        if target is None:
            target = self.identity()
        net = sum(int(f) * int(q) for f, q in zip(flows, charges_per_leg))
        # For modular groups we need to reduce the net charge
        n = self.n_values()
        if n is not None:
            net = net % n
        return net == target


class U1Symmetry(BaseSymmetry):
    """U(1) symmetry: integer charges, fusion by addition.

    Represents the continuous U(1) group (particle number conservation,
    total Sz conservation, etc.). Charges are unbounded integers.

    Example:
        >>> sym = U1Symmetry()
        >>> sym.fuse(np.array([0, 1, -1]), np.array([1, -1, 0]))
        array([1, 0, -1])
    """

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return charges_a + charges_b

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return -charges

    def identity(self) -> int:
        return 0

    def n_values(self) -> None:
        return None

    def __eq__(self, other: object) -> bool:
        return isinstance(other, U1Symmetry)

    def __hash__(self) -> int:
        return hash("U1Symmetry")

    def __repr__(self) -> str:
        return "U1Symmetry()"


class ZnSymmetry(BaseSymmetry):
    """Z_n symmetry: integer charges mod n, fusion by addition mod n.

    Represents the discrete cyclic group Z_n (e.g., Z_2 for fermion parity,
    Z_3 for three-state Potts model, etc.).

    Args:
        n: The order of the group. Must be >= 2.

    Example:
        >>> sym = ZnSymmetry(3)
        >>> sym.fuse(np.array([1, 2]), np.array([2, 2]))
        array([0, 1])
    """

    def __init__(self, n: int) -> None:
        if n < 2:
            raise ValueError(f"n must be >= 2, got {n}")
        self.n = n

    def fuse(self, charges_a: np.ndarray, charges_b: np.ndarray) -> np.ndarray:
        return (charges_a + charges_b) % self.n

    def dual(self, charges: np.ndarray) -> np.ndarray:
        return (-charges) % self.n

    def identity(self) -> int:
        return 0

    def n_values(self) -> int:
        return self.n

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ZnSymmetry) and self.n == other.n

    def __hash__(self) -> int:
        return hash(("ZnSymmetry", self.n))

    def __repr__(self) -> str:
        return f"ZnSymmetry({self.n})"


class BaseNonAbelianSymmetry(BaseSymmetry):
    """Stub base class for non-Abelian symmetries (e.g., SU(2)).

    Non-Abelian symmetries require Clebsch-Gordan / recoupling coefficients
    for tensor contractions, making them significantly more complex than
    Abelian symmetries. This base provides the interface contract for future
    concrete implementations.

    Note:
        Concrete subclasses must implement recoupling_coefficients and
        allowed_fusions in addition to the BaseSymmetry abstract methods.
        The fuse/dual/identity/n_values methods for non-Abelian groups
        operate on irrep labels (e.g., spin quantum numbers).
    """

    @abstractmethod
    def recoupling_coefficients(
        self,
        j1: int,
        j2: int,
        j3: int,
    ) -> np.ndarray:
        """Return Clebsch-Gordan or 6j-symbol coefficients for recoupling.

        These coefficients are needed when contracting non-Abelian symmetric
        tensors. The specific form depends on the group (CG coefficients for
        SU(2), etc.).

        Args:
            j1: First irrep label.
            j2: Second irrep label.
            j3: Output irrep label (must be in allowed_fusions(j1, j2)).

        Returns:
            Coefficient array for the (j1, j2) -> j3 channel.
        """

    @abstractmethod
    def allowed_fusions(self, j1: int, j2: int) -> list[int]:
        """Return list of irrep labels appearing in tensor product j1 x j2.

        For SU(2) with spin quantum numbers j1, j2, this returns
        ``[abs(j1 - j2), ..., j1 + j2]`` (triangular rule).

        Args:
            j1: First irrep label.
            j2: Second irrep label.

        Returns:
            Sorted list of irrep labels in the decomposition.
        """
