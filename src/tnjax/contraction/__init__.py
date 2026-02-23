"""Tensor contraction engine with label-based API."""

from tnjax.contraction.contractor import (
    contract,
    contract_with_subscripts,
    qr_decompose,
    truncated_svd,
)

__all__ = [
    "contract",
    "contract_with_subscripts",
    "truncated_svd",
    "qr_decompose",
]
