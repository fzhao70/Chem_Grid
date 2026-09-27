"""Sparse regridding weights and their application.

Weights are built once, on the host, in numpy (:mod:`.bilinear`,
:mod:`.conservative`), and applied as a gather plus a segment sum, which is
``jax.jit``-, ``jax.grad``- and ``jax.vmap``-compatible. :class:`RegridWeights`
is a pytree, so it can be passed into a jitted function as an ordinary
argument.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

__all__ = ["RegridWeights"]


@dataclass(frozen=True)
class RegridWeights:
    """``dst[r] = sum_k weight[k] * src[col[k]]`` over entries with ``row[k] == r``.

    Attributes
    ----------
    row, col : ``(nnz,)`` int
        Flat (C-order) destination and source cell indices.
    weight : ``(nnz,)`` float
    covered : ``dst_shape`` float
        Fraction of each destination cell (conservative) or point (bilinear:
        0 or 1) that the source grid covers. Cells with ``covered == 0`` get
        *fill_value* from :meth:`__call__`.
    src_shape, dst_shape :
        Horizontal shapes, static.
    """

    row: np.ndarray
    col: np.ndarray
    weight: np.ndarray
    covered: np.ndarray
    src_shape: tuple[int, ...]
    dst_shape: tuple[int, ...]

    @property
    def n_src(self) -> int:
        return int(np.prod(self.src_shape))

    @property
    def n_dst(self) -> int:
        return int(np.prod(self.dst_shape))

    def __call__(self, field, fill_value: float = float("nan")):
        """Regrid *field* of shape ``src_shape + extra`` to ``dst_shape + extra``.

        The horizontal axes come first, matching the ``(nx, ny, nz, ...)``
        layout used across packages; trailing axes (levels, species, time)
        are carried through unchanged.
        """
        field = jnp.asarray(field)
        nsd = len(self.src_shape)
        if field.shape[:nsd] != tuple(self.src_shape):
            raise ValueError(
                f"field's leading axes {field.shape[:nsd]} do not match src_shape {self.src_shape}"
            )
        extra = field.shape[nsd:]
        flat = field.reshape((self.n_src, -1))
        dtype = flat.dtype if jnp.issubdtype(flat.dtype, jnp.floating) else jnp.float64
        w = jnp.asarray(self.weight, dtype)
        contrib = flat[jnp.asarray(self.col)].astype(dtype) * w[:, None]
        out = jax.ops.segment_sum(contrib, jnp.asarray(self.row), num_segments=self.n_dst)
        valid = jnp.asarray(self.covered).reshape(-1) > 0
        out = jnp.where(valid[:, None], out, jnp.asarray(fill_value, dtype))
        return out.reshape(tuple(self.dst_shape) + extra)

    def to_dense(self) -> np.ndarray:
        """The ``(n_dst, n_src)`` weight matrix; for tests and small grids only."""
        m = np.zeros((self.n_dst, self.n_src))
        np.add.at(m, (self.row, self.col), self.weight)
        return m


jax.tree_util.register_dataclass(
    RegridWeights,
    data_fields=["row", "col", "weight", "covered"],
    meta_fields=["src_shape", "dst_shape"],
)
