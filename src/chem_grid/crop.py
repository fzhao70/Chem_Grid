"""REAM's lateral crop of the met-driver grid onto the model grid.

REAM reads a larger WRF grid and drops a fixed number of cells off each side
(``constants.cpp:598-615``: ``IXB``/``JXB`` off the low side, ``IXBe``/``JXBe``
off the high side, e.g. the China 36 km configuration crops a 179x123 WRF grid
to 169x113 with all four set to 5). ``read_wrf.cpp``'s copy loops
(``WRF.PC(I-IXB-IST, J-JXB-JST, ...) = PRF(I, J, ...)``) are the same uniform
crop on axes 0/1 for every field, which also handles C-grid-staggered fields
correctly: their raw axis is already one longer, so the same crop widths give
a result exactly one longer too.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

from chem_grid.grid import Grid

__all__ = ["Crop"]

ArrayT = TypeVar("ArrayT")


@dataclass(frozen=True)
class Crop:
    """Cells dropped off each side of the ``(x, y)`` axes.

    Attributes
    ----------
    ixb, ixbe :
        Cells dropped off the low / high end of ``x`` (axis 0), REAM's
        ``IXB`` / ``IXBe`` (plus ``IST`` / ``IED`` when those are non-zero).
    jxb, jxbe :
        The same for ``y`` (axis 1), REAM's ``JXB`` / ``JXBe``.
    """

    ixb: int = 0
    ixbe: int = 0
    jxb: int = 0
    jxbe: int = 0

    def __post_init__(self):
        if min(self.ixb, self.ixbe, self.jxb, self.jxbe) < 0:
            raise ValueError(f"crop widths must be >= 0, got {self}")

    def shape(self, nx_full: int, ny_full: int) -> tuple[int, int]:
        """The cropped ``(nx, ny)`` of an unstaggered ``(nx_full, ny_full)`` field."""
        nx = nx_full - self.ixb - self.ixbe
        ny = ny_full - self.jxb - self.jxbe
        if nx < 1 or ny < 1:
            raise ValueError(f"{self} leaves no cells in a {nx_full}x{ny_full} grid")
        return nx, ny

    def grid(self, nx_full: int, ny_full: int, nz: int, dx: float) -> Grid:
        """The model :class:`~chem_grid.Grid` REAM builds from a full met grid."""
        nx, ny = self.shape(nx_full, ny_full)
        return Grid(nx=nx, ny=ny, nz=nz, dx=dx)

    def apply(self, a: ArrayT) -> ArrayT:
        """Crop *a*'s first two axes (``x``, ``y``).

        Arrays with fewer than two dimensions (1-D vertical-only fields such
        as ``znu``, scalars) are returned unchanged. Works on numpy and jax
        arrays alike, and inside ``jax.jit`` (the slice bounds are static).
        """
        if getattr(a, "ndim", 0) < 2:
            return a
        nx_full, ny_full = a.shape[0], a.shape[1]
        self.shape(nx_full, ny_full)
        return a[self.ixb : nx_full - self.ixbe, self.jxb : ny_full - self.jxbe, ...]
