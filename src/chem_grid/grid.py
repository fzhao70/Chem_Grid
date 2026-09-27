"""Static model-grid geometry shared by every package.

:class:`Grid` holds only what every operator agrees on: the domain dimensions
and REAM's scalar grid spacing. Anything one operator alone needs (transport's
``topk`` boundary depth, its ``g_p2m`` gravity constant, ...) belongs in that
package's own options, not here -- a shared object that carries one module's
knob is how a default silently mismatches another module's arrays.

Array-valued geometry (map-scale factor, cell-centre lat/lon, sigma levels,
``ptop``) is deliberately not part of :class:`Grid`: it has to be a traced
array, while :class:`Grid` has to stay hashable so it can be a ``jax.jit``
static argument.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Grid"]


@dataclass(frozen=True)
class Grid:
    """Static grid geometry, usable as a ``jax.jit`` static argument.

    Attributes
    ----------
    nx, ny, nz :
        Domain dimensions. Arrays are laid out ``(nx, ny, nz, ...)`` -- ``x``
        first -- and the vertical axis is in model order (``k=0`` is the model
        top) wherever it crosses a package boundary; see
        :mod:`chem_grid.vertical`.
    dx :
        Scalar grid spacing in metres. REAM's ``DX`` is a single constant, not
        an array; horizontal distance is ``dx/xmsf`` per column.

    Each package still checks the minimum size it can handle (transport needs
    at least 3 cells per axis, for example); this class only rejects grids
    that are degenerate for everyone.
    """

    nx: int
    ny: int
    nz: int
    dx: float

    def __post_init__(self):
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError(f"grid dimensions must be >= 1, got {self.shape}")
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)
