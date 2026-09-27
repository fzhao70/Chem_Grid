"""Static model-grid geometry shared by every package.

:class:`Grid` holds only what every operator agrees on: the domain dimensions,
REAM's scalar grid spacing, and -- only when the caller assigns them -- the
horizontal cell-centre and cell-edge coordinates. Anything one operator alone
needs (transport's ``topk`` boundary depth, its ``g_p2m`` gravity constant,
...) belongs in that package's own options, not here -- a shared object that
carries one module's knob is how a default silently mismatches another
module's arrays.

Fields that change in time or are differentiated through (map-scale factor,
sigma levels, ``ptop``, met fields) are deliberately not part of
:class:`Grid`: they have to be traced arrays, while :class:`Grid` has to stay
hashable so it can be a ``jax.jit`` static argument. The optional
coordinates are fixed geometry, so they are stored as read-only numpy copies
and take part in equality and hashing.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from chem_grid.projection import LambertConformal

__all__ = ["Grid"]

_COORD_FIELDS = ("lon", "lat", "lon_edges", "lat_edges")


def _frozen_copy(a, name: str, shape: tuple[int, int]) -> np.ndarray:
    a = np.array(a, dtype=np.float64, copy=True)
    if a.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    a.setflags(write=False)
    return a


@dataclass(frozen=True, eq=False)
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
    lon, lat : ``(nx, ny)``, optional
        Cell-centre longitude/latitude in degrees (WRF's ``XLONG``/``XLAT``).
        Given together or not at all.
    lon_edges, lat_edges : ``(nx+1, ny+1)``, optional
        Cell-corner longitude/latitude in degrees: cell ``(i, j)`` has corners
        ``(i, j)``, ``(i+1, j)``, ``(i+1, j+1)``, ``(i, j+1)``. The layout
        :func:`chem_grid.regrid.conservative_weights` takes. Given together or
        not at all, independently of ``lon``/``lat``.
    projection : :class:`~chem_grid.LambertConformal`, optional
        WRF's ``CEN_LAT``/``CEN_LON``/``TRUELAT1``/``TRUELAT2``/``STAND_LON``
        (REAM's ``phic``/``xlonc``/``truelat01``/``truelat02``), with
        ``cen`` at the centre of *this* grid. :meth:`from_projection` builds a
        grid with coordinates computed from it.

    Coordinates and projection are ``None`` unless assigned. Assigned
    coordinates are stored as read-only float64 copies, and two grids are
    equal only if their coordinates and projections are equal too.

    Each package still checks the minimum size it can handle (transport needs
    at least 3 cells per axis, for example); this class only rejects grids
    that are degenerate for everyone.
    """

    nx: int
    ny: int
    nz: int
    dx: float
    lon: np.ndarray | None = field(default=None, repr=False)
    lat: np.ndarray | None = field(default=None, repr=False)
    lon_edges: np.ndarray | None = field(default=None, repr=False)
    lat_edges: np.ndarray | None = field(default=None, repr=False)
    projection: LambertConformal | None = None

    def __post_init__(self):
        if min(self.nx, self.ny, self.nz) < 1:
            raise ValueError(f"grid dimensions must be >= 1, got {self.shape}")
        if self.dx <= 0:
            raise ValueError(f"dx must be positive, got {self.dx}")
        for a, b, shape in (
            ("lon", "lat", (self.nx, self.ny)),
            ("lon_edges", "lat_edges", (self.nx + 1, self.ny + 1)),
        ):
            va, vb = getattr(self, a), getattr(self, b)
            if (va is None) != (vb is None):
                raise ValueError(f"{a} and {b} must be given together")
            if va is None:
                continue
            object.__setattr__(self, a, _frozen_copy(va, a, shape))
            object.__setattr__(self, b, _frozen_copy(vb, b, shape))
            if np.any(np.abs(getattr(self, b)) > 90.0):
                raise ValueError(f"{b} must lie within [-90, 90]")
        if self.projection is not None and not isinstance(self.projection, LambertConformal):
            raise TypeError(f"projection must be a LambertConformal, got {type(self.projection)}")
        object.__setattr__(self, "_key", self._make_key())

    def _make_key(self):
        digest = hashlib.sha1()
        for name in _COORD_FIELDS:
            a = getattr(self, name)
            digest.update(name.encode())
            digest.update(b"-" if a is None else a.tobytes())
        return (self.nx, self.ny, self.nz, self.dx, self.projection, digest.hexdigest())

    def __eq__(self, other):
        if not isinstance(other, Grid):
            return NotImplemented
        if self._key != other._key:
            return False
        return all(
            (getattr(self, n) is None and getattr(other, n) is None)
            or np.array_equal(getattr(self, n), getattr(other, n))
            for n in _COORD_FIELDS
        )

    def __hash__(self):
        return hash(self._key)

    def __repr__(self):
        extra = [n for n in ("lon", "lon_edges") if getattr(self, n) is not None]
        coords = f", coords={'+'.join(extra)}" if extra else ""
        proj = f", projection={self.projection}" if self.projection is not None else ""
        return f"Grid(nx={self.nx}, ny={self.ny}, nz={self.nz}, dx={self.dx}{coords}{proj})"

    @classmethod
    def from_projection(
        cls,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        projection: LambertConformal,
        *,
        centers: bool = True,
        edges: bool = True,
    ) -> Grid:
        """A grid centred on ``projection.cen`` with coordinates computed from it.

        ``centers``/``edges`` choose which of ``lon``/``lat`` and
        ``lon_edges``/``lat_edges`` to fill. For the China 36 km domain the
        computed centres match WRF's ``XLAT``/``XLONG`` to float32 precision.
        """
        kw = {}
        if centers:
            kw["lon"], kw["lat"] = projection.cell_centers(nx, ny, dx)
        if edges:
            kw["lon_edges"], kw["lat_edges"] = projection.cell_corners(nx, ny, dx)
        return cls(nx=nx, ny=ny, nz=nz, dx=dx, projection=projection, **kw)

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def horizontal_shape(self) -> tuple[int, int]:
        return (self.nx, self.ny)

    @property
    def has_centers(self) -> bool:
        """True if ``lon``/``lat`` were assigned."""
        return self.lon is not None

    @property
    def has_edges(self) -> bool:
        """True if ``lon_edges``/``lat_edges`` were assigned."""
        return self.lon_edges is not None
