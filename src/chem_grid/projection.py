"""WRF's Lambert conformal map projection (``MAP_PROJ = 1``).

WRF describes a Lambert conformal domain with the global attributes
``CEN_LAT``/``CEN_LON`` (the domain centre), ``TRUELAT1``/``TRUELAT2`` (the
standard parallels) and ``STAND_LON`` (the meridian parallel to the grid's
y axis). REAM carries the same numbers as ``phic``, ``xlonc``, ``truelat01``,
``truelat02`` (``constants.cpp``; the China 36 km domain is
``LambertConformal(cen_lat=37, cen_lon=107, truelat1=30, truelat2=45)``), with
``xlonc`` serving as both ``CEN_LON`` and ``STAND_LON``.

The formulas are the spherical Lambert conformal conic (Snyder 1987, eqs.
15-1 to 15-11) on WRF's sphere of radius 6370 km, with the cone constant
computed as in WRF's ``module_llxy`` (``sin(truelat1)`` when the two parallels
coincide). Grid cell ``(i, j)`` (0-based, x first) of an ``(nx, ny)`` mass grid
sits at projected ``((i - (nx-1)/2) * dx, (j - (ny-1)/2) * dx)`` from the
domain centre, which is where WPS places ``CEN_LAT``/``CEN_LON``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

__all__ = ["WRF_EARTH_RADIUS_M", "LambertConformal"]

WRF_EARTH_RADIUS_M = 6370000.0


@dataclass(frozen=True)
class LambertConformal:
    """WRF Lambert conformal projection parameters, in degrees.

    Attributes
    ----------
    cen_lat, cen_lon :
        Domain centre, WRF's ``CEN_LAT``/``CEN_LON`` (REAM's ``phic``/``xlonc``).
    truelat1, truelat2 :
        Standard parallels, WRF's ``TRUELAT1``/``TRUELAT2``.
    stand_lon :
        WRF's ``STAND_LON``; ``None`` (the default) means ``cen_lon``, which is
        what REAM assumes and what a WPS domain gets unless ``stand_lon`` was
        set separately in ``namelist.wps``.
    """

    cen_lat: float
    cen_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float | None = None

    def __post_init__(self):
        for name in ("cen_lat", "cen_lon", "truelat1", "truelat2"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if self.stand_lon is None:
            object.__setattr__(self, "stand_lon", self.cen_lon)
        else:
            object.__setattr__(self, "stand_lon", float(self.stand_lon))
        for name in ("cen_lat", "truelat1", "truelat2"):
            if not -90.0 < getattr(self, name) < 90.0:
                raise ValueError(f"{name} must lie strictly within (-90, 90), got {getattr(self, name)}")
        if self.truelat1 == 0.0 and self.truelat2 == 0.0:
            raise ValueError("truelat1 = truelat2 = 0 is a Mercator projection, not Lambert conformal")
        if self.truelat1 * self.truelat2 < 0.0:
            raise ValueError("truelat1 and truelat2 must be in the same hemisphere")

    @classmethod
    def from_wrf_attrs(cls, attrs) -> LambertConformal:
        """Build from a WRF file's global attributes (any mapping with WRF's names)."""
        proj = attrs.get("MAP_PROJ", 1)
        if int(np.asarray(proj).reshape(-1)[0]) != 1:
            raise ValueError(f"MAP_PROJ={proj} is not Lambert conformal (1)")

        def get(name):
            return float(np.asarray(attrs[name]).reshape(-1)[0])

        stand = get("STAND_LON") if "STAND_LON" in attrs else None
        return cls(get("CEN_LAT"), get("CEN_LON"), get("TRUELAT1"), get("TRUELAT2"), stand)

    # -- projection constants ------------------------------------------------

    @property
    def cone(self) -> float:
        """Cone constant ``n``, WRF's ``cone`` (negative in the southern hemisphere)."""
        t1, t2 = np.radians(self.truelat1), np.radians(self.truelat2)
        if abs(self.truelat1 - self.truelat2) > 0.1:
            n = np.log(np.cos(t1) / np.cos(t2)) / np.log(
                np.tan(np.pi / 4 + t2 / 2) / np.tan(np.pi / 4 + t1 / 2)
            )
        else:
            n = np.sin(t1)
        return float(n)

    def _rho(self, lat):
        n = self.cone
        t1 = np.radians(self.truelat1)
        f = np.cos(t1) * np.tan(np.pi / 4 + t1 / 2) ** n / n
        return WRF_EARTH_RADIUS_M * f / np.tan(np.pi / 4 + np.radians(lat) / 2) ** n

    def _forward(self, lon, lat):
        """Projected ``(x, y)`` in metres, origin at the pole's image shifted so ``cen`` is ``(0, 0)``."""
        n = self.cone
        dlon = np.mod(np.asarray(lon, np.float64) - self.stand_lon + 180.0, 360.0) - 180.0
        theta = n * np.radians(dlon)
        rho = self._rho(np.asarray(lat, np.float64))
        return rho * np.sin(theta), -rho * np.cos(theta)

    # -- public transforms ---------------------------------------------------

    def to_xy(self, lon, lat):
        """``(x, y)`` in metres relative to the domain centre."""
        xc, yc = self._forward(self.cen_lon, self.cen_lat)
        x, y = self._forward(lon, lat)
        return x - xc, y - yc

    def to_lonlat(self, x, y):
        """Inverse of :meth:`to_xy`: ``(lon, lat)`` in degrees, lon in ``[-180, 180)``."""
        n = self.cone
        s = np.sign(n)
        xc, yc = self._forward(self.cen_lon, self.cen_lat)
        px = np.asarray(x, np.float64) + xc
        py = np.asarray(y, np.float64) + yc
        rho = s * np.hypot(px, py)
        theta = np.arctan2(s * px, -s * py)
        t1 = np.radians(self.truelat1)
        f = np.cos(t1) * np.tan(np.pi / 4 + t1 / 2) ** n / n
        lat = np.degrees(2.0 * np.arctan((WRF_EARTH_RADIUS_M * f / rho) ** (1.0 / n)) - np.pi / 2)
        lon = self.stand_lon + np.degrees(theta / n)
        return np.mod(lon + 180.0, 360.0) - 180.0, lat

    def map_factor(self, lat):
        """Map-scale factor at *lat*, WRF's ``MAPFAC_M`` at mass points."""
        lat = np.asarray(lat, np.float64)
        return self.cone * self._rho(lat) / (WRF_EARTH_RADIUS_M * np.cos(np.radians(lat)))

    # -- grid coordinates ----------------------------------------------------

    def cell_centers(self, nx: int, ny: int, dx: float):
        """``(nx, ny)`` cell-centre ``(lon, lat)`` of a grid centred on ``cen``."""
        i = (np.arange(nx) - (nx - 1) / 2.0) * dx
        j = (np.arange(ny) - (ny - 1) / 2.0) * dx
        return self.to_lonlat(*np.meshgrid(i, j, indexing="ij"))

    def cell_corners(self, nx: int, ny: int, dx: float):
        """``(nx+1, ny+1)`` cell-corner ``(lon, lat)`` of a grid centred on ``cen``."""
        i = (np.arange(nx + 1) - nx / 2.0) * dx
        j = (np.arange(ny + 1) - ny / 2.0) * dx
        return self.to_lonlat(*np.meshgrid(i, j, indexing="ij"))

    def shifted(self, di: float, dj: float, dx: float) -> LambertConformal:
        """The same projection with the domain centre moved by ``(di, dj)`` cells.

        Only ``cen_lat``/``cen_lon`` change; ``stand_lon`` is pinned, so the
        grid orientation is unchanged. Used when an asymmetric crop moves the
        domain centre.
        """
        if di == 0 and dj == 0:
            return self
        lon, lat = self.to_lonlat(di * dx, dj * dx)
        return replace(self, cen_lon=float(lon), cen_lat=float(lat), stand_lon=self.stand_lon)
