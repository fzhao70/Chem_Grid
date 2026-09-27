"""Bilinear interpolation from a rectilinear lon/lat grid to arbitrary points.

The source is a regular lon/lat grid given by its 1-D cell-centre
coordinates (a typical inventory or reanalysis file); the destination is any
set of points given by 2-D (or N-D) lon/lat arrays, e.g. the model's
cell-centre ``XLONG``/``XLAT``. Interpolation is linear in degrees of
longitude and latitude, the usual convention for gridded geophysical fields.

Bilinear is not mass-conserving; use :func:`chem_grid.regrid.conservative_weights`
for fluxes or amounts per area (emissions, deposition). Bilinear suits
intensive, smooth fields: temperature, mixing ratios, boundary profiles.
"""

from __future__ import annotations

import numpy as np

from chem_grid.regrid.weights import RegridWeights

__all__ = ["bilinear_weights", "is_periodic_lon"]


def is_periodic_lon(lon: np.ndarray) -> bool:
    """True if the 1-D centres *lon* (degrees, ascending) wrap around the globe.

    The gap between the last centre and the first one plus 360 must be no
    larger than 1.5 grid spacings (a global grid's gap is exactly one).
    """
    lon = np.asarray(lon, dtype=np.float64)
    if lon.size < 2:
        return False
    gap = lon[0] + 360.0 - lon[-1]
    return bool(0.0 < gap <= 1.5 * np.max(np.diff(lon)))


def _axis_index(coord, x, *, periodic):
    """Left neighbour index and fraction of *x* along ascending centres *coord*.

    Returns ``(i0, i1, t, inside)``: *x* lies ``t`` of the way from
    ``coord[i0]`` to ``coord[i1]``. For a periodic axis the last cell wraps to
    the first. ``t`` is not clipped; *inside* flags ``0 <= t <= 1`` on a
    non-periodic axis (always true on a periodic one).
    """
    n = coord.size
    if periodic:
        ext = np.concatenate([coord, [coord[0] + 360.0]])
    else:
        ext = coord
    j = np.searchsorted(ext, x, side="right") - 1
    j = np.clip(j, 0, ext.size - 2)
    t = (x - ext[j]) / (ext[j + 1] - ext[j])
    i0 = j
    i1 = (j + 1) % n if periodic else j + 1
    inside = np.ones(x.shape, bool) if periodic else (t >= 0.0) & (t <= 1.0)
    return i0, i1, t, inside


def bilinear_weights(
    lon_src,
    lat_src,
    lon_dst,
    lat_dst,
    *,
    periodic: bool | None = None,
    outside: str = "nan",
) -> RegridWeights:
    """Bilinear weights from a rectilinear lon/lat source to arbitrary points.

    Parameters
    ----------
    lon_src : ``(nlon,)``
        Source cell-centre longitudes in degrees, strictly ascending.
    lat_src : ``(nlat,)``
        Source cell-centre latitudes in degrees, strictly ascending or
        strictly descending (north-to-south files are common).
    lon_dst, lat_dst : any shape, equal
        Destination point longitudes/latitudes in degrees; their shape is the
        result's ``dst_shape``. Longitudes may use either the ``[-180, 180)``
        or ``[0, 360)`` convention, independently of the source.
    periodic :
        Whether the source longitudes wrap around the globe. ``None`` detects
        it with :func:`is_periodic_lon`.
    outside :
        What to do with points outside the source's centre-to-centre extent:
        ``"nan"`` marks them uncovered (the regridded value is the fill
        value, NaN by default); ``"nearest"`` clamps them onto the nearest
        edge of the source grid.

    Returns
    -------
    RegridWeights
        ``src_shape == (nlon, nlat)``: source fields are laid out
        longitude-first, like the model's ``(nx, ny)``. Transpose a
        ``(lat, lon)`` file array before applying.
    """
    if outside not in ("nan", "nearest"):
        raise ValueError(f"outside must be 'nan' or 'nearest', got {outside!r}")
    lon_src = np.asarray(lon_src, dtype=np.float64)
    lat_src = np.asarray(lat_src, dtype=np.float64)
    lon_dst = np.asarray(lon_dst, dtype=np.float64)
    lat_dst = np.asarray(lat_dst, dtype=np.float64)
    if lon_src.ndim != 1 or lat_src.ndim != 1:
        raise ValueError("lon_src and lat_src must be 1-D cell-centre coordinates")
    if lon_dst.shape != lat_dst.shape:
        raise ValueError(f"lon_dst {lon_dst.shape} and lat_dst {lat_dst.shape} differ in shape")
    if lon_src.size < 2 or lat_src.size < 2:
        raise ValueError("the source grid needs at least 2 centres per axis")
    if np.any(np.diff(lon_src) <= 0):
        raise ValueError("lon_src must be strictly ascending")
    dlat = np.diff(lat_src)
    if not (np.all(dlat > 0) or np.all(dlat < 0)):
        raise ValueError("lat_src must be strictly monotonic")
    if periodic is None:
        periodic = is_periodic_lon(lon_src)

    nlon, nlat = lon_src.size, lat_src.size
    x = lon_dst.reshape(-1)
    y = lat_dst.reshape(-1)

    # Move each destination longitude onto the source's branch: the window
    # starting at lon_src[0] when periodic, else the 360-degree window centred
    # on the source's midpoint.
    if periodic:
        x = lon_src[0] + np.mod(x - lon_src[0], 360.0)
    else:
        mid = 0.5 * (lon_src[0] + lon_src[-1])
        x = mid - 180.0 + np.mod(x - (mid - 180.0), 360.0)

    descending = dlat[0] < 0
    lat_asc = lat_src[::-1] if descending else lat_src

    i0, i1, tx, in_x = _axis_index(lon_src, x, periodic=periodic)
    j0, j1, ty, in_y = _axis_index(lat_asc, y, periodic=False)
    if descending:
        j0, j1 = nlat - 1 - j0, nlat - 1 - j1

    inside = in_x & in_y
    if outside == "nearest":
        tx = np.clip(tx, 0.0, 1.0)
        ty = np.clip(ty, 0.0, 1.0)
        inside = np.ones_like(inside)

    npt = x.size
    row = np.repeat(np.arange(npt), 4)
    col = np.stack([i0 * nlat + j0, i1 * nlat + j0, i0 * nlat + j1, i1 * nlat + j1], axis=1)
    w = np.stack([(1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty], axis=1)
    keep = np.repeat(inside, 4)
    return RegridWeights(
        row=row[keep],
        col=col.reshape(-1)[keep],
        weight=w.reshape(-1)[keep],
        covered=inside.astype(np.float64).reshape(lon_dst.shape),
        src_shape=(nlon, nlat),
        dst_shape=tuple(lon_dst.shape),
    )
