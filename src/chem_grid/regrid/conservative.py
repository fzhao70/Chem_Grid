"""First-order conservative regridding between quadrilateral cell meshes.

Both grids are given by their cell corners: ``(n+1, m+1)`` lon/lat arrays for
an ``(n, m)`` grid, cell ``(i, j)`` having corners ``(i, j)``, ``(i+1, j)``,
``(i+1, j+1)``, ``(i, j+1)``. A rectilinear lon/lat grid is just 1-D edge
arrays (see :func:`rectilinear_corners`); a curvilinear model grid (WRF's
Lambert conformal, say) needs its corner coordinates.

Geometry is done in the equal-area plane ``(lon [rad], sin(lat))``, where a
cell's planar area times ``R**2`` is its area on the sphere. That is exact for
cells bounded by meridians and parallels (every rectilinear lon/lat grid);
for other cells it treats each edge as straight in that plane rather than a
great circle, an error far below 1% for mesoscale cells away from the poles.
Either way the weights are exactly conservative with respect to the areas
:func:`cell_areas` reports, because intersections and cell areas use the
same metric.

Every cell must be convex in that plane (true for any sane model grid), since
cells are intersected by Sutherland-Hodgman clipping.
"""

from __future__ import annotations

import numpy as np

from chem_grid.regrid.weights import RegridWeights

__all__ = ["cell_areas", "conservative_weights", "rectilinear_corners"]

EARTH_RADIUS_M = 6.371e6
_TWO_PI = 2.0 * np.pi


def rectilinear_corners(lon_edges, lat_edges) -> tuple[np.ndarray, np.ndarray]:
    """``(nlon+1, nlat+1)`` corner arrays of a rectilinear grid from its 1-D edges."""
    lon_edges = np.asarray(lon_edges, dtype=np.float64)
    lat_edges = np.asarray(lat_edges, dtype=np.float64)
    if lon_edges.ndim != 1 or lat_edges.ndim != 1:
        raise ValueError("lon_edges and lat_edges must be 1-D")
    return np.meshgrid(lon_edges, lat_edges, indexing="ij")


def _as_corners(lon, lat):
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    if lon.ndim == 1 and lat.ndim == 1:
        lon, lat = rectilinear_corners(lon, lat)
    if lon.ndim != 2 or lon.shape != lat.shape:
        raise ValueError(
            f"corners must be 1-D edge arrays or equal-shape 2-D arrays, got {lon.shape} and {lat.shape}"
        )
    if min(lon.shape) < 2:
        raise ValueError(f"need at least one cell, got corner shape {lon.shape}")
    if np.any(np.abs(lat) > 90.0):
        raise ValueError("latitudes must lie within [-90, 90]")
    return lon, lat


def _quads(lon_c, lat_c):
    """``(n*m, 4, 2)`` counter-clockwise quads in the ``(lon rad, sin lat)`` plane."""
    x = np.radians(lon_c)
    y = np.sin(np.radians(lat_c))
    q = np.stack(
        [
            np.stack([x[:-1, :-1], y[:-1, :-1]], -1),
            np.stack([x[1:, :-1], y[1:, :-1]], -1),
            np.stack([x[1:, 1:], y[1:, 1:]], -1),
            np.stack([x[:-1, 1:], y[:-1, 1:]], -1),
        ],
        axis=2,
    ).reshape(-1, 4, 2)
    # Keep each cell on one longitude branch: within pi of its first corner.
    x0 = q[:, :1, 0]
    q[:, :, 0] = x0 + np.mod(q[:, :, 0] - x0 + np.pi, _TWO_PI) - np.pi
    flip = _signed_area(q, np.full(q.shape[0], 4)) < 0
    q[flip] = q[flip][:, ::-1]
    return q


def _signed_area(v, cnt):
    """Shoelace area of padded polygons ``v`` ``(P, M, 2)`` with ``cnt`` valid vertices."""
    m = v.shape[1]
    k = np.arange(m)[None, :]
    nxt = (k + 1) % np.maximum(cnt[:, None], 1)
    w = np.take_along_axis(v, nxt[..., None], axis=1)
    term = v[..., 0] * w[..., 1] - w[..., 0] * v[..., 1]
    return 0.5 * np.sum(np.where(k < cnt[:, None], term, 0.0), axis=1)


def _clip(subj, clip):
    """Sutherland-Hodgman: each ``subj[p]`` clipped by the convex CCW quad ``clip[p]``."""
    v = subj
    cnt = np.full(v.shape[0], subj.shape[1])
    for e in range(clip.shape[1]):
        a = clip[:, e][:, None, :]
        ab = clip[:, (e + 1) % clip.shape[1]][:, None, :] - a
        m = v.shape[1]
        k = np.arange(m)[None, :]
        valid = k < cnt[:, None]
        prev = np.take_along_axis(v, ((k - 1) % np.maximum(cnt[:, None], 1))[..., None], axis=1)
        d_cur = ab[..., 0] * (v[..., 1] - a[..., 1]) - ab[..., 1] * (v[..., 0] - a[..., 0])
        d_prev = ab[..., 0] * (prev[..., 1] - a[..., 1]) - ab[..., 1] * (prev[..., 0] - a[..., 0])
        in_cur = d_cur >= 0.0
        in_prev = d_prev >= 0.0
        denom = d_prev - d_cur
        t = np.where(denom != 0.0, d_prev / np.where(denom != 0.0, denom, 1.0), 0.0)
        inter = prev + t[..., None] * (v - prev)
        # Per input vertex: the edge crossing (if any), then the vertex itself if inside.
        out = np.stack([inter, v], axis=2).reshape(v.shape[0], 2 * m, 2)
        ok = np.stack([(in_cur != in_prev) & valid, in_cur & valid], axis=2).reshape(-1, 2 * m)
        order = np.argsort(~ok, axis=1, kind="stable")
        cnt = ok.sum(axis=1)
        v = np.take_along_axis(out, order[..., None], axis=1)[:, : max(int(cnt.max()), 1)]
    return v, cnt


def _bbox(q):
    return q[..., 0].min(1), q[..., 0].max(1), q[..., 1].min(1), q[..., 1].max(1)


def _candidate_pairs(q_index, q_query):
    """Index pairs ``(i, j)`` whose bounding boxes overlap, via a uniform bin index on ``q_index``."""
    ix0, ix1, iy0, iy1 = _bbox(q_index)
    qx0, qx1, qy0, qy1 = _bbox(q_query)
    hx = max(float(np.max(ix1 - ix0)), 1e-12)
    hy = max(float(np.max(iy1 - iy0)), 1e-12)
    ox, oy = float(ix0.min()), float(iy0.min())
    bx = np.floor((ix0 - ox) / hx).astype(np.int64)
    by = np.floor((iy0 - oy) / hy).astype(np.int64)
    nbx, nby = int(bx.max()) + 1, int(by.max()) + 1
    key = bx * nby + by
    order = np.argsort(key, kind="stable")
    skey = key[order]

    # An index cell is at most one bin wide, so its lower-left bin lies within
    # one bin below/left of the query box.
    qbx0 = np.clip(np.floor((qx0 - hx - ox) / hx), 0, nbx - 1).astype(np.int64)
    qbx1 = np.clip(np.floor((qx1 - ox) / hx), -1, nbx - 1).astype(np.int64)
    qby0 = np.clip(np.floor((qy0 - hy - oy) / hy), 0, nby - 1).astype(np.int64)
    qby1 = np.clip(np.floor((qy1 - oy) / hy), -1, nby - 1).astype(np.int64)

    out_i, out_j = [], []
    for j in np.nonzero((qbx1 >= qbx0) & (qby1 >= qby0))[0]:
        cols = np.arange(qbx0[j], qbx1[j] + 1)
        lo = np.searchsorted(skey, cols * nby + qby0[j], side="left")
        hi = np.searchsorted(skey, cols * nby + qby1[j], side="right")
        for a, b in zip(lo, hi):
            if b > a:
                out_i.append(order[a:b])
                out_j.append(np.full(b - a, j))
    if not out_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    i = np.concatenate(out_i)
    j = np.concatenate(out_j)
    hit = (ix0[i] < qx1[j]) & (qx0[j] < ix1[i]) & (iy0[i] < qy1[j]) & (qy0[j] < iy1[i])
    return i[hit], j[hit]


def cell_areas(lon_corners, lat_corners, radius: float = EARTH_RADIUS_M) -> np.ndarray:
    """Cell areas in m^2 (for ``radius`` in m), in the metric the weights conserve."""
    lon, lat = _as_corners(lon_corners, lat_corners)
    q = _quads(lon, lat)
    area = np.abs(_signed_area(q, np.full(q.shape[0], 4)))
    return (radius**2 * area).reshape(lon.shape[0] - 1, lon.shape[1] - 1)


def conservative_weights(
    lon_src,
    lat_src,
    lon_dst,
    lat_dst,
    *,
    normalize: str = "destarea",
) -> RegridWeights:
    """First-order conservative weights between two quadrilateral meshes.

    Parameters
    ----------
    lon_src, lat_src :
        Source cell corners in degrees: 1-D edge arrays for a rectilinear
        grid, or ``(nx_src+1, ny_src+1)`` corner arrays.
    lon_dst, lat_dst :
        Destination cell corners, same forms.
    normalize :
        ``"destarea"`` (default): ``w = A_overlap / A_dst``. The area
        integral is conserved exactly; a destination cell only partly inside
        the source domain gets a value scaled down by its covered fraction.
        ``"fracarea"``: ``w = A_overlap / sum(A_overlap)``, the mean over the
        covered part only; conserves the integral only where coverage is full.

    Returns
    -------
    RegridWeights
        Shapes are the cell counts ``(nx, ny)`` of each mesh, x first.
        ``covered`` is each destination cell's covered area fraction; cells
        with no overlap get the fill value.

    Notes
    -----
    Apply to intensive-per-area quantities (fluxes in kg m-2 s-1, mixing
    ratios, concentrations): ``sum(dst * cell_areas(dst))`` then equals
    ``sum(src * cell_areas(src))`` over the overlap. Convert amounts per cell
    (kg per cell) to per-area first.
    """
    if normalize not in ("destarea", "fracarea"):
        raise ValueError(f"normalize must be 'destarea' or 'fracarea', got {normalize!r}")
    slon, slat = _as_corners(lon_src, lat_src)
    dlon, dlat = _as_corners(lon_dst, lat_dst)
    src_shape = (slon.shape[0] - 1, slon.shape[1] - 1)
    dst_shape = (dlon.shape[0] - 1, dlon.shape[1] - 1)
    qs = _quads(slon, slat)
    qd = _quads(dlon, dlat)
    n_dst = qd.shape[0]

    # Put each destination cell on the source's longitude branch, then add
    # copies shifted by one turn either way so cells straddling a global
    # source's seam meet the cells on both sides of it.
    mid = 0.5 * (qs[..., 0].min() + qs[..., 0].max())
    x0 = qd[:, :1, 0]
    qd[:, :, 0] += (mid - np.pi + np.mod(x0 - (mid - np.pi), _TWO_PI)) - x0
    shifts = np.array([0.0, -_TWO_PI, _TWO_PI])
    qd3 = np.concatenate([qd + np.array([s, 0.0]) for s in shifts])
    d_of = np.tile(np.arange(n_dst), shifts.size)

    if qs.shape[0] >= qd3.shape[0]:
        s_idx, d3_idx = _candidate_pairs(qs, qd3)
    else:
        d3_idx, s_idx = _candidate_pairs(qd3, qs)

    area_dst = np.abs(_signed_area(qd, np.full(n_dst, 4)))
    if s_idx.size:
        poly, cnt = _clip(qd3[d3_idx], qs[s_idx])
        overlap = np.abs(_signed_area(poly, cnt))
    else:
        overlap = np.zeros(0)
    d_idx = d_of[d3_idx]
    keep = overlap > 1e-12 * np.maximum(area_dst[d_idx], 1e-300)
    s_idx, d_idx, overlap = s_idx[keep], d_idx[keep], overlap[keep]

    covered_area = np.bincount(d_idx, weights=overlap, minlength=n_dst)
    covered = np.where(area_dst > 0, covered_area / np.where(area_dst > 0, area_dst, 1.0), 0.0)
    denom = area_dst if normalize == "destarea" else covered_area
    weight = overlap / denom[d_idx]
    return RegridWeights(
        row=d_idx,
        col=s_idx,
        weight=weight,
        covered=covered.reshape(dst_shape),
        src_shape=src_shape,
        dst_shape=dst_shape,
    )
