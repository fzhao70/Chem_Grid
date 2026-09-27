"""Tests for chem_grid.Grid and chem_grid.Crop."""

from dataclasses import FrozenInstanceError

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chem_grid import Crop, Grid

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------


def test_grid_valid_constructs():
    g = Grid(nx=4, ny=5, nz=6, dx=4000.0)
    assert g.shape == (4, 5, 6)
    assert g.horizontal_shape == (4, 5)
    assert g.dx == 4000.0


@pytest.mark.parametrize("nx,ny,nz", [(0, 5, 6), (4, 0, 6), (4, 5, 0), (-1, 5, 6)])
def test_grid_rejects_degenerate_dims(nx, ny, nz):
    with pytest.raises(ValueError):
        Grid(nx=nx, ny=ny, nz=nz, dx=4000.0)


@pytest.mark.parametrize("dx", [0.0, -100.0])
def test_grid_rejects_nonpositive_dx(dx):
    with pytest.raises(ValueError):
        Grid(nx=4, ny=5, nz=6, dx=dx)


def test_grid_minimal_dims_allowed():
    assert Grid(nx=1, ny=1, nz=1, dx=1.0).shape == (1, 1, 1)


def test_grid_is_frozen_and_hashable():
    g1 = Grid(nx=4, ny=5, nz=6, dx=4000.0)
    g2 = Grid(nx=4, ny=5, nz=6, dx=4000.0)
    with pytest.raises(FrozenInstanceError):
        g1.nx = 10  # type: ignore[misc]
    assert g1 == g2 and hash(g1) == hash(g2)


def test_grid_usable_as_static_jit_argument():
    f = jax.jit(lambda x, grid: x.reshape(grid.shape), static_argnums=1)
    assert f(jnp.arange(8.0), Grid(nx=2, ny=2, nz=2, dx=1.0)).shape == (2, 2, 2)


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------

# The China 36 km configuration: a 179x123 WRF grid cropped by 5 on every side.
CHINA36 = Crop(ixb=5, ixbe=5, jxb=5, jxbe=5)


def test_crop_china36_grid():
    g = CHINA36.grid(179, 123, nz=35, dx=36000.0)
    assert g == Grid(nx=169, ny=113, nz=35, dx=36000.0)


def test_crop_apply_matches_slice():
    a = np.arange(179 * 123 * 2.0).reshape(179, 123, 2)
    np.testing.assert_array_equal(CHINA36.apply(a), a[5:174, 5:118])


def test_crop_asymmetric_widths():
    a = np.arange(10 * 8.0).reshape(10, 8)
    out = Crop(ixb=1, ixbe=2, jxb=3, jxbe=0).apply(a)
    np.testing.assert_array_equal(out, a[1:8, 3:8])


def test_crop_staggered_field_is_one_longer():
    u = np.zeros((180, 123, 35))  # x-staggered
    assert CHINA36.apply(u).shape == (170, 113, 35)


def test_crop_leaves_low_rank_arrays_alone():
    znu = np.linspace(1.0, 0.0, 35)
    assert CHINA36.apply(znu) is znu
    assert CHINA36.apply(3.0) == 3.0


def test_crop_jax_array_under_jit():
    a = jnp.arange(20.0).reshape(5, 4)
    crop = Crop(ixb=1, ixbe=1, jxb=1, jxbe=1)
    out = jax.jit(crop.apply)(a)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(a[1:4, 1:3]))


def test_crop_rejects_negative_width():
    with pytest.raises(ValueError):
        Crop(ixb=-1)


def test_crop_rejects_overcrop():
    with pytest.raises(ValueError):
        Crop(ixb=3, ixbe=3).shape(6, 10)
    with pytest.raises(ValueError):
        Crop(jxb=5, jxbe=5).apply(np.zeros((4, 10)))


# ---------------------------------------------------------------------------
# Grid: optional coordinates
# ---------------------------------------------------------------------------


def _coords(nx=4, ny=3):
    i, j = np.meshgrid(np.arange(nx + 1.0), np.arange(ny + 1.0), indexing="ij")
    lon_e, lat_e = 100.0 + 0.5 * i + 0.1 * j, 20.0 + 0.4 * j
    lon_c = 0.25 * (lon_e[:-1, :-1] + lon_e[1:, :-1] + lon_e[1:, 1:] + lon_e[:-1, 1:])
    lat_c = 0.25 * (lat_e[:-1, :-1] + lat_e[1:, :-1] + lat_e[1:, 1:] + lat_e[:-1, 1:])
    return lon_c, lat_c, lon_e, lat_e


def test_grid_coords_default_to_none():
    g = Grid(nx=4, ny=3, nz=2, dx=1.0)
    assert g.lon is None and g.lat is None and g.lon_edges is None and g.lat_edges is None
    assert not g.has_centers and not g.has_edges


def test_grid_stores_assigned_coords_as_readonly_copies():
    lon, lat, lon_e, lat_e = _coords()
    g = Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon, lat=lat, lon_edges=lon_e, lat_edges=lat_e)
    assert g.has_centers and g.has_edges
    np.testing.assert_array_equal(g.lon_edges, lon_e)
    lon[0, 0] = -999.0  # caller's array changes; the grid's copy does not
    assert g.lon[0, 0] != -999.0
    with pytest.raises(ValueError):
        g.lat[0, 0] = 0.0


def test_grid_centers_and_edges_are_independent():
    lon, lat, lon_e, lat_e = _coords()
    assert Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon, lat=lat).has_edges is False
    assert Grid(nx=4, ny=3, nz=2, dx=1.0, lon_edges=lon_e, lat_edges=lat_e).has_centers is False


@pytest.mark.parametrize(
    "kw",
    [
        {"lon": np.zeros((4, 3))},  # lon without lat
        {"lat_edges": np.zeros((5, 4))},  # lat_edges without lon_edges
        {"lon": np.zeros((3, 4)), "lat": np.zeros((3, 4))},  # transposed
        {"lon_edges": np.zeros((4, 3)), "lat_edges": np.zeros((4, 3))},  # centre-shaped edges
        {"lon": np.zeros((4, 3)), "lat": np.full((4, 3), 91.0)},  # latitude out of range
        {"lon": np.full((4, 3), np.nan), "lat": np.zeros((4, 3))},  # non-finite
    ],
)
def test_grid_rejects_bad_coords(kw):
    with pytest.raises(ValueError):
        Grid(nx=4, ny=3, nz=2, dx=1.0, **kw)


def test_grid_equality_and_hash_include_coords():
    lon, lat, _, _ = _coords()
    a = Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon, lat=lat)
    b = Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon.copy(), lat=lat.copy())
    c = Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon + 1.0, lat=lat)
    bare = Grid(nx=4, ny=3, nz=2, dx=1.0)
    assert a == b and hash(a) == hash(b)
    assert a != c and a != bare
    assert len({a, b, c, bare}) == 3


def test_grid_with_coords_is_static_jit_argument():
    lon, lat, _, _ = _coords()
    g = Grid(nx=4, ny=3, nz=1, dx=1.0, lon=lon, lat=lat)
    f = jax.jit(lambda x, grid: x.reshape(grid.horizontal_shape) + grid.lat, static_argnums=1)
    np.testing.assert_allclose(np.asarray(f(jnp.zeros(12), g)), lat)


def test_grid_repr_mentions_coords_not_values():
    lon, lat, _, _ = _coords()
    r = repr(Grid(nx=4, ny=3, nz=2, dx=1.0, lon=lon, lat=lat))
    assert r == "Grid(nx=4, ny=3, nz=2, dx=1.0, coords=lon)"


def test_crop_grid_crops_coords():
    lon, lat, lon_e, lat_e = _coords(nx=10, ny=8)
    crop = Crop(ixb=2, ixbe=1, jxb=1, jxbe=3)
    g = crop.grid(10, 8, nz=5, dx=1.0, lon=lon, lat=lat, lon_edges=lon_e, lat_edges=lat_e)
    assert g.horizontal_shape == (7, 4)
    np.testing.assert_array_equal(g.lon, lon[2:9, 1:5])
    np.testing.assert_array_equal(g.lat_edges, lat_e[2:10, 1:6])  # (nx+1, ny+1)


def test_crop_grid_rejects_precropped_coords():
    lon, lat, _, _ = _coords(nx=10, ny=8)
    with pytest.raises(ValueError):
        Crop(ixb=1, ixbe=1).grid(10, 8, nz=5, dx=1.0, lon=lon[1:9], lat=lat[1:9])


def test_grid_edges_feed_conservative_regrid():
    from chem_grid.regrid import conservative_weights

    _, _, lon_e, lat_e = _coords()
    g = Grid(nx=4, ny=3, nz=1, dx=1.0, lon_edges=lon_e, lat_edges=lat_e)
    w = conservative_weights(np.arange(99.0, 104.01, 0.25), np.arange(19.0, 23.01, 0.25),
                             g.lon_edges, g.lat_edges)
    assert w.dst_shape == g.horizontal_shape
