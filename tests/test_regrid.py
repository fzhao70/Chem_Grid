"""Tests for chem_grid.regrid: bilinear and conservative weights and their application."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chem_grid.regrid import (
    RegridWeights,
    bilinear_weights,
    cell_areas,
    conservative_weights,
    is_periodic_lon,
    rectilinear_corners,
)

jax.config.update("jax_enable_x64", True)


def _lambert_like_corners(nx=12, ny=9, lon0=105.0, lat0=32.0, d=0.35, shear=0.08):
    """A sheared, rotated curvilinear mesh standing in for a model grid."""
    i, j = np.meshgrid(np.arange(nx + 1), np.arange(ny + 1), indexing="ij")
    lon = lon0 + d * i + shear * d * j
    lat = lat0 + d * j - 0.5 * shear * d * i + 0.002 * i * j
    return lon, lat


# ---------------------------------------------------------------------------
# bilinear
# ---------------------------------------------------------------------------


def test_bilinear_reproduces_linear_field():
    lon = np.arange(90.0, 130.1, 0.5)
    lat = np.arange(10.0, 50.1, 0.5)
    f = 2.0 * lon[:, None] - 3.0 * lat[None, :] + 7.0
    lon_d = np.array([[100.13, 115.7], [101.0, 129.9]])
    lat_d = np.array([[20.21, 33.3], [49.99, 10.01]])
    w = bilinear_weights(lon, lat, lon_d, lat_d)
    np.testing.assert_allclose(np.asarray(w(f)), 2.0 * lon_d - 3.0 * lat_d + 7.0, rtol=1e-12)


def test_bilinear_hits_source_centres_exactly():
    lon = np.linspace(100.0, 110.0, 11)
    lat = np.linspace(20.0, 30.0, 6)
    f = np.random.default_rng(0).random((11, 6))
    lo, la = np.meshgrid(lon, lat, indexing="ij")
    np.testing.assert_allclose(np.asarray(bilinear_weights(lon, lat, lo, la)(f)), f, rtol=1e-13)


def test_bilinear_descending_latitudes():
    lon = np.linspace(100.0, 110.0, 11)
    lat_desc = np.linspace(30.0, 20.0, 11)
    f = lon[:, None] + 10.0 * lat_desc[None, :]
    w = bilinear_weights(lon, lat_desc, np.array([103.3]), np.array([27.45]))
    np.testing.assert_allclose(np.asarray(w(f)), [103.3 + 274.5], rtol=1e-12)


def test_bilinear_outside_nan_and_nearest():
    lon = np.linspace(100.0, 110.0, 11)
    lat = np.linspace(20.0, 30.0, 11)
    f = np.ones((11, 11))
    lon_d, lat_d = np.array([105.0, 95.0, 105.0]), np.array([25.0, 25.0, 31.0])
    w = bilinear_weights(lon, lat, lon_d, lat_d)
    np.testing.assert_array_equal(w.covered, [1.0, 0.0, 0.0])
    out = np.asarray(w(f))
    assert out[0] == 1.0 and np.isnan(out[1]) and np.isnan(out[2])
    np.testing.assert_allclose(np.asarray(bilinear_weights(lon, lat, lon_d, lat_d, outside="nearest")(f)), 1.0)


def test_bilinear_periodic_seam_and_lon_convention():
    lon = np.arange(0.0, 360.0, 10.0)  # global, 0..350
    lat = np.array([-10.0, 10.0])
    assert is_periodic_lon(lon)
    f = np.cos(np.radians(lon))[:, None] * np.ones(2)
    # 355 and -5 are the same point, halfway between 350 and 0 (== 360).
    w = bilinear_weights(lon, lat, np.array([355.0, -5.0]), np.array([0.0, 0.0]))
    expected = 0.5 * (np.cos(np.radians(350.0)) + 1.0)
    np.testing.assert_allclose(np.asarray(w(f)), expected, rtol=1e-12)


def test_bilinear_regional_source_accepts_negative_lons():
    lon = np.linspace(-120.0, -60.0, 61)  # regional, western hemisphere
    lat = np.linspace(20.0, 50.0, 31)
    assert not is_periodic_lon(lon)
    f = lon[:, None] + 0.0 * lat[None, :]
    w = bilinear_weights(lon, lat, np.array([250.5]), np.array([35.0]))  # == -109.5
    np.testing.assert_allclose(np.asarray(w(f)), [-109.5], rtol=1e-12)


def test_bilinear_carries_trailing_axes():
    lon = np.linspace(100.0, 110.0, 6)
    lat = np.linspace(20.0, 30.0, 6)
    f = np.random.default_rng(1).random((6, 6, 3, 2))
    lo, la = np.meshgrid(lon[1:3], lat[2:5], indexing="ij")
    out = bilinear_weights(lon, lat, lo, la)(f)
    assert out.shape == (2, 3, 3, 2)
    np.testing.assert_allclose(np.asarray(out), f[1:3, 2:5], rtol=1e-13)


# ---------------------------------------------------------------------------
# conservative
# ---------------------------------------------------------------------------


def test_conservative_identity():
    lon_e = np.linspace(100.0, 104.0, 5)
    lat_e = np.linspace(20.0, 23.0, 4)
    w = conservative_weights(lon_e, lat_e, lon_e, lat_e)
    np.testing.assert_allclose(w.to_dense(), np.eye(12), atol=1e-12)
    np.testing.assert_allclose(w.covered, 1.0)


def test_conservative_coarsening_is_area_weighted_mean():
    lon_f = np.linspace(100.0, 104.0, 9)  # 0.5 deg
    lat_f = np.linspace(0.0, 60.0, 13)  # 5 deg: cell areas differ strongly with latitude
    lon_c, lat_c = lon_f[::2], lat_f[::2]
    f = np.random.default_rng(2).random((8, 12))
    a = cell_areas(lon_f, lat_f)
    out = np.asarray(conservative_weights(lon_f, lat_f, lon_c, lat_c)(f))
    expected = (f * a).reshape(4, 2, 6, 2).sum((1, 3)) / a.reshape(4, 2, 6, 2).sum((1, 3))
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_cell_areas_match_spherical_formula():
    lon_e = np.array([0.0, 1.0])
    lat_e = np.array([30.0, 31.0])
    r = 6.371e6
    exact = r**2 * np.radians(1.0) * (np.sin(np.radians(31.0)) - np.sin(np.radians(30.0)))
    np.testing.assert_allclose(cell_areas(lon_e, lat_e), [[exact]], rtol=1e-14)


def test_conservative_constant_field_on_curvilinear_mesh():
    lon_d, lat_d = _lambert_like_corners()
    lon_s = np.arange(100.0, 115.01, 0.25)
    lat_s = np.arange(28.0, 40.01, 0.25)
    w = conservative_weights(lon_s, lat_s, lon_d, lat_d)
    # Overlaps of ~1e-5-sized planar cells lose a few digits to cancellation.
    np.testing.assert_allclose(w.covered, 1.0, rtol=1e-10)
    out = np.asarray(w(np.full((lon_s.size - 1, lat_s.size - 1), 3.5)))
    np.testing.assert_allclose(out, 3.5, rtol=1e-10)


def test_conservative_conserves_integral_both_directions():
    # Destination mesh lies wholly inside the source: integral over destination
    # equals the source integral weighted by overlap, and mapping back to the
    # source conserves that same integral.
    lon_d, lat_d = _lambert_like_corners()
    lon_s = np.arange(100.0, 115.01, 0.3)
    lat_s = np.arange(28.0, 40.01, 0.3)
    rng = np.random.default_rng(3)
    f = rng.random((lon_s.size - 1, lat_s.size - 1))
    fwd = conservative_weights(lon_s, lat_s, lon_d, lat_d)
    g = np.asarray(fwd(f))
    a_d = cell_areas(lon_d, lat_d)
    a_s = cell_areas(lon_s, lat_s)
    # Source-side integral restricted to the overlap: sum_s f_s * sum_d A_overlap(d, s).
    overlap_per_src = (fwd.to_dense() * a_d.reshape(-1, 1)).sum(0).reshape(a_s.shape)
    np.testing.assert_allclose((g * a_d).sum(), (f * overlap_per_src).sum(), rtol=1e-12)

    back = conservative_weights(lon_d, lat_d, lon_s, lat_s)
    h = np.nan_to_num(np.asarray(back(g)))
    np.testing.assert_allclose((h * a_s).sum(), (g * a_d).sum(), rtol=1e-12)


def test_conservative_partial_coverage_normalizations():
    lon_s, lat_s = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    lon_d, lat_d = np.array([0.5, 1.5]), np.array([0.0, 1.0])  # half covered
    f = np.array([[4.0]])
    dest = conservative_weights(lon_s, lat_s, lon_d, lat_d)
    frac = conservative_weights(lon_s, lat_s, lon_d, lat_d, normalize="fracarea")
    np.testing.assert_allclose(dest.covered, [[0.5]], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(dest(f)), [[2.0]], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(frac(f)), [[4.0]], rtol=1e-12)


def test_conservative_no_overlap_gets_fill_value():
    w = conservative_weights([0.0, 1.0], [0.0, 1.0], [5.0, 6.0], [0.0, 1.0])
    assert w.row.size == 0 and w.covered[0, 0] == 0.0
    assert np.isnan(np.asarray(w(np.ones((1, 1))))[0, 0])
    assert np.asarray(w(np.ones((1, 1)), fill_value=-1.0))[0, 0] == -1.0


def test_conservative_global_seam():
    lon_s = np.arange(0.0, 360.01, 10.0)  # global, edges 0..360
    lat_s = np.array([-10.0, 10.0])
    f = np.zeros((36, 1))
    f[0, 0], f[-1, 0] = 1.0, 3.0  # cells [0, 10] and [350, 360]
    # A cell from -5 to 5 (== 355..5) straddles the seam equally.
    w = conservative_weights(lon_s, lat_s, np.array([-5.0, 5.0]), np.array([-10.0, 10.0]))
    np.testing.assert_allclose(w.covered, [[1.0]], rtol=1e-12)
    np.testing.assert_allclose(np.asarray(w(f)), [[2.0]], rtol=1e-12)


def test_conservative_handles_reversed_orientation():
    lon_e = np.linspace(100.0, 102.0, 3)
    lat_desc = np.linspace(22.0, 20.0, 3)  # north-to-south edges
    f = np.random.default_rng(4).random((2, 2))
    w = conservative_weights(lon_e, lat_desc, lon_e, lat_desc[::-1])
    np.testing.assert_allclose(np.asarray(w(f)), f[:, ::-1], rtol=1e-12)


# ---------------------------------------------------------------------------
# RegridWeights under JAX transforms
# ---------------------------------------------------------------------------


def test_weights_are_a_pytree_under_jit_and_grad():
    lon_d, lat_d = _lambert_like_corners(nx=4, ny=3)
    lon_s = np.arange(104.0, 108.01, 0.5)
    lat_s = np.arange(30.0, 34.01, 0.5)
    w = conservative_weights(lon_s, lat_s, lon_d, lat_d)
    f = jnp.asarray(np.random.default_rng(5).random((lon_s.size - 1, lat_s.size - 1)))

    @jax.jit
    def total(weights: RegridWeights, field):
        return jnp.sum(weights(field) * jnp.asarray(cell_areas(lon_d, lat_d)))

    np.testing.assert_allclose(total(w, f), jnp.sum(w(f) * cell_areas(lon_d, lat_d)), rtol=1e-12)
    g = jax.grad(total, argnums=1)(w, f)
    # d(integral)/d(f_s) is the overlap area of source cell s with the destination mesh.
    expected = (w.to_dense() * cell_areas(lon_d, lat_d).reshape(-1, 1)).sum(0).reshape(f.shape)
    np.testing.assert_allclose(np.asarray(g), expected, rtol=1e-12)


def test_weights_reject_wrong_field_shape():
    w = bilinear_weights(np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.5]), np.array([0.5]))
    with pytest.raises(ValueError):
        w(np.ones((3, 2)))


def test_rectilinear_corners_layout():
    lo, la = rectilinear_corners([0.0, 1.0, 2.0], [10.0, 20.0])
    assert lo.shape == (3, 2)
    np.testing.assert_array_equal(lo[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(la[0], [10.0, 20.0])
