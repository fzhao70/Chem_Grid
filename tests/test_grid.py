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
