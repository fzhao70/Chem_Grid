"""Tests for chem_grid.LambertConformal and its use in Grid and Crop."""

from pathlib import Path

import numpy as np
import pytest

from chem_grid import Crop, Grid, LambertConformal

# REAM's China 36 km domain (constants.cpp:621-625: phic, xlonc, truelat01, truelat02).
CHINA36 = LambertConformal(cen_lat=37.0, cen_lon=107.0, truelat1=30.0, truelat2=45.0)
WRF_SAMPLE = Path("/mnt/n/WRF/post_process/output/wrfout_d01_2018-03-03_mean.nc")


def test_stand_lon_defaults_to_cen_lon():
    assert CHINA36.stand_lon == 107.0
    assert LambertConformal(37, 107, 30, 45, stand_lon=100).stand_lon == 100.0


def test_projection_is_hashable_value():
    assert CHINA36 == LambertConformal(37, 107, 30, 45.0)
    assert len({CHINA36, LambertConformal(37, 107, 30, 45), LambertConformal(37, 107, 30, 50)}) == 2


@pytest.mark.parametrize(
    "args",
    [(95, 107, 30, 45), (37, 107, 90, 45), (37, 107, 0, 0), (37, 107, -30, 45)],
)
def test_projection_rejects_bad_params(args):
    with pytest.raises(ValueError):
        LambertConformal(*args)


def test_from_wrf_attrs():
    attrs = {"MAP_PROJ": 1, "CEN_LAT": 37.0, "CEN_LON": 107.0, "TRUELAT1": 30.0,
             "TRUELAT2": 45.0, "STAND_LON": 107.0}
    assert LambertConformal.from_wrf_attrs(attrs) == CHINA36
    assert LambertConformal.from_wrf_attrs({**attrs, "STAND_LON": 110.0}).stand_lon == 110.0
    with pytest.raises(ValueError):
        LambertConformal.from_wrf_attrs({**attrs, "MAP_PROJ": 3})


def test_centre_maps_to_origin_and_round_trips():
    x, y = CHINA36.to_xy(107.0, 37.0)
    assert abs(x) < 1e-6 and abs(y) < 1e-6
    rng = np.random.default_rng(0)
    lon, lat = rng.uniform(70, 140, 200), rng.uniform(5, 60, 200)
    lon2, lat2 = CHINA36.to_lonlat(*CHINA36.to_xy(lon, lat))
    np.testing.assert_allclose(lon2, lon, atol=1e-9)
    np.testing.assert_allclose(lat2, lat, atol=1e-9)


def test_map_factor_is_one_on_true_latitudes():
    np.testing.assert_allclose(CHINA36.map_factor(np.array([30.0, 45.0])), 1.0, rtol=1e-12)
    assert CHINA36.map_factor(37.0) < 1.0  # between the parallels the cone lies inside the sphere


def test_single_true_latitude_cone():
    p = LambertConformal(40.0, -97.0, 40.0, 40.0)
    assert p.cone == pytest.approx(np.sin(np.radians(40.0)))
    np.testing.assert_allclose(p.map_factor(40.0), 1.0, rtol=1e-12)


def test_southern_hemisphere_round_trip():
    p = LambertConformal(-30.0, 135.0, -20.0, -40.0)
    assert p.cone < 0
    lon, lat = p.cell_centers(5, 4, 50000.0)
    x, y = p.to_xy(lon, lat)
    np.testing.assert_allclose(np.diff(x, axis=0), 50000.0, rtol=1e-9)
    np.testing.assert_allclose(np.diff(y, axis=1), 50000.0, rtol=1e-9)


def test_cell_centres_and_corners_are_on_the_grid_lattice():
    lon, lat = CHINA36.cell_centers(7, 5, 36000.0)
    lon_e, lat_e = CHINA36.cell_corners(7, 5, 36000.0)
    x, _ = CHINA36.to_xy(lon, lat)
    xe, ye = CHINA36.to_xy(lon_e, lat_e)
    np.testing.assert_allclose(x[:, 0], (np.arange(7) - 3) * 36000.0, atol=1e-6)
    np.testing.assert_allclose(xe[:, 0], (np.arange(8) - 3.5) * 36000.0, atol=1e-6)
    np.testing.assert_allclose(ye[0], (np.arange(6) - 2.5) * 36000.0, atol=1e-6)
    assert lon[3, 2] == pytest.approx(107.0) and lat[3, 2] == pytest.approx(37.0)


def test_grid_from_projection():
    g = Grid.from_projection(7, 5, 3, 36000.0, CHINA36)
    assert g.projection == CHINA36 and g.has_centers and g.has_edges
    assert g.lon_edges.shape == (8, 6)
    assert not Grid.from_projection(7, 5, 3, 36000.0, CHINA36, centers=False).has_centers
    bare = Grid.from_projection(7, 5, 3, 36000.0, CHINA36, centers=False, edges=False)
    assert bare.projection == CHINA36 and not bare.has_edges


def test_grid_projection_in_equality_and_repr():
    a = Grid(nx=4, ny=3, nz=2, dx=1.0, projection=CHINA36)
    b = Grid(nx=4, ny=3, nz=2, dx=1.0, projection=LambertConformal(37, 107, 30, 45))
    c = Grid(nx=4, ny=3, nz=2, dx=1.0)
    assert a == b and hash(a) == hash(b) and a != c
    assert "projection=LambertConformal(cen_lat=37.0" in repr(a)
    with pytest.raises(TypeError):
        Grid(nx=4, ny=3, nz=2, dx=1.0, projection=(37, 107, 30, 45))


def test_symmetric_crop_keeps_centre():
    g = Crop(5, 5, 5, 5).grid(179, 123, 35, 36000.0, projection=CHINA36)
    assert g.projection is CHINA36


def test_asymmetric_crop_moves_centre_consistently():
    full = Grid.from_projection(20, 16, 1, 36000.0, CHINA36)
    crop = Crop(ixb=3, ixbe=1, jxb=0, jxbe=4)
    g = crop.grid(20, 16, 1, 36000.0, lon=full.lon, lat=full.lat,
                  lon_edges=full.lon_edges, lat_edges=full.lat_edges, projection=CHINA36)
    assert g.projection.stand_lon == CHINA36.stand_lon
    assert g.projection.cen_lat != CHINA36.cen_lat
    # The recomputed projection regenerates exactly the cropped coordinates.
    again = Grid.from_projection(g.nx, g.ny, 1, 36000.0, g.projection)
    np.testing.assert_allclose(again.lon, g.lon, atol=1e-9)
    np.testing.assert_allclose(again.lat_edges, g.lat_edges, atol=1e-9)


@pytest.mark.skipif(not WRF_SAMPLE.exists(), reason="sample WRF file not mounted")
def test_matches_wrf_china36_sample():
    from scipy.io import netcdf_file

    with netcdf_file(WRF_SAMPLE, "r", mmap=True) as f:
        xlat = np.array(f.variables["XLAT"][0], dtype=np.float64).T
        xlon = np.array(f.variables["XLONG"][0], dtype=np.float64).T
        mapfac = np.array(f.variables["MAPFAC_M"][0], dtype=np.float64).T
    g = Grid.from_projection(179, 123, 35, 36000.0, CHINA36, edges=False)
    # WRF stores these as float32 (~1e-5 degree resolution at these magnitudes).
    np.testing.assert_allclose(g.lon, xlon, atol=5e-5)
    np.testing.assert_allclose(g.lat, xlat, atol=5e-5)
    np.testing.assert_allclose(CHINA36.map_factor(xlat), mapfac, rtol=1e-6)
