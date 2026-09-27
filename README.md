<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/banner-light.svg">
  <img alt="Chem_Grid" src="assets/banner-light.svg" width="860">
</picture>

<p>
  <a href="LICENSE"><img alt="license: MPL 2.0" src="https://img.shields.io/badge/license-MPL%202.0-6366F1?style=flat-square"></a>
  <img alt="python: 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?style=flat-square&logo=python&logoColor=white">
  <img alt="works with: JAX jit + grad" src="https://img.shields.io/badge/works%20with-JAX%20jit%20%2B%20grad-0EA5E9?style=flat-square">
  <img alt="used by: SMVGEAR · MINT · DepConv" src="https://img.shields.io/badge/used%20by-SMVGEAR%20·%20MINT%20·%20DepConv-9333EA?style=flat-square">
</p>

<p>
  <a href="#-whats-inside"><b>What's inside</b></a> ·
  <a href="#-regridding"><b>Regridding</b></a> ·
  <a href="#-tests"><b>Tests</b></a>
</p>

</div>

`chem_grid` is the small shared package every JAX chemistry package in this
repository depends on for grid geometry. It holds only what all of them agree
on; anything a single operator needs stays in that operator's own package.

Used by [MINT](https://github.com/fzhao70/MINT) (`mint.Grid` is `chem_grid.Grid`;
`topk` is a `mint.mint` argument), Chem_DepConv (vertical-order helpers) and
[SMVGEAR](https://github.com/fzhao70/SMVGEAR) (`ChemistryOperator(grid=...)`).

## 🧩 What's inside

| Name | What it is |
|---|---|
| `Grid(nx, ny, nz, dx)` | Static domain dimensions and REAM's scalar grid spacing. Frozen and hashable, so it works as a `jax.jit` static argument. Arrays are laid out `(nx, ny, nz, ...)`, x first. |
| `Grid(..., lon=, lat=, lon_edges=, lat_edges=)` | Optional coordinates in degrees, `None` unless assigned: cell centres `(nx, ny)` and cell corners `(nx+1, ny+1)`, each given as a lon/lat pair. Stored as read-only copies; they take part in `==` and `hash`, so the grid stays a valid static argument. The corners plug straight into `conservative_weights`. |
| `LambertConformal(cen_lat, cen_lon, truelat1, truelat2, stand_lon=None)` | WRF's Lambert conformal projection (`CEN_LAT`, `CEN_LON`, `TRUELAT1`, `TRUELAT2`, `STAND_LON`; REAM's `phic`, `xlonc`, `truelat01`, `truelat02`). `stand_lon` defaults to `cen_lon`, as in REAM. Forward/inverse transforms, `map_factor`, `cell_centers`/`cell_corners`, and `from_wrf_attrs`. Pass it as `Grid(..., projection=...)`, or use `Grid.from_projection(nx, ny, nz, dx, proj)` to also fill the coordinates. For the China 36 km domain it reproduces WRF's `XLAT`/`XLONG` to float32 precision and `MAPFAC_M` to 4e-7. |
| `VerticalOrder`, `reorder`, `to_chemistry_order`, `to_model_order` | REAM's two vertical conventions. **Model order (`k=0` is the model top) is the public order across packages**; code with chemistry-order internals flips on entry and back on exit. |
| `layer_dp(psfc, c3f, c4f=None, ptop=1000.0)` | REAM's layer pressure thickness `DP` in mb, **model order**, from surface pressure (Pa) and WRF's `C3F`/`C4F` full-level coefficients (surface first). File-free; numpy or jax, works under `jit` and `grad`. The pressure tendency is the difference of two calls: `dpdt = (layer_dp(ps_next, ...) - layer_dp(ps, ...)) / interval` (REAM's `DPDT`). |
| `Crop(ixb, ixbe, jxb, jxbe)` | REAM's lateral crop of the WRF grid onto the model grid (`constants.cpp:598-615`); `Crop.grid(...)` builds the model `Grid` (cropping any full-grid coordinates you pass, and recomputing the projection's centre if the crop is asymmetric), `Crop.apply(a)` crops a field (staggered fields come out one longer, as in REAM). |
| `chem_grid.regrid` | Bilinear (`bilinear_weights`) and first-order conservative (`conservative_weights`) horizontal regridding. **Not used by any package yet**: REAM's inputs already sit on the model grid. |

Not in `Grid`, on purpose: transport's `topk` and `g_p2m` (transport
options), and geometry that has to be a traced array, such as the map-scale
factor, sigma levels and `ptop`.

## 🌐 Regridding

```python
from chem_grid.regrid import conservative_weights, bilinear_weights, cell_areas

w = conservative_weights(lon_edges, lat_edges, xlong_corners, xlat_corners)  # built once, numpy
flux_model = w(flux_inventory)   # (nlon, nlat, ...) -> (nx, ny, ...), JAX, jit/grad-safe
```

- Weights are built on the host in numpy and applied in JAX as a gather plus
  a segment sum; `RegridWeights` is a pytree. No ESMF/xESMF dependency.
- Source fields are longitude-first, `(nlon, nlat, ...)`, like the model's
  `(nx, ny, ...)`. Transpose `(lat, lon)` file arrays first.
- **Conservative**: both grids are given as cell corners (1-D edges for a
  rectilinear grid, `(n+1, m+1)` arrays for a curvilinear one). Geometry is
  done in the equal-area `(lon, sin lat)` plane: exact for lon/lat cells,
  straight-edge approximation otherwise, and exactly conservative with
  respect to `cell_areas`. `normalize="destarea"` (default) conserves the
  area integral; `"fracarea"` gives the mean over the covered part.
  Use it for per-area quantities (fluxes, concentrations).
- **Bilinear**: source is 1-D cell centres (latitudes may descend), targets
  are any lon/lat points. Handles global longitude wrap and either longitude
  convention; points outside the source are NaN (`outside="nan"`) or clamped
  (`outside="nearest"`). Not mass-conserving.
- Uncovered destination cells get `fill_value` (NaN by default).

Building conservative weights for a 169x113 curvilinear grid from a 0.1°
inventory takes about 3 s; bilinear takes a few milliseconds.

## 🧪 Tests

```bash
JAX_PLATFORMS=cpu python -m pytest -q     # 89 tests, ~15 s
```

`tests/test_vertical.py` is ported from Chem_DepConv's `tests/test_grid.py`.

## 🔗 Part of the JaxREAM family

`chem_grid` is one of the standalone libraries that [JaxREAM](https://github.com/fzhao70/JaxREAM) couples into a full regional chemical transport model. Each library works on its own, with arrays in and arrays out.

<table><tr>
<td align="center" width="96"><a href="https://github.com/fzhao70/JaxREAM"><img src="assets/modules/jaxream.svg" width="40" alt="jaxream"><br><b>jaxream</b></a><br>coupler</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/SMVGEAR"><img src="assets/modules/smvgear.svg" width="40" alt="smvgear"><br><b>smvgear</b></a><br>chemistry</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/MINT"><img src="assets/modules/mint.svg" width="40" alt="mint"><br><b>mint</b></a><br>transport</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/DepConv"><img src="assets/modules/depconv.svg" width="40" alt="depconv"><br><b>depconv</b></a><br>deposition</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/FastJ"><img src="assets/modules/fastj.svg" width="40" alt="fastj"><br><b>fastj</b></a><br>photolysis</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/EMIS_GEOS"><img src="assets/modules/emis_geos.svg" width="40" alt="emis_geos"><br><b>emis_geos</b></a><br>emissions</td>
<td align="center" width="96"><a href="https://github.com/fzhao70/Chem_Grid"><img src="assets/modules/chem_grid.svg" width="40" alt="chem_grid"><br><b>chem_grid</b></a><br><i>this repo</i></td>
</tr></table>

## 📄 License

Mozilla Public License 2.0; see [LICENSE](LICENSE).

<div align="center">
<br>
<a href="https://github.com/fzhao70/JaxREAM"><img src="assets/logo.svg" width="56" alt="chem_grid in the JaxREAM family"></a>
<br>
<sub>Part of the <a href="https://github.com/fzhao70/JaxREAM">JaxREAM</a> family of modular JAX atmospheric-chemistry libraries</sub>
</div>
