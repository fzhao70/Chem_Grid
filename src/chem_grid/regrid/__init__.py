"""Horizontal regridding: bilinear interpolation and first-order conservative remapping.

Weights are built once on the host in numpy and applied in JAX
(:class:`RegridWeights` is a pytree; applying it works under ``jax.jit`` and
``jax.grad``). No external regridding library is needed.

Not used by any package yet: REAM's inputs already sit on the model grid, so
the current pipeline only crops (:class:`chem_grid.Crop`). These are here for
inputs that arrive on a different grid (a lat/lon emission inventory, say).
"""

from chem_grid.regrid.bilinear import bilinear_weights, is_periodic_lon
from chem_grid.regrid.conservative import cell_areas, conservative_weights, rectilinear_corners
from chem_grid.regrid.weights import RegridWeights

__all__ = [
    "RegridWeights",
    "bilinear_weights",
    "cell_areas",
    "conservative_weights",
    "is_periodic_lon",
    "rectilinear_corners",
]
