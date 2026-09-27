"""Shared grid geometry for the JAX chemistry packages.

* :class:`Grid` -- static domain dimensions and grid spacing.
* :class:`VerticalOrder`, :func:`reorder` -- REAM's model-order vs
  chemistry-order vertical conventions; model order (``k=0`` at the model top)
  is the public order across packages.
* :func:`layer_dp` -- layer pressure thickness from surface pressure and
  WRF's ``C3F``/``C4F`` coefficients, REAM's ``DP``.
* :class:`LambertConformal` -- WRF's CEN_LAT/CEN_LON/TRUELAT1/TRUELAT2/STAND_LON
  projection, with forward/inverse transforms and cell centres/corners.
* :class:`Crop` -- REAM's lateral crop of the met grid onto the model grid.
* :mod:`chem_grid.regrid` -- bilinear and conservative horizontal regridding.
"""

from chem_grid.crop import Crop
from chem_grid.grid import Grid
from chem_grid.projection import LambertConformal
from chem_grid.vertical import (
    VerticalOrder,
    layer_dp,
    reorder,
    to_chemistry_order,
    to_model_order,
)

__all__ = [
    "Crop",
    "Grid",
    "LambertConformal",
    "VerticalOrder",
    "layer_dp",
    "reorder",
    "to_chemistry_order",
    "to_model_order",
]

__version__ = "0.1.0"
