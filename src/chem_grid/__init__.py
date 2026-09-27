"""Shared grid geometry for the JAX chemistry packages.

* :class:`Grid` -- static domain dimensions and grid spacing.
* :class:`VerticalOrder`, :func:`reorder` -- REAM's model-order vs
  chemistry-order vertical conventions; model order (``k=0`` at the model top)
  is the public order across packages.
* :class:`Crop` -- REAM's lateral crop of the met grid onto the model grid.
* :mod:`chem_grid.regrid` -- bilinear and conservative horizontal regridding.
"""

from chem_grid.crop import Crop
from chem_grid.grid import Grid
from chem_grid.vertical import VerticalOrder, reorder, to_chemistry_order, to_model_order

__all__ = [
    "Crop",
    "Grid",
    "VerticalOrder",
    "reorder",
    "to_chemistry_order",
    "to_model_order",
]

__version__ = "0.1.0"
