"""REAM's two vertical index conventions, and the one sanctioned way to flip.

REAM runs the same column in two different vertical orders at once. ``Q``,
``XSTT``, ``DZQ``, ``DEN``, ``TB``, ``CLW2A``, ``RNW2A``, ``DP`` and every
KF-eta field are **model order**: index ``0`` is the model top. ``STT``,
``BXHEIGHT``, ``AIRVOL``, ``KZDIFF`` and ``DEPSAV`` are **chemistry order**:
index ``0`` is the surface. The flip between the two is explicit in REAM
itself -- ``chemdr.cpp:637`` fills the chemistry-order ``BXHEIGHT`` from the
model-order ``XBXHEIGHT`` with a reversed index, and ``chemdr.cpp:831``
reverses it back when copying ``STT`` into ``XSTT``. Both flips are a plain
index reversal over the vertical axis.

This is the main way a port can fail silently: an array handed across a
module boundary in the wrong order still has the right ``nz``, it is just
upside down, and no shape check catches it.

**Contract shared by every package**: the public vertical order is **model
order** (``k=0`` is the model top) everywhere a vertical axis crosses a
package boundary. A package whose internals follow REAM's chemistry order
(``depconv.drydep``, for one) flips on entry and back on exit with
:func:`to_chemistry_order` / :func:`to_model_order`, and nothing outside it
ever sees a chemistry-order array.
"""

from __future__ import annotations

import enum
from typing import TypeVar

__all__ = [
    "VerticalOrder",
    "reorder",
    "to_chemistry_order",
    "to_model_order",
]

ArrayT = TypeVar("ArrayT")


class VerticalOrder(enum.Enum):
    """Which end of the vertical axis is index ``0``.

    REAM keeps both orders live simultaneously on the same column (``chemdr.cpp:637-639`` and ``:831``):

    * ``MODEL`` -- index ``0`` is the model top. REAM's ``Q``, ``XSTT``,
      ``DZQ``, ``DEN``, ``TB``, ``CLW2A``, ``RNW2A``, ``DP`` and every KF-eta
      field. This is the public order across packages.
    * ``CHEMISTRY`` -- index ``0`` is the surface. REAM's ``STT``,
      ``BXHEIGHT``, ``AIRVOL``, ``KZDIFF``, ``DEPSAV``. Internal to
      packages such as ``depconv.drydep``.

    The two members carry no data; they exist so a flip is always written as
    ``reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=...)``
    rather than a bare, unexplained ``[..., ::-1]``.
    """

    MODEL = "model"
    CHEMISTRY = "chemistry"


def reorder(x: ArrayT, frm: VerticalOrder, to: VerticalOrder, axis: int = -1) -> ArrayT:
    """Convert *x* from vertical order *frm* to vertical order *to*.

    A no-op when ``frm is to``; otherwise a reversal of *x* along *axis*,
    which is the whole of the flip REAM performs at ``chemdr.cpp:637`` and
    ``:831`` (see the module docstring). Implemented with a reversed basic
    slice rather than ``numpy.flip`` / ``jax.numpy.flip`` so the input's array
    type is preserved -- a numpy array in gives a numpy array out, a jax array
    in gives a jax array out -- and so it is safe to call from inside
    ``jax.jit``: *frm*, *to* and *axis* are ordinary Python values resolved at
    trace time (this is static branching on Python objects, not a
    data-dependent branch on array contents), and the reversed slice lowers to
    ``lax.rev``, which every jax transform (``jit``, ``grad``, ``vmap``)
    supports.

    Parameters
    ----------
    x :
        Array with a vertical axis of length ``nz`` at position *axis*. Numpy
        or jax array.
    frm, to :
        The order *x* is currently in, and the order to convert it to.
    axis :
        Position of the vertical axis in *x*. Defaults to the last axis.
    """
    if frm is to:
        return x
    ndim = x.ndim
    ax = axis if axis >= 0 else ndim + axis
    if not 0 <= ax < ndim:
        raise ValueError(f"axis {axis} out of range for array with ndim={ndim}")
    index = (slice(None),) * ax + (slice(None, None, -1),)
    return x[index]


def to_chemistry_order(x: ArrayT, axis: int = -1) -> ArrayT:
    """Flip a model-order array (the public order) to chemistry order.

    Used on the way into code whose internals mirror REAM's chemistry-order
    arrays (``depconv.drydep``'s ``DEPVEL`` / ``DEPRATE``, for example).
    """
    return reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=axis)


def to_model_order(x: ArrayT, axis: int = -1) -> ArrayT:
    """Flip a chemistry-order array back to model order (the public order).

    Used on the way out of chemistry-order internals, before the array is
    handed back across a package boundary.
    """
    return reorder(x, VerticalOrder.CHEMISTRY, VerticalOrder.MODEL, axis=axis)
