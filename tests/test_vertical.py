"""Tests for chem_grid.vertical, ported from Chem_DepConv's tests/test_grid.py."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chem_grid import VerticalOrder, reorder, to_chemistry_order, to_model_order

# ---------------------------------------------------------------------------
# VerticalOrder
# ---------------------------------------------------------------------------


def test_vertical_order_has_two_members():
    assert {m.name for m in VerticalOrder} == {"MODEL", "CHEMISTRY"}


def test_vertical_order_round_trips_by_value():
    assert VerticalOrder("model") is VerticalOrder.MODEL
    assert VerticalOrder("chemistry") is VerticalOrder.CHEMISTRY


def test_vertical_order_members_distinct():
    assert VerticalOrder.MODEL is not VerticalOrder.CHEMISTRY
    assert VerticalOrder.MODEL != VerticalOrder.CHEMISTRY


# ---------------------------------------------------------------------------
# reorder: correctness
# ---------------------------------------------------------------------------


def test_reorder_same_order_is_identity_object():
    x = np.arange(5.0)
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.MODEL)
    assert out is x


def test_reorder_flips_last_axis_by_default():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY)
    np.testing.assert_array_equal(out, x[::-1])


def test_reorder_is_involution_numpy():
    x = np.arange(24.0).reshape(2, 3, 4)
    once = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-1)
    twice = reorder(once, VerticalOrder.CHEMISTRY, VerticalOrder.MODEL, axis=-1)
    np.testing.assert_array_equal(twice, x)


def test_reorder_is_involution_jax():
    x = jnp.arange(24.0).reshape(2, 3, 4)
    once = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-1)
    twice = reorder(once, VerticalOrder.CHEMISTRY, VerticalOrder.MODEL, axis=-1)
    np.testing.assert_array_equal(np.asarray(twice), np.asarray(x))


def test_reorder_on_explicit_axis():
    x = np.arange(24.0).reshape(4, 3, 2)  # vertical axis is axis 0, length 4
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=0)
    np.testing.assert_array_equal(out, x[::-1, :, :])


def test_reorder_leaves_other_axes_untouched():
    x = np.arange(24.0).reshape(2, 3, 4)
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-1)
    assert out.shape == x.shape


def test_reorder_negative_axis_matches_positive_equivalent():
    x = np.arange(24.0).reshape(4, 3, 2)
    out_neg = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-3)
    out_pos = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=0)
    np.testing.assert_array_equal(out_neg, out_pos)


def test_reorder_rejects_out_of_range_axis():
    x = np.arange(6.0).reshape(2, 3)
    with pytest.raises(ValueError):
        reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=5)


# ---------------------------------------------------------------------------
# reorder: array-type preservation (numpy in -> numpy out, jax in -> jax out)
# ---------------------------------------------------------------------------


def test_reorder_preserves_numpy_type():
    x = np.arange(4.0)
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY)
    assert isinstance(out, np.ndarray)


def test_reorder_preserves_jax_type():
    x = jnp.arange(4.0)
    out = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY)
    assert isinstance(out, jax.Array)


# ---------------------------------------------------------------------------
# reorder: jit-safety
# ---------------------------------------------------------------------------


def test_reorder_under_jit_matches_eager():
    x = jnp.arange(24.0).reshape(2, 3, 4)

    def flip(y):
        return reorder(y, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-1)

    eager = flip(x)
    jitted = jax.jit(flip)(x)
    np.testing.assert_array_equal(np.asarray(eager), np.asarray(jitted))


def test_reorder_jit_with_static_order_args():
    def flip(y, frm, to, axis):
        return reorder(y, frm, to, axis=axis)

    flip = jax.jit(flip, static_argnums=(1, 2, 3))
    x = jnp.arange(8.0).reshape(2, 4)
    out = flip(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, -1)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(x[:, ::-1]))


def test_reorder_grad_compatible():
    # Confirms the flip lowers to something autodiff (and hence jit) accepts.
    x = jnp.arange(4.0)

    def loss(y):
        flipped = reorder(y, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY)
        return jnp.sum(flipped * jnp.arange(4.0))

    grad = jax.grad(loss)(x)
    # d/dy_i [sum_j flipped(y)_j * c_j] = c_{reverse(i)}
    expected = jnp.arange(4.0)[::-1]
    np.testing.assert_array_equal(np.asarray(grad), np.asarray(expected))


# ---------------------------------------------------------------------------
# to_chemistry_order / to_model_order convenience aliases
# ---------------------------------------------------------------------------


def test_to_chemistry_order_matches_reorder():
    x = np.arange(24.0).reshape(2, 3, 4)
    expected = reorder(x, VerticalOrder.MODEL, VerticalOrder.CHEMISTRY, axis=-1)
    np.testing.assert_array_equal(to_chemistry_order(x), expected)


def test_to_model_order_matches_reorder():
    x = np.arange(24.0).reshape(2, 3, 4)
    expected = reorder(x, VerticalOrder.CHEMISTRY, VerticalOrder.MODEL, axis=-1)
    np.testing.assert_array_equal(to_model_order(x), expected)


def test_to_chemistry_then_to_model_is_involution():
    x = np.arange(10.0)
    round_trip = to_model_order(to_chemistry_order(x))
    np.testing.assert_array_equal(round_trip, x)


def test_to_chemistry_order_custom_axis():
    x = np.arange(24.0).reshape(4, 3, 2)
    out = to_chemistry_order(x, axis=0)
    np.testing.assert_array_equal(out, x[::-1, :, :])
