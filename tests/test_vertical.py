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


# ---------------------------------------------------------------- layer_dp

from chem_grid import layer_dp  # noqa: E402


def _hybrid(nz=6):
    """A made-up hybrid coordinate, surface first: C3F from 1 to 0, and a C4F
    that is zero at both ends, as WRF's is."""
    c3f = np.linspace(1.0, 0.0, nz + 1) ** 1.3
    c4f = 3000.0 * np.sin(np.pi * np.linspace(0.0, 1.0, nz + 1))
    return c3f, c4f


def test_layer_dp_column_sums_to_surface_minus_top():
    psfc = 90000.0 + 12000.0 * np.random.default_rng(0).random((5, 4))
    c3f, c4f = _hybrid()
    dp = layer_dp(psfc, c3f, c4f, ptop=1000.0)
    assert dp.shape == (5, 4, 6)
    np.testing.assert_allclose(dp.sum(-1), (psfc - 1000.0) / 100.0, rtol=1e-13)


def test_layer_dp_matches_ream_formula_in_model_order():
    """PRF = C3F*(PS - PTOP) + C4F + PTOP (read_wrf.cpp:1382), differenced;
    the surface layer is the LAST entry (model order)."""
    psfc = np.array([[101325.0]])
    c3f, c4f = _hybrid()
    prf = c3f * (psfc[..., None] - 1000.0) + c4f + 1000.0
    expect_surface_first = (prf[..., :-1] - prf[..., 1:]) * 0.01
    got = layer_dp(psfc, c3f, c4f)
    np.testing.assert_allclose(got, expect_surface_first[..., ::-1], rtol=1e-14)
    assert got[0, 0, -1] == pytest.approx(expect_surface_first[0, 0, 0])


def test_layer_dp_sigma_default_is_even_split():
    nz = 8
    dp = layer_dp(np.full((2, 3), 100000.0), np.linspace(1.0, 0.0, nz + 1))
    np.testing.assert_allclose(dp, (100000.0 - 1000.0) / 100.0 / nz, rtol=1e-13)


def test_layer_dp_keeps_array_type_and_works_under_jit_and_grad():
    c3f, c4f = _hybrid()
    psfc = np.full((3, 2), 95000.0)
    assert isinstance(layer_dp(psfc, c3f, c4f), np.ndarray)
    out = jax.jit(lambda p: layer_dp(p, c3f, c4f))(jnp.asarray(psfc))
    assert isinstance(out, jax.Array)
    np.testing.assert_allclose(np.asarray(out), layer_dp(psfc, c3f, c4f), rtol=1e-12)
    # d(column sum)/d(psfc) = 1/100 mb per Pa, whatever the coefficients
    g = jax.grad(lambda p: layer_dp(p, c3f, c4f).sum())(jnp.asarray(psfc))
    np.testing.assert_allclose(np.asarray(g), 0.01, rtol=1e-12)


def test_layer_dp_tendency_is_the_difference_of_two_calls():
    c3f, c4f = _hybrid()
    p0, p1, interval = np.full((2, 2), 95000.0), np.full((2, 2), 95180.0), 1800.0
    dpdt = (layer_dp(p1, c3f, c4f) - layer_dp(p0, c3f, c4f)) / interval
    # the column tendency is the surface-pressure tendency, 180 Pa in 30 min
    np.testing.assert_allclose(dpdt.sum(-1), 1.8 / interval, rtol=1e-12)


def test_layer_dp_validates_coefficients():
    with pytest.raises(ValueError, match="c3f"):
        layer_dp(np.ones((2, 2)), np.ones((2, 3)))
    with pytest.raises(ValueError, match="c4f"):
        layer_dp(np.ones((2, 2)), np.linspace(1, 0, 5), np.zeros(4))
