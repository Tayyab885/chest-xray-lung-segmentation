import numpy as np
import pytest

from src.metrics import all_metrics, assd, dice, hd95, iou


def square(size=64, top=10, left=10, side=20):
    m = np.zeros((size, size), dtype=bool)
    m[top:top + side, left:left + side] = True
    return m


def test_identical_masks_are_perfect():
    a = square()
    assert dice(a, a) == pytest.approx(1.0)
    assert iou(a, a) == pytest.approx(1.0)
    assert hd95(a, a) == pytest.approx(0.0)
    assert assd(a, a) == pytest.approx(0.0)


def test_disjoint_masks_score_zero_overlap():
    a = square(left=0, side=10)
    b = square(left=40, side=10)
    assert dice(a, b) == pytest.approx(0.0)
    assert iou(a, b) == pytest.approx(0.0)


def test_dice_of_half_overlapping_squares_is_exactly_one_half():
    # Two 10x20 squares sharing exactly half their area.
    a = np.zeros((64, 64), dtype=bool)
    b = np.zeros((64, 64), dtype=bool)
    a[10:30, 10:20] = True   # 200 px
    b[20:40, 10:20] = True   # 200 px, overlap is 100 px
    # dice = 2*100 / (200+200) = 0.5, iou = 100 / 300
    assert dice(a, b) == pytest.approx(0.5)
    assert iou(a, b) == pytest.approx(1.0 / 3.0)


def test_hd95_of_shifted_square_is_exactly_the_shift():
    a = square(left=10, side=20)
    b = square(left=15, side=20)  # shifted 5 px in x
    assert hd95(a, b) == pytest.approx(5.0)


def test_assd_is_smaller_than_hd95_for_a_shift():
    a = square(left=10, side=20)
    b = square(left=15, side=20)
    assert assd(a, b) < hd95(a, b)


def test_empty_masks():
    empty = np.zeros((32, 32), dtype=bool)
    a = square(size=32, top=4, left=4, side=8)
    assert dice(empty, empty) == pytest.approx(1.0)
    assert dice(empty, a) == pytest.approx(0.0)
    assert np.isnan(hd95(empty, a))
    assert np.isnan(assd(empty, a))


def test_all_metrics_returns_every_key():
    a = square()
    out = all_metrics(a, a)
    assert set(out) == {"dice", "iou", "hd95", "assd"}


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4), dtype=bool), np.zeros((5, 5), dtype=bool))


def test_non_boolean_float_input_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4), dtype=float), np.zeros((4, 4), dtype=bool))


def test_uint8_input_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 4), dtype=np.uint8))


def test_3d_input_raises():
    with pytest.raises(ValueError):
        dice(np.zeros((1, 4, 4), dtype=bool), np.zeros((1, 4, 4), dtype=bool))


def test_hd95_and_assd_are_symmetric_for_an_asymmetric_pair():
    a = square()  # plain 20x20 square
    b = a.copy()
    b[19, 30:50] = True  # same square plus a thin 20 px spike off its right edge
    assert hd95(a, b) == pytest.approx(hd95(b, a))
    assert assd(a, b) == pytest.approx(assd(b, a))


def test_hd95_reflects_the_spike_not_the_near_zero_reverse_distance():
    gt = square()
    pred = gt.copy()
    pred[19, 30:50] = True  # spike tip sits 20 px past the square's right edge
    # The gt -> pred direction reads near 0 here, since almost all of gt's
    # boundary sits exactly on pred's boundary too. Only the pred -> gt
    # direction sees the spike, so hd95 must take the max of both to land
    # near the spike length. The companion symmetry test is what actually
    # catches an implementation that keeps only one direction.
    assert hd95(pred, gt) > 10.0
