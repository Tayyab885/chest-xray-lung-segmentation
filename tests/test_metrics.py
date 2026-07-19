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


def test_hd95_of_shifted_square_is_about_the_shift():
    a = square(left=10, side=20)
    b = square(left=15, side=20)  # shifted 5 px in x
    assert hd95(a, b) == pytest.approx(5.0, abs=1.0)


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
