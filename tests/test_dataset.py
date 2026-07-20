import numpy as np
import pytest
import torch

from src.dataset import LungDataset, augment_pair


@pytest.fixture
def frame(fake_frame):
    return fake_frame(1, source="x")


def test_dataset_length(frame):
    assert len(LungDataset(frame, size=32)) == 1


def test_item_shapes_and_dtypes(frame):
    img, mask = LungDataset(frame, size=32)[0]
    assert img.shape == (1, 32, 32)
    assert mask.shape == (1, 32, 32)
    assert img.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_mask_stays_binary(frame):
    _, mask = LungDataset(frame, size=32)[0]
    assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_augmentation_keeps_mask_binary(frame):
    ds = LungDataset(frame, size=32, augment=True)
    for _ in range(10):
        _, mask = ds[0]
        assert set(torch.unique(mask).tolist()).issubset({0.0, 1.0})


def test_augmentation_changes_the_image(frame):
    plain = LungDataset(frame, size=32, augment=False)[0][0]
    aug = LungDataset(frame, size=32, augment=True)
    assert any(not torch.allclose(plain, aug[0][0]) for _ in range(10))


def test_fixture_mask_is_not_its_own_mirror(frame):
    # Guards the guard: if this fixture ever becomes symmetric again, every test
    # built on it goes blind to a horizontal flip.
    _, mask = LungDataset(frame, size=32)[0]
    assert not torch.equal(mask, torch.flip(mask, dims=[-1]))


def test_dataset_preserves_left_right_orientation(frame):
    # The fixture's left blob is wider than its right one. Anything that mirrors
    # the pair on the way out of __getitem__ inverts that, which is the failure
    # the no-flip rule exists to prevent.
    _, mask = LungDataset(frame, size=32)[0]
    left, right = mask[..., :16].sum(), mask[..., 16:].sum()
    assert left > right, "left/right orientation inverted, this looks like a flip"


def test_augmentation_keeps_image_and_mask_aligned():
    # One transform is drawn and applied to both arrays. If image and mask ever
    # drew separate parameters, the mask would slide off the anatomy and only
    # show up much later as an unexplained Dice drop, never as a failing test.
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 8:24] = True
    img = mask.astype(np.float32)  # image landmark sits exactly on the mask
    rng = np.random.default_rng(0)
    for _ in range(20):
        img_out, mask_out = augment_pair(img, mask, rng)
        if mask_out.sum() == 0:
            continue
        bright = img_out > 0.5 * img_out.max()
        overlap = np.logical_and(bright, mask_out).sum()
        assert overlap / mask_out.sum() > 0.9, "mask drifted away from the image"


def test_worker_init_gives_each_worker_its_own_stream(frame, monkeypatch):
    # Workers are copies of the dataset, so without a reseed they all share one
    # generator state: same rotation drawn for different images, same sequence
    # replayed every epoch.
    import src.dataset as dataset_module

    class FakeInfo:
        def __init__(self, ds, seed):
            self.dataset = ds
            self.seed = seed

    def run(seed):
        ds = LungDataset(frame, size=32, augment=True)
        monkeypatch.setattr(dataset_module, "get_worker_info", lambda: FakeInfo(ds, seed))
        dataset_module.worker_init_fn(0)
        return ds[0][0]

    assert not torch.allclose(run(1), run(2))


def test_augmentation_never_mirrors_left_right():
    """A horizontal flip would move the mask's center of mass across the midline.

    The mask here is heavier on the left. Augmentation may shift it, but it must
    never land on the mirrored position, because flipping a chest radiograph
    puts the heart on the wrong side.
    """
    img = np.zeros((64, 64), dtype=np.float32)
    mask = np.zeros((64, 64), dtype=bool)
    mask[20:44, 4:20] = True   # entirely left of the midline
    rng = np.random.default_rng(0)
    for _ in range(50):
        _, m = augment_pair(img, mask, rng)
        if m.sum() == 0:
            continue
        centroid_x = np.argwhere(m)[:, 1].mean()
        assert centroid_x < 32, "mask crossed the midline, this looks like a flip"
