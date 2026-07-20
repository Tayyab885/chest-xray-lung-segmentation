"""Torch dataset and augmentation for lung field segmentation.

Augmentation is deliberately mild and deliberately excludes horizontal flip.
Mirroring a chest radiograph places the heart on the wrong side, which is an
anatomy the model should never learn.
"""
import numpy as np
import torch
from scipy.ndimage import affine_transform
from torch.utils.data import Dataset, get_worker_info

from src.data import load_image, load_mask

MAX_ROTATION_DEG = 10.0
SCALE_RANGE = (0.9, 1.1)
MAX_SHIFT_FRAC = 0.05
INTENSITY_SCALE_RANGE = (0.9, 1.1)
INTENSITY_SHIFT = 0.1


def _affine_matrix(angle_deg, scale, shape):
    theta = np.deg2rad(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ]) / scale
    center = np.array(shape) / 2.0
    offset = center - rot @ center
    return rot, offset


def augment_pair(image, mask, rng):
    """Apply the same geometric transform to image and mask, then jitter intensity."""
    angle = rng.uniform(-MAX_ROTATION_DEG, MAX_ROTATION_DEG)
    scale = rng.uniform(*SCALE_RANGE)
    shift = rng.uniform(-MAX_SHIFT_FRAC, MAX_SHIFT_FRAC, size=2) * np.array(image.shape)

    rot, offset = _affine_matrix(angle, scale, image.shape)
    offset = offset + shift

    image_out = affine_transform(
        image.astype(np.float32), rot, offset=offset, order=1, mode="constant", cval=0.0)
    mask_out = affine_transform(
        mask.astype(np.float32), rot, offset=offset, order=0, mode="constant", cval=0.0) > 0.5

    image_out = image_out * rng.uniform(*INTENSITY_SCALE_RANGE)
    image_out = image_out + rng.uniform(-INTENSITY_SHIFT, INTENSITY_SHIFT)

    return image_out.astype(np.float32), mask_out


class LungDataset(Dataset):
    def __init__(self, frame, size, augment=False, seed=42):
        self.frame = frame.reset_index(drop=True)
        self.size = size
        self.augment = augment
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, i):
        row = self.frame.iloc[i]
        image = load_image(row["image_path"], self.size)
        mask = load_mask(row["mask_paths"], self.size)

        if self.augment:
            image, mask = augment_pair(image, mask, self.rng)

        image_t = torch.from_numpy(np.ascontiguousarray(image)).float().unsqueeze(0)
        mask_t = torch.from_numpy(np.ascontiguousarray(mask)).float().unsqueeze(0)
        return image_t, mask_t


def worker_init_fn(worker_id):
    """Give every DataLoader worker its own augmentation stream.

    Workers are copies of the dataset object, so without this they all inherit
    one identical generator state: two workers would draw the same rotation for
    different images, and every epoch would replay the same sequence. Torch
    derives `info.seed` per worker and per epoch, which is exactly what is
    needed here. Pass this as `DataLoader(..., worker_init_fn=worker_init_fn)`
    whenever `num_workers > 0`.
    """
    info = get_worker_info()
    info.dataset.rng = np.random.default_rng(info.seed)
