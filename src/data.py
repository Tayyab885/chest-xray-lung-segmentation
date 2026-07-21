"""Manifest construction and image/mask loading.

The three sources ship in different directory layouts, so the layout is data
(configs/data_layout.yaml) rather than code. Adding a fourth source means
adding a config block, not editing this module.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

MANIFEST_COLUMNS = ["image_path", "mask_paths", "source", "patient_id"]


def build_manifest(data_root, layout):
    """Scan every configured source and return one row per annotated image.

    Raises if a source yields a different image count than configured, or if an
    image is missing a mask the layout did not say to expect. A silently
    dropped image would change the denominator of every metric.

    A source may set `expected_unmasked` when the mirror ships radiographs
    without segmentations, as the Shenzhen one does: 662 images against 566
    masks. Those images cannot be segmented or scored, so the study is the
    annotated subset. The count is declared rather than inferred so that the
    composition is pinned from both sides, and `expected_count` is then checked
    after the drop because it is the number that enters the study.
    """
    root = Path(data_root)
    rows = []

    for source, spec in layout.items():
        images = sorted(root.glob(spec["image_glob"]))
        allowed_unmasked = spec.get("expected_unmasked")
        kept, unmasked = [], []

        for img in images:
            masks = [root / pattern.format(stem=img.stem)
                     for pattern in spec["mask_globs"]]
            missing = [m for m in masks if not m.exists()]
            if missing:
                if allowed_unmasked is None:
                    raise FileNotFoundError(f"{source}: no mask at {missing[0]}")
                unmasked.append(img)
                continue
            kept.append((img, masks))

        if allowed_unmasked is not None and len(unmasked) != allowed_unmasked:
            raise ValueError(
                f"{source}: {len(unmasked)} images have no mask, config expects "
                f"expected_unmasked={allowed_unmasked}. The mirror's composition "
                "has changed, so the study is not the one this config describes."
            )
        if unmasked:
            print(f"{source}: dropped {len(unmasked)} radiographs with no "
                  f"segmentation, keeping {len(kept)}")

        expected = spec.get("expected_count")
        if expected is not None and len(kept) != expected:
            raise ValueError(
                f"{source}: found {len(kept)} annotated images, config expects "
                f"{expected}. Check the download, then update expected_count if "
                "the config is wrong."
            )

        for img, masks in kept:
            rows.append({
                "image_path": str(img),
                "mask_paths": [str(m) for m in masks],
                "source": source,
                "patient_id": f"{source}/{img.stem}",
            })

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

    duplicates = manifest.loc[
        manifest["patient_id"].duplicated(keep=False), "patient_id"
    ].unique()
    if len(duplicates):
        raise ValueError(
            f"repeated patient_id: {sorted(duplicates)}. Every source is supposed to "
            "hold one image per patient, which is what makes an image split a patient "
            "split. Two rows under one ID would put the same patient in both halves."
        )

    return manifest


def load_image(image_path, size):
    """Load a radiograph as a z-scored float32 array of shape (size, size)."""
    img = Image.open(image_path).convert("L").resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    std = arr.std()
    if std < 1e-6:
        return np.zeros_like(arr)
    return (arr - arr.mean()) / std


def load_mask(mask_paths, size):
    """Merge one or more mask files into a single boolean lung field mask."""
    merged = None
    for path in mask_paths:
        m = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
        arr = np.asarray(m) > 127
        merged = arr if merged is None else np.logical_or(merged, arr)
    return merged
