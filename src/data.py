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
    """Scan every configured source and return one row per image.

    Raises if a source yields a different image count than configured, or if
    any image is missing one of its mask files. A silently dropped image would
    change the denominator of every metric.
    """
    root = Path(data_root)
    rows = []

    for source, spec in layout.items():
        images = sorted(root.glob(spec["image_glob"]))
        expected = spec.get("expected_count")
        if expected is not None and len(images) != expected:
            raise ValueError(
                f"{source}: found {len(images)} images, config expects {expected}. "
                "Check the download, then update expected_count if the config is wrong."
            )

        for img in images:
            stem = img.stem
            masks = []
            for pattern in spec["mask_globs"]:
                mask_path = root / pattern.format(stem=stem)
                if not mask_path.exists():
                    raise FileNotFoundError(f"{source}: no mask at {mask_path}")
                masks.append(str(mask_path))

            rows.append({
                "image_path": str(img),
                "mask_paths": masks,
                "source": source,
                "patient_id": f"{source}/{stem}",
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
