"""Configuration loading with defaults."""
from pathlib import Path

import yaml

DEFAULTS = {
    "seed": 42,
    "image_size": 512,
    "data_root": "data",
    "results_dir": "results",
    "checkpoint_dir": "checkpoints",
    "batch_size": 8,
    "epochs": 60,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "patience": 10,
    "num_workers": 2,
    "val_frac": 0.15,
    "test_frac": 0.15,
}


def load_config(path):
    """Load a YAML config and fill in defaults for missing keys."""
    with open(Path(path), "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = dict(DEFAULTS)
    cfg.update(raw)

    size = cfg["image_size"]
    if not isinstance(size, int) or size <= 0:
        raise ValueError(f"image_size must be a positive int, got {size!r}")

    return cfg
