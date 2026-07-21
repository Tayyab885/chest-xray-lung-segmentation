from pathlib import Path

import pytest
import yaml

from src.config import load_config


def test_load_config_fills_defaults(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"data_root": "data"}))
    cfg = load_config(p)
    assert cfg["seed"] == 42
    assert cfg["image_size"] == 512
    assert cfg["num_workers"] == 2
    assert cfg["data_root"] == "data"


def test_load_config_keeps_explicit_values(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"data_root": "data", "seed": 7, "batch_size": 64}))
    cfg = load_config(p)
    assert cfg["seed"] == 7
    assert cfg["batch_size"] == 64


def test_load_config_rejects_bad_image_size(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump({"data_root": "data", "image_size": 0}))
    with pytest.raises(ValueError):
        load_config(p)


REPO = Path(__file__).resolve().parent.parent


def test_the_two_layouts_describe_the_same_datasets():
    local = yaml.safe_load((REPO / "configs/data_layout.yaml").read_text(encoding="utf-8"))
    kaggle = yaml.safe_load((REPO / "configs/kaggle_layout.yaml").read_text(encoding="utf-8"))

    # Only the paths may differ. A source present in one and not the other, or
    # a different expected_count, means the local and remote runs are scoring
    # different datasets while reporting one number.
    assert set(local) == set(kaggle)
    for source in local:
        assert local[source]["expected_count"] == kaggle[source]["expected_count"]
        assert local[source]["kaggle_slug"] == kaggle[source]["kaggle_slug"]
        assert len(local[source]["mask_globs"]) == len(kaggle[source]["mask_globs"])


def test_kaggle_config_changes_nothing_that_defines_the_protocol():
    local = load_config(REPO / "configs/default.yaml")
    kaggle = load_config(REPO / "configs/kaggle.yaml")

    # These four decide the splits and the metric, so they must be identical or
    # the Kaggle numbers are not the numbers this repo claims to compute.
    # batch_size and num_workers are free: they are what the bigger GPU buys.
    for key in ("image_size", "seed", "val_frac", "test_frac"):
        assert local[key] == kaggle[key], key
