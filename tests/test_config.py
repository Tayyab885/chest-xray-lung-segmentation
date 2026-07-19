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
