import numpy as np
import pytest
import yaml
from PIL import Image

from src.data import build_manifest, load_image, load_mask


@pytest.fixture
def fake_root(tmp_path):
    """A miniature two-source tree: one source with split masks, one without."""
    (tmp_path / "srcA" / "img").mkdir(parents=True)
    (tmp_path / "srcA" / "left").mkdir(parents=True)
    (tmp_path / "srcA" / "right").mkdir(parents=True)
    (tmp_path / "srcB" / "img").mkdir(parents=True)
    (tmp_path / "srcB" / "mask").mkdir(parents=True)

    for i in (1, 2):
        Image.fromarray(np.full((32, 32), 100, dtype=np.uint8)).save(
            tmp_path / "srcA" / "img" / f"A_{i}.png")
        left = np.zeros((32, 32), dtype=np.uint8)
        left[8:16, 4:12] = 255
        right = np.zeros((32, 32), dtype=np.uint8)
        right[8:16, 20:28] = 255
        Image.fromarray(left).save(tmp_path / "srcA" / "left" / f"A_{i}.png")
        Image.fromarray(right).save(tmp_path / "srcA" / "right" / f"A_{i}.png")

    Image.fromarray(np.full((32, 32), 50, dtype=np.uint8)).save(
        tmp_path / "srcB" / "img" / "B_1.png")
    m = np.zeros((32, 32), dtype=np.uint8)
    m[4:20, 4:20] = 255
    Image.fromarray(m).save(tmp_path / "srcB" / "mask" / "B_1_mask.png")

    layout = {
        "srcA": {
            "image_glob": "srcA/img/A_*.png",
            "mask_globs": ["srcA/left/{stem}.png", "srcA/right/{stem}.png"],
            "expected_count": 2,
        },
        "srcB": {
            "image_glob": "srcB/img/B_*.png",
            "mask_globs": ["srcB/mask/{stem}_mask.png"],
            "expected_count": 1,
        },
    }
    return tmp_path, layout


def test_manifest_has_one_row_per_image(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    assert len(man) == 3
    assert set(man["source"]) == {"srcA", "srcB"}
    assert (man["source"] == "srcA").sum() == 2


def test_manifest_columns(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    assert list(man.columns) == ["image_path", "mask_paths", "source", "patient_id"]


def test_patient_ids_are_unique_within_source(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    for _, group in man.groupby("source"):
        assert group["patient_id"].is_unique


def test_manifest_raises_on_count_mismatch(fake_root):
    root, layout = fake_root
    layout["srcB"]["expected_count"] = 99
    with pytest.raises(ValueError, match="srcB"):
        build_manifest(root, layout)


def test_manifest_raises_on_missing_mask(fake_root):
    root, layout = fake_root
    (root / "srcB" / "mask" / "B_1_mask.png").unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(root, layout)


def test_manifest_raises_on_duplicate_patient_id(fake_root):
    root, layout = fake_root
    # Community re-uploads sometimes leave a nested copy of an image behind. Two
    # rows sharing one patient_id would put the same patient on both sides of a
    # split, so the manifest has to reject it rather than assume stems are unique.
    (root / "srcB" / "img" / "extra").mkdir()
    Image.fromarray(np.full((32, 32), 50, dtype=np.uint8)).save(
        root / "srcB" / "img" / "extra" / "B_1.png")
    layout["srcB"]["image_glob"] = "srcB/img/**/B_*.png"
    layout["srcB"]["expected_count"] = 2
    with pytest.raises(ValueError, match="patient_id"):
        build_manifest(root, layout)


def test_split_masks_are_merged(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    row = man[man["source"] == "srcA"].iloc[0]
    mask = load_mask(row["mask_paths"], size=32)
    assert mask.dtype == bool
    # Both the left blob and the right blob survive the merge.
    assert mask[10, 8]
    assert mask[10, 24]
    assert not mask[0, 0]


def test_mask_is_binary_after_resize(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    mask = load_mask(man.iloc[0]["mask_paths"], size=64)
    assert mask.shape == (64, 64)
    assert set(np.unique(mask)).issubset({True, False})


def test_masks_are_not_empty(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    for _, row in man.iterrows():
        assert load_mask(row["mask_paths"], size=32).sum() > 0


def test_image_is_z_scored(fake_root):
    root, layout = fake_root
    man = build_manifest(root, layout)
    # A constant image has zero variance, so the loader must not divide by zero.
    img = load_image(man.iloc[0]["image_path"], size=32)
    assert img.shape == (32, 32)
    assert img.dtype == np.float32
    assert np.isfinite(img).all()


def _add_unmasked(root, layout, n=1):
    """Add images with no mask, as the Shenzhen mirror has."""
    for i in range(n):
        Image.fromarray(np.full((32, 32), 70, dtype=np.uint8)).save(
            root / "srcB" / "img" / f"B_9{i}.png")
    layout["srcB"]["expected_count"] = 1


def test_unmasked_images_still_raise_when_the_layout_does_not_declare_them(fake_root):
    root, layout = fake_root
    _add_unmasked(root, layout)
    # Silence is the failure mode here. An undeclared image without a mask is a
    # broken download, not a dataset property, and dropping it quietly would
    # change the denominator of every metric.
    with pytest.raises(FileNotFoundError):
        build_manifest(root, layout)


def test_a_declared_count_of_unmasked_images_is_dropped(fake_root):
    root, layout = fake_root
    _add_unmasked(root, layout, n=2)
    layout["srcB"]["expected_unmasked"] = 2

    manifest = build_manifest(root, layout)
    # The Shenzhen mirror ships 662 radiographs and 566 masks. The 96 without
    # one cannot be segmented or scored, so the study is the annotated subset,
    # and saying so in the config is what separates it from a silent drop.
    assert list(manifest[manifest["source"] == "srcB"]["patient_id"]) == ["srcB/B_1"]


def test_the_declared_number_of_unmasked_images_is_itself_checked(fake_root):
    root, layout = fake_root
    _add_unmasked(root, layout, n=3)
    layout["srcB"]["expected_unmasked"] = 2

    # Pins the composition from both sides. Without this a mirror that lost
    # another hundred masks would still build a manifest, just a smaller one,
    # and nothing downstream would notice the study had changed.
    with pytest.raises(ValueError, match="unmasked"):
        build_manifest(root, layout)


def test_expected_count_is_checked_after_the_drop(fake_root):
    root, layout = fake_root
    _add_unmasked(root, layout, n=2)
    layout["srcB"]["expected_unmasked"] = 2
    layout["srcB"]["expected_count"] = 3  # the pre-drop total, not the study size

    # expected_count is the number of images that enter the study, so it has to
    # be counted after the unannotated ones are removed.
    with pytest.raises(ValueError, match="expects 3"):
        build_manifest(root, layout)


def test_dropped_images_are_reported(fake_root, capsys):
    root, layout = fake_root
    _add_unmasked(root, layout, n=2)
    layout["srcB"]["expected_unmasked"] = 2
    build_manifest(root, layout)

    assert "2" in capsys.readouterr().out
