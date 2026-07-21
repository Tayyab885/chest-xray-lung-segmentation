import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from src.evaluate import evaluate_run, postprocess
from src.model import build_model
from src.splits import Fold


def _checkpoint(tmp_path, arm="unet", model_name="unet", held_out="c"):
    model = build_model(model_name, pretrained=False)
    path = tmp_path / f"{arm}.pt"
    torch.save({"model_state": model.state_dict(), "arm": arm,
                "model_name": model_name, "held_out": held_out,
                "seed": 42, "image_size": 32}, path)
    return path


@pytest.fixture
def cfg():
    return {"image_size": 32, "batch_size": 2, "num_workers": 0,
            "device": "cpu", "seed": 42}


def test_postprocess_keeps_the_two_largest_blobs():
    m = np.zeros((40, 40), dtype=bool)
    m[2:20, 2:12] = True    # 180 px
    m[2:20, 25:35] = True   # 180 px
    m[35:37, 35:37] = True  # 4 px speck
    out = postprocess(m, keep=2)
    assert out[10, 5] and out[10, 30]
    assert not out[35, 35]


def test_postprocess_leaves_a_clean_mask_alone():
    m = np.zeros((40, 40), dtype=bool)
    m[2:20, 2:12] = True
    m[2:20, 25:35] = True
    assert np.array_equal(postprocess(m, keep=2), m)


def test_postprocess_handles_empty_mask():
    m = np.zeros((16, 16), dtype=bool)
    assert postprocess(m, keep=2).sum() == 0


def test_postprocess_handles_fewer_blobs_than_keep():
    m = np.zeros((16, 16), dtype=bool)
    m[2:8, 2:8] = True
    assert np.array_equal(postprocess(m, keep=2), m)


def test_postprocess_only_ever_removes_pixels():
    # Post-processing is a cleanup step, not a prediction step. If it could add
    # pixels it would be quietly improving Dice on its own, and the raw-vs-
    # postprocessed comparison would stop being a comparison.
    rng = np.random.default_rng(0)
    for _ in range(20):
        m = rng.random((24, 24)) > 0.6
        out = postprocess(m, keep=2)
        assert not np.logical_and(out, ~m).any()


def test_postprocess_returns_bool_for_the_metrics_contract():
    # src.metrics rejects anything that is not 2D bool.
    m = np.zeros((16, 16), dtype=bool)
    m[2:8, 2:8] = True
    assert postprocess(m, keep=2).dtype == np.bool_


def test_evaluate_run_shape_and_columns(tmp_path, cfg, fake_frame):
    frame = fake_frame(3, "a")
    ood = fake_frame(2, "c")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame, ood_test=ood)
    df = evaluate_run(cfg, _checkpoint(tmp_path), fold, "unet")

    # 5 images, each scored raw and post-processed.
    assert len(df) == 10
    assert set(df["split"]) == {"id_test", "ood_test"}
    assert set(df["postprocessed"]) == {True, False}
    for col in ["patient_id", "source", "model", "arm", "held_out",
                "dice", "iou", "hd95", "assd"]:
        assert col in df.columns
    assert (df["model"] == "unet").all()
    assert (df["held_out"] == "c").all()


def test_every_patient_appears_exactly_twice(tmp_path, cfg, fake_frame):
    frame = fake_frame(3, "a")
    ood = fake_frame(2, "c")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame, ood_test=ood)
    df = evaluate_run(cfg, _checkpoint(tmp_path), fold, "unet")
    assert (df.groupby("patient_id").size() == 2).all()


def test_the_two_resnet_arms_stay_distinguishable(tmp_path, cfg, fake_frame):
    # Both ResNet arms share the model name "unet_resnet34". If the CSV only
    # recorded the model, Task 9 would group them together and the pretraining
    # contrast, the entire reason the third arm exists, would average itself
    # away without anything failing.
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "c"))
    rows = []
    for arm in ["unet_resnet34_imagenet", "unet_resnet34_scratch"]:
        ckpt = _checkpoint(tmp_path, arm=arm, model_name="unet_resnet34")
        rows.append(evaluate_run(cfg, ckpt, fold, arm))
    df = pd.concat(rows)
    assert df["arm"].nunique() == 2


def test_scoring_a_checkpoint_as_the_wrong_arm_raises(tmp_path, cfg, fake_frame):
    # A results table built from mislabelled checkpoints is worse than no
    # table: the numbers are real, so nothing looks wrong.
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "c"))
    ckpt = _checkpoint(tmp_path, arm="unet_resnet34_scratch",
                       model_name="unet_resnet34")
    with pytest.raises(ValueError, match="unet_resnet34_scratch"):
        evaluate_run(cfg, ckpt, fold, "unet_resnet34_imagenet")


def test_metrics_are_attached_to_the_right_patient(tmp_path, cfg, monkeypatch):
    """Each row's score must belong to that row's image, not its neighbour's.

    Predictions come back in loader order while the metadata is zipped from the
    frame. If those ever diverge, every metric is still a plausible number and
    every other test in this file still passes; the results table is simply
    wrong about who is who.
    """
    import src.evaluate as ev

    size = 32
    # Three ground truths with deliberately different areas, so a fixed
    # prediction scores each one differently and the scores identify the row.
    areas = {"p0": 8, "p1": 16, "p2": 24}
    rows = []
    for name, width in areas.items():
        gt = np.zeros((size, size), dtype=np.uint8)
        gt[4:28, 4:4 + width] = 255
        ip = tmp_path / f"{name}.png"
        mp = tmp_path / f"{name}_m.png"
        Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(ip)
        Image.fromarray(gt).save(mp)
        rows.append({"image_path": str(ip), "mask_paths": [str(mp)],
                     "source": "a", "patient_id": name})
    frame = pd.DataFrame(rows)

    pattern = np.zeros((size, size), dtype=np.float32)
    pattern[4:28, 4:20] = 1.0  # fixed prediction, 24x16

    class Constant(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("pat", torch.from_numpy(pattern))

        def forward(self, x):
            logits = self.pat * 12.0 - 6.0
            return logits.view(1, 1, size, size).expand(x.shape[0], 1, size, size)

    monkeypatch.setattr(ev, "build_model", lambda name, pretrained=False: Constant())
    ckpt = tmp_path / "const.pt"
    torch.save({"model_state": Constant().state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    df = evaluate_run(cfg, ckpt, fold, "unet")

    raw = df[~df["postprocessed"]].set_index("patient_id")
    pred_area = 24 * 16
    for name, width in areas.items():
        gt_area = 24 * width
        overlap = 24 * min(width, 16)
        expected = 2.0 * overlap / (pred_area + gt_area)
        assert raw.loc[name, "dice"] == pytest.approx(expected, abs=1e-6), (
            f"{name} carries another image's score"
        )


def test_postprocessing_is_reported_not_silently_applied(tmp_path, cfg, monkeypatch):
    # The raw row is the honest number. If post-processing were applied to both
    # rows, the README would report a cleanup benefit it never measured.
    import src.evaluate as ev

    size = 32
    # Two lung fields, as in the real data. With a single blob a speck would
    # only be the second component and keep=2 would rightly retain it.
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[4:28, 4:12] = 255
    gt[4:28, 18:26] = 255
    ip, mp = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(ip)
    Image.fromarray(gt).save(mp)
    frame = pd.DataFrame([{"image_path": str(ip), "mask_paths": [str(mp)],
                           "source": "a", "patient_id": "p0"}])

    # Correct lungs plus a speck in the corner that post-processing removes.
    pattern = np.zeros((size, size), dtype=np.float32)
    pattern[4:28, 4:12] = 1.0
    pattern[4:28, 18:26] = 1.0
    pattern[30:32, 30:32] = 1.0

    class Speckled(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.register_buffer("pat", torch.from_numpy(pattern))

        def forward(self, x):
            logits = self.pat * 12.0 - 6.0
            return logits.view(1, 1, size, size).expand(x.shape[0], 1, size, size)

    monkeypatch.setattr(ev, "build_model", lambda name, pretrained=False: Speckled())
    ckpt = tmp_path / "speck.pt"
    torch.save({"model_state": Speckled().state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    df = evaluate_run(cfg, ckpt, fold, "unet").set_index("postprocessed")

    assert df.loc[False, "dice"] < df.loc[True, "dice"], "raw row already cleaned"
    assert df.loc[True, "dice"] == pytest.approx(1.0)


def test_scoring_a_checkpoint_from_the_wrong_fold_raises(tmp_path, cfg, fake_frame):
    """The mixup that inverts the result rather than mislabelling a column.

    A checkpoint trained holding out Shenzhen, scored against the Montgomery
    fold, has already seen the images the fold calls off-domain. Off-domain
    Dice comes back near in-domain Dice, the headline gap collapses toward
    zero, and the story flips to "these models generalise fine". Nothing
    crashes and every number in the CSV is real.
    """
    frame = fake_frame(2, "a")
    fold = Fold(held_out="montgomery", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "montgomery"))
    ckpt = _checkpoint(tmp_path, held_out="shenzhen")
    with pytest.raises(ValueError, match="shenzhen"):
        evaluate_run(cfg, ckpt, fold, "unet")


def test_a_checkpoint_without_provenance_is_refused(tmp_path, cfg, fake_frame):
    # A missing key must not be read as agreement.
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "c"))
    model = build_model("unet", pretrained=False)
    ckpt = tmp_path / "bare.pt"
    torch.save({"model_state": model.state_dict()}, ckpt)
    with pytest.raises(ValueError, match="provenance"):
        evaluate_run(cfg, ckpt, fold, "unet")


def test_seed_and_image_size_must_match_the_checkpoint(tmp_path, cfg, fake_frame):
    # HD95 and ASSD are in pixels at the evaluation resolution, so scoring a
    # 512-trained checkpoint at 256 yields numbers that differ by a factor of
    # two from the rest of the table with no column revealing it. A seed
    # mismatch is worse: it manufactures a replicate that does not exist.
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "c"))
    ckpt = _checkpoint(tmp_path)
    with pytest.raises(ValueError, match="image_size"):
        evaluate_run({**cfg, "image_size": 64}, ckpt, fold, "unet")
    with pytest.raises(ValueError, match="seed"):
        evaluate_run({**cfg, "seed": 7}, ckpt, fold, "unet")


def test_off_domain_rows_are_the_held_out_source(tmp_path, cfg, fake_frame):
    # Swapping the two split labels sign-flips the generalisation gap, and
    # asserting only that both labels appear cannot see it.
    id_frame = fake_frame(3, "a")
    ood_frame = fake_frame(2, "c")
    fold = Fold(held_out="c", train=id_frame, val=id_frame,
                id_test=id_frame, ood_test=ood_frame)
    df = evaluate_run(cfg, _checkpoint(tmp_path), fold, "unet")

    assert set(df[df["split"] == "ood_test"]["source"]) == {"c"}
    assert "c" not in set(df[df["split"] == "id_test"]["source"])
    assert set(df[df["split"] == "id_test"]["source"]) == {"a"}


def test_predictions_use_the_loaded_weights(tmp_path, cfg, fake_frame):
    """Pins the deletion of load_state_dict, which leaves every other test green.

    Scoring a randomly initialised network looks like a bad arm rather than a
    broken evaluator, so nothing downstream would question the numbers.
    """
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])

    model = build_model("unet", pretrained=False)
    with torch.no_grad():
        model.head.bias.fill_(20.0)  # predict every pixel as lung
    ckpt = tmp_path / "allfg.pt"
    torch.save({"model_state": model.state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    df = evaluate_run({**cfg, "seed": 42}, ckpt, fold, "unet")
    raw = df[~df["postprocessed"]]
    # All-foreground against the fixture mask. Dice and IoU then satisfy
    # dice = 2*iou/(1+iou) exactly, and Dice is strictly below 1, which a
    # random network reaches only by accident.
    assert np.allclose(raw["dice"], 2.0 * raw["iou"] / (1.0 + raw["iou"]))
    assert (raw["dice"] < 1.0).all() and (raw["dice"] > 0.3).all()


def test_model_is_in_eval_mode_during_prediction(tmp_path, cfg, fake_frame, monkeypatch):
    # Without model.eval() BatchNorm normalises by batch statistics, so
    # predictions depend on batch composition and the last partial batch is
    # normalised differently from every other batch.
    import src.evaluate as ev
    seen = []

    class Watcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))
            self.bn = torch.nn.BatchNorm2d(1)

        def forward(self, x):
            seen.append(self.training)
            return torch.zeros_like(x)

    monkeypatch.setattr(ev, "build_model", lambda name, pretrained=False: Watcher())
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    ckpt = tmp_path / "w.pt"
    torch.save({"model_state": Watcher().state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    evaluate_run({**cfg, "seed": 42}, ckpt, fold, "unet")
    assert seen and not any(seen), "BatchNorm normalised by batch statistics"


def test_probability_threshold_is_a_half_not_a_raw_logit(tmp_path, cfg, monkeypatch):
    # Without sigmoid, thresholding the logit at 0.5 is a threshold at p=0.62.
    # Dice then shifts in one direction across all 9 runs, which is the size of
    # the effect being measured between arms.
    import src.evaluate as ev

    size = 32

    class Faint(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            # p = 0.55: positive at the 0.5 probability threshold, below a raw
            # logit threshold of 0.5.
            return torch.full((x.shape[0], 1, size, size), 0.2)

    monkeypatch.setattr(ev, "build_model", lambda name, pretrained=False: Faint())
    gt = np.full((size, size), 255, dtype=np.uint8)
    ip, mp = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(ip)
    Image.fromarray(gt).save(mp)
    frame = pd.DataFrame([{"image_path": str(ip), "mask_paths": [str(mp)],
                           "source": "a", "patient_id": "p0"}])
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    ckpt = tmp_path / "f.pt"
    torch.save({"model_state": Faint().state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    df = evaluate_run({**cfg, "seed": 42}, ckpt, fold, "unet")
    assert np.allclose(df["dice"], 1.0)


def test_empty_predictions_are_flagged_for_downstream_aggregation(tmp_path, cfg,
                                                                 monkeypatch):
    """HD95 is NaN exactly when the prediction is empty, the total-failure case.

    Task 9 aggregates these CSVs. A plain .mean() skips NaN, so off-domain mean
    HD95 would be computed only over the images where the model did not fail,
    and a paired test would silently drop those pairs. The gap comes out
    smaller than the truth, in the direction that flatters the result. The
    column makes the omission countable.
    """
    import src.evaluate as ev

    size = 32

    class Empty(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return torch.full((x.shape[0], 1, size, size), -20.0)

    monkeypatch.setattr(ev, "build_model", lambda name, pretrained=False: Empty())
    gt = np.zeros((size, size), dtype=np.uint8)
    gt[8:24, 8:24] = 255
    ip, mp = tmp_path / "i.png", tmp_path / "m.png"
    Image.fromarray(np.zeros((size, size), dtype=np.uint8)).save(ip)
    Image.fromarray(gt).save(mp)
    frame = pd.DataFrame([{"image_path": str(ip), "mask_paths": [str(mp)],
                           "source": "a", "patient_id": "p0"}])
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    ckpt = tmp_path / "e.pt"
    torch.save({"model_state": Empty().state_dict(), "arm": "unet",
                "model_name": "unet", "held_out": "c", "seed": 42,
                "image_size": 32}, ckpt)

    df = evaluate_run({**cfg, "seed": 42}, ckpt, fold, "unet")
    assert df["pred_empty"].all()
    assert df["dice"].eq(0.0).all()
    assert df["hd95"].isna().all(), "NaN would be invisible without the flag"


def test_rows_record_the_evaluation_resolution(tmp_path, cfg, fake_frame):
    # Concatenating a 512-run CSV with a 256-run CSV averages HD95 values that
    # differ by a factor of two. The column is what makes that detectable.
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=fake_frame(1, "c"))
    df = evaluate_run(cfg, _checkpoint(tmp_path), fold, "unet")
    assert (df["image_size"] == 32).all()


def test_a_short_prediction_list_is_an_error_not_a_short_csv(tmp_path, cfg,
                                                            fake_frame, monkeypatch):
    # Silently zipping to the shorter list gives correct scores over a wrong
    # denominator, which nothing downstream would notice.
    import src.evaluate as ev
    real = ev._predict_frame
    monkeypatch.setattr(ev, "_predict_frame",
                        lambda *a, **k: tuple(x[:1] for x in real(*a, **k)))
    frame = fake_frame(3, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame,
                ood_test=frame.iloc[:0])
    with pytest.raises(ValueError, match="predictions"):
        evaluate_run(cfg, _checkpoint(tmp_path), fold, "unet")
