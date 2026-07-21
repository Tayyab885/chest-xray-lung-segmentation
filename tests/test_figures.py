import matplotlib
import numpy as np
import pandas as pd
import pytest
import torch

matplotlib.use("Agg")

from src.figures import gap_chart, overlay_panel, pick_examples
from src.model import build_model
from src.splits import Fold


def _ood(dices, patient_ids=None, **overrides):
    """A per-image frame shaped like one run's off-domain rows."""
    n = len(dices)
    ids = patient_ids or [f"p{i}" for i in range(n)]
    base = dict(patient_id=ids, split="ood_test", postprocessed=False,
                arm="unet", held_out="c", seed=42, image_size=32, dice=dices)
    base.update(overrides)
    return pd.DataFrame(base)


def _gap_row(arm, held_out, dice_id, dice_ood, model="unet"):
    return {
        "arm": arm, "held_out": held_out, "postprocessed": False, "model": model,
        "dice_id": dice_id, "dice_ood": dice_ood, "dice_gap": dice_id - dice_ood,
        "hd95_id": 8.0, "hd95_ood": 20.0, "hd95_gap": 12.0,
        "n_id": 100, "n_ood": 100, "hd95_n_id": 100, "hd95_n_ood": 90,
        "n_empty_id": 0, "n_empty_ood": 10,
    }


def test_pick_examples_returns_best_median_worst():
    picked = pick_examples(_ood([0.1, 0.4, 0.5, 0.6, 0.9]))
    assert [p["label"] for p in picked] == ["best", "median", "worst"]
    assert picked[0]["patient_id"] == "p4"
    assert picked[1]["patient_id"] == "p2"
    assert picked[2]["patient_id"] == "p0"


def test_pick_examples_carries_the_dice_of_the_case_it_picked():
    # The panel prints this number under the drawn contour. If the label came
    # from a different row the figure would caption a failure as a success.
    picked = pick_examples(_ood([0.1, 0.4, 0.9]))
    assert picked[0]["dice"] == pytest.approx(0.9)
    assert picked[2]["dice"] == pytest.approx(0.1)


def test_pick_examples_ignores_postprocessed_rows():
    df = pd.concat([
        _ood([0.2, 0.8], ["p0", "p1"]),
        _ood([0.99, 0.99], ["p0", "p1"], postprocessed=True),
    ], ignore_index=True)
    picked = pick_examples(df)
    assert picked[0]["patient_id"] == "p1"
    assert picked[-1]["patient_id"] == "p0"
    assert picked[0]["dice"] == pytest.approx(0.8), "picked a cleaned-up score"


def test_pick_examples_ignores_in_domain_rows():
    """The panel exists to show off-domain behaviour.

    In-domain Dice is uniformly higher, so leaking those rows in replaces the
    worst case, the one image the figure is there to show, with a mediocre
    in-domain case and the panel stops being evidence of anything.
    """
    df = pd.concat([
        _ood([0.3, 0.5]),
        _ood([0.05, 0.95], ["q0", "q1"], split="id_test"),
    ], ignore_index=True)
    picked = pick_examples(df)
    assert {p["patient_id"] for p in picked} <= {"p0", "p1"}


def test_pick_examples_breaks_ties_the_same_way_every_time():
    # run_report concatenates the CSVs in whatever order glob returns them.
    # Sorting on dice alone leaves tied rows in input order, so the "median"
    # panel would change between runs on identical data.
    a = _ood([0.5, 0.5, 0.5, 0.5, 0.5], ["p0", "p1", "p2", "p3", "p4"])
    b = a.iloc[::-1].reset_index(drop=True)
    assert [p["patient_id"] for p in pick_examples(a)] == \
           [p["patient_id"] for p in pick_examples(b)]


def test_pick_examples_refuses_a_frame_holding_two_arms():
    """The overlay draws one checkpoint's prediction under one CSV's Dice.

    Mixing arms means the caption can come from a different network than the
    contour, and both numbers are real, so nothing about the figure looks
    wrong.
    """
    df = pd.concat([
        _ood([0.3, 0.9]),
        _ood([0.4, 0.95], ["q0", "q1"], arm="unet_resnet34_imagenet"),
    ], ignore_index=True)
    with pytest.raises(ValueError, match="arm"):
        pick_examples(df)


@pytest.mark.parametrize("column,other", [("seed", 7), ("held_out", "d"),
                                          ("image_size", 64)])
def test_pick_examples_refuses_a_frame_mixing_run_conditions(column, other):
    df = pd.concat([
        _ood([0.3, 0.9]),
        _ood([0.4, 0.95], ["q0", "q1"], **{column: other}),
    ], ignore_index=True)
    with pytest.raises(ValueError, match=column):
        pick_examples(df)


def test_pick_examples_refuses_a_frame_with_no_off_domain_rows():
    # Falling through would index an empty frame and raise IndexError from
    # inside pandas, which reads as a bug in the plotting rather than a
    # results directory that never had the run in it.
    with pytest.raises(ValueError, match="no off-domain"):
        pick_examples(_ood([0.3, 0.9], split="id_test"))


def test_gap_chart_writes_a_file(tmp_path):
    gap = pd.DataFrame([
        _gap_row("unet", "montgomery", 0.95, 0.88),
        _gap_row("unet", "jsrt", 0.95, 0.79),
    ])
    out = gap_chart(gap, tmp_path / "gap.png")
    assert out.exists()
    assert out.stat().st_size > 0


def test_gap_chart_keeps_the_two_resnet_arms_apart(tmp_path):
    """Both ResNet arms carry model name unet_resnet34.

    Labelling the bars by model gives two identical captions on the fold whose
    entire purpose is the pretraining contrast, so the reader cannot tell which
    bar is the pretrained one. Worse, a grouping keyed on model would average
    them and draw the effect as zero.
    """
    gap = pd.DataFrame([
        _gap_row("unet_resnet34_imagenet", "jsrt", 0.96, 0.90, "unet_resnet34"),
        _gap_row("unet_resnet34_scratch", "jsrt", 0.94, 0.72, "unet_resnet34"),
    ])
    fig = gap_chart(gap, tmp_path / "g.png", return_figure=True)
    labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert len({l for l in labels}) == 2
    assert any("imagenet" in l for l in labels)
    assert any("scratch" in l for l in labels)


def test_gap_chart_bars_carry_the_numbers_from_the_table(tmp_path):
    # A chart whose bar heights do not match the CSV is a figure that
    # contradicts its own table, and the README cites both.
    gap = pd.DataFrame([
        _gap_row("unet", "jsrt", 0.91, 0.62),
        _gap_row("unet", "montgomery", 0.93, 0.85),
    ])
    fig = gap_chart(gap, tmp_path / "g.png", return_figure=True)
    heights = sorted(round(p.get_height(), 6) for p in fig.axes[0].patches)
    assert heights == [0.62, 0.85, 0.91, 0.93]


def test_gap_chart_only_plots_raw_rows(tmp_path):
    # Post-processing raises Dice. Drawing both flags side by side without
    # saying so would show the cleanup benefit as if it were a model
    # difference.
    raw = _gap_row("unet", "jsrt", 0.91, 0.62)
    cleaned = {**raw, "postprocessed": True, "dice_id": 0.99, "dice_ood": 0.98}
    fig = gap_chart(pd.DataFrame([raw, cleaned]), tmp_path / "g.png",
                    return_figure=True)
    heights = sorted(round(p.get_height(), 6) for p in fig.axes[0].patches)
    assert heights == [0.62, 0.91]


def test_gap_chart_refuses_an_empty_table(tmp_path):
    # An empty results directory otherwise produces a blank PNG that a README
    # would happily embed.
    empty = pd.DataFrame([_gap_row("unet", "jsrt", 0.9, 0.6)]).iloc[:0]
    with pytest.raises(ValueError, match="no rows"):
        gap_chart(empty, tmp_path / "g.png")


def test_outline_is_the_boundary_not_the_filled_region():
    """The panel draws outlines so the radiograph underneath stays visible.

    A filled overlay hides the very anatomy a reader is being asked to judge
    the contour against, and a dilated boundary sits one pixel outside the
    mask, which is the scale of the errors this figure is meant to show.
    """
    from src.figures import _outline

    mask = np.zeros((12, 12), dtype=bool)
    mask[3:9, 3:9] = True
    out = _outline(mask)
    assert out[3, 3] and out[3, 8] and out[8, 8]      # corners of the border
    assert not out[5, 5]                              # interior is not drawn
    assert not out[2, 3] and not out[3, 2]            # nothing outside the mask
    assert out.sum() == 6 * 6 - 4 * 4


def test_outline_treats_a_mask_touching_the_edge_as_bounded():
    # border_value=0 in the erosion. Without it a lung field running off the
    # bottom of the image gets no boundary drawn along that edge, which is
    # commonest on exactly the off-domain images the panel is showing.
    from src.figures import _outline

    mask = np.zeros((8, 8), dtype=bool)
    mask[5:8, 2:6] = True
    assert _outline(mask)[7, 3], "no contour along the image edge"


def _layers(fig):
    """The three imshow layers of one panel: radiograph, reference, prediction."""
    return [im.get_array() for im in fig.axes[0].images]


@pytest.fixture
def cfg():
    return {"image_size": 32, "batch_size": 2, "num_workers": 0,
            "device": "cpu", "seed": 42}


def _checkpoint(tmp_path, arm="unet", model_name="unet", held_out="c",
                seed=42, image_size=32):
    model = build_model(model_name, pretrained=False)
    path = tmp_path / f"{arm}_{held_out}_seed{seed}.pt"
    torch.save({"model_state": model.state_dict(), "arm": arm,
                "model_name": model_name, "held_out": held_out,
                "seed": seed, "image_size": image_size}, path)
    return path


def _fold_and_scores(fake_frame, n=4):
    ood = fake_frame(n, "c")
    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame, ood_test=ood)
    scores = _ood(list(np.linspace(0.2, 0.9, n)),
                  list(ood["patient_id"]))
    return fold, scores


def test_overlay_panel_writes_a_panel_per_pick(tmp_path, cfg, fake_frame):
    fold, scores = _fold_and_scores(fake_frame)
    out = overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                        tmp_path / "fig" / "overlay.png")
    assert out.exists() and out.stat().st_size > 0


def test_overlay_panel_draws_reference_and_prediction_in_the_stated_colours(
        tmp_path, cfg, fake_frame, monkeypatch):
    """The caption says green is the reference and orange the prediction.

    Swapping them, or drawing the same mask twice, produces a figure that
    looks entirely normal and argues the opposite of the truth. The model here
    predicts nothing at all, so the prediction layer must be empty and the
    reference layer must not be.
    """
    import src.figures as figures

    fold, scores = _fold_and_scores(fake_frame)

    class Empty(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, 32, 32), -20.0)

        def load_state_dict(self, state, *a, **k):
            pass

    monkeypatch.setattr(figures, "build_model",
                        lambda name, pretrained=False: Empty())
    fig = overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                        tmp_path / "o.png", return_figure=True)

    _, reference, prediction = _layers(fig)
    drawn = reference[reference[..., 3] > 0]
    assert len(drawn), "no reference contour drawn"
    assert np.allclose(drawn[0], figures.GT_COLOR)
    assert prediction[..., 3].sum() == 0, "drew a contour for an empty prediction"


def test_overlay_panel_thresholds_the_probability_not_the_logit(
        tmp_path, cfg, fake_frame, monkeypatch):
    # Without the sigmoid, a logit of 0.2 (p = 0.55, a positive prediction)
    # falls below a 0.5 threshold and the panel shows an empty prediction for a
    # model that predicted the whole image.
    import src.figures as figures

    fold, scores = _fold_and_scores(fake_frame)

    class Faint(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, 32, 32), 0.2)

        def load_state_dict(self, state, *a, **k):
            pass

    monkeypatch.setattr(figures, "build_model",
                        lambda name, pretrained=False: Faint())
    fig = overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                        tmp_path / "o.png", return_figure=True)
    assert _layers(fig)[2][..., 3].sum() > 0, "predicted everything, drew nothing"


def test_overlay_panel_draws_the_patient_it_captioned(tmp_path, cfg, fake_frame,
                                                      monkeypatch):
    """The reference contour has to belong to the patient named in the title.

    Every panel would still render, every Dice caption would still be a real
    number, and the figure would be showing three of the wrong images.
    """
    import src.figures as figures

    ood = fake_frame(3, "c")
    # Distinct ground truths, so the contour identifies which row was loaded.
    from PIL import Image
    for i, path in enumerate(ood["mask_paths"]):
        m = np.zeros((32, 32), dtype=np.uint8)
        m[4:28, 4:8 + 6 * i] = 255
        Image.fromarray(m).save(path[0])

    frame = fake_frame(2, "a")
    fold = Fold(held_out="c", train=frame, val=frame, id_test=frame, ood_test=ood)
    scores = _ood([0.1, 0.5, 0.9], list(ood["patient_id"]))

    class Empty(torch.nn.Module):
        def forward(self, x):
            return torch.full((x.shape[0], 1, 32, 32), -20.0)

        def load_state_dict(self, state, *a, **k):
            pass

    monkeypatch.setattr(figures, "build_model",
                        lambda name, pretrained=False: Empty())
    fig = overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                        tmp_path / "o.png", return_figure=True)

    # First axis is the best case, which is the widest mask (patient index 2).
    best_layer = fig.axes[0].images[1].get_array()
    columns = np.where(best_layer[..., 3].sum(axis=0) > 0)[0]
    assert columns.max() == 19, "best panel drew another patient's reference"


@pytest.mark.parametrize("column,other", [("seed", 7), ("image_size", 64),
                                          ("arm", "unet_resnet34_scratch")])
def test_overlay_panel_refuses_scores_that_did_not_come_from_this_checkpoint(
        tmp_path, cfg, fake_frame, column, other):
    """The contour is recomputed; the caption is read from the CSV.

    Both sides are separately self-consistent, so neither existing guard fires:
    the checkpoint matches the config, and the stale CSV agrees with itself.
    Retrain an arm and forget to rerun the evaluator and the panel draws the
    new weights under the old weights' Dice, which is a real number about a
    different run.
    """
    fold, scores = _fold_and_scores(fake_frame)
    scores[column] = other
    with pytest.raises(ValueError, match=column):
        overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                      tmp_path / "o.png")


def test_overlay_panel_refuses_a_checkpoint_from_another_fold(tmp_path, cfg,
                                                              fake_frame):
    """Same failure the evaluator guards, one module later.

    A checkpoint trained holding out a different source has already seen these
    images, so the panel would present near-perfect contours as evidence of
    off-domain generalisation.
    """
    fold, scores = _fold_and_scores(fake_frame)
    ckpt = _checkpoint(tmp_path, held_out="shenzhen")
    with pytest.raises(ValueError, match="shenzhen"):
        overlay_panel(cfg, ckpt, fold, "unet", scores, tmp_path / "o.png")


def test_overlay_panel_refuses_a_checkpoint_from_another_arm(tmp_path, cfg,
                                                             fake_frame):
    fold, scores = _fold_and_scores(fake_frame)
    ckpt = _checkpoint(tmp_path, arm="unet_resnet34_scratch",
                       model_name="unet_resnet34")
    with pytest.raises(ValueError, match="unet_resnet34_scratch"):
        overlay_panel(cfg, ckpt, fold, "unet", scores, tmp_path / "o.png")


def test_overlay_panel_refuses_a_checkpoint_at_another_resolution(tmp_path, cfg,
                                                                  fake_frame):
    # The contour would be drawn from a network run at a resolution it was not
    # trained at, and the Dice printed beneath it comes from the CSV, so the
    # caption and the picture would describe different things.
    fold, scores = _fold_and_scores(fake_frame)
    ckpt = _checkpoint(tmp_path, image_size=64)
    with pytest.raises(ValueError, match="image_size"):
        overlay_panel(cfg, ckpt, fold, "unet", scores, tmp_path / "o.png")


def test_overlay_panel_says_so_when_a_picked_patient_is_not_in_the_fold(
        tmp_path, cfg, fake_frame):
    # Scores from one fold against another fold's images. Indexing an empty
    # match raises IndexError from pandas, which reads as a plotting bug.
    fold, scores = _fold_and_scores(fake_frame)
    scores["patient_id"] = [f"missing{i}" for i in range(len(scores))]
    with pytest.raises(ValueError, match="not in the held-out set"):
        overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                      tmp_path / "o.png")


def test_overlay_panel_uses_the_loaded_weights(tmp_path, cfg, fake_frame,
                                               monkeypatch):
    """Pins load_state_dict, whose removal leaves a plausible-looking figure.

    A randomly initialised network draws contours that look like a bad model
    rather than a broken figure script, and every other test here still passes.
    """
    import src.figures as figures

    fold, scores = _fold_and_scores(fake_frame)
    seen = {}

    class Watcher(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.head = torch.nn.Parameter(torch.zeros(1))

        def load_state_dict(self, state, *a, **k):
            seen["loaded"] = True

        def forward(self, x):
            return torch.full((x.shape[0], 1, 32, 32), -20.0)

    monkeypatch.setattr(figures, "build_model",
                        lambda name, pretrained=False: Watcher())
    overlay_panel(cfg, _checkpoint(tmp_path), fold, "unet", scores,
                  tmp_path / "o.png")
    assert seen.get("loaded"), "figure drew an untrained network"


# --- the report driver -------------------------------------------------------

from src.run_report import compare_arms


def _run_rows(arm, held_out, split, dices, seed=42):
    return pd.DataFrame({
        "patient_id": [f"{held_out}/{i}" for i in range(len(dices))],
        "arm": arm, "model": "unet", "held_out": held_out, "split": split,
        "postprocessed": False, "image_size": 32, "seed": seed, "dice": dices,
    })


def _three_arm_frame(rng=None):
    rng = rng or np.random.default_rng(0)
    base = rng.uniform(0.55, 0.85, 20)
    parts = []
    for held_out in ("c", "d"):
        for split in ("id_test", "ood_test"):
            parts += [
                _run_rows("unet", held_out, split, base),
                _run_rows("unet_resnet34_scratch", held_out, split, base + 0.01),
                _run_rows("unet_resnet34_imagenet", held_out, split, base + 0.03),
            ]
    return pd.concat(parts, ignore_index=True)


def test_compare_arms_covers_every_arm_pair_in_every_fold_and_split():
    out = compare_arms(_three_arm_frame())
    # 3 arms -> 3 pairs, over 2 folds and 2 splits.
    assert len(out) == 12
    assert set(out["held_out"]) == {"c", "d"}
    assert set(out["split"]) == {"id_test", "ood_test"}


def test_compare_arms_measures_pretraining_as_imagenet_minus_scratch():
    """The one comparison the third arm exists to make, in the stated direction.

    unet against unet_resnet34_imagenet differs in both initialisation and
    roughly 3x the parameters, so on its own it cannot support a claim about
    pretraining. Only the two ResNet arms isolate it. Alphabetical pair order
    would report it as scratch minus imagenet, so every sign in the one row
    the claim rests on would read backwards.
    """
    out = compare_arms(_three_arm_frame())
    row = out[(out["family"] == "pretraining") & (out["split"] == "ood_test")].iloc[0]
    assert row["arm_a"] == "unet_resnet34_scratch"
    assert row["arm_b"] == "unet_resnet34_imagenet"
    assert row["hl_estimate"] > 0, "the pretrained arm came out worse"


def test_compare_arms_corrects_the_confirmatory_contrast_on_its_own():
    """Family scope is a decision, not a default.

    The two unet pairs are capacity-confounded and were never able to support
    a pretraining claim. Pooling them with the capacity-matched contrast puts
    the one confirmatory test under a threshold three times stricter than the
    hypothesis it was designed for, so a real effect can be corrected away by
    the company it keeps rather than by weak evidence.
    """
    out = compare_arms(_three_arm_frame())
    pretraining = out[out["family"] == "pretraining"]
    capacity = out[out["family"] == "capacity"]
    assert len(pretraining) == 4 and len(capacity) == 8
    assert (pretraining["family_size"] == 4).all()
    assert (capacity["family_size"] == 8).all()


def test_compare_arms_corrects_within_each_family():
    """Uncorrected, this family manufactures a positive result on its own.

    The claim this project makes is exactly the kind that gets manufactured
    that way, so the corrected column has to exist and has to be the one the
    README quotes.
    """
    out = compare_arms(_three_arm_frame())
    assert "p_holm" in out.columns
    assert (out["p_holm"] >= out["p_value"] - 1e-12).all()
    assert out["p_holm"].max() > out["p_value"].max()

    from src.stats import holm
    for _, group in out.groupby("family"):
        assert group["p_holm"].to_numpy() == pytest.approx(
            holm(group["p_value"].to_numpy()))
    # ... and not the correction the pooled family would have applied.
    pooled = holm(out["p_value"].to_numpy())
    assert (out["p_holm"].to_numpy() <= pooled + 1e-12).all()
    assert (out["p_holm"].to_numpy() < pooled - 1e-12).any(), "families were pooled"


def test_compare_arms_reports_the_direction_it_measured():
    # "b minus a" is meaningless without knowing which arm is which, and a
    # sign read backwards inverts the paper's conclusion.
    out = compare_arms(_three_arm_frame())
    row = out[(out["arm_a"] == "unet") & (out["held_out"] == "c")
              & (out["split"] == "ood_test")].iloc[0]
    assert row["comparison"] == f"{row['arm_b']} minus {row['arm_a']}"
    assert row["hl_estimate"] > 0, "the better arm came out worse"


def test_compare_arms_refuses_to_mix_seeds():
    # Two seeds of one arm are two runs, not one. Concatenating them puts each
    # patient in twice and the paired test would compare a patient with
    # themselves.
    df = pd.concat([_three_arm_frame(),
                    _run_rows("unet", "c", "ood_test", np.linspace(0.4, 0.9, 20),
                              seed=7)], ignore_index=True)
    with pytest.raises(ValueError, match="seed"):
        compare_arms(df)


def test_compare_arms_refuses_to_mix_resolutions():
    # HD95 and ASSD are in pixels at the evaluation resolution, so rows from a
    # 512 run and a 256 run are not on the same scale. Dice survives the mix,
    # which is what makes it silent. Named by paired_compare rather than here.
    df = pd.concat([_three_arm_frame(),
                    _run_rows("unet", "c", "ood_test", np.linspace(0.4, 0.9, 20)
                              ).assign(image_size=64,
                                       patient_id=[f"x{i}" for i in range(20)])],
                   ignore_index=True)
    with pytest.raises(ValueError, match="image_size"):
        compare_arms(df)


def test_compare_arms_handles_a_fold_where_an_arm_did_not_run():
    """A partial results directory is the normal state mid-experiment.

    Assuming all three arms are present pairs a real frame against an empty
    one, which raises IndexError from inside the statistics rather than
    reporting the comparisons that do exist.
    """
    df = _three_arm_frame()
    df = df[df["arm"] != "unet_resnet34_imagenet"]
    out = compare_arms(df)
    assert len(out) == 4  # one pair, over 2 folds and 2 splits
    assert set(out["arm_b"]) == {"unet_resnet34_scratch"}


def test_compare_arms_skips_post_processed_rows():
    # Every CSV holds each patient twice, raw and cleaned. Leaving both in
    # doubles every patient_id and the pairing silently breaks.
    df = _three_arm_frame()
    cleaned = df.copy()
    cleaned["postprocessed"] = True
    cleaned["dice"] = 0.99
    out = compare_arms(pd.concat([df, cleaned], ignore_index=True))
    assert len(out) == 12
    assert (out["n"] == 20).all()
