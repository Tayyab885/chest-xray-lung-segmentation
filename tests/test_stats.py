import numpy as np
import pandas as pd
import pytest

from src.stats import generalization_gap, holm, paired_compare, summarize


def _per_image(arm, held_out, split, dices, model=None, postprocessed=False,
               hd95=None, image_size=512, prefix="p", seed=42):
    """A per-image frame shaped like the CSVs Task 8 writes."""
    n = len(dices)
    return pd.DataFrame({
        "patient_id": [f"{prefix}{i}" for i in range(n)],
        "source": "s",
        "arm": arm,
        "model": model if model is not None else arm,
        "held_out": held_out,
        "split": split,
        "postprocessed": postprocessed,
        "image_size": image_size,
        "seed": seed,
        "pred_empty": [d == 0.0 for d in dices],
        "dice": list(dices),
        "iou": [d / (2 - d) for d in dices],
        "hd95": list(hd95) if hd95 is not None else [10 * (1 - d) for d in dices],
        "assd": [3 * (1 - d) for d in dices],
    })


# --- summarize ------------------------------------------------------------

def test_summarize_groups_and_counts():
    df = pd.concat([
        _per_image("unet", "a", "id_test", [0.9, 0.8]),
        _per_image("unet", "a", "ood_test", [0.6, 0.4]),
    ])
    s = summarize(df)
    assert len(s) == 2
    row = s[s["split"] == "id_test"].iloc[0]
    assert row["dice_mean"] == pytest.approx(0.85)
    assert row["n"] == 2


def test_the_two_resnet_arms_stay_separate():
    """The whole pretraining contrast lives in this grouping key.

    unet_resnet34_imagenet and unet_resnet34_scratch share the model name
    unet_resnet34, so grouping by model averages the pretrained and the
    randomly initialised arm into one row. The measured effect of ImageNet
    pretraining then becomes exactly zero by construction, and the summary
    table still looks complete.
    """
    df = pd.concat([
        _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.80, 0.80],
                   model="unet_resnet34"),
        _per_image("unet_resnet34_scratch", "a", "ood_test", [0.60, 0.60],
                   model="unet_resnet34"),
    ])
    s = summarize(df)
    assert len(s) == 2, "the two ResNet arms were merged into one row"
    by_arm = s.set_index("arm")["dice_mean"]
    assert by_arm["unet_resnet34_imagenet"] == pytest.approx(0.80)
    assert by_arm["unet_resnet34_scratch"] == pytest.approx(0.60)
    # model is carried through for the table, but must not be the key.
    assert set(s["model"]) == {"unet_resnet34"}


def test_one_arm_carrying_two_model_names_is_refused():
    # arm maps to model 1:1 through ARMS, so a disagreement means the CSVs
    # came from a mislabelled run. Taking the first value resolves it silently
    # and the wrong architecture ends up named in the results table.
    df = pd.concat([
        _per_image("unet", "a", "ood_test", [0.8], model="unet"),
        _per_image("unet", "a", "ood_test", [0.7], model="unet_resnet34",
                   prefix="q"),
    ])
    with pytest.raises(ValueError, match="model names"):
        summarize(df)


def test_summary_reports_how_many_images_failed_outright():
    """A mean over four images and a mean over four attempts differ.

    An empty prediction scores Dice 0 and HD95 NaN. pandas skips the NaN, so
    hd95_mean is an average over the images the model did not fail on. Off
    domain that is exactly the subset that flatters the model, and nothing in
    a mean-only table shows the denominator changed.
    """
    df = _per_image("unet", "a", "ood_test", [0.8, 0.8, 0.0, 0.0],
                    hd95=[4.0, 6.0, np.nan, np.nan])
    row = summarize(df).iloc[0]
    assert row["n"] == 4
    assert row["n_empty"] == 2
    assert row["hd95_n"] == 2, "the HD95 denominator is not reported"
    assert row["hd95_mean"] == pytest.approx(5.0)
    assert row["dice_mean"] == pytest.approx(0.4)
    # On chest radiographs every ground truth has lungs, so the only way a
    # boundary goes missing is an empty prediction. A violation would mean
    # something upstream produced an empty ground truth.
    assert row["n_empty"] == row["n"] - row["hd95_n"]


def test_mixing_evaluation_resolutions_is_refused():
    # HD95 and ASSD are pixel counts at the evaluation resolution, so a 256 row
    # and a 512 row are not the same unit. Averaging them produces a number in
    # no unit at all and no column in the output would reveal it.
    df = pd.concat([
        _per_image("unet", "a", "ood_test", [0.8], image_size=512),
        _per_image("unet", "a", "ood_test", [0.8], image_size=256, prefix="q"),
    ])
    with pytest.raises(ValueError, match="image_size"):
        summarize(df)


def test_mixing_seeds_is_refused():
    """The confound the three-arm design exists to remove.

    Two arms trained under different seeds have distinct patient sets per
    group and distinct arm labels, so no duplicate check and no grouping key
    can see the difference. The pretraining effect then carries the seed
    difference inside it, and the summary table looks complete.
    """
    df = pd.concat([
        _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.8], seed=42,
                   model="unet_resnet34"),
        _per_image("unet_resnet34_scratch", "a", "ood_test", [0.6], seed=43,
                   model="unet_resnet34"),
    ])
    with pytest.raises(ValueError, match="seed"):
        summarize(df)


def test_a_patient_scored_twice_in_one_group_is_refused():
    # Concatenating the same run's CSV twice doubles n and silently averages
    # replicates, which makes the group look better powered than it is and
    # then feeds the paired test.
    df = pd.concat([
        _per_image("unet", "a", "ood_test", [0.8, 0.7]),
        _per_image("unet", "a", "ood_test", [0.6, 0.5]),
    ])
    with pytest.raises(ValueError, match="duplicate"):
        summarize(df)


def test_raw_and_postprocessed_rows_are_separate():
    # Task 8 writes both for every image. Pooling them averages each image with
    # its own cleaned-up version and reports a number belonging to neither.
    df = pd.concat([
        _per_image("unet", "a", "ood_test", [0.6, 0.6], postprocessed=False),
        _per_image("unet", "a", "ood_test", [0.8, 0.8], postprocessed=True),
    ])
    s = summarize(df)
    assert len(s) == 2
    by_flag = s.set_index("postprocessed")["dice_mean"]
    assert by_flag[False] == pytest.approx(0.6)
    assert by_flag[True] == pytest.approx(0.8)


# --- generalization_gap ---------------------------------------------------

def test_generalization_gap_is_id_minus_ood():
    df = pd.concat([
        _per_image("unet", "a", "id_test", [0.9, 0.9]),
        _per_image("unet", "a", "ood_test", [0.6, 0.6], prefix="q"),
    ])
    gap = generalization_gap(summarize(df))
    row = gap.iloc[0]
    assert row["dice_id"] == pytest.approx(0.9)
    assert row["dice_ood"] == pytest.approx(0.6)
    assert row["dice_gap"] == pytest.approx(0.3)


def test_both_gaps_are_positive_when_the_model_generalises_worse():
    """Dice is higher-is-better and HD95 is lower-is-better.

    Subtracting in the same direction for both would make one gap negative
    for a model that degrades off domain, and the README's claim that a
    positive gap means worse generalisation would be false for half the
    columns.
    """
    df = pd.concat([
        _per_image("unet", "a", "id_test", [0.9, 0.9], hd95=[2.0, 2.0]),
        _per_image("unet", "a", "ood_test", [0.6, 0.6], hd95=[9.0, 9.0],
                   prefix="q"),
    ])
    row = generalization_gap(summarize(df)).iloc[0]
    assert row["dice_gap"] > 0
    assert row["hd95_gap"] == pytest.approx(7.0)
    assert row["hd95_gap"] > 0, "hd95 gap has the wrong sign convention"


def test_the_gap_row_carries_the_denominator_its_hd95_was_averaged_over():
    """hd95_gap is a difference of means taken over different image counts.

    Two of the four off-domain images here failed outright, so hd95_ood is an
    average over the two that worked. A row showing n_ood=4 beside it reads
    as a mild degradation over four images when half of them had no
    prediction at all, and off domain that is the common case.
    """
    df = pd.concat([
        # One in-domain failure too, so n and hd95_n differ on both halves of
        # the row. Equal counts on the in-domain side would let the gap table
        # substitute n for hd95_n and still pass.
        _per_image("unet", "a", "id_test", [0.9, 0.9, 0.9, 0.0],
                   hd95=[2.0, 2.0, 2.0, np.nan]),
        _per_image("unet", "a", "ood_test", [0.7, 0.7, 0.0, 0.0],
                   hd95=[4.0, 4.0, np.nan, np.nan], prefix="q"),
    ])
    row = generalization_gap(summarize(df)).iloc[0]
    assert row["n_ood"] == 4
    assert row["hd95_n_ood"] == 2, "the HD95 denominator did not reach the gap table"
    assert row["n_empty_ood"] == 2
    assert row["n_id"] == 4
    assert row["hd95_n_id"] == 3, "the HD95 denominator did not reach the gap table"
    assert row["n_empty_id"] == 1
    assert row["hd95_gap"] == pytest.approx(2.0)


def test_gap_is_computed_per_arm_and_fold():
    df = pd.concat([
        _per_image("unet", "a", "id_test", [0.9]),
        _per_image("unet", "a", "ood_test", [0.5], prefix="q"),
        _per_image("unet_resnet34_scratch", "a", "id_test", [0.9],
                   model="unet_resnet34"),
        _per_image("unet_resnet34_scratch", "a", "ood_test", [0.7],
                   model="unet_resnet34", prefix="q"),
        _per_image("unet", "b", "id_test", [0.9]),
        _per_image("unet", "b", "ood_test", [0.8], prefix="q"),
    ])
    gap = generalization_gap(summarize(df))
    assert len(gap) == 3
    keyed = gap.set_index(["arm", "held_out"])["dice_gap"]
    assert keyed[("unet", "a")] == pytest.approx(0.4)
    assert keyed[("unet_resnet34_scratch", "a")] == pytest.approx(0.2)
    assert keyed[("unet", "b")] == pytest.approx(0.1)


def test_a_run_missing_one_split_produces_no_gap_row():
    # A gap needs both halves. Emitting a row with one side filled in from
    # nowhere would put a fabricated number in the headline table.
    df = _per_image("unet", "a", "id_test", [0.9, 0.9])
    gap = generalization_gap(summarize(df))
    assert len(gap) == 0
    # An empty frame still needs its schema: a driver over a partial results
    # directory would otherwise write a headerless CSV or raise KeyError on a
    # column that exists in every non-empty case.
    assert "dice_gap" in gap.columns
    assert "hd95_n_ood" in gap.columns


# --- paired_compare -------------------------------------------------------

def test_paired_compare_detects_a_consistent_improvement():
    rng = np.random.default_rng(0)
    base = rng.uniform(0.6, 0.8, 40)
    a = _per_image("unet", "a", "ood_test", base)
    # A spread of improvements, not a flat offset: a constant makes every
    # resample and every Walsh average identical, so the interval assertions
    # below would hold no matter what the interval code did.
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test",
                   base + rng.uniform(0.01, 0.09, 40))
    out = paired_compare(a, b, metric="dice")
    assert out["n"] == 40
    assert out["median_diff"] == pytest.approx(0.05, abs=0.015)
    assert out["hl_estimate"] == pytest.approx(0.05, abs=0.015)
    assert out["p_value"] < 0.01
    assert out["ci_low"] > 0
    assert out["ci_high"] > out["ci_low"]


def test_paired_compare_finds_nothing_when_there_is_nothing():
    rng = np.random.default_rng(1)
    base = rng.uniform(0.6, 0.8, 40)
    a = _per_image("unet", "a", "ood_test", base)
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test",
                   base + rng.uniform(-0.05, 0.05, 40))
    out = paired_compare(a, b, metric="dice")
    assert out["ci_low"] < 0 < out["ci_high"]
    assert out["p_value"] > 0.05


def test_paired_compare_rejects_mismatched_patients():
    a = _per_image("unet", "a", "ood_test", [0.7, 0.8])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="patient_id"):
        paired_compare(a, b)


def test_paired_compare_pairs_by_patient_not_by_row_order():
    """Row order is not a pairing.

    The evaluator writes id_test before ood_test and the manifest order
    depends on the filesystem, so two CSVs can hold the same patients in
    different orders. Zipping positionally would then difference one
    patient's Dice against another's, which destroys the pairing the whole
    test depends on while still returning a plausible number.
    """
    scores = [0.10, 0.20, 0.30, 0.95]
    a = _per_image("unet", "a", "ood_test", scores)
    # Same patients, same scores plus a constant, rows reversed. Positional
    # pairing gives differences [0.90, 0.15, -0.15, -0.80], nothing like the
    # constant 0.05 the patients actually differ by.
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test",
                   [s + 0.05 for s in scores]).iloc[::-1].reset_index(drop=True)
    out = paired_compare(a, b, metric="dice")
    assert out["median_diff"] == pytest.approx(0.05)
    assert out["hl_estimate"] == pytest.approx(0.05)


def test_paired_compare_reports_b_minus_a_not_the_reverse():
    # A sign error inverts every conclusion in the paper while leaving the
    # magnitude and the p-value untouched.
    a = _per_image("unet", "a", "ood_test", [0.50, 0.52, 0.54, 0.56])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test",
                   [0.60, 0.63, 0.66, 0.69])
    out = paired_compare(a, b, metric="dice")
    assert out["median_diff"] > 0
    assert out["hl_estimate"] > 0
    assert paired_compare(b, a, metric="dice")["hl_estimate"] < 0


def test_paired_compare_counts_the_pairs_it_dropped():
    """HD95 is NaN exactly where the model failed completely.

    Dropping those pairs silently compares the two models only on the images
    where both succeeded, which is the subset that flatters whichever model
    fails more often. The count is what makes the restriction reportable.
    """
    a = _per_image("unet", "a", "ood_test", [0.8, 0.8, 0.0, 0.0],
                   hd95=[10.0, 10.0, np.nan, np.nan])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test",
                   [0.8, 0.8, 0.7, 0.7], hd95=[6.0, 5.0, 8.0, 8.0])
    out = paired_compare(a, b, metric="hd95")
    assert out["n"] == 2, "NaN pairs were counted as real comparisons"
    assert out["n_dropped"] == 2
    assert out["median_diff"] == pytest.approx(-4.5)


def test_tied_pairs_stay_in_the_sample_and_are_counted():
    """scipy's default drops exact ties and never says so.

    zero_method="wilcox" discards zero differences and shrinks the effective
    n, while the returned n here would still count them. Ties are common in
    this pipeline: two arms that both fail on the same off-domain image both
    score Dice 0, and post-processing collapses near-identical masks to
    identical ones. Keeping them is what makes a p-value driven by one image
    out of forty look like what it is.
    """
    base = np.full(40, 0.7)
    moved = base.copy()
    moved[0] += 0.2
    a = _per_image("unet", "a", "ood_test", base)
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", moved)
    out = paired_compare(a, b, metric="dice")
    assert out["n"] == 40
    assert out["n_zero"] == 39, "tied pairs are not reported"
    assert out["p_value"] > 0.5, (
        "one image out of forty produced a small p, so ties were discarded"
    )


def test_the_interval_is_a_95_percent_interval():
    """A narrower interval excludes zero more often, which is overclaiming.

    Nothing else in this file separates a 95% interval from a 90% one: both
    bracket the estimate and both shrink with n. The distinguishing property
    is width. For n differences uniform on (-1, 1) the Hodges-Lehmann
    estimator has asymptotic SD 1/sqrt(12*n*(integral of f squared)^2), which
    at n=400 is 0.0289, so the half-width is 1.96*0.0289 = 0.057 at 95%
    against 1.645*0.0289 = 0.048 at 90%.
    """
    n = 400
    rng = np.random.default_rng(11)
    a = _per_image("unet", "a", "ood_test", np.zeros(n))
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", rng.uniform(-1, 1, n))
    out = paired_compare(a, b, metric="dice")
    # Each tail separately, so widening only one of them is caught too.
    e = out["hl_estimate"]
    assert e - out["ci_low"] == pytest.approx(0.057, abs=0.005), "lower tail is not 2.5%"
    assert out["ci_high"] - e == pytest.approx(0.057, abs=0.005), "upper tail is not 97.5%"


def test_the_interval_agrees_with_the_test_it_is_reported_beside():
    """An interval on a different parameter than the test contradicts it.

    The signed-rank test is about the pseudomedian, so the interval has to be
    about the pseudomedian too. A results row reading p < 0.05 beside an
    interval containing zero, or the reverse, is an unforced contradiction in
    the paper, and skewed off-domain difference distributions are exactly
    where a median-based interval and a signed-rank test come apart.
    """
    rng = np.random.default_rng(5)
    # Mostly small positive gains with a heavy left tail of failures, which is
    # the shape of a real off-domain Dice difference.
    base = rng.uniform(0.5, 0.9, 60)
    delta = rng.uniform(0.01, 0.06, 60)
    delta[:8] = rng.uniform(-0.6, -0.3, 8)
    a = _per_image("unet", "a", "ood_test", base)
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", base + delta)
    out = paired_compare(a, b, metric="dice")
    rejects = out["p_value"] < 0.05
    excludes_zero = out["ci_low"] > 0 or out["ci_high"] < 0
    assert rejects == excludes_zero, (
        f"test and interval disagree: p={out['p_value']}, "
        f"ci=({out['ci_low']}, {out['ci_high']})"
    )


def test_paired_compare_refuses_two_frames_from_different_conditions():
    """Matching patient sets are not enough to make a pairing meaningful.

    Every per-image CSV holds a raw and a post-processed row for every
    patient, at one resolution, for one split. Each of those conditions covers
    the same patients, so forgetting one filter on one side passes the
    patient_id check and returns a significant improvement of a run over
    itself.
    """
    scores = [0.6, 0.65, 0.7, 0.75]
    shifted = [s + 0.05 for s in scores]
    base = dict(arm="unet_resnet34_imagenet", held_out="a", split="ood_test",
                dices=shifted)
    a = _per_image("unet", "a", "ood_test", scores)
    for override, pattern in [
        ({"postprocessed": True}, "postprocessed"),
        ({"image_size": 256}, "image_size"),
        ({"seed": 43}, "seed"),
        ({"split": "id_test"}, "split"),
        ({"held_out": "b"}, "held_out"),
    ]:
        b = _per_image(**{**base, **override})
        with pytest.raises(ValueError, match=pattern):
            paired_compare(a, b, metric="dice")


def test_paired_compare_refuses_an_arm_against_itself():
    # The same arm on both sides is a filter that did not filter. Every
    # difference is zero, and "no significant difference between the two
    # ResNet arms" is a publishable-looking conclusion to reach by accident.
    a = _per_image("unet", "a", "ood_test", [0.7, 0.8, 0.9])
    with pytest.raises(ValueError, match="itself"):
        paired_compare(a, a.copy(), metric="dice")


def test_paired_compare_survives_two_identical_arms():
    # Every difference is zero, which is the degenerate case wilcoxon refuses
    # to handle. It must report "no difference" rather than crash the whole
    # report driver.
    a = _per_image("unet", "a", "ood_test", [0.7, 0.8, 0.9])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.7, 0.8, 0.9])
    out = paired_compare(a, b, metric="dice")
    assert out["median_diff"] == pytest.approx(0.0)
    assert out["p_value"] == pytest.approx(1.0)
    assert out["n_zero"] == 3
    # Handing an all-zero sample to scipy divides by a zero standard error.
    # It happens to return p=1.0 through a RuntimeWarning, with a statistic of
    # 7.5 that means nothing, and that statistic would be printed in a table.
    assert out["wilcoxon_stat"] == 0.0


def test_the_estimate_is_the_pseudomedian_not_the_median():
    """Hand-computed on three pairs, where the two estimators visibly differ.

    The Hodges-Lehmann estimate is the median of all Walsh averages including
    each difference paired with itself. Differences of [0, 0, 0.4] give the
    six averages [0, 0, 0, 0.2, 0.2, 0.4], whose median is 0.1, while the
    median of the differences themselves is 0. Dropping the diagonal leaves
    [0, 0.2, 0.2] and an estimate of 0.2, which is the wrong parameter and
    twice the right answer here.
    """
    a = _per_image("unet", "a", "ood_test", [0.5, 0.5, 0.5])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.5, 0.5, 0.9])
    out = paired_compare(a, b, metric="dice")
    assert out["median_diff"] == pytest.approx(0.0)
    assert out["hl_estimate"] == pytest.approx(0.1)


def test_paired_compare_rejects_a_repeated_patient():
    # set_index on a duplicated id silently broadcasts, so one patient would
    # contribute several pairs and the effective n would be a fiction.
    a = _per_image("unet", "a", "ood_test", [0.7, 0.8])
    a = pd.concat([a, a.iloc[[0]]])
    b = _per_image("unet_resnet34_imagenet", "a", "ood_test", [0.7, 0.8])
    with pytest.raises(ValueError, match="duplicate"):
        paired_compare(a, b)


# --- holm -----------------------------------------------------------------

def test_holm_adjusts_the_family_this_study_actually_runs():
    """Nine paired tests per metric, uncorrected, manufacture a result.

    Three arm pairs over three folds at alpha 0.05 is expected to produce
    roughly one spurious rejection whether or not any effect exists, and the
    pretraining claim is exactly the kind that gets manufactured that way.
    """
    raw = [0.001, 0.02, 0.03, 0.04, 0.20, 0.30, 0.40, 0.50, 0.60]
    adj = holm(raw)
    assert adj[0] == pytest.approx(0.009)  # 0.001 * 9
    assert adj[1] == pytest.approx(0.16)   # 0.02 * 8
    assert (adj >= np.array(raw)).all(), "adjustment made a p-value smaller"
    assert (adj <= 1.0).all()
    # The three borderline results do not survive the family.
    assert sum(p < 0.05 for p in raw) == 4
    assert sum(p < 0.05 for p in adj) == 1


def test_holm_returns_results_in_the_order_given():
    # Returning them sorted would silently reattach each adjusted p to the
    # wrong comparison, which is worse than not correcting at all.
    adj = holm([0.40, 0.01, 0.10])
    assert adj[1] < adj[2] < adj[0]
    assert adj[1] == pytest.approx(0.03)  # smallest, multiplied by 3
    assert adj[2] == pytest.approx(0.20)  # middle, multiplied by 2


def test_holm_is_monotone_so_a_larger_p_never_becomes_more_significant():
    adj = holm([0.01, 0.011, 0.012, 0.013])
    assert list(adj) == sorted(adj)
