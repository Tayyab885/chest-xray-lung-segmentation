"""Aggregation and the paired model comparison.

Two run-level means differing by a fraction of a point is not evidence. The
model comparison is done per image on the identical test set, with a
signed-rank test and a distribution-free interval.

Grouping is by `arm`, never by `model`. unet_resnet34_imagenet and
unet_resnet34_scratch are the same architecture and differ only in
initialisation, which is the contrast this project exists to measure. Keying
on the model name averages them together and reports the effect of
pretraining as exactly zero.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm, wilcoxon

GROUP_KEYS = ["arm", "held_out", "split", "postprocessed"]

# Conditions that must be constant within a group and identical across the two
# sides of a paired comparison. Every one of them changes what the numbers
# mean: image_size changes the unit of hd95 and assd, seed changes which run
# produced them, and postprocessed and split change which images they are.
CONDITION_KEYS = ["image_size", "seed"]


def _require_unique_patients(frame, where):
    dupes = frame["patient_id"][frame["patient_id"].duplicated()].unique()
    if len(dupes):
        raise ValueError(
            f"{where} contains duplicate patient_id values: {sorted(dupes)[:5]}"
        )


def single_value(frame, column, where):
    values = frame[column].unique()
    if len(values) > 1:
        raise ValueError(f"{where} mixes {column} values: {sorted(values)}")
    return values[0]


def _one_model(series):
    # "first" would silently resolve a disagreement. arm maps to model 1:1
    # through ARMS, so two different models under one arm means the CSVs were
    # built from a mislabelled run, which is worth a crash.
    names = series.unique()
    if len(names) > 1:
        raise ValueError(f"one arm carries several model names: {sorted(names)}")
    return names[0]


def summarize(per_image):
    """Group-level means, with the denominators the means were taken over."""
    for column in CONDITION_KEYS:
        single_value(per_image, column, "summary")

    for keys, group in per_image.groupby(GROUP_KEYS, dropna=False):
        _require_unique_patients(group, f"group {dict(zip(GROUP_KEYS, keys))}")

    out = per_image.groupby(GROUP_KEYS, dropna=False).agg(
        model=("model", _one_model),
        dice_mean=("dice", "mean"),
        dice_std=("dice", "std"),
        iou_mean=("iou", "mean"),
        hd95_mean=("hd95", "mean"),
        hd95_std=("hd95", "std"),
        assd_mean=("assd", "mean"),
        n=("dice", "size"),
        # pandas skips NaN in mean, and hd95 is NaN whenever a boundary does
        # not exist, which on this data means the prediction was empty. Off
        # domain that is the total-failure subset, so hd95_mean is an average
        # over the images the model did not fail on. hd95_n is the denominator
        # that actually applied; n_empty is how many images it left out.
        hd95_n=("hd95", "count"),
        n_empty=("pred_empty", "sum"),
    ).reset_index()
    out["n_empty"] = out["n_empty"].astype(int)
    return out


GAP_COLUMNS = [
    "arm", "held_out", "postprocessed", "model",
    "dice_id", "dice_ood", "dice_gap",
    "hd95_id", "hd95_ood", "hd95_gap",
    "n_id", "n_ood", "hd95_n_id", "hd95_n_ood", "n_empty_id", "n_empty_ood",
]


def generalization_gap(summary):
    """One row per arm and fold: in-domain against off-domain.

    Both gaps are signed so that positive means worse off domain. Dice is
    higher-is-better so the gap is id minus ood; HD95 is lower-is-better so it
    is ood minus id.

    The HD95 denominators travel with the HD95 columns. hd95_gap is a
    difference of two means taken over different image counts whenever a
    prediction came back empty, and off domain that is common. A row showing
    n_ood=100 beside an hd95_ood averaged over 60 images would read as a mild
    degradation when 40 images failed outright.
    """
    rows = []
    keys = ["arm", "held_out", "postprocessed"]
    for key, group in summary.groupby(keys, dropna=False):
        idx = group.set_index("split")
        if "id_test" not in idx.index or "ood_test" not in idx.index:
            # A gap needs both halves. Filling one side in from nowhere would
            # put a fabricated number in the headline table.
            continue
        rows.append({
            "arm": key[0],
            "held_out": key[1],
            "postprocessed": key[2],
            "model": _one_model(idx["model"]),
            "dice_id": idx.loc["id_test", "dice_mean"],
            "dice_ood": idx.loc["ood_test", "dice_mean"],
            "dice_gap": idx.loc["id_test", "dice_mean"] - idx.loc["ood_test", "dice_mean"],
            "hd95_id": idx.loc["id_test", "hd95_mean"],
            "hd95_ood": idx.loc["ood_test", "hd95_mean"],
            "hd95_gap": idx.loc["ood_test", "hd95_mean"] - idx.loc["id_test", "hd95_mean"],
            "n_id": idx.loc["id_test", "n"],
            "n_ood": idx.loc["ood_test", "n"],
            "hd95_n_id": idx.loc["id_test", "hd95_n"],
            "hd95_n_ood": idx.loc["ood_test", "hd95_n"],
            "n_empty_id": idx.loc["id_test", "n_empty"],
            "n_empty_ood": idx.loc["ood_test", "n_empty"],
        })
    # Explicit columns so an all-incomplete results directory still yields a
    # frame with the right schema instead of a headerless CSV.
    return pd.DataFrame(rows, columns=GAP_COLUMNS)


def _walsh_averages(diff):
    """Sorted pairwise means, including each value with itself."""
    w = (diff[:, None] + diff[None, :]) / 2.0
    return np.sort(w[np.triu_indices(len(diff))])


def _hodges_lehmann_interval(diff, alpha=0.05):
    """Point estimate and distribution-free interval for the pseudomedian.

    This is the parameter the signed-rank test is about, so the estimate, the
    interval, and the p-value all describe one quantity. A bootstrap on the
    sample median would not: on the skewed difference distributions this study
    produces off domain, an interval on the median can straddle zero while the
    signed-rank test rejects, and a results table cannot carry both.

    The interval is the order statistics of the Walsh averages cut at the
    signed-rank critical value, using the normal approximation with a
    continuity correction.
    """
    n = len(diff)
    w = _walsh_averages(diff)
    m = len(w)
    z = norm.ppf(1.0 - alpha / 2.0)
    k = int(np.floor(n * (n + 1) / 4.0
                     - z * np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0) - 0.5))
    # Below roughly n=6 no interval at this level exists and k goes negative.
    # Clamping widens to the full range of Walsh averages, which is
    # conservative rather than falsely tight.
    k = max(k, 0)
    return float(np.median(w)), float(w[k]), float(w[m - 1 - k])


def paired_compare(df_a, df_b, metric="dice", alpha=0.05):
    """Compare two arms on the same images. Differences are b minus a."""
    _require_unique_patients(df_a, "df_a")
    _require_unique_patients(df_b, "df_b")

    # Matching patient sets are not enough. Every per-image CSV holds both a
    # raw and a post-processed row for every patient, and both cover the same
    # split at the same resolution, so forgetting one filter on one side
    # compares a run against itself and returns a significant improvement.
    for column in CONDITION_KEYS + ["split", "postprocessed", "held_out"]:
        left = single_value(df_a, column, "df_a")
        right = single_value(df_b, column, "df_b")
        if left != right:
            raise ValueError(
                f"paired comparison across different {column}: {left!r} vs {right!r}"
            )
    if single_value(df_a, "arm", "df_a") == single_value(df_b, "arm", "df_b"):
        raise ValueError("paired comparison of an arm against itself")

    # Sorting by patient_id, not zipping by position: the two frames can hold
    # the same patients in different orders, and a positional pairing would
    # difference one patient's score against another's while still returning a
    # plausible-looking number.
    a = df_a.set_index("patient_id")[metric].sort_index()
    b = df_b.set_index("patient_id")[metric].sort_index()
    if list(a.index) != list(b.index):
        raise ValueError("paired comparison needs the identical set of patient_ids")

    diff = (b - a).to_numpy(dtype=float)
    finite = np.isfinite(diff)
    n_dropped = int((~finite).sum())
    diff = diff[finite]
    if not len(diff):
        raise ValueError(f"no comparable pairs for metric {metric!r}")

    n_zero = int((diff == 0).sum())
    if n_zero == len(diff):
        # wilcoxon refuses an all-zero input, and this is what comparing a run
        # with itself looks like. Report no difference rather than take down
        # the whole report driver.
        stat, p = 0.0, 1.0
    else:
        # zero_method="zsplit" keeps tied pairs in the sample. The scipy
        # default discards them and shrinks the effective n with no signal in
        # the return value, so the reported n would describe a larger sample
        # than the one tested. Ties are common here: two arms that both fail
        # on the same off-domain image both score Dice 0.
        stat, p = wilcoxon(diff, zero_method="zsplit")

    estimate, ci_low, ci_high = _hodges_lehmann_interval(diff, alpha=alpha)

    return {
        "n": int(len(diff)),
        # Pairs where either side is NaN, which for hd95 and assd means the
        # prediction was empty. Dropping them compares the two arms only on
        # the images where both succeeded, which favours whichever arm fails
        # more often.
        "n_dropped": n_dropped,
        # Exact ties. They carry no information about direction but they do
        # dilute the effect, so a large n_zero beside a small p is a result
        # driven by a minority of the images.
        "n_zero": n_zero,
        "median_diff": float(np.median(diff)),
        "hl_estimate": estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "wilcoxon_stat": float(stat),
        # Uncorrected, and this study runs one of these per arm pair per fold
        # per metric. Pass the family through holm() before reading any of
        # them as significant.
        "p_value": float(p),
    }


def holm(p_values):
    """Holm-Bonferroni adjusted p-values, in the order given.

    Three arms over three folds is nine paired tests for the pretraining
    contrast alone, and more once the other metrics are reported. At alpha
    0.05 that family is expected to manufacture a significant result whether
    or not one exists, and the claim this study makes is exactly the kind that
    gets manufactured that way.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(running, 1.0)
    return adjusted
