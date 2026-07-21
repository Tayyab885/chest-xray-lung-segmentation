"""Read every per-image CSV and write the tables and figures the README cites."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.config import load_config
from src.data import build_manifest
from src.figures import gap_chart, overlay_panel
from src.splits import lodo_folds
from src.stats import generalization_gap, holm, paired_compare, single_value, summarize
from src.train import ARMS


# Ordered so that b minus a reads as the effect being claimed: the pretrained
# arm against the scratch arm, and the larger model against the smaller one.
# Alphabetical order would report the pretraining contrast backwards.
ARM_PAIRS = [
    ("unet_resnet34_scratch", "unet_resnet34_imagenet"),
    ("unet", "unet_resnet34_scratch"),
    ("unet", "unet_resnet34_imagenet"),
]

# The first pair holds architecture fixed and varies only initialisation, so
# it is the only one that can support a claim about pretraining. The other two
# differ by roughly 3x the parameters as well, so they describe capacity and
# initialisation together and were never able to carry that claim.
FAMILIES = {
    ("unet_resnet34_scratch", "unet_resnet34_imagenet"): "pretraining",
}
DEFAULT_FAMILY = "capacity"


def compare_arms(per_image, metric="dice"):
    """Every arm pair, on every fold and split, corrected within its family.

    Three folds and two splits give the confirmatory pretraining contrast six
    tests, which at alpha 0.05 is already enough to produce a significant
    result whether or not one exists, so the corrected column is the one to
    quote.

    The correction is applied per family rather than over all eighteen rows at
    once. Pooling would hold the one capacity-matched contrast to a threshold
    three times stricter than the hypothesis it was designed for, and it would
    be corrected away by the company it keeps rather than by weak evidence.
    Splitting it is also the more conservative reading of the two exploratory
    families, which are descriptive and were never the claim.
    """
    # Every CSV holds each patient twice, raw and post-processed. Both rows
    # carry the same patient_id, so leaving them in would break the pairing.
    raw = per_image[~per_image["postprocessed"]]
    # Checked here rather than left to paired_compare, which sees only one arm
    # at a time. Two seeds of one arm are two runs, so they arrive as repeated
    # patient_ids and paired_compare reports duplicate IDs, which reads as a
    # broken manifest instead of two runs concatenated. Resolution needs no
    # such guard: mixed image_size reaches paired_compare intact and is named
    # there.
    single_value(raw, "seed", "comparison")

    rows = []
    for held_out in sorted(raw["held_out"].unique()):
        for split in ("id_test", "ood_test"):
            sel = raw[(raw["held_out"] == held_out) & (raw["split"] == split)]
            present = set(sel["arm"])
            for arm_a, arm_b in ARM_PAIRS:
                if not {arm_a, arm_b} <= present:
                    # A partial results directory is the normal state part way
                    # through the run matrix. Comparing against an absent arm
                    # would pair a real frame with an empty one.
                    continue
                out = paired_compare(sel[sel["arm"] == arm_a],
                                     sel[sel["arm"] == arm_b], metric=metric)
                out.update({
                    "family": FAMILIES.get((arm_a, arm_b), DEFAULT_FAMILY),
                    "held_out": held_out, "split": split, "metric": metric,
                    "arm_a": arm_a, "arm_b": arm_b,
                    # Spelled out because a sign read backwards inverts the
                    # conclusion, and "b minus a" is meaningless in a CSV.
                    "comparison": f"{arm_b} minus {arm_a}",
                })
                rows.append(out)

    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm"] = np.nan
    table["family_size"] = 0
    for _, index in table.groupby("family").groups.items():
        table.loc[index, "p_holm"] = holm(table.loc[index, "p_value"].to_numpy())
        table.loc[index, "family_size"] = len(index)
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--layout", default="configs/data_layout.yaml")
    ap.add_argument("--seed", type=int, default=None,
                    help="report on this seed's runs, defaults to the config seed")
    ap.add_argument("--figures", action="store_true",
                    help="also render overlay panels, needs checkpoints present")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    results = Path(cfg["results_dir"])
    per_image_dir = results / "per_image"

    csvs = sorted(per_image_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"no per-image CSVs in {per_image_dir}, run evaluate.py first")
    per_image = pd.concat([pd.read_csv(p) for p in csvs], ignore_index=True)

    # Replicate seeds are separate runs, not more data. Averaging across them
    # would put each patient in the table several times and shrink every
    # interval as though the sample had grown.
    seeds = sorted(per_image["seed"].unique())
    per_image = per_image[per_image["seed"] == cfg["seed"]]
    if per_image.empty:
        raise SystemExit(f"no rows for seed {cfg['seed']}, CSVs hold seeds {seeds}")

    summary = summarize(per_image)
    gap = generalization_gap(summary)
    summary.to_csv(results / "summary.csv", index=False)
    gap.to_csv(results / "generalization_gap.csv", index=False)

    comparisons = compare_arms(per_image)
    if comparisons.empty:
        # Otherwise a run where no fold had two arms in it looks exactly like a
        # run where the comparison was deliberately skipped.
        print("no fold and split had two arms in it, no model_comparison.csv")
    else:
        comparisons.to_csv(results / "model_comparison.csv", index=False)

    fig_dir = results / "figures"
    gap_chart(gap, fig_dir / "generalization_gap.png")
    print("wrote tables and gap chart to", results)

    if args.figures:
        with open(args.layout, "r", encoding="utf-8") as f:
            layout = yaml.safe_load(f)
        manifest = build_manifest(cfg["data_root"], layout)
        folds = lodo_folds(manifest, cfg["val_frac"], cfg["test_frac"], cfg["seed"])
        ckpt_dir = Path(cfg["checkpoint_dir"])
        for fold in folds:
            for arm in sorted(ARMS):
                ckpt = ckpt_dir / f"{arm}_{fold.held_out}_seed{cfg['seed']}.pt"
                sel = per_image[(per_image["arm"] == arm)
                                & (per_image["held_out"] == fold.held_out)]
                if not ckpt.exists() or sel.empty:
                    continue
                out = overlay_panel(
                    cfg, ckpt, fold, arm, sel,
                    fig_dir / f"overlay_{arm}_{fold.held_out}.png")
                print("wrote", out)


if __name__ == "__main__":
    main()
