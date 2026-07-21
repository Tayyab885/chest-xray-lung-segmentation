"""Qualitative overlays and the summary chart.

Worst cases are included on purpose. A figure that shows only successes is an
advertisement, not evidence.

Everything here is keyed on `arm`, never on `model`. Both ResNet arms carry the
model name unet_resnet34, so a chart labelled by model puts two identical
captions on the one contrast this project exists to measure.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.ndimage import binary_erosion  # noqa: E402

from src.data import load_image, load_mask  # noqa: E402
from src.evaluate import load_checkpoint  # noqa: E402
from src.model import build_model  # noqa: E402
from src.stats import single_value  # noqa: E402
from src.train import ARMS, assert_gpu_usable  # noqa: E402

GT_COLOR = (0.0, 0.8, 0.2, 1.0)
PRED_COLOR = (1.0, 0.3, 0.0, 1.0)


def pick_examples(per_image):
    """Best, median, and worst off-domain cases by raw Dice.

    Raw, not post-processed: the panel should show what the network predicted,
    not what the cleanup step rescued.
    """
    df = per_image[(per_image["split"] == "ood_test") & (~per_image["postprocessed"])]
    if df.empty:
        raise ValueError(
            "no off-domain rows to illustrate: the frame passed in holds only "
            f"splits {sorted(per_image['split'].unique())}"
        )

    # One panel is drawn from one checkpoint while its caption comes from these
    # rows. If the rows span two arms or two seeds the contour and the number
    # under it can come from different networks, and both are real numbers.
    for column in ("arm", "held_out", "seed", "image_size"):
        single_value(df, column, "pick_examples")

    # patient_id breaks ties. run_report concatenates the per-image CSVs in
    # glob order, so sorting on dice alone would leave tied rows in input order
    # and the median panel would change between runs on identical data.
    df = df.sort_values(["dice", "patient_id"]).reset_index(drop=True)
    picks = [
        ("best", df.iloc[-1]),
        ("median", df.iloc[len(df) // 2]),
        ("worst", df.iloc[0]),
    ]
    return [{"label": label, "patient_id": row["patient_id"], "dice": row["dice"]}
            for label, row in picks]


def _outline(mask):
    return np.logical_xor(mask, binary_erosion(mask, border_value=0))


def _rgba(mask, color):
    out = np.zeros((*mask.shape, 4))
    out[_outline(mask)] = color
    return out


@torch.no_grad()
def overlay_panel(cfg, checkpoint_path, fold, arm, per_image, out_path,
                  return_figure=False):
    """Render best, median, and worst off-domain predictions side by side."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}, expected one of {sorted(ARMS)}")
    model_name, _ = ARMS[arm]

    device = cfg.get("device") or assert_gpu_usable()
    state = load_checkpoint(checkpoint_path, device, arm, fold.held_out, cfg)

    # The contour below is recomputed from these weights while the caption is
    # read out of per_image. load_checkpoint has only compared the checkpoint
    # with the config, and pick_examples only checks that per_image agrees with
    # itself, so a stale CSV sitting beside a freshly retrained checkpoint
    # passes both: each side is separately consistent and they describe
    # different runs.
    for column, want in [("arm", arm), ("held_out", fold.held_out),
                         ("seed", state["seed"]),
                         ("image_size", state["image_size"])]:
        got = single_value(per_image, column, "the per-image rows")
        if got != want:
            raise ValueError(
                f"per-image rows have {column}={got!r}, but this panel would be "
                f"drawn from a checkpoint with {column}={want!r}"
            )

    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    picks = pick_examples(per_image)
    size = cfg["image_size"]
    fig, axes = plt.subplots(1, len(picks), figsize=(4 * len(picks), 4.4))

    for ax, pick in zip(np.atleast_1d(axes), picks):
        match = fold.ood_test[fold.ood_test["patient_id"] == pick["patient_id"]]
        if match.empty:
            # Scores from one fold against another fold's images. Indexing the
            # empty match raises IndexError from inside pandas, which reads as
            # a plotting bug rather than a mismatched results directory.
            raise ValueError(
                f"{pick['patient_id']!r} is not in the held-out set for fold "
                f"{fold.held_out!r}"
            )
        row = match.iloc[0]
        image = load_image(row["image_path"], size)
        gt = load_mask(row["mask_paths"], size)
        x = torch.from_numpy(image).float()[None, None].to(device)
        pred = torch.sigmoid(model(x)).cpu().numpy()[0, 0] > 0.5

        ax.imshow(image, cmap="gray")
        ax.imshow(_rgba(gt, GT_COLOR))
        ax.imshow(_rgba(pred, PRED_COLOR))
        ax.set_title(f"{pick['label']}, Dice {pick['dice']:.3f}", fontsize=11)
        ax.axis("off")

    fig.suptitle(
        f"{arm}, held out {fold.held_out} "
        "(green: reference, orange: prediction)", fontsize=12)
    fig.tight_layout()
    _save(fig, out_path)
    return fig if return_figure else Path(out_path)


def _save(fig, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def gap_chart(gap_table, out_path, return_figure=False):
    """Grouped bars: in-domain vs off-domain Dice for each fold and arm."""
    df = gap_table[~gap_table["postprocessed"]]
    if df.empty:
        # An empty results directory would otherwise produce a blank PNG that a
        # README embeds without complaint.
        raise ValueError(
            "no rows to chart: the gap table has no un-post-processed rows"
        )
    df = df.sort_values(["arm", "held_out"]).reset_index(drop=True)

    labels = [f"{r.arm}\nheld out: {r.held_out}" for r in df.itertuples()]
    x = np.arange(len(df))
    width = 0.38

    fig, ax = plt.subplots(figsize=(1.9 * len(df) + 2, 4.6))
    ax.bar(x - width / 2, df["dice_id"], width, label="in-domain test")
    ax.bar(x + width / 2, df["dice_ood"], width, label="off-domain test")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Dice")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.set_title("Lung field segmentation: in-domain vs unseen source")
    fig.tight_layout()

    _save(fig, out_path)
    # The saved file is the deliverable; the figure is returned only so a test
    # can read back the bar heights and the labels it was drawn with.
    return fig if return_figure else Path(out_path)
