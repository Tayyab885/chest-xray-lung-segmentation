"""Per-image evaluation, reported both raw and post-processed.

Keeping the two largest connected components flatters HD95 by deleting
spurious specks. Both numbers are written so the README can report the effect
instead of quietly benefiting from it.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import label
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_manifest
from src.dataset import LungDataset
from src.metrics import all_metrics
from src.model import build_model
from src.splits import lodo_folds
from src.train import ARMS, assert_gpu_usable, select_fold


def postprocess(mask, keep=2):
    """Keep the `keep` largest connected components.

    8-connectivity, not the scipy default of 4. A lung whose two halves meet
    only at a diagonal would otherwise count as two components, and keep=2
    could then delete a whole real lung while a speck elsewhere survives.
    """
    labelled, n = label(mask, structure=np.ones((3, 3)))
    if n <= keep:
        return mask.copy()
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0  # background
    biggest = np.argsort(sizes)[::-1][:keep]
    return np.isin(labelled, biggest)


@torch.no_grad()
def _predict_frame(model, frame, cfg, device):
    loader = DataLoader(
        LungDataset(frame, cfg["image_size"], augment=False),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"])
    preds, gts = [], []
    for images, masks in loader:
        logits = model(images.to(device))
        probs = torch.sigmoid(logits).cpu().numpy()[:, 0]
        preds.extend(probs > 0.5)
        gts.extend(masks.numpy()[:, 0] > 0.5)
    return preds, gts


def evaluate_run(cfg, checkpoint_path, fold, arm):
    """Score one checkpoint on the in-domain and off-domain test sets."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}, expected one of {sorted(ARMS)}")
    model_name, _ = ARMS[arm]

    device = cfg.get("device") or assert_gpu_usable()
    # weights_only=True: torch.load unpickles arbitrary objects by default,
    # which is arbitrary code execution on an untrusted checkpoint. The
    # checkpoint here holds only tensors, strings, and numbers.
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    name = Path(checkpoint_path).name
    missing = [k for k in ("arm", "held_out", "seed", "image_size") if k not in state]
    if missing:
        # A missing key must not be read as agreement. Defaulting to the
        # caller's value would turn each guard below into a no-op for exactly
        # the checkpoints whose origin is unknown.
        raise ValueError(f"checkpoint {name} lacks provenance keys {missing}")

    # Every one of these is a silent-wrong-number failure, not a crash. The
    # held_out check matters most: scoring a Shenzhen-trained checkpoint
    # against the Montgomery fold means the "off-domain" images were in its
    # training set, so the headline in-domain-minus-off-domain gap collapses
    # toward zero and the paper's conclusion inverts.
    for key, want in [("arm", arm), ("held_out", fold.held_out),
                      ("seed", cfg.get("seed")), ("image_size", cfg["image_size"])]:
        if want is not None and state[key] != want:
            raise ValueError(
                f"checkpoint {name} has {key}={state[key]!r}, "
                f"but this evaluation is configured for {key}={want!r}"
            )

    # pretrained=False regardless: the trained weights replace the encoder
    # anyway, so downloading ImageNet here would be wasted work.
    model = build_model(model_name, pretrained=False).to(device)
    model.load_state_dict(state["model_state"])
    model.eval()

    rows = []
    for split_name, frame in [("id_test", fold.id_test), ("ood_test", fold.ood_test)]:
        if not len(frame):
            continue
        preds, gts = _predict_frame(model, frame, cfg, device)
        if len(preds) != len(frame):
            # Zipping to the shorter list would give correct scores over a
            # wrong denominator, which nothing downstream could notice.
            raise ValueError(
                f"{split_name}: got {len(preds)} predictions for {len(frame)} images"
            )
        # shuffle=False above is what lets loader order be zipped against frame
        # order. Shuffling would leave every metric plausible and every
        # patient_id attached to somebody else's score.
        for (_, meta), pred, gt in zip(frame.iterrows(), preds, gts):
            for flag, mask in [(False, pred), (True, postprocess(pred, keep=2))]:
                rows.append({
                    "patient_id": meta["patient_id"],
                    "source": meta["source"],
                    "arm": arm,
                    "model": model_name,
                    "held_out": fold.held_out,
                    "split": split_name,
                    "postprocessed": flag,
                    "image_size": cfg["image_size"],
                    # HD95 and ASSD are NaN exactly when the prediction is
                    # empty, which is the total-failure case and is commonest
                    # off-domain. A plain .mean() downstream skips those rows,
                    # so the off-domain average would be taken only over the
                    # images where the model did not fail. This column makes
                    # the omission countable.
                    "pred_empty": not mask.any(),
                    **all_metrics(mask, gt),
                })

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--layout", default="configs/data_layout.yaml")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.layout, "r", encoding="utf-8") as f:
        layout = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed

    manifest = build_manifest(cfg["data_root"], layout)
    folds = lodo_folds(manifest, cfg["val_frac"], cfg["test_frac"], cfg["seed"])
    fold = select_fold(folds, args.held_out)

    df = evaluate_run(cfg, args.checkpoint, fold, args.arm)
    out_dir = Path(cfg["results_dir"]) / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Seed in the name for the same reason the checkpoints carry it: replicate
    # runs must not overwrite each other's per-image scores.
    out = out_dir / f"{args.arm}_{args.held_out}_seed{cfg['seed']}.csv"
    df.to_csv(out, index=False)
    print("wrote", out, len(df), "rows")


if __name__ == "__main__":
    main()
