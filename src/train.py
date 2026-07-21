"""Training for one (fold, arm) pair.

An "arm" is one column of the run matrix: a model plus an initialisation. The
three of them exist because scratch U-Net against a pretrained ResNet34 U-Net
differs in both initialisation and capacity, roughly 3x the parameters, so that
pair cannot support a claim about pretraining on its own. The two ResNet arms
are identical architectures differing only in init, which is the contrast that
isolates the ImageNet effect.
"""
import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import build_manifest
from src.dataset import LungDataset, worker_init_fn
from src.model import build_model
from src.splits import lodo_folds

# arm name -> (model name, pretrained)
ARMS = {
    "unet": ("unet", False),
    "unet_resnet34_imagenet": ("unet_resnet34", True),
    "unet_resnet34_scratch": ("unet_resnet34", False),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN picks convolution backward algorithms that are not deterministic by
    # default, so without this a rerun of the same seed on the same GPU gives a
    # different Dice. Costs some throughput and buys a number that reproduces,
    # which is the trade this project wants.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def assert_gpu_usable():
    """Return 'cuda' only if a real kernel launches.

    torch.cuda.is_available() reports that a device exists, not that this torch
    build supports its compute capability. Launching a kernel is the only
    honest check.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        torch.zeros(8, device="cuda").sum().item()
        return "cuda"
    except Exception:
        return "cpu"


def dice_bce_loss(logits, target, eps=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probs = torch.sigmoid(logits)
    # eps keeps an all-background batch from dividing zero by zero.
    num = 2.0 * (probs * target).sum() + eps
    den = probs.sum() + target.sum() + eps
    return bce + (1.0 - num / den)


@torch.no_grad()
def _val_dice(model, loader, device):
    model.eval()
    total, count = 0.0, 0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        pred = (torch.sigmoid(model(images)) > 0.5).float()
        num = 2.0 * (pred * masks).sum(dim=(1, 2, 3))
        den = pred.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3))
        # An empty prediction on an empty mask is a perfect match, not a zero.
        total += torch.where(den > 0, num / den, torch.ones_like(den)).sum().item()
        count += images.size(0)
    return total / max(count, 1)


def train_one(cfg, fold, arm):
    """Train one arm on one fold. Returns the checkpoint path."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}, expected one of {sorted(ARMS)}")
    model_name, pretrained = ARMS[arm]

    set_seed(cfg["seed"])
    device = cfg.get("device") or assert_gpu_usable()
    size = cfg["image_size"]

    # Explicit generator, not the global RNG: model construction draws from the
    # global stream and the arms draw different amounts, since the ImageNet arm
    # loads its encoder from disk and only initialises the decoder. Sharing the
    # stream would leave each arm with a different batch order, so the two
    # ResNet arms would differ in more than initialisation and the pretraining
    # contrast would no longer be clean.
    shuffle_rng = torch.Generator().manual_seed(cfg["seed"])
    train_loader = DataLoader(
        LungDataset(fold.train, size, augment=True, seed=cfg["seed"]),
        batch_size=cfg["batch_size"], shuffle=True, generator=shuffle_rng,
        num_workers=cfg["num_workers"], drop_last=False,
        worker_init_fn=worker_init_fn)
    val_loader = DataLoader(
        LungDataset(fold.val, size, augment=False),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"],
        worker_init_fn=worker_init_fn)

    model = build_model(model_name, pretrained=pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(cfg["epochs"], 1))

    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # Named by arm, not by model: both ResNet arms share a model name, and
    # naming by model would have the second run overwrite the first. The seed
    # is in the name so replicate runs, which is what error bars on the
    # pretraining delta need, do not overwrite each other either.
    ckpt_path = ckpt_dir / f"{arm}_{fold.held_out}_seed{cfg['seed']}.pt"

    best, since_best = -1.0, 0
    for epoch in range(cfg["epochs"]):
        model.train()
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            opt.zero_grad()
            loss = dice_bce_loss(model(images), masks)
            loss.backward()
            opt.step()
        sched.step()

        score = _val_dice(model, val_loader, device)
        print(f"[{arm}|held_out={fold.held_out}] epoch {epoch} val_dice {score:.4f}")

        if score > best:
            best, since_best = score, 0
            torch.save({
                "model_state": model.state_dict(),
                "arm": arm,
                "model_name": model_name,
                "pretrained": pretrained,
                "held_out": fold.held_out,
                "val_dice": best,
                "epoch": epoch,
            }, ckpt_path)
        else:
            since_best += 1
            if since_best >= cfg["patience"]:
                print(f"early stop at epoch {epoch}, best val_dice {best:.4f}")
                break

    return ckpt_path


def select_fold(folds, held_out):
    for fold in folds:
        if fold.held_out == held_out:
            return fold
    raise ValueError(
        f"no fold holds out {held_out!r}, available: "
        f"{sorted(f.held_out for f in folds)}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--layout", default="configs/data_layout.yaml")
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--held-out", required=True)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    with open(args.layout, "r", encoding="utf-8") as f:
        layout = yaml.safe_load(f)

    if args.seed is not None:
        cfg["seed"] = args.seed

    manifest = build_manifest(cfg["data_root"], layout)
    # Folds are built from the config seed so every arm and every replicate
    # sees the same partition; only training randomness follows --seed.
    folds = lodo_folds(manifest, cfg["val_frac"], cfg["test_frac"],
                       load_config(args.config)["seed"])
    print("checkpoint:", train_one(cfg, select_fold(folds, args.held_out), args.arm))


if __name__ == "__main__":
    main()
