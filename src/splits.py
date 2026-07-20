"""Leave-one-dataset-out fold construction.

Each fold holds out one entire source. The remaining sources are partitioned
into train, validation, and an in-domain test set, stratified so that every
partition keeps each contributing source in proportion. The in-domain test set
exists so that the off-domain drop is measured against a like-for-like
baseline, not against validation numbers the model was selected on.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Fold:
    held_out: str
    train: pd.DataFrame
    val: pd.DataFrame
    id_test: pd.DataFrame
    ood_test: pd.DataFrame


def _stratified_three_way(pool, val_frac, test_frac, rng):
    val_parts, test_parts, train_parts = [], [], []

    for source, group in pool.groupby("source", sort=True):
        idx = rng.permutation(len(group))
        shuffled = group.iloc[idx]
        n = len(shuffled)
        n_val = int(round(val_frac * n))
        n_test = int(round(test_frac * n))
        if n_val == 0 or n_test == 0:
            raise ValueError(
                f"{source}: too few images ({n}) to stratify at val_frac={val_frac}, "
                f"test_frac={test_frac}. Rounding gives {n_val} validation and "
                f"{n_test} in-domain test images, so the source would train without "
                "appearing in either evaluation set."
            )
        val_parts.append(shuffled.iloc[:n_val])
        test_parts.append(shuffled.iloc[n_val:n_val + n_test])
        train_parts.append(shuffled.iloc[n_val + n_test:])

    return (
        pd.concat(train_parts).reset_index(drop=True),
        pd.concat(val_parts).reset_index(drop=True),
        pd.concat(test_parts).reset_index(drop=True),
    )


def lodo_folds(manifest, val_frac=0.15, test_frac=0.15, seed=42):
    """Build one leave-one-dataset-out fold per source."""
    sources = sorted(manifest["source"].unique())
    if len(sources) < 2:
        raise ValueError(
            f"leave-one-dataset-out needs at least 2 sources, got {sources}. "
            "Holding one out of one leaves nothing to train on."
        )

    folds = []
    for held_out in sources:
        # Reseeded per fold on purpose: each fold's split is then reproducible on
        # its own, without depending on how many folds ran before it.
        rng = np.random.default_rng(seed)
        ood = manifest[manifest["source"] == held_out].reset_index(drop=True)
        pool = manifest[manifest["source"] != held_out]
        train, val, id_test = _stratified_three_way(pool, val_frac, test_frac, rng)
        folds.append(Fold(held_out, train, val, id_test, ood))
    return folds
