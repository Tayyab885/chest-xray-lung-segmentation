import pandas as pd
import pytest

from src.splits import lodo_folds


@pytest.fixture
def manifest():
    rows = []
    for source, n in [("a", 40), ("b", 60), ("c", 20)]:
        for i in range(n):
            rows.append({
                "image_path": f"{source}/{i}.png",
                "mask_paths": [f"{source}/{i}_m.png"],
                "source": source,
                "patient_id": f"{source}/{i}",
            })
    return pd.DataFrame(rows)


def test_one_fold_per_source(manifest):
    folds = lodo_folds(manifest)
    assert len(folds) == 3
    assert {f.held_out for f in folds} == {"a", "b", "c"}


def test_held_out_source_never_leaks_into_training(manifest):
    for fold in lodo_folds(manifest):
        for part in (fold.train, fold.val, fold.id_test):
            assert fold.held_out not in set(part["source"])


def test_ood_test_is_the_entire_held_out_source(manifest):
    for fold in lodo_folds(manifest):
        expected = (manifest["source"] == fold.held_out).sum()
        assert len(fold.ood_test) == expected
        assert set(fold.ood_test["source"]) == {fold.held_out}


def test_partitions_are_disjoint_and_complete(manifest):
    for fold in lodo_folds(manifest):
        ids = pd.concat([fold.train, fold.val, fold.id_test, fold.ood_test])["patient_id"]
        assert ids.is_unique
        assert len(ids) == len(manifest)


def test_in_domain_partitions_are_stratified_by_source(manifest):
    for fold in lodo_folds(manifest, val_frac=0.25, test_frac=0.25):
        remaining = [s for s in ("a", "b", "c") if s != fold.held_out]
        for source in remaining:
            n_source = (manifest["source"] == source).sum()
            n_val = (fold.val["source"] == source).sum()
            assert n_val == pytest.approx(0.25 * n_source, abs=1)


def test_in_domain_test_set_is_also_stratified(manifest):
    for fold in lodo_folds(manifest, val_frac=0.25, test_frac=0.25):
        remaining = [s for s in ("a", "b", "c") if s != fold.held_out]
        for source in remaining:
            n_source = (manifest["source"] == source).sum()
            n_test = (fold.id_test["source"] == source).sum()
            assert n_test == pytest.approx(0.25 * n_source, abs=1)


def test_raises_on_fewer_than_two_sources(manifest):
    # Only JSRT is downloaded locally, so this is the manifest a contributor is
    # most likely to hand this function by accident. Leave-one-out of one source
    # leaves nothing to train on.
    single = manifest[manifest["source"] == "a"]
    with pytest.raises(ValueError, match="at least 2 sources"):
        lodo_folds(single)


def test_raises_on_empty_manifest(manifest):
    with pytest.raises(ValueError, match="at least 2 sources"):
        lodo_folds(manifest.iloc[:0])


def test_raises_when_a_source_is_too_small_to_stratify(manifest):
    # Rounding can hand a small source zero validation rows. Training on it while
    # it is absent from val and id_test would quietly break the stratification
    # the in-domain baseline depends on.
    tiny = manifest[manifest["source"] != "c"]
    tiny = pd.concat([tiny, manifest[manifest["source"] == "c"].iloc[:2]])
    with pytest.raises(ValueError, match="too few images"):
        lodo_folds(tiny, val_frac=0.15, test_frac=0.15)


def test_splits_are_deterministic_for_a_seed(manifest):
    a = lodo_folds(manifest, seed=1)[0]
    b = lodo_folds(manifest, seed=1)[0]
    assert list(a.val["patient_id"]) == list(b.val["patient_id"])


def test_different_seeds_give_different_splits(manifest):
    a = lodo_folds(manifest, seed=1)[0]
    b = lodo_folds(manifest, seed=2)[0]
    assert list(a.val["patient_id"]) != list(b.val["patient_id"])
