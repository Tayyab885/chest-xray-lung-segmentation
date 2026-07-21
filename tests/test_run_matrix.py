from itertools import groupby

import pandas as pd
import pytest

from src.run_matrix import (ARM_ORDER, per_image_path, plan_runs,
                            restore_previous, run_all)
from src.splits import lodo_folds
from src.train import ARMS


@pytest.fixture
def folds(fake_frame):
    manifest = pd.concat(
        [fake_frame(8, source=s) for s in ("jsrt", "montgomery", "shenzhen")],
        ignore_index=True)
    return lodo_folds(manifest, 0.15, 0.15, 42)


@pytest.fixture
def cfg(tmp_path):
    return {"results_dir": str(tmp_path / "results"),
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "seed": 42}


def _fakes(log=None, fail_on=(), durations=None, clock=None):
    """Stand-ins for train_one and evaluate_run.

    The matrix loop is what these tests are about. Training for real would put
    a GPU and forty minutes between the test and the behaviour it checks.
    """
    def train(cfg, fold, arm):
        if log is not None:
            log.append((fold.held_out, arm))
        if durations is not None and clock is not None:
            clock.advance(durations.get((fold.held_out, arm), 1.0))
        return f"{arm}_{fold.held_out}.pt"

    def evaluate(cfg, ckpt, fold, arm):
        if (fold.held_out, arm) in fail_on:
            raise RuntimeError("CUDA out of memory")
        return pd.DataFrame({"patient_id": ["a/0"], "arm": [arm],
                             "held_out": [fold.held_out], "dice": [0.9]})
    return train, evaluate


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def advance(self, seconds):
        self.now += seconds

    def __call__(self):
        return self.now


def test_plan_covers_every_arm_on_every_fold(folds):
    plan = plan_runs(folds)
    assert len(plan) == len(folds) * len(ARMS)
    assert {(f.held_out, arm) for f, arm in plan} == {
        (f.held_out, arm) for f in folds for arm in ARMS}


def test_plan_finishes_a_fold_before_moving_to_the_next(folds):
    # A session killed part way through leaves whole folds rather than one arm
    # of three folds. The pretraining contrast is measured within a fold, so
    # scattered arms across folds would leave nothing comparable.
    order = [f.held_out for f, _ in plan_runs(folds)]
    blocks = [held_out for held_out, _ in groupby(order)]
    assert len(blocks) == len(folds)
    assert len(blocks) == len(set(blocks))


def test_arm_order_covers_every_arm():
    # A fourth arm added to ARMS and forgotten here would silently never run.
    assert set(ARM_ORDER) == set(ARMS)


def test_the_capacity_matched_pair_runs_before_the_baseline(folds):
    # Two reasons, and they point the same way. A fold cut short by the session
    # cap should lose the baseline rather than the pretraining contrast, and
    # timing the small arm first would tell the deadline check that a run three
    # times its size fits in the time left.
    first_fold = [arm for f, arm in plan_runs(folds) if f.held_out == folds[0].held_out]
    assert first_fold[-1] == "unet"
    assert set(first_fold[:2]) == {"unet_resnet34_imagenet", "unet_resnet34_scratch"}


def test_restore_brings_forward_results_from_an_earlier_session(cfg, folds, tmp_path):
    # Kaggle starts every committed session with an empty /kaggle/working, so
    # without this the resume check never sees anything and each session
    # retrains the matrix from the beginning.
    earlier = tmp_path / "earlier" / "per_image"
    earlier.mkdir(parents=True)
    (earlier / "unet_jsrt_seed42.csv").write_text("patient_id,dice\na/0,0.9\n")

    restored = restore_previous(cfg, [tmp_path / "earlier"])

    assert len(restored) == 1
    assert per_image_path(cfg, "unet", "jsrt").exists()


def test_restore_ignores_a_half_written_file_from_a_killed_session(cfg, tmp_path):
    earlier = tmp_path / "earlier" / "per_image"
    earlier.mkdir(parents=True)
    (earlier / "unet_jsrt_seed42.csv.partial").write_text("patient_id,di")

    assert restore_previous(cfg, [tmp_path / "earlier"]) == []
    assert not per_image_path(cfg, "unet", "jsrt").exists()


def test_restore_does_not_overwrite_this_session_s_results(cfg, tmp_path):
    fresh = per_image_path(cfg, "unet", "jsrt")
    fresh.parent.mkdir(parents=True, exist_ok=True)
    fresh.write_text("patient_id,dice\nfresh,0.5\n")

    earlier = tmp_path / "earlier" / "per_image"
    earlier.mkdir(parents=True)
    (earlier / "unet_jsrt_seed42.csv").write_text("patient_id,dice\nstale,0.9\n")

    restore_previous(cfg, [tmp_path / "earlier"])
    assert "fresh" in fresh.read_text()


def test_restore_tolerates_a_missing_directory(cfg, tmp_path):
    # The first session has no previous output to attach, and that is normal.
    assert restore_previous(cfg, [tmp_path / "nothing-here"]) == []


def test_restored_runs_are_not_retrained(cfg, folds, tmp_path):
    earlier = tmp_path / "earlier" / "per_image"
    earlier.mkdir(parents=True)
    (earlier / f"unet_{folds[0].held_out}_seed42.csv").write_text("patient_id,dice\na/0,0.9\n")
    restore_previous(cfg, [tmp_path / "earlier"])

    log = []
    train, evaluate = _fakes(log=log)
    run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    assert (folds[0].held_out, "unet") not in log


def test_run_all_writes_one_csv_per_run(cfg, folds):
    train, evaluate = _fakes()
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    written = sorted(p.name for p in result["written"])
    assert len(written) == len(folds) * len(ARMS)
    assert "unet_resnet34_imagenet_jsrt_seed42.csv" in written


def test_csv_name_carries_the_seed(cfg, folds):
    # Replicate seeds are separate runs. A name without the seed would have the
    # second replicate overwrite the first and the error bars would come from
    # one run reported twice.
    cfg["seed"] = 7
    assert per_image_path(cfg, "unet", "jsrt").name == "unet_jsrt_seed7.csv"


def test_run_all_skips_a_run_that_already_has_results(cfg, folds):
    train, evaluate = _fakes()
    run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    log = []
    train2, evaluate2 = _fakes(log=log)
    again = run_all(cfg, folds, train_fn=train2, eval_fn=evaluate2)

    # Kaggle sessions cap at 12 hours, so the matrix spans several of them.
    # Retraining what is already scored would spend the next session repeating
    # the last one.
    assert log == []
    assert len(again["skipped"]) == len(folds) * len(ARMS)


def test_a_half_written_csv_is_not_treated_as_finished(cfg, folds):
    train, evaluate = _fakes(fail_on=[(folds[0].held_out, "unet")])
    run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    # Nothing partial may survive a crash: run_report globs this directory and
    # would concatenate a truncated file without complaint, and the resume
    # check would call the fold done.
    assert not per_image_path(cfg, "unet", folds[0].held_out).exists()


class HalfWriter:
    """Scores whose write dies part way through, as a killed session's would."""

    def to_csv(self, path, index=False):
        with open(path, "w", encoding="utf-8") as f:
            f.write("patient_id,arm,held_out,di")
        raise OSError("no space left on device")


def test_a_write_that_dies_part_way_leaves_no_csv_behind(cfg, folds):
    def evaluate(cfg, ckpt, fold, arm):
        return HalfWriter() if fold.held_out == folds[0].held_out else pd.DataFrame(
            {"patient_id": ["a/0"], "dice": [0.9]})

    train, _ = _fakes()
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    # run_report globs this directory and concatenates whatever it finds, so a
    # truncated file would enter the results as a short fold rather than as an
    # error, and the resume check would never retry it.
    for arm in ARMS:
        assert not per_image_path(cfg, arm, folds[0].held_out).exists()
    assert len(result["failed"]) == len(ARMS)


def test_one_failed_run_does_not_abandon_the_rest(cfg, folds):
    failing = (folds[0].held_out, "unet")
    train, evaluate = _fakes(fail_on=[failing])
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    assert len(result["written"]) == len(folds) * len(ARMS) - 1
    assert [r for r, _ in result["failed"]] == [failing]


def test_failures_are_reported_not_swallowed(cfg, folds, capsys):
    train, evaluate = _fakes(fail_on=[(folds[0].held_out, "unet")])
    run_all(cfg, folds, train_fn=train, eval_fn=evaluate)

    out = capsys.readouterr().out
    assert "CUDA out of memory" in out


def test_run_all_stops_before_a_run_it_cannot_finish(cfg, folds):
    clock = FakeClock()
    train, evaluate = _fakes(durations={}, clock=clock)
    # Each fake run advances the clock by one second, so a budget of four
    # leaves room for three runs and the fourth would be cut off by the cap.
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate,
                     deadline=3.5, clock=clock)

    assert len(result["written"]) == 3
    assert len(result["remaining"]) == len(folds) * len(ARMS) - 3


def test_the_budget_check_uses_the_slowest_run_not_the_average(cfg, folds):
    clock = FakeClock()
    plan = plan_runs(folds)
    slow = {(plan[1][0].held_out, plan[1][1]): 100.0}
    train, evaluate = _fakes(durations=slow, clock=clock)

    # One fast run then one slow one: the mean says a third fits in the time
    # left, the slowest observed run says it does not. Guessing low gets the
    # session killed mid-run, which on Kaggle discards every result in it.
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate,
                     deadline=160.0, clock=clock)
    assert len(result["written"]) == 2


def test_no_deadline_means_no_early_stop(cfg, folds):
    train, evaluate = _fakes()
    result = run_all(cfg, folds, train_fn=train, eval_fn=evaluate)
    assert result["remaining"] == []
