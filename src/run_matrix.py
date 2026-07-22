"""Drive the full run matrix: every arm on every leave-one-dataset-out fold.

Kaggle caps a session at twelve hours and discards everything a session
produced if it is killed rather than allowed to finish. That shapes the whole
module: results are written per run rather than at the end, a finished run is
never repeated, and the loop stops itself while there is still time to exit
cleanly.
"""
import shutil
import time
import traceback
from pathlib import Path

from src.train import ARMS, train_one
from src.evaluate import evaluate_run


# Slowest first, which here is also confirmatory first. The two ResNet arms are
# the capacity-matched contrast the third arm was added to measure, so a fold
# cut short by the session cap should have lost the baseline rather than the
# comparison. It also fixes the budget estimate: the deadline check has no
# timing to work from until a run finishes, and timing the smallest arm first
# would tell it that a run three times that size fits in the time left.
ARM_ORDER = ["unet_resnet34_imagenet", "unet_resnet34_scratch", "unet"]


def per_image_path(cfg, arm, held_out):
    """Where one run's per-image scores live.

    Keyed on arm rather than model because both ResNet arms share a model name,
    and on seed because replicate seeds are separate runs.
    """
    return (Path(cfg["results_dir"]) / "per_image"
            / f"{arm}_{held_out}_seed{cfg['seed']}.csv")


def plan_runs(folds):
    """Every (fold, arm) pair, grouped so a fold finishes before the next starts.

    The pretraining contrast is paired within a fold, so a session that ends
    early should leave whole folds behind. One arm each across three folds
    would be nine hours of GPU that compares against nothing.
    """
    missing = set(ARMS) - set(ARM_ORDER)
    if missing:
        raise ValueError(f"ARM_ORDER does not cover {sorted(missing)}")
    return [(fold, arm) for fold in folds for arm in ARM_ORDER]


def restore_previous(cfg, sources):
    """Copy an earlier session's per-image CSVs into this session's results.

    Kaggle starts every committed session with an empty /kaggle/working, so the
    resume check in run_all sees nothing unless the previous session's output
    is attached as an input dataset and copied in first. Without this the nine
    runs can never accumulate across sessions: each one restarts the matrix.
    """
    out_dir = Path(cfg["results_dir"]) / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)

    restored = []
    for source in sources:
        # A path that does not exist yields nothing rather than raising, which
        # is what the first session needs: it has no earlier output to attach.
        source_dir = Path(source) / "per_image"
        # *.csv only. A .partial left by a killed session is a truncated run,
        # and copying it in would have run_all treat that run as finished.
        for csv in sorted(source_dir.glob("*.csv")):
            target = out_dir / csv.name
            if target.exists():
                # This session's own output wins. Overwriting it with an older
                # copy would silently report the earlier run's numbers.
                continue
            shutil.copy2(csv, target)
            restored.append(target)

    print(f"restored {len(restored)} run(s) from earlier sessions")
    return restored


def restore_checkpoints(cfg, sources):
    """Copy an earlier session's trained weights into this session.

    The overlay figures are drawn from the checkpoints, not the per-image CSVs,
    so a figures-only session needs the weights carried forward the same way
    restore_previous carries the scores. Same empty-/kaggle/working problem,
    same fix: attach the training session's output and copy its .pt files in
    before run_report renders anything.
    """
    out_dir = Path(cfg["checkpoint_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    restored = []
    for source in sources:
        source_dir = Path(source) / "checkpoints"
        for ckpt in sorted(source_dir.glob("*.pt")):
            target = out_dir / ckpt.name
            if target.exists():
                # This session's own weights win, exactly as in restore_previous.
                continue
            shutil.copy2(ckpt, target)
            restored.append(target)

    print(f"restored {len(restored)} checkpoint(s) from earlier sessions")
    return restored


def run_all(cfg, folds, deadline=None, clock=time.monotonic,
            train_fn=train_one, eval_fn=evaluate_run):
    """Train and score every pending run, and report what was left undone.

    `deadline` is a `clock()` reading, not a duration. Pass None to run the
    whole matrix.
    """
    out_dir = Path(cfg["results_dir"]) / "per_image"
    out_dir.mkdir(parents=True, exist_ok=True)

    written, skipped, failed, remaining = [], [], [], []
    longest = 0.0

    plan = plan_runs(folds)
    for position, (fold, arm) in enumerate(plan):
        run = (fold.held_out, arm)
        out_path = per_image_path(cfg, arm, fold.held_out)
        if out_path.exists():
            skipped.append(run)
            continue

        # Measured against the slowest run seen, not the mean. Underestimating
        # gets the session killed part way through a run, and Kaggle then keeps
        # nothing at all, including the runs that had already finished.
        if deadline is not None and longest and clock() + longest > deadline:
            remaining = [(f.held_out, a) for f, a in plan[position:]
                         if not per_image_path(cfg, a, f.held_out).exists()]
            print(f"stopping with {len(remaining)} runs left: the next one would "
                  f"not finish inside the budget")
            break

        started = clock()
        try:
            checkpoint = train_fn(cfg, fold, arm)
            scores = eval_fn(cfg, checkpoint, fold, arm)
            # Written beside the target and renamed, because a write killed
            # part way leaves a truncated CSV that run_report would
            # concatenate without complaint and the resume check above would
            # call finished. The rename is the only step that publishes it.
            temp_path = out_path.with_suffix(".csv.partial")
            scores.to_csv(temp_path, index=False)
            temp_path.replace(out_path)
        except Exception as exc:
            # One arm running out of memory should not discard the runs that
            # already succeeded, but it must not pass quietly either: a missing
            # CSV downstream is indistinguishable from a run never scheduled.
            # Printed to stdout rather than left to stderr, so the reason
            # survives in the committed notebook next to the run it belongs to.
            detail = traceback.format_exc()
            print(f"FAILED {arm} on {fold.held_out}: {exc!r}\n{detail}")
            failed.append((run, detail))
            continue
        finally:
            longest = max(longest, clock() - started)

        written.append(out_path)
        print(f"done {arm} on {fold.held_out} -> {out_path.name}")

    if failed:
        print(f"\n{len(failed)} run(s) failed: "
              + ", ".join(f"{a} on {h}" for (h, a), _ in failed))
    return {"written": written, "skipped": skipped,
            "failed": failed, "remaining": remaining}
