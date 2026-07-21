import pytest
import torch

from src.splits import Fold
from src.train import (
    ARMS,
    _val_dice,
    assert_gpu_usable,
    dice_bce_loss,
    set_seed,
    train_one,
)


@pytest.fixture
def fold(fake_frame):
    # All four frames are disjoint on purpose. Aliasing them to one another
    # would make the worst failure in this pipeline invisible: selecting the
    # model on the held-out dataset would inflate off-domain Dice and collapse
    # the in-domain-minus-off-domain headline toward zero, with every test
    # still green.
    return Fold(held_out="c",
                train=fake_frame(4, "a"),
                val=fake_frame(2, "b"),
                id_test=fake_frame(2, "d"),
                ood_test=fake_frame(2, "c"))


@pytest.fixture
def cfg(tmp_path):
    return {
        "image_size": 32, "batch_size": 2, "epochs": 1, "lr": 1e-3,
        "weight_decay": 0.0, "patience": 1, "num_workers": 0, "seed": 0,
        "checkpoint_dir": str(tmp_path / "ckpt"), "device": "cpu",
    }


def test_loss_is_lower_for_a_better_prediction():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    good = (target * 6.0) - 3.0     # confident and correct
    bad = (1 - target) * 6.0 - 3.0  # confident and inverted
    assert dice_bce_loss(good, target) < dice_bce_loss(bad, target)


def test_loss_is_positive_and_finite():
    target = torch.zeros(1, 1, 16, 16)
    target[..., 4:12, 4:12] = 1.0
    loss = dice_bce_loss(torch.zeros_like(target), target)
    assert torch.isfinite(loss) and loss > 0


def test_loss_is_finite_on_a_confident_empty_target():
    # A confident model on an all-background mask underflows sigmoid to exactly
    # zero, so both Dice terms are zero. Without smoothing that is 0/0 and the
    # run dies with a NaN partway through an epoch. -6 logits are not enough to
    # trigger it; the sigmoid there is still 0.0025.
    target = torch.zeros(1, 1, 16, 16)
    loss = dice_bce_loss(torch.full_like(target, -100.0), target)
    assert torch.isfinite(loss)


def test_set_seed_makes_torch_deterministic():
    set_seed(3)
    a = torch.randn(4)
    set_seed(3)
    assert torch.allclose(a, torch.randn(4))


def test_assert_gpu_usable_returns_a_device_string():
    assert assert_gpu_usable() in {"cuda", "cpu"}


def test_train_one_runs_and_writes_a_checkpoint(cfg, fold):
    path = train_one(cfg, fold, "unet")
    assert path.exists()
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert "model_state" in state
    assert state["model_name"] == "unet"
    assert state["arm"] == "unet"
    assert state["held_out"] == "c"


def test_val_dice_is_per_image_and_thresholded_at_half():
    # Two images: one predicted perfectly, one predicted empty against a
    # non-empty mask. Per-image Dice is (1.0 + 0.0) / 2. Pooling over the batch
    # instead would weight by lung area and give a different, larger number,
    # and a threshold below 0.5 would let near-zero logits count as lung.
    masks = torch.zeros(2, 1, 8, 8)
    masks[0, ..., 2:6, 2:6] = 1.0
    masks[1, ..., 0:2, 0:2] = 1.0
    # -2.0 is sigmoid 0.119: clearly background at the 0.5 threshold, but above
    # any sloppier cutoff, so lowering the threshold changes the answer.
    logits = torch.full((2, 1, 8, 8), -2.0)
    logits[0, ..., 2:6, 2:6] = 3.0  # correct on image 0, all background on 1

    model = lambda x: logits  # noqa: E731
    model.eval = lambda: None
    loader = [(torch.zeros(2, 1, 8, 8), masks)]
    assert _val_dice(model, loader, "cpu") == pytest.approx(0.5)


def test_val_dice_scores_an_empty_prediction_on_an_empty_mask_as_perfect():
    masks = torch.zeros(1, 1, 8, 8)
    model = lambda x: torch.full((1, 1, 8, 8), -9.0)  # noqa: E731
    model.eval = lambda: None
    assert _val_dice(model, [(masks, masks)], "cpu") == pytest.approx(1.0)


def test_training_actually_updates_the_weights(cfg, fold):
    # Without this, deleting opt.step() leaves a suite that still passes: every
    # other test only checks that a checkpoint file exists with the right keys,
    # which a train_one that does no training at all would satisfy.
    from src.model import build_model

    cfg["epochs"] = 2
    path = train_one(cfg, fold, "unet")
    trained = torch.load(path, map_location="cpu", weights_only=True)["model_state"]

    set_seed(cfg["seed"])  # train_one seeds then builds, so this is its init
    fresh = build_model("unet", pretrained=False)
    initial = fresh.state_dict()

    # Learnable parameters only. BatchNorm running_mean and running_var are
    # buffers that move on every forward pass, so comparing the whole state
    # dict would call a model with no optimiser step "trained".
    learnable = [k for k, _ in fresh.named_parameters()]
    changed = [k for k in learnable if not torch.allclose(initial[k], trained[k])]
    assert changed, "weights identical to initialisation, the optimiser never stepped"


def test_same_seed_reproduces_the_same_weights(cfg, fold):
    # The README will claim seed 42 reproduces. If train_one stopped seeding,
    # nothing else in this suite would notice, and every number in the results
    # table would become unrepeatable.
    cfg["epochs"] = 1
    first = torch.load(train_one(cfg, fold, "unet"), map_location="cpu",
                       weights_only=True)["model_state"]
    second = torch.load(train_one(cfg, fold, "unet"), map_location="cpu",
                        weights_only=True)["model_state"]
    assert all(torch.allclose(first[k], second[k]) for k in first)


def test_best_checkpoint_and_early_stopping_follow_the_validation_curve(
        cfg, fold, monkeypatch):
    # The one test that exercises the selection logic. Every other call runs a
    # single epoch, where best starts at -1.0 so the first score always wins
    # and the entire early-stopping branch is dead code.
    import src.train as train_module

    scores = [0.5, 0.3, 0.9, 0.4, 0.4, 0.9, 0.9, 0.9]
    seen = []

    def scripted(model, loader, device):
        seen.append(len(seen))
        return scores[len(seen) - 1]

    monkeypatch.setattr(train_module, "_val_dice", scripted)
    cfg["epochs"] = 8
    cfg["patience"] = 2
    path = train_one(cfg, fold, "unet")

    # 0.9 at epoch 2, then two non-improving epochs, so it stops after epoch 4.
    assert len(seen) == 5, "early stopping did not fire on the patience budget"
    state = torch.load(path, map_location="cpu", weights_only=True)
    assert state["val_dice"] == pytest.approx(0.9)
    assert state["epoch"] == 2, "checkpoint is not from the best epoch"


def test_training_reads_train_and_validates_on_val(cfg, fold, monkeypatch):
    # Guards against selecting on the held-out set or training on the in-domain
    # test set. Both mutations are silent and both invalidate the headline.
    import src.train as train_module
    real = train_module.LungDataset
    calls = []

    def spy(frame, size, augment=False, seed=42):
        calls.append((frame, augment))
        return real(frame, size, augment=augment, seed=seed)

    monkeypatch.setattr(train_module, "LungDataset", spy)
    train_one(cfg, fold, "unet")

    assert len(calls) == 2
    (train_frame, train_aug), (val_frame, val_aug) = calls
    assert train_frame is fold.train
    assert val_frame is fold.val
    assert train_aug is True
    assert val_aug is False, "validation is augmented, selection measures noise"


def test_three_arms_map_to_three_distinct_configurations():
    # The run matrix is 3 arms x 3 folds. Two arms collapsing to the same
    # (model, pretrained) pair would silently make one contrast vacuous.
    assert len(ARMS) == 3
    assert len(set(ARMS.values())) == 3


def test_arms_isolate_pretraining_at_equal_capacity():
    # The headline pretraining claim rests on one pair differing only in init.
    # Scratch-vs-pretrained across architectures is confounded by 3x params.
    pretrained = ARMS["unet_resnet34_imagenet"]
    scratch = ARMS["unet_resnet34_scratch"]
    assert pretrained[0] == scratch[0]
    assert pretrained[1] != scratch[1]


def test_arms_get_separate_checkpoint_files(cfg, fold):
    # Both ResNet arms share a model name. Naming checkpoints by model would
    # make the second run overwrite the first and destroy the comparison.
    a = train_one(cfg, fold, "unet_resnet34_scratch")
    b = train_one(cfg, fold, "unet_resnet34_imagenet")
    assert a != b


def test_each_arm_builds_the_model_it_names(cfg, fold, monkeypatch):
    # The one thing no other test can see: train_one could look the arm up and
    # then hand build_model the wrong flag. The ImageNet arm would train from
    # random init, every test would still pass, and the headline pretraining
    # result would be measuring nothing.
    import src.train as train_module

    calls = []
    real = train_module.build_model

    def spy(name, pretrained=True):
        calls.append((name, pretrained))
        return real(name, pretrained=False)  # keep the smoke run offline

    monkeypatch.setattr(train_module, "build_model", spy)
    cfg["epochs"] = 0
    for arm, expected in ARMS.items():
        train_one(cfg, fold, arm)
    assert calls == [ARMS[arm] for arm in ARMS]


def test_replicate_seeds_do_not_overwrite_each_other(cfg, fold):
    # Error bars on the pretraining delta need seed replicates. Without the
    # seed in the filename, run 2 overwrites run 1 in place and any results row
    # already collected points at the wrong weights.
    cfg["seed"] = 0
    a = train_one(cfg, fold, "unet")
    cfg["seed"] = 1
    b = train_one(cfg, fold, "unet")
    assert a != b
    assert a.exists() and b.exists()


def test_training_batch_order_does_not_depend_on_the_model(cfg, fold, monkeypatch):
    # Model construction draws from the global RNG, and the two ResNet arms
    # draw different amounts: the ImageNet arm loads its encoder from disk and
    # only initialises the decoder. Left unfixed, the shuffle stream differs
    # per arm, so the arms differ in batch order as well as initialisation and
    # the pretraining contrast is no longer clean.
    import src.train as train_module
    real = train_module.LungDataset
    orders = {}

    def capture(arm):
        seen = []

        class Recording(real):
            def __getitem__(self, i):
                if self.augment:  # the shuffled training loader
                    seen.append(i)
                return super().__getitem__(i)

        # Recorded during the epoch, not at construction: the sampler draws its
        # permutation lazily, after build_model has already moved the RNG on.
        monkeypatch.setattr(train_module, "LungDataset", Recording)
        train_one(cfg, fold, arm)
        orders[arm] = seen

    capture("unet")
    capture("unet_resnet34_scratch")
    assert orders["unet"] == orders["unet_resnet34_scratch"]
    # Equal but sorted would mean shuffling is off entirely, which trains in
    # source order and makes batches source-homogeneous. That is the exact
    # confound the leave-one-dataset-out design exists to avoid.
    assert orders["unet"] != sorted(orders["unet"]), "training data is not shuffled"


def test_unknown_held_out_names_the_available_folds():
    # Every other error path in this codebase explains the constraint; a bare
    # StopIteration from a typo'd --held-out would not.
    from src.train import select_fold

    class F:
        def __init__(self, h):
            self.held_out = h

    with pytest.raises(ValueError, match="montgomery"):
        select_fold([F("montgomery"), F("jsrt")], "montgommery")


def test_unknown_arm_raises(cfg, fold):
    with pytest.raises(ValueError, match="unknown arm"):
        train_one(cfg, fold, "unet_resnet50")


def test_workers_get_their_own_augmentation_stream(cfg, fold, monkeypatch):
    # Without worker_init_fn every worker is a copy holding one identical RNG
    # state, so they draw the same augmentation and every epoch replays it.
    from src.dataset import worker_init_fn

    seen = []

    import src.train as train_module
    real = train_module.DataLoader

    def spy(*args, **kwargs):
        seen.append(kwargs.get("worker_init_fn"))
        return real(*args, **kwargs)

    monkeypatch.setattr(train_module, "DataLoader", spy)
    cfg["num_workers"] = 2
    cfg["epochs"] = 0  # only the loader construction matters here
    train_one(cfg, fold, "unet")
    assert seen and all(fn is worker_init_fn for fn in seen)
