import pytest
import torch

from src.model import build_model


@pytest.mark.parametrize("name", ["unet", "unet_resnet34"])
def test_output_shape_matches_input(name):
    model = build_model(name, pretrained=False)
    x = torch.zeros(2, 1, 64, 64)
    out = model(x)
    assert out.shape == (2, 1, 64, 64)


@pytest.mark.parametrize("name", ["unet", "unet_resnet34"])
def test_output_is_logits_not_probabilities(name):
    """A sigmoid inside the model would silently double-apply with BCEWithLogitsLoss."""
    model = build_model(name, pretrained=False)
    with torch.no_grad():
        out = model(torch.randn(1, 1, 64, 64) * 10)
    assert out.min() < 0.0 or out.max() > 1.0


@pytest.mark.parametrize("name", ["unet", "unet_resnet34"])
def test_gradients_flow(name):
    model = build_model(name, pretrained=False)
    out = model(torch.randn(1, 1, 64, 64))
    out.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())


def test_unknown_model_name_raises():
    with pytest.raises(ValueError):
        build_model("resnet50", pretrained=False)


def test_asking_for_a_pretrained_scratch_unet_raises():
    # A run matrix over (name, pretrained) would otherwise hand back two
    # identical scratch models and record them as a comparison.
    with pytest.raises(ValueError, match="no pretrained variant"):
        build_model("unet", pretrained=True)


@pytest.mark.parametrize("name,divisor", [("unet", 16), ("unet_resnet34", 32)])
def test_input_size_must_be_divisible(name, divisor):
    model = build_model(name, pretrained=False)
    bad = divisor * 2 + 1
    with pytest.raises(ValueError, match="divisible"):
        model(torch.zeros(1, 1, bad, bad))


def test_pretrained_encoder_actually_loads_imagenet_weights():
    # Guards the premise of the whole comparison: if torchvision ever changed
    # under us and quietly handed back a random encoder, every other test here
    # would still pass and the pretrained arm would be pretrained in name only.
    try:
        pre = build_model("unet_resnet34", pretrained=True)
    except Exception as exc:  # no cached weights and no network
        pytest.skip(f"ImageNet weights unavailable: {exc}")
    rand = build_model("unet_resnet34", pretrained=False)
    assert not torch.allclose(pre.stem[0].weight, rand.stem[0].weight)
    assert pre.stem[0].weight.std() > 2 * rand.stem[0].weight.std()


def test_scratch_and_pretrained_are_different_classes():
    a = build_model("unet", pretrained=False)
    b = build_model("unet_resnet34", pretrained=False)
    assert type(a) is not type(b)
