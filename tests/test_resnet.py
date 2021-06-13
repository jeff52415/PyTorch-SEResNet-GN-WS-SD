import torch

from senet_pack.models import se_resnext50_32x4d


def test_bn():
    dummy_ = torch.randn(32, 3, 64, 64)
    model = se_resnext50_32x4d(
        pretrained=None,
        group_normalization=False,
        weight_standardization=False,
        stochastic_depth=False,
    )
    model.features(dummy_)


def test_gn():
    dummy_ = torch.randn(32, 3, 64, 64)
    model = se_resnext50_32x4d(
        pretrained=None,
        group_normalization=True,
        weight_standardization=False,
        stochastic_depth=False,
    )
    model.features(dummy_)


def test_ws():
    dummy_ = torch.randn(32, 3, 64, 64)
    model = se_resnext50_32x4d(
        pretrained=None,
        group_normalization=True,
        weight_standardization=True,
        stochastic_depth=False,
    )
    model.features(dummy_)


def test_sd():
    dummy_ = torch.randn(32, 3, 64, 64)
    model = se_resnext50_32x4d(
        pretrained=None,
        group_normalization=True,
        weight_standardization=True,
        stochastic_depth=True,
    )
    model.features(dummy_)


def test_full():
    dummy_ = torch.randn(32, 3, 64, 64)
    model = se_resnext50_32x4d(
        num_classes=10,
        pretrained=None,
        group_normalization=False,
        weight_standardization=False,
        stochastic_depth=False,
    )
    output = model(dummy_)
    assert output.shape[1] == 10
