import torch

from farm_us.training.losses import build_loss, masked_huber, masked_mae, masked_mse


def test_masked_mse_ignores_invalid_pixels():
    pred = torch.tensor([[[[1.0, 100.0], [2.0, 3.0]]]])
    target = torch.tensor([[[[1.0, 0.0], [2.0, 3.0]]]])
    mask = torch.tensor([[[[1.0, 0.0], [1.0, 1.0]]]])  # exclude the big-error pixel
    loss, n = masked_mse(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(0.0))
    assert n == 3


def test_masked_mse_denominator_is_valid_count():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.ones(1, 1, 2, 2) * 2.0
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss, n = masked_mse(pred, target, mask)
    # mean of (0-2)^2 = 4 over the 2 valid pixels
    assert torch.isclose(loss, torch.tensor(4.0))
    assert n == 2


def test_masked_mae_value():
    pred = torch.zeros(1, 1, 2, 2)
    target = torch.tensor([[[[3.0, 5.0], [9.0, 9.0]]]])
    mask = torch.tensor([[[[1.0, 1.0], [0.0, 0.0]]]])
    loss, _ = masked_mae(pred, target, mask)
    assert torch.isclose(loss, torch.tensor(4.0))  # (3+5)/2


def test_masked_huber_small_error_is_quadratic():
    pred = torch.zeros(1, 1, 1, 1)
    target = torch.tensor([[[[0.5]]]])
    mask = torch.ones(1, 1, 1, 1)
    loss, _ = masked_huber(pred, target, mask, delta=1.0)
    assert torch.isclose(loss, torch.tensor(0.125), atol=1e-6)  # 0.5*0.5^2


def test_zero_valid_pixels_gives_finite_zero():
    pred = torch.randn(1, 1, 2, 2, requires_grad=True)
    target = torch.randn(1, 1, 2, 2)
    mask = torch.zeros(1, 1, 2, 2)
    loss, n = masked_mse(pred, target, mask)
    assert n == 0
    assert torch.isfinite(loss)
    loss.backward()  # must not error


def test_build_loss_dispatch():
    for name in ("mse", "mae", "huber"):
        fn = build_loss(name)
        loss, _ = fn(torch.zeros(1, 1, 2, 2), torch.ones(1, 1, 2, 2), torch.ones(1, 1, 2, 2))
        assert torch.isfinite(loss)
