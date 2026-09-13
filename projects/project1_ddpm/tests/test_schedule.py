import torch

from projects.project1_ddpm.schedule import (
    DDPMSchedule,
    cosine_beta_schedule,
    linear_beta_schedule,
)


def test_linear_schedule_shape_and_range():
    betas = linear_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert torch.isfinite(betas).all()
    assert torch.all((betas > 0) & (betas < 1))
    assert torch.all(betas[1:] >= betas[:-1])


def test_cosine_schedule_shape_and_range():
    betas = cosine_beta_schedule(1000)
    assert betas.shape == (1000,)
    assert torch.isfinite(betas).all()
    assert torch.all((betas > 0) & (betas < 1))


def test_cosine_schedule_is_distinct_from_linear_and_non_decreasing():
    linear = linear_beta_schedule(128)
    cosine = cosine_beta_schedule(128)
    assert not torch.allclose(linear, cosine)
    assert torch.all(cosine[1:] >= cosine[:-1])


def test_schedule_selects_cosine_coefficients():
    schedule = DDPMSchedule(T=64, beta_schedule="cosine")
    expected = cosine_beta_schedule(64)
    assert torch.allclose(schedule.betas, expected)


def test_schedule_registers_device_movable_buffers():
    schedule = DDPMSchedule(T=32)
    assert schedule.betas.shape == (32,)
    assert schedule.posterior_variance[0].item() == 0.0
    schedule.to("cpu")
    assert schedule.sqrt_alphas_cumprod.device.type == "cpu"
