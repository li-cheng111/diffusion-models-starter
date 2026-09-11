import torch

from schedule import DDPMSchedule, cosine_beta_schedule, linear_beta_schedule


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


def test_schedule_registers_device_movable_buffers():
    schedule = DDPMSchedule(T=32)
    assert schedule.betas.shape == (32,)
    assert schedule.posterior_variance[0].item() == 0.0
    schedule.to("cpu")
    assert schedule.sqrt_alphas_cumprod.device.type == "cpu"

