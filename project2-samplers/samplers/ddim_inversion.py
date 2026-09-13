"""DDIM inversion paired with the exact timestep policy used for reconstruction."""

from __future__ import annotations

import torch

from .ddim import DDIMSampler


class DDIMInverter:
    def __init__(
        self,
        model,
        schedule,
        device="cuda",
        timestep_strategy: str = "linear",
    ):
        self.model = model
        self.schedule = schedule
        self.device = torch.device(device)
        self.sampler = DDIMSampler(
            model,
            schedule,
            device=device,
            eta=0.0,
            timestep_strategy=timestep_strategy,
            clip_denoised=False,
        )

    @torch.no_grad()
    def invert(self, x_0: torch.Tensor, num_steps: int, return_trajectory=False):
        x = x_0.detach().to(self.device).clone()
        ascending = list(reversed(self.sampler.get_timesteps(num_steps)))
        trajectory = [x.clone()] if return_trajectory else None
        current_t = -1
        self.model.eval()

        for target_t in ascending:
            model_t = max(current_t, 0)
            t_tensor = torch.full(
                (x.shape[0],), model_t, device=x.device, dtype=torch.long
            )
            eps_pred = self.model(x, t_tensor)
            if current_t < 0:
                alpha_current = torch.ones((), device=x.device, dtype=x.dtype)
            else:
                alpha_current = self.schedule.alphas_cumprod[current_t].to(x)
            alpha_target = self.schedule.alphas_cumprod[target_t].to(x)
            tiny = torch.finfo(x.dtype).eps
            x_hat_0 = (
                x
                - torch.sqrt((1.0 - alpha_current).clamp(min=0.0)) * eps_pred
            ) / torch.sqrt(alpha_current.clamp(min=tiny))
            x = (
                torch.sqrt(alpha_target) * x_hat_0
                + torch.sqrt((1.0 - alpha_target).clamp(min=0.0)) * eps_pred
            )
            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"DDIM inversion produced NaN/Inf while moving to t={target_t}"
                )
            current_t = target_t
            if return_trajectory:
                trajectory.append(x.clone())

        return (x, trajectory) if return_trajectory else x

    @torch.no_grad()
    def reconstruct(self, x_t: torch.Tensor, num_steps: int, return_trajectory=False):
        return self.sampler.sample(
            tuple(x_t.shape),
            num_steps=num_steps,
            return_trajectory=return_trajectory,
            initial_x=x_t,
        )

    @torch.no_grad()
    def invert_and_reconstruct(
        self, x_0: torch.Tensor, num_steps: int, return_trajectories=False
    ):
        inverted = self.invert(
            x_0, num_steps=num_steps, return_trajectory=return_trajectories
        )
        if return_trajectories:
            x_t, inversion_trajectory = inverted
            reconstructed, reconstruction_trajectory = self.reconstruct(
                x_t, num_steps=num_steps, return_trajectory=True
            )
            return x_t, reconstructed, inversion_trajectory, reconstruction_trajectory
        x_t = inverted
        reconstructed = self.reconstruct(x_t, num_steps=num_steps)
        return x_t, reconstructed


def ddim_invert(x_0, model, schedule, num_steps=50, device="cuda"):
    """Functional convenience wrapper for the assignment API."""

    return DDIMInverter(model, schedule, device=device).invert(x_0, num_steps)
