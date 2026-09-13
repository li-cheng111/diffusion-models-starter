"""Second-order singlestep DPM-Solver using the midpoint construction."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from .base import BaseSampler, linear_timesteps, randn, validate_timesteps


class DPMSolver2Sampler(BaseSampler):
    def __init__(
        self,
        model,
        schedule,
        device="cuda",
        timestep_strategy: str = "lambda",
        clip_denoised: bool = True,
    ):
        super().__init__(model, schedule, device)
        alpha = torch.sqrt(schedule.alphas_cumprod.float().clamp(min=1e-20))
        sigma = torch.sqrt((1.0 - schedule.alphas_cumprod.float()).clamp(min=1e-20))
        self.alpha_t = alpha
        self.sigma_t = sigma
        self.lambdas = torch.log(alpha) - torch.log(sigma)
        self.timestep_strategy = timestep_strategy
        self.clip_denoised = bool(clip_denoised)

    def _lambda_timesteps(self, num_steps: int) -> list[int]:
        if not isinstance(num_steps, int) or num_steps < 1 or num_steps > self.T:
            raise ValueError(
                f"num_steps must be an integer in [1, {self.T}], got {num_steps!r}"
            )
        targets = torch.linspace(
            self.lambdas[-1].item(), self.lambdas[0].item(), num_steps
        )
        result: list[int] = []
        previous = self.T
        for index, target in enumerate(targets):
            nearest = int(torch.argmin((self.lambdas.cpu() - target).abs()).item())
            minimum = num_steps - index - 1
            maximum = previous - 1
            chosen = min(max(nearest, minimum), maximum)
            result.append(chosen)
            previous = chosen
        return validate_timesteps(result, num_steps, self.T)

    def get_timesteps(self, num_steps: int) -> list[int]:
        if self.timestep_strategy == "lambda":
            return self._lambda_timesteps(num_steps)
        if self.timestep_strategy == "linear":
            return linear_timesteps(num_steps, self.T)
        raise ValueError(
            f"Unknown DPM-Solver timestep strategy: {self.timestep_strategy}"
        )

    def nfe_for_steps(self, num_steps: int) -> int:
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        return 2 * int(num_steps) - 1

    def _coefficient(self, values: torch.Tensor, index: int, x: torch.Tensor):
        return values[index].to(device=x.device, dtype=x.dtype)

    @torch.no_grad()
    def _dpm_solver_2_step(self, x_t, t: int, t_prev: int):
        t_tensor = torch.full(
            (x_t.shape[0],), t, device=x_t.device, dtype=torch.long
        )
        eps_t = self.model(x_t, t_tensor)
        alpha_t = self._coefficient(self.alpha_t, t, x_t)
        sigma_t = self._coefficient(self.sigma_t, t, x_t)

        if t_prev < 0:
            x_hat_0 = (x_t - sigma_t * eps_t) / alpha_t.clamp(
                min=torch.finfo(x_t.dtype).eps
            )
            return x_hat_0.clamp(-1.0, 1.0) if self.clip_denoised else x_hat_0

        lambda_t = self._coefficient(self.lambdas, t, x_t)
        lambda_prev = self._coefficient(self.lambdas, t_prev, x_t)
        h = lambda_prev - lambda_t
        if not bool(h > 0):
            raise ValueError(
                f"Expected positive lambda step for {t}->{t_prev}, got h={h.item()}"
            )

        lambda_mid_target = (lambda_t + lambda_prev) / 2.0
        interval_lambdas = self.lambdas[t_prev : t + 1].to(
            device=x_t.device, dtype=x_t.dtype
        )
        t_mid = t_prev + int(
            torch.argmin((interval_lambdas - lambda_mid_target).abs()).item()
        )
        lambda_mid = self._coefficient(self.lambdas, t_mid, x_t)
        h_mid = lambda_mid - lambda_t

        alpha_mid = self._coefficient(self.alpha_t, t_mid, x_t)
        sigma_mid = self._coefficient(self.sigma_t, t_mid, x_t)
        x_mid = (
            alpha_mid / alpha_t * x_t
            - sigma_mid * torch.expm1(h_mid) * eps_t
        )
        if not torch.isfinite(x_mid).all():
            raise FloatingPointError(f"DPM-Solver midpoint is non-finite at t={t}")

        mid_tensor = torch.full(
            (x_t.shape[0],), t_mid, device=x_t.device, dtype=torch.long
        )
        eps_mid = self.model(x_mid, mid_tensor)
        alpha_prev = self._coefficient(self.alpha_t, t_prev, x_t)
        sigma_prev = self._coefficient(self.sigma_t, t_prev, x_t)
        return (
            alpha_prev / alpha_t * x_t
            - sigma_prev * torch.expm1(h) * eps_mid
        )

    @torch.no_grad()
    def sample(
        self,
        shape: Sequence[int],
        num_steps: int,
        return_trajectory: bool = False,
        initial_x: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ):
        expected_shape = tuple(int(value) for value in shape)
        if initial_x is None:
            x = randn(
                expected_shape,
                device=self.device,
                dtype=torch.float32,
                generator=generator,
            )
        else:
            if tuple(initial_x.shape) != expected_shape:
                raise ValueError(
                    f"initial_x has shape {tuple(initial_x.shape)}, expected {expected_shape}"
                )
            x = initial_x.detach().to(self.device).clone()

        timesteps = self.get_timesteps(num_steps)
        extended = timesteps + [-1]
        trajectory = [x.clone()] if return_trajectory else None
        self.model.eval()
        for index, timestep in enumerate(timesteps):
            x = self._dpm_solver_2_step(x, timestep, extended[index + 1])
            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"DPM-Solver produced NaN/Inf at t={timestep}"
                )
            if return_trajectory:
                trajectory.append(x.clone())
        return (x, trajectory) if return_trajectory else x

    def step(self, x_t, eps_pred, t, t_prev, generator=None):
        raise NotImplementedError("DPM-Solver-2 uses its custom sample method")

    @property
    def name(self) -> str:
        return "DPM-Solver-2"


if __name__ == "__main__":
    from project1_path import add_project1_to_path

    add_project1_to_path()
    from schedule import DDPMSchedule

    class CountingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x, t):
            self.calls += 1
            return torch.zeros_like(x)

    schedule = DDPMSchedule(T=100).to("cpu")
    model = CountingModel()
    sampler = DPMSolver2Sampler(model, schedule, device="cpu")
    output = sampler.sample((1, 3, 8, 8), num_steps=5)
    assert model.calls == sampler.nfe_for_steps(5)
    print(f"DPM-Solver-2 self-test passed: shape={tuple(output.shape)}, NFE={model.calls}")
