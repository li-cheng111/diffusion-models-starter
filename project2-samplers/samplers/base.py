"""Common sampler interface and schedule/timestep compatibility helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import torch


def get_schedule_buffer(schedule, *names: str) -> torch.Tensor:
    """Get the first available schedule buffer from a set of aliases."""

    for name in names:
        if hasattr(schedule, name):
            value = getattr(schedule, name)
            if isinstance(value, torch.Tensor):
                return value
    raise AttributeError(
        f"Schedule is missing all compatible buffer names: {', '.join(names)}"
    )


def sqrt_alpha_bar(schedule) -> torch.Tensor:
    return get_schedule_buffer(schedule, "sqrt_alpha_bar", "sqrt_alphas_cumprod")


def sqrt_one_minus_alpha_bar(schedule) -> torch.Tensor:
    return get_schedule_buffer(
        schedule,
        "sqrt_one_minus_alpha_bar",
        "sqrt_one_minus_alphas_cumprod",
    )


def validate_timesteps(timesteps: Sequence[int], num_steps: int, T: int) -> list[int]:
    values = [int(value) for value in timesteps]
    if len(values) != num_steps:
        raise ValueError(f"Expected {num_steps} timesteps, got {len(values)}")
    if any(value < 0 or value >= T for value in values):
        raise ValueError(f"Timesteps must be within [0, {T - 1}]: {values}")
    if any(left <= right for left, right in zip(values, values[1:])):
        raise ValueError(f"Timesteps must be strictly decreasing: {values}")
    return values


def _unique_ascending_targets(targets: torch.Tensor, T: int) -> list[int]:
    """Round targets while reserving enough integer slots to avoid duplicates."""

    count = int(targets.numel())
    if count < 1 or count > T:
        raise ValueError(f"num_steps must be in [1, {T}], got {count}")
    result: list[int] = []
    for index, target in enumerate(targets.tolist()):
        minimum = 0 if index == 0 else result[-1] + 1
        maximum = T - (count - index)
        result.append(min(max(int(round(target)), minimum), maximum))
    return result


def linear_timesteps(num_steps: int, T: int) -> list[int]:
    """Return exactly ``num_steps`` uniformly spaced, decreasing indices."""

    if not isinstance(num_steps, int) or num_steps < 1 or num_steps > T:
        raise ValueError(f"num_steps must be an integer in [1, {T}], got {num_steps!r}")
    if num_steps == 1:
        return [T - 1]
    ascending = _unique_ascending_targets(torch.linspace(0, T - 1, num_steps), T)
    return validate_timesteps(list(reversed(ascending)), num_steps, T)


def quadratic_timesteps(num_steps: int, T: int) -> list[int]:
    """Return a duplicate-free quadratic timestep schedule."""

    if not isinstance(num_steps, int) or num_steps < 1 or num_steps > T:
        raise ValueError(f"num_steps must be an integer in [1, {T}], got {num_steps!r}")
    if num_steps == 1:
        return [T - 1]
    ramp = torch.linspace(0.0, 1.0, num_steps)
    ascending = _unique_ascending_targets(ramp.square() * (T - 1), T)
    return validate_timesteps(list(reversed(ascending)), num_steps, T)


def randn(
    shape: Sequence[int],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    return torch.randn(tuple(shape), device=device, dtype=dtype, generator=generator)


class BaseSampler(ABC):
    """Base class for samplers consuming a trained epsilon-prediction model."""

    def __init__(self, model, schedule, device: str | torch.device = "cuda"):
        self.model = model
        self.schedule = schedule
        self.device = torch.device(device)
        self.T = int(schedule.T)

    @abstractmethod
    def get_timesteps(self, num_steps: int) -> list[int]:
        pass

    @abstractmethod
    def step(
        self,
        x_t: torch.Tensor,
        eps_pred: torch.Tensor,
        t: int,
        t_prev: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        pass

    def nfe_for_steps(self, num_steps: int) -> int:
        return int(num_steps)

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
            x = randn(expected_shape, device=self.device, dtype=torch.float32, generator=generator)
        else:
            if tuple(initial_x.shape) != expected_shape:
                raise ValueError(
                    f"initial_x has shape {tuple(initial_x.shape)}, expected {expected_shape}"
                )
            x = initial_x.detach().to(self.device).clone()

        timesteps = validate_timesteps(
            self.get_timesteps(num_steps), num_steps if self.name != "DDPM" else self.T, self.T
        )
        extended = timesteps + [-1]
        trajectory = [x.clone()] if return_trajectory else None

        self.model.eval()
        for index, timestep in enumerate(timesteps):
            previous = extended[index + 1]
            t_tensor = torch.full(
                (x.shape[0],), timestep, device=self.device, dtype=torch.long
            )
            eps_pred = self.model(x, t_tensor)
            x = self.step(x, eps_pred, timestep, previous, generator=generator)
            if not torch.isfinite(x).all():
                raise FloatingPointError(
                    f"{self.name} produced NaN/Inf at t={timestep}, t_prev={previous}"
                )
            if return_trajectory:
                trajectory.append(x.clone())

        return (x, trajectory) if return_trajectory else x

    @property
    def name(self) -> str:
        return self.__class__.__name__
