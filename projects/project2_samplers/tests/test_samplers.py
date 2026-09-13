import unittest

import torch

from projects.project2_samplers.samplers import (
    DDIMInverter,
    DDIMSampler,
    DDPMSampler,
    DPMSolver2Sampler,
)
from projects.project2_samplers.tests.helpers import (
    CountingZeroModel,
    ToySchedule,
    ZeroModel,
)


class SamplerTests(unittest.TestCase):
    def test_ddpm_accepts_project1_plural_buffer_names(self):
        schedule = ToySchedule(plural_aliases=True)
        output = DDPMSampler(ZeroModel(), schedule, device="cpu").sample(
            (2, 3, 8, 8), num_steps=100
        )
        self.assertEqual(output.shape, (2, 3, 8, 8))
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_ddim_eta_zero_is_deterministic_for_fixed_initial_x(self):
        sampler = DDIMSampler(ZeroModel(), ToySchedule(), device="cpu", eta=0.0)
        initial = torch.randn(2, 3, 8, 8)
        first = sampler.sample(initial.shape, 10, initial_x=initial)
        second = sampler.sample(initial.shape, 10, initial_x=initial)
        self.assertTrue(torch.equal(first, second))

    def test_ddim_eta_positive_uses_generator(self):
        sampler = DDIMSampler(ZeroModel(), ToySchedule(), device="cpu", eta=1.0)
        initial = torch.randn(2, 3, 8, 8)
        first = sampler.sample(
            initial.shape,
            10,
            initial_x=initial,
            generator=torch.Generator().manual_seed(1),
        )
        second = sampler.sample(
            initial.shape,
            10,
            initial_x=initial,
            generator=torch.Generator().manual_seed(2),
        )
        self.assertFalse(torch.equal(first, second))

    def test_dpm_solver_reports_and_performs_true_nfe(self):
        model = CountingZeroModel()
        sampler = DPMSolver2Sampler(model, ToySchedule(), device="cpu")
        output = sampler.sample((2, 3, 8, 8), num_steps=5)
        self.assertEqual(model.calls, 9)
        self.assertEqual(sampler.nfe_for_steps(5), 9)
        self.assertTrue(bool(torch.isfinite(output).all()))

    def test_ddim_inversion_zero_model_round_trip(self):
        inverter = DDIMInverter(ZeroModel(), ToySchedule(), device="cpu")
        source = torch.randn(2, 3, 8, 8) * 0.25
        _, reconstructed = inverter.invert_and_reconstruct(source, num_steps=10)
        self.assertTrue(torch.allclose(source, reconstructed, atol=1e-5, rtol=1e-5))

    def test_initial_x_shape_is_validated(self):
        sampler = DDIMSampler(ZeroModel(), ToySchedule(), device="cpu")
        with self.assertRaisesRegex(ValueError, "initial_x"):
            sampler.sample(
                (1, 3, 8, 8), 10, initial_x=torch.randn(2, 3, 8, 8)
            )


if __name__ == "__main__":
    unittest.main()
