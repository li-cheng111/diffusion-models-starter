import unittest

import torch

from checkpoint_utils import load_model_weights, model_state_from_checkpoint


class CheckpointUtilsTests(unittest.TestCase):
    def test_nested_ema_is_unwrapped(self):
        model = torch.nn.Linear(2, 2)
        expected = {
            name: torch.ones_like(value) for name, value in model.state_dict().items()
        }
        checkpoint = {
            "model": model.state_dict(),
            "ema": {"decay": 0.999, "model": expected},
        }
        self.assertIs(model_state_from_checkpoint(checkpoint), expected)
        self.assertEqual(load_model_weights(model, checkpoint), "EMA")
        for value in model.state_dict().values():
            self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_flat_ema_is_supported(self):
        model = torch.nn.Linear(2, 2)
        flat = {
            name: torch.zeros_like(value) for name, value in model.state_dict().items()
        }
        checkpoint = {"model": model.state_dict(), "ema": flat}
        self.assertIs(model_state_from_checkpoint(checkpoint), flat)


if __name__ == "__main__":
    unittest.main()
