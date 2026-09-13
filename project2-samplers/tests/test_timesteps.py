import unittest

from samplers.base import linear_timesteps, quadratic_timesteps, validate_timesteps


class TimestepTests(unittest.TestCase):
    def test_timestep_builders_are_strict_and_exact(self):
        for builder in (linear_timesteps, quadratic_timesteps):
            for steps in (1, 2, 10, 50, 100):
                with self.subTest(builder=builder.__name__, steps=steps):
                    values = builder(steps, 100)
                    self.assertEqual(len(values), steps)
                    self.assertEqual(values[0], 99)
                    self.assertTrue(
                        all(left > right for left, right in zip(values, values[1:]))
                    )
                    self.assertGreaterEqual(min(values), 0)
                    self.assertLess(max(values), 100)

    def test_invalid_step_count_rejected(self):
        with self.assertRaises(ValueError):
            linear_timesteps(0, 100)
        with self.assertRaises(ValueError):
            quadratic_timesteps(101, 100)

    def test_validation_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            validate_timesteps([99, 50, 50], 3, 100)


if __name__ == "__main__":
    unittest.main()
