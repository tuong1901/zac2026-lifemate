import unittest
from types import SimpleNamespace

from src.agent.training_safety import UpdateHealth, make_update_callback


class TrainingSafetyTests(unittest.TestCase):
    def test_nan_grad_cannot_be_a_success_despite_finite_losses(self):
        health = UpdateHealth()
        with self.assertRaises(FloatingPointError):
            health.observe_logs({'loss': 2.3347, 'grad_norm': float('nan')})
        self.assertFalse(health.smoke_is_healthy(0.1))

    def test_overflow_skip_is_not_counted_as_update(self):
        health = UpdateHealth()
        with self.assertRaises(FloatingPointError):
            health.observe_update(True)
        self.assertEqual(health.successful_updates, 0)
        self.assertEqual(health.skipped_updates, 1)

    def test_two_finite_updates_require_actual_parameter_change(self):
        health = UpdateHealth()
        for _ in range(2):
            health.observe_update(False)
            health.observe_logs({'grad_norm': 24.0})
        self.assertFalse(health.smoke_is_healthy(0.0))
        self.assertFalse(health.smoke_is_healthy(float('nan')))
        self.assertTrue(health.smoke_is_healthy(1e-5))

    def test_callback_uses_accelerate_update_status(self):
        health = UpdateHealth()
        callback = make_update_callback(object, health)
        callback.on_step_end(None, None, None, optimizer=SimpleNamespace(step_was_skipped=False))
        self.assertEqual(health.successful_updates, 1)
        with self.assertRaises(FloatingPointError):
            callback.on_step_end(None, None, None, optimizer=SimpleNamespace(step_was_skipped=True))
        with self.assertRaises(RuntimeError):
            callback.on_step_end(None, None, None, optimizer=object())


if __name__ == '__main__':
    unittest.main()
