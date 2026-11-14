import unittest
from types import SimpleNamespace

from src.agent.training_attention import configure_training_sdpa, POLICY


class AttentionPolicyTests(unittest.TestCase):
    def test_explicit_kv_and_no_math_fallback(self):
        calls = {}
        def setter(name):
            return lambda value: calls.__setitem__(name, value)
        backend = SimpleNamespace(**{name: setter(name) for name in (
            'enable_math_sdp', 'enable_flash_sdp', 'enable_mem_efficient_sdp', 'enable_cudnn_sdp')})
        torch = SimpleNamespace(backends=SimpleNamespace(cuda=backend))
        integration = SimpleNamespace(use_gqa_in_sdpa=lambda mask, key: True)
        self.assertEqual(configure_training_sdpa(torch, integration), POLICY)
        self.assertFalse(integration.use_gqa_in_sdpa(None, object()))
        self.assertFalse(integration.use_gqa_in_sdpa(object(), object()))
        self.assertEqual(calls, {'enable_math_sdp': False, 'enable_flash_sdp': True,
                                 'enable_mem_efficient_sdp': True, 'enable_cudnn_sdp': False})

    def test_missing_transformers_hook_fails_explicitly(self):
        with self.assertRaisesRegex(RuntimeError, 'API changed'):
            configure_training_sdpa(None, SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
