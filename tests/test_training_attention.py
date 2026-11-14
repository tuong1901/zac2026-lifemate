import unittest
from types import SimpleNamespace

from src.agent.training_attention import (configure_training_sdpa, POLICY,
    CHUNKED_POLICY, chunked_sdpa_attention_forward, native_training_dtype)

try:
    import torch
except ImportError:
    torch = None


class AttentionPolicyTests(unittest.TestCase):
    def test_t4_uses_fp16_even_if_bf16_emulation_exists(self):
        from unittest.mock import Mock
        probe = Mock(return_value=True)
        fake = SimpleNamespace(float16='FP16', bfloat16='BF16', cuda=SimpleNamespace(
            get_device_capability=lambda index: (7, 5), is_bf16_supported=probe))
        self.assertEqual(native_training_dtype(fake, 1), 'FP16')
        probe.assert_not_called()

    def test_ampere_requires_native_bf16_support(self):
        from unittest.mock import Mock
        probe = Mock(return_value=True)
        fake = SimpleNamespace(float16='FP16', bfloat16='BF16', cuda=SimpleNamespace(
            get_device_capability=lambda index: (8, 0), is_bf16_supported=probe))
        self.assertEqual(native_training_dtype(fake, 0), 'BF16')
        probe.assert_called_once_with(including_emulation=False)
        probe.return_value = False
        self.assertEqual(native_training_dtype(fake, 0), 'FP16')

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

    def test_chunked_backend_is_registered_and_math_is_bounded(self):
        calls, registered = {}, {}
        def setter(name):
            return lambda value: calls.__setitem__(name, value)
        backend = SimpleNamespace(**{name: setter(name) for name in (
            'enable_math_sdp', 'enable_flash_sdp', 'enable_mem_efficient_sdp', 'enable_cudnn_sdp')})
        fake_torch = SimpleNamespace(backends=SimpleNamespace(cuda=backend))
        integration = SimpleNamespace(use_gqa_in_sdpa=lambda mask, key: True)
        registry = SimpleNamespace(register=lambda name, fn: registered.__setitem__(name, fn))
        self.assertEqual(configure_training_sdpa(fake_torch, integration,
                         mode='chunked-math', query_chunk_size=128, registry=registry), CHUNKED_POLICY)
        self.assertIs(integration.sdpa_attention_forward, registered['sdpa'])
        self.assertEqual(calls, {'enable_math_sdp': True, 'enable_flash_sdp': False,
                                 'enable_mem_efficient_sdp': False, 'enable_cudnn_sdp': False})


@unittest.skipIf(torch is None, 'PyTorch needed for numerical forward/backward tests')
class ChunkedAttentionNumericalTests(unittest.TestCase):
    def compare(self, causal=True, mask_type=None, nested=False):
        from unittest.mock import patch
        from torch.nn import functional as F
        from torch.utils.checkpoint import checkpoint
        torch.manual_seed(42)
        q = torch.randn(1, 4, 13, 8, dtype=torch.float64, requires_grad=True)
        k = torch.randn(1, 2, 13, 8, dtype=torch.float64, requires_grad=True)
        v = torch.randn(1, 2, 13, 8, dtype=torch.float64, requires_grad=True)
        q2, k2, v2 = (t.detach().clone().requires_grad_() for t in (q, k, v))
        module = SimpleNamespace(is_causal=causal)
        mask = None
        base_mask = torch.ones(13, 13, dtype=torch.bool)
        if causal:
            base_mask = base_mask.tril()
        if mask_type == 'padding':
            mask = torch.ones(1, 13, dtype=torch.long)
            mask[:, 5] = 0
            reference_mask = base_mask & mask.bool()[:, None, None, :]
        elif mask_type == 'additive':
            mask = torch.zeros(1, 1, 13, 13, dtype=torch.float64)
            mask[..., 4] = float('-inf')
            reference_mask = mask.masked_fill(~base_mask, float('-inf'))
        else:
            reference_mask = base_mask if causal else None
        ref = F.scaled_dot_product_attention(q, k.repeat_interleave(2, 1),
                    v.repeat_interleave(2, 1), attn_mask=reference_mask, scale=8**-0.5).transpose(1, 2)
        reference_grads = torch.autograd.grad(ref.square().sum(), (q, k, v))
        seen = []
        original = F.scaled_dot_product_attention
        def observe(query, key, value, **kwargs):
            seen.append((query.shape[-2], key.shape[-2]))
            return original(query, key, value, **kwargs)
        def forward(a, b, c):
            return chunked_sdpa_attention_forward(module, a, b, c, mask,
                       scaling=8**-0.5, query_chunk_size=3)[0]
        with patch('torch.nn.functional.scaled_dot_product_attention', side_effect=observe):
            actual = checkpoint(forward, q2, k2, v2, use_reentrant=False) if nested else forward(q2, k2, v2)
            actual_grads = torch.autograd.grad(actual.square().sum(), (q2, k2, v2))
        torch.testing.assert_close(actual, ref, atol=1e-10, rtol=1e-10)
        for actual_grad, reference_grad in zip(actual_grads, reference_grads):
            torch.testing.assert_close(actual_grad, reference_grad, atol=1e-10, rtol=1e-10)
        self.assertTrue(all(q_length <= 3 for q_length, _ in seen))
        self.assertTrue(any(k_length == 13 for _, k_length in seen))

    def test_causal_gqa_matches_dense_output_and_gradients(self):
        self.compare()

    def test_padding_mask_matches_dense(self):
        self.compare(mask_type='padding')

    def test_additive_mask_matches_dense(self):
        self.compare(mask_type='additive')

    def test_noncausal_matches_dense(self):
        self.compare(causal=False)

    def test_nested_checkpoint_matches_dense(self):
        self.compare(nested=True)


if __name__ == '__main__':
    unittest.main()
