import asyncio
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import scripts.colab_microbatch_server as server


class ColabMicrobatchServerTests(unittest.TestCase):
    def test_is_cuda_oom(self):
        self.assertTrue(server._is_cuda_oom(torch.OutOfMemoryError("CUDA out of memory")))
        self.assertTrue(server._is_cuda_oom(RuntimeError("CUDA out of memory. Tried to allocate...")))
        self.assertFalse(server._is_cuda_oom(ValueError("Some other error")))
        self.assertFalse(server._is_cuda_oom(RuntimeError("Connection closed")))

    def test_generate_batch_resilient_splits_batch_on_oom(self):
        req1 = server.PendingRequest(
            request=server.ChatCompletionRequest(messages=[server.Message(role="user", content="hi 1")]),
            future=MagicMock(),
        )
        req2 = server.PendingRequest(
            request=server.ChatCompletionRequest(messages=[server.Message(role="user", content="hi 2")]),
            future=MagicMock(),
        )

        calls = []

        def mock_generate(batch, max_new_tokens_override=None):
            calls.append(len(batch))
            if len(batch) > 1:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return [f"response for {b.request.messages[0].content}" for b in batch]

        with patch.object(server, "generate_batch", side_effect=mock_generate), \
             patch.object(server, "_clear_cuda_cache") as mock_clear:
            results = server.generate_batch_resilient([req1, req2])

        self.assertEqual(calls, [2, 1, 1])
        self.assertEqual(len(results), 2)
        self.assertIn("hi 1", results[0])
        self.assertIn("hi 2", results[1])
        self.assertTrue(mock_clear.called)

    def test_generate_batch_resilient_retries_single_request_with_smaller_budget(self):
        req = server.PendingRequest(
            request=server.ChatCompletionRequest(
                messages=[server.Message(role="user", content="long prompt")],
                max_tokens=256,
            ),
            future=MagicMock(),
        )

        attempt_limits = []

        def mock_generate(batch, max_new_tokens_override=None):
            attempt_limits.append(max_new_tokens_override)
            if max_new_tokens_override != 64:
                raise torch.OutOfMemoryError("CUDA out of memory")
            return ["success at 64"]

        with patch.object(server, "generate_batch", side_effect=mock_generate), \
             patch.object(server, "_clear_cuda_cache"):
            results = server.generate_batch_resilient([req])

        self.assertEqual(results, ["success at 64"])
        # Should try None (full 256), then 128, then 64
        self.assertEqual(attempt_limits, [None, 128, 64])

    def test_generate_batch_resilient_reraises_non_oom(self):
        req = server.PendingRequest(
            request=server.ChatCompletionRequest(messages=[server.Message(role="user", content="test")]),
            future=MagicMock(),
        )

        with patch.object(server, "generate_batch", side_effect=ValueError("Tokenization bug")):
            with self.assertRaises(ValueError):
                server.generate_batch_resilient([req])

    def test_chat_completion_request_validation(self):
        from pydantic import ValidationError

        # Valid request
        req = server.ChatCompletionRequest(
            messages=[server.Message(role="user", content="hi")],
            temperature=0.0,
            max_tokens=256,
        )
        self.assertEqual(req.temperature, 0.0)

        # max_tokens > MAX_NEW_TOKENS rejected by schema
        with self.assertRaises(ValidationError):
            server.ChatCompletionRequest(
                messages=[server.Message(role="user", content="hi")],
                max_tokens=server.MAX_NEW_TOKENS + 1,
            )

        # max_tokens < 1 rejected by schema
        with self.assertRaises(ValidationError):
            server.ChatCompletionRequest(
                messages=[server.Message(role="user", content="hi")],
                max_tokens=0,
            )

    def test_prompt_truncation_logic(self):
        head = server.MAX_INPUT_HEAD_TOKENS
        max_tokens = server.MAX_INPUT_TOKENS
        tail = max_tokens - head

        tokens = list(range(max_tokens + 100))
        truncated = tokens[:head] + tokens[-tail:]
        self.assertEqual(len(truncated), max_tokens)
        self.assertEqual(truncated[:head], list(range(head)))
        self.assertEqual(truncated[-tail:], list(range(len(tokens) - tail, len(tokens))))


if __name__ == "__main__":
    unittest.main()
