"""Offline plumbing test with random tiny weights; does not measure task accuracy."""
import tempfile
import unittest


class LocalClientRuntimeTests(unittest.TestCase):
    def test_real_transformers_generation_and_context_limit(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from src.agent.llm_client import LocalHuggingFaceClient, ContextBudgetExceeded

        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        self.addCleanup(torch.set_num_threads, previous_threads)
        torch.manual_seed(42)
        with tempfile.TemporaryDirectory(prefix='lifemate-client-') as directory:
            tokenizer = Tokenizer(models.WordLevel(
                {'<unk>': 0, '<eos>': 1, 'a': 2, 'assistant': 3}, unk_token='<unk>'))
            tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
            fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
                                          eos_token='<eos>', pad_token='<eos>')
            fast.chat_template = '{% for message in messages %}{{ message.content }} {% endfor %}assistant'
            fast.save_pretrained(directory)
            model = Qwen2ForCausalLM(Qwen2Config(vocab_size=4, hidden_size=16,
                intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                num_key_value_heads=1, max_position_embeddings=64, bos_token_id=1, eos_token_id=1))
            model.save_pretrained(directory)
            client = LocalHuggingFaceClient(directory, device='cpu', max_new_tokens=2, context_window=64)
            self.assertEqual(client.generate_with_budget('a', 'a', 10), client.generate_with_budget('a', 'a', 10))
            with self.assertRaises(ContextBudgetExceeded):
                client.generate_with_budget('a', 'a ' * 70, 10)


if __name__ == '__main__':
    unittest.main()
