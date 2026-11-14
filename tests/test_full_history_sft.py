import json
import unittest
from unittest.mock import patch
from pathlib import Path

from scripts.build_sft_dataset import build_rows
from scripts.prepare_full_history_sft import split_sources
from src.rules.catalog_filter import CatalogFilter


class FullHistorySFTTests(unittest.TestCase):
    def raw_item(self, text):
        return {'id': 'raw', 'audio_text': text,
                'context': {'installed_apps': {'notes': {}}},
                'history': [{'role': 'user', 'audio_text': 'Tên Lann, năm ba bảy', 'audio': 'x.wav'}],
                'label': {'tool_calls': [], 'tool_results': []}}

    def test_raw_no_action_keeps_original_text_and_bypasses_legacy_prompt(self):
        catalog = CatalogFilter(str(Path(__file__).resolve().parents[1] / 'data/train/train_tools_catalog.json'))
        text = 'Ghi cho Lann năm ba bảy, à mà thôi không cần nữa.'
        with patch('scripts.build_sft_dataset.build_user_prompt', side_effect=AssertionError('legacy called')):
            rows = build_rows(self.raw_item(text), catalog, raw_input=True)
        self.assertEqual(len(rows), 1)
        prompt = rows[0]['messages'][1]['content']
        self.assertEqual(json.loads(prompt.split('[USER UTTERANCE]\n')[1]), text)
        self.assertIn('Tên Lann, năm ba bảy', prompt)
        self.assertNotIn('x.wav', prompt)
        self.assertEqual(json.loads(rows[0]['messages'][-1]['content']),
                         {'thought': '', 'actions': [], 'is_finished': True})

    def test_raw_does_not_classify_thoi_as_no_action_or_expose_future_result(self):
        catalog = CatalogFilter(str(Path(__file__).resolve().parents[1] / 'data/train/train_tools_catalog.json'))
        item = self.raw_item('À thôi, xem ghi chú trước đã.')
        item['label'] = {'tool_calls': [{'name': 'notes.list', 'arguments': {}}],
                         'tool_results': [{'notes': [{'id': 'RAW_FUTURE_RESULT'}]}]}
        rows = build_rows(item, catalog, raw_input=True)
        self.assertEqual(json.loads(rows[0]['messages'][-1]['content'])['actions'][0]['name'], 'notes.list')
        self.assertNotIn('RAW_FUTURE_RESULT', rows[0]['messages'][1]['content'])
        self.assertIn('RAW_FUTURE_RESULT', rows[1]['messages'][1]['content'])

    def test_preserves_old_text_all_results_and_future_boundary(self):
        catalog = CatalogFilter(str(Path(__file__).resolve().parents[1] / 'data/train/train_tools_catalog.json'))
        history = [
            {'role': 'user', 'audio_text': 'OLD_REQUEST', 'audio': 'old.wav'},
            {'role': 'assistant', 'text': 'OLD_REPLY',
             'tool_calls': [{'name': 'notes.list', 'arguments': {}}],
             'tool_results': [{'notes': [{'id': f'old_{i}', 'content': 'x' * 800} for i in range(20)]}]},
        ]
        item = {'id': 'demo', 'audio_text': 'Xem ghi chú', 'history': history,
                'context': {'installed_apps': {'notes': {}}},
                'label': {'tool_calls': [{'name': 'notes.list', 'arguments': {}}],
                          'tool_results': [{'notes': [{'id': 'FUTURE_RESULT'}]}]}}
        rows = build_rows(item, catalog, full_history=True)
        prompt = rows[0]['messages'][1]['content']
        memory = prompt.split('[ACTIVE MEMORY FROM PREVIOUS DIALOGUE]\n', 1)[1].split('\n\n[CLOSEST REFERENCE EXEMPLAR]', 1)[0]
        restored = json.loads(memory.strip())
        self.assertEqual(restored, [{k: v for k, v in t.items() if k != 'audio'} for t in history])
        self.assertNotIn('FUTURE_RESULT', prompt)
        self.assertIn('FUTURE_RESULT', rows[1]['messages'][1]['content'])
        self.assertEqual(history[0]['audio'], 'old.wav')

    def test_shared_history_audio_stays_together_transitively(self):
        items = [
            {'id': 'a', 'audio': 'a.wav'},
            {'id': 'b', 'audio': 'b.wav', 'history': [{'audio': 'a.wav'}]},
            {'id': 'c', 'audio': 'c.wav', 'history': [{'audio': 'b.wav'}]},
            {'id': 'd', 'audio': 'd.wav', 'audio_text': 'independent'},
            {'id': 'e', 'audio': 'e.wav', 'audio_text': 'other'},
        ]
        # Distinct current text prevents input-dedup from grouping unrelated fixtures.
        for item in items:
            item.setdefault('audio_text', item['id'])
        split, groups = split_sources(items, val_ratio=0.2)
        self.assertEqual(groups, 3)
        self.assertEqual(split['a'], split['b'])
        self.assertEqual(split['b'], split['c'])
        self.assertEqual(set(split.values()), {'train', 'validation'})
        self.assertEqual(split, split_sources(items, val_ratio=0.2)[0])


if __name__ == '__main__':
    unittest.main()
