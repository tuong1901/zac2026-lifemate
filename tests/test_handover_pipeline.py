"""Handover regressions: invented scenarios only, no split environment access."""
import copy
import json
import random
import unittest
from pathlib import Path

from src.agent.execution import ActionFeedback, GuardedToolEnv, RunBudget
from src.agent.prompt import build_date_table, clean_asr_transcript
from src.asr.post_processor import AggressivePostProcessor, LexiconPostProcessor
from src.rules.schema_guardrail import SafeSchemaGuardrail
from scripts.build_sft_dataset import build_rows
from src.rules.catalog_filter import CatalogFilter

ROOT = Path(__file__).resolve().parent.parent


class HandoverTests(unittest.TestCase):
    def test_safe_normalization_preserves_names_and_spoken_quantities(self):
        text = 'Anh Đan đi đồ điểm phở lúc bảy rưỡi'
        self.assertEqual(clean_asr_transcript(text), text)
        self.assertEqual(AggressivePostProcessor().process(text), text)
        self.assertEqual(LexiconPostProcessor().process('za lo pay và mo mo, vê tê vê 3'), 'ZaloPay và MoMo, VTV3')

    def test_unicode_normalization(self):
        import unicodedata
        text = 'Đan đặt vé'
        self.assertEqual(clean_asr_transcript(unicodedata.normalize('NFD', text)), text)

    def test_calendar_rolls_across_year_and_preserves_local_day(self):
        table = build_date_table('2026-12-31T23:30:00-05:00')
        self.assertIn('Ngày mai: 2027-01-01', table)
        self.assertIn('Hôm qua: 2026-12-30', table)
        self.assertIn('2026-12-28', table)
        self.assertIn('2027-01-10', table)
        self.assertEqual(build_date_table('invalid'), '')

    def make_guard(self, required=False):
        class Env:
            def __init__(self):
                self.calls = []
            def validate(self, name, args):
                return {'ok': True}
            def call(self, name, args):
                self.calls.append((name, args))
                return {'ok': True, 'items': [{'id': 'real', 'name': 'Alpha'}]} if name == 'demo.list' else {'ok': True}
        tools = [
            {'name': 'demo.list', 'kind': 'read', 'returns': 'items[]',
             'parameters': {'properties': {'query': {'type': 'string'}}, 'required': ['query'] if required else []}},
            {'name': 'demo.update', 'kind': 'write',
             'parameters': {'properties': {'item_id': {'type': 'string', 'x-ref': 'demo.items'}}, 'required': ['item_id']}},
        ]
        env = Env()
        return env, GuardedToolEnv(env, tools, SafeSchemaGuardrail(), RunBudget.start(10, 10), {'demo.list'})

    def test_guard_defers_write_and_never_replaces_hallucinated_id(self):
        env, guard = self.make_guard()
        with self.assertRaises(ActionFeedback):
            guard.call('demo.update', {'item_id': 'fake'})
        self.assertEqual(env.calls, [('demo.list', {})])
        with self.assertRaises(ActionFeedback):
            guard.call('demo.update', {'item_id': 'fake'})
        self.assertEqual(len(env.calls), 1)
        guard.call('demo.update', {'item_id': 'real'})
        self.assertEqual(env.calls[-1], ('demo.update', {'item_id': 'real'}))

    def test_guard_does_not_invent_required_search_filters(self):
        env, guard = self.make_guard(required=True)
        with self.assertRaises(ActionFeedback):
            guard.call('demo.update', {'item_id': 'fake'})
        self.assertEqual(env.calls, [])

    def test_guard_uses_successful_history_but_not_failed_or_other_app_results(self):
        env, guard = self.make_guard()
        guard.seed_history([{'tool_calls': [{'name': 'other.list'}], 'tool_results': [{'items': [{'id': 'real'}]}]},
                            {'tool_calls': [{'name': 'demo.list'}], 'tool_results': [{'ok': False, 'items': [{'id': 'real'}]}]}])
        with self.assertRaises(ActionFeedback):
            guard.call('demo.update', {'item_id': 'real'})
        guard.call('demo.update', {'item_id': 'real'})
        self.assertEqual(len(env.calls), 2)
        env, guard = self.make_guard()
        guard.seed_history([{'tool_calls': [{'name': 'demo.list'}], 'tool_results': [{'items': [{'id': 'real'}]}]}])
        guard.call('demo.update', {'item_id': 'real'})
        self.assertEqual(len(env.calls), 1)

    def test_synthetic_replay_covers_all_apps(self):
        from scripts.synth_unseen_apps import GENS, build, replay_check
        catalog = json.loads((ROOT / 'data/public_test/public_test_tools_catalog.json').read_text(encoding='utf-8'))
        base = [{'context': {'current_time': '2026-12-31T09:00:00+07:00', 'installed_apps': {}}}]
        for app in GENS:
            rng = random.Random(42)
            for attempt in range(100):
                item = build(1, rng, base, catalog, app)
                if item is not None:
                    self.assertTrue(replay_check(item, catalog), app)
                    break
            else:
                self.fail(f'No valid scenario for {app}')

    def test_sft_does_not_reveal_future_results(self):
        catalog = CatalogFilter(str(ROOT / 'data/train/train_tools_catalog.json'))
        item = {'id': 'invented', 'audio_text': 'Xem ghi chú',
                'context': {'current_time': '2026-11-11T09:00:00+07:00', 'installed_apps': {'notes': {}}},
                'label': {'tool_calls': [{'name': 'notes.list', 'arguments': {}}],
                          'tool_results': [{'ok': True, 'notes': [{'id': 'secret_future_id', 'content': 'Test'}]}]}}
        rows = build_rows(item, catalog)
        self.assertEqual(len(rows), 2)
        self.assertNotIn('secret_future_id', rows[0]['messages'][1]['content'])
        self.assertIn('secret_future_id', rows[1]['messages'][1]['content'])
        self.assertIn('notes.list', rows[0]['messages'][1]['content'])
        stripped = copy.deepcopy(item)
        stripped['context']['installed_apps'] = {}
        stripped['label'] = {'tool_calls': [], 'tool_results': []}
        self.assertNotEqual(rows[0]['messages'][1]['content'], build_rows(stripped, catalog)[0]['messages'][1]['content'])


if __name__ == '__main__':
    unittest.main()