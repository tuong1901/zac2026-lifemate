import copy
import json
import unittest

from scripts.prepare_qwen_trajectories import convert, build_observation_message
from src.agent.sft_encoding import encode_trajectory
from test_qwen_sft_encoding import CharacterTokenizer


class TrajectoryTests(unittest.TestCase):
    def rows(self):
        observation = {'tool_name': 'contacts.search', 'arguments': {'name': 'Lan'},
                       'result': {'id': 'FUTURE_SECRET'}}
        targets = [{'thought': '', 'actions': [{'name': 'contacts.search', 'arguments': {'name': 'Lan'}}],
                    'is_finished': False}, {'thought': '', 'actions': [], 'is_finished': True}]
        rows = []
        for i, target in enumerate(targets):
            prompt = ('CONTEXT\n[PREVIOUS STEPS IN THIS SESSION]\n' +
                      json.dumps([observation] if i else []) + '\n\n[USER UTTERANCE]\nCall Lan')
            rows.append({'id': 's#step'+str(i), 'source_id': 's', 'messages': [
                {'role': 'system', 'content': 'SYSTEM'}, {'role': 'user', 'content': prompt},
                {'role': 'assistant', 'content': json.dumps(target)}]})
        return rows, observation

    def test_preserves_all_targets_and_no_future_observation(self):
        rows, observation = self.rows()
        result = convert(rows)[0]
        messages = result['messages']
        self.assertEqual([m['content'] for m in messages if m['role']=='assistant'],
                         [r['messages'][-1]['content'] for r in rows])
        self.assertNotIn('FUTURE_SECRET', ''.join(m['content'] for m in messages[:3]))
        self.assertEqual(messages[3], build_observation_message(observation))

    def test_all_assistants_supervised_but_observations_masked(self):
        row = convert(self.rows()[0])[0]
        tok = CharacterTokenizer()
        enc = encode_trajectory(row, tok, 10000)
        rendered = tok.apply_chat_template(row['messages'])
        supervised = ''.join(chr(i) for i in enc['labels'] if i != -100)
        self.assertNotIn('FUTURE_SECRET', supervised)
        for m in row['messages']:
            if m['role']=='assistant':
                self.assertIn(m['content']+'<END>\n', supervised)
        self.assertEqual(supervised.count('<END>'), 2)
        self.assertEqual(len(rendered), len(enc['labels']))

    def test_changed_context_or_action_result_rejected(self):
        rows, _ = self.rows()
        bad = copy.deepcopy(rows)
        bad[1]['messages'][1]['content'] = bad[1]['messages'][1]['content'].replace('CONTEXT', 'CHANGED')
        with self.assertRaisesRegex(ValueError, 'Context'):
            convert(bad)
        bad = copy.deepcopy(rows)
        bad[1]['messages'][1]['content'] = bad[1]['messages'][1]['content'].replace('contacts.search', 'wrong')
        with self.assertRaisesRegex(ValueError, 'match'):
            convert(bad)

    def test_no_truncation(self):
        with self.assertRaisesRegex(ValueError, 'no silent'):
            encode_trajectory(convert(self.rows()[0])[0], CharacterTokenizer(), 10)
