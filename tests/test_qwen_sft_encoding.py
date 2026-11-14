import copy
import json
import unittest

from src.agent.sft_encoding import encode_decision, check_split_rows


class CharacterTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = ''.join('<' + m['role'] + '>\n' + m['content'] + '<END>\n' for m in messages)
        return text + ('<assistant>\n' if add_generation_prompt else '')

    def __call__(self, text, **kwargs):
        return {'input_ids': [ord(c) for c in text],
                'offset_mapping': [(i, i+1) for i in range(len(text))]}


class QwenEncodingTests(unittest.TestCase):
    def row(self):
        return {'id': 'a#step0', 'source_id': 'a', 'messages': [
            {'role': 'system', 'content': 'SYSTEM_SENTINEL'},
            {'role': 'user', 'content': 'Ghi cho Lann năm ba bảy, à mà thôi.'},
            {'role': 'assistant', 'content': json.dumps(
                {'thought': '', 'actions': [], 'is_finished': True}, ensure_ascii=False)}]}

    def test_only_assistant_and_terminator_are_supervised(self):
        row = self.row()
        encoded = encode_decision(row, CharacterTokenizer(), 1000)
        text = ''.join(chr(i) for i in encoded['input_ids'])
        start = text.index(row['messages'][-1]['content'])
        self.assertTrue(all(v == -100 for v in encoded['labels'][:start]))
        self.assertEqual(encoded['labels'][start:], encoded['input_ids'][start:])
        self.assertTrue(text.endswith('<END>\n'))
        self.assertEqual(len(encoded['labels']), len(encoded['input_ids']))

    def test_overlength_raises_instead_of_truncating(self):
        with self.assertRaisesRegex(ValueError, 'no silent'):
            encode_decision(self.row(), CharacterTokenizer(), 10)

    def test_template_mismatch_rejected(self):
        class BadTemplate(CharacterTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                text = super().apply_chat_template(messages, **kwargs)
                return 'DIFFERENT' + text if kwargs.get('add_generation_prompt') else text
        with self.assertRaisesRegex(ValueError, 'prefix mismatch'):
            encode_decision(self.row(), BadTemplate(), 1000)

    def test_source_overlap_rejected_even_if_step_ids_differ(self):
        other = copy.deepcopy(self.row())
        other['id'] = 'a#step1'
        with self.assertRaisesRegex(ValueError, 'share'):
            check_split_rows([self.row()], [other])
        other['id'], other['source_id'] = 'b#step0', 'b'
        check_split_rows([self.row()], [other])


if __name__ == '__main__':
    unittest.main()
