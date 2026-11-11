"""Paired labelled/unlabelled comparison; invented manifests, no GPU."""
import json
import tempfile
import unittest
from pathlib import Path

from src.asr.compare import samples_to_manifest, compare_predictions
from src.asr.benchmark import score_predictions


class ComparisonTests(unittest.TestCase):
    def test_unlabelled_test_does_not_report_zero_wer(self):
        scores = score_predictions([{'reference': None, 'prediction': 'A'}])
        self.assertIsNone(scores['wer'])
        self.assertIsNone(scores['errors'])
        self.assertEqual(scores['labeled_clips'], 0)

    def test_mixed_labels_are_scored_only_where_gold_exists(self):
        scores = score_predictions([{'reference': 'A', 'prediction': 'B'},
                                    {'reference': None, 'prediction': 'anything'}])
        self.assertEqual(scores['wer'], 1)
        self.assertEqual(scores['labeled_clips'], 1)
        self.assertEqual(scores['clips'], 2)

    def test_public_manifest_never_uses_history_text_as_current_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'clip.wav').touch()
            source = root / 'public_test.jsonl'
            source.write_text(json.dumps({'id': 'x', 'audio': 'clip.wav',
                'history': [{'role': 'user', 'audio_text': 'Old utterance'}]}))
            rows = samples_to_manifest(source)['validation']
            self.assertEqual(len(rows), 1)
            self.assertIsNone(rows[0]['text'])
            self.assertEqual(rows[0]['sample_id'], 'x')

    def test_pairing_requires_same_ids_audio_and_reference(self):
        a = [{'sample_id': 'x', 'audio': 'a.wav', 'reference': None, 'prediction': 'old'}]
        b = [{'sample_id': 'x', 'audio': 'a.wav', 'reference': None, 'prediction': 'new'}]
        self.assertEqual(compare_predictions(a, b)[0]['new_text'], 'new')
        b[0]['audio'] = 'other.wav'
        with self.assertRaises(ValueError):
            compare_predictions(a, b)
        b[0]['sample_id'] = 'y'
        with self.assertRaises(ValueError):
            compare_predictions(a, b)


if __name__ == '__main__':
    unittest.main()
