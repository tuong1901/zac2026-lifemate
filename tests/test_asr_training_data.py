"""ASR dataset safety tests with invented manifests, no GPU or real state files."""
import json
import tempfile
import unittest
from pathlib import Path

from src.asr.training_data import collect_clips, split_clips


class ASRTrainingDataTests(unittest.TestCase):
    def test_shared_audio_connects_conversations_and_deduplicates(self):
        clips = [
            {"sample_id": "a", "audio": "/shared.wav", "text": "Xin chào"},
            {"sample_id": "a", "audio": "/a.wav", "text": "A"},
            {"sample_id": "b", "audio": "/shared.wav", "text": "Xin chào"},
            {"sample_id": "b", "audio": "/b.wav", "text": "B"},
            {"sample_id": "c", "audio": "/c.wav", "text": "C"},
            {"sample_id": "d", "audio": "/d.wav", "text": "D"},
        ]
        train, val = split_clips(clips, 0.3, 42)
        self.assertEqual((train, val), split_clips(clips, 0.3, 42))
        self.assertFalse({c['audio'] for c in train} & {c['audio'] for c in val})
        self.assertFalse({c['sample_id'] for c in train} & {c['sample_id'] for c in val})
        ab_side = train if any(c['sample_id'] == 'a' for c in train) else val
        self.assertTrue(any(c['sample_id'] == 'b' for c in ab_side))
        self.assertEqual(len(train) + len(val), 5)

    def test_conflicting_transcripts_fail(self):
        rows = [{"sample_id": "a", "audio": "shared", "text": "one"},
                {"sample_id": "b", "audio": "shared", "text": "two"},
                {"sample_id": "c", "audio": "other", "text": "three"}]
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            split_clips(rows)

    def test_single_connected_group_fails_instead_of_leaking(self):
        with self.assertRaisesRegex(ValueError, "independent"):
            split_clips([{"sample_id": "a", "audio": "x", "text": "X"}])

    def test_collects_current_and_user_history_without_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'now.wav').touch()
            (root / 'before.wav').touch()
            item = {'id': 'invented', 'audio': 'now.wav', 'audio_text': 'Anh Đan bảy rưỡi',
                    'history': [{'role': 'user', 'audio': 'before.wav', 'audio_text': 'MoMo!'},
                                {'role': 'assistant', 'audio': 'missing.wav', 'audio_text': 'Ignore'}]}
            manifest = root / 'train.jsonl'
            manifest.write_text(json.dumps(item), encoding='utf-8')
            rows = collect_clips(manifest)
            self.assertEqual([row['text'] for row in rows], ['Anh Đan bảy rưỡi', 'MoMo!'])
            self.assertTrue(all(row['sample_id'] == 'invented' for row in rows))

    def test_missing_audio_and_path_escape_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / 'train.jsonl'
            for audio, error in [('missing.wav', FileNotFoundError), ('../outside.wav', ValueError)]:
                manifest.write_text(json.dumps({'id': 'a', 'audio': audio, 'audio_text': 'Hello'}), encoding='utf-8')
                with self.assertRaises(error):
                    collect_clips(manifest)


if __name__ == '__main__':
    unittest.main()
