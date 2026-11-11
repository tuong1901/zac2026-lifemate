"""ASR dataset safety tests with invented manifests, no GPU or real state files."""
import json
import tempfile
import unittest
from pathlib import Path

from src.asr.training_data import (collect_clips, split_clips, cast_acoustic_features,
                                   select_saved_checkpoint, speed_output_length)
from src.asr.benchmark import score_predictions, decoding_configs


class ASRTrainingDataTests(unittest.TestCase):
    def test_speed_augmentation_preserves_whisper_window_without_cropping(self):
        self.assertEqual(speed_output_length(16000, 0.8), 20000)
        self.assertEqual(speed_output_length(16000, 1.25), 12800)
        self.assertEqual(speed_output_length(480000, 0.9), 480000)
        for invalid in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                speed_output_length(16000, invalid)

    def test_benchmark_uses_corpus_wer_and_same_normalization_as_training(self):
        rows = [{'reference': 'A b c d e f g h i', 'prediction': 'a b c d e f g h x'},
                {'reference': 'yes', 'prediction': 'no'}]
        scores = score_predictions(rows)
        self.assertEqual(scores['reference_words'], 10)
        self.assertEqual(scores['errors'], 2)
        self.assertAlmostEqual(scores['wer'], 0.2)
        self.assertAlmostEqual(scores['raw_wer'], 0.3)

    def test_decoding_configs_control_one_change_at_a_time(self):
        self.assertEqual(decoding_configs('1:3,1:0,5:0,1:3'), [(1, 3), (1, 0), (5, 0)])
        for invalid in ('0:3', '1:-1', 'bad'):
            with self.assertRaises(ValueError):
                decoding_configs(invalid)

    def test_recovery_selects_best_existing_checkpoint_not_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step, wer in ((141, 0.16), (282, 0.12), (423, 0.139)):
                checkpoint = root / f'checkpoint-{step}'
                checkpoint.mkdir()
                (checkpoint / 'adapter_config.json').write_text('{}')
                (checkpoint / 'adapter_model.safetensors').write_bytes(b'invented')
                (checkpoint / 'trainer_state.json').write_text(json.dumps({
                    'log_history': [{'step': step, 'eval_wer': wer}]}))
            checkpoint, details = select_saved_checkpoint(root)
            self.assertEqual(checkpoint.name, 'checkpoint-282')
            self.assertEqual(details['eval_wer'], 0.12)
            (checkpoint / 'adapter_model.safetensors').write_bytes(b'')
            self.assertEqual(select_saved_checkpoint(root)[0].name, 'checkpoint-423')

    def test_recovery_does_not_claim_success_without_saved_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'No complete checkpoint'):
                select_saved_checkpoint(directory)

    def test_generation_batch_matches_encoder_without_casting_integer_fields(self):
        class Tensor:
            def __init__(self, dtype):
                self.dtype = dtype
            def to(self, *, dtype):
                return Tensor(dtype)
        labels, mask = Tensor('long'), Tensor('long')
        batch = {'input_features': Tensor('float32'), 'labels': labels, 'attention_mask': mask}
        result = cast_acoustic_features(batch, 'float16')
        self.assertEqual(result['input_features'].dtype, 'float16')
        self.assertIs(result['labels'], labels)
        self.assertIs(result['attention_mask'], mask)
        self.assertEqual(cast_acoustic_features(result, 'float32')['input_features'].dtype, 'float32')

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
