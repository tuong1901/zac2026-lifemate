"""Restoration checks without real weights, cloud access or GPU dependencies."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

from src.asr.restore import discover_models, unpack_model_zip, ensure_final_checkpoint


class RestoreTests(unittest.TestCase):
    def test_separates_full_weights_and_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            full, adapter = root / 'full', root / 'adapter'
            full.mkdir()
            adapter.mkdir()
            (full / 'config.json').write_text(json.dumps({'model_type': 'whisper', 'd_model': 1280}))
            (full / 'model.safetensors').write_bytes(b'invented')
            (adapter / 'adapter_config.json').write_text(json.dumps({'r': 32, 'base_model_name_or_path': 'original'}))
            (adapter / 'adapter_model.safetensors').write_bytes(b'invented')
            self.assertEqual({item['kind'] for item in discover_models(root)}, {'full', 'adapter'})

    def test_partial_sharded_download_is_not_a_complete_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'config.json').write_text(json.dumps({'model_type': 'whisper'}))
            (root / 'model.safetensors.index.json').write_text(json.dumps({
                'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
            (root / 'part1.safetensors').write_bytes(b'invented')
            self.assertEqual(discover_models(root), [])
            (root / 'part2.safetensors').write_bytes(b'invented')
            self.assertEqual(len(discover_models(root)), 1)

    def test_zip_traversal_does_not_write_outside_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / 'model.zip'
            with ZipFile(archive, 'w') as stream:
                stream.writestr('../escaped', 'bad')
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                unpack_model_zip(archive, root / 'model')
            self.assertFalse((root / 'escaped').exists())

    def test_final_update_is_saved_even_between_checkpoint_intervals(self):
        control = SimpleNamespace(should_evaluate=False, should_save=False)
        ensure_final_checkpoint(SimpleNamespace(global_step=51, max_steps=141), control)
        self.assertFalse(control.should_save)
        ensure_final_checkpoint(SimpleNamespace(global_step=141, max_steps=141), control)
        self.assertTrue(control.should_save)
        self.assertTrue(control.should_evaluate)


if __name__ == '__main__':
    unittest.main()
