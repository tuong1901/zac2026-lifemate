"""Inspect downloaded model folders and safely unpack model ZIPs (stdlib only)."""
import json
import stat
from pathlib import Path
from zipfile import ZipFile


def unpack_model_zip(archive_path, destination):
    destination = Path(destination).resolve()
    with ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination) or stat.S_ISLNK(member.external_attr >> 16):
                raise ValueError(f"Unsafe ZIP member: {member.filename}")
        archive.extractall(destination)
    return destination


def discover_models(root):
    candidates = []
    root = Path(root)
    for config_path in sorted(root.rglob('adapter_config.json')):
        folder = config_path.parent
        if any((folder / name).is_file() and (folder / name).stat().st_size
               for name in ('adapter_model.safetensors', 'adapter_model.bin')):
            config = json.loads(config_path.read_text(encoding='utf-8'))
            candidates.append({'path': str(folder), 'kind': 'adapter',
                               'base_model': config.get('base_model_name_or_path'),
                               'rank': config.get('r')})
    for config_path in sorted(root.rglob('config.json')):
        folder = config_path.parent
        config = json.loads(config_path.read_text(encoding='utf-8'))
        if config.get('model_type') != 'whisper':
            continue
        complete = any((folder / name).is_file() and (folder / name).stat().st_size
                       for name in ('model.safetensors', 'pytorch_model.bin'))
        for name in ('model.safetensors.index.json', 'pytorch_model.bin.index.json'):
            index_path = folder / name
            if index_path.is_file():
                shards = set(json.loads(index_path.read_text(encoding='utf-8')).get('weight_map', {}).values())
                complete |= bool(shards) and all((folder / shard).resolve().is_relative_to(folder.resolve())
                                                 and (folder / shard).is_file()
                                                 and (folder / shard).stat().st_size for shard in shards)
        if complete:
            candidates.append({'path': str(folder), 'kind': 'full',
                               'd_model': config.get('d_model'),
                               'encoder_layers': config.get('encoder_layers'),
                               'has_processor': (folder / 'preprocessor_config.json').is_file()})
    return candidates


def ensure_final_checkpoint(state, control):
    """Step-based saving must include the final update even off the interval."""
    if state.max_steps > 0 and state.global_step >= state.max_steps:
        control.should_evaluate = True
        control.should_save = True
    return control
