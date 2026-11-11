"""Compare two full ASR models on validation/train and public test using two GPUs.

Only reads JSONL manifests and referenced audio; never loads app environments.
Public transcripts are inference artifacts, not training labels or tool submissions.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def samples_to_manifest(jsonl, audio_root=None):
    source = Path(jsonl)
    root = Path(audio_root).resolve() if audio_root else source.resolve().parent
    rows, ids, audio_paths = [], set(), set()
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            identity = str(sample['id'])
            audio = (root / sample['audio']).resolve()
            if not audio.is_relative_to(root):
                raise ValueError(f'Audio escapes root: {audio}')
            if not audio.is_file():
                raise FileNotFoundError(audio)
            if identity in ids or str(audio) in audio_paths:
                raise ValueError('Duplicate sample ID/audio: compare requires unique current-turn clips')
            ids.add(identity)
            audio_paths.add(str(audio))
            text = sample.get('audio_text')
            rows.append({'sample_id': identity, 'audio': str(audio),
                         'text': text if isinstance(text, str) and text.strip() else None})
    if not rows:
        raise ValueError('Dataset has no samples')
    return {'validation': rows, 'source': str(source), 'scope': 'current user audio only; history is not a gold label'}


def compare_predictions(left, right):
    def index(rows):
        result = {}
        for row in rows:
            key = row['sample_id']
            if key in result:
                raise ValueError(f'Duplicate sample ID: {key}')
            result[key] = row
        return result
    a, b = index(left), index(right)
    if a.keys() != b.keys():
        raise ValueError('Models did not transcribe the same sample IDs')
    merged = []
    for identity, row in a.items():
        other = b[identity]
        if row['audio'] != other['audio'] or row['reference'] != other['reference']:
            raise ValueError('Audio/reference differs between models')
        merged.append({'id': identity, 'audio': row['audio'], 'reference': row['reference'],
                       'drive_text': row['prediction'], 'new_text': other['prediction']})
    return merged


def read_jsonl(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    with Path(path).open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-model', required=True, help='Restored full/merged model directory')
    parser.add_argument('--new-model', required=True, help='New full/merged model directory')
    labelled = parser.add_mutually_exclusive_group()
    labelled.add_argument('--validation-manifest', help='Saved split manifest from the training run')
    labelled.add_argument('--train-jsonl', help='Whole train current turns; training-set WER is not generalization')
    parser.add_argument('--train-audio-root')
    parser.add_argument('--test-jsonl', required=True)
    parser.add_argument('--test-audio-root')
    parser.add_argument('--out', required=True, help='New output directory')
    args = parser.parse_args()
    for model in (args.drive_model, args.new_model):
        if not (Path(model) / 'config.json').is_file():
            parser.error(f'Expected a full/merged model directory: {model}')
    if Path(args.drive_model).resolve() == Path(args.new_model).resolve():
        parser.error('Both paths identify the same model; select the actual newly trained model')
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error('Choose a new --out; existing comparison outputs are preserved')
    out.mkdir(parents=True, exist_ok=True)
    datasets = {}
    if args.validation_manifest:
        fixed = json.loads(Path(args.validation_manifest).read_text(encoding='utf-8'))
        datasets['validation'] = {'validation': fixed['validation'], 'source': args.validation_manifest}
    elif args.train_jsonl:
        datasets['train'] = samples_to_manifest(args.train_jsonl, args.train_audio_root)
    datasets['public_test'] = samples_to_manifest(args.test_jsonl, args.test_audio_root)
    reports = {}
    config = 'beam5_ngram0'
    for name, manifest in datasets.items():
        manifest_path = out / f'{name}.manifest.json'
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        predictions, scores = {}, {}
        for label, model in (('drive', args.drive_model), ('new', args.new_model)):
            job_out = out / f'{name}_{label}'
            subprocess.run([sys.executable, '-m', 'torch.distributed.run', '--standalone',
                            '--nproc_per_node=2', '-m', 'src.asr.benchmark',
                            '--manifest', str(manifest_path), '--model', model,
                            '--out', str(job_out), '--configs', '5:0'], check=True)
            report = json.loads((job_out / 'report.json').read_text(encoding='utf-8'))
            expected_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            if report['manifest_sha256'] != expected_hash:
                raise ValueError('Benchmark used a different audio manifest')
            scores[label] = report['scores'][config]
            predictions[label] = read_jsonl(job_out / f'{config}.predictions.jsonl')
            write_jsonl(out / f'{name}_{label}_transcripts.jsonl',
                        [{'id': row['sample_id'], 'audio_text': row['prediction']} for row in predictions[label]])
        paired = compare_predictions(predictions['drive'], predictions['new'])
        write_jsonl(out / f'{name}_comparison.jsonl', paired)
        reports[name] = {'models': scores,
                         'exact_transcript_disagreements': sum(row['drive_text'] != row['new_text'] for row in paired),
                         'note': 'No WER without gold audio_text. Train WER is not held-out quality.'}
        print(json.dumps({name: reports[name]}, ensure_ascii=False, indent=2), flush=True)
    summary = {'drive_model': args.drive_model, 'new_model': args.new_model,
               'decoding': {'num_beams': 5, 'no_repeat_ngram_size': 0}, 'datasets': reports,
               'note': 'ASR comparison only: no tool calls or competition state_hash submission is generated.'}
    (out / 'comparison_report.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
