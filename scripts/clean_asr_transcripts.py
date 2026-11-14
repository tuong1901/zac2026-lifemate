"""Create cleaned transcript copies plus per-edit audit; never overwrite source files.

python scripts/clean_asr_transcripts.py --input old.jsonl new.jsonl \
  --context-jsonl data/public_test/public_test.jsonl --out-dir out/asr_cleaned
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.asr.phrase_cleaner import PhraseCleaner


def records(path):
    with Path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True)
    parser.add_argument('--context-jsonl', help='Only id/context are used; no labels or environment access')
    parser.add_argument('--rules', help='Optional reviewed JSON rules')
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    contexts = {}
    if args.context_jsonl:
        for item in records(args.context_jsonl):
            if item['id'] in contexts:
                raise ValueError('Duplicate context ID')
            contexts[item['id']] = item.get('context', {})
    cleaner = PhraseCleaner(args.rules)
    prepared, names = [], set()
    for source in map(Path, args.input):
        if source.stem in names:
            raise ValueError('Input filenames must have different stems')
        names.add(source.stem)
        targets = [out / f'{source.stem}{suffix}' for suffix in ('_clean.jsonl', '_audit.jsonl', '_changes.csv', '_report.json')]
        if any(target.exists() or target == source.resolve() for target in targets):
            raise FileExistsError('Choose a new out-dir: existing inputs/outputs are never overwritten')
        rows = records(source)
        seen, cleaned, audit, counts = set(), [], [], collections.Counter()
        for item in rows:
            identity = item['id']
            if identity in seen:
                raise ValueError(f'Duplicate transcript ID: {identity}')
            seen.add(identity)
            if args.context_jsonl and identity not in contexts:
                raise ValueError(f'Missing context: {identity}')
            original = item['audio_text']
            text, edits = cleaner.clean_with_audit(original, contexts.get(identity))
            cleaned.append({**item, 'audio_text': text})
            if text != original:
                counts.update(edit['rule'] for edit in edits)
                audit.append({'id': identity, 'raw_text': original, 'clean_text': text, 'edits': edits})
        prepared.append((source, targets, rows, cleaned, audit, counts))
    out.mkdir(parents=True, exist_ok=True)
    for source, targets, rows, cleaned, audit, counts in prepared:
        for path, values in ((targets[0], cleaned), (targets[1], audit)):
            with path.open('w', encoding='utf-8') as stream:
                for item in values:
                    stream.write(json.dumps(item, ensure_ascii=False) + '\n')
        with targets[2].open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['id', 'rules', 'raw_text', 'clean_text'])
            writer.writeheader()
            for item in audit:
                writer.writerow({'id': item['id'], 'rules': ','.join(edit['rule'] for edit in item['edits']),
                                 'raw_text': item['raw_text'], 'clean_text': item['clean_text']})
        report = {'input': str(source.resolve()), 'output': str(targets[0]), 'samples': len(rows),
                  'changed_samples': len(audit), 'unchanged_samples': len(rows)-len(audit),
                  'edit_counts': dict(counts), 'context_supplied': bool(args.context_jsonl),
                  'note': 'Rule-based hypotheses, not verified gold. No WER improvement is asserted. Keep raw version for A/B.'}
        targets[3].write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
