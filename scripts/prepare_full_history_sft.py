"""Prepare original-label SFT with complete dialogue and a source-group split.

No API calls, history mining, synthetic labels or public-test data.
Token counts include the model chat template and supervised answer.
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.build_sft_dataset import build_rows, read_jsonl
from src.rules.catalog_filter import CatalogFilter


def split_sources(items, val_ratio=0.1, seed=42):
    """Keep shared audio, explicit conversations and identical inputs together."""
    if not 0 < val_ratio < 1:
        raise ValueError('val_ratio must be between 0 and 1')
    parents = list(range(len(items)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen = {}
    for i, item in enumerate(items):
        keys = [('audio', str(t['audio']).replace('\\', '/'))
                for t in [item] + item.get('history', []) if t.get('audio')]
        if item.get('conversation_id') is not None:
            keys.append(('conversation', str(item['conversation_id'])))
        payload = {k: item.get(k) for k in ('audio_text', 'context', 'history')}
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                          sort_keys=True).encode()).hexdigest()
        keys.append(('input', digest))
        for key in keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i in range(len(items)):
        groups[find(i)].append(i)
    shuffled = list(groups.values())
    random.Random(seed).shuffle(shuffled)
    val_indices = set()
    for group in shuffled:
        if len(val_indices) < len(items) * val_ratio and len(val_indices) + len(group) < len(items):
            val_indices.update(group)
    if not val_indices:
        raise ValueError('Cannot create nonempty disjoint splits from these groups')
    return {item['id']: ('validation' if i in val_indices else 'train')
            for i, item in enumerate(items)}, len(groups)


def make_encoder(model, tokenizer_dir=None):
    if tokenizer_dir:
        # Offline tokenizer files + small runtime, useful without torch installed.
        sys.path.insert(0, str(Path(tokenizer_dir).resolve()))
        from tokenizers import Tokenizer
        from jinja2.sandbox import ImmutableSandboxedEnvironment
        path = Path(tokenizer_dir)
        tokenizer = Tokenizer.from_file(str(path / 'tokenizer.json'))
        tokenizer.no_truncation()
        tokenizer.no_padding()
        config = json.loads((path / 'tokenizer_config.json').read_text(encoding='utf-8'))
        template = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True).from_string(config['chat_template'])

        def encode(messages, generation=False):
            text = template.render(messages=messages, tools=None,
                                   add_generation_prompt=generation)
            return len(tokenizer.encode(text, add_special_tokens=False).ids)
        return encode
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model)

    def encode(messages, generation=False):
        return len(tokenizer.apply_chat_template(messages, tokenize=True,
                                                add_generation_prompt=generation))
    return encode


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train', default='data/train/train.jsonl')
    ap.add_argument('--catalog', default='data/train/train_tools_catalog.json')
    ap.add_argument('--out', default='out/sft_full_history')
    ap.add_argument('--model', default='Qwen/Qwen3-4B-Instruct-2507')
    ap.add_argument('--tokenizer-dir')
    ap.add_argument('--input-mode', choices=['raw', 'legacy'], default='raw',
                    help='raw: preserve transcript, no ASR rewrites or intent gate; legacy: previous V1 prompt')
    ap.add_argument('--max-tokens', type=int, default=8192)
    ap.add_argument('--val-ratio', type=float, default=0.1)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    if args.max_tokens <= 0:
        ap.error('--max-tokens must be positive')
    out = Path(args.out)
    # Avoid accidentally replacing a previous dataset release.
    out.mkdir(parents=True, exist_ok=False)
    items = read_jsonl(args.train)
    if len({i['id'] for i in items}) != len(items):
        raise ValueError('Duplicate source IDs')
    split, group_count = split_sources(items, args.val_ratio, args.seed)
    encode = make_encoder(args.model, args.tokenizer_dir)
    catalog = CatalogFilter(args.catalog)
    files = {name: (out / (name + '.jsonl')).open('w', encoding='utf-8')
             for name in ('train', 'validation', 'train_all_lengths',
                          'validation_all_lengths', 'overlength')}
    counts = defaultdict(int)
    lengths = []
    skipped_sources = defaultdict(set)
    kept_sources = defaultdict(set)
    try:
        with (out / 'lengths.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['id', 'source_id', 'split',
                                                  'input_tokens', 'total_tokens', 'kept'])
            writer.writeheader()
            for n, item in enumerate(items, 1):
                subset = split[item['id']]
                no_action = not item.get('label', {}).get('tool_calls', [])
                if no_action:
                    counts[subset + '_no_action_sources'] += 1
                for row in build_rows(item, catalog, full_history=True,
                                      raw_input=args.input_mode == 'raw'):
                    messages = row['messages']
                    length = encode(messages)
                    input_length = encode(messages[:-1], generation=True)
                    # Match kaggle_train_sft.py: len >= max_len is skipped.
                    kept = length < args.max_tokens
                    line = json.dumps(row, ensure_ascii=False) + '\n'
                    files[subset + '_all_lengths'].write(line)
                    files[subset if kept else 'overlength'].write(line)
                    counts[subset + ('_kept' if kept else '_overlength')] += 1
                    if no_action:
                        counts[subset + ('_no_action_kept' if kept else '_no_action_overlength')] += 1
                    (kept_sources if kept else skipped_sources)[subset].add(item['id'])
                    lengths.append(length)
                    writer.writerow(dict(id=row['id'], source_id=item['id'], split=subset,
                                         input_tokens=input_length, total_tokens=length, kept=kept))
                if n % 250 == 0:
                    print(f'Processed {n}/{len(items)} sources', flush=True)
    finally:
        for f in files.values():
            f.close()
    ordered = sorted(lengths)
    report = dict(model=args.model, source_file=args.train, source_sha256=hashlib.sha256(Path(args.train).read_bytes()).hexdigest(),
                  seed=args.seed, val_ratio=args.val_ratio, max_tokens_exclusive=args.max_tokens,
                  source_samples=len(items), source_groups=group_count,
                  split_sources={s: sum(v == s for v in split.values()) for s in ('train', 'validation')},
                  decision_samples=len(lengths), counts=dict(counts),
                  sources_all_steps_overlength={s: len(skipped_sources[s] - kept_sources[s]) for s in ('train', 'validation')},
                  total_tokens=dict(min=ordered[0], median=ordered[len(ordered)//2],
                                    p90=ordered[int(len(ordered)*0.9)], max=ordered[-1]),
                  thresholds={str(k): sum(v >= k for v in lengths) for k in (4096, 8192, 16384, 20000)},
                  input_mode=args.input_mode,
                  history_mode='complete ordered history excluding audio filenames; no text truncation',
                  augmentation=False)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    (out / 'split_manifest.json').write_text(json.dumps(split, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
