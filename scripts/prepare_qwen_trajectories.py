"""Remove repeated decision prefixes without dropping any v2 assistant targets.

Inference contract: retain initial system/user, append generated assistant JSON,
then append build_observation_message(actual_result), and generate again.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.agent.sft_encoding import check_split_rows


def build_observation_message(observation):
    return {'role': 'user', 'content': '[PREVIOUS STEP RESULT]\n' +
            json.dumps(observation, ensure_ascii=False)}


def split_prompt(text):
    before, rest = text.split('[PREVIOUS STEPS IN THIS SESSION]\n', 1)
    observations, after = rest.split('\n\n[USER UTTERANCE]\n', 1)
    return before, json.loads(observations), after


def convert(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['source_id']].append(row)
    result = []
    for source, steps in groups.items():
        steps.sort(key=lambda r: int(r['id'].rsplit('#step', 1)[1]))
        first = steps[0]
        base_before, initial, base_after = split_prompt(first['messages'][1]['content'])
        if initial:
            raise ValueError('First decision already has observations')
        messages = [dict(m) for m in first['messages'][:2]]
        previous = []
        for index, step in enumerate(steps):
            if int(step['id'].rsplit('#step', 1)[1]) != index:
                raise ValueError('Noncontiguous decision indices')
            if [m['role'] for m in step['messages']] != ['system', 'user', 'assistant']:
                raise ValueError('Unexpected v2 roles')
            before, observations, after = split_prompt(step['messages'][1]['content'])
            if before != base_before or after != base_after or step['messages'][0] != first['messages'][0]:
                raise ValueError('Context/history/tools changed across source decisions')
            if index:
                if len(observations) != len(previous)+1 or observations[:-1] != previous:
                    raise ValueError('Observations are not incremental')
                action = json.loads(messages[-1]['content'])
                observed = observations[-1]
                if action['is_finished'] or len(action['actions']) != 1:
                    raise ValueError('Cannot continue a terminal or batched action')
                expected = action['actions'][0]
                if expected['name'] != observed['tool_name'] or expected['arguments'] != observed['arguments']:
                    raise ValueError('Observation does not match previous action')
                messages.append(build_observation_message(observed))
            previous = observations
            messages.append(dict(step['messages'][-1]))
        if not json.loads(messages[-1]['content'])['is_finished']:
            raise ValueError('Missing final decision')
        result.append({'id': source, 'source_id': source, 'format': 'trajectory-v1',
                       'decision_ids': [s['id'] for s in steps], 'messages': messages})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('Choose a new empty output directory')
    originals, converted = {}, {}
    for split in ['train', 'validation']:
        originals[split] = [json.loads(line) for line in
                            (args.input / (split+'.jsonl')).read_text(encoding='utf-8').splitlines() if line]
        converted[split] = convert(originals[split])
        old = {r['id']: r['messages'][-1]['content'] for r in originals[split]}
        new = {key: m['content'] for r in converted[split]
               for key, m in zip(r['decision_ids'], [m for m in r['messages'] if m['role']=='assistant'])}
        if old != new:
            raise ValueError('Assistant targets changed')
    check_split_rows(converted['train'], converted['validation'])
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in converted.items():
        (args.out / (split+'.jsonl')).write_bytes((''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows)).encode())
    report = {'format': 'trajectory-v1', 'all_targets_preserved': True,
              'source_split_unchanged': True, 'splits': {
                  split: {'trajectories': len(converted[split]), 'assistant_decisions': len(originals[split])}
                  for split in converted}}
    (args.out / 'report.json').write_bytes((json.dumps(report, indent=2)+'\n').encode())
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
