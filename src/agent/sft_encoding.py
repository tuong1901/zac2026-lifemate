"""Assistant-only SFT encoding, independent of torch for audit/testing."""
import json


def encode_decision(row, tokenizer, max_length):
    messages = row.get('messages', [])
    if [m.get('role') for m in messages] != ['system', 'user', 'assistant']:
        raise ValueError(f"{row.get('id')}: expected one system/user/assistant decision")
    target = json.loads(messages[-1]['content'])
    if not isinstance(target.get('actions'), list) or not isinstance(target.get('is_finished'), bool):
        raise ValueError(f"{row.get('id')}: invalid action target")
    prefix = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError(f"{row.get('id')}: chat template prefix mismatch")
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    ids = encoded['input_ids']
    if len(ids) > max_length:
        raise ValueError(f"{row.get('id')}: {len(ids)} tokens > {max_length}; no silent truncation/drop")
    labels = [-100] * len(ids)
    for i, (start, end) in enumerate(encoded['offset_mapping']):
        if start < len(prefix) < end:
            raise ValueError(f"{row.get('id')}: token crosses assistant boundary")
        if end > start and start >= len(prefix):
            labels[i] = ids[i]
    if not any(v != -100 for v in labels):
        raise ValueError(f"{row.get('id')}: no assistant tokens")
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def check_split_rows(train_rows, validation_rows):
    def identifiers(rows):
        ids = [r['id'] for r in rows]
        if len(ids) != len(set(ids)):
            raise ValueError('Duplicate decision IDs')
        return set(ids), {r.get('source_id', r['id'].split('#')[0]) for r in rows}
    ti, ts = identifiers(train_rows)
    vi, vs = identifiers(validation_rows)
    if not train_rows or not validation_rows:
        raise ValueError('Train and validation must both be nonempty')
    if ti & vi or ts & vs:
        raise ValueError('Train/validation share a decision or source')
