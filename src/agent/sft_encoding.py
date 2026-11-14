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


def encode_trajectory(row, tokenizer, max_length):
    """Supervise every assistant turn, never the subsequent tool observations."""
    messages = row['messages']
    roles = [m['role'] for m in messages]
    if len(roles) < 3 or roles[0] != 'system' or roles[1:] != [
            'user' if i % 2 else 'assistant' for i in range(1, len(roles))]:
        raise ValueError('Expected alternating user/assistant trajectory after system')
    if roles[-1] != 'assistant':
        raise ValueError('Trajectory must end with assistant')
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    spans = []
    for i, message in enumerate(messages):
        if message['role'] != 'assistant':
            continue
        target = json.loads(message['content'])
        if not isinstance(target.get('actions'), list) or not isinstance(target.get('is_finished'), bool):
            raise ValueError('Invalid trajectory action target')
        prefix = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        through = tokenizer.apply_chat_template(messages[:i+1], tokenize=False, add_generation_prompt=False)
        if not full.startswith(through) or not through.startswith(prefix):
            raise ValueError('Trajectory template prefix mismatch')
        spans.append((len(prefix), len(through)))
    encoded = tokenizer(full, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = encoded['input_ids']
    if len(ids) > max_length:
        raise ValueError(f'{row.get("id")}: {len(ids)} tokens > {max_length}; no silent truncation/drop')
    labels = [-100] * len(ids)
    for left, right in spans:
        count = 0
        for i, (start, end) in enumerate(encoded['offset_mapping']):
            if start < left < end or start < right < end:
                raise ValueError('Token crosses trajectory supervision boundary')
            if end > start and start >= left and end <= right:
                labels[i] = ids[i]
                count += 1
        if not count:
            raise ValueError('Empty supervised assistant turn')
    return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}


def encode_sft_row(row, tokenizer, max_length):
    encoder = encode_trajectory if row.get('format') == 'trajectory-v1' else encode_decision
    return encoder(row, tokenizer, max_length)
