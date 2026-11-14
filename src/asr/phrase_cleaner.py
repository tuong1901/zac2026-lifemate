"""Opt-in contextual compound-word repair, with audit records and protected fields.

Rules do not use sample IDs, reference transcripts, another model's output or hidden state.
Corrections are hypotheses for downstream A/B, not ground-truth labels.
"""
import json
import re
import unicodedata
from pathlib import Path

DEFAULT_RULES = Path(__file__).resolve().parents[2] / 'configs/asr_phrase_rules.json'
QUOTED = re.compile(r'"[^"\n]*"|“[^”\n]*”|‘[^’\n]*’|\x27[^\x27\n]*\x27')
NUMBER_WORDS = re.compile(r'\b(?:một|hai|ba|bốn|tư|năm|sáu|bảy|tám|chín|mười|mươi|trăm|nghìn|ngàn|triệu|tỷ|rưỡi|kém)\b', re.I)
LOGIC_WORDS = re.compile(r'\b(?:không|chưa|đừng|nếu|thì|hủy|huỷ|thôi)\b', re.I)


def protected_values(text):
    text = unicodedata.normalize('NFC', text).casefold()
    return (re.findall(r'\d+(?:[.,:/-]\d+)*', text), NUMBER_WORDS.findall(text), LOGIC_WORDS.findall(text))


class PhraseCleaner:
    def __init__(self, rules_path=None):
        rows = json.loads(Path(rules_path or DEFAULT_RULES).read_text(encoding='utf-8'))
        self.rules = []
        ids = set()
        for row in rows:
            if row['id'] in ids:
                raise ValueError(f'Duplicate rule ID: {row["id"]}')
            ids.add(row['id'])
            pattern = re.compile(row['pattern'], re.I)
            if 'bad' not in pattern.groupindex:
                raise ValueError(f'Rule needs a named bad span: {row["id"]}')
            self.rules.append((row, pattern))

    def clean_with_audit(self, text, context=None):
        if not isinstance(text, str):
            raise TypeError('Transcript must be a string')
        normalized = re.sub(r'\s+', ' ', unicodedata.normalize('NFC', text)).strip()
        result, edits = normalized, []
        if normalized != text:
            edits.append({'rule': 'unicode_whitespace', 'before': text, 'after': normalized})
        installed = (context or {}).get('installed_apps', {}) or {}
        apps = set(installed)
        for rule, pattern in self.rules:
            if rule.get('apps') and not apps.intersection(rule['apps']):
                continue
            if rule.get('context_pattern') and not re.search(rule['context_pattern'], result, re.I):
                continue
            quotes = [match.span() for match in QUOTED.finditer(result)]
            rule_input = result

            def replace(match):
                start, end = match.span('bad')
                if any(start < right and end > left for left, right in quotes):
                    return match.group(0)
                if rule.get('block_before') and re.search(rule['block_before'], rule_input[:start], re.I):
                    return match.group(0)
                old = match.group('bad')
                replacement = rule['replacement'].format(**match.groupdict())
                if not rule.get('canonical_case'):
                    if old.isupper():
                        replacement = replacement.upper()
                    elif old[:1].isupper():
                        replacement = replacement[:1].upper() + replacement[1:]
                if old == replacement:
                    return match.group(0)
                edits.append({'rule': rule['id'], 'before': old, 'after': replacement,
                              'start': start, 'end': end, 'offset_basis': 'input_to_this_rule'})
                relative = start - match.start()
                return match.group(0)[:relative] + replacement + match.group(0)[relative + end - start:]

            result = pattern.sub(replace, result)
        if protected_values(normalized) != protected_values(result):
            raise ValueError('Rule attempted to change numbers or logical/cancellation markers; output rejected')
        return result, edits

    def process(self, text, context=None):
        return self.clean_with_audit(text, context)[0]

    def process_batch(self, texts, contexts=None):
        if contexts is None:
            contexts = [None] * len(texts)
        if len(texts) != len(contexts):
            raise ValueError('One context is required per transcript')
        return [self.process(text, context) for text, context in zip(texts, contexts)]
