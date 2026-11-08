"""Select observed entities without replacing valid targets by list position."""
import re
import unicodedata

from src.agent.execution import ActionFeedback, result_failed


def observed_entities(history, steps, source_tools, collection, singular=None):
    """Collect successful results from explicitly named tools, newest first."""
    sources = []
    for step in reversed(steps or []):
        if step.get('tool_called') in source_tools and not step.get('error'):
            sources.append(step.get('raw_observation'))
    for turn in reversed(history or []):
        calls, results = turn.get('tool_calls') or [], turn.get('tool_results') or []
        for index in reversed(range(min(len(calls), len(results)))):
            if isinstance(calls[index], dict) and calls[index].get('name') in source_tools:
                sources.append(results[index])
    entities, seen = [], set()
    for result in sources:
        if not isinstance(result, dict) or result_failed(result):
            continue
        values = result.get(collection)
        values = values if isinstance(values, list) else []
        if singular and isinstance(result.get(singular), dict):
            values = values + [result[singular]]
        for entity in values:
            if not isinstance(entity, dict):
                continue
            identity = (str(entity.get('id', '')), str(entity.get('phone', entity.get('phone_number', ''))),
                        str(entity.get('name', '')), str(entity.get('address', '')))
            if identity not in seen:
                seen.add(identity)
                entities.append(entity)
    return entities


def _normalized(text):
    return ' '.join(re.findall(r'\w+', unicodedata.normalize('NFC', str(text)).casefold()))


def select_observed_entity(tool_name, arguments, argument_key, entities, user_text):
    """Preserve an observed selection, or resolve a unique requested entity.

    A lone unrelated result is not evidence for replacing an unknown ID.
    """
    if not entities:
        return None
    target = arguments.get(argument_key)
    selected = next((entity for entity in entities
                     if target is not None and str(entity.get('id')) == str(target)), None)
    # The utterance can describe several actions. A name mentioned for another
    # action must not replace this action's already-observed target.
    if selected is not None:
        return selected
    # Preserve a selected contact number when the environment has observed it.
    for key in ('phone', 'phone_number'):
        number = arguments.get(key)
        if number:
            matches = [entity for entity in entities
                       if str(number) in {str(entity.get('phone')), str(entity.get('phone_number'))}]
            if len(matches) == 1:
                return matches[0]
    utterance = f' {_normalized(user_text)} '
    matched = []
    for entity in entities:
        labels = [_normalized(entity.get(key, '')) for key in ('name', 'title', 'label', 'address')]
        if any(label and f' {label} ' in utterance for label in labels):
            matched.append(entity)
    # Repeated observations of the same entity do not create ambiguity.
    unique = {}
    for entity in matched:
        identity = str(entity.get('id') or entity.get('phone') or entity.get('phone_number') or '')
        unique.setdefault(identity, entity)
    candidates = list(unique.values())
    if len(candidates) == 1:
        return candidates[0]
    raise ActionFeedback(tool_name, arguments,
        f'Unresolved {argument_key}: use an observed ID matching the request; search or ask for clarification if needed')
