"""Per-sample execution boundary shared by planned and helper tool calls."""
import copy
import math
import json
import time
from dataclasses import dataclass


def result_failed(result):
    return isinstance(result, dict) and (
        bool(result.get('error')) or result.get('ok') is False or
        result.get('success') is False or result.get('done') is False or
        result.get('status') in ('error', 'failed'))


class RunLimitReached(RuntimeError):
    pass


def affected_domains(name, arguments):
    """Invalidate app reads, including a payment's explicitly named funding app."""
    domains = {name.split('.')[0]}
    source = (arguments or {}).get('source')
    if source in ('bank', 'banking'):
        domains.add('banking')
    elif source == 'ewallet':
        domains.add('ewallet')
    return domains


class ActionFeedback(RuntimeError):
    def __init__(self, name, arguments, message, result=None, executed=False):
        super().__init__(message)
        self.message = str(message)
        self.name = name
        self.arguments = copy.deepcopy(arguments)
        self.result = copy.deepcopy(result)
        self.executed = executed


@dataclass
class RunBudget:
    deadline: float
    max_calls: int
    calls: int = 0

    @classmethod
    def start(cls, seconds, max_calls):
        if not math.isfinite(seconds) or seconds <= 0 or max_calls <= 0:
            raise ValueError('Execution limits must be finite and positive')
        return cls(time.monotonic() + seconds, max_calls)

    def remaining(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise RunLimitReached('Per-sample time budget exhausted')
        return remaining

    def before_call(self):
        self.remaining()
        if self.calls >= self.max_calls:
            raise RunLimitReached('Per-sample tool call budget exhausted')


class GuardedToolEnv:
    """Never creates/resets ToolEnv; validates every call, including helper branches."""
    def __init__(self, env, allowed_tools, guardrail, budget, read_tools=()):
        self._env = env
        self._allowed = {tool['name'] for tool in allowed_tools}
        self.guardrail = guardrail.bind(allowed_tools)
        self.budget = budget
        self.read_tools = frozenset(read_tools)
        self.mutated_domains = set()
        self.successful_signatures = set()
        self._specs = {tool['name']: tool for tool in allowed_tools}
        self._observed_refs = {}
        self._guard_reads = set()

    def observe(self, name, result):
        """Remember IDs from successful, named tool results, never hidden state."""
        if not isinstance(result, dict) or result_failed(result):
            return
        app = name.split('.')[0]
        for collection, values in result.items():
            if isinstance(values, list):
                ids = {str(value['id']) for value in values
                       if isinstance(value, dict) and value.get('id') is not None}
                self._observed_refs.setdefault(f'{app}.{collection}', set()).update(ids)

    def seed_history(self, history):
        for turn in history or []:
            calls, results = turn.get('tool_calls') or [], turn.get('tool_results') or []
            for call, result in zip(calls, results):
                if isinstance(call, dict):
                    self.observe(call.get('name', ''), result)

    def _check_references(self, name, arguments):
        spec = self._specs[name]
        if name in self.read_tools or spec.get('kind') == 'meta':
            return
        for key, prop in spec.get('parameters', {}).get('properties', {}).items():
            ref = prop.get('x-ref')
            if not ref or key not in arguments or str(arguments[key]) in self._observed_refs.get(ref, set()):
                continue
            app, collection = ref.split('.', 1)
            readers = [tool for tool in self._specs.values()
                       if tool['name'] in self.read_tools and tool['name'].split('.')[0] == app
                       and collection in str(tool.get('returns', ''))
                       and not tool.get('parameters', {}).get('required')]
            readers.sort(key=lambda tool: tool['name'])
            reader = next((tool for tool in readers if tool['name'] not in self._guard_reads), None)
            if reader:
                self._guard_reads.add(reader['name'])
                result = self.call(reader['name'], {})
                raise ActionFeedback(name, arguments,
                    f'WRITE deferred: observe {reader["name"]} result and choose the requested {key}; '
                    f'do not reuse an unobserved ID. Result: {json.dumps(result, ensure_ascii=False)}',
                    result=result)
            raise ActionFeedback(name, arguments,
                f'Unobserved {key} for {ref}; call a suitable READ with user-grounded filters '
                'or ask for clarification before writing')

    def validate(self, name, arguments):
        return self._env.validate(name, arguments)

    def get_tool_calls(self):
        return self._env.get_tool_calls()

    def get_state_hash(self):
        return self._env.get_state_hash()

    def call(self, name, arguments):
        self.budget.before_call()
        if name not in self._allowed:
            raise ActionFeedback(name, arguments, 'Tool is not installed/allowed; no call executed')
        if not isinstance(arguments, dict):
            raise ActionFeedback(name, {}, 'Arguments must be an object; no call executed')
        try:
            json.dumps(arguments, allow_nan=False)
        except (ValueError, TypeError) as error:
            raise ActionFeedback(name, arguments, "Arguments must be finite JSON values") from error
        valid, clean, error = self.guardrail.guard_and_validate(self._env, name, arguments)
        if not valid:
            raise ActionFeedback(name, clean, str(error))
        self._check_references(name, clean)
        self.budget.before_call()
        self.budget.calls += 1
        if name not in self.read_tools:
            domains = affected_domains(name, clean)
            self.mutated_domains.update(domains)
            stale = {signature for signature in self.successful_signatures
                     if signature.split(":", 1)[0] in self.read_tools
                     and signature.split(".", 1)[0] in domains}
            self.successful_signatures.difference_update(stale)
        # An exception has an unknown mutation outcome: deliberately do not retry it.
        result = self._env.call(name, clean)
        if result_failed(result):
            raise ActionFeedback(name, clean, str(result.get('error') or 'Tool execution failed'),
                                 result=result, executed=True)
        self.observe(name, result)
        return result


MISSING = object()


def successful_history_reads(history, read_tools, volatile_tools=()):
    """Cache only aligned, successful reads; later writes invalidate the same app."""
    cache = {}
    for turn in history or []:
        calls = turn.get('tool_calls') or []
        results = turn.get('tool_results') or []
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            name = call.get('name', '')
            if name not in read_tools:
                # Even an uncertain/failed mutation makes older observations unsafe.
                domains = affected_domains(name, call.get('arguments'))
                cache = {key: value for key, value in cache.items() if key.split('.')[0] not in domains}
                continue
            if name in volatile_tools or index >= len(results):
                continue
            result = results[index]
            if result is None or result_failed(result):
                continue
            # Only zero-argument snapshots can satisfy a later zero-argument read.
            if not (call.get('arguments') or {}):
                cache[name] = copy.deepcopy(result)
    return cache


def current_read_signatures(steps, read_tools):
    """Only successful reads after the latest write to that app can be reused."""
    signatures = set()
    for step in steps:
        name = step.get('tool_called', '')
        if not name:
            continue
        if name not in read_tools:
            domains = affected_domains(name, step.get('arguments'))
            signatures = {entry for entry in signatures if entry[0].split('.')[0] not in domains}
        elif not step.get('error'):
            signatures.add((name, json.dumps(step.get('arguments', {}), sort_keys=True)))
    return signatures


def observation_frontier(actions, read_tools):
    """Preserve order; defer writes that occur after a new read until it is observed."""
    seen_read = False
    for index, action in enumerate(actions):
        if action['name'] in read_tools:
            seen_read = True
        elif seen_read:
            return actions[:index], True
    # A batch ending in a read needs another planning turn before deciding to finish.
    return actions, seen_read and any(a['name'] not in read_tools for a in actions)


def history_cart_has_items(history, app):
    """Remember successful cart additions without treating writes as read results."""
    populated = False
    for turn in history or []:
        calls, results = turn.get('tool_calls') or [], turn.get('tool_results') or []
        for index, call in enumerate(calls):
            if index >= len(results) or results[index] is None or result_failed(results[index]):
                continue
            name = call.get('name', '')
            if name == f'{app}.add_to_cart':
                populated = True
            elif name in {f'{app}.checkout', f'{app}.clear_cart', f'{app}.remove_from_cart'}:
                populated = False
    return populated
