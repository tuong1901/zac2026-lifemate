"""Agent regressions using synthetic tools; no public-test labels or state files."""
import contextlib
import io
import json
import unittest
from unittest.mock import patch
from src.agent.react_loop import ReActAgent
from src.rules.schema_guardrail import SafeSchemaGuardrail


class Catalog:
    def get_allowed_tools(self, installed):
        return [{'name': n} for n in ('demo.write', 'weather.get', 'notes.create', 'sys.ask_missing_info')]
    def format_tools_for_prompt(self, tools):
        return json.dumps(tools)


class Client:
    def __init__(self, plans):
        self.plans = iter(plans)
        self.calls = 0
    def generate(self, system, prompt):
        self.calls += 1
        plan = next(self.plans)
        return plan if isinstance(plan, str) else json.dumps(plan)


class Env:
    def __init__(self, results=(), invalid=()):
        self.results = iter(results)
        self.calls = []
        self.invalid = invalid
    def validate(self, name, args):
        return {'ok': args not in self.invalid, 'error': 'invalid' if args in self.invalid else None}
    def call(self, name, args):
        self.calls.append({'name': name, 'arguments': args})
        return next(self.results, {'ok': True})
    def get_tool_calls(self):
        return self.calls
    def get_state_hash(self):
        return 'synthetic-test-hash'


def action(value=1, name='demo.write'):
    return {'name': name, 'arguments': {'value': value}}


def plan(actions, finished=True):
    return {'actions': actions, 'is_finished': finished}


class AgentTests(unittest.TestCase):
    def run_agent(self, plans, env=None):
        env = env or Env()
        client = Client(plans)
        agent = ReActAgent(client, Catalog(), max_steps=5)
        item = {'id': 'synthetic', 'context': {'installed_apps': {'demo': {}, 'weather': {}}}, 'history': []}
        # Isolate execution logic from existing exemplar/phonetic heuristics.
        with patch('src.agent.react_loop.build_user_prompt', return_value='Synthetic request'), contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, item, 'Thực hiện thao tác yêu cầu')
        return env, client

    def test_failed_call_replans_and_can_retry(self):
        env, client = self.run_agent([plan([action(1), action(2)]), plan([action(1)])],
                                    Env([{'error': 'temporary failure'}, {'ok': True}]))
        self.assertEqual(env.calls, [action(1), action(1)])
        self.assertEqual(client.calls, 2)

    def test_false_done_stops_batch(self):
        env, _ = self.run_agent([plan([action(1), action(2)]), plan([])], Env([{'done': False}]))
        self.assertEqual(env.calls, [action(1)])

    def test_guardrail_failure_does_not_finish_or_run_remainder(self):
        env, _ = self.run_agent([plan([action(0), action(2)]), plan([action(1)])], Env(invalid=[{'value': 0}]))
        self.assertEqual(env.calls, [action(1)])

    def test_uninstalled_tool_never_reaches_environment(self):
        env, _ = self.run_agent([plan([action(name='unknown.write'), action(2)]), plan([action(1)])])
        self.assertEqual(env.calls, [action(1)])

    def test_distinct_queries_for_same_read_tool_are_preserved(self):
        a = {'name': 'weather.get', 'arguments': {'city': 'Huế'}}
        b = {'name': 'weather.get', 'arguments': {'city': 'Hà Nội'}}
        env, _ = self.run_agent([plan([a], False), plan([b], False), plan([])])
        self.assertEqual(env.calls, [a, b])

    def test_identical_query_not_repeated(self):
        a = {'name': 'weather.get', 'arguments': {'city': 'Huế'}}
        env, _ = self.run_agent([plan([a], False), plan([a])])
        self.assertEqual(env.calls, [a])

    def test_malformed_plan_recovers(self):
        for malformed in ('not json', '[]', '{"actions":"bad"}',
                          '{"action":{"name":"demo.write","arguments":[]}}',
                          '{"actions":[],"is_finished":"false"}',
                          '{"action":{"name":"demo.write","arguments":{"x":NaN}}}'):
            with self.subTest(malformed=malformed):
                env, _ = self.run_agent([malformed, plan([action()])])
                self.assertEqual(env.calls, [action()])

    def test_distinct_writes_to_same_tool_are_not_dropped(self):
        first = {'name': 'notes.create', 'arguments': {'title': 'First', 'content': 'One'}}
        second = {'name': 'notes.create', 'arguments': {'title': 'Second', 'content': 'Two'}}
        env, _ = self.run_agent([plan([first], False), plan([second])])
        self.assertEqual(env.calls, [first, second])

    def test_successful_batch_keeps_order(self):
        env, client = self.run_agent([plan([action(1), action(2)])])
        self.assertEqual(env.calls, [action(1), action(2)])
        self.assertEqual(client.calls, 1)


class GuardrailTests(unittest.TestCase):
    def test_preserve_schema_declared_optional_user_intent(self):
        cases = [('smart_home.fan_set', 'speed', 'integer', 3),
                 ('smart_home.light_set', 'color', 'string', 'red'),
                 ('alarm.update', 'enabled', 'boolean', False),
                 ('alarm.update', 'repeat', 'array', []),
                 ('calendar.update_event', 'title', 'string', 'New title'),
                 ('tasks.list', 'status', 'string', 'completed')]
        guard = SafeSchemaGuardrail()
        for name, key, kind, value in cases:
            with self.subTest(name=name, key=key):
                spec = {'parameters': {'properties': {key: {'type': kind}}}}
                self.assertEqual(guard.sanitize_arguments(name, {key: value, 'unknown': 1}, spec), {key: value})

    def test_integer_repair_does_not_truncate_or_coerce_boolean(self):
        guard = SafeSchemaGuardrail()
        self.assertIs(guard._repair_value('count', True, 'integer', None), True)
        self.assertEqual(guard._repair_value('count', 2.7, 'integer', None), 2.7)
        self.assertEqual(guard._repair_value('count', '3', 'integer', None), 3)


if __name__ == '__main__':
    unittest.main()
