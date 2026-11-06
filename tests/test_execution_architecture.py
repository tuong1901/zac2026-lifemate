import contextlib
import io
import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.agent.execution import (ActionFeedback, GuardedToolEnv, RunBudget,
                                 RunLimitReached, successful_history_reads, history_cart_has_items)
from src.agent.llm_client import drop_reference_examples, OpenAICompatibleClient
from src.agent.prompt import build_user_prompt, get_history_satisfied_reads, format_tools_with_enums
from src.agent.react_loop import ReActAgent
from src.rules.schema_guardrail import SafeSchemaGuardrail
from test_agent_regressions import Env, Client, Catalog, action, plan


class HistoryTests(unittest.TestCase):
    def turn(self, name, result, args=None):
        return {'role': 'assistant', 'tool_calls': [{'name': name, 'arguments': args or {}}],
                'tool_results': [result]}

    def test_result_shape_does_not_cross_app_boundary(self):
        history = [self.turn('train_bus_tickets.my_bookings', {'bookings': [{'id': 'test'}]})]
        self.assertNotIn('flights.my_bookings', get_history_satisfied_reads(history))
        history = [self.turn('grocery.search', {'products': [{'id': 'test'}]})]
        self.assertNotIn('ecommerce.search', get_history_satisfied_reads(history))

    def test_missing_failed_and_parameterized_results_not_cached(self):
        history = [self.turn('notes.list', {'error': 'failed'}),
                   {'tool_calls': [{'name': 'notes.list'}]},
                   self.turn('notes.list', {'notes': []}, {'query': 'abc'})]
        self.assertEqual(successful_history_reads(history, {'notes.list'}), {})

    def test_cart_history_tracks_successful_add_and_checkout(self):
        history = [self.turn('grocery.add_to_cart', {'ok': True})]
        self.assertTrue(history_cart_has_items(history, 'grocery'))
        self.assertFalse(history_cart_has_items(history, 'ecommerce'))
        history.append(self.turn('grocery.checkout', {'ok': True}))
        self.assertFalse(history_cart_has_items(history, 'grocery'))
        history.append(self.turn('grocery.add_to_cart', {'error': 'failed'}))
        self.assertFalse(history_cart_has_items(history, 'grocery'))

    def test_latest_read_wins_and_write_invalidates(self):
        history = [self.turn('notes.list', {'notes': [1]}), self.turn('notes.list', {'notes': [2]})]
        self.assertEqual(successful_history_reads(history, {'notes.list'})['notes.list'], {'notes': [2]})
        history.append(self.turn('notes.create', {'ok': True}))
        self.assertEqual(successful_history_reads(history, {'notes.list'}), {})

    def test_explicit_refresh_ignores_cache(self):
        history = [self.turn('notes.list', {'notes': [1]})]
        self.assertIn('notes.list', get_history_satisfied_reads(history))
        self.assertNotIn('notes.list', get_history_satisfied_reads(history, 'kiểm tra lại ghi chú'))


class BoundaryTests(unittest.TestCase):
    def wrap(self, env=None, budget=None):
        env = env or Env()
        return GuardedToolEnv(env, [{'name': 'demo.write'}], SafeSchemaGuardrail(),
                              budget or RunBudget.start(20, 2))

    def test_helper_calls_cannot_bypass_allowlist_or_schema(self):
        env = Env(invalid=[{'value': 0}])
        wrapper = self.wrap(env)
        for name, args in [('unknown', {}), ('demo.write', {'value': 0}), ('demo.write', {'value': float('nan')})]:
            with self.assertRaises(ActionFeedback):
                wrapper.call(name, args)
        self.assertEqual(env.calls, [])

    def test_tool_failure_is_explicit_and_logged_once(self):
        env = Env([{'success': False}])
        with self.assertRaises(ActionFeedback) as caught:
            self.wrap(env).call('demo.write', {})
        self.assertTrue(caught.exception.executed)
        self.assertEqual(len(env.calls), 1)

    def test_deadline_and_call_limit_cover_every_call(self):
        env = Env()
        wrapper = self.wrap(env, RunBudget.start(20, 1))
        wrapper.call('demo.write', {})
        with self.assertRaises(RunLimitReached):
            wrapper.call('demo.write', {})
        with patch('src.agent.execution.time.monotonic', return_value=30):
            with self.assertRaises(RunLimitReached):
                self.wrap(env, RunBudget(29, 1)).call('demo.write', {})
        self.assertEqual(len(env.calls), 1)

    def test_unknown_mutation_outcome_is_not_retried(self):
        class Broken(Env):
            def call(self, name, args):
                super().call(name, args)
                raise OSError('connection lost after mutation')
        env = Broken()
        with self.assertRaises(OSError):
            self.wrap(env).call('demo.write', {})
        self.assertEqual(len(env.calls), 1)


class PromptTests(unittest.TestCase):
    def make_prompt(self, label, **kwargs):
        return build_user_prompt(label, {'current_time': '2026-11-06T10:00:00+07:00'},
                                 '', '', 'no examples', [], **kwargs)

    def test_explicit_history_and_full_nested_schema(self):
        schema = {'type': 'object', 'properties': {'rows': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'amount': {'type': 'integer', 'minimum': 1}},
            'required': ['amount']}}}, 'required': ['rows']}
        tools = [{'name': 'demo.write', 'parameters': schema}]
        prompt = self.make_prompt('Create', allowed_tools=tools,
                                 history=[{'role': 'user', 'audio_text': 'HISTORY_SENTINEL'}])
        self.assertIn('HISTORY_SENTINEL', prompt)
        self.assertIn('"minimum":1', prompt)
        self.assertEqual(json.loads(format_tools_with_enums(tools, ''))['parameters'], schema)

    def test_prompt_does_not_mutate_caller_or_llm(self):
        client = Client([])
        original = client.generate
        executed_call_signatures = {'loyalty.redeem:{}'}
        self.llm_client = client  # Used to trigger the old implicit frame mutation.
        self.make_prompt('hello')
        self.assertEqual(executed_call_signatures, {'loyalty.redeem:{}'})
        self.assertEqual(client.generate, original)
        self.assertFalse(hasattr(client, '_retry_wrapped'))

    def test_parallel_prompts_do_not_mix_observations(self):
        def create(i):
            return self.make_prompt(f'REQUEST_{i}', asr_variants={'whisper_turbo': f'VARIANT_{i}'})
        with ThreadPoolExecutor(max_workers=4) as pool:
            prompts = list(pool.map(create, range(8)))
        for i, prompt in enumerate(prompts):
            self.assertIn(f'VARIANT_{i}', prompt)
            for j in range(8):
                if i != j:
                    self.assertNotIn(f'VARIANT_{j}', prompt)

    def test_budget_reduction_preserves_evidence(self):
        prompt = 'SCHEMA\n[PREVIOUS STEPS]\nEVIDENCE\n[CLOSEST REFERENCE EXEMPLAR]\nDEMO\n[USER UTTERANCE]\nREQUEST'
        compact = drop_reference_examples(prompt)
        self.assertNotIn('DEMO', compact)
        self.assertIn('EVIDENCE', compact)
        self.assertIn('REQUEST', compact)


class AgentArchitectureTests(unittest.TestCase):
    def run_agent(self, plans, names, text='Thực hiện yêu cầu', history=(), budget=28, max_calls=48):
        class Tools(Catalog):
            def get_allowed_tools(self, installed):
                return [{'name': name} for name in names]
        env, client = Env(), Client(plans)
        agent = ReActAgent(client, Tools(), time_budget=budget, max_calls=max_calls)
        item = {'id': 'synthetic', 'context': {'installed_apps': {n.split('.')[0]: {} for n in names}},
                'history': list(history)}
        with patch('src.agent.react_loop.build_user_prompt', return_value='prompt'), contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, item, text)
        return env, client

    def test_read_then_write_requires_observation_turn(self):
        read = {'name': 'notes.list', 'arguments': {}}
        write = {'name': 'notes.create', 'arguments': {'title': 'One'}}
        env, client = self.run_agent([plan([read, write]), plan([write])], ['notes.list', 'notes.create'])
        self.assertEqual(env.calls, [read, write])
        self.assertEqual(client.calls, 2)

    def test_write_before_read_keeps_user_order(self):
        write = {'name': 'notes.create', 'arguments': {'title': 'One'}}
        read = {'name': 'notes.list', 'arguments': {}}
        env, _ = self.run_agent([plan([write, read]), plan([])], ['notes.list', 'notes.create'])
        self.assertEqual(env.calls, [write, read])

    def test_read_is_refreshed_after_same_app_write(self):
        read = {'name': 'notes.list', 'arguments': {}}
        write = {'name': 'notes.create', 'arguments': {'title': 'One'}}
        env, _ = self.run_agent([plan([read], False), plan([write], False), plan([read])],
                               ['notes.list', 'notes.create'])
        self.assertEqual(env.calls, [read, write, read])

    def test_repeat_redemption_is_bounded_by_explicit_batch(self):
        redeem = {'name': 'loyalty.redeem', 'arguments': {'voucher_id': 'v1'}}
        env, _ = self.run_agent([plan([redeem, redeem], False), plan([redeem, redeem])], ['loyalty.redeem'],
                               'Đổi hai vé')
        self.assertEqual(env.calls, [redeem, redeem])

    def test_late_generation_cannot_start_tool_call(self):
        clock = [0.0]
        class LateClient:
            def generate(self, *_):
                clock[0] = 29.0
                return json.dumps(plan([action()]))
        env = Env()
        with patch('src.agent.execution.time.monotonic', side_effect=lambda: clock[0]), \
             patch('src.agent.react_loop.build_user_prompt', return_value='prompt'), \
             contextlib.redirect_stdout(io.StringIO()):
            ReActAgent(LateClient(), Catalog()).run(env,
                {'id': 'synthetic', 'context': {'installed_apps': {'demo': {}}}}, 'Thực hiện yêu cầu')
        self.assertEqual(env.calls, [])

    def test_budget_stops_mid_batch(self):
        env, _ = self.run_agent([plan([action(1), action(2)])], ['demo.write'], max_calls=1)
        self.assertEqual(env.calls, [action(1)])

    def test_empty_unfinished_plan_requests_correction(self):
        env, _ = self.run_agent([plan([], False), plan([action()])], ['demo.write'])
        self.assertEqual(env.calls, [action()])

    def test_ordinal_task_does_not_drop_second_request(self):
        history = [{'role': 'assistant', 'tool_calls': [{'name': 'tasks.list', 'arguments': {}}],
                    'tool_results': [{'tasks': [{'id': 't1'}, {'id': 't2'}]}]}]
        first = {'name': 'tasks.complete', 'arguments': {'task_id': 't2'}}
        second = {'name': 'notes.create', 'arguments': {'title': 'Done'}}
        env, client = self.run_agent([plan([first, second])], ['tasks.complete', 'notes.create'],
                                    'Đánh dấu việc thứ hai rồi tạo ghi chú', history)
        self.assertEqual(env.calls, [first, second])
        self.assertEqual(client.calls, 1)

    def test_automatic_helper_uses_same_validation_boundary(self):
        class Tools(Catalog):
            def get_allowed_tools(self, installed):
                return [{'name': 'email.create_draft'}, {'name': 'contacts.search'}]
        draft = {'name': 'email.create_draft', 'arguments': {'to': 'Lan', 'subject': 'Hi'}}
        env = Env(invalid=[{'query': 'Lan'}])
        client = Client([plan([draft]), plan([])])
        agent = ReActAgent(client, Tools())
        with patch('src.agent.react_loop.build_user_prompt', return_value='prompt'), contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, {'id': 'synthetic', 'context': {'installed_apps': {'email': {}, 'contacts': {}}}},
                      'Soạn thư cho Lan')
        self.assertEqual(env.calls, [])
        self.assertEqual(client.calls, 2)

    def test_real_prompt_builder_integrates_with_loop(self):
        env, client = Env(), Client([plan([action()])])
        agent = ReActAgent(client, Catalog())
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, {'id': 'synthetic', 'history': [],
                           'context': {'current_time': '2026-11-06T10:00:00+07:00', 'installed_apps': {'demo': {}}}},
                      'Thực hiện thao tác')
        self.assertEqual(env.calls, [action()])
        self.assertFalse(hasattr(client, '_retry_wrapped'))

    def test_ambiguous_placeholder_is_not_replaced_by_first_entity(self):
        args = {'task_id': 'placeholder'}
        steps = [{'tool_called': 'tasks.list', 'raw_observation': {'tasks': [
            {'id': 'a', 'title': 'Alpha'}, {'id': 'b', 'title': 'Beta'}]}}]
        with self.assertRaises(ActionFeedback):
            ReActAgent._resolve_grounded_arguments('tasks.complete', args, [], steps, 'Hoàn thành công việc')
        self.assertEqual(args['task_id'], 'placeholder')

    def test_unique_entity_match_can_ground_placeholder(self):
        args = {'task_id': 'placeholder'}
        steps = [{'tool_called': 'tasks.list', 'raw_observation': {'tasks': [
            {'id': 'a', 'title': 'Alpha'}, {'id': 'b', 'title': 'Beta'}]}}]
        ReActAgent._resolve_grounded_arguments('tasks.complete', args, [], steps, 'Hoàn thành Alpha')
        self.assertEqual(args['task_id'], 'a')

    def test_server_client_uses_remaining_timeout_without_retries(self):
        calls = []
        class Fake:
            def with_options(self, **kwargs):
                calls.append(kwargs)
                return self
            @property
            def chat(self): return SimpleNamespace(completions=self)
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{}'))])
        client = object.__new__(OpenAICompatibleClient)
        client.client, client.model, client.max_tokens, client.timeout = Fake(), 'local-model', 128, 25
        self.assertEqual(client.generate_with_budget('sys', 'request', 3), '{}')
        self.assertEqual(calls, [{'timeout': 3, 'max_retries': 0}])
        def timeout(**kwargs):
            raise TimeoutError("test timeout")
        client.client.create = timeout
        with self.assertRaises(RunLimitReached):
            client.generate_with_budget('sys', 'request', 3)
        self.assertEqual(len(calls), 2)  # One attempt per generation, no hidden retries.



class NestedSchemaTests(unittest.TestCase):
    def test_nested_numbers_null_and_empty_values(self):
        spec = {'parameters': {'properties': {'rows': {'type': 'array', 'items': {
            'type': 'object', 'properties': {'count': {'type': 'integer'}}, 'additionalProperties': False}},
            'optional': {'type': ['string', 'null']}}}}
        result = SafeSchemaGuardrail().sanitize_arguments('demo.write',
            {'rows': [{'count': '3', 'garbage': 4}], 'optional': None}, spec)
        self.assertEqual(result, {'rows': [{'count': 3}], 'optional': None})

    def test_non_finite_numeric_text_not_silently_cast(self):
        guard = SafeSchemaGuardrail()
        self.assertEqual(guard._repair_value('amount', 'NaN', 'number', None), 'NaN')
        self.assertEqual(guard._repair_value('amount', 'Infinity', 'number', None), 'Infinity')


if __name__ == '__main__':
    unittest.main()
