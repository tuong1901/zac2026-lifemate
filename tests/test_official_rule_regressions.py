"""Synthetic regressions derived from the organizer's API/scoring contract."""
import contextlib
import copy
import io
import json
import unittest
from unittest.mock import patch

from src.agent.execution import (ActionFeedback, GuardedToolEnv, RunBudget,
                                 current_read_signatures, successful_history_reads)
from src.agent.prompt import build_step_directive, format_exemplar_by_turns
from src.agent.rag.exemplar_retriever import BM25ExemplarRetriever
from src.agent.react_loop import ReActAgent
from src.rules.schema_guardrail import SafeSchemaGuardrail
from test_agent_regressions import Env, Catalog, Client, plan


def read_step(name, **result):
    return {'tool_called': name, 'arguments': {}, 'raw_observation': result}


class GroundingTests(unittest.TestCase):
    def ground(self, name, args, steps, text='', history=()):
        args = copy.deepcopy(args)
        ReActAgent._resolve_grounded_arguments(name, args, history, steps, text)
        return args

    def test_selected_place_is_not_replaced_by_first_result(self):
        steps = [read_step('maps.search_place', places=[{'id': 'p1', 'name': 'Alpha'}, {'id': 'p2', 'name': 'Beta'}])]
        self.assertEqual(self.ground('maps.save_place', {'place_id': 'p2'}, steps, 'Lưu Beta'), {'place_id': 'p2'})

    def test_unknown_place_resolves_by_name_and_ambiguity_replans(self):
        steps = [read_step('maps.search_place', places=[{'id': 'p1', 'name': 'Alpha'}, {'id': 'p2', 'name': 'Beta'}])]
        self.assertEqual(self.ground('maps.save_place', {'place_id': 'placeholder'}, steps, 'Lưu Beta')['place_id'], 'p2')
        with self.assertRaises(ActionFeedback):
            self.ground('maps.save_place', {'place_id': 'placeholder'}, steps, 'Lưu địa điểm')

    def test_two_searches_do_not_replace_selected_recipient(self):
        steps = [read_step('contacts.search', contacts=[{'id': 'c1', 'name': 'Lan'}]),
                 read_step('contacts.search', contacts=[{'id': 'c2', 'name': 'Minh'}])]
        self.assertEqual(self.ground('ewallet.transfer', {'contact_id': 'c1', 'amount': 125000}, steps,
                                    'Chuyển cho Lan')['contact_id'], 'c1')

    def test_name_for_another_action_does_not_replace_selected_target(self):
        steps = [read_step('contacts.search', contacts=[{'id': 'c1', 'name': 'Lan'}, {'id': 'c2', 'name': 'Minh'}])]
        self.assertEqual(self.ground('ewallet.transfer', {'contact_id': 'c2'}, steps,
                                    'Chuyển cho Lan và người thứ hai')['contact_id'], 'c2')

    def test_single_unrelated_contact_cannot_replace_unknown_recipient(self):
        steps = [read_step('contacts.search', contacts=[{'id': 'c1', 'name': 'Lan'}])]
        with self.assertRaises(ActionFeedback):
            self.ground('ewallet.transfer', {'contact_id': 'unknown'}, steps, 'Chuyển cho Minh')

    def test_wrong_source_and_failed_results_cannot_ground(self):
        steps = [read_step('demo.search', places=[{'id': 'wrong'}]),
                 read_step('maps.search_place', ok=False, places=[{'id': 'failed'}])]
        self.assertEqual(self.ground('maps.save_place', {'place_id': 'placeholder'}, steps)['place_id'], 'placeholder')

    def test_history_lookup_and_explicit_phone_are_preserved(self):
        history = [{'tool_calls': [{'name': 'contacts.search'}],
                    'tool_results': [{'contacts': [{'id': 'c1', 'name': 'Lan', 'phone': '0900000001'}]}]}]
        self.assertEqual(self.ground('ewallet.transfer', {}, [], 'Chuyển cho Lan', history)['contact_id'], 'c1')
        for tool, key in [('phone.call', 'phone_number'), ('sms.send', 'phone_number'), ('ewallet.topup_mobile', 'phone')]:
            with self.subTest(tool=tool):
                self.assertEqual(self.ground(tool, {key: '0900000002'}, [], '', history), {key: '0900000002'})

    def run_agent(self, calls, results, text):
        names = {call['name'] for turn in calls for call in turn} | {'sys.ask_missing_info'}
        class Tools(Catalog):
            def get_allowed_tools(self, installed):
                return [{'name': name} for name in sorted(names)]
        env = Env(results)
        client = Client([plan(turn, i == len(calls) - 1) for i, turn in enumerate(calls)])
        agent = ReActAgent(client, Tools())
        item = {'id': 'synthetic', 'context': {'installed_apps': {n.split('.')[0]: {} for n in names}}}
        with patch('src.agent.react_loop.build_user_prompt', return_value='prompt'), contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, item, text)
        return env.calls

    def test_transfer_does_not_turn_into_canned_qr_payment(self):
        search = {'name': 'contacts.search', 'arguments': {'query': 'Lan'}}
        transfer = {'name': 'ewallet.transfer', 'arguments': {'contact_id': 'c1', 'amount': 125000}}
        result = {'contacts': [{'id': 'c1', 'name': 'Lan'}, {'id': 'c2', 'name': 'Minh'}]}
        self.assertEqual(self.run_agent([[search], [transfer]], [result],
                         'Chuyển ví phần tôi 125 nghìn cho Lan'), [search, transfer])

    def test_qr_amount_not_overwritten_by_my_share_phrase(self):
        pay = {'name': 'ewallet.pay_qr', 'arguments': {'merchant_id': 'cafe', 'amount': 125000}}
        self.assertEqual(self.run_agent([[pay]], [], 'Thanh toán QR phần tôi 125 nghìn'), [pay])

    def test_unknown_contact_replans_without_searching_default_name(self):
        transfer = {'name': 'ewallet.transfer', 'arguments': {'contact_id': 'placeholder', 'amount': 125000}}
        search = {'name': 'contacts.search', 'arguments': {'query': 'Quang'}}
        fixed = {'name': 'ewallet.transfer', 'arguments': {'contact_id': 'cq', 'amount': 125000}}
        calls = self.run_agent([[transfer], [search], [fixed]], [{'contacts': [{'id': 'cq', 'name': 'Quang'}]}],
                               'Chuyển ví 125 nghìn cho Quang')
        self.assertEqual(calls, [search, fixed])


    def test_partial_action_then_missing_information_keeps_both_calls(self):
        create = {'name': 'notes.create', 'arguments': {'title': 'Cuộc họp'}}
        ask = {'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'reminders.create', 'params': ['time']}}
        self.assertEqual(self.run_agent([[create, ask]], [], 'Tạo ghi chú Cuộc họp rồi nhắc tôi'), [create, ask])

    def test_mixed_intent_is_not_rejected_by_one_unavailable_or_cancelled_clause(self):
        agent = ReActAgent(Client([]), Catalog())
        self.assertFalse(agent.is_oos_or_cancelled('Tìm đường tới ga và tạo ghi chú Công tác', {'notes': {}}))
        self.assertFalse(agent.is_oos_or_cancelled('Thôi khỏi đặt xe, nhưng tạo ghi chú Công tác', {'notes': {}}))
        self.assertTrue(agent.is_oos_or_cancelled('Tìm đường tới ga', {'notes': {}}))

    def test_catalog_read_kind_controls_observation_boundary(self):
        read = {'name': 'custom.lookup', 'arguments': {}}
        write = {'name': 'custom.commit', 'arguments': {'id': 'found'}}
        class Tools(Catalog):
            def get_allowed_tools(self, installed):
                return [{'name': 'custom.lookup', 'kind': 'read'}, {'name': 'custom.commit', 'kind': 'write'}]
        env = Env([{'id': 'found'}])
        client = Client([plan([read, write]), plan([write])])
        agent = ReActAgent(client, Tools())
        with patch('src.agent.react_loop.build_user_prompt', return_value='prompt'), contextlib.redirect_stdout(io.StringIO()):
            agent.run(env, {'id': 'synthetic', 'context': {'installed_apps': {'custom': {}}}}, 'Thực hiện yêu cầu')
        self.assertEqual(env.calls, [read, write])
        self.assertEqual(client.calls, 2)


class SchemaBoundaryTests(unittest.TestCase):
    def test_only_four_official_environment_methods_used(self):
        class Opaque(Env):
            @property
            def tools(self):
                raise AssertionError('ToolEnv internals are forbidden')
            def validate(self, name, args):
                return {'ok': type(args.get('count')) is int}
        env = Opaque()
        spec = {'name': 'demo.write', 'parameters': {'type': 'object',
                'properties': {'count': {'type': 'integer'}}}}
        wrapper = GuardedToolEnv(env, [spec], SafeSchemaGuardrail(), RunBudget.start(10, 2))
        wrapper.call('demo.write', {'count': '3'})
        self.assertEqual(wrapper.get_tool_calls(), [{'name': 'demo.write', 'arguments': {'count': 3}}])
        self.assertEqual(wrapper.get_state_hash(), 'synthetic-test-hash')
        with self.assertRaises(AttributeError):
            _ = wrapper.tools

    def test_valid_arguments_are_unchanged(self):
        guard = SafeSchemaGuardrail().bind([{'name': 'phone.call', 'parameters': {
            'properties': {'contact_id': {'type': 'string'}, 'phone_number': {'type': 'string'}}}}])
        args = {'contact_id': 'ct_1', 'phone_number': '0900000000', 'amount': 100000,
                'time': '2026-11-06T10:00:00Z'}
        ok, clean, _ = guard.guard_and_validate(Env(), 'phone.call', args)
        self.assertTrue(ok)
        self.assertEqual(json.dumps(clean), json.dumps(args))
        self.assertIsNot(clean, args)

    def test_open_object_pattern_and_composite_contents_are_not_erased(self):
        guard = SafeSchemaGuardrail()
        cases = [({'type': 'object'}, {'config': {'power': 'on'}}),
                 ({'type': 'object', 'patternProperties': {'^x': {'type': 'integer'}}}, {'x1': 2}),
                 ({'allOf': [{'properties': {'a': {'type': 'integer'}}}]}, {'a': 2})]
        for schema, args in cases:
            with self.subTest(schema=schema):
                self.assertEqual(guard.sanitize_arguments('demo.write', args, {'parameters': schema}), args)
        self.assertIs(type(guard._repair_value('amount', 100000, 'number', None)), int)

    def test_bindings_are_isolated_per_sample(self):
        guard = SafeSchemaGuardrail()
        first = guard.bind([{'name': 'a'}])
        second = guard.bind([{'name': 'b'}])
        self.assertEqual(set(first._tool_specs), {'a'})
        self.assertEqual(set(second._tool_specs), {'b'})
        self.assertEqual(guard._tool_specs, {})

    def test_cross_app_payment_invalidates_history_current_and_live_cache(self):
        for source in ('bank', 'banking', 'ewallet'):
            app = 'banking' if source != 'ewallet' else source
            read = f'{app}.get_balance'
            reader = {'name': read, 'arguments': {}}
            write = {'name': 'bill_payment.pay', 'arguments': {'source': source, 'bill_id': 'b1'}}
            with self.subTest(source=source):
                history = [{'tool_calls': [reader, write], 'tool_results': [{'balance': 100}, {'ok': True}]}]
                self.assertEqual(successful_history_reads(history, {read}), {})
                steps = [{'tool_called': c['name'], 'arguments': c['arguments']} for c in [reader, write]]
                self.assertEqual(current_read_signatures(steps, {read}), set())
                wrapper = GuardedToolEnv(Env(), [reader, write], SafeSchemaGuardrail(), RunBudget.start(10, 2), {read})
                wrapper.successful_signatures.add(read + ':{}')
                wrapper.call(write['name'], write['arguments'])
                self.assertEqual(wrapper.successful_signatures, set())
                self.assertIn(app, wrapper.mutated_domains)


class ReferenceTests(unittest.TestCase):
    def test_empty_example_does_not_force_empty_current_plan(self):
        directive = build_step_directive([{'tool_calls': []}], [], '', 'Tạo ghi chú')
        self.assertNotIn('"actions": []', directive)
        self.assertNotIn('YÊU CẦU NGOÀI PHẠM VI', directive)

    def test_repeated_tool_occurrences_remain_available(self):
        ex = [{'tool_calls': [{'name': 'notes.create'}, {'name': 'notes.create'}]}]
        directive = build_step_directive(ex, [{'tool_called': 'notes.create', 'arguments': {'title': 'A'}}], '')
        self.assertIn("trong bài mẫu, chỉ dùng nếu phù hợp yêu cầu hiện tại: ['notes.create']", directive)
        self.assertNotIn('TẤT CẢ CÔNG CỤ CẦN THIẾT ĐÃ CHẠY XONG', directive)

    def test_reference_does_not_supply_executable_fallback(self):
        calls = [{'name': 'notes.create', 'arguments': {'title': 'REFERENCE'}}, {'name': 'notes.list'}]
        formatted = format_exemplar_by_turns([{'tool_calls': calls}], '')
        self.assertNotIn('LƯỢT HIỆN TẠI CẦN GỌI NGAY', formatted)
        self.assertLess(formatted.index('notes.create'), formatted.index('notes.list'))

    def retriever(self, calls):
        retriever = BM25ExemplarRetriever()
        retriever.index_records([{'id': 'synthetic', 'audio_text': 'tạo ghi chú', 'tool_calls': calls}])
        return retriever

    def test_hard_app_filter_never_silently_relaxes(self):
        retriever = self.retriever([{'name': 'notes.create'}])
        self.assertEqual(retriever.retrieve('tạo ghi chú', {'alarm': {}}), [])
        self.assertEqual(len(retriever.retrieve('tạo ghi chú', {'alarm': {}}, hard_app_filter=False)), 1)

    def test_specific_tools_and_missing_info_target_are_filtered(self):
        installed = {'notes': {'tools': ['notes.list']}}
        for calls in [[{'name': 'notes.create'}], [{'name': 'sys.ask_missing_info', 'arguments': {'tool_name': 'notes.create'}}]]:
            self.assertEqual(self.retriever(calls).retrieve('tạo ghi chú', installed), [])
        self.assertEqual(len(self.retriever([{'name': 'notes.list'}]).retrieve('tạo ghi chú', installed)), 1)

    def test_empty_examples_and_explicit_empty_apps(self):
        self.assertEqual(len(self.retriever([]).retrieve('tạo ghi chú', {})), 1)
        self.assertEqual(self.retriever([{'name': 'notes.create'}]).retrieve('tạo ghi chú', {}), [])

    def test_index_rebuild_clears_derived_state(self):
        retriever = self.retriever([{'name': 'notes.create'}])
        retriever._atomic_retriever = object()
        retriever._catalog_exemplars_indexed = True
        retriever.index_records([])
        self.assertFalse(retriever.is_indexed)
        self.assertEqual(retriever.inverted_index, {})
        self.assertFalse(hasattr(retriever, '_atomic_retriever'))
        self.assertFalse(hasattr(retriever, '_catalog_exemplars_indexed'))
        self.assertEqual(retriever.retrieve('tạo ghi chú'), [])


if __name__ == '__main__':
    unittest.main()
