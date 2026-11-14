"""Compound repair with explicit counterexamples; no public sample fixtures."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.asr.phrase_cleaner import PhraseCleaner, protected_values
from src.asr.post_processor import get_post_processor


def ctx(*apps):
    return {'installed_apps': {app: {} for app in apps}}


class PhraseCleanerTests(unittest.TestCase):
    def setUp(self):
        self.cleaner = PhraseCleaner()

    def test_alarm_collocation_and_design_counterexample(self):
        self.assertEqual(self.cleaner.process('Đặt báo thiết lúc bảy rưỡi', ctx('alarm')),
                         'Đặt báo thức lúc bảy rưỡi')
        self.assertEqual(self.cleaner.process('Đặt báo thiết kế lên bàn', ctx('alarm')),
                         'Đặt báo thiết kế lên bàn')

    def test_does_not_guess_domain_without_context(self):
        self.assertEqual(self.cleaner.process('Đặt báo thiết lúc 7 giờ'), 'Đặt báo thiết lúc 7 giờ')
        self.assertEqual(self.cleaner.process('Đặt báo thiết', ctx('email')), 'Đặt báo thiết')

    def test_invoice_noun_does_not_rewrite_a_lock_action(self):
        self.assertEqual(self.cleaner.process('Khoá đơn điện sắp đến hạn', ctx('bill_payment')),
                         'Hóa đơn điện sắp đến hạn')
        for text in ('Khóa đơn hàng này', 'Đừng khóa đơn điện tháng này', 'Hãy khóa đơn điện tháng này'):
            self.assertEqual(self.cleaner.process(text, ctx('bill_payment')), text)

    def test_price_alert_and_shopping_list_exceptions(self):
        self.assertEqual(self.cleaner.process('Tạo cạnh bao giá mã ABC', ctx('stocks')),
                         'Tạo cảnh báo giá mã ABC')
        self.assertEqual(self.cleaner.process('Thêm sữa vào danh sách đi chờ', ctx('shopping_list')),
                         'Thêm sữa vào danh sách đi chợ')
        text = 'Thêm vào danh sách đi trả hàng ngày mai'
        self.assertEqual(self.cleaner.process(text, ctx('shopping_list')), text)

    def test_quotes_and_proper_names_are_not_rewritten(self):
        for text in ('Ghi nguyên văn "Đặt báo thiết lúc 9 giờ"', 'Ghi “hoá đơn” vào tên',
                     'Ghi ‘đặt báo thiết’', 'Tên người nhận là Báo Thiết'):
            self.assertEqual(self.cleaner.process(text, ctx('alarm', 'bill_payment')), text)

    def test_translation_requires_full_context_not_bare_calendar_word(self):
        text = 'Lịch sang tiếng Anh giúp tôi câu này, lịch xong thì lưu vào ghi chú'
        expected = 'Dịch sang tiếng Anh giúp tôi câu này, dịch xong thì lưu vào ghi chú'
        self.assertEqual(self.cleaner.process(text, ctx('translator')), expected)
        text = 'Lịch sang tiếng Anh của tôi đổi rồi'
        self.assertEqual(self.cleaner.process(text, ctx('translator', 'calendar')), text)

    def test_draft_review_not_social_like_counterexample(self):
        self.assertEqual(self.cleaner.process('Để nháp, gửi số liếc rồi xem like rồi tự gửi', ctx('email')),
                         'Để nháp, gửi số liệu rồi xem lại rồi tự gửi')
        text = 'Xem like của bài viết này'
        self.assertEqual(self.cleaner.process(text, ctx('email')), text)

    def test_numbers_codes_dates_and_cancellation_remain_unchanged(self):
        text = 'Đặt báo thiết lúc 16:15 ngày 12/11, W12; nếu không thì hủy, hai trăm nghìn'
        result, audit = self.cleaner.clean_with_audit(text, ctx('alarm'))
        self.assertEqual(protected_values(result), protected_values(text))
        self.assertIn('W12', result)
        self.assertEqual(audit[0]['rule'], 'alarm_compound')
        text = 'Hủy đơn chứ không ghi, chuyển chứ không trừ, từ sáng mai'
        self.assertEqual(self.cleaner.process(text, ctx('alarm', 'bill_payment')), text)

    def test_bad_custom_rule_cannot_change_quantity_or_logic(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'rules.json'
            for target in ('33', 'hủy'):
                path.write_text(json.dumps([{'id': 'bad', 'pattern': '(?P<bad>73)', 'replacement': target}]))
                with self.assertRaises(ValueError):
                    PhraseCleaner(path).process('Mã 73')

    def test_idempotent_and_factory_opt_in(self):
        text = 'Mo mo, Đặt báo thiết lúc 8 giờ'
        context = ctx('alarm')
        first = self.cleaner.process(text, context)
        self.assertEqual(self.cleaner.process(first, context), first)
        self.assertIsInstance(get_post_processor('phrases'), PhraseCleaner)
        self.assertEqual(get_post_processor().process('Đặt báo thiết'), 'Đặt báo thiết')

    def test_cli_preserves_raw_input_ids_and_metadata(self):
        from scripts.clean_asr_transcripts import main
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, contexts, out = root / 'input.jsonl', root / 'context.jsonl', root / 'out'
            original = {'id': 'invented', 'audio_text': 'Đặt báo thiết lúc 7 giờ', 'other': {'keep': True}}
            source.write_text(json.dumps(original, ensure_ascii=False), encoding='utf-8')
            contexts.write_text(json.dumps({'id': 'invented', 'context': ctx('alarm')}), encoding='utf-8')
            argv = ['clean', '--input', str(source), '--context-jsonl', str(contexts), '--out-dir', str(out)]
            with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            result = json.loads((out / 'input_clean.jsonl').read_text(encoding='utf-8'))
            self.assertEqual(result['audio_text'], 'Đặt báo thức lúc 7 giờ')
            self.assertEqual(result['id'], original['id'])
            self.assertEqual(result['other'], original['other'])
            self.assertEqual(json.loads(source.read_text(encoding='utf-8')), original)
            with patch('sys.argv', argv), self.assertRaises(FileExistsError):
                main()


if __name__ == '__main__':
    unittest.main()
