# Chuẩn hóa cụm từ ASR có ngữ cảnh

Đầu ra clean là giả thuyết hậu xử lý để chạy A/B với agent, không phải gold label. Không cam kết giảm WER khi chưa có audio/reference độc lập. Không tự dùng transcript public_test làm nhãn train.

## Chạy

```powershell
python scripts/clean_asr_transcripts.py --input old.jsonl new.jsonl --context-jsonl data/public_test/public_test.jsonl --out-dir out/asr_cleaned
```

Input giữ hợp đồng `id`/`audio_text`; các trường khác và thứ tự mẫu được giữ. `--context-jsonl` chỉ lấy id/context, nhất là installed_apps; không đọc label, state hoặc environment, không gọi API/model. Chọn out-dir mới nếu đã chạy trước đó. Không ghi đè bản gốc hoặc kết quả cũ.

Mỗi input tạo bốn file:

- `*_clean.jsonl`: cùng định dạng transcript gốc, dùng để A/B agent.
- `*_audit.jsonl`: chỉ các câu thay đổi, đầy đủ raw/clean và từng edit.
- `*_changes.csv`: bảng trước/sau để xem bằng Excel.
- `*_report.json`: số lượng mẫu, quy tắc đã dùng và giới hạn kết quả.

## Phạm vi

Quy tắc ở `configs/asr_phrase_rules.json`, API ở `src/asr/phrase_cleaner.py`. Quy tắc không dùng sample ID, đáp án gold hoặc transcript model kia để lựa chọn. Những cụm phụ thuộc miền yêu cầu app tương ứng trong context, đồng thời phải khớp cụm từ/điều kiện văn bản.

Ví dụ: báo thiết → báo thức chỉ sau động từ đặt/hẹn/tạo… và có app alarm, không sửa “báo thiết kế”; cạnh bao giá → cảnh báo giá với stocks; danh sách đi chờ → danh sách đi chợ với shopping_list, không sửa “danh sách đi trả hàng”. Những câu dịch có điều kiện đầy đủ mới được sửa lịch → dịch, không thay mọi từ lịch.

Nội dung nằm trong cặp dấu ngoặc kép/nháy không được sửa bằng quy tắc cụm từ. Có kiểm tra giữ nguyên dãy chữ số, từ chỉ số và dấu phủ định/điều kiện/hủy; một quy tắc tùy biến vi phạm bị từ chối. Đây không phải bộ chứng minh bảo toàn ngữ nghĩa: điều kiện regex vẫn có thể sai ở ví dụ mới. Những quote không khép cặp hoặc văn bản lồng ghép phức tạp cần kiểm tra riêng.

Không đoán lại W12/W2, sáng nay/sáng mai, giờ giấc, hủy/ghi, chuyển/trừ hoặc tên người/địa danh. Không cố viết lại mọi câu cho trôi chảy. Câu vẫn có thể chưa rõ nghĩa sau khi sửa một cụm từ.

Cách viết hoá đơn → hóa đơn chỉ là thống nhất chính tả, không phải cải thiện hiểu ý. WER khớp từ nguyên dạng có thể thay đổi vì hai cách đặt dấu này; nếu cần giữ nguyên quy ước của reference thì bỏ rule invoice_spelling trong một bản rules tùy chỉnh rồi chạy lại ở out-dir khác.

API opt-in:

```python
from src.asr.phrase_cleaner import PhraseCleaner

cleaner = PhraseCleaner()
text, edits = cleaner.clean_with_audit(
    'Đặt báo thiết lúc 7 giờ',
    {'installed_apps': {'alarm': {}}},
)
```

`get_post_processor('phrases')` trả cleaner này; cần truyền context vào process để sử dụng quy tắc theo app. Mặc định lexicon và prompt cleaner cũ không bị bật chế độ phrases tự động. Sau khi đóng gói cần mang kèm configs/asr_phrase_rules.json hoặc truyền rules_path tường minh.

## Kết quả kiểm tra ngày 12/11/2026

Hai bản public_test đều giữ đủ 634 mẫu. Bản mới thay đổi 11 câu, trong đó 5 câu chỉ thay cách viết hóa đơn; bản Drive thay đổi 12 câu, trong đó 4 câu chỉ thay cách viết này. Còn lại là sửa cụm theo ngữ cảnh, chưa xác minh bằng audio.

Kiểm tra không sửa nhầm trên 5.036 lượt gold text train/history: 47 lượt thay đổi, tất cả chỉ là cách viết hoá đơn/hóa đơn (60 occurrences). Các rule sửa cụm khác không kích hoạt. Đây không phải phép đo precision trên audio mới và không thay thế holdout độc lập.

Unit tests gồm cả phản ví dụ khóa đơn hàng, báo thiết kế, literal có quote, mã/số/thời gian/phủ định, ngữ cảnh app sai, idempotence, quy tắc tùy chỉnh nguy hiểm và CLI không ghi đè raw/metadata.
