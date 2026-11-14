# Trạng thái cải tiến và gỡ thay đổi Codex — 13/11/2026

## Kết luận kiểm chứng

Các cải tiến do Codex triển khai trong phiên 12–13/11 **chưa được QC phê duyệt**
và chưa chứng minh cải thiện chất lượng toàn agent. Theo yêu cầu người dùng,
đã gỡ phần code và công cụ do Codex thêm trong phiên này, khôi phục working tree
ban đầu. Không chạy test/public-test inference hoặc tạo submission mới khi gỡ.

## Các thay đổi có sẵn trước Codex được giữ lại

Đây là nội dung đang làm dở được lưu trước khi Codex chỉnh sửa; **không phải
các cải tiến đã được Codex xác nhận đạt chuẩn**:

| File | Nội dung thay đổi so với HEAD trước phiên |
|---|---|
| `src/agent/asr_phonetic_data.py` | Các điều chỉnh trong bảng chuẩn hóa/sửa bản ASR đang có. |
| `src/agent/catalog_exemplars.py` | Điều chỉnh ví dụ và nhận dạng biến thể ngôn ngữ cho các workflow. |
| `src/agent/prompt.py` | Khôi phục từ chỉ buổi giữa các bản ASR; sửa trích nội dung SMS, thời gian/tiêu đề lịch và xử lý kế hoạch công cụ. |
| `src/agent/react_loop.py` | Điều chỉnh chọn chuyến/vé, đối chiếu history, đọc trước ghi, thời lượng bài tập → lịch và xử lý người nhận email trùng tên. |
| `src/agent/rag/exemplar_retriever.py` | Bổ sung thu thập ví dụ từ history, xác định ứng dụng được gọi và xử lý ví dụ hỏi thiếu thông tin. |
| `src/rules/catalog_inspector.py` | File chưa được theo dõi trước phiên: đọc metadata/schema, loại công cụ và quan hệ tham chiếu. Được giữ nguyên. |

Các nội dung này còn cần review về hardcode, tính tổng quát và kiểm thử hồi quy.
Riêng history mining đang đọc trường `content`, trong khi history train đã quan
sát dùng `audio_text`; chưa thể tuyên bố tính năng này hoạt động đầy đủ.

## Phần Codex đã thử và nay đã gỡ

- Thay bảng sửa ASR và cơ chế nạp đa nguồn; truyền bản ASR phụ cho history.
- Viết lại phần lớn prompt/ReAct, grounding thực thể, hỏi thiếu thông tin,
  thời lượng lịch và parser số/ngày.
- Thêm ngân sách thời gian, trace quyết định và thay đổi LLM client.
- Thay đổi guardrail, bộ lọc catalog và history mining.
- Sửa tính U/call matching, cách ly nhãn và viết bộ chạy val/train theo fold.
- Thêm `eval_support.py`, `qc_gate.py`, `quality_ladder.py` và các test synthetic.
- Các báo cáo Codex cũ về baseline/kế hoạch/QC được chuyển ra ngoài repository
  cùng bản code thử nghiệm; tài liệu này thay thế trạng thái công bố trong nhánh.

## Bằng chứng và giới hạn

Trước khi gỡ, một số test synthetic đã qua, nhưng không đủ để xác nhận cải thiện
end-to-end hoặc tuân thủ đầy đủ. Lượt local ASR `clean_ladder_r2` trên val20 ghi:

| Chỉ số | Kết quả của bản Codex đã gỡ |
|---|---:|
| Điểm | 23,1667 / 100 |
| State Hash | 6 / 20 |
| U thực | 27 |
| Call bị từ chối | 0 |
| Thời gian trung bình | 37,18 giây/mẫu |

Train20 của lượt đó không hoàn tất: guard phát hiện code đổi giữa lúc chạy và
đã dừng. Không cộng các trace từng phần thành kết quả train20. Các sửa sau r2
cũng chưa có lượt đo đầy đủ. Các số trên không đánh giá bản code đã khôi phục
và không phải điểm leaderboard. Không xác nhận lại các tuyên bố 50/50 hoặc
100% trong trao đổi trước.

## Cách khôi phục và lưu trữ

- Bản gốc: `D:/TOOL_TH/ZAC_2026/logs/audit_20261112/`.
- Trước khi khôi phục, kiểm tra snapshot của cả 5 file sửa dở bằng cách đảo
  `initial_worktree.patch` trong thư mục kiểm tra riêng rồi so với HEAD.
- Sau khi khôi phục, `git diff --binary` khớp **từng byte** với patch ban đầu.
- Các file tracked khác do Codex sửa được trả về HEAD; `catalog_inspector.py`
  có sẵn được giữ lại. Không sửa nội dung submission.
- Bản Codex đã gỡ và manifest SHA-256 được lưu ngoài repository tại
  `D:/TOOL_TH/ZAC_2026/logs/codex_rollback_20261113/` để có thể đối chiếu.

Kiểm tra sau khôi phục chỉ gồm kiểm tra cú pháp, diff và scanner tĩnh; không
được diễn giải thành QC phê duyệt chất lượng. Mọi thay đổi tiếp theo cần phân
tích và QC duyệt phương án trước khi triển khai từng phần nhỏ.
