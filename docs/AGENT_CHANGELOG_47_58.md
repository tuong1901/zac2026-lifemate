# Báo Cáo Cải Tiến Agent: Hành Trình 34.73 $\rightarrow$ 37.xx $\rightarrow$ 42.xx $\rightarrow$ 47.58000

> **Cập nhật lần cuối:** 11/11/2026  
> **Mô hình Suy luận:** `Qwen/Qwen3-4B-Instruct-2507` (Greedy Decoding, `temperature=0.0`, `seed=42`)  
> **Điểm Public Leaderboard hiện tại:** **47.58000** (`4d1e35e8-b049-4032-92e8-6a564e5b2323.jsonl`)

---

## 1. Tổng Quan Các Mốc Cải Tiến

| Phiên bản | Mô hình | Điểm Public Test | Các nâng cấp cốt lõi |
| :--- | :--- | :---: | :--- |
| **Baseline (V12-V14)** | `Qwen2.5-3B-Instruct` | **34.73 – 37.xx** | ReAct loop cơ bản + BM25 trên `train.jsonl` (17 apps). |
| **V15 (Qwen3-4B Migration)** | `Qwen3-4B-Instruct-2507` | **42.xx** | Chuyển sang `Qwen3-4B-Instruct-2507` (`enable_thinking=False`), đồng bộ chat template, mở rộng ngữ cảnh và chuẩn hóa tham số cơ bản. |
| **V16 (Multi-Agent Deep Upgrade + QC)** | `Qwen3-4B-Instruct-2507` | **47.58000** | Multi-ASR Consensus + Phonetic Normalizer, Exemplars cho 11 Unseen Apps, Pre-Execution Verifier chống gọi thừa/thiếu, History Grounding & QC 634/634 mẫu. |

---

## 2. Chi Tiết Các Hạng Mục Đã Cải Tiến (Đạt 47.58 Điểm)

### 2.1. Nâng Cấp Mô Hình Lên `Qwen/Qwen3-4B-Instruct-2507` (`src/agent/llm_client.py`)
* **Tắt chế độ `<think>` rườm rà khi chạy ReAct JSON (`enable_thinking=False`)**: Giúp model tập trung sinh cấu trúc JSON `{"thought": "...", "actions": [...], "is_finished": ...}` ổn định, giảm độ trễ và tránh tràn `max_new_tokens`.
* **Hậu xử lý bóc tách `<think>...</think>` an toàn**: Đảm bảo bộ phân tích JSON (`parse_llm_json`) không bao giờ bị lỗi cú pháp nếu model sinh thẻ suy luận.
* **Cơ chế khóa đồng bộ GPU (`threading.Lock`) & Quản lý VRAM trên Server**: Tránh lỗi `CUDA out of memory` khi chạy đa luồng (`--workers 2`), hỗ trợ tự động resume từ `submission.jsonl`.

### 2.2. Chuẩn Hóa Ngữ Âm & Đa Nguồn ASR (`src/agent/asr_phonetic_data.py`, `src/agent/prompt.py`)
* **Multi-ASR Consensus Fusion**: Kết hợp đối chiếu đồng thời 3 bản gỡ băng (`phowhisper_medium`, `whisper_turbo`, `phowhisper_small`) cho cả câu lệnh hiện tại (`audio_text`) và các lượt hội thoại trong `history`.
* **Khôi phục số học, tiền tệ, giờ giấc & dấu thập phân tiếng Việt**:
  * Chuyển đổi chính xác các cụm số đọc bằng chữ hoặc bị dính chữ do ASR (ví dụ: `"phẩy"` $\rightarrow$ dấu thập phân `.`, `"bốn số chín"` $\rightarrow$ `9999`, `"rưỡi"`, `"k"`, `"triệu"`, `"lít"`, `"củ"`).
  * Sửa lỗi biến dạng tên riêng, mã cổ phiếu, tên ứng dụng và địa danh phổ biến trong tiếng Việt 3 miền (ví dụ: `"cổ phiếu của gờ"` $\rightarrow$ `G`, `"cây xăng vi na cô"` $\rightarrow$ `Vinaco`, v.v.).

### 2.3. Phủ Sóng 11 Ứng Dụng Mới Trên Public Test (`src/agent/catalog_exemplars.py`)
* Tập `train.jsonl` chỉ có 17 ứng dụng, trong khi `public_test.jsonl` xuất hiện **11 ứng dụng hoàn toàn mới**:
  1. `ecommerce` (`search`, `get_product`, `add_to_cart`, `checkout`, `my_orders`, `request_return`, `cancel_order`)
  2. `grocery` (`search`, `add_to_cart`, `checkout`, `order_status`, `cancel_order`)
  3. `parking` (`find`, `reserve`, `active_session`, `extend`, `cancel`)
  4. `public_transit` (`find_route`, `next_departures`, `buy_pass`, `my_passes`)
  5. `stocks` (`get_quote`, `Portfolio`, `buy`, `sell`, `watchlist_add`, `watchlist_remove`)
  6. `recipes` (`search`, `get`, `save_favorite`, `scale_servings`)
  7. `pantry` (`list`, `add`, `use`, `remove`, `expiring_soon`)
  8. `train_bus_tickets` (`search`, `book`, `my_tickets`, `cancel`)
  9. `event_tickets` (`search`, `book`, `my_tickets`, `cancel`)
  10. `habit_tracker` (`list_habits`, `create_habit`, `log`, `streak`, `delete_habit`)
  11. `translator` (`translate`, `detect_language`, `phrasebook_save`, `phrasebook_list`)
* Đã xây dựng bộ **Catalog Exemplars & Atomic Workflow Chunks** cho toàn bộ 11 ứng dụng mới, giúp BM25 Retriever luôn cung cấp đúng chuỗi `READ -> WRITE` mẫu chuẩn xác cho LLM.

### 2.4. Bộ Kiểm Chứng Trước Thực Thi & Bảo Vệ `state_hash` (`src/agent/react_loop.py`)
* **Chống gọi thừa (`WRITE_NO_CHANGE` / Redundant Setter)**:
  * Kiểm tra trạng thái thiết bị (`smart_home.get_status`) trước khi gọi `light_set`, `ac_set`, `fan_set`, `curtain_set`. Nếu thiết bị đã ở đúng trạng thái mục tiêu (`power`, `speed`, `oscillate`, `temperature`), tự động lược bỏ lệnh ghi dư thừa để tránh bị phạt Precision ($P$).
* **Chống gọi thiếu (`done: False` / Missing Write)**:
  * Phát hiện các trường hợp LLM dừng sớm sau bước `READ` khi yêu cầu của người dùng có mệnh đề hành động (`WRITE`) rõ ràng và điều kiện thực thi đã thỏa mãn.
* **Tự động đối chiếu ID thực tế từ `history` và `step_history` (Entity Grounding)**:
  * Khớp chính xác `contact_id`, `beneficiary_id`, `account_id`, `bill_id`, `event_id`, `alarm_id`, `rem_id`, `task_id`, `order_id`, `product_id`, `lot_id`, `recipe_id`, `habit_id`, `place_id` từ kết quả trả về của môi trường `ToolEnv`, ngăn chặn triệt để lỗi bịa ID (hallucinated IDs).

---

## 3. Mục Tiêu Giai Đoạn Tiếp Theo: Hướng Tới 70+ Điểm

1. **Phân tích chuyên sâu 3 bộ ASR (`phowhisper_medium`, `phowhisper_small`, `whisper_turbo`)**: Chọn lọc và hợp nhất thông minh (Smart ASR Selector / ROVER Fusion) để lấy văn bản sạch nhất cho từng mẫu.
2. **Dynamic Date Table & Suy luận Thời gian**: Bảng lịch động đầy đủ (Hôm nay, Ngày mai, Hôm qua, Thứ 2 $\rightarrow$ Chủ Nhật tuần này và tuần sau) gắn trực tiếp vào Prompt để loại bỏ hoàn toàn lỗi tính nhầm ngày.
3. **Compact Schema & GuardedToolEnv**: Thu gọn schema theo ứng dụng liên quan và ép buộc điều kiện tiên quyết `READ` trước `WRITE` chuẩn xác tuyệt đối (không thừa, không thiếu).
4. **Xử lý câu điều kiện phức tạp, tự hủy lệnh (Retraction) & OOS theo `installed_apps`**: Đảm bảo các ca rút lại quyết định (`"à thôi khỏi"`, `"không cần nữa"`) và ca thiếu ứng dụng cài đặt trả về đúng `[]` hoặc chỉ thực hiện vế hợp lệ.
5. **Kiểm định QC nghiêm ngặt đa vòng lặp**: Chỉ xuất submission khi Subagent QC xác nhận 100% mẫu đạt chuẩn ngữ nghĩa, đúng thứ tự tool call và khớp `state_hash`.
