# Hướng Dẫn Nhiệm Vụ & Bàn Giao Kỹ Thuật Cho Codex (Codex Handover Brief)

> **Dự án:** ZAC 2026 LifeMate — Trợ lý ảo AI thông minh đa tác vụ (Voice $\rightarrow$ Tool Calling $\rightarrow$ State Mutation).  
> **Mục tiêu:** Nâng điểm thi từ **34.73** (hiện tại) lên **55–65+ điểm**.  
> **Người nhận tài liệu:** Codex (AI Coding Agent / Lập trình viên tiếp quản).

---

## 1. Bản Chất Cuộc Thi & Bối Cảnh Dự Án

### 1.1 Bài toán ZAC 2026 LifeMate là gì?
Người dùng nói một câu khẩu ngữ tiếng Việt $\rightarrow$ Hệ thống gỡ băng (ASR) $\rightarrow$ Mô hình ngôn ngữ (LLM Agent) suy luận và gọi một chuỗi công cụ (Tool Calls) $\rightarrow$ Các công cụ này thay đổi trạng thái trong môi trường sandbox (`state.json`) $\rightarrow$ Ban tổ chức (BTC) chấm điểm:
1. **Cổng sinh tử State Hash Gate:** Nếu mã băm trạng thái cuối cùng (`state_hash`) khớp 100% với đáp án thì mới được chấm tiếp. **Nếu sai State Hash $\rightarrow$ Toàn bộ sample đó nhận 0.00 điểm.**
2. **F1-Score Tool Calls:** Sau khi qua cổng State Hash, tính F1 trên chuỗi công cụ đã gọi.

### 1.2 Thực trạng của nhóm ta:
* **Điểm hiện tại:** **34.73 / 100** (Chạy bằng Zero-shot Qwen3-4B qua API Colab + BM25 RAG).
* **Điểm của các đối thủ top đầu:** **48.56 – 52.68+** (Họ dùng PhoWhisper LoRA + SFT LoRA Qwen3-4B 3 epochs + Synthetic data cho 11 app mới).
* **Nguyên nhân đội ta bị kẹt ở 34.73:**
  1. Chưa từng SFT (Zero-shot có trần vật lý chỉ ~35 điểm).
  2. Public test có 11 ứng dụng hoàn toàn mới (`ecommerce`, `parking`, `stocks`, `recipes`...) chiếm 65% dữ liệu thi, nhưng nhóm chưa có dữ liệu cho các app này.
  3. PhoWhisper bị nhồi prompt dài dẫn đến ảo giác ("bẩn"), và bộ regex hậu xử lý bị overfit nặng vào tập train.

---

## 2. Những Tài Liệu Codex CẦN ĐỌC Để Hiểu Sâu Dự Án

Trước khi viết bất kỳ dòng code nào, Codex hãy đọc kỹ các tài liệu quan trọng sau trong thư mục `docs/`:

1. **`docs/ASR_DIRTY_ANALYSIS_AND_CLEANUP.md`**: Bóc tách chi tiết vì sao PhoWhisper bị "bẩn" (do prompt bleeding và aggressive regex) và cách sửa triệt để.
2. **`docs/BENCHMARK_AND_RESEARCH_REPORT_ZAIC2026.md`**: Phân tích giải pháp của đội Nguyen-Le-Tuan (Top 1 - 52.68 điểm).
3. **`docs/BENCHMARK_AND_RESEARCH_REPORT_MINGGU151623.md`**: Phân tích giải pháp của đội minggu151623 (50.88 điểm - tác giả của `synth.py` và CoT Rationale).
4. **`docs/BENCHMARK_AND_RESEARCH_REPORT_MUAHAHAHA55.md`**: Phân tích giải pháp của đội muahahaha55 (48.56 điểm - tác giả của History Mining và App Stripping).
5. **`docs/AGENT_ARCHITECTURE_REVIEW.md`**: Đánh giá toàn bộ kiến trúc ReAct Agent hiện tại do Việt Đức phát triển.

---

## 3. Bản Đồ Thư Mục & Các File Cốt Lõi

* `src/asr/transcribe.py`: File gỡ băng âm thanh ASR PhoWhisper (cần làm sạch prompt).
* `src/asr/post_processor.py`: File hậu xử lý văn bản ASR (cần gỡ bỏ các regex gây overfit).
* `src/agent/react_loop.py`: Vòng lặp ReAct Agent chính thực thi suy luận và gọi tool.
* `src/agent/prompt.py`: Nơi định nghĩa System Prompt và Schema tools.
* `scripts/kaggle_train_sft.py`: Script huấn luyện LoRA SFT trên Kaggle GPU (đã viết sẵn).
* `scripts/build_sft_dataset.py`: Script trích xuất và nhân bản tập dữ liệu SFT (đã tạo 4,994 samples).
* `downloaded_minggu/`: Chứa mã nguồn tham khảo của top competitor (đặc biệt là `train+public_test_solution_synth.py`).

---

## 4. Danh Sách Các Nhiệm Vụ Codex Cần Thực Hiện (Action Items)

Codex hãy thực hiện lần lượt các nhiệm vụ theo thứ tự ưu tiên sau:

### Nhiệm Vụ 1: Làm Sạch Pipeline PhoWhisper (Loại bỏ ASR "Bẩn")
* **File cần sửa:** `src/asr/transcribe.py` và `src/asr/post_processor.py`.
* **Yêu cầu cụ thể:**
  1. Trong `transcribe.py`: Xóa bỏ việc nhồi `self.default_prompt` dài 80 từ vào `prompt_ids` (hoặc đặt `self.prompt_ids = None`).
  2. Bổ sung các tham số giải mã chuẩn chống lặp và chống ảo giác vào `generate_kwargs`:
     ```python
     generate_kwargs = {
         "language": "vi",
         "task": "transcribe",
         "do_sample": False,
         "temperature": 0.0,
         "num_beams": 5,
         "no_repeat_ngram_size": 3,
         "condition_on_previous_text": False,
     }
     ```
  3. Trong `post_processor.py`: Tắt hoặc xóa danh sách `AGGRESSIVE_PHONETIC_REPLACEMENTS` (danh sách này đang biến đổi sai lệch câu nói của người dùng). Chỉ giữ lại chuẩn hóa Unicode NFC và sửa các từ mượn hiển nhiên (`ZaloPay`, `MoMo`, `VTV3`).

---

### Nhiệm Vụ 2: Chuyển Giao Bộ Sinh Dữ Liệu `synth.py` Cho 11 App Mới
* **Mục tiêu:** Sinh ~1,100 sample cho 11 ứng dụng chưa từng có trong tập train bằng `zaic_tools.executor.Simulator`.
* **Nguồn tham khảo:** `downloaded_minggu/train+public_test_solution_synth.py`.
* **Yêu cầu cụ thể:**
  1. Tạo file `scripts/synth_unseen_apps.py`.
  2. Đọc catalog của `public_test/public_test_tools_catalog.json` để lấy schema của 11 app mới: `ecommerce`, `grocery`, `parking`, `public_transit`, `stocks`, `recipes`, `pantry`, `train_bus_tickets`, `event_tickets`, `habit_tracker`, `translator`.
  3. Dùng `Simulator` để sinh các mẫu hợp lệ và đảm bảo `state_hash` chính xác.
  4. Xuất ra file `data/synth_11_apps.jsonl`.
  5. Chạy `python scripts/build_sft_dataset.py --train data/train/train.jsonl --synth data/synth_11_apps.jsonl --out data/sft_train_full.jsonl` để gộp thành bộ dữ liệu ~6,000 samples.

---

### Nhiệm Vụ 3: Triển Khai Huấn Luyện SFT LoRA Trên Kaggle
* **File thực thi:** `scripts/kaggle_train_sft.py`.
* **Yêu cầu cụ thể:**
  1. Hướng dẫn hoặc đóng gói script thành Kaggle Notebook (Notebook Cells).
  2. Cấu hình chạy trên GPU 2x T4 (hoặc P100):
     * Model: `Qwen/Qwen3-4B-Instruct-2507` (hoặc `Qwen/Qwen2.5-3B-Instruct`).
     * LoRA Rank $r = 32$, Alpha = 64.
     * Bật 4-bit QLoRA (`bitsandbytes`).
     * Gradient accumulation = 16, Batch size = 1.
     * 3 Epochs.
  3. Kiểm tra file weights adapter đầu ra tại `out/lora_adapter/final_adapter`.

---

### Nhiệm Vụ 4: Triển Khai Dynamic Date Table Vào ReAct Prompt
* **File cần sửa:** `src/agent/prompt.py`.
* **Vấn đề:** 12% sample liên quan đến tính toán ngày giờ ("hôm qua", "tuần sau thứ 6", "3 ngày nữa"). Nếu để LLM tự nhẩm sẽ bị sai ngày.
* **Yêu cầu cụ thể:** Viết hàm sinh bảng ngày giờ động dựa trên `current_time` của từng sample:
  ```python
  def build_date_table(current_time_iso: str) -> str:
      # Tính toán ngày hôm nay, hôm qua, ngày mai, thứ 2 đến CN tuần này, tuần sau
      # Trả về bảng text để chèn thẳng vào system prompt
  ```

---

### Nhiệm Vụ 5: GuardedToolEnv (Chống Bị 0 Điểm Oan Ở Cổng State Hash)
* **File cần sửa:** `src/agent/react_loop.py`.
* **Nguyên tắc:** Các thao tác ghi (WRITE: chuyển tiền, đặt vé, hủy lịch) bắt buộc phải có ID lấy từ thao tác đọc (READ: xem danh bạ, tìm chuyến bay, xem lịch). Nếu LLM vội vàng gọi WRITE khi chưa READ $\rightarrow$ Guard layer tự động hoãn và chèn lệnh READ hợp lệ trước.

---

## 5. Các Ràng Buộc & Quy Chế Thi Tuyệt Đối Không Được Vi Phạm

1. **Giới hạn mô hình:** Mô hình ASR $< 1\text{B}$ tham số. Mô hình LLM $< 5\text{B}$ tham số (Qwen3-4B hoàn toàn hợp lệ).
2. **Không gian sandbox:** Tuyệt đối KHÔNG đọc file `state.json` trong thư mục `<split>_environment/` trong quá trình chạy inference.
3. **Môi trường offline:** Khi submit cho BTC, toàn bộ pipeline phải chạy offline hoàn toàn trên 1 GPU máy thi, không được gọi bất kỳ API bên ngoài nào.
4. **Git Branching:** Luôn kiểm tra `git status` trước khi sửa code, commit rõ ràng theo chuẩn Conventional Commits (ví dụ: `feat(sft): ...`, `fix(asr): ...`).
