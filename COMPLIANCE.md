# BẢNG NỘI QUY & CHECKLIST CHỐNG PHẠM QUY (ZAC 2026)

Tất cả 5 thành viên trong nhóm và mọi AI Agent hỗ trợ lập trình (Cursor / Copilot / ChatGPT) bắt buộc phải tuân thủ nghiêm ngặt các quy định sau. Vi phạm bất kỳ điều nào dưới đây sẽ dẫn đến việc **BỊ BAN VÀ HỦY TOÀN BỘ KẾT QUẢ THI**.

---

### ❌ 5 ĐIỀU CẤM TUYỆT ĐỐI (ZERO TOLERANCE)

1. **CẤM ĐỌC FILE `state.json` VÀ THƯ MỤC `*_environment/`:**
   * Mọi thao tác đọc và ghi dữ liệu app chỉ được phép đi qua `ToolEnv` của BTC (`env.call()`).
   * Không được dùng `open()`, `json.load()` hay bất kỳ hàm nào để đọc trộm dữ liệu người dùng trong `*_environment/`.
2. **CẤM DÙNG MÔ HÌNH QUÁ 5B THAM SỐ:**
   * Mô hình LLM bắt buộc phải **< 5 tỷ tham số** (`Qwen2.5-3B-Instruct` là lựa chọn tối ưu nhất).
   * Cấm tuyệt đối dùng model 7B, 14B, 32B hay 70B.
3. **CẤM TOÀN HỆ THỐNG VƯỢT QUÁ 6B THAM SỐ:**
   * ASR (Whisper Turbo ~809M) + LLM (Qwen 3B ~3.09B) = ~3.9B params (An toàn tuyệt đối).
4. **CẤM GỌI API ONLINE KHI SUY LUẬN (INFERENCE):**
   * Vòng chung kết Docker chạy ngắt mạng (`--network none`). Không được `import requests`, gọi OpenAI API, Anthropic hay Gemini API trong code inference.
5. **CẤM GÁN NHÃN THỦ CÔNG LÊN TẬP PUBLIC_TEST / PRIVATE_TEST:**
   * Không được mở file test ra xem bằng mắt rồi tự gắn label hoặc dùng mô hình ngoài gán nhãn để train.

---

###  5 ĐIỀU BẮT BUỘC (MANDATORY)

1. **Tính tái lập (Reproducibility):** Luôn cố định random seed = 42 (`random`, `numpy`, `torch`).
2. **Deterministic Decoding:** Luôn đặt `temperature = 0.0` (Greedy Decoding) cho cả Whisper và LLM.
3. **Một ToolEnv duy nhất:** Mỗi mẫu chỉ tạo đúng 1 phiên `ToolEnv(split, item["id"])`. Mọi cuộc gọi đều được ghi lại vào file nộp.
4. **Thời gian chạy:** Mỗi mẫu phải hoàn thành dưới **30 giây** (mục tiêu nhóm là 1.5 - 2s trên GPU NVIDIA A30).
5. **Kiểm tra Compliance trước khi nộp:** Luôn chạy lệnh kiểm tra tự động trước khi commit hoặc nộp bài:
   ```bash
   python scripts/check_compliance.py
   ```
