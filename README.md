# ZAC 2026 — LifeMate: Voice-to-Action Agent

Dự án tham gia cuộc thi **Zalo AI Challenge 2026** — Bảng **LifeMate** (*Agentic AI That Turns Your Voice Into Action*).

---

## 1. Phân chia công việc 5 thành viên

| Thành viên | Phụ trách | Thư mục code | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | Team Lead & Pipeline | `scripts/` | Quản lý pipeline, viết `local_eval.py`, tạo `make_submission.py`. |
| **Thành viên 2** | Audio & ASR | `src/asr/` | Gỡ băng giọng nói bằng Faster-Whisper, Dynamic Hotwords từ danh bạ/thiết bị. |
| **Thành viên 3** | ReAct Agent Engine | `src/agent/` | Vòng lặp tương tác `ToolEnv`, Prompting, Function Calling nhiều bước. |
| **Thành viên 4** | Rules & Schema | `src/rules/` | Lọc catalog theo `installed_apps`, xử lý `sys.ask_missing_info`, chuẩn hóa từ lóng / thời gian. |
| **Thành viên 5** | MLOps & Fine-tuning | `src/training/`| Fine-tune Qwen 2.5 LoRA, cấu hình vLLM, đóng gói Docker cho vòng chung kết. |

---

## 2. Cấu trúc thư mục

```text
zac2026-lifemate/
├── .gitignore
├── requirements.txt
├── README.md
├── configs/
│   └── default.yaml
├── src/
│   ├── asr/
│   │   ├── __init__.py
│   │   └── transcribe.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── prompt.py
│   │   └── react_loop.py
│   ├── rules/
│   │   ├── __init__.py
│   │   ├── catalog_filter.py
│   │   └── normalizer.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
└── scripts/
    ├── local_eval.py
    └── make_submission.py
```

---

## 3. Cài đặt và Chạy thử

### Cài đặt môi trường
```bash
pip install -r requirements.txt
```

### Chạy tự chấm điểm nội bộ (Local Evaluation)
```bash
python scripts/local_eval.py --data_root "D:/Download/train+public_test/train+public_test" --limit 50
```

### Tạo file nộp bài (Submission)
```bash
python scripts/make_submission.py --data_root "D:/Download/train+public_test/train+public_test" --output submission.jsonl
```
Kiểm tra file nộp:
```bash
python -m zaic_tools --split public_test --root "D:/Download/train+public_test/train+public_test" --validate submission.jsonl
```
