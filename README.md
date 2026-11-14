# ZAC 2026 — LifeMate: Voice-to-Action Agent

Dự án tham gia cuộc thi **Zalo AI Challenge 2026** — Bảng **LifeMate** (*Agentic AI That Turns Your Voice Into Action*).

---

## 1. Phân chia công việc 5 thành viên (Mô hình mới: 2 người Data)

| Thành viên | Phụ trách | Thư mục code | Nhiệm vụ chính |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Nhóm trưởng (Data Architecture & Evaluation)** | `scripts/` | Quản lý pipeline, viết `local_eval.py` chuẩn công thức BTC, quản lý 5 lần submit/ngày và Docker. |
| **Thành viên 2** | **Kỹ sư Dữ liệu (Data Processing & EDA)** | `data/`, `notebooks/`| Cặp đôi cùng Nhóm trưởng: EDA sâu 1.545 mẫu, sinh Synthetic Data bằng LLM/TTS, phân tích lỗi lệch State Hash. |
| **Thành viên 3** | **Kỹ sư Âm thanh (Speech & ASR Specialist)** | `src/asr/` | Gỡ băng Faster-Whisper Turbo (<1B), xử lý tiếng Việt 3 miền + từ mượn ("book", "AC"), tối ưu tốc độ <0.4s. |
| **Thành viên 4** | **Kỹ sư Tác nhân (ReAct Agent & Function Calling)**| `src/agent/`, `src/rules/`| Vòng lặp ReAct tương tác `ToolEnv`, Pre-validate schema, xử lý `sys.ask_missing_info` và `[]`. |
| **Thành viên 5** | **Kỹ sư Huấn luyện & MLOps (Fine-tuning & Docker)**| `src/training/`, `Dockerfile`| Fine-tune Qwen 2.5 3B LoRA, tối ưu vLLM trên NVIDIA A30 (24GB), đóng gói Docker offline nộp ngày 16/11. |

---

## 2. Công thức chấm điểm chính thức của BTC (Đã cài đặt trong `local_eval.py`)

$$\text{Score} = \begin{cases} 0, & \text{nếu } \text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}} \text{ hoặc } M = 0 \\ \frac{2 \cdot P \cdot R}{P + R}, & \text{trường hợp còn lại} \end{cases}$$

Trong đó:
* $P = \frac{M}{M + U}$ (với $U$ bao gồm cả các call bị từ chối do lỗi schema).
* $R = \frac{M}{|G|}$
* Khớp tham số tự do (`x-match: fuzzy`): Yêu cầu Token-F1 > 0.6 sau khi normalize NFC và lowercase.
* **Cổng sinh tử (Gatekeeper):** Nếu $\text{hash}_{\text{sys}} \neq \text{hash}_{\text{gold}}$, mẫu bị **0 ĐIỂM NGAY LẬP TỨC**. Mọi cải tiến phải ưu tiên bảo toàn độ khớp State.

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
python scripts/run_with_asr.py --make_submission --mode hf --model <duong_dan_model_local> --output_file output/submission.jsonl
```
Kiểm tra file nộp:
```bash
python scripts/validate_submission.py output/submission.jsonl --data_root "D:/Download/train+public_test/train+public_test"
```

Xem hướng dẫn chạy benchmark, ASR, chunking và submission cho toàn đội tại [docs/TEAM_RUNBOOK.md](docs/TEAM_RUNBOOK.md) và báo cáo chi tiết các cải tiến đạt mốc **47.58 điểm** trên Public Leaderboard tại [docs/AGENT_CHANGELOG_47_58.md](docs/AGENT_CHANGELOG_47_58.md).

> Lưu ý: ASR model/pipeline chưa được bàn giao; hiện chỉ chạy và tối ưu agent với Gold Text. Không tạo submission speech-based cho đến khi ASR chạy offline trong Docker.
