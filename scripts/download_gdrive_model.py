"""
scripts/download_gdrive_model.py — Tải mô hình ASR Fine-tuned từ Google Drive về Kaggle.
Folder ID: 1fOtoBLRuAlepfC9xxneiVGjs3WIvauTI
Đích đến mặc định: /kaggle/working/models/phowhisper-medium-finetuned
"""

import os
import sys
import argparse
from pathlib import Path

def download_model(folder_id: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("      TẢI MÔ HÌNH PHOWHISPER-MEDIUM FINETUNED TỪ GOOGLE DRIVE")
    print("=" * 70)
    print(f" ID Thư mục GDrive: {folder_id}")
    print(f" Lưu trữ tại:       {output_dir.resolve()}")
    print("-" * 70)

    try:
        import gdown
    except ImportError:
        print("📦 Đang cài đặt thư viện gdown...")
        os.system(f"{sys.executable} -m pip install -q gdown")
        import gdown

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    print(f"⏳ Bắt đầu tải toàn bộ trọng số mô hình từ {url}...")
    try:
        gdown.download_folder(url=url, output=str(output_dir), quiet=False, use_cookies=False)
        print("\n✅ TẢI MÔ HÌNH THÀNH CÔNG!")
        
        # Liệt kê các file đã tải về
        downloaded_files = list(output_dir.glob("*"))
        print(f"\n📂 Danh sách files trong thư mục ({len(downloaded_files)} files):")
        for f in downloaded_files:
            size_mb = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
            kind = "FILE" if f.is_file() else "DIR "
            print(f"  - [{kind}] {f.name:30} ({size_mb:.1f} MB)")
            
    except Exception as e:
        print(f"\n❌ Lỗi khi tải bằng gdown: {e}")
        print("💡 Gợi ý: Hãy kiểm tra quyền truy cập của link Google Drive (chia sẻ công khai 'Bất kỳ ai có đường liên kết').")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Tải mô hình ASR Fine-tuned từ Google Drive")
    parser.add_argument("--folder_id", type=str, default="1fOtoBLRuAlepfC9xxneiVGjs3WIvauTI", help="Google Drive Folder ID")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/models/phowhisper-medium-finetuned", help="Thư mục lưu mô hình")
    args = parser.parse_args()

    out_path = Path(args.output_dir)
    if not out_path.is_absolute() and "KAGGLE_WORKING_DIR" in os.environ:
        out_path = Path("/kaggle/working") / args.output_dir

    download_model(args.folder_id, out_path)

if __name__ == "__main__":
    main()
