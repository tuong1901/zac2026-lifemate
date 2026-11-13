from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path


def inspect_hardware(paths=()) -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform(), "cpu_count": os.cpu_count(),
              "torch": None, "gpu": [], "cuda_available": False, "ram_bytes": None,
              "paths": {str(p): (shutil.disk_usage(p)._asdict() if Path(p).exists() else None) for p in paths}}
    try:
        import torch
        result["torch"] = torch.__version__
        result["cuda_version"] = torch.version.cuda
        result["cuda_available"] = torch.cuda.is_available()
        if result["cuda_available"]:
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                result["gpu"].append({"index": i, "name": props.name, "total_vram_bytes": props.total_memory,
                                      "free_vram_bytes": torch.cuda.mem_get_info(i)[0], "bf16": torch.cuda.is_bf16_supported()})
    except ImportError as exc:
        result["torch_error"] = str(exc)
    if os.name == "nt":
        try:
            import ctypes
            class Status(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            s = Status(); s.dwLength = ctypes.sizeof(Status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)):
                result["ram_bytes"] = {"total": s.ullTotalPhys, "available": s.ullAvailPhys}
        except Exception as exc:
            result["ram_error"] = str(exc)
    elif Path("/proc/meminfo").exists():
        values = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                values[key] = int(value.strip().split()[0]) * 1024
        result["ram_bytes"] = values
    try:
        result["nvidia_smi"] = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=5).stdout[:4000]
    except (OSError, subprocess.SubprocessError):
        result["nvidia_smi"] = None
    return result
