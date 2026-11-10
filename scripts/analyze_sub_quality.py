import json
import re

sub_path = r"C:\Users\opc\Downloads\submission.jsonl"

stats = {
    "total": 0,
    "empty": 0,
    "has_ask_missing": 0,
    "has_placeholder_id": 0,
    "has_duplicate_action": 0,
    "read_only": 0,
    "healthy_actions": 0,
}

PLACEHOLDER_REGEX = re.compile(r"(nhập_|tên_người|user_id|\*|sếp_dũng|lan)", re.IGNORECASE)

READ_PREFIXES = ("get", "list", "search", "read", "track")

with open(sub_path, "r", encoding="utf-8") as f:
    for line in f:
        stats["total"] += 1
        row = json.loads(line)
        calls = row.get("tool_calls", [])
        if not calls:
            stats["empty"] += 1
            continue
        
        call_names = [c["name"] for c in calls]
        has_ask = any(n == "sys.ask_missing_info" for n in call_names)
        if has_ask:
            stats["has_ask_missing"] += 1

        args_str = json.dumps([c.get("arguments", {}) for c in calls], ensure_ascii=False)
        if PLACEHOLDER_REGEX.search(args_str):
            stats["has_placeholder_id"] += 1

        # Check duplicate write actions
        write_calls = [n for n in call_names if not any(n.split(".")[-1].startswith(p) for p in READ_PREFIXES) and n != "sys.ask_missing_info"]
        if len(write_calls) != len(set(write_calls)) and len(write_calls) > 1:
            stats["has_duplicate_action"] += 1

        if not write_calls and not has_ask:
            stats["read_only"] += 1
        elif write_calls and not has_ask and not PLACEHOLDER_REGEX.search(args_str):
            stats["healthy_actions"] += 1

print("THỐNG KÊ PHÂN TÍCH CHẤT LƯỢNG SUBMISSION:")
for k, v in stats.items():
    print(f"  - {k}: {v} ({v/stats['total']*100:.1f}%)")

