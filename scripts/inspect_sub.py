import json

sub_path = r"C:\Users\opc\Downloads\submission.jsonl"
with open(sub_path, "r", encoding="utf-8") as f:
    for idx in range(30):
        line = f.readline()
        if not line:
            break
        row = json.loads(line)
        print(f"[{idx+1}] {row['id']}: {row['tool_calls']}")

