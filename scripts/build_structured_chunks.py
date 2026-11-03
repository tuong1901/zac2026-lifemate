"""Generate intent/workflow/argument chunks from clean training data."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.rag.structured_chunks import build_structured_chunks, load_jsonl, write_chunks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=Path, default=Path("D:/Download/train+public_test/train+public_test/train/train.jsonl"))
    parser.add_argument("--validation_file", type=Path, default=PROJECT_ROOT / "data" / "val_benchmark_200.jsonl")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "data" / "structured_chunks")
    args = parser.parse_args()

    train_items = load_jsonl(args.train_file)
    validation_ids = {item["id"] for item in load_jsonl(args.validation_file)}
    chunks = build_structured_chunks(train_items, exclude_ids=validation_ids)
    write_chunks(chunks, args.output_dir)

    workflows = chunks["workflow"]
    print(json.dumps({
        "excluded_validation_ids": len(validation_ids),
        "intent_chunks": len(chunks["intent"]),
        "workflow_chunks": len(workflows),
        "argument_chunks": len(chunks["argument"]),
        "top_workflows": Counter(
            tuple(chunk["tool_sequence"])
            for chunk in chunks["intent"]
        ).most_common(10),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
