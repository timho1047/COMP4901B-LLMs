#!/usr/bin/env python
"""
Prepare deterministic samples for the SFT loss-mask homework exercises.

The script loads conversations from `deita_6k.json`, normalizes them with the
same helper logic used by the training pipeline, and then emits two datasets:

* single_turn: system prefix (if any) + the first user/assistant exchange
* multi_turn: the full normalized conversation (respecting max_rounds)

Both splits use the same five source conversations so that students can debug
exercise implementations consistently across the assignments.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from utils import is_single_turn_conversation, normalize_conversation

DEFAULT_DATA_PATH = Path("smol-smoltalk-6k.json")
DEFAULT_OUTPUT = Path("exercise_samples.json")
SAMPLES_PER_TASK = 5
RANDOM_SEED = 42


def _load_records(path: Path) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")
    records: List[Dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records were loaded from {path}")
    return records


def _first_user_assistant_turn(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    system_prefix: List[Dict[str, str]] = []
    idx = 0
    while idx < len(messages) and messages[idx]["role"] == "system":
        system_prefix.append(messages[idx])
        idx += 1

    if idx + 1 >= len(messages):
        raise ValueError("Conversation does not contain a full user/assistant turn.")

    user_msg = messages[idx]
    assistant_msg = messages[idx + 1]
    if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
        raise ValueError("First turn is not a user/assistant exchange.")

    subset = system_prefix + [user_msg, assistant_msg]
    if not is_single_turn_conversation(subset):
        raise ValueError("Constructed single-turn conversation failed validation.")
    return subset


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare exercise samples.")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the JSONL dataset (default: deita_6k.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Destination for the generated sample file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=SAMPLES_PER_TASK,
        help="Number of conversations to keep for each exercise.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed used when shuffling conversation order.",
    )
    args = parser.parse_args()

    records = _load_records(args.data_path)
    rng = random.Random(args.seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    single_turn_samples: List[Dict] = []
    multi_turn_samples: List[Dict] = []

    for idx in indices:
        conversation = records[idx].get("conversations")
        if conversation is None:
            continue
        normalized = normalize_conversation(conversation, max_rounds=0)
        if not normalized:
            continue
        try:
            single_turn = _first_user_assistant_turn(normalized)
        except ValueError:
            continue

        entry_id = records[idx].get("id", idx)
        multi_turn_samples.append({"id": entry_id, "messages": normalized})
        single_turn_samples.append({"id": entry_id, "messages": single_turn})

        if len(single_turn_samples) >= args.samples:
            break

    if len(single_turn_samples) < args.samples:
        raise RuntimeError(
            f"Only collected {len(single_turn_samples)} usable conversations; "
            f"need {args.samples}. Try adjusting --max-rounds or the random seed."
        )

    payload = {
        "meta": {
            "data_path": str(args.data_path),
            "output_path": str(args.output),
            "seed": args.seed,
            "samples_per_task": args.samples,
        },
        "single_turn": single_turn_samples,
        "multi_turn": multi_turn_samples,
    }

    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(
        f"Wrote {args.samples} samples for single-turn and multi-turn exercises to {args.output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
