#!/usr/bin/env python3
"""Freeze F/C winners from the completed development queue, before formal evaluation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.queue.read_text())
    result = {"modes": {}}
    for mode in ("future", "current"):
        candidates = [
            row for row in state["runs"]
            if row.get("status") == "complete"
            and row.get("mode") == mode
            and row.get("outer_scope") == "eds"
        ]
        if len(candidates) < 2:
            raise ValueError(
                f"{mode}: fewer than two SSL families passed health checks; "
                "the preregistered top-two confirmation cannot be completed"
            )
        ranked = sorted(candidates, key=lambda row: tuple(row["best_rank"]))
        result["modes"][mode] = {
            "first": ranked[0],
            "second": ranked[1],
            "all_eds_ranking": ranked,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        mode: [
            result["modes"][mode][rank]["objective"] for rank in ("first", "second")
        ]
        for mode in ("future", "current")
    }))


if __name__ == "__main__":
    main()
