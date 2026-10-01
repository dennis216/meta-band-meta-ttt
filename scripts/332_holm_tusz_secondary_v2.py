#!/usr/bin/env python3
"""Apply Holm correction to preregistered secondary paired comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparisons", type=Path, nargs="+", required=True)
    parser.add_argument("--metric", default="fa_per_hour_ratio")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.comparisons:
        result = json.loads(path.read_text())
        value = result["bootstrap"][args.metric]["two_sided_bootstrap_p"]
        if value is None:
            continue
        rows.append({"comparison": path.stem, "path": str(path.resolve()), "p": float(value)})
    ordered = sorted(enumerate(rows), key=lambda item: item[1]["p"])
    running = 0.0
    total = len(rows)
    for rank, (original_index, row) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * row["p"])
        running = max(running, adjusted)
        rows[original_index]["holm_adjusted_p"] = running
    result = {"metric": args.metric, "comparisons": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
