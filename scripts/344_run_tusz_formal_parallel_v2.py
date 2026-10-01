#!/usr/bin/env python3
"""Run the three independent formal seeds concurrently, then aggregate evidence."""
from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"
SEEDS = (17, 42, 3407)


def call(command: list[str], log: Path) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        result = subprocess.run(
            command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--development-queue", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", choices=SEEDS, default=[3407])
    parser.add_argument("--defer-aggregate", action="store_true")
    args = parser.parse_args()
    base = [
        sys.executable,
        str(ROOT / "scripts/338_run_tusz_formal_v2.py"),
        "--selection", str(args.selection.resolve()),
        "--development-queue", str(args.development_queue.resolve()),
    ]
    if not args.execute:
        print({"parallel_seed_workers": len(args.seeds), "seeds": args.seeds})
        return
    with ThreadPoolExecutor(max_workers=len(args.seeds)) as executor:
        futures = {
            executor.submit(
                call,
                [*base, "--execute", "--seeds", str(seed), "--skip-aggregate"],
                OUT / "logs/formal" / f"seed_pipeline_{seed}.log",
            ): seed
            for seed in args.seeds
        }
        for future in as_completed(futures):
            future.result()
    if args.defer_aggregate or set(args.seeds) != set(SEEDS):
        return
    # A fast resumable pass finds all seed artifacts, writes the canonical queue,
    # and performs the cross-seed bootstrap and Holm aggregation exactly once.
    call([*base, "--execute", "--seeds", *map(str, args.seeds)], OUT / "logs/formal/formal_aggregate.log")


if __name__ == "__main__":
    main()
