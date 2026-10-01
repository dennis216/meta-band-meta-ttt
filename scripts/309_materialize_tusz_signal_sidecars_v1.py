#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from bfa.tusz_meta_ttt.dataset import signal_sidecar_path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs/reports/tusz_meta_ttt_v1/cache"


def materialize(path: Path) -> tuple[str, str]:
    target = signal_sidecar_path(path)
    if target.is_file():
        mapped = np.load(target, mmap_mode="r", allow_pickle=False)
        if mapped.dtype == np.float32 and mapped.ndim == 2 and mapped.shape[0] == 16:
            return str(path), "existing"
        target.unlink()
    with np.load(path, allow_pickle=False) as archive:
        signal = archive["signal"]
    temporary = target.with_suffix(target.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as stream:
        np.save(stream, signal, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, target)
    mapped = np.load(target, mmap_mode="r", allow_pickle=False)
    if mapped.shape != signal.shape or mapped.dtype != signal.dtype:
        raise RuntimeError(f"sidecar verification failed: {target}")
    return str(path), "created"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--partition", choices=["train", "dev", "eval", "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    partitions = ("train", "dev", "eval") if args.partition == "all" else (args.partition,)
    paths = sorted(path for partition in partitions for path in (CACHE / partition).rglob("*.npz"))
    if args.limit is not None:
        paths = paths[: args.limit]
    counts = {"created": 0, "existing": 0}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(materialize, path): path for path in paths}
        for index, future in enumerate(as_completed(futures), 1):
            _, status = future.result()
            counts[status] += 1
            if index % 100 == 0 or index == len(paths):
                print({"completed": index, "total": len(paths), **counts}, flush=True)


if __name__ == "__main__":
    main()
