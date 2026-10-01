#!/usr/bin/env python3
"""Fail closed unless every scheduled v2 stage and final evidence artifact is complete."""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/reports/tusz_meta_ttt_v2"


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.is_file() else None


def check(name: str, passed: bool, evidence: str, checks: list[dict]) -> None:
    checks.append({"name": name, "passed": bool(passed), "evidence": evidence})


def main() -> None:
    checks = []
    audit_path = OUT / "audits/assets_v2.json"
    audit = read(audit_path)
    check(
        "complete asset audit",
        bool(audit and audit.get("passed") and audit.get("complete_cache_scan")),
        str(audit_path),
        checks,
    )
    stage_b_path = OUT / "queues/stage_b_v2.json"
    stage_b = read(stage_b_path)
    check(
        "nine SSL candidates completed",
        bool(stage_b and len(stage_b.get("runs", [])) == 9
             and all(row.get("status") == "complete" for row in stage_b["runs"])),
        str(stage_b_path),
        checks,
    )
    ssl_selection_path = OUT / "selection/ssl_difficulty_seed3407.json"
    ssl_selection = read(ssl_selection_path)
    check(
        "SSL health selection recorded",
        bool(ssl_selection and sum(value is not None for value in ssl_selection["selected"].values()) >= 2),
        str(ssl_selection_path),
        checks,
    )
    benchmark_path = OUT / "benchmarks/parallel_scaling.json"
    benchmark = read(benchmark_path)
    benchmark_levels = {
        row.get("parallel_conditions") for row in benchmark.get("results", [])
    } if benchmark else set()
    check(
        "parallel throughput scaling measured and fixed",
        bool(
            benchmark
            and benchmark_levels >= {1, 2, 3, 4, 5, 6}
            and benchmark.get("selected_parallel_conditions") == 4
        ),
        str(benchmark_path),
        checks,
    )
    development_path = OUT / "queues/development_v2.json"
    development = read(development_path)
    check(
        "development Meta matrix completed",
        bool(development and development.get("completed_utc")
             and all(row.get("status") in {"complete", "health_check_failed", "inner_health_check_failed"}
                     for row in development.get("runs", []))),
        str(development_path),
        checks,
    )
    meta_selection_path = OUT / "selection/meta_development_v2.json"
    meta_selection = read(meta_selection_path)
    check(
        "F and C winners frozen",
        bool(meta_selection and all(
            rank in meta_selection.get("modes", {}).get(mode, {})
            for mode in ("future", "current") for rank in ("first", "second")
        )),
        str(meta_selection_path),
        checks,
    )
    mechanisms_path = OUT / "queues/mechanisms_v2.json"
    mechanisms = read(mechanisms_path)
    check(
        "six mechanism controls completed",
        bool(mechanisms and len(mechanisms.get("runs", [])) == 6
             and all(row.get("status") == "complete" for row in mechanisms["runs"])),
        str(mechanisms_path),
        checks,
    )
    formal_path = OUT / "queues/formal_v2.json"
    formal = read(formal_path)
    check(
        "three-seed formal stage completed",
        bool(formal and formal.get("status") == "complete"
             and sorted(formal.get("completed_seeds", [])) == [17, 42, 3407]),
        str(formal_path),
        checks,
    )
    bootstrap_paths = list((OUT / "statistics/bootstrap").glob("*.json"))
    bootstrap_valid = bool(bootstrap_paths) and all(
        read(path).get("replicates") == 10_000 and read(path).get("seeds") == 3
        for path in bootstrap_paths
    )
    check(
        "paired patient bootstrap completed",
        bootstrap_valid,
        str(OUT / "statistics/bootstrap"),
        checks,
    )
    holm_path = OUT / "statistics/holm_secondary.json"
    check("Holm correction completed", holm_path.is_file(), str(holm_path), checks)
    report_paths = [
        OUT / "reports/event_results.csv",
        OUT / "reports/training_manifest.json",
        OUT / "reports/mechanisms.json",
        OUT / "reports/main_results.md",
    ]
    check(
        "final reports compiled",
        all(path.is_file() for path in report_paths),
        ", ".join(map(str, report_paths)),
        checks,
    )
    result = {
        "created_utc": datetime.now(UTC).isoformat(),
        "complete": all(row["passed"] for row in checks),
        "checks": checks,
    }
    destination = OUT / "reports/completion_audit.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"complete": result["complete"], "evidence": str(destination)}))
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
