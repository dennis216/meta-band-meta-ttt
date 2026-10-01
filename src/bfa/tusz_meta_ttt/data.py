from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import mne
import numpy as np

from bfa.preprocessing.tuh import load_tuh_canonical_raw_uv
from bfa.tusz_meta_ttt.labels import decision_interval_labels
from bfa.tusz_meta_ttt.preprocessing import CausalTUSZPreprocessor


@dataclass(frozen=True)
class TUSZRecord:
    partition: str
    patient_id: str
    session_id: str
    record_id: str
    montage: str
    relative_edf: str
    relative_annotation: str
    edf_bytes: int
    edf_sha256: str
    annotation_sha256: str
    sampling_hz: float
    duration_s: float
    channel_names: tuple[str, ...]
    seizures: tuple[tuple[float, float], ...]
    detailed_annotation_sha256: str | None = None
    seizure_types: tuple[str, ...] = ()
    detailed_consistent: bool = True
    exclusion: str | None = None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_binary_annotation(path: Path) -> tuple[float, tuple[tuple[float, float], ...]]:
    lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    duration = None
    body = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("# duration") and "=" in stripped:
            duration = float(stripped.split("=", 1)[1].split()[0])
        elif stripped and not stripped.startswith("#"):
            body.append(stripped)
    if duration is None:
        raise ValueError(f"duration header missing: {path}")
    rows = csv.DictReader(body)
    seizures = []
    for row in rows:
        if str(row.get("label", "")).strip().lower() != "seiz":
            continue
        start, end = float(row["start_time"]), float(row["stop_time"])
        # TUSZ uses a zero-length TERM seizure row as a sentinel in some files.
        if start == 0.0 and end == 0.0:
            continue
        if start < 0 or end <= start or end > duration + 1e-3:
            raise ValueError(f"annotation outside record: {path}: {(start, end, duration)}")
        seizures.append((start, end))
    return duration, tuple(seizures)


def read_detailed_annotation(path: Path) -> tuple[tuple[tuple[float, float], ...], tuple[str, ...]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8-sig", errors="replace").splitlines()]
    rows = csv.DictReader(line for line in lines if line and not line.startswith("#"))
    intervals = []
    labels = set()
    for row in rows:
        label = str(row.get("label", "")).strip().lower()
        if label in {"", "bckg", "background"}:
            continue
        start, end = float(row["start_time"]), float(row["stop_time"])
        if end > start:
            intervals.append((start, end))
            labels.add(label)
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1] + 1e-3:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged), tuple(sorted(labels))


def intervals_match(
    left: tuple[tuple[float, float], ...], right: tuple[tuple[float, float], ...], tolerance: float = 0.02
) -> bool:
    return len(left) == len(right) and all(
        abs(a_start - b_start) <= tolerance and abs(a_end - b_end) <= tolerance
        for (a_start, a_end), (b_start, b_end) in zip(left, right, strict=True)
    )


def inspect_record(dataset_root: Path, partition: str, edf: Path, *, hash_edf: bool) -> TUSZRecord:
    annotation = Path(str(edf.with_suffix(".csv")) + "_bi")
    detailed_annotation = edf.with_suffix(".csv")
    if not annotation.is_file():
        raise FileNotFoundError(annotation)
    relative = edf.relative_to(dataset_root)
    parts = relative.parts
    patient_id = parts[2]
    session_id = parts[3]
    montage = parts[4]
    duration_annotation, seizures = read_binary_annotation(annotation)
    detailed_seizures, seizure_types = read_detailed_annotation(detailed_annotation)
    raw = mne.io.read_raw_edf(edf, preload=False, verbose="ERROR")
    duration_edf = raw.n_times / float(raw.info["sfreq"])
    exclusion = None
    if abs(duration_edf - duration_annotation) > max(1.0, 2 / float(raw.info["sfreq"])):
        exclusion = "edf_annotation_duration_mismatch"
    detailed_consistent = intervals_match(seizures, detailed_seizures)
    if not detailed_consistent:
        exclusion = "binary_detailed_annotation_mismatch"
    fingerprint = file_sha256(edf) if hash_edf else "stat-v1:" + hashlib.sha256(
        f"{edf.stat().st_size}\0{edf.stat().st_mtime_ns}".encode()
    ).hexdigest()
    return TUSZRecord(
        partition=partition,
        patient_id=patient_id,
        session_id=session_id,
        record_id=edf.stem,
        montage=montage,
        relative_edf=relative.as_posix(),
        relative_annotation=annotation.relative_to(dataset_root).as_posix(),
        edf_bytes=edf.stat().st_size,
        edf_sha256=fingerprint,
        annotation_sha256=file_sha256(annotation),
        sampling_hz=float(raw.info["sfreq"]),
        duration_s=min(duration_edf, duration_annotation),
        channel_names=tuple(raw.ch_names),
        seizures=seizures,
        detailed_annotation_sha256=file_sha256(detailed_annotation),
        seizure_types=seizure_types,
        detailed_consistent=detailed_consistent,
        exclusion=exclusion,
    )


def build_inventory(dataset_root: Path, *, hash_edf: bool = False) -> list[TUSZRecord]:
    records = []
    for partition in ("train", "dev", "eval"):
        for edf in sorted((dataset_root / "edf" / partition).rglob("*.edf")):
            records.append(inspect_record(dataset_root, partition, edf, hash_edf=hash_edf))
    montage_priority = {"01_tcp_ar": 0, "03_tcp_ar_a": 1, "02_tcp_le": 2, "04_tcp_le_a": 3}
    groups: dict[tuple[str, str, str, str], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        groups[(record.partition, record.patient_id, record.session_id, record.record_id)].append(index)
    for indices in groups.values():
        if len(indices) <= 1:
            continue
        keep = min(indices, key=lambda index: montage_priority.get(records[index].montage, 99))
        for index in indices:
            if index != keep:
                records[index] = replace(records[index], exclusion="duplicate_montage")
    content_groups: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        if not record.edf_sha256.startswith("stat-v1:"):
            content_groups[record.edf_sha256].append(index)
    partition_priority = {"eval": 0, "dev": 1, "train": 2}
    for indices in content_groups.values():
        if len(indices) <= 1:
            continue
        keep = min(
            indices,
            key=lambda index: (
                partition_priority[records[index].partition],
                records[index].relative_edf,
            ),
        )
        for index in indices:
            if index != keep:
                records[index] = replace(records[index], exclusion="duplicate_edf_content")
    return records


def stratified_development_split(
    records: list[TUSZRecord], *, seed: int = 3407, development_fraction: float = 0.20
) -> dict[str, list[str]]:
    by_patient: dict[str, list[TUSZRecord]] = defaultdict(list)
    for record in records:
        if record.partition == "train" and record.exclusion is None:
            by_patient[record.patient_id].append(record)
    burdens = {patient: sum(len(record.seizures) for record in items) for patient, items in by_patient.items()}
    strata: dict[str, list[str]] = defaultdict(list)
    nonzero = np.array([value for value in burdens.values() if value > 0], dtype=float)
    boundaries = np.quantile(nonzero, [1 / 3, 2 / 3]) if len(nonzero) else np.array([0, 0])
    for patient, burden in burdens.items():
        stratum = "none" if burden == 0 else f"seizure_q{1 + int(burden > boundaries[0]) + int(burden > boundaries[1])}"
        strata[stratum].append(patient)
    rng = random.Random(seed)
    selection = set()
    for patients in strata.values():
        rng.shuffle(patients)
        count = max(1, round(len(patients) * development_fraction))
        selection.update(patients[:count])
    all_patients = sorted(by_patient)
    return {
        "development_fit": [patient for patient in all_patients if patient not in selection],
        "development_validation": [patient for patient in all_patients if patient in selection],
    }


def write_inventory(records: list[TUSZRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(record) for record in records], indent=2) + "\n")


def cache_record(record: TUSZRecord, dataset_root: Path, destination: Path) -> dict:
    if record.exclusion is not None:
        raise ValueError(f"cannot cache excluded record: {record.exclusion}")
    raw_uv, source_hz, _ = load_tuh_canonical_raw_uv(dataset_root / record.relative_edf)
    processor = CausalTUSZPreprocessor(source_hz)
    signal = processor.update(raw_uv)
    window_samples, stride_samples = 2000, 400
    window_count = max(0, 1 + (signal.shape[-1] - window_samples) // stride_samples)
    decision_end_s = 10.0 + processor.metadata.output_delay_s + np.arange(window_count) * 2.0
    labels = decision_interval_labels(decision_end_s, list(record.seizures))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            signal=signal,
            decision_end_s=decision_end_s.astype(np.float64),
            labels=labels,
            source_hz=np.float64(source_hz),
            output_delay_s=np.float64(processor.metadata.output_delay_s),
        )
    os.replace(temporary, destination)
    return {"cache": str(destination), "samples": signal.shape[-1], "windows": window_count}


def cache_is_valid(path: Path) -> bool:
    try:
        with np.load(path, allow_pickle=False) as archive:
            required = {"signal", "decision_end_s", "labels", "source_hz", "output_delay_s"}
            if set(archive.files) != required:
                return False
            signal, labels, times = archive["signal"], archive["labels"], archive["decision_end_s"]
            expected = max(0, 1 + (signal.shape[-1] - 2000) // 400)
            return bool(
                signal.ndim == 2
                and signal.shape[0] == 16
                and signal.dtype == np.float32
                and labels.shape == times.shape == (expected,)
                and np.isfinite(signal).all()
                and np.isfinite(labels).all()
                and np.isfinite(times).all()
            )
    except (OSError, ValueError, EOFError, zipfile.BadZipFile):
        return False
