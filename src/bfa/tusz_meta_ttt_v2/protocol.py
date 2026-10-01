from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from torch.utils.data import Sampler


def assert_train_cache_paths(paths, train_cache_root: Path) -> None:
    """Reject any Dev/Eval record before labels reach a training objective."""
    root = Path(train_cache_root).resolve()
    for path in paths:
        if not Path(path).resolve().is_relative_to(root):
            raise ValueError(f"training record is outside Train cache: {path}")


def validate_evaluation_request(partition: str, calibrate: bool, thresholds: Path | None) -> None:
    if calibrate and thresholds is not None:
        raise ValueError("--calibrate and --thresholds are mutually exclusive")
    if partition == "eval":
        if calibrate or thresholds is None:
            raise ValueError("Eval requires fixed Dev thresholds and cannot recalibrate")
        if Path(thresholds).parent.name != "dev":
            raise ValueError("Eval thresholds must be a Dev summary")


def stable_transform_seed(base_seed: int, *parts: object) -> int:
    payload = "\x1f".join((str(base_seed), *(str(part) for part in parts))).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


@dataclass(frozen=True)
class Chunk:
    index: int
    rows: tuple[int, ...]
    start_s: float
    end_s: float


def chunk_rows(decision_end_s: np.ndarray, *, chunk_seconds: float = 30.0) -> list[Chunk]:
    """Group decisions by real time; a boundary decision belongs to the preceding chunk."""
    times = np.asarray(decision_end_s, dtype=np.float64)
    if times.ndim != 1 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("decision times must be finite and strictly increasing")
    if chunk_seconds <= 0:
        raise ValueError("chunk_seconds must be positive")
    groups: dict[int, list[int]] = defaultdict(list)
    for row, time_s in enumerate(times):
        index = max(0, int(np.ceil((time_s - 1e-9) / chunk_seconds)) - 1)
        groups[index].append(row)
    return [
        Chunk(index, tuple(rows), index * chunk_seconds, (index + 1) * chunk_seconds)
        for index, rows in sorted(groups.items())
    ]


def availability_times(decision_end_s: np.ndarray, mode: str) -> np.ndarray:
    """Return nominal F times or the observed end of each retrospective C chunk."""
    times = np.asarray(decision_end_s, dtype=np.float64)
    if mode == "future":
        return times.copy()
    if mode != "current":
        raise ValueError("mode must be future or current")
    result = times.copy()
    for chunk in chunk_rows(times):
        result[list(chunk.rows)] = times[chunk.rows[-1]]
    return result


def class_patient_record_weights(
    labels_by_record: dict[Path, np.ndarray],
) -> dict[Path, np.ndarray]:
    """Give seizure/background each mass 0.5, then balance patient, record, and window."""
    records_by_patient_class: dict[tuple[str, bool], list[Path]] = defaultdict(list)
    for path, labels in labels_by_record.items():
        patient = path.parts[-4]
        values = np.asarray(labels)
        for positive in (False, True):
            if np.any((values > 0) == positive):
                records_by_patient_class[(patient, positive)].append(path)
    patients_by_class = {
        positive: sorted({patient for patient, label in records_by_patient_class if label == positive})
        for positive in (False, True)
    }
    if not all(patients_by_class.values()):
        raise ValueError("both classes must be present")
    result = {path: np.zeros(len(labels), dtype=np.float64) for path, labels in labels_by_record.items()}
    for positive in (False, True):
        patients = patients_by_class[positive]
        for patient in patients:
            records = records_by_patient_class[(patient, positive)]
            for path in records:
                mask = (np.asarray(labels_by_record[path]) > 0) == positive
                result[path][mask] = 0.5 / (len(patients) * len(records) * int(mask.sum()))
    return result


def weight_audit(weights: dict[Path, np.ndarray], labels: dict[Path, np.ndarray]) -> dict[str, float]:
    return {
        "background": float(sum(value[np.asarray(labels[path]) == 0].sum() for path, value in weights.items())),
        "seizure": float(sum(value[np.asarray(labels[path]) > 0].sum() for path, value in weights.items())),
        "total": float(sum(value.sum() for value in weights.values())),
    }


class RecordLocalBatchSampler(Sampler[list[int]]):
    """Exhaust records in random order while keeping every I/O batch record-local."""

    def __init__(self, dataset, *, batch_size: int, samples: int, seed: int) -> None:
        if not 0 < samples <= len(dataset):
            raise ValueError("samples must be in (0, len(dataset)]")
        self.batch_size = batch_size
        self.samples = samples
        self.seed = seed
        self.by_file: dict[int, list[int]] = defaultdict(list)
        for dataset_index, (file_index, *_rest) in enumerate(dataset.rows):
            self.by_file[file_index].append(dataset_index)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        files = np.asarray(sorted(self.by_file))
        rng.shuffle(files)
        emitted = 0
        for file_index in files:
            indexes = np.asarray(self.by_file[int(file_index)])
            for start in range(0, len(indexes), self.batch_size):
                remaining = self.samples - emitted
                if remaining <= 0:
                    return
                batch = indexes[start : start + self.batch_size].tolist()[:remaining]
                emitted += len(batch)
                yield batch

    def __len__(self) -> int:
        return sum(
            (len(indexes) + self.batch_size - 1) // self.batch_size
            for indexes in self.by_file.values()
        )
