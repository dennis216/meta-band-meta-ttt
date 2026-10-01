from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def signal_sidecar_path(path: Path) -> Path:
    return path.with_suffix(".signal.npy")


def load_cached_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load cache metadata and mmap the continuous signal when a sidecar exists."""
    path = Path(path)
    sidecar = signal_sidecar_path(path)
    with np.load(path, allow_pickle=False) as archive:
        arrays = {
            name: archive[name]
            for name in archive.files
            if name != "signal" or not sidecar.is_file()
        }
    if sidecar.is_file():
        arrays["signal"] = np.load(sidecar, mmap_mode="r", allow_pickle=False)
    return arrays


class CachedTUSZWindows(Dataset):
    def __init__(self, cache_paths: list[Path], *, max_open: int = 8) -> None:
        self.cache_paths = [Path(path) for path in cache_paths]
        self.max_open = max_open
        self._open: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()
        self.rows: list[tuple[int, int, str, bool]] = []
        self.event_keys: list[str | None] = []
        for file_index, path in enumerate(self.cache_paths):
            with np.load(path, allow_pickle=False) as archive:
                labels = archive["labels"]
            patient = path.parts[-4]
            event_index = -1
            previous_positive = False
            for row, label in enumerate(labels):
                positive = bool(label > 0)
                if positive and not previous_positive:
                    event_index += 1
                self.rows.append((file_index, row, patient, positive))
                self.event_keys.append(f"{file_index}:{event_index}" if positive else None)
                previous_positive = positive

    def __len__(self) -> int:
        return len(self.rows)

    def _archive(self, path: Path) -> dict[str, np.ndarray]:
        if path not in self._open:
            self._open[path] = load_cached_arrays(path)
            while len(self._open) > self.max_open:
                self._open.popitem(last=False)
        return self._open[path]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        file_index, row, patient, _ = self.rows[index]
        archive = self._archive(self.cache_paths[file_index])
        start = row * 400
        signal = archive["signal"][:, start : start + 2000].reshape(16, 10, 200)
        return {
            "signal": torch.from_numpy(np.ascontiguousarray(signal)),
            "label": torch.tensor(float(archive["labels"][row]), dtype=torch.float32),
            "patient": patient,
            "file_index": file_index,
            "row": row,
        }


class PatientClassSampler(Sampler[int]):
    """Sample patients uniformly, with a fixed seizure/background draw ratio."""

    def __init__(
        self,
        dataset: CachedTUSZWindows,
        *,
        samples: int,
        seed: int,
        positive_fraction: float = 0.25,
        offset: int = 0,
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.positive_fraction = positive_fraction
        self.offset = offset
        grouped: dict[tuple[str, bool], list[int]] = defaultdict(list)
        for index, (_, _, patient, positive) in enumerate(dataset.rows):
            grouped[(patient, positive)].append(index)
        positive_events: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for index, (_, _, patient, positive) in enumerate(dataset.rows):
            if positive:
                event_key = dataset.event_keys[index]
                if event_key is None:
                    raise RuntimeError("positive row is missing its event key")
                positive_events[patient][event_key].append(index)
        self.by_class = {
            positive: {patient: rows for (patient, label), rows in grouped.items() if label == positive}
            for positive in (False, True)
        }
        self.positive_events = {
            patient: dict(events) for patient, events in positive_events.items()
        }
        if not self.by_class[False] or not self.by_class[True]:
            raise ValueError("sampler requires both seizure and background samples")

    def __len__(self) -> int:
        return self.samples

    def __iter__(self):
        generator = np.random.default_rng(self.seed)
        patients = {label: sorted(groups) for label, groups in self.by_class.items()}
        for draw in range(self.offset + self.samples):
            positive = bool(generator.random() < self.positive_fraction)
            patient = patients[positive][generator.integers(len(patients[positive]))]
            if positive:
                events = self.positive_events[patient]
                event_keys = sorted(events)
                event_key = event_keys[generator.integers(len(event_keys))]
                rows = events[event_key]
            else:
                rows = self.by_class[False][patient]
            selected = rows[generator.integers(len(rows))]
            if draw >= self.offset:
                yield selected


class PatientUniformSampler(Sampler[int]):
    def __init__(
        self, dataset: CachedTUSZWindows, *, samples: int, seed: int
    ) -> None:
        self.samples = samples
        self.seed = seed
        self.by_patient: dict[str, list[int]] = defaultdict(list)
        for index, (_, _, patient, _) in enumerate(dataset.rows):
            self.by_patient[patient].append(index)
        if not self.by_patient:
            raise ValueError("sampler requires at least one patient")

    def __len__(self) -> int:
        return self.samples

    def __iter__(self):
        generator = np.random.default_rng(self.seed)
        patients = sorted(self.by_patient)
        for _ in range(self.samples):
            patient = patients[generator.integers(len(patients))]
            rows = self.by_patient[patient]
            yield rows[generator.integers(len(rows))]


def cache_paths_for_patients(cache_root: Path, partition: str, patients: set[str]) -> list[Path]:
    return sorted(
        path for path in (cache_root / partition).rglob("*.npz") if path.parts[-4] in patients
    )


def cache_record_key(path: Path) -> tuple[str, str, str, str]:
    return path.parts[-4], path.parts[-3], path.parts[-2], path.stem


def filter_inventory_records(
    paths: list[Path], inventory: list[dict], *, partition: str
) -> list[Path]:
    allowed = {
        (item["patient_id"], item["session_id"], item["montage"], item["record_id"])
        for item in inventory
        if item["partition"] == partition and item["exclusion"] is None
    }
    return [path for path in paths if cache_record_key(path) in allowed]
