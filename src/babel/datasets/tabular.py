"""Load the real tabular benchmarks used by Phase 3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np


@dataclass(frozen=True)
class TabularDataset:
    name: str
    features: np.ndarray
    labels: np.ndarray
    feature_names: tuple[str, ...]
    label_names: tuple[str, ...]
    feature_values: tuple[tuple[str, ...], ...] | None = None
    source_url: str = ""


CAR_FEATURE_NAMES = ("buying", "maintenance", "doors", "persons", "lug_boot", "safety")
CAR_FEATURE_VALUES = (
    ("vhigh", "high", "med", "low"),
    ("vhigh", "high", "med", "low"),
    ("2", "3", "4", "5more"),
    ("2", "4", "more"),
    ("small", "med", "big"),
    ("low", "med", "high"),
)
CAR_LABEL_NAMES = ("unacc", "acc", "good", "vgood")
CAR_URL = "https://archive.ics.uci.edu/static/public/19/car+evaluation.zip"
CONGRESSIONAL_FEATURE_NAMES = (
    "handicapped_infants", "water_project_cost_sharing", "adoption_of_budget_resolution",
    "physician_fee_freeze", "el_salvador_aid", "religious_groups_in_schools",
    "anti_satellite_test_ban", "aid_to_nicaraguan_contras", "mx_missile", "immigration",
    "synfuels_corporation_cutback", "education_spending", "superfund_right_to_sue",
    "crime", "duty_free_exports", "export_administration_act_south_africa",
)
CONGRESSIONAL_FEATURE_VALUES = tuple(("y", "n", "?") for _ in CONGRESSIONAL_FEATURE_NAMES)
CONGRESSIONAL_LABEL_NAMES = ("democrat", "republican")
CONGRESSIONAL_URL = "https://archive.ics.uci.edu/static/public/105/congressional+voting+records.zip"


def load_car_evaluation(data_dir: str | Path = "data", *, download: bool = True) -> TabularDataset:
    """Load the UCI Car Evaluation data, caching the original archive locally."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / "car-evaluation.zip"
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(f"Car Evaluation archive not found: {archive_path}")
        try:
            with urlopen(CAR_URL, timeout=30) as response:
                archive_path.write_bytes(response.read())
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

    with ZipFile(archive_path) as archive:
        csv_name = next((name for name in archive.namelist() if name.lower().endswith(".data")), None)
        if csv_name is None:
            raise ValueError(f"No .data file found in {archive_path}")
        rows = [line.decode("utf-8").strip().split(",") for line in archive.read(csv_name).splitlines() if line.strip()]

    expected_columns = len(CAR_FEATURE_NAMES) + 1
    if not rows or any(len(row) != expected_columns for row in rows):
        raise ValueError("Car Evaluation data has an unexpected row format")
    feature_lookup = [{value: index for index, value in enumerate(values)} for values in CAR_FEATURE_VALUES]
    label_lookup = {value: index for index, value in enumerate(CAR_LABEL_NAMES)}
    try:
        features = np.asarray([[feature_lookup[i][value] for i, value in enumerate(row[:-1])] for row in rows], dtype=np.float64)
        labels = np.asarray([label_lookup[row[-1]] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown value in Car Evaluation data: {exc.args[0]}") from exc
    return TabularDataset("car_evaluation", features, labels, CAR_FEATURE_NAMES, CAR_LABEL_NAMES, CAR_FEATURE_VALUES, CAR_URL)


def load_congressional_voting_records(data_dir: str | Path = "data", *, download: bool = True) -> TabularDataset:
    """Load UCI's 1984 House voting records and retain unknown votes as a category."""
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    archive_path = data_path / "congressional-voting-records.zip"
    if not archive_path.exists():
        if not download:
            raise FileNotFoundError(f"Congressional Voting Records archive not found: {archive_path}")
        try:
            with urlopen(CONGRESSIONAL_URL, timeout=30) as response:
                archive_path.write_bytes(response.read())
        except Exception:
            archive_path.unlink(missing_ok=True)
            raise

    with ZipFile(archive_path) as archive:
        data_name = next((name for name in archive.namelist() if name.lower().endswith(".data")), None)
        if data_name is None:
            raise ValueError(f"No .data file found in {archive_path}")
        rows = [line.decode("utf-8").strip().split(",") for line in archive.read(data_name).splitlines() if line.strip()]

    expected_columns = len(CONGRESSIONAL_FEATURE_NAMES) + 1
    if not rows or any(len(row) != expected_columns for row in rows):
        raise ValueError("Congressional Voting Records data has an unexpected row format")
    feature_lookup = [{value: index for index, value in enumerate(values)} for values in CONGRESSIONAL_FEATURE_VALUES]
    label_lookup = {value: index for index, value in enumerate(CONGRESSIONAL_LABEL_NAMES)}
    try:
        labels = np.asarray([label_lookup[row[0].strip().lower()] for row in rows], dtype=np.int64)
        features = np.asarray([
            [feature_lookup[index][value.strip().lower()] for index, value in enumerate(row[1:])]
            for row in rows
        ], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"Unknown value in Congressional Voting Records data: {exc.args[0]}") from exc
    return TabularDataset(
        "congressional_voting_records", features, labels, CONGRESSIONAL_FEATURE_NAMES,
        CONGRESSIONAL_LABEL_NAMES, CONGRESSIONAL_FEATURE_VALUES, CONGRESSIONAL_URL,
    )


def load_tabular_dataset(name: str, data_dir: str | Path = "data", *, download: bool = True) -> TabularDataset:
    """Load a supported benchmark by its stable short name."""
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {"congressional", "congressional_voting_records", "congressional_vote"}:
        return load_congressional_voting_records(data_dir, download=download)
    if normalized in {"car", "car_evaluation"}:
        return load_car_evaluation(data_dir, download=download)
    raise ValueError(f"Unknown tabular dataset {name!r}; expected 'congressional_voting_records' or 'car_evaluation'")
