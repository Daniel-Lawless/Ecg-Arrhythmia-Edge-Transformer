import json
from pathlib import Path
from typing import Any

import numpy as np

def load_dataset(
    index_path: Path = Path("data/processed"),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:

    # Specify paths
    X_set_path = index_path / "X.npy"
    y_set_path = index_path / "y.npy"
    patient_ids_path = index_path / "patient_ids.npy"
    metadata_path = index_path / "record_segments.json"

    # Check if the files exist
    if not X_set_path.exists():
        raise FileNotFoundError(f"No file at {X_set_path}", "No X data has been saved")

    if not y_set_path.exists():
        raise FileNotFoundError(f"No file at {y_set_path}", "No y data has been saved")
    if not patient_ids_path.exists():
        raise FileNotFoundError(
            f"No file at {patient_ids_path}", "No patient_ids have been saved"
        )
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No file at {metadata_path}", "No record metadata was found"
        )
    # Load X and y data
    X = np.load(X_set_path)
    y = np.load(y_set_path)
    patient_ids = np.load(patient_ids_path)

    # Load record metadata.
    with metadata_path.open("r", encoding="utf8") as file:
        record_metadata = json.load(file)

    return X, y, patient_ids, record_metadata


def validate_dataset(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    record_metadata: list[dict[str, Any]],
) -> None:

    # Checks if each window has a corresponding label.
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y counts row counts do not match. {X.shape[0]} != {y.shape[0]} "
        )

    if patient_ids.shape[0] != X.shape[0]:
        raise ValueError(
            "Each beat must have a patient_ids"
            f"Found {patient_ids.shape[0]} patient ids and {X.shape[0]} beats"
        )

    # First records starting position
    expected_start_index = 0

    # For each record
    for record in record_metadata:
        # Grab its start/end index and the number of windows.
        start_index = record["start_index"]
        end_index = record["end_index"]
        expected_patient_id = record["patient_id"]
        num_beats = record["num_beats"]

        # Window must begin at either the start (for the first record) or
        # at the end of the previous records window.
        if start_index != expected_start_index:
            raise ValueError(
                f"Record {record['record_id']} starts at {start_index}, "
                f"but expected {expected_start_index}"
            )

        # Window must be as large as the end of the window - the start of the window
        if end_index - start_index != num_beats:
            raise ValueError(f"Record {record['record_id']} has inconsistent num_beats")

        # Each beat must have a patient_id
        if not np.all(patient_ids[start_index:end_index] == expected_patient_id):
            raise ValueError(
                f"Each beat in record {record['record_id']}"
                f"must have patient_id {record['patient_id']}"
            )

        # Update expected_start_index to the start of the next records start.
        expected_start_index = end_index

    if record_metadata and record_metadata[-1]["end_index"] != X.shape[0]:
        raise ValueError("Final metadata end_index does not match number of rows in X")