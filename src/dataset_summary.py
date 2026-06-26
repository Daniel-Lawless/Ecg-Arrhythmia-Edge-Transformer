import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

def load_dataset(
        index_path: Path = Path("data/processed")
    ) -> tuple[np.ndarray, np.ndarray, list]:

    # Specify paths
    X_set_path = index_path / "X.npy"
    y_set_path = index_path / "y.npy"
    metadata_path = index_path / "record_segments.json"

    # Check if the files exist
    if not X_set_path.exists():
        raise FileNotFoundError("No X data has been saved")
    
    if not y_set_path.exists():
        raise FileNotFoundError("No y data has been saved")
    
    if not metadata_path.exists():
        raise FileNotFoundError("No record metadata was found")

    # Load X and y data
    X = np.load(X_set_path)
    y = np.load(y_set_path)

    # Load record metadata.
    with metadata_path.open("r", encoding="utf8") as file:
        record_metadata = json.load(file)
    
    return X, y, record_metadata

def validate_dataset(
        X: np.ndarray,
        y: np.ndarray,
        record_metadata: list[dict[str, Any]]
) -> None:
    
    # Checks if each window has a corresponding label.
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y counts row counts do not match. {X.shape[0]} != {y.shape[0]} ")
    
    # First records starting position
    expected_start_index = 0

    # For each record
    for record in record_metadata:

        # Grab its start/end index and the number of windows.
        start_index = record["start_index"]
        end_index = record["end_index"]
        num_beats = record["num_beats"]
 
        if start_index != expected_start_index:
            raise ValueError(
                f"Record {record['record_id']} starts at {start_index}, "
                f"but expected {expected_start_index}"
            )
        
        if end_index - start_index != num_beats:
            raise ValueError(f"Record {record['record_id']} has inconsistent num_beats")
        
        # Update expected_start_index to the start of the next records start.
        expected_start_index = end_index
    
    if record_metadata and record_metadata[-1]["end_index"] != X.shape[0]:
        raise ValueError(
            "Final metadata end_index does not match number of rows in X"
        )

def print_dataset_summary(
        X: np.ndarray,
        y: np.ndarray,
        record_metadata: list[dict[str, Any]]
) -> None:
    
    # Counts how often each annotation appears.
    label_counts = Counter(y.tolist())

    # Counts how often each lead appears
    lead_counts = Counter(record["lead_name"] for record in record_metadata)

    # Collect a set of patient_ids
    patient_ids = {record["patient_id"] for record in record_metadata}

    print("\nDataset Summary")
    print("=" * 50)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Number of records: {len(record_metadata)}")
    print(f"Number of patients: {len(patient_ids)}")
    print(f"Window size: {X.shape[1]} samples")

    # Lead distribution
    print("\nLead distribution")
    print("-" * 50)

    for lead_name, count in lead_counts.most_common():
        print(f"{lead_name}: {count} records")

    # Class distribution
    print("\nClass distribution")
    print("-" * 50)
    total_labels = len(y)

    for label, count in label_counts.most_common():
        percentage = (count / total_labels) * 100
        print(f"{label}: {count} beats ({percentage:.2f}%)")

    print("\nBeats per record")
    print("-" * 50)
    for segment in record_metadata:
        print(
            f"Record {segment['record_id']}: "
            f"{segment['num_beats']} beats, "
            f"lead={segment['lead_name']}, "
            f"patient={segment['patient_id']}"
        )

def main() -> None:
    X, y, record_segments = load_dataset()
    validate_dataset(X, y, record_segments)
    print_dataset_summary(X, y, record_segments)


if __name__ == "__main__":
    main()
