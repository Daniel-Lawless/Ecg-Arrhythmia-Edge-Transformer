from collections import Counter
from typing import Any

import numpy as np

from ecg_arrhythmia.data.dataset_io import load_dataset, validate_dataset


def print_dataset_summary(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    record_metadata: list[dict[str, Any]],
) -> None:

    # Counts how often each annotation appears.
    label_counts = Counter(y.tolist())

    # Counts how often each lead appears
    lead_counts = Counter(record["lead_name"] for record in record_metadata)

    # Counts many times each patient_id occurs
    beat_patient_counts = Counter(patient_ids.tolist())

    # Counts the number of patients
    unique_patient_ids = set(patient_ids.tolist())

    print("\nAAMI class distribution")
    print("=" * 50)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"patient_ids shape: {patient_ids.shape}")
    print(f"Number of records: {len(record_metadata)}")
    print(f"Number of patients: {len(unique_patient_ids)}")
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

    print("\nBeats per patient")
    print("-" * 50)

    for patient_id, count in beat_patient_counts.most_common():
        print(f"Patient {patient_id}: {count} beats")

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
    X, y, patients_ids, record_segments = load_dataset()
    validate_dataset(X, y, patients_ids, record_segments)
    print_dataset_summary(X, y, patients_ids, record_segments)


if __name__ == "__main__":
    main()
