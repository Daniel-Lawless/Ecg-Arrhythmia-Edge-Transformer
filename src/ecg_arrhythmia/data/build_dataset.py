import json
import logging
from pathlib import Path
from typing import TypedDict

import numpy as np

from ecg_arrhythmia.data.label_mapping import map_labels_to_aami
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.preprocessing.beat_extraction import extract_beats

logger = logging.getLogger(__name__)


# Define RecordSegment type
class RecordSegment(TypedDict):
    record_id: str
    patient_id: str
    lead_name: str
    start_index: int
    end_index: int
    num_beats: int


# All 48 records in the MIT-BIH Arrhythmia Database.
# fmt: off
MITDB_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
# fmt: on

# These records contain paced beats. We can exclude these later
# if I want to test a non-pace classifier.
PACED_RECORDS = {"102", "104", "107", "217"}


def get_patient_id(record_name: str) -> str:
    """
    Return the patient/group ID for a MIT-BIH record.
    Records 201 and 202 came from the same person, so they share one patient ID.
    """

    if record_name in {"201", "202"}:
        return "201_202"

    return record_name


def build_dataset(
    record_names: list[str] | None = None,
    output_dir: Path = Path("data/processed"),
    excluded_records: set[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[RecordSegment]]:
    """
    Build a beat-level ECG dataset from MIT-BIH records.

    Saves:
    - X.npy: all extracted ECG beat windows
    - y.npy: all corresponding beat labels
    - record_segments.json: metadata showing which rows belong to each record
    """

    if record_names is None:
        record_names = MITDB_RECORDS

    if excluded_records is None:
        excluded_records = set()

    all_beats = []
    all_labels = []
    record_segments = []

    # We can use these for patient-wise splitting later
    all_patient_ids = []

    current_start_index = 0

    # For each patient in the dataset
    for record_name in record_names:
        if record_name in excluded_records:
            logger.info("Skipping excluded record %s", record_name)
            continue

        logger.info("Processing record %s", record_name)

        # Load the record
        signals, fields, annotation = load_record(record_name=record_name)

        # Select the signal channel
        signal, lead_name = select_signal_channel(
            signals=signals,
            fields=fields,
        )

        if annotation.symbol is None:
            raise ValueError(f"Record {record_name} contains no annotation symbols")

        # Extract beats and labels for this record.
        beats_matrix, labels = extract_beats(
            signal=signal,
            annotation_samples=annotation.sample,
            annotation_symbols=annotation.symbol,
        )

        # Map each label to its AAMI map
        labels = map_labels_to_aami(labels=labels)

        if beats_matrix.shape[0] == 0:
            logger.warning("No beats extracted for record %s", record_name)
            continue

        # Number of windows for this record
        number_of_beats = beats_matrix.shape[0]

        # Start index of a beat_matrix to end index of a beat matrix.
        start_index = current_start_index
        end_index = start_index + number_of_beats

        patient_id = get_patient_id(record_name)

        # Append metadata.
        record_segments.append(
            {
                "record_id": record_name,
                "patient_id": patient_id,
                "lead_name": lead_name,
                "start_index": start_index,  # Start of this records matrix
                "end_index": end_index,  # End of this records matrix
                "num_beats": number_of_beats,  # Number of windows in this record
            }
        )

        # Append this records beats and labels.
        all_beats.append(beats_matrix)
        all_labels.append(labels)
        all_patient_ids.extend([patient_id] * number_of_beats)

        current_start_index = end_index

        logger.info(
            "Record %s complete: extracted %s beats from lead %s",
            record_name,
            number_of_beats,
            lead_name,
        )

    if len(all_beats) == 0:
        raise ValueError("No beats were extracted from any records")

    # X is all record matrices stacked
    X_data = np.vstack(all_beats)

    # Concatenates each label np.array into 1 long np.array
    # i.e., labels = [np.array(["V", "N", "Q"]), np.array(["N", "N", "L"])]
    # all_labels = np.concat(labels) = ['V' 'N' 'Q' 'N' 'N' 'L']
    y_labels = np.concatenate(all_labels)

    # Turns our patient_ids list into type np.ndarray
    patient_ids = np.array(all_patient_ids)

    if (
        X_data.shape[0] != y_labels.shape[0]
        or y_labels.shape[0] != patient_ids.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: X={X_data.shape[0]}, y={y_labels.shape[0]}, "
            f"patient_ids={patient_ids.shape[0]}"
        )

    # Make the directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Specify data paths
    X_path = output_dir / "X.npy"
    y_path = output_dir / "y.npy"
    patient_id_path = output_dir / "patient_ids.npy"
    metadata_path = output_dir / "record_segments.json"

    # Save X, y data, and patient_ids
    np.save(file=X_path, arr=X_data)
    np.save(file=y_path, arr=y_labels)
    np.save(file=patient_id_path, arr=patient_ids)

    # Write each dictionary into metadata_path
    with metadata_path.open("w", encoding="utf8") as file:
        json.dump(record_segments, file, indent=4)

    logger.info(
        "Saved X.npy with shape %s to file %s", X_data.shape, X_path
    )
    logger.info(
        "Saved y.npy with shape %s to file %s", y_labels.shape, y_path
    )
    logger.info(
        "Saved patient IDs with shape %s to file %s", patient_ids.shape, patient_id_path
    )
    logger.info(
        "Saved to file %s with %s records", metadata_path, len(record_segments)
    )

    return X_data, y_labels, patient_ids, record_segments


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    build_dataset(excluded_records=PACED_RECORDS)
