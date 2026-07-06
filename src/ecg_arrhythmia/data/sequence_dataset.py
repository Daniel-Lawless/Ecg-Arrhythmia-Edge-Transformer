from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ecg_arrhythmia.data.dataset_io import load_dataset, validate_dataset

logger = logging.getLogger(__name__)


# Define sequence segment type
class SequenceSegment(TypedDict):
    record_id: str
    patient_id: str
    start_index: int
    end_index: int
    num_sequences: int


# We pass in our data, and this makes sequences from them. Here, one X sample will be
# [beat i-4, beat i-3. beat i-2, beat i-1, beat i] where each beat i is a 240 window.
# This is called casual since it never since future beats, this is important for
# real-time inference.
# RR sample: [
#   RR features for beat i-4,
#   RR features for beat i-3,
#   RR features for beat i-2,
#   RR features for beat i-1,
#   RR features for beat i]
# And then the label will be for beat i


# sequences must built within each record only, never across records to avoid
# data leakage.
def create_record_sequences(
    X: np.ndarray,
    y: np.ndarray,
    rr_features: np.ndarray,
    patient_ids: np.ndarray,
    record_metadata: list[dict[str, Any]],
    sequence_length: int = 5,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[SequenceSegment],
]:
    """
    Build causal beat sequences inside each record.

    For sequence_length=5, each target beat i receives:
    [beat i-4, beat i-3, beat i-2, beat i-1, beat i].
    The target label is the label for beat i.
    """

    if sequence_length <= 0:
        raise ValueError("sequence_length must be greater than 0")

    logger.info("Validating data...")

    validate_dataset(X, y, patient_ids, rr_features, record_metadata)

    logger.info("Data validated")

    X_sequences_chunks: list[np.ndarray] = []
    y_targets_chunks: list[np.ndarray] = []
    rr_sequences_chunks: list[np.ndarray] = []
    sequence_patient_ids_chunks: list[np.ndarray] = []
    target_indices_chunks: list[np.ndarray] = []
    sequence_segments: list[SequenceSegment] = []

    current_sequence_start = 0

    logger.info("Initalising sequencing...")

    # For each record
    for record in record_metadata:
        # Get the start and end of the record
        record_start = int(record["start_index"])
        record_end = int(record["end_index"])
        record_id = str(record["record_id"])
        patient_id = str(record["patient_id"])
        num_record_beats = record_end - record_start

        logger.info("Sequencing record %s", record_id)

        # If the number of beats in the record is less than
        # sequence length, then we cannot make a sequence, so skip it.
        if num_record_beats < sequence_length:
            continue

        # This is the number of sequences we can make from this record
        num_sequences = num_record_beats - sequence_length + 1
        record_sequence_start = current_sequence_start

        # Starts creates number 0,1,2,..., num_sequences - 1. These
        # are the starting indicies of each sequence
        starts = np.arange(num_sequences)
        # offsets tells us how far we go from that start index.
        # 0,1,2... sequence_length - 1. So if a sequence starts at 10,
        # this will include 10+0, 10+1, 10+2, 10+3, 10+4 for seq_len=5
        # So it says once I know where the window starts, how many steps forward
        # do I need to grab the rest of the beats
        offsets = np.arange(sequence_length)

        # Create all sliding-window indices for this record in one vectorised step.
        # starts[:, np.newaxis] is shaped (num_sequences, 1), offsets[np.newaxis, :] is
        # shaped (1, sequence_length), so broadcasting gives
        # (num_sequences, sequence_length).
        window_indices = record_start + starts[:, np.newaxis] + offsets[np.newaxis, :]

        # Example with record_start=10, starts=[0, 1, 2], offsets=[0, 1, 2, 3, 4]:
        # [[10, 11, 12, 13, 14],
        #  [11, 12, 13, 14, 15],
        #  [12, 13, 14, 15, 16]]

        # Extract blocks of X and rr_features in one operation.
        X_sequences_chunks.append(X[window_indices])
        # [
        #   [beat_10, beat_11, beat_12, beat_13, beat_14],
        #   [beat_11, beat_12, beat_13, beat_14, beat_15],
        #   [beat_12, beat_13, beat_14, beat_15, beat_16]
        #   etc...
        # ]
        # Same for the rr_features
        rr_sequences_chunks.append(rr_features[window_indices])

        # Looks in winow indicies and takes the last element of each.
        # array([14, 15, 16]) (3,)
        target_idx_array = window_indices[:, -1]

        # Index the labels with these indices, so append the labels
        # of all beats at the end of the sequences for this record
        y_targets_chunks.append(y[target_idx_array].astype(str))
        # Do the same for the patient ids.
        sequence_patient_ids_chunks.append(patient_ids[target_idx_array].astype(str))
        # Append these indices so we know where the end beat in each sequence is
        # in each record.
        target_indices_chunks.append(target_idx_array)

        # Move the global sequence counter forward so the next record's
        # sequence segment starts after the sequences created for this record.
        current_sequence_start += num_sequences

        # This is what SequenceSegment expects
        sequence_segments.append(
            {
                "record_id": record_id,
                "patient_id": patient_id,
                "start_index": record_sequence_start,
                "end_index": current_sequence_start,
                "num_sequences": num_sequences,
            }
        )

    logger.info("Sequencing complete")

    if not X_sequences_chunks:
        raise ValueError("No sequences could be created from the provided records")

    # Concatenate the block arrays along the 0th axis.
    # So we get (num_all_sequences, sequence length, channel, beat_window_length)
    return (
        np.concatenate(X_sequences_chunks, axis=0),
        np.concatenate(y_targets_chunks, axis=0),
        np.concatenate(rr_sequences_chunks, axis=0),
        np.concatenate(sequence_patient_ids_chunks, axis=0),
        np.concatenate(target_indices_chunks, axis=0).astype(np.int64),
        sequence_segments,
    )


def save_sequence_dataset(
    output_dir: Path,
    X_sequences: np.ndarray,
    y: np.ndarray,
    rr_sequences: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
    sequence_segments: list[SequenceSegment],
    sequence_length: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Preparing to save data...")

    # Define paths
    np.save(output_dir / "X.npy", X_sequences)
    np.save(output_dir / "y.npy", y)
    np.save(output_dir / "rr_features.npy", rr_sequences)
    np.save(output_dir / "patient_ids.npy", patient_ids)
    np.save(output_dir / "target_indices.npy", target_indices)

    metadata = {
        "sequence_length": sequence_length,
        "num_sequences": int(X_sequences.shape[0]),
        "X_shape": list(X_sequences.shape),
        "rr_features_shape": list(rr_sequences.shape),
        "sequence_segments": sequence_segments,
    }

    # Write the metadata to a file
    with (output_dir / "sequence_segments.json").open("w", encoding="utf8") as file:
        json.dump(metadata, file, indent=4)

    logger.info("Data saved to %s", output_dir)


def build_sequence_dataset(
    input_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/processed_sequences"),
    sequence_length: int = 5,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[SequenceSegment]
]:
    logger.info("Loading data...")

    # Load our data
    X, y, patient_ids, rr_features, record_metadata = load_dataset(index_path=input_dir)

    logger.info("Data loaded")
    # Extract the sequences from this data
    sequences = create_record_sequences(
        X=X,
        y=y,
        rr_features=rr_features,
        patient_ids=patient_ids,
        record_metadata=record_metadata,
        sequence_length=sequence_length,
    )

    (
        X_sequences,
        y_targets,
        rr_sequences,
        sequence_patient_ids,
        target_indices,
        segments,
    ) = sequences

    # Save these sequences.
    save_sequence_dataset(
        output_dir=output_dir,
        X_sequences=X_sequences,
        y=y_targets,
        rr_sequences=rr_sequences,
        patient_ids=sequence_patient_ids,
        target_indices=target_indices,
        sequence_segments=segments,
        sequence_length=sequence_length,
    )

    return sequences


# Define our CLI
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build causal ECG beat sequences.")

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed"),
        help="Directory containing beat-level X/y/rr_features/patient_ids arrays.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed_sequences"),
        help="Directory where sequence arrays will be saved.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=5,
        help="Number of beats per sequence, including the target beat.",
    )

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    X_sequences, y, rr_sequences, patient_ids, _, _ = build_sequence_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        sequence_length=args.sequence_length,
    )

    logger.info(
        "Saved sequence dataset to %s | X=%s | y=%s | rr_features=%s | patient_ids=%s",
        args.output_dir,
        X_sequences.shape,
        y.shape,
        rr_sequences.shape,
        patient_ids.shape,
    )


if __name__ == "__main__":
    main()
