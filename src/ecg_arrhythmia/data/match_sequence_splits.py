import argparse
import json
import logging
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ecg_arrhythmia.data.ecg_sequence_dataset import LABEL_TO_INDEX

logger = logging.getLogger(__name__)

# We use the same split names throughout the script so they stay consistent.
SPLIT_NAMES = ("train", "val", "test")

# Gives us the class labels in the same order as LABEL_TO_INDEX.
CLASS_LABELS = tuple(LABEL_TO_INDEX.keys())


# ---------------------------------------------------------------------
#                             Define Types
# ---------------------------------------------------------------------


# This defines what information we save in the summary for one split.
class PerSplitSummary(TypedDict):
    beat_sequences_shape: list[int]
    rr_sequences_shape: list[int]
    y_labels_shape: list[int]
    target_indices_shape: list[int]
    num_of_beat_sequences: int
    selected_patient_ids: list[str]
    y_label_distribution: dict[str, float]


# This defines the structure of each train/val/test summary.
class MatchedSplitSummary(TypedDict):
    split_method: str
    reference_split_dir: str
    true_y_distribution: dict[str, float]
    per_split: dict[str, PerSplitSummary]


# ---------------------------------------------------------------------
#                         Load Sequence Dataset
# ---------------------------------------------------------------------


def load_sequence_dataset(input_dir: Path) -> dict[str, Any]:
    """
    Load the complete sequence dataset before it has been split.
    """

    # Load each of the arrays created when we built the sequence dataset.
    X_sequences = np.load(input_dir / "X.npy")
    rr_sequences = np.load(input_dir / "rr_features.npy")
    y_labels = np.load(input_dir / "y.npy")
    patient_ids = np.load(input_dir / "patient_ids.npy").astype(str)
    target_indices = np.load(input_dir / "target_indices.npy")

    # sequence_segments tells us which sequence rows belong to each record/patient.
    sequence_segments_path = input_dir / "sequence_segments.json"

    with sequence_segments_path.open("r", encoding="utf-8") as file:
        sequence_segments = json.load(file)

    # Return everything together so it is easier to pass through the pipeline.
    return {
        "X_sequences": X_sequences,
        "rr_sequences": rr_sequences,
        "y_labels": y_labels,
        "patient_ids": patient_ids,
        "target_indices": target_indices,
        "sequence_segments": sequence_segments,
    }


# ---------------------------------------------------------------------
#                       Validate Sequence Dataset
# ---------------------------------------------------------------------


def validate_sequence_dataset(
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
) -> None:
    """
    Check that all sequence-level arrays describe the same samples.
    """

    # Every array must contain the same number of sequence samples.
    num_sequences = X_sequences.shape[0]

    if (
        rr_sequences.shape[0] != num_sequences
        or y_labels.shape[0] != num_sequences
        or patient_ids.shape[0] != num_sequences
        or target_indices.shape[0] != num_sequences
    ):
        raise ValueError(
            f"Shape mismatch: X={X_sequences.shape[0]}, "
            f"rr={rr_sequences.shape[0]}, "
            f"y={y_labels.shape[0]}, "
            f"patient_ids={patient_ids.shape[0]}, "
            f"target_indices={target_indices.shape[0]}"
        )

    # X should contain:
    # (num_sequences, sequence_length, beat_window_size)
    if X_sequences.ndim != 3:
        raise ValueError(
            "X_sequences must have shape "
            f"(num_sequences, sequence_length, window_size). Found {X_sequences.shape}"
        )

    # RR sequences should contain two RR features for every beat:
    # (num_sequences, sequence_length, 2)
    if rr_sequences.ndim != 3 or rr_sequences.shape[-1] != 2:
        raise ValueError(
            "rr_sequences must have shape "
            f"(num_sequences, sequence_length, 2). Found {rr_sequences.shape}"
        )

    # X and RR must agree on both the number of sequences and sequence length.
    if X_sequences.shape[:2] != rr_sequences.shape[:2]:
        raise ValueError(
            "X_sequences and rr_sequences must agree on num_sequences "
            f"and sequence_length. Found X={X_sequences.shape}, "
            f"rr={rr_sequences.shape}"
        )


# ---------------------------------------------------------------------
#                   Load CNN Patient Assignments
# ---------------------------------------------------------------------


def load_reference_patient_ids(
    reference_split_dir: Path,
) -> dict[str, np.ndarray]:
    """
    Load the patient assignments used by the original CNN splits.
    """

    patient_splits: dict[str, np.ndarray] = {}

    # Load the patient_ids.npy file from each original CNN split.
    for split_name in SPLIT_NAMES:
        patient_ids_path = reference_split_dir / split_name / "patient_ids.npy"

        if not patient_ids_path.exists():
            raise FileNotFoundError(f"No patient_ids.npy found at {patient_ids_path}")

        patient_ids = np.load(patient_ids_path).astype(str)

        # The saved file contains one patient ID per beat.
        # We only need each patient once for the matched split.
        patient_splits[split_name] = np.unique(patient_ids)

    # Make sure the original CNN split itself does not contain patient leakage.
    validate_reference_patient_splits(patient_splits)

    return patient_splits


def validate_reference_patient_splits(
    patient_splits: dict[str, np.ndarray],
) -> None:
    """
    Ensure that a patient has not been assigned to more than one split.
    """

    # Convert each patient array into a set so we can easily check overlap.
    split_sets = {
        split_name: set(patient_splits[split_name].tolist())
        for split_name in SPLIT_NAMES
    }

    # Compare train vs val, train vs test, and val vs test.
    for first_index, first_name in enumerate(SPLIT_NAMES):
        for second_name in SPLIT_NAMES[first_index + 1 :]:
            overlap = split_sets[first_name] & split_sets[second_name]

            if overlap:
                raise ValueError(
                    f"Patient leakage between {first_name} and {second_name}: "
                    f"{sorted(overlap)}"
                )


# ---------------------------------------------------------------------
#                Map Patients To Their Sequence Rows
# ---------------------------------------------------------------------


def build_patient_to_sequence_indices(
    sequence_segments: dict[str, Any],
) -> dict[str, np.ndarray]:
    """
    Map every patient ID to all rows belonging to that patient in the
    complete sequence dataset.
    """

    # Grab the list of record-level sequence segments from the metadata.
    segments = sequence_segments.get("sequence_segments")

    if not segments:
        raise ValueError("No sequence segments were found")

    # A patient can have more than one record segment, so we collect
    # their index chunks first before joining them together.
    patient_to_chunks: dict[str, list[np.ndarray]] = {}

    for segment in segments:
        patient_id = str(segment["patient_id"])
        start_index = int(segment["start_index"])
        end_index = int(segment["end_index"])

        # end_index is exclusive, so it must be greater than start_index.
        if end_index <= start_index:
            raise ValueError(
                f"Invalid sequence segment for patient {patient_id}: "
                f"start={start_index}, end={end_index}"
            )

        # Create all sequence row indices for this segment.
        # For example, start=10 and end=13 gives [10, 11, 12].
        patient_to_chunks.setdefault(patient_id, []).append(
            np.arange(start_index, end_index, dtype=np.int64)
        )

    # Join all chunks belonging to the same patient.
    # This also handles patients with more than one record.
    return {
        patient_id: np.concatenate(index_chunks)
        for patient_id, index_chunks in patient_to_chunks.items()
    }


# ---------------------------------------------------------------------
#                     Build The Matched Splits
# ---------------------------------------------------------------------


def build_matched_indices(
    patient_splits: dict[str, np.ndarray],
    patient_to_sequence_indices: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Select sequence rows using the fixed CNN train/validation/test
    patient assignments.
    """

    # All patients available in the full sequence dataset.
    available_patients = set(patient_to_sequence_indices)

    # All patients that were assigned by the original CNN split.
    assigned_patients = set(
        np.concatenate(
            [patient_splits[split_name] for split_name in SPLIT_NAMES]
        ).tolist()
    )

    # Check whether the CNN split contains a patient we cannot find
    # in the sequence dataset.
    missing_from_sequences = assigned_patients - available_patients

    if missing_from_sequences:
        raise ValueError(
            "The reference CNN split contains patients that are missing from "
            f"the sequence dataset: {sorted(missing_from_sequences)}"
        )

    # Check the opposite direction as well.
    # Every sequence patient should belong to one of the CNN splits.
    unassigned_sequence_patients = available_patients - assigned_patients

    if unassigned_sequence_patients:
        raise ValueError(
            "The sequence dataset contains patients that are not assigned in "
            f"the reference CNN split: {sorted(unassigned_sequence_patients)}"
        )

    matched_indices: dict[str, np.ndarray] = {}

    for split_name in SPLIT_NAMES:
        # Look up the sequence row indices for every patient assigned
        # to this split.
        index_chunks = [
            patient_to_sequence_indices[str(patient_id)]
            for patient_id in patient_splits[split_name]
        ]

        if not index_chunks:
            raise ValueError(f"No sequence rows found for {split_name}")

        # Join all patient chunks into one array of rows for this split.
        matched_indices[f"{split_name}_indices"] = np.concatenate(index_chunks)

    # Final check that no sequence row appears in multiple splits.
    validate_matched_indices(matched_indices)

    return matched_indices


def validate_matched_indices(
    matched_indices: dict[str, np.ndarray],
) -> None:
    """
    Ensure no sequence row appears in more than one split.
    """

    # Convert each split's indices to a set so overlap is easy to detect.
    split_index_sets = {
        split_name: set(matched_indices[f"{split_name}_indices"].tolist())
        for split_name in SPLIT_NAMES
    }

    # Compare every pair of splits.
    for first_index, first_name in enumerate(SPLIT_NAMES):
        for second_name in SPLIT_NAMES[first_index + 1 :]:
            overlap = split_index_sets[first_name] & split_index_sets[second_name]

            if overlap:
                raise ValueError(
                    f"Sequence row leakage between {first_name} and "
                    f"{second_name}: {len(overlap)} overlapping rows"
                )


# ---------------------------------------------------------------------
#                         Build Split Summary
# ---------------------------------------------------------------------


def y_label_distribution(y_labels: np.ndarray) -> dict[str, float]:
    """
    Return the proportion of target labels in a sequence split.
    """

    # y_labels == label creates a boolean mask.
    # The mean of that mask is the proportion of samples with that label.
    return {
        label: np.round(float(np.mean(y_labels == label)), 4)
        for label in CLASS_LABELS
    }


def build_split_summary(
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    target_indices: np.ndarray,
    matched_indices: dict[str, np.ndarray],
    patient_splits: dict[str, np.ndarray],
    reference_split_dir: Path,
) -> MatchedSplitSummary:
    # Build a summary for each matched train/val/test split.
    return {
        "split_method": "matched_to_cnn_patient_assignments",
        "reference_split_dir": str(reference_split_dir),
        "true_y_distribution": y_label_distribution(y_labels),
        "per_split": {
            split_name: {
                # Store the shapes so we can quickly check what was saved.
                "beat_sequences_shape": list(
                    X_sequences[matched_indices[f"{split_name}_indices"]].shape
                ),
                "rr_sequences_shape": list(
                    rr_sequences[matched_indices[f"{split_name}_indices"]].shape
                ),
                "y_labels_shape": list(
                    y_labels[matched_indices[f"{split_name}_indices"]].shape
                ),
                "target_indices_shape": list(
                    target_indices[matched_indices[f"{split_name}_indices"]].shape
                ),
                "num_of_beat_sequences": int(
                    len(matched_indices[f"{split_name}_indices"])
                ),
                # Save the exact patient IDs copied from the CNN split.
                "selected_patient_ids": sorted(patient_splits[split_name].tolist()),
                # Show how balanced the target labels are in this split.
                "y_label_distribution": y_label_distribution(
                    y_labels[matched_indices[f"{split_name}_indices"]]
                ),
            }
            for split_name in SPLIT_NAMES
        },
    }


# ---------------------------------------------------------------------
#                         Save Matched Splits
# ---------------------------------------------------------------------


def save_matched_splits(
    output_dir: Path,
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
    matched_indices: dict[str, np.ndarray],
    split_summary: MatchedSplitSummary,
) -> None:
    """
    Save the matched sequence train, validation, and test splits.
    """

    # Create the root output directory if it does not already exist.
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        # Each split gets its own directory.
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Grab the sequence rows assigned to this split.
        indices = matched_indices[f"{split_name}_indices"]

        # Save every array using the same format as the normal sequence split.
        np.save(split_dir / "X.npy", X_sequences[indices])
        np.save(split_dir / "rr_features.npy", rr_sequences[indices])
        np.save(split_dir / "y.npy", y_labels[indices])
        np.save(split_dir / "patient_ids.npy", patient_ids[indices])
        np.save(split_dir / "target_indices.npy", target_indices[indices])

    # Save the row indices used for each split so the split is reproducible.
    np.savez(
        output_dir / "split_indices.npz",
        train_indices=matched_indices["train_indices"],
        val_indices=matched_indices["val_indices"],
        test_indices=matched_indices["test_indices"],
    )

    # Save a summary of the matched split.
    with (output_dir / "split_summary_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(split_summary, file, indent=4)

    logger.info("Saved matched sequence splits to %s", output_dir)


# ---------------------------------------------------------------------
#                     Main Matched Split Pipeline
# ---------------------------------------------------------------------


def create_matched_sequence_splits(
    input_dir: Path,
    reference_split_dir: Path,
    output_dir: Path,
) -> None:
    logger.info("Loading complete sequence dataset from %s", input_dir)

    # Load the complete sequence dataset before splitting.
    data = load_sequence_dataset(input_dir)

    # Check that all of the sequence arrays line up correctly.
    validate_sequence_dataset(
        X_sequences=data["X_sequences"],
        rr_sequences=data["rr_sequences"],
        y_labels=data["y_labels"],
        patient_ids=data["patient_ids"],
        target_indices=data["target_indices"],
    )

    logger.info(
        "Loading reference CNN patient assignments from %s",
        reference_split_dir,
    )

    # Get the train/val/test patient assignments used by the CNN.
    patient_splits = load_reference_patient_ids(reference_split_dir)

    # Build a lookup from each patient ID to their rows in the sequence dataset.
    patient_to_sequence_indices = build_patient_to_sequence_indices(
        data["sequence_segments"]
    )

    # Use the CNN patient assignments to select matching sequence rows.
    matched_indices = build_matched_indices(
        patient_splits=patient_splits,
        patient_to_sequence_indices=patient_to_sequence_indices,
    )

    # Build the JSON summary before saving the arrays.
    split_summary = build_split_summary(
        X_sequences=data["X_sequences"],
        rr_sequences=data["rr_sequences"],
        y_labels=data["y_labels"],
        target_indices=data["target_indices"],
        matched_indices=matched_indices,
        patient_splits=patient_splits,
        reference_split_dir=reference_split_dir,
    )

    # Save the matched train, validation, and test splits.
    save_matched_splits(
        output_dir=output_dir,
        X_sequences=data["X_sequences"],
        rr_sequences=data["rr_sequences"],
        y_labels=data["y_labels"],
        patient_ids=data["patient_ids"],
        target_indices=data["target_indices"],
        matched_indices=matched_indices,
        split_summary=split_summary,
    )

    # Log a small summary for each split so we can check the result quickly.
    for split_name in SPLIT_NAMES:
        split_metrics = split_summary["per_split"][split_name]

        logger.info(
            "%s | sequences: %s | patients: %s | distribution: %s",
            split_name,
            split_metrics["num_of_beat_sequences"],
            split_metrics["selected_patient_ids"],
            split_metrics["y_label_distribution"],
        )


# ---------------------------------------------------------------------
#                             CLI Parser
# ---------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split the sequence dataset using the exact patient assignments "
            "from the original CNN train/validation/test splits."
        )
    )

    # Define command line arguments
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed_sequences"),
        help="Directory containing the complete unsplit sequence dataset.",
    )

    parser.add_argument(
        "--reference-split-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing the original CNN train/val/test splits.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits_sequences_matched"),
        help="Directory in which to save the matched sequence splits.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------
#                                 Main
# ---------------------------------------------------------------------


def main() -> None:
    # Show INFO logs when the script runs.
    logging.basicConfig(level=logging.INFO)

    # Read the command line arguments.
    args = parse_args()

    # Run the full matched-split pipeline.
    create_matched_sequence_splits(
        input_dir=args.input_dir,
        reference_split_dir=args.reference_split_dir,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()