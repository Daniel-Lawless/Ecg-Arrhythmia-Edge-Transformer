import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, TypedDict

import numpy as np

from ecg_arrhythmia.data.ecg_sequence_dataset import LABEL_TO_INDEX

SPLIT_NAMES = ("train", "val", "test")
CLASS_LABELS = tuple(LABEL_TO_INDEX.keys())
NUM_CLASSES = len(CLASS_LABELS)

logger = logging.getLogger(__name__)


# --------------------------------------------------------
#                     Define types
# --------------------------------------------------------


class PerSplit(TypedDict):
    beat_sequences_shape: list[int]
    rr_sequences_shape: list[int]
    y_labels_shape: list[int]
    num_of_beat_sequences: int
    selected_patient_ids: list[str]
    y_label_distribution: dict[str, float]


class SplitSummaryMetrics(TypedDict):
    n_trials: int
    true_y_distribution: dict[str, float]
    per_split: dict[str, PerSplit]


# --------------------------------------------------------
#                    Loading dataset
# --------------------------------------------------------


def load_dataset(input_dir: Path = Path("data/processed_sequences")) -> dict[str, Any]:
    # Load numpy data
    X_sequences = np.load(input_dir / "X.npy")
    rr_sequences = np.load(input_dir / "rr_features.npy")
    y_labels = np.load(input_dir / "y.npy")
    patient_ids = np.load(input_dir / "patient_ids.npy")
    target_indices = np.load(input_dir / "target_indices.npy")

    # Load sequence segments
    sequence_segments_path = input_dir / "sequence_segments.json"
    with sequence_segments_path.open("r", encoding="utf-8") as file:
        sequence_segments = json.load(file)

    return {
        "X_sequences": X_sequences,
        "rr_sequences": rr_sequences,
        "y_labels": y_labels,
        "patient_ids": patient_ids,
        "target_indices": target_indices,
        "sequence_segments": sequence_segments,
    }


def validate_sequence_dataset(
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
) -> None:
    if (
        X_sequences.shape[0] != rr_sequences.shape[0]
        or rr_sequences.shape[0] != y_labels.shape[0]
        or y_labels.shape[0] != patient_ids.shape[0]
        or patient_ids.shape[0] != target_indices.shape[0]
    ):
        raise ValueError(
            f"Shape mismatch: X={X_sequences.shape[0]}, "
            f"rr={rr_sequences.shape[0]}, "
            f"y={y_labels.shape[0]}, "
            f"patient_ids={patient_ids.shape[0]}, "
            f"target_indices={target_indices.shape[0]}"
        )

    if X_sequences.ndim != 3:
        raise ValueError(
            f"X_sequences must have shape (num_sequences, K, 240)."
            f"Found {X_sequences.shape}"
        )

    if rr_sequences.ndim != 3 or rr_sequences.shape[-1] != 2:
        raise ValueError(
            "rr_sequences must have shape (num_sequences, K, 2). "
            f"Found {rr_sequences.shape}"
        )


# --------------------------------------------------------
#             Helper to summarise the split
# --------------------------------------------------------


def _y_label_distribution(y_labels: np.ndarray) -> dict[str, float]:
    # Counts how often each label occurs in the split
    y_labels_count = Counter(y_labels.tolist())

    # Return its distribution
    return {
        label: float(np.round(y_labels_count.get(label, 0) / len(y_labels), 4))
        for label in CLASS_LABELS
    }


def _split_summary(
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    best_indices: dict[str, np.ndarray],
    selected_patient_ids: dict[str, np.ndarray],
    n_trials: int,
) -> SplitSummaryMetrics:
    return {
        "n_trials": n_trials,
        "true_y_distribution": _y_label_distribution(y_labels),
        "per_split": {
            split_name: {
                "beat_sequences_shape": list(
                    X_sequences[best_indices[f"{split_name}_indices"]].shape
                ),
                "rr_sequences_shape": list(
                    rr_sequences[best_indices[f"{split_name}_indices"]].shape
                ),
                "y_labels_shape": list(
                    y_labels[best_indices[f"{split_name}_indices"]].shape
                ),
                "num_of_beat_sequences": int(
                    len(best_indices[f"{split_name}_indices"])
                ),
                "selected_patient_ids": [
                    str(patient_id)
                    for patient_id in selected_patient_ids[f"{split_name}_patient_ids"]
                ],
                "y_label_distribution": _y_label_distribution(
                    y_labels[best_indices[f"{split_name}_indices"]]
                ),
            }
            for split_name in SPLIT_NAMES
        },
    }


# --------------------------------------------------------
#            Helpers for the splitting logic
# --------------------------------------------------------


def _validate_split_ratios(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> None:

    if train_ratio <= 0 or val_ratio <= 0 or test_ratio <= 0:
        raise ValueError("All split ratios must be greater than 0")

    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")


def _get_split_patient_ids(
    patient_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    number_of_patients = len(patient_ids)

    if number_of_patients < 3:
        raise ValueError(
            f"Must have 3 or more patients. Number of patients: {number_of_patients}"
        )

    # Must have at least one ID and 2 spaces left for val and test
    n_train = min(
        max(1, round(number_of_patients * train_ratio)),
        number_of_patients - 2,
    )

    # Must have at least one ID and 1 space left for test
    n_val = min(
        max(1, round(number_of_patients * val_ratio)),
        number_of_patients - n_train - 1,
    )

    # Extract patient_ids
    train_patient_ids = patient_ids[:n_train]
    val_patient_ids = patient_ids[n_train : n_train + n_val]
    test_patient_ids = patient_ids[n_train + n_val :]

    return train_patient_ids, val_patient_ids, test_patient_ids


def _build_patient_to_indices(
    sequence_segments: dict[str, Any],
) -> dict[str, np.ndarray]:
    if not sequence_segments:
        raise ValueError("No sequences found")

    patient_to_index_chunks: dict[str, list[np.ndarray]] = {}

    # For each sequence segment, store the sequence row indices belonging to
    # that patient. A patient can have more than one segment, so we collect
    # chunks first and concatenate.
    for sequence_segment in sequence_segments["sequence_segments"]:
        patient_id = str(sequence_segment["patient_id"])
        start_index = int(sequence_segment["start_index"])
        end_index = int(sequence_segment["end_index"])

        patient_to_index_chunks.setdefault(patient_id, []).append(
            np.arange(start_index, end_index)
        )

    return {
        patient_id: np.concatenate(index_chunks)
        for patient_id, index_chunks in patient_to_index_chunks.items()
    }


def _get_indices(
    split_patient_ids: np.ndarray,
    patient_to_indices: dict[str, np.ndarray],
) -> np.ndarray:
    if split_patient_ids.size == 0:
        raise ValueError("No patient ids")

    # Get the sequence row indices for every patient in this split.
    index_chunks = [
        patient_to_indices[str(patient_id)] for patient_id in split_patient_ids
    ]

    if not index_chunks:
        raise ValueError("No matching patient ids found in sequence segments")

    return np.concatenate(index_chunks)


def _encode_labels(y_labels: np.ndarray) -> np.ndarray:
    return np.array([LABEL_TO_INDEX[str(label)] for label in y_labels], dtype=np.int64)


def _build_patient_stats(
    patient_to_indices: dict[str, np.ndarray],
    y_label_indices: np.ndarray,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    patient_to_num_sequences: dict[str, int] = {}
    patient_to_label_counts: dict[str, np.ndarray] = {}

    for patient_id, indices in patient_to_indices.items():
        patient_to_num_sequences[patient_id] = int(len(indices))
        patient_to_label_counts[patient_id] = np.bincount(
            y_label_indices[indices],
            minlength=NUM_CLASSES,
        )

    return patient_to_num_sequences, patient_to_label_counts


def _combine_patient_stats(
    split_patient_ids: np.ndarray,
    patient_to_num_sequences: dict[str, int],
    patient_to_label_counts: dict[str, np.ndarray],
) -> tuple[int, np.ndarray]:
    total_sequences = 0
    label_counts = np.zeros(NUM_CLASSES, dtype=np.int64)

    for patient_id in split_patient_ids:
        patient_id = str(patient_id)
        total_sequences += patient_to_num_sequences[patient_id]
        label_counts += patient_to_label_counts[patient_id]

    return total_sequences, label_counts


# --------------------------------------------------------
#               Helpers to define the error
# --------------------------------------------------------


def _distribution_score(
    true_y_distribution: np.ndarray,
    split_label_counts: np.ndarray,
) -> float:
    # Calculate the proportions for each label in this split.
    split_total = split_label_counts.sum()

    if split_total == 0:
        return float("inf")

    split_distribution = split_label_counts / split_total

    # Calculate the error in the proportions for each class.
    return float(np.sum((true_y_distribution - split_distribution) ** 2))


def _ratio_score(
    num_sequences_in_split: int,
    split_ratio: float,
    total_sequences: int,
) -> float:
    # How much of the total sequence dataset is in this split.
    proportion_of_sequences = num_sequences_in_split / total_sequences

    # Calculate the squared difference between our chosen split_ratio and what we
    # actually got.
    return float((split_ratio - proportion_of_sequences) ** 2)


def _score_split(
    true_y_distribution: np.ndarray,
    split_label_counts: np.ndarray,
    num_sequences_in_split: int,
    split_ratio: float,
    total_sequences: int,
) -> float:
    distribution_score = _distribution_score(
        true_y_distribution=true_y_distribution,
        split_label_counts=split_label_counts,
    )

    ratio_score = _ratio_score(
        num_sequences_in_split=num_sequences_in_split,
        split_ratio=split_ratio,
        total_sequences=total_sequences,
    )

    return distribution_score + ratio_score


def split_sequence_dataset(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
    sequence_segments: dict[str, Any],
    n_trials: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], SplitSummaryMetrics]:
    logger.info("Splitting data...")

    validate_sequence_dataset(
        X_sequences=X_sequences,
        rr_sequences=rr_sequences,
        y_labels=y_labels,
        patient_ids=patient_ids,
        target_indices=target_indices,
    )
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    # Makes access easier later
    ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    # Precompute patient -> sequence indices once.
    # This avoids rebuilding large index arrays inside every trial.
    patient_to_indices = _build_patient_to_indices(sequence_segments)

    # Convert labels once so class counts can use np.bincount.
    y_label_indices = _encode_labels(y_labels)

    # Precompute patient-level counts once.
    # During trials, we score patient groups using these counts instead of
    # slicing large arrays.
    patient_to_num_sequences, patient_to_label_counts = _build_patient_stats(
        patient_to_indices=patient_to_indices,
        y_label_indices=y_label_indices,
    )

    total_sequences = len(y_labels)

    true_label_counts = np.bincount(y_label_indices, minlength=NUM_CLASSES)
    true_y_distribution = true_label_counts / true_label_counts.sum()

    # Set of best indices to minimise the error conditions
    best_indices: dict[str, np.ndarray] = {}

    # Best train/val/test split
    best_patient_split: dict[str, np.ndarray] = {}

    # Keep a running score
    best_score = float("inf")

    # Define generator object
    random_generator = np.random.default_rng(seed=seed)

    # Unique patient ids. This is what will change each loop
    unique_patient_ids = np.array(sorted(patient_to_indices.keys()))

    logger.info("Trying %s splits", n_trials)

    for trial in range(n_trials):
        if trial % 1000 == 0:
            logger.info("Trial: %s", trial)

        # Randomly shuffles our unique ids.
        shuffled_patient_ids = random_generator.permutation(unique_patient_ids)

        # Get the patient_ids of the sample splits
        train_patient_ids, val_patient_ids, test_patient_ids = _get_split_patient_ids(
            patient_ids=shuffled_patient_ids,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        candidate_patient_split = {
            "train": train_patient_ids,
            "val": val_patient_ids,
            "test": test_patient_ids,
        }

        # Calculate the score of this split.
        split_score = 0.0

        for split_name in SPLIT_NAMES:
            num_sequences_in_split, split_label_counts = _combine_patient_stats(
                split_patient_ids=candidate_patient_split[split_name],
                patient_to_num_sequences=patient_to_num_sequences,
                patient_to_label_counts=patient_to_label_counts,
            )

            split_score += _score_split(
                true_y_distribution=true_y_distribution,
                split_label_counts=split_label_counts,
                num_sequences_in_split=num_sequences_in_split,
                split_ratio=ratios[split_name],
                total_sequences=total_sequences,
            )

        if split_score < best_score:
            best_score = split_score
            best_patient_split = {
                "train_patient_ids": train_patient_ids,
                "val_patient_ids": val_patient_ids,
                "test_patient_ids": test_patient_ids,
            }

    logger.info("Trial: %s\nSplitting completed", n_trials)
    logger.info("Best split score: %.6f", best_score)

    # Build the actual sequence row indices only once, after the best
    # patient split is known.
    best_indices = {
        "train_indices": _get_indices(
            split_patient_ids=best_patient_split["train_patient_ids"],
            patient_to_indices=patient_to_indices,
        ),
        "val_indices": _get_indices(
            split_patient_ids=best_patient_split["val_patient_ids"],
            patient_to_indices=patient_to_indices,
        ),
        "test_indices": _get_indices(
            split_patient_ids=best_patient_split["test_patient_ids"],
            patient_to_indices=patient_to_indices,
        ),
    }

    logger.info("Building summary metrics")

    split_summary_metrics = _split_summary(
        X_sequences=X_sequences,
        rr_sequences=rr_sequences,
        y_labels=y_labels,
        best_indices=best_indices,
        selected_patient_ids=best_patient_split,
        n_trials=n_trials,
    )

    return best_indices, split_summary_metrics


# --------------------------------------------------------
#                   Saving split data
# --------------------------------------------------------


def save_split_dataset(
    output_dir: Path,
    X_sequences: np.ndarray,
    rr_sequences: np.ndarray,
    y_labels: np.ndarray,
    patient_ids: np.ndarray,
    target_indices: np.ndarray,
    best_indices: dict[str, np.ndarray],
    split_summary_metrics: SplitSummaryMetrics,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in SPLIT_NAMES:
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        indices = best_indices[f"{split_name}_indices"]

        np.save(split_dir / "X.npy", X_sequences[indices])
        np.save(split_dir / "rr_features.npy", rr_sequences[indices])
        np.save(split_dir / "y.npy", y_labels[indices])
        np.save(split_dir / "patient_ids.npy", patient_ids[indices])
        np.save(split_dir / "target_indices.npy", target_indices[indices])

    np.savez(
        output_dir / "split_indices.npz",
        train_indices=best_indices["train_indices"],
        val_indices=best_indices["val_indices"],
        test_indices=best_indices["test_indices"],
    )

    with (output_dir / "split_summary_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(fp=file, obj=split_summary_metrics, indent=4)

    logger.info("Saved sequence splits to %s", output_dir)


# --------------------------------------------------------
#                   Main splitting logic
# --------------------------------------------------------


def main(
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    input_dir: Path = Path("data/processed_sequences"),
    output_dir: Path = Path("data/splits_sequences"),
    n_trials: int = 10000,
    seed: int = 42,
) -> None:
    logger.info("Loading dataset...")

    # Load the data
    data_dict = load_dataset(input_dir)

    logger.info("Dataset loaded")

    best_indices, split_summary_metrics = split_sequence_dataset(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        X_sequences=data_dict["X_sequences"],
        rr_sequences=data_dict["rr_sequences"],
        y_labels=data_dict["y_labels"],
        patient_ids=data_dict["patient_ids"],
        target_indices=data_dict["target_indices"],
        sequence_segments=data_dict["sequence_segments"],
        n_trials=n_trials,
        seed=seed,
    )

    save_split_dataset(
        output_dir=output_dir,
        X_sequences=data_dict["X_sequences"],
        rr_sequences=data_dict["rr_sequences"],
        y_labels=data_dict["y_labels"],
        patient_ids=data_dict["patient_ids"],
        target_indices=data_dict["target_indices"],
        best_indices=best_indices,
        split_summary_metrics=split_summary_metrics,
    )


# --------------------------------------------------------
#                          CLI
# --------------------------------------------------------


def parse_args() -> argparse.Namespace:
    # Define parser
    parser = argparse.ArgumentParser(description="CLI for sequence_splitter")

    # Add input directory CLI argument
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/processed_sequences"),
        help="Directory to where the sequence data is stored",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/splits_sequences"),
        help="Directory to where the split sequence data will be stored",
    )

    # Add n_trials CLI argument
    parser.add_argument(
        "--n-trials",
        type=int,
        default=10000,
        help="Number of patient_splits",
    )

    # Add seed CLI argument
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="select seed for reproducibility",
    )

    # Add ratio CLI arguments
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Goal train ratio",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Goal val ratio",
    )

    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Goal test ratio",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level="INFO")

    # Get command line arguments
    args = parse_args()

    # Call main function
    main(
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        n_trials=args.n_trials,
        seed=args.seed,
    )
