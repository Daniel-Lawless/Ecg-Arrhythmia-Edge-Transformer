import json
import logging
from collections import Counter
from pathlib import Path
from typing import TypedDict

import numpy as np

from ecg_arrhythmia.data.dataset_io import load_dataset, validate_dataset

logger = logging.getLogger(__name__)


# This allows us to create a new type
# which is a dictionary with known key names
# and know value types
class SplitData(TypedDict):
    X: np.ndarray
    y: np.ndarray
    patient_ids: np.ndarray
    indices: np.ndarray
    selected_patient_ids: list[str]


# This means any dict of type SplitData must have keys X, y,
# patient_ids etc with the corresponding types


class DatasetSplits(TypedDict):
    train: SplitData
    val: SplitData
    test: SplitData


# So each set must be a SplitData dict

SPLIT_NAMES = ("train", "val", "test")


def _validate_split_ratios(
    train_ratio: float, val_ratio: float, test_ratio: float
) -> None:

    ratios = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
    }

    # No set should have a ratio less than or equal to 0
    for key, value in ratios.items():
        if value <= 0:
            raise ValueError(f"ratio for set {key} cannot be less than 0. ")

    # The sum of ratios should be approximatley 1
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("Set ratios must sum to 1")


def _split_patient_ids(
    unique_patient_ids: np.ndarray,
    rng: np.random.Generator,
    train_ratio,
    val_ratio,
) -> dict[str, list[str]]:

    # Shuffle the patient ids
    shuffled_patient_ids = rng.permutation(unique_patient_ids)

    # Total number of patients
    number_of_patients = len(shuffled_patient_ids)

    if number_of_patients < 3:
        raise ValueError(
            f"num of patients: {number_of_patients}"
            f"Atleast 3 patients are needed for a train/val/test split."
        )

    # Get the number of patients that will be used for the train and val sets
    n_train = int(round(number_of_patients * train_ratio))
    n_val = int(round(number_of_patients * val_ratio))

    # This ensures we have at least 1 training example, and at least
    # two spots remaining for val and test.
    # 1 <= n_train <= num_patients - 2
    n_train = min(max(n_train, 1), number_of_patients - 2)

    # This ensures there is 1 spot remaining for test.
    # It can be atleast 1 and at must leave 1 spot remaining.
    # 1 <= n_val <= number_of_patients - n_train - 1
    n_val = min(max(n_val, 1), number_of_patients - n_train - 1)

    # Assign that many patient_ids to train and val
    train_patient_ids = shuffled_patient_ids[:n_train]
    val_patient_ids = shuffled_patient_ids[n_train : n_train + n_val]
    test_patient_ids = shuffled_patient_ids[n_train + n_val :]

    return {
        "train": sorted(train_patient_ids.tolist()),
        "val": sorted(val_patient_ids.tolist()),
        "test": sorted(test_patient_ids.tolist()),
    }


def _indices_for_patients(
    patient_ids: np.ndarray, selected_patient_ids: list[str]
) -> np.ndarray:
    # np.isin() returns true where selected_patient_ids occur in patient_ids.
    # np.flapnonzero() goes through this list of True and False and returns the
    # index of the True values.
    #
    # Returns the indicies of where the selected ids are in patient_ids
    # These indices correspond to windows in X and y that had the selected patient ids.
    return np.flatnonzero(np.isin(patient_ids, selected_patient_ids))


def _class_distribution_error(y: np.ndarray, split_indices: np.ndarray) -> float:

    # Returns each label once, plus how many times that label occurred
    # in the original y array.
    all_labels, all_counts = np.unique(y, return_counts=True)

    # Returns the y label for the corresponding windows in the split
    # and how often  that label occured in that split array
    split_labels, split_counts = np.unique(y[split_indices], return_counts=True)

    # calculates the proportion of each label to the whole y array
    all_distribution = {
        label: count / len(y)
        for label, count in zip(all_labels, all_counts, strict=True)
    }

    # calculates the proportion of each label to the y labels for
    # this split
    split_distribution = {
        label: count / len(y[split_indices])
        for label, count in zip(split_labels, split_counts, strict=True)
    }

    # You want the distributions to be similar. We get the proportion of the
    # label in the split labels and subtract from that the proportion in the
    # original y labels. This tells us how far off it is. We square it to
    # rid of negatives and to penalise more extreme differences.
    return sum(
        (split_distribution.get(label, 0.0) - all_distribution.get(label, 0.0)) ** 2
        for label in all_labels
    )


def _score_split(
    y: np.ndarray,
    split_indices: dict[str, np.ndarray],
    target_ratios: dict[str, float],
) -> float:

    # Measure how closely each split's window proportion matches its target ratio.
    # Squaring removes negative differences and penalises larger mismatches more.
    size_error = sum(
        ((len(split_indices[name]) / len(y)) - target_ratios[name]) ** 2
        for name in SPLIT_NAMES
    )

    # Measure how closely each split's class distribution matches the full dataset.
    # Lower error means the split has a more representative mix of labels.
    class_error = sum(
        _class_distribution_error(y, split_indices[name]) for name in SPLIT_NAMES
    )

    # Total score for this train/val/test split candidate. Lower is better.
    return size_error + class_error


def create_patient_splits(
    X: np.ndarray,
    y: np.ndarray,
    patient_ids: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    n_trials: int,
) -> DatasetSplits:
    """
    Create train/validation/test splits without patient leakage.
    Uses Monte Carlo sampling to find the split that best preserves class balance.
    """

    # Checks that patient_ids, X, and y all have the same number of beats
    if X.shape[0] != y.shape[0] or y.shape[0] != patient_ids.shape[0]:
        raise ValueError(
            f"Shape mismatch: X={X.shape[0]}, y={y.shape[0]}, "
            f"patient_ids={patient_ids.shape[0]}"
        )

    # Ensures our splits ratios are valid
    _validate_split_ratios(train_ratio, val_ratio, test_ratio)

    # Gives us the unique patient ids.
    unique_patient_ids = np.unique(patient_ids)

    # Gives a random number generator.
    # Seed allows us to control reproducibility
    rng = np.random.default_rng(seed)

    # These are our desired ratios
    target_ratios = {
        "train": train_ratio,
        "val": val_ratio,
        "test": test_ratio,
    }

    # These will be our final saved values
    best_patient_split: dict[str, list[str]] | None = None
    best_indices: dict[str, np.ndarray] | None = None
    best_score = float("inf")

    logger.info("Trying %d splits", n_trials)

    # We try diffent splits n_trials times, chosing the split that best
    # balances our chosen split ratios and matching as close as possbile
    # to the starting dataset distribution. This is a form of Monte Carlo
    # sampling
    for trial in range(n_trials):

        # Lets us know at every 1000th count. It'll show
        # 0, 1000, 2000 etc.
        if trial % 1000 == 0:
            logger.info("Trial: %d", trial)

        # Split the patient_ids into a train/val/test dict
        # The randomness comes from rng.permutation in this function.
        # This selects a new random split on each run,
        # which is later scored.
        patient_split = _split_patient_ids(
            unique_patient_ids=unique_patient_ids,
            rng=rng,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
        )

        # This is a dictionary comprehension. It builds the dictionary with
        # the for loop. This gives a dictionary of the row indicies in X and y
        # for each train/val/test split
        split_indices = {
            name: _indices_for_patients(patient_ids, patient_split[name])
            for name in SPLIT_NAMES
        }

        score = _score_split(y, split_indices, target_ratios)

        # The score has been minimised, so it is the new best
        if score < best_score:
            best_score = score
            # This split that resulted in the lowest score,
            # so it is the best split
            best_patient_split = patient_split
            # The indicies that gave this best split
            best_indices = split_indices

    logger.info("Trial: %d", n_trials)

    if best_patient_split is None or best_indices is None:
        raise ValueError("Unable to split dataset")

    # Local helper to build the SplitData type
    def _build_split(name: str) -> SplitData:
        indices = best_indices[name]

        return {
            "X": X[indices],
            "y": y[indices],
            "patient_ids": patient_ids[indices],
            "indices": indices,
            "selected_patient_ids": best_patient_split[name],
        }

    logger.info("Splitting complete")

    # This is what DatasetSplits want
    return {
        "train": _build_split("train"),
        "val": _build_split("val"),
        "test": _build_split("test"),
    }


def _class_summary(
    y: np.ndarray, all_labels: np.ndarray
) -> dict[str, dict[str, int | float]]:
    # Counts how many times each label appears
    label_counts = Counter(y.tolist())

    if all_labels is None:
        all_labels = np.array(sorted(label_counts))

    # Gives a new dictionary that returns the count
    # of each label, and 0 if it is not there.
    counts = {label: int(label_counts.get(label, 0)) for label in all_labels}

    # Get the proportion of each class label
    y_proportion_dict = {
        label: np.round(count / len(y), 4) for label, count in counts.items()
    }

    return {
        # Sort the labels by their count
        "class_counts": dict(
            sorted(
                counts.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
        # Sort the labels by their proportions
        "class_proportions": dict(
            sorted(y_proportion_dict.items(), key=lambda item: item[1], reverse=True)
        ),
    }


def _split_summary(
    split: SplitData,
    total_beats: int,
    all_labels: np.ndarray,
) -> dict[str, object]:

    # Returns class counts and proportions
    class_summary = _class_summary(split["y"], all_labels)

    return {
        "number_of_beats": int(split["X"].shape[0]),
        "num_patients": int(len(split["selected_patient_ids"])),
        "obtained_ratio": round(int(split["X"].shape[0]) / total_beats, 2),
        "patient_ids": split["selected_patient_ids"],
        "class_counts": class_summary["class_counts"],
        "class_proportions": class_summary["class_proportions"],
    }


def save_splits(
    splits: DatasetSplits,
    output_dir: Path = Path("data/splits"),
    seed: int = 42,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    n_trials: int = 50000
) -> None:
    """
    Save train/val/test arrays and the exact original row indices.
    """
    # Create directory if it is not already there.
    output_dir.mkdir(parents=True, exist_ok=True)

    # For each split
    for split_name in SPLIT_NAMES:
        # Specify path and create the directory
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        # Save the np.ndarray types
        np.save(split_dir / "X.npy", splits[split_name]["X"])
        np.save(split_dir / "y.npy", splits[split_name]["y"])
        np.save(split_dir / "patient_ids.npy", splits[split_name]["patient_ids"])

    # np.savez saves multiple NumPy arrays into one file.
    # When we load it later, we can do
    # loaded = np.load(output_dir / "split_indices.npz")
    # train_indices = loaded["train_indices"] to get the train indices, etc.
    # The names become the keys we use later.
    np.savez(
        output_dir / "split_indices.npz",
        train_indices=splits["train"]["indices"],
        val_indices=splits["val"]["indices"],
        test_indices=splits["test"]["indices"],
    )

    #
    all_y = np.concatenate([splits[name]["y"] for name in SPLIT_NAMES])
    all_labels = np.unique(all_y)
    total_beats = len(all_y)

    dataset_class_summary = _class_summary(all_y, all_labels)

    # Overall summary and summary of each train/val/test split
    summary = {
        "seed": seed,
        "n_trials": n_trials,
        "ratios": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
        },
        "true_dataset_counts": dataset_class_summary["class_counts"],
        "true_dataset_distribution": dataset_class_summary["class_proportions"],
        "splits": {
            split_name: _split_summary(splits[split_name], total_beats, all_labels)
            for split_name in SPLIT_NAMES
        },
    }

    # Write this summary to a file
    with (output_dir / "split_summary.json").open("w", encoding="utf8") as file:
        json.dump(summary, file, indent=4)

    logger.info("Saved dataset splits to %s", output_dir)


def split_processed_dataset(
    input_dir: Path = Path("data/processed"),
    output_dir: Path = Path("data/splits"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    n_trials: int = 50000,
) -> DatasetSplits:

    logger.info("Loading saved data...")

    # Load saved data
    X, y, patient_ids, record_metadata = load_dataset(index_path=input_dir)

    logger.info("Data loaded")

    # Validate the data
    validate_dataset(X, y, patient_ids, record_metadata)

    logger.info("Data validated, splitting data...")

    # Split the data into train, val, test
    splits = create_patient_splits(
        X=X,
        y=y,
        patient_ids=patient_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        n_trials=n_trials,
    )

    logger.info("Attempting to save data...")

    # save the data to disk
    save_splits(
        splits=splits,
        output_dir=output_dir,
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        n_trials=n_trials
    )

    return splits


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    split_processed_dataset()


if __name__ == "__main__":
    main()
