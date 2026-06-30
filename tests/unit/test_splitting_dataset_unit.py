import json

import numpy as np
import pytest

from ecg_arrhythmia.data.split_dataset import (
    _indices_for_patients,
    _split_patient_ids,
    _validate_split_ratios,
    create_patient_splits,
    save_splits,
)

# Configuration
SPLIT_NAMES = ("train", "val", "test")
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
N_TRIALS = 100


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_dataset():
    """Create a tiny dataset with five patients and four classes."""

    # Creats numbers between 0 to 99 and makes is a 2d numpy array with 20 rows
    # and 5 columns
    X = np.arange(20 * 5).reshape(20, 5)

    # fmt: off
    y = np.array(
        [
            "N", "N", "V", "V", "N",
            "S", "S", "V", "N", "N",
            "N", "F", "V", "V", "S",
            "S", "N", "F", "V", "S",
        ]
    )

    patient_ids = np.array(
        [
            "101", "101", "101", "101",
            "102", "102", "102", "102",
            "103", "103", "103", "103",
            "104", "104", "104", "104",
            "105","105", "105", "105",
        ]
    )
    # fmt: on

    return X, y, patient_ids


# fixture is built into pytest and is marks reusable setup code for tests
@pytest.fixture
def small_splits(small_dataset):
    """Create reusable train/val/test splits from the tiny dataset."""
    X, y, patient_ids = small_dataset

    return create_patient_splits(
        X=X,
        y=y,
        patient_ids=patient_ids,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        n_trials=N_TRIALS,
    )


# ---------------------------------------------------------------------------
# Ratio validation tests
# ---------------------------------------------------------------------------


def test_validate_split_ratios_accepts_valid_ratios():
    """Valid train/val/test ratios should pass without raising an error."""
    _validate_split_ratios(
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
    )


# mark.parametrize runs the same test multiple times with different inputs.
# Each tuple below provides one set of values for:
# train_ratio, val_ratio, and test_ratio.
# This avoids writing several near-identical tests.
@pytest.mark.parametrize(
    ("train_ratio", "val_ratio", "test_ratio"),
    [
        # So for the first run train_ratio = 0.0, val_ratio = 0.15, test_ratio = 0.85
        # And so on for the second, third, and fourth run.
        (0.0, 0.15, 0.85),
        (0.7, 0.0, 0.3),
        (0.7, 0.15, 0.0),
        (-0.7, 0.15, 1.55),
    ],
)

# It will run the function directly below it.
def test_validate_split_ratios_rejects_non_positive_ratios(
    train_ratio,  # Must match the parametrize arguments
    val_ratio,
    test_ratio,
):
    """
    Each split ratio must be strictly greater than zero (for the first three)
    And must be approximatley 1 (for the last tuple of values)
    """
    with pytest.raises(ValueError):
        _validate_split_ratios(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )


def test_validate_split_ratios_rejects_ratios_that_do_not_sum_to_one():
    """The three split ratios must sum to approximately one."""
    with pytest.raises(ValueError):
        _validate_split_ratios(
            train_ratio=0.7,
            val_ratio=0.2,
            test_ratio=0.2,
        )


# ---------------------------------------------------------------------------
# Patient ID splitting tests
# ---------------------------------------------------------------------------


def test_split_patient_ids_rejects_less_than_three_patients():
    """At least one patient is needed for each of train, validation, and test."""
    unique_patient_ids = np.array(["101", "102"])
    rng = np.random.default_rng(seed=SEED)

    with pytest.raises(ValueError):
        _split_patient_ids(
            unique_patient_ids=unique_patient_ids,
            rng=rng,
            train_ratio=0.7,
            val_ratio=0.2,
        )


def test_split_patient_ids_assigns_one_patient_to_each_split_when_three_patients():
    """With three patients, each split should receive one patient each."""
    unique_patient_ids = np.array(["101", "102", "103"])
    rng = np.random.default_rng(seed=SEED)

    split_dict = _split_patient_ids(
        unique_patient_ids=unique_patient_ids,
        rng=rng,
        train_ratio=0.7,
        val_ratio=0.2,
    )

    assert len(split_dict["train"]) == 1
    assert len(split_dict["val"]) == 1
    assert len(split_dict["test"]) == 1


def test_split_patient_ids_assigns_every_patient_once_with_no_overlap():
    """Patient IDs should be assigned only once, with no leakage between splits."""
    # fmt: off
    unique_patient_ids = np.array(
        [
            "101", "102", "103",
            "104", "105", "106",
            "107", "108", "109",
        ]
    )
    # fmt: on

    rng = np.random.default_rng(seed=SEED)

    split_dict = _split_patient_ids(
        unique_patient_ids=unique_patient_ids,
        rng=rng,
        train_ratio=0.7,
        val_ratio=0.2,
    )

    # Combined patients
    assigned_patients = split_dict["train"] + split_dict["val"] + split_dict["test"]

    # All sets should be disjoint from one another
    assert set(split_dict["train"]).isdisjoint(set(split_dict["val"]))
    assert set(split_dict["train"]).isdisjoint(set(split_dict["test"]))
    assert set(split_dict["val"]).isdisjoint(set(split_dict["test"]))

    # When all sets are combined it should equal the entire patient_id set.
    assert sorted(assigned_patients) == sorted(unique_patient_ids.tolist())
    assert len(assigned_patients) == len(set(assigned_patients))


# ---------------------------------------------------------------------------
# Beat index selection tests
# ---------------------------------------------------------------------------


def test_indices_for_patients_returns_expected_beat_indices():
    """Selected patient IDs should map back to the correct beat-level row indices."""
    patient_splits = {
        "train": ["101", "105", "107"],
        "val": ["103", "104", "106"],
        "test": ["102", "108", "109"],
    }

    # fmt: off
    patient_ids = np.array(
        [
            "101", "101", "101",
            "102", "102", "103",
            "103", "103", "104",
            "105", "105", "106",
            "106", "107", "107",
            "107", "108", "109",
        ]
    )
    # fmt: on

    # Based on the patient_ids and splits, we expect the following
    # indices
    expected_indices = {
        "train": np.array([0, 1, 2, 9, 10, 13, 14, 15]),
        "val": np.array([5, 6, 7, 8, 11, 12]),
        "test": np.array([3, 4, 16, 17]),
    }

    # tests the indices we return match the expected indices
    for split_name in SPLIT_NAMES:
        indices = _indices_for_patients(
            patient_ids=patient_ids,
            selected_patient_ids=patient_splits[split_name],
        )

        np.testing.assert_array_equal(indices, expected_indices[split_name])


def test_indices_for_patients_returns_empty_array_when_no_patients_match():
    """If no selected patients are present, we should return no indices."""
    patient_ids = np.array(["101", "101", "102", "102"])

    # Patient 999 does not exist.
    indices = _indices_for_patients(
        patient_ids=patient_ids,
        selected_patient_ids=["999"],
    )

    # Hence it should return and empty array
    np.testing.assert_array_equal(indices, np.array([]))


# ---------------------------------------------------------------------------
# Full create_patient_splits tests
# ---------------------------------------------------------------------------


def test_create_patient_splits_rejects_mismatched_lengths():
    """X, y, and patient_ids must have the same number of beats."""
    X = np.zeros((10, 5))
    y = np.array(["N"] * 9)
    patient_ids = np.array(["101"] * 10)

    with pytest.raises(ValueError):
        create_patient_splits(
            X=X,
            y=y,
            patient_ids=patient_ids,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=SEED,
            n_trials=N_TRIALS,
        )


# pytest sees that small_splits is a fixture, goes and run it, and allows
# us to use its results here. The name of the fixture is the access point.
# I.e., it is equivalent to running it as
# small_splits = create_patient_splits(
#       X=X,
#       y=y,
#       ...
# )
def test_create_patient_splits_has_no_patient_leakage(small_splits):
    """The final split should not place the same patient in multiple splits."""
    # These are the selected patient_ids for each split after create_splits().
    # These should be unique
    train_patients = set(small_splits["train"]["selected_patient_ids"])
    val_patients = set(small_splits["val"]["selected_patient_ids"])
    test_patients = set(small_splits["test"]["selected_patient_ids"])

    # They should not overlap.
    assert train_patients.isdisjoint(val_patients)
    assert train_patients.isdisjoint(test_patients)
    assert val_patients.isdisjoint(test_patients)


def test_create_patient_splits_uses_every_beat_once(small_dataset, small_splits):
    """Every beat index should appear exactly once across train, validation, and test."""
    X, _, _ = small_dataset

    # Combines all indices across the splits
    all_indices = np.concatenate(
        [
            small_splits["train"]["indices"],
            small_splits["val"]["indices"],
            small_splits["test"]["indices"],
        ]
    )

    # converts the indices to a list, then sorts the indices in order,
    # then creates a list from 0, 1, ... , X - 1 and checks they are equal.
    assert sorted(all_indices.tolist()) == list(range(len(X)))

    # Ensures there are no duplicate indices
    assert len(all_indices) == len(set(all_indices.tolist()))


def test_create_patient_splits_data_matches_original_indices(
    small_dataset, small_splits
):
    """Split arrays should exactly match the original rows selected by split indices."""
    X, y, patient_ids = small_dataset

    for split_name in SPLIT_NAMES:
        # Indices for each window in the split
        indices = small_splits[split_name]["indices"]

        # Checks the splits X, y, and patient_ids windows match the
        # original rows selected by the split indices
        np.testing.assert_array_equal(small_splits[split_name]["X"], X[indices])
        np.testing.assert_array_equal(small_splits[split_name]["y"], y[indices])
        np.testing.assert_array_equal(
            small_splits[split_name]["patient_ids"],
            patient_ids[indices],
        )


def test_create_patient_splits_is_reproducible_with_same_seed(small_dataset):
    """Using the same seed and trial count should produce the same selected split."""
    X, y, patient_ids = small_dataset

    # 2 splits with the same seed
    split_1 = create_patient_splits(
        X=X,
        y=y,
        patient_ids=patient_ids,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        n_trials=N_TRIALS,
    )

    split_2 = create_patient_splits(
        X=X,
        y=y,
        patient_ids=patient_ids,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        n_trials=N_TRIALS,
    )

    # Should give us the same results
    for split_name in SPLIT_NAMES:
        np.testing.assert_array_equal(
            split_1[split_name]["indices"],
            split_2[split_name]["indices"],
        )
        assert (
            split_1[split_name]["selected_patient_ids"]
            == split_2[split_name]["selected_patient_ids"]
        )


# ---------------------------------------------------------------------------
# Save-to-disk tests
# ---------------------------------------------------------------------------


def test_save_splits_writes_arrays_indices_and_summary(tmp_path, small_splits):
    """Saved files should match the in-memory splits and include useful metadata."""
    save_splits(
        splits=small_splits,
        output_dir=tmp_path,
        seed=SEED,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        n_trials=N_TRIALS,
    )

    # We save the X, y, patient_ids windows,
    # so we should expect to get them vack
    # when we load
    for split_name in SPLIT_NAMES:
        np.testing.assert_array_equal(
            np.load(tmp_path / split_name / "X.npy"),
            small_splits[split_name]["X"],
        )
        np.testing.assert_array_equal(
            np.load(tmp_path / split_name / "y.npy"),
            small_splits[split_name]["y"],
        )
        np.testing.assert_array_equal(
            np.load(tmp_path / split_name / "patient_ids.npy"),
            small_splits[split_name]["patient_ids"],
        )

    loaded_indices = np.load(tmp_path / "split_indices.npz")

    # Similarly, we save the train/val/tests
    # indices, so we should expect them to
    # be returned the same way.
    np.testing.assert_array_equal(
        loaded_indices["train_indices"],
        small_splits["train"]["indices"],
    )
    np.testing.assert_array_equal(
        loaded_indices["val_indices"],
        small_splits["val"]["indices"],
    )
    np.testing.assert_array_equal(
        loaded_indices["test_indices"],
        small_splits["test"]["indices"],
    )

    # Load summary
    summary = json.loads((tmp_path / "split_summary.json").read_text(encoding="utf8"))

    # Check that the layout of summary is what we expect.
    assert summary["seed"] == SEED
    assert summary["n_trials"] == N_TRIALS
    assert summary["ratios"] == {
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
    }
    assert summary["true_dataset_counts"] == {
        "N": 7,
        "V": 6,
        "S": 5,
        "F": 2,
    }

    for split_name in SPLIT_NAMES:
        assert summary["splits"][split_name]["number_of_beats"] == len(
            small_splits[split_name]["y"]
        )
        assert (
            summary["splits"][split_name]["patient_ids"]
            == small_splits[split_name]["selected_patient_ids"]
        )
