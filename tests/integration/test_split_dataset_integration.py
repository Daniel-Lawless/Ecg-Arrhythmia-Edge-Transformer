import numpy as np
import pytest

import ecg_arrhythmia.data.split_dataset as split_dataset_module

# Configuration
SPLIT_NAMES = ("train", "val", "test")
SEED = 42
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
N_TRIALS = 100


@pytest.fixture
def small_dataset():
    """Create a tiny dataset with five patients and four classes."""
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

    rr_features = np.array(
        [
            [1.0, 1.0],
            [1.1, 1.0],
            [0.9, 0.9],
            [1.0, 1.1],
        ]
        * 5,
        dtype=float,
    )
    # fmt: on

    return X, y, patient_ids, rr_features


@pytest.mark.integration
def test_split_processed_dataset_loads_validates_splits_and_saves(
    monkeypatch,
    tmp_path,
    small_dataset,
):
    """
    The full split_processed_dataset pipeline should load,
    validate, split, save, and return splits.
    """
    X, y, patient_ids, rr_features = small_dataset
    record_metadata = [{"record_id": "fake_record"}]

    load_called = {"index_path": None}
    validate_called = {"called": False}

    # Replace load_dataset() so this test does not need real data/processed files.
    def fake_load_dataset(index_path):
        load_called["index_path"] = index_path
        return X, y, patient_ids, rr_features, record_metadata

    # Replace validate_dataset() so we can check it was called with the loaded data.
    def fake_validate_dataset(
        loaded_X,
        loaded_y,
        loaded_patient_ids,
        loaded_rr_features,
        loaded_record_metadata,
    ):
        validate_called["called"] = True

        np.testing.assert_array_equal(loaded_X, X)
        np.testing.assert_array_equal(loaded_y, y)
        np.testing.assert_array_equal(loaded_patient_ids, patient_ids)
        np.testing.assert_array_equal(loaded_rr_features, rr_features)
        assert loaded_record_metadata == record_metadata

    monkeypatch.setattr(
        split_dataset_module,
        "load_dataset",
        fake_load_dataset,
    )
    monkeypatch.setattr(
        split_dataset_module,
        "validate_dataset",
        fake_validate_dataset,
    )

    input_dir = tmp_path / "processed"
    output_dir = tmp_path / "splits"

    splits = split_dataset_module.split_processed_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=SEED,
        n_trials=N_TRIALS,
    )

    # The top-level pipeline should call the loader with the requested input path.
    assert load_called["index_path"] == input_dir

    # The top-level pipeline should validate the loaded dataset before splitting.
    assert validate_called["called"]

    # The pipeline should return all three splits.
    assert set(splits.keys()) == set(SPLIT_NAMES)

    # The pipeline should save the expected output files.
    for split_name in SPLIT_NAMES:
        assert (output_dir / split_name / "X.npy").exists()
        assert (output_dir / split_name / "y.npy").exists()
        assert (output_dir / split_name / "patient_ids.npy").exists()
        assert (output_dir / split_name / "rr_features.npy").exists()

    assert (output_dir / "split_indices.npz").exists()
    assert (output_dir / "split_summary.json").exists()
