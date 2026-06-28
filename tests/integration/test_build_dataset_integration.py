import json
from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data.build_dataset import build_dataset


@pytest.mark.integration
def test_build_dataset_with_real_mitdb_records(tmp_path):
    X, y, patient_ids, record_segments = build_dataset(
        record_names=["100", "101"],
        output_dir=Path(tmp_path),
    )

    # Matrix should be 2d with more than 1 window
    # with windows of length 240
    assert X.ndim == 2
    assert X.shape[1] == 240
    assert X.shape[0] > 0

    # Each window, X.shape[0], should have a
    # corresponding label
    assert y.shape == (X.shape[0],)

    # Each window, X.sha[0], should have a
    # corresponding patient_id
    assert patient_ids.shape == (X.shape[0],)

    # We input 2 records, so we should get the metadata
    # for 2 records.
    assert len(record_segments) == 2

    # Get the metadata for both records
    first_segment = record_segments[0]
    second_segment = record_segments[1]

    # first segment should have metadata for record 100,
    # second segment should have metadata for record 101
    assert first_segment["record_id"] == "100"
    assert second_segment["record_id"] == "101"

    # first segment should start at index 0, second segment should
    # start at the end of the first window.
    assert first_segment["start_index"] == 0
    assert first_segment["end_index"] == second_segment["start_index"]

    # Last index will be the total number of windows.
    assert second_segment["end_index"] == X.shape[0]

    # The size of the window should be the end index - start index
    assert first_segment["num_beats"] == (
        first_segment["end_index"] - first_segment["start_index"]
    )
    assert second_segment["num_beats"] == (
        second_segment["end_index"] - second_segment["start_index"]
    )

    # Every beat belonging to a record should have that record's patient ID.
    for record in record_segments:
        # Window for a given record
        start = record["start_index"]
        end = record["end_index"]
        # expected patient_id for each beat in this given record
        expected_patient_id = record["patient_id"]

        # Check each beat in this window has this patient_id
        assert np.all(patient_ids[start:end] == expected_patient_id)

    # These files should be created
    assert (tmp_path / "X.npy").exists()
    assert (tmp_path / "y.npy").exists()
    assert (tmp_path / "patient_ids.npy").exists
    assert (tmp_path / "record_segments.json").exists()

    # Load these arrays
    saved_X = np.load(tmp_path / "X.npy")
    saved_y = np.load(tmp_path / "y.npy")
    saved_patient_ids = np.load(tmp_path / "patient_ids.npy")

    # And load the meta data
    with open(tmp_path / "record_segments.json", encoding="utf-8") as file:
        saved_segments = json.load(file)

    # check the loaded files are equal to what we saved
    assert np.array_equal(saved_X, X)
    assert np.array_equal(saved_y, y)
    assert np.array_equal(saved_patient_ids, patient_ids)
    assert saved_segments == record_segments
