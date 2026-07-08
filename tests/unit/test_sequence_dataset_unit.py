from pathlib import Path

import numpy as np
import pytest

from ecg_arrhythmia.data.ecg_sequence_dataset import ECGSequenceDataset
from ecg_arrhythmia.data.sequence_dataset import (
    SequenceSegment,
    create_record_sequences,
    save_sequence_dataset,
)


def test_create_record_sequences_keeps_sequences_inside_each_record():
    # Build a tiny fake beat dataset with 7 beats.
    # Each beat has 4 values so it is easy to see which rows are selected.
    X = np.arange(7 * 4).reshape(7, 4)

    # One label per beat. The target label for each sequence should come
    # from the final beat in tht sequence.
    y = np.array(["N", "S", "V", "F", "N", "V", "S"])

    # One RR feature vector per beat. Each beat has 2 RR features.
    rr_features = np.array(
        [
            [1.0, 1.0],
            [2.0, 1.0],
            [3.0, 1.0],
            [4.0, 1.0],
            [5.0, 1.0],
            [6.0, 1.0],
            [7.0, 1.0],
        ]
    )

    # First 4 beats belong to record/patient 100, last 3 belong to 101.
    patient_ids = np.array(["100", "100", "100", "100", "101", "101", "101"])

    # Metadata tells the sequence builder where each record starts and ends
    # in the full beat array, so windows are not allowed to cross records.
    record_metadata = [
        {
            "record_id": "100",
            "patient_id": "100",
            "lead_name": "MLII",
            "start_index": 0,
            "end_index": 4,
            "num_beats": 4,
        },
        {
            "record_id": "101",
            "patient_id": "101",
            "lead_name": "MLII",
            "start_index": 4,
            "end_index": 7,
            "num_beats": 3,
        },
    ]

    # Build causal sequences of length 3 within each record.
    # Record 100 gives [0,1,2] and [1,2,3].
    # Record 101 gives [4,5,6].
    (
        X_sequences,
        y_targets,
        rr_sequences,
        sequence_patient_ids,
        target_indices,
        segments,
    ) = create_record_sequences(
        X=X,
        y=y,
        rr_features=rr_features,
        patient_ids=patient_ids,
        record_metadata=record_metadata,
        sequence_length=3,
    )

    # There should be 3 total sequences, each containing 3 beats.
    # X has 4 values per beat, while RR has 2 features per beat.
    assert X_sequences.shape == (3, 3, 4)
    assert rr_sequences.shape == (3, 3, 2)

    # Target labels come from the final beat of each sequence:
    # [0,1,2] -> y[2] = V, [1,2,3] -> y[3] = F, [4,5,6] -> y[6] = S.
    assert y_targets.tolist() == ["V", "F", "S"]

    # Each sequence keeps the patient ID of the record it came from.
    assert sequence_patient_ids.tolist() == ["100", "100", "101"]

    # These are the original beat indices of each target beat.
    assert target_indices.tolist() == [2, 3, 6]

    # Check the actual ECG windows selected from X.
    # This confirms the windows slide inside each record and do not cross boundaries.
    np.testing.assert_array_equal(X_sequences[0], X[0:3])
    np.testing.assert_array_equal(X_sequences[1], X[1:4])
    np.testing.assert_array_equal(X_sequences[2], X[4:7])

    # Segments describe where each record's sequences ended up in the final
    # concatenated sequence dataset.
    assert segments == [
        {
            "record_id": "100",
            "patient_id": "100",
            "start_index": 0,
            "end_index": 2,
            "num_sequences": 2,
        },
        {
            "record_id": "101",
            "patient_id": "101",
            "start_index": 2,
            "end_index": 3,
            "num_sequences": 1,
        },
    ]


def test_create_record_sequences_rejects_invalid_sequence_length():
    # Minimal valid-looking dataset. The important part of this test is
    # sequence_length=0, which should be rejected before sequence creation.
    X = np.ones((3, 4))
    y = np.array(["N", "S", "V"])
    rr_features = np.ones((3, 2))
    patient_ids = np.array(["100", "100", "100"])
    record_metadata = [
        {
            "record_id": "100",
            "patient_id": "100",
            "lead_name": "MLII",
            "start_index": 0,
            "end_index": 3,
            "num_beats": 3,
        }
    ]

    # sequence_length must be positive because a sequence with no beats is invalid.
    with pytest.raises(ValueError, match="sequence_length must be greater than 0"):
        create_record_sequences(
            X=X,
            y=y,
            rr_features=rr_features,
            patient_ids=patient_ids,
            record_metadata=record_metadata,
            sequence_length=0,
        )


def test_ecg_sequence_dataset_returns_one_sequence_without_batch_dimension(tmp_path):

    # Create two saved ECG sequences.
    # Shape: (num_sequences=2, sequence_length=3, beat_length=4).
    X_sequences = np.arange(2 * 3 * 4).reshape(2, 3, 4)

    # String labels should be mapped to integer class indices by ECGSequenceDataset.
    y = np.array(["N", "V"])

    # Matching RR sequences.
    # Shape: (num_sequences=2, sequence_length=3, rr_feature_dim=2).
    rr_sequences = np.ones((2, 3, 2))

    # Metadata aligned with the two sequences.
    patient_ids = np.array(["100", "100"])
    target_indices = np.array([2, 3])
    segments: list[SequenceSegment] = [
        {
            "record_id": "100",
            "patient_id": "100",
            "start_index": 0,
            "end_index": 2,
            "num_sequences": 2,
        }
    ]

    # Save the sequence dataset to a temporary folder so the Dataset class
    # can load it the same way it would load a real processed dataset.
    save_sequence_dataset(
        output_dir=Path(tmp_path),
        X_sequences=X_sequences,
        y=y,
        rr_sequences=rr_sequences,
        patient_ids=patient_ids,
        target_indices=target_indices,
        sequence_segments=segments,
        sequence_length=3,
    )

    # Load the saved data through the PyTorch Dataset wrapper.
    dataset = ECGSequenceDataset(Path(tmp_path))

    # __getitem__ should return one sample, not a batch.
    x_tensor, rr_tensor, y_tensor = dataset[1]

    # The dataset length should equal the number of saved sequences.
    assert len(dataset) == 2

    # One sample should have shape (sequence_length, channels, beat_length).
    # The channel dimension is added by ECGSequenceDataset using unsqueeze.
    assert tuple(x_tensor.shape) == (3, 1, 4)

    # RR features should line up with the same 3 beats in the ECG sequence.
    assert tuple(rr_tensor.shape) == (3, 2)

    # Label V should map to class index 2.
    assert y_tensor.item() == 2
