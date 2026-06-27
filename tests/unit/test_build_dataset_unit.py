import numpy as np
from pathlib import Path
from src import build_dataset as build_dataset_module

class FakeAnnotation:
    sample = np.array([100])
    symbol = ["N"]

# record 201 and 202 should be converted to id "201_202",
# any other remains the same.
def test_get_patient_id_groups_records_201_and_202():
    assert build_dataset_module.get_patient_id("201") == "201_202"
    assert build_dataset_module.get_patient_id("202") == "201_202"
    assert build_dataset_module.get_patient_id("100") == "100"


def test_build_dataset_saves_arrays_and_record_segments(tmp_path, monkeypatch):
    # Record 100 has 2 windows, 201 has 1, and 202 has 3.
    beat_counts = {
        "100": 2,
        "201": 1,
        "202": 3,
    }

    # Build fake equivalents of methods used in build_dataset.
    # This makes the test isolated on just the behaviour of build_dataset.
    def fake_load_record(record_name: str):
        signals = np.empty((10, 2))
        fields = {"record_name": record_name}
        annotation = FakeAnnotation()
        return signals, fields, annotation

    def fake_select_signal_channel(signals, fields, preferred_lead="MLII"):
        record_name = fields["record_name"]
        signal = np.array([beat_counts[record_name]])
        lead_name = "MLII"
        return signal, lead_name

    def fake_extract_beats(signal, annotation_samples, annotation_symbols):
        number_of_beats = int(signal[0])
        beats = np.ones((number_of_beats, 240))
        labels = np.array(["N"] * number_of_beats)
        return beats, labels

    # This makes it so when we do build_dataset_module.load_record, or any 
    # other method, it uses our fake equivalent.
    monkeypatch.setattr(
        build_dataset_module, # The file to look in
        "load_record",        # The function to look for
        fake_load_record      # The function to replace it with
    )
    monkeypatch.setattr(
        build_dataset_module,
        "select_signal_channel",
        fake_select_signal_channel,
    )
    monkeypatch.setattr(
        build_dataset_module,
        "extract_beats",
        fake_extract_beats
    )

    # Extract data, patient_ids, and metadata
    X, y, patient_ids, record_segments = build_dataset_module.build_dataset(
        record_names=["100", "201", "202"],
        output_dir= Path(tmp_path),
    )

    # Since the total number of windows was 6, each of length 240
    assert X.shape == (6, 240)
    # Each window had a corresponding label.
    assert y.shape == (6,)
    # Each window also has a patient_id
    assert patient_ids.shape == (6,)

    # Patient_ids should be as follows
    assert patient_ids.tolist() == [
        "100", "100",
        "201_202",
        "201_202", "201_202", "201_202",
    ]

    # Each file must be created
    assert (tmp_path / "X.npy").exists()
    assert (tmp_path / "y.npy").exists()
    assert (tmp_path / "record_segments.json").exists()
    assert (tmp_path / "patient_ids.npy").exists()

    # Check the metadata loads correctly.
    assert record_segments == [
        {
            "record_id": "100",
            "patient_id": "100",
            "lead_name": "MLII",
            "start_index": 0,
            "end_index": 2,
            "num_beats": 2,
        },
        {
            "record_id": "201",
            "patient_id": "201_202",
            "lead_name": "MLII",
            "start_index": 2,
            "end_index": 3,
            "num_beats": 1,
        },
        {
            "record_id": "202",
            "patient_id": "201_202",
            "lead_name": "MLII",
            "start_index": 3,
            "end_index": 6,
            "num_beats": 3,
        },
    ]

def test_build_dataset_can_exclude_records(tmp_path, monkeypatch):

    # Create fake methods
    def fake_load_record(record_name: str):
        signals = np.empty((10, 2))
        fields = {"record_name": record_name}
        annotation = FakeAnnotation()
        return signals, fields, annotation

    def fake_select_signal_channel(signals, fields, preferred_lead="MLII",):
        return np.array([2]), "MLII"

    def fake_extract_beats(signal, annotation_samples, annotation_symbols):
        return np.ones((2, 240)), np.array(["N", "N"])

    # Make is so these fake methods are called instead in build_dataset
    monkeypatch.setattr(
        build_dataset_module,
        "load_record",
        fake_load_record
    )
    monkeypatch.setattr(
        build_dataset_module,
        "select_signal_channel",
        fake_select_signal_channel,
    )
    monkeypatch.setattr(
        build_dataset_module,
        "extract_beats",
        fake_extract_beats
    )

    # Build data and metadata.
    X, y, patient_ids, record_segments = build_dataset_module.build_dataset(
        record_names=["100", "102", "104"],
        output_dir=Path(tmp_path),
        excluded_records=build_dataset_module.PACED_RECORDS,
    )

    # Since record 100 is the only non-paced record, it should
    # be the only one left.
    assert X.shape == (2, 240)
    assert y.shape == (2,)
    assert patient_ids.shape == (2,)
    assert patient_ids.tolist() == ["100", "100"]   

    # Only metadata from record 100 should exist.
    assert len(record_segments) == 1
    assert record_segments[0]["record_id"] == "100"
