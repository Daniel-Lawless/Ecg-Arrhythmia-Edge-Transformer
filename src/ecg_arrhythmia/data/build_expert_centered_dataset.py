from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ecg_arrhythmia.data.build_dataset import EXCLUDED_AAMI_LABELS, get_patient_id
from ecg_arrhythmia.data.label_mapping import map_labels_to_aami
from ecg_arrhythmia.data.load_record import load_record, select_signal_channel
from ecg_arrhythmia.data.sequence_dataset import create_record_sequences
from ecg_arrhythmia.preprocessing.beat_extraction import (
    BEAT_SYMBOLS,
    SAMPLES_AFTER,
    SAMPLES_BEFORE,
    extract_beats,
)

logger = logging.getLogger(__name__)

SEQUENCE_LENGTH = 5


@dataclass(frozen=True)
class ExpertRecordBeats:
    """Expert-centred beats for one record, with per-beat identity."""

    record_id: str
    patient_id: str
    windows: NDArray[np.float64]
    rr_features: NDArray[np.float64]
    aami_labels: NDArray[np.str_]
    annotation_samples: NDArray[np.int64]
    annotation_indices: NDArray[np.int64]
    symbols: NDArray[np.str_]

    @property
    def num_beats(self) -> int:
        return int(self.windows.shape[0])


@dataclass(frozen=True)
class ExpertCenteredDataset:
    """
    Expert-centred sequence dataset with a stable identity for each
    scored target beat.

    The sequences reproduce the saved matched split deterministically
    (same beat extraction, AAMI mapping, Q exclusion, and causal
    sequencing), while additionally recording the expert annotation that
    each target beat is centred on.
    """

    X_sequences: NDArray[np.float64]
    rr_sequences: NDArray[np.float64]
    y_labels: NDArray[np.str_]
    patient_ids: NDArray[np.str_]
    target_indices: NDArray[np.int64]

    # Stable per-target identity.
    target_records: NDArray[np.str_]
    target_annotation_samples: NDArray[np.int64]
    target_annotation_indices: NDArray[np.int64]


def build_expert_record_beats(
    record_name: str,
    normalise_beats: bool,
    excluded_labels: set[str],
) -> ExpertRecordBeats:
    """
    Build expert-centred beats for one record, matching the dataset
    builder while recording each beat's expert annotation identity.

    Beats are extracted around expert heartbeat annotations, mapped to
    AAMI classes, and excluded classes (Q by default) are dropped, just
    like ``build_dataset``.
    """

    signals, fields, annotation = load_record(record_name)
    signal, _ = select_signal_channel(signals=signals, fields=fields)
    signal_length = len(signal)

    if annotation.symbol is None:
        raise ValueError(f"Record {record_name} contains no annotation symbols.")

    heartbeat_annotations = [
        (int(sample), symbol)
        for sample, symbol in zip(
            annotation.sample,
            annotation.symbol,
            strict=True,
        )
        if symbol in BEAT_SYMBOLS
    ]

    annotation_samples = np.asarray(
        [sample for sample, _ in heartbeat_annotations],
        dtype=np.int64,
    )
    annotation_symbols = [symbol for _, symbol in heartbeat_annotations]

    windows, extracted_symbols, rr_features = extract_beats(
        signal=signal,
        annotation_samples=annotation_samples,
        annotation_symbols=annotation_symbols,
        normalise=normalise_beats,
    )

    # Recreate extract_beats' emit decision: the first heartbeat is the RR
    # seed and is dropped; the rest are kept only when the full window lies
    # inside the signal.
    within_bounds = (annotation_samples - SAMPLES_BEFORE >= 0) & (
        annotation_samples + SAMPLES_AFTER <= signal_length
    )
    emit_mask = within_bounds.copy()
    emit_mask[0] = False
    emitted_indices = np.nonzero(emit_mask)[0]

    if emitted_indices.size != windows.shape[0]:
        raise ValueError(
            f"Record {record_name}: emitted-beat mismatch between "
            f"extract_beats ({windows.shape[0]}) and the reconstructed "
            f"annotation mapping ({emitted_indices.size})."
        )

    emitted_symbols = np.asarray(annotation_symbols, dtype=str)[emitted_indices]

    # Sanity check the mapping against the symbols extract_beats emitted.
    if not np.array_equal(
        emitted_symbols,
        np.asarray(extracted_symbols, dtype=str),
    ):
        raise ValueError(
            f"Record {record_name}: reconstructed emitted symbols do not "
            "match extract_beats output."
        )

    aami_labels = map_labels_to_aami(emitted_symbols)

    # Drop excluded AAMI classes (Q by default), matching build_dataset.
    keep_mask = ~np.isin(aami_labels, list(excluded_labels))

    return ExpertRecordBeats(
        record_id=record_name,
        patient_id=get_patient_id(record_name),
        windows=windows[keep_mask],
        rr_features=rr_features[keep_mask],
        aami_labels=aami_labels[keep_mask],
        annotation_samples=annotation_samples[emitted_indices][keep_mask],
        annotation_indices=emitted_indices[keep_mask].astype(np.int64),
        symbols=emitted_symbols[keep_mask],
    )


def build_expert_centered_dataset(
    record_names: list[str],
    normalise_beats: bool = False,
    excluded_labels: set[str] | None = None,
    sequence_length: int = SEQUENCE_LENGTH,
) -> ExpertCenteredDataset:
    """
    Build expert-centred sequences with per-target identity for a set of
    records, reproducing the saved matched split.
    """

    if not record_names:
        raise ValueError("At least one record name must be supplied.")

    if excluded_labels is None:
        excluded_labels = set(EXCLUDED_AAMI_LABELS)

    record_beats = [
        build_expert_record_beats(
            record_name=record_name,
            normalise_beats=normalise_beats,
            excluded_labels=excluded_labels,
        )
        for record_name in record_names
    ]

    windows_chunks: list[NDArray[np.float64]] = []
    rr_chunks: list[NDArray[np.float64]] = []
    label_chunks: list[NDArray[np.str_]] = []
    patient_id_values: list[str] = []
    annotation_sample_chunks: list[NDArray[np.int64]] = []
    annotation_index_chunks: list[NDArray[np.int64]] = []
    record_chunks: list[NDArray[np.str_]] = []
    record_segments: list[dict[str, object]] = []

    current_start = 0
    for beats in record_beats:
        num_beats = beats.num_beats

        if num_beats == 0:
            raise ValueError(
                f"Record {beats.record_id} produced no expert-centred beats."
            )

        windows_chunks.append(beats.windows)
        rr_chunks.append(beats.rr_features)
        label_chunks.append(beats.aami_labels)
        patient_id_values.extend([beats.patient_id] * num_beats)
        annotation_sample_chunks.append(beats.annotation_samples)
        annotation_index_chunks.append(beats.annotation_indices)
        record_chunks.append(np.full(num_beats, beats.record_id, dtype=object))

        end_index = current_start + num_beats
        record_segments.append(
            {
                "record_id": beats.record_id,
                "patient_id": beats.patient_id,
                "start_index": current_start,
                "end_index": end_index,
                "num_beats": num_beats,
            }
        )
        current_start = end_index

    windows_all = np.vstack(windows_chunks)
    rr_all = np.vstack(rr_chunks)
    labels_all = np.concatenate(label_chunks)
    patient_ids_all = np.asarray(patient_id_values, dtype=str)
    annotation_samples_all = np.concatenate(annotation_sample_chunks)
    annotation_indices_all = np.concatenate(annotation_index_chunks)
    records_all = np.concatenate(record_chunks).astype(str)

    (
        X_sequences,
        y_sequences,
        rr_sequences,
        sequence_patient_ids,
        target_indices,
        _,
    ) = create_record_sequences(
        X=windows_all,
        y=labels_all,
        rr_features=rr_all,
        patient_ids=patient_ids_all,
        record_metadata=record_segments,
        sequence_length=sequence_length,
    )

    return ExpertCenteredDataset(
        X_sequences=X_sequences,
        rr_sequences=rr_sequences,
        y_labels=y_sequences.astype(str),
        patient_ids=sequence_patient_ids.astype(str),
        target_indices=target_indices.astype(np.int64),
        target_records=records_all[target_indices],
        target_annotation_samples=annotation_samples_all[target_indices],
        target_annotation_indices=annotation_indices_all[target_indices],
    )


def verify_matches_saved_split(
    dataset: ExpertCenteredDataset,
    saved_split_dir: Path,
) -> None:
    """
    Confirm the rebuilt expert-centred dataset reproduces the saved
    matched split at record and class granularity.

    This proves the rebuilt sequences (which carry identity) are the same
    sequences as the original expert-centred validation split.
    """

    saved_labels = np.load(saved_split_dir / "y.npy").astype(str)
    saved_patient_ids = np.load(saved_split_dir / "patient_ids.npy").astype(str)

    if dataset.y_labels.shape[0] != saved_labels.shape[0]:
        raise ValueError(
            "Rebuilt expert dataset has a different sequence count than the "
            f"saved split: rebuilt={dataset.y_labels.shape[0]}, "
            f"saved={saved_labels.shape[0]}."
        )

    rebuilt_label_counts = Counter(dataset.y_labels.tolist())
    saved_label_counts = Counter(saved_labels.tolist())
    if rebuilt_label_counts != saved_label_counts:
        raise ValueError(
            "Rebuilt expert class supports differ from the saved split: "
            f"rebuilt={dict(rebuilt_label_counts)}, "
            f"saved={dict(saved_label_counts)}."
        )

    rebuilt_patient_counts = Counter(dataset.patient_ids.tolist())
    saved_patient_counts = Counter(saved_patient_ids.tolist())
    if rebuilt_patient_counts != saved_patient_counts:
        raise ValueError(
            "Rebuilt expert per-patient counts differ from the saved split: "
            f"rebuilt={dict(rebuilt_patient_counts)}, "
            f"saved={dict(saved_patient_counts)}."
        )

    logger.info(
        "Expert rebuild verified against saved split: %d sequences.",
        dataset.y_labels.shape[0],
    )
