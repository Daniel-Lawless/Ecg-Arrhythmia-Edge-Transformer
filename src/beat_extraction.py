import numpy as np

# Recognised heartbeat annotations. This will exclude non heart beat annotations 
BEAT_SYMBOLS = {
    "N", "L", "R", "e",
    "j", "A", "a", "J", 
    "S", "V", "E", "F",
    "Q", "?", "/", "f",
}

# Start and end of the window.
SAMPLES_BEFORE = 90
SAMPLES_AFTER = 150

# This will give us a 240 sample window.
WINDOW_SIZE = SAMPLES_BEFORE + SAMPLES_AFTER

# Extract beats and labels for a given signal.
def extract_beats(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    extracted_beats = []
    extracted_labels = []

    # Creates a 240 sample window around each annotation index.
    # This will form each datapoint. The corresponding label for that
    # window will be its y value.
    for sample_index, symbol in zip(
        annotation_samples,
        annotation_symbols
    ):
        # Ignore annotations that do not represent heartbeats
        if symbol not in BEAT_SYMBOLS:
            continue

        start = sample_index - SAMPLES_BEFORE
        end = sample_index + SAMPLES_AFTER

        # Skip beats that cannot produce a complete 240-sample window
        if start < 0 or end > len(signal):
            continue

        # 240 sample beat
        beat = signal[start:end]

        # Append window and corresponding symbol.
        extracted_beats.append(beat)
        extracted_labels.append(symbol)

    # Converts the list of labels into a numpy array
    labels = np.array(extracted_labels)

    if len(extracted_beats) == 0:
        return (
            # Empty numpy array with 0 rows and 240 columns. It will take values
            # that has the same data type as signal.
            np.empty((0, WINDOW_SIZE), dtype=signal.dtype),
            # Empty numpy array for labels
            np.array([], dtype=str),
        )

    # Stacks the list of numpy arrays into a 2d matrix.
    beats_matrix = np.vstack(extracted_beats)

    return beats_matrix, labels
