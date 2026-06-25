import numpy as np

# Extract beats and labels for a given signal.
def extract_beats(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
) -> tuple[np.ndarray, np.ndarray]:

    # Recognised heartbeat annotations. This will exclude non heart beat annotations 
    beat_symbols = {
    "N", "L", "R", "B",
    "A", "a", "J", "S",
    "V", "r", "F",
    "e", "j", "n", "E",
    "/", "f", "Q", "?"
    }

    # This will give us a 240 sample window.
    samples_before = 90
    samples_after = 150

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
        if symbol not in beat_symbols:
            continue

        # Start and end of the window.
        start = sample_index - samples_before
        end = sample_index + samples_after

        # Skip beats that cannot produce a complete 240-sample window
        if start < 0 or end > len(signal):
            continue

        # 240 sample beat
        beat = signal[start:end]

        # Append window and corresponding symbol.
        extracted_beats.append(beat)
        extracted_labels.append(symbol)

    # Stacks the list of numpy arrays into a 2d matrix.
    beats_matrix = np.vstack(extracted_beats)
    
    # Converts the list of labels into a numpy array
    labels = np.array(extracted_labels)

    return beats_matrix, labels
