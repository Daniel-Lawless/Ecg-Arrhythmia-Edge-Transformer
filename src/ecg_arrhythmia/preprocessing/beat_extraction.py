import numpy as np

# Recognised heartbeat annotations. This will exclude non heart beat annotations
# fmt: off
BEAT_SYMBOLS = {
    "N", "L", "R", "e",
    "j", "A", "a", "J", 
    "S", "V", "E", "F",
    "Q", "?", "/", "f",
}
# fmt: on

# Start and end of the window.
SAMPLES_BEFORE = 90
SAMPLES_AFTER = 150

# This will give us a 240 sample window.
WINDOW_SIZE = SAMPLES_BEFORE + SAMPLES_AFTER

# How many amplitudes values are counted a second
SAMPLING_RATE = 360

LOCAL_RR_WINDOW = 10


# Extract beats and labels for a given signal.
def extract_beats(
    signal: np.ndarray,
    annotation_samples: np.ndarray,
    annotation_symbols: list[str],
    normalise: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    extracted_beats = []
    extracted_labels = []
    rr_features = []

    previous_heartbeat_sample: int | None = None
    recent_rr_seconds: list[float] = []

    # Creates a 240 sample window around each annotation index.
    # This will form each datapoint. The corresponding label for that
    # window will be its y value. stict=True means they must have the same length
    for sample_index, symbol in zip(
        annotation_samples, annotation_symbols, strict=True
    ):
        # Ignore annotations that do not represent heartbeats
        if symbol not in BEAT_SYMBOLS:
            continue

        # We cannot calculate a previous RR interval for the first beat
        if previous_heartbeat_sample is None:
            previous_heartbeat_sample = sample_index
            continue

        # Define samples before and after beat
        start = sample_index - SAMPLES_BEFORE
        end = sample_index + SAMPLES_AFTER

        # Calculate time between each beat
        rr_samples = sample_index - previous_heartbeat_sample
        prev_rr_seconds = rr_samples / SAMPLING_RATE

        previous_heartbeat_sample = sample_index

        # Skip beats that cannot produce a complete 240-sample window
        if start < 0 or end > len(signal):
            continue

        # 240 sample beat
        beat = signal[start:end]

        # Normalise the beat
        if normalise:
            beat = (beat - beat.mean()) / (beat.std() + 1e-8)

        if recent_rr_seconds:
            # Calculates the mean of the most recent LOCAL_RR_WINDOW time intervals
            local_mean_rr = np.mean(recent_rr_seconds[-LOCAL_RR_WINDOW:])
        else:
            # If recent_rr_seconds is empty, just use this intervals time as the mean
            local_mean_rr = prev_rr_seconds

        # Compare the current RR interval against the recent rhythm for this record.
        # local_mean_rr is the average RR interval from recent previous beats.
        # Example: if local_mean_rr = 0.80s and prev_rr_seconds = 0.50s,
        # then rr_ratio = 0.50 / 0.80 = 0.625.
        # rr_ratio < 1 means the beat arrived earlier than expected.
        # rr_ratio ~= 1 means the beat arrived around the expected time.
        # rr_ratio > 1 means the beat arrived later than expected.
        # This is useful for S beats because they can look similar to normal beats,
        # but are often premature, so timing can help distinguish them from N.
        rr_ratio = prev_rr_seconds / (local_mean_rr + 1e-8)

        # Append window and corresponding symbol.
        extracted_beats.append(beat)
        extracted_labels.append(symbol)
        rr_features.append([prev_rr_seconds, rr_ratio])

        # Append the time to the last beat to recent_rr_seconds
        recent_rr_seconds.append(prev_rr_seconds)

    # Converts the list of labels into a numpy array
    labels = np.array(extracted_labels)

    if len(extracted_beats) == 0:
        return (
            # Empty numpy array with 0 rows and 240 columns. It will take values
            # that has the same data type as signal.
            np.empty((0, WINDOW_SIZE), dtype=signal.dtype),
            # Empty numpy array for labels
            np.array([], dtype=str),
            # Empty numpy array for rr_features
            np.empty((0, 2), dtype=np.float32),
        )

    # Stacks the list of numpy arrays into a 2d matrix.
    beats_matrix = np.vstack(extracted_beats)
    rr_features = np.array(rr_features)

    return beats_matrix, labels, rr_features
