import wfdb
import matplotlib.pyplot as plt
import logging
import numpy as np
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

def load_record(
    record_name: str,
) -> tuple[np.ndarray, dict[str, Any], wfdb.Annotation]:
    """
    Load the ECG signals, metadata, and expert annotations for one
    record from the MIT-BIH Arrhythmia Database.
    """

    # Load both ECG channels for the requested record.
    #
    # signals is a 2D NumPy array with shape:
    # (number_of_samples, number_of_channels)
    #
    # Each row represents one moment in time, each column represents
    # a different way of measuring heart rate activity (ECG lead).
    # MIT-BIH records contain two channels.
    #
    # fields contains metadata such as the sampling frequency and lead names.
    signals, fields = wfdb.rdsamp(
        record_name=record_name,
        pn_dir="mitdb",
    )

    # Load the expert annotations associated with the record.
    # annotation.sample contains the ECG sample positions of annotations.
    # annotation.symbol contains the corresponding annotation symbols,
    # such as "N" for a normal beat or "V" for a premature ventricular beat.
    annotation = wfdb.rdann(
        record_name=record_name,
        extension="atr",
        pn_dir="mitdb",
    )

    # Signals can be both a numpy array or None.
    if signals is None:
        raise ValueError("No signal found")

    # MIT-BIH should return two-dimensional signal data. 
    if signals.ndim != 2:
        raise ValueError(
            f"Expected a 2D signal array, received shape {signals.shape}"
        )

    # Annotation.symbol is allowed to be None by the WFDB Annotation class.
    # We need the symbols because they will become our labels later on.
    if annotation.symbol is None:
        raise ValueError(
            f"Record {record_name} contains no annotation symbols"
        )

    # Log useful information for checking that the record loaded correctly.
    logger.debug("Record: %s", record_name)
    logger.debug("Signal shape: %s", signals.shape)
    logger.debug("Sampling frequency: %s", fields["fs"])
    logger.debug("Signal names: %s", fields["sig_name"])
    logger.debug("First annotation samples: %s", annotation.sample[:10])
    logger.debug("First annotation symbols: %s", annotation.symbol[:10])

    return signals, fields, annotation

def select_signal_channel(
    signals: np.ndarray,
    fields: dict[str, Any],
    preferred_lead: str = "MLII",
) -> tuple[np.ndarray, str]:
    """
    Select one ECG channel from the two channels stored in a record.

    MLII is preferred because it is available in most MIT-BIH records.
    If it is unavailable, the first channel is used instead.
    """

    # The lead names are stored in the same order as the signal columns.
    # For example if fields["sig_name"] == ["MLII", "V5"] then
    # signals[:, 0] is therefore MLII and signals[:, 1] is V5.
    signal_names = fields["sig_name"]

    # Find the column containing the preferred lead.
    if preferred_lead in signal_names:
        channel_index = signal_names.index(preferred_lead)

    # Some records, such as 102 and 104, do not contain MLII.
    # In that case, just use the first available channel.
    else:
        channel_index = 0

        logger.warning(
            "%s is unavailable; using %s instead",
            preferred_lead,
            signal_names[channel_index],
        )

    # Extract every ECG amplitude measurement from the selected channel.
    # This converts the 2D signal matrix into a 1D signal array.
    signal = signals[:, channel_index]

    # Store the selected lead name so we know how the signal was measured.
    lead_name = signal_names[channel_index]

    return signal, lead_name

def plot_record(
        record_name: str,
        signal: np.ndarray,
        annotation: wfdb.Annotation,
        lead_name: str
        ) -> None:

    # Plot the first 3000 amplitudes/ 3000/360 ≈ 8.3 seconds
    # of ECG recording for this record
    start = 0
    end = 3000

    if annotation.symbol is None:
        raise ValueError("No annotation for this sample.")

    plt.figure(figsize=(12,4))
    plt.plot(signal[start:end], label=f"ECG signal")

    # Gives (sample_index, symbol at that index)
    for sample, symbol in zip(annotation.sample, annotation.symbol):
        # The sample index has to be between the start and end index
        if start <= sample < end:
            # Draw a vertical red line at x position sample - start
            plt.axvline(sample - start, color="red", alpha=0.3)
            # Put symbol at x postion sample - start, and y position signal[sample].
            plt.text(sample - start, float(signal[sample]), symbol, color="green", fontsize=8)

    plt.title(f"MIT-BIH Record {record_name}: {lead_name} Signal with Beat Annotations")
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ecg_plot.png")
    plt.close()
