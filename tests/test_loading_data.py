from src.data.load_record import load_record, select_signal_channel
import numpy as np
import pytest

# These are not unit tests but integration tests since they rely on
# internet access and the PhysioNet server being available.
@pytest.mark.integration
def test_loading_a_record():
    record_name = "100"

    signals, fields, annotation = load_record(record_name=record_name)

    # Signals should not be None, should have 2 dimensions, and 2 channels.
    assert signals is not None
    assert signals.ndim == 2
    assert signals.shape[1] == 2

    # symbol and sample lists should not be None. 
    # Each annotation sample should have a corresponding annotation symbol
    # 
    assert annotation.symbol is not None
    assert annotation.sample is not None
    assert len(annotation.sample) == len(annotation.symbol)

    # for mitdb the sample frequency should be 360.
    # sig_name should also be availble for us to extract the 
    # measuring technique.
    assert fields["fs"] == 360
    assert "sig_name" in fields

@pytest.mark.integration
def test_invalid_record_name():
    # mitdb only goes up to 234
    record_name = "300"

    # This should retrun a FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        load_record(record_name=record_name)

# The following are unit tests since they don't rely on externals.

def test_signal_and_lead_name():
    # Fake signals array
    signals = np.array(
        [[1,2],
         [3,4],
         [5,6]]
    )

    # Leads
    fields = {
        "sig_name" : ["MLII", "V5"]
    }

    # select a channel
    signal, lead_name = select_signal_channel(
        signals=signals,
        fields=fields,
        preferred_lead="MLII"
    )

    # Since the preferred channel is MLII, and is avaiable,
    # it should be selected. The corresponding signal then
    # should be the amplitues from the first column
    assert lead_name == "MLII"
    assert np.array_equal(signal, [1,3,5])

def test_preferred_lead_not_available():
    signals = np.array(
        [[1,2],
         [3,4],
         [5,6]]
    )

    # Lead does not include the preferred lead
    fields = {
        "sig_name" : ["V5", "V2"]
    }

    # Select a channel
    signal, lead_name = select_signal_channel(
        signals=signals,
        fields=fields,
        preferred_lead="MLII"
    )

    # Since the preferred channel is not available,
    # it should default to using the first channel
    assert lead_name == "V5" 
    assert np.array_equal(signal, [1, 3, 5])

def test_selects_preferred_lead_from_second_column():
    signals = np.array(
        [[1, 2],
         [3, 4],
         [5, 6]]
    )

    # Lead is in the second column
    fields = {
        "sig_name": ["V5", "MLII"]
    }

    signal, lead_name = select_signal_channel(
        signals=signals,
        fields=fields,
        preferred_lead="MLII"
    )

    # We expect to still use the preferred lead
    # The amplitudes should be taken from the second column
    assert lead_name == "MLII"
    assert np.array_equal(signal, np.array([2, 4, 6]))
