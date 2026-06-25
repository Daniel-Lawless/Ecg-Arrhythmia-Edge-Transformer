from src.data.load_record import load_record
import pytest

# These are integration tests since they rely on
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

    # This should return a FileNotFoundError.
    with pytest.raises(FileNotFoundError):
        load_record(record_name=record_name)
