import numpy as np
import pytest

from src.label_mapping import map_labels_to_aami


def test_maps_raw_labels_to_aami_classes():
    # Input
    labels = np.array(["N", "L", "R", "A", "a", "J", "S", "V", "E", "F", "/", "f", "Q", "?"])

    # Map the labels accoridng to the AAMI standard 
    mapped_labels = map_labels_to_aami(labels)

    # what we expect to get
    expected = np.array(["N", "N", "N", "S", "S", "S", "S", "V", "V", "F", "Q", "Q", "Q", "Q"])

    assert np.array_equal(mapped_labels, expected)


def test_raises_error_for_unmapped_label():
    # "x" is not a valid annotation that is recognised by AAMI,
    # so there is no mapping for it, and should be rejected
    labels = np.array(["N", "x"])

    with pytest.raises(ValueError, match="Unmapped labels found"):
        map_labels_to_aami(labels)
