"""
Spot checks for 
"""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal
import sys

sys.path.append('../')
from id_file_from_maxQuant import filter_maxQuant

################################ filter_maxQuant ##############################


@pytest.fixture
def evidence_input():
    return pd.read_csv('evidence_test_input.txt', sep='\t')


@pytest.fixture
def evidence_output():
    return pd.read_csv('evidence_test_output.txt', sep='\t')


def test_filter_maxQuant(evidence_input, evidence_output):

    result = filter_maxQuant(evidence_input, ['unlab2', 'unlab3'])

    # We turn ints to floats when we do agg, so ignore dtype
    # We change column order with groupby, so sort to ignore column order
    assert_frame_equal(result.sort_index(axis=1),
                       evidence_output.sort_index(axis=1),
                       check_names=True,
                       check_dtype=False)
