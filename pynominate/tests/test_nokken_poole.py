import csv
import json
import StringIO
import numpy as np


import pytest
from pdb import set_trace as st

from pynominate import nokken_poole


@pytest.fixture
def payload():
    with open('pynominate/tests/data/payload.json') as f:
        data = json.load(f)
    return data


@pytest.fixture
def nokken_poole_output_file():
    with open('pynominate/tests/data/nokken_poole.csv') as f:
        data = f.readlines()
    return data


def test_merge_dicts():
    first = {'a': 1, 'b': 2}
    second = {'c': 3, 'd': 4}
    third = {'e': 5, 'f': 6}
    expected = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    result = nokken_poole.merge_dicts(first, second, third)
    assert result == expected


def test_nokken_poole_member_estimate(payload):
    member_estimates = nokken_poole.nokken_poole(payload)
    member_estimate = member_estimates[0]
    expected = {
        'chamber': u'S',
        'startx': [0.323, -0.183],
        'nvotes': 607,
        'icpsr': u'40304',
        'cong': u'111',
        u'x': np.array([0.28298301516582103, -0.23682349897588367]),
        u'llend': 151.01943582140623,
        u'llstart': 157.2711841651132,
    }
    member_estimate[u'x'] = member_estimate[u'x'].tolist()
    expected[u'x'] = expected[u'x'].tolist()
    assert member_estimate == expected


def test_make_member_congress_votes(payload):
    result = nokken_poole.make_member_congress_votes(payload)
    expected = nokken_poole.member_congress_votes(payload)
    assert result == expected


@pytest.mark.skip
def test_integration_to_file(payload, nokken_poole_output_file):
    output_file = StringIO.StringIO()
    estimates = nokken_poole.nokken_poole(payload)

    nokken_poole.np_write_csv(estimates, output_file)
    assert output_file.readlines() == nokken_poole_output_file
