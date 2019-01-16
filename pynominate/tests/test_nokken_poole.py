import numpy as np
import copy

import pytest

from pynominate import nokken_poole


@pytest.fixture
def int_payload():
    payload = {}
    payload["memberwise"] = [
        {"icpsr": 1,
         "update": 1,
         "votes": [[1, "MH1150001"], [-1, "MH1150002"]]},
        {"icpsr": 2,
         "update": 1,
         "votes": [[-1, "MH1150001"], [1, "MH1150002"]]},
    ]

    payload["idpt"] = {
        1: [-0.5, -0.5],
        2: [0.5, 0.5],
    }

    payload["bp"] = {
        "MH1150001": [0.02, -0.02, -0.8, -0.5],
        "MH1150002": [-0.02, 0.02, 0.8, 0.5],
    }
    return payload


@pytest.fixture
def broken_payload(int_payload):
    """broken_payload has inconsistent icpsr types in "idpt" and "memberwise" keys"""
    broken_payload = copy.deepcopy(int_payload)
    broken_payload["idpt"] = {
        str(k): v for k, v in broken_payload["idpt"].iteritems()
    }
    return broken_payload


@pytest.fixture
def str_payload(broken_payload):
    """fixes broken_payload by making memberwise icpsrs also strings"""
    str_payload = copy.deepcopy(broken_payload)
    str_payload["memberwise"] = [
        {k: str(v) if k == "icpsr" else v for k, v in m.iteritems()}
        for m in str_payload["memberwise"]
    ]
    return str_payload


def test_broken_payload_breaks(broken_payload):
    with pytest.raises(KeyError):
        nokken_poole.nokken_poole(broken_payload)


def test_make_member_congress_votes(int_payload, str_payload):
    for payload_type, payload in {"int": int_payload, "str": str_payload}.iteritems():
        expected_bp = np.transpose(np.array([
            [0.02, -0.02, -0.8, -0.5],
            [-0.02, 0.02, 0.8, 0.5],
        ]))
        expected_dat = {
            "data": [
                {"votes": np.array([1, -1]), "bp": expected_bp},
                {"votes": np.array([-1, 1]), "bp": expected_bp},
            ],
            "start": [[-0.5, -0.5], [0.5, 0.5]],
            "icpsr_chamber_congress": [
                {
                    "icpsr": 1 if payload_type == "int" else "1",
                    "chamber": "H",
                    "cong": "115",
                    "nvotes": 2,
                },
                {
                    "icpsr": 2 if payload_type == "int" else "2",
                    "chamber": "H",
                    "cong": "115",
                    "nvotes": 2,
                },
            ]
        }

        dat = nokken_poole.make_member_congress_votes(payload)

        for k, v in dat.iteritems():
            if k == "data":
                for i, data in enumerate(v):
                    np.testing.assert_array_equal(data["votes"], expected_dat["data"][i]["votes"])
                    np.testing.assert_array_equal(data["bp"], expected_dat["data"][i]["bp"])
            else:
                assert dat[k] == expected_dat[k]
              

def test_nokken_poole_member_estimate(int_payload, str_payload):
    for payload_type, payload in {"int": int_payload, "str": str_payload}.iteritems():
        member_estimates = nokken_poole.nokken_poole(payload)

        expected = [
            {
                'chamber': 'H',
                'startx': [-0.5, -0.5],
                'nvotes': 2,
                'icpsr': 1 if payload_type == "int" else "1",
                'cong': u'115',
                u'x': [0.8661656699495177, 0.4997562405655863],
                u'llend': 5.580493176682049e-17,
                u'llstart': 51.63337903006439,
            },
            {
                'chamber': 'H',
                'startx': [0.5, 0.5],
                'nvotes': 2,
                'icpsr': 2 if payload_type == "int" else "2",
                'cong': u'115',
                u'x': [-0.8661656699495177, -0.4997562405655863],
                u'llend': 5.580493176682049e-17,
                u'llstart': 51.63337903006439,
            }
        ]
        
        for i, member in enumerate(member_estimates):
            np.testing.assert_array_equal(member.pop("x"), expected[i].pop("x"))
            assert member == expected[i]


def test_merge_dicts():
    first = {'a': 1, 'b': 2}
    second = {'c': 3, 'd': 4}
    third = {'e': 5, 'f': 6}
    expected = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6}
    result = nokken_poole.merge_dicts(first, second, third)
    assert result == expected
