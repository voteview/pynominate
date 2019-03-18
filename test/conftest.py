import pytest
import copy


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
