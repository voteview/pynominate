import pytest
import copy


@pytest.fixture
def int_payload():
    payload = {}
    payload["memberwise"] = [
        {"icpsr": 1,
         "update": 1,
         "votes": [[1, "MH1150001"], [-1, "MH1150002"], [1, "MH1160001"], [-1, "MH1170001"]]},
        {"icpsr": 2,
         "update": 1,
         "votes": [[-1, "MH1150001"], [1, "MH1150002"], [-1, "MH1160001"]]},
        {"icpsr": 3,
         "update": 1,
         "votes": [[1, "MH1170001"]]}
    ]

    payload["idpt"] = {
        1: [-0.511, -0.512],
        2: [0.521, 0.522],
        3: [0.031, 0.032]
    }

    payload["bp"] = {
        "MH1150001": [0.02, -0.02, -0.8, -0.5],
        "MH1150002": [-0.02, 0.02, 0.8, 0.5],
        "MH1160001": [-0.02, 0.02, 0.8, 0.5],
        "MH1170001": [-0.02, 0.02, 0.8, 0.5]
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
    str_payload["icpsr"] = {
        str(k): v for k, v in str_payload["idpt"].iteritems()
    }
    return str_payload
