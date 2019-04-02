from pprint import pprint
from numpy.random import uniform
from pynominate import payload
from pynominate import nominate

import sys
sys.path.insert(0, "..")

test_payload = {
    'votes': [
        {
            'votes': [[-1, "1"], [1, "2"], [1, "3"]],
            'id': "RS09901",
            'update': 1
        },
        {
            'votes': [[-1, "1"], [1, "2"], [-1, "3"]],
            'id': "RS09902",
            'update': 1
        },
        {
            'votes': [[-1, "1"], [1, "2"], [1, "3"]],
            'id': "RS10001",
            'update': 1
        },
        {
            'votes': [[-1, "1"], [1, "2"], [1, "3"]],
            'id': "RS10002",
            'update': 1
        }
    ],
    'bp': {
        'RS09901': [0, 0, 0, 0],
        'RS09902': [0, 0, 0, 0],
        'RS10001': [0, 0, 0, 0],
        'RS10002': [0, 0, 0, 0],
    },
    'memberwise': [
        {
            'icpsr': "1",
            'votes': [
                [-1, "RS09901", 0],
                [-1, "RS09902", 0],
                [-1, "RS10001", 1],
                [-1, "RS10002", 1],
            ],
            'update': 1
        },
        {
            'icpsr': "2",
            'votes': [
                [1, "RS09901", 0],
                [1, "RS09902", 0],
                [1, "RS10001", 1],
                [1, "RS10002", 1],
            ],
            'update': 1
        },
        {
            'icpsr': "3",
            'votes': [
                [1, "RS09901", 0],
                [-1, "RS09902", 0],
                [1, "RS10001", 1],
                [1, "RS10002", 1],
            ],
            'update': 1
        },
    ],
    'idpt': {
        "1": {
            "99": [-0.4, 0],
            "100": [-0.5, 0]
        },
        "2": {
            "99": [0, 0],
            "100": [0.5, 0]
        },
        "3": {
            "99": [0, 0],
            "100": [0, 0],
        }
    }
}


fullpayload = payload.add_congresswise(test_payload)
pprint(fullpayload)

ret = nominate.update_nominate(
    fullpayload,
    update=['bw'],
    lambdaval=0.0,
    maxiter=1,
    add_meta=[]
)


pprint(ret)
