
import json
import numpy as np
from pprint import pprint
from numpy.random import uniform
from pynominate import payload
from pynominate import nominate

import sys
sys.path.insert(0, "..")

true_bps = {
    "RS001001": [0.6, -0.1, 0.3, 0.3]
}

pload = {
    "bp": {rcid: [0.0] * 4 for rcid in true_bps.keys()},
    "idpt": {
        str(i): {
            "idpts": [[uniform(-0.7, 0.7), uniform(-0.7, 0.7)]]
        }
        for i in range(100)
    }
}


def sim_payload_bp(pload, true_bps, w=0.4619, b=8.8633):
    """Simulate votes when given a toy payload stub and
    true ideal points"""
    votes_array = []
    for rcid, bp in true_bps.iteritems():
        print(bp)
        votes_array += [{
            "id": rcid,
            "update": 1,
            "votes": [
                [nominate.pr_yea(bp, pload["idpt"][icpsr]["idpts"][0], w, b) > uniform() and 1 or -1, icpsr]
                for icpsr in pload["idpt"].keys()
            ]
        }]

    pload["votes"] = votes_array
    return pload


fullpayload = sim_payload_bp(pload, true_bps)
pprint(fullpayload)

ret = nominate.update_nominate(
    fullpayload,
    update=['bp'],
    lambdaval=0.0,
    maxiter=1,
    add_meta=[],
    cores=2
)

pprint(ret['bp'])
pprint(true_bps)


def test_constraints():

    d = {
        'votes': np.array([1, -1]),
        'ideal': np.transpose(np.array([[0.5, 0.5], [-0.5, -0.5]]))
    }

    bp = np.array([0.00756398913426508, -0.999971392624997, 0.1, 0.1])

    nominate.circle_constraint_bp(bp)

    nominate.update_bp(d, 0.4, 7.5, bp)


def test_nudge_bp():
    with open("data/payload_broken_bp.json") as f:
        pload = json.load(f)

    fit = nominate.update_bp(
        d={
            'votes': np.array([v[0] for v in pload['votes'][0]['votes']]),
            'ideal': np.transpose(np.array([pload['idpt'][str(v[1])] for v in pload['votes'][0]['votes']]))
        },
        w=0.43,
        b=7.78,
        par0=pload["bp"]
    )

    pprint(fit)
    mp = fit['bp'][0]**2 + fit['bp'][1]**2

    alts = [
        np.square(fit['bp'][0] - fit['bp'][2]) + np.square(fit['bp'][1] - fit['bp'][3]),
        np.square(fit['bp'][0] + fit['bp'][2]) + np.square(fit['bp'][1] + fit['bp'][3]),
    ]

    print("pload constraints")
    print(nominate.circle_constraint_bp(pload['bp']))
    print("fit constraints")
    print(nominate.circle_constraint_bp(fit['bp']))

    print("mp")
    print(mp)
    print("alts")
    print(alts)
    assert mp <= 1.0
    assert alts[0] <= 1.0 or alts[1] <= 1.0


test_constraints()
test_nudge_bp()
