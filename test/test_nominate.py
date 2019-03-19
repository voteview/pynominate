import sys
sys.path.insert(0, "..")
from pprint import pprint
from numpy.random import uniform
from pynominate import nominate
from pynominate import payload
import cProfile, pstats, StringIO

sessions = 10
# Recovers right ideal points?
pload = {
    "bp": {"RS" + str(t).zfill(3) + str(i).zfill(4):
           [uniform(-0.5, 0.5),
            uniform(-0.5, 0.5),
            uniform(-0.5, 0.5),
            uniform(-0.5, 0.5)]
           for t in range(sessions) for i in range(100)},
    "idpt": {"0": [0.5, 0.5], "1": [0.0, 0.0]}
}

# Actual ideal points
x = {"0": [[0.5,  0.7],
           [-0.6, -0.4],
           [1, 0],
           [0.7, 0.3]],
     "1": [[0.8, 0.1]]}

x = {"0": [[0.5, 0.5]] * sessions, "1": [[0.5, 0.5]]}


def sim_payload(pload, xx, w=0.4619, b=8.8633):
    """Simulate votes when given a toy payload stub and
    true ideal points"""
    member_array = []
    for icpsr, idpt in pload["idpt"].iteritems():
        member_array += [{
            "icpsr": icpsr,
            "update": 1,
            "votes": [
                [nominate.pr_yea(bp, xx[icpsr][int(rcid[2:5])], w, b) > uniform() and 1 or -1, rcid, int(rcid[2:5])]
                for rcid, bp in pload["bp"].iteritems() if int(rcid[2:5]) < len(xx[icpsr])
            ]
        }]

    pload["memberwise"] = member_array
    return pload


fullpayload = payload.add_congresswise(sim_payload(pload, x))

ret = nominate.update_nominate(
    fullpayload,
    update=['idpt'],
    lambdaval=0.0,
    maxiter=1,
    add_meta=[]
)

pprint(ret['idpt'])

# pr = cProfile.Profile()
# pr.enable()
ret = nominate.update_nominate(
    fullpayload,
    update=['idpt'],
    lambdaval=100000.0,
    maxiter=1,
    add_meta=[]
)
pprint(ret['idpt'])
# pr.disable()
# pprint(ret['idpt'])
# s = StringIO.StringIO()
# ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')

# ps.print_stats()
# print(s.getvalue())


# TODO TEST

# newpayload = add_votes(add_starts(modify_payload(fullpayload)))
# pprint(newpayload["idpt"]["0"])
# pprint(newpayload["memberwise"][0]["votes"][0:10])

# res = nominate.update_nominate(
#     newpayload,
#     update = ['idpt'],
#     lambdas = [0, 0],
#     maxiter = 2,
#     cores = 1
# )
# print("Correct ideal points")
# pprint(res['idpt']['0'])

# # Does it put all t = 0 if not enough congresses served?
# #newpayload = add_starts(dynamic_splines.modify_payload(fullpayload, 5))
# #pprint(newpayload["idpt"])

# #print("Served fewer than 5 congresses, so all 0")
# #pprint(newpayload["memberwise"][0]["votes"][0:5])

# # Does it truncate at 20 (19 with 0 index?)
# #payload = {
# #    "bp" : {"RS" + str(t).zfill(3) + str(i).zfill(4):
# #            [uniform(-0.5, 0.5),
# #             uniform(-0.5, 0.5),
# #             uniform(-0.5, 0.5),
# #             uniform(-0.5, 0.5)]
# #            for t in range(22) for i in range(1)},
# #    "idpt": {"0": {"0": [0.0, 0.0]}}
# #}

# # mock
# #x = {"0": [[uniform(-0.5, 0.5), uniform(-0.5, 0.5)] for t in range(22)]}
# #fullpayload = sim_payload(payload, x)
# #newpayload = dynamic_splines.modify_payload(fullpayload, 5)
# #pprint(newpayload["idpt"])

# #print("Truncate t at 19")
# #pprint(newpayload["memberwise"][0]["votes"])

