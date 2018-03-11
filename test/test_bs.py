import sys
sys.path.insert(0, "..")
from pynominate import nominate
from pprint import pprint
import json
import time
import copy

starttime = time.time()

payload = nominate.add_prs(json.load(open("payload_S105_S115.json")))
#payload = nominate.add_prs(json.load(open("toy_data.json")))

print("loaded time...")
print((time.time() - starttime) / 60)

loadtime = time.time()

idpts = copy.deepcopy(payload['idpt'])
bres = nominate.bootstrap(payload, idpts, 25, 2)
ses = nominate.get_ses(bres, payload['idpt'])
lens = {icpsr: len(v) for icpsr, v in payload['memberwise'].iteritems()}
#print(ses)
with open("data/ses_S105_S115_50.json", "w") as rb:
    json.dump(ses, rb)
with open("data/bs_S105_S115_50.json", "w") as rb:
    json.dump(bres, rb)
with open("data/len_S105_S115_50.json", "w") as rb:
    json.dump(lens, rb)
print((time.time() - loadtime) / 60)
