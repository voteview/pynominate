import sys
sys.path.insert(0, "..")
from pynominate import nominate
from pprint import pprint
import json
import time
import copy

starttime = time.time()

payload = nominate.add_prs(json.load(open("idpt_payload.json")))

print("loaded time...")
print((time.time() - starttime) / 60)

loadtime = time.time()

idpts = copy.deepcopy(payload['idpt'])
bres = nominate.bootstrap(payload, idpts, 200, 3)
ses = nominate.get_ses(bres, payload['idpt'])
lens = {m['icpsr']: len(m['votes']) for m in payload['memberwise']}
#print(ses)
with open("data/ses.json", "w") as rb:
    json.dump(ses, rb)
with open("data/bs.json", "w") as rb:
    json.dump(bres, rb)
with open("data/len.json", "w") as rb:
    json.dump(lens, rb)
print((time.time() - loadtime) / 60)
