from pynominate.nominate import update_idpt_star, OPTIONS, OPTIONSWB
import numpy as np
from multiprocessing import cpu_count
from multiprocessing import Pool

import json
import csv


def merge_dicts(x, y, z):
    """
    Given three dicts, merge them into a new dict as a shallow copy.
    See https://stackoverflow.com/questions/38987/how-to-merge-two-dictionaries-in-a-single-expression
    """
    zz = x.copy()
    zz.update(y)
    zz.update(z)
    return zz


def make_member_to_votes_and_bill_parameters(payload):
    member_chamber_congress_count = 0
    tmp_dct = {}
    for m in payload['memberwise']:
        for v in m['votes']:
            chamber_congress = "%s_%s" % (
                v[1][1], v[1][2:5]
            )
            if m['icpsr'] not in tmp_dct:
                tmp_dct[m['icpsr']] = {}
            
            if chamber_congress in tmp_dct:
                tmp_dct[m['icpsr']][chamber_congress]['votes'].append(v[0])
                tmp_dct[m['icpsr']][chamber_congress]['bp'].append(payload['bp'][str(v[1])])
            else:
                member_chamber_congress_count += 1
                tmp_dct[m['icpsr']][chamber_congress] = {
                    'votes': [v[0]],
                    'bp': [payload['bp'][str(v[1])]],
                }
    return tmp_dct


def make_member_congress_votes(payload):
    icpsr_dict = make_member_to_votes_and_bill_parameters(payload)
    dat = {
        'data': [],
        'icpsr_chamber_congress': [],
        'start': []
    }
    for icpsr, chamber_cong_dict in icpsr_dict.iteritems():

        dat['data'] += [
            {
                "votes": np.array(v['votes']),
                'bp':np.transpose(np.array(v['bp']))
            }
            for v in chamber_cong_dict.values()
        ]

        dat['start'] += [payload['idpt'][icpsr]] * len(chamber_cong_dict)

        dat['icpsr_chamber_congress'] += [
            dict(zip(
                ["icpsr", "chamber", "cong", "nvotes"],
                [icpsr] + k.split("_") + [len(v['votes'])]
            ))
            for k, v in chamber_cong_dict.iteritems()
        ]
    
    return dat


def nokken_poole(payload, cores=int(cpu_count()) - 1, xtol=1e-4, add_meta=['members', 'rollcalls']):
    import time

    OPTIONS['xtol'] = xtol
    OPTIONSWB['xtol'] = xtol

    print "(000) Running Nokken-Poole on %i cores..." % cores
    firststarttime = time.time()

    if cores >= 2:
        pool = Pool(cores)
        mymap = pool.map  # allow switching to in/out parallel processing for  debugging
    else:
        mymap = map

    if 'bw' in payload:
        b = payload['bw']['b']
        w = payload['bw']['w']
    else:
        b = 8.8633
        w = 0.4619
    starttime = time.time()

    dat = make_member_congress_votes(payload)
    print "(001) Data marshal took %2.2f seconds (%i members)..." % (time.time() - starttime, len(dat['start']))
    # Run dwnominate...
    res_idpt = mymap(
        update_idpt_star,
        zip(
            dat['data'], [w] * len(dat['data']),
            [b] * len(dat['data']), dat['start']
        )
    )

    print "(002) Total update time elapsed %5.2f minutes." % ((time.time() - firststarttime) / 60)
    res_idpt = [
        merge_dicts(r, s, {'startx': t})
        for r, s, t in zip(
            res_idpt, dat['icpsr_chamber_congress'], dat['start']
        )
    ]
    return res_idpt


def write_csv(res, file_object):
    fields = ['icpsr', 'chamber', 'cong', 'nvotes',
              'startx', 'x', 'llstart', 'llend']
    csvout = csv.writer(file_object)
    csvout.writerow(['icpsr', 'chamber', 'cong', 'nvotes', 'dwnom1', 'dwnom2',
                     'np1', 'np2', 'dw_ll', 'np_ll'])
    for r in sorted(res, key=lambda k: (k['icpsr'], k['cong'])):
        record = []
        for f in fields:
            if type(r[f]) in [list, np.ndarray]:
                for ff in r[f]:
                    record.append(round(ff, 3))
            else:
                record.append(
                    type(r[f]) is np.float64 and round(r[f], 3) or r[f])
        csvout.writerow(record)


if __name__ == "__main__":
    # Note: will not work because in .json the keys in the idpt collection
    # and the idpt values in the memberwise collection are of different types!
    from pprint import pprint
    print "Testing Nokken-Poole...\n"
    payload = json.load(open("pynominate/tests/data/payload.json"))
    result = nokken_poole(payload, cores=1)
    write_csv(result, open(
        'pynominate/tests/data/nokken_poole_out.csv', 'w'))
