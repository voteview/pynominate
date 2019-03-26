from pprint import pprint

# Methods to prepare payloads for dynamic estimation
def add_congresswise(payload, min_congresses=0):
    """Modifies the 'idpt' collection of the payload to have a slot for an idpt per session and adds time to the votes collection"""
    tmp_idpt = {}
    idpt_key_str = isinstance(payload['idpt'].keys()[0], basestring)
    for i, m in enumerate(payload['memberwise']):
        max_cong = 0
        min_cong = 999
        drop = []
        icpsr = str(m['icpsr'])
        # get right key type to accomodate both json and mongodb input
        idpt_icpsr = icpsr if idpt_key_str else int(icpsr)
        member_idpts = {}
        for k, v in enumerate(m['votes']):
            congress = int(v[1][2:5])
            if congress < min_cong:
                min_cong = congress
            if congress > max_cong:
                max_cong = congress
            if str(congress) not in member_idpts:
                if str(congress) in payload['idpt'][idpt_icpsr]:
                    idpt = payload['idpt'][idpt_icpsr][str(congress)]
                elif isinstance(payload['idpt'][idpt_icpsr], list):
                    idpt = payload['idpt'][idpt_icpsr]
                else:
                    idpt = [0.0, 0.0]
                member_idpts[str(congress)] = idpt

        tmp_idpt[idpt_icpsr] = {
            'min_cong': min_cong,
            'max_cong': max_cong
        }
                
        memberwise_votes = []
        mobile = max_cong - min_cong >= min_congresses
        for k, v in enumerate(m['votes']):
            if mobile:
                tmp_idpt[idpt_icpsr]['idpts'] = [[0.0, 0.0]] * (max_cong - min_cong + 1)
                # Fill in skipped congresses for members with discontinuous service
                for cong, idpt in member_idpts.iteritems():
                    tmp_idpt[idpt_icpsr]['idpts'][int(cong) - min_cong] = idpt
                t = int(v[1][2:5]) - min_cong
            else:
                tmp_idpt[idpt_icpsr]['idpts'] = [member_idpts[str(min_cong)]]
                t = 0
                
            memberwise_votes.append([v[0], v[1], t])

        payload['memberwise'][i]['votes'] = memberwise_votes
        
    payload['idpt'] = tmp_idpt
    return payload
