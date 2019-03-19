from pprint import pprint

# Methods to prepare payloads for dynamic estimation
def add_congresswise(payload, min_congresses=0):
    """Modifies the 'idpt' collection of the payload to have a slot for an idpt per session and adds time to the votes collection"""
    member_congress_count = 0
    vote_count = 0
    tmp_idpt = {}
    for i, m in enumerate(payload['memberwise']):
        max_cong = 0
        min_cong = 999
        drop = []
        icpsr = str(m['icpsr'])
        member_idpts = {}
        for k, v in enumerate(m['votes']):
            congress = int(v[1][2:5])
            if congress < min_cong:
                min_cong = congress
            if congress > max_cong:
                max_cong = congress
            if str(congress) not in member_idpts:
                if str(congress) in payload['idpt'][icpsr]:
                    idpt = payload['idpt'][icpsr][str(congress)]
                elif isinstance(payload['idpt'][icpsr], list):
                    idpt = payload['idpt'][icpsr]
                else:
                    idpt = [0.0, 0.0]
                member_idpts[str(congress)] = idpt

        tmp_idpt[str(icpsr)] = {
            'min_cong': min_cong,
            'max_cong': max_cong
        }
                
        memberwise_votes = []
        mobile = max_cong - min_cong >= min_congresses
        for k, v in enumerate(m['votes']):
            if mobile:
                tmp_idpt[str(icpsr)]['idpts'] = [[0.0, 0.0]] * (max_cong - min_cong + 1)
                # Fill in skipped congresses for members with discontinuous service
                for cong, idpt in member_idpts.iteritems():
                    tmp_idpt[str(icpsr)]['idpts'][int(cong) - min_cong] = idpt
                t = int(v[1][2:5]) - min_cong
            else:
                tmp_idpt[str(icpsr)]['idpts'] = member_idpts[str(min_cong)]
                t = 0
                
            memberwise_votes.append([v[0], v[1], t])

        payload['memberwise'][i]['votes'] = memberwise_votes
        
    payload['idpt'] = tmp_idpt
    return payload
