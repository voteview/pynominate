from pynominate import payload
from pprint import pprint
from copy import deepcopy


def test_dynamic_payload_build(int_payload, str_payload):
    for pload_type, pload in {"int": int_payload, "str": str_payload}.iteritems():
        temp_load = deepcopy(pload)
        payload.add_congresswise(temp_load)


def test_freezing_short_members(int_payload, str_payload):
    for pload_type, pload in {"int": int_payload, "str": str_payload}.iteritems():
        min_congresses = 2
        temp_load = deepcopy(pload)
        temp_load = payload.add_congresswise(temp_load, min_congresses=min_congresses)

        pprint(temp_load)
        # Test right number of idpts
        assert len(temp_load["idpt"][1 if pload_type == "int" else "1"]["idpts"]) == 3
        assert len(temp_load["idpt"][2 if pload_type == "int" else "2"]["idpts"]) == 1
        assert len(temp_load["idpt"][3 if pload_type == "int" else "3"]["idpts"]) == 1

        # Test t=0 for people with short tenures (even if multiple sessions), and right for others
        for m in temp_load["memberwise"]:
            idpt = temp_load["idpt"][m["icpsr"]]
            min_cong = idpt["min_cong"]
            max_cong = idpt["max_cong"]
            for v in m["votes"]:
                if max_cong - min_cong < min_congresses:
                    assert v[2] == 0
                else:
                    assert v[2] == int(v[1][2:5]) - min_cong

