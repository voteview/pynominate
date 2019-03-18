from pynominate import payload
from pprint import pprint


def test_dynamic_payload_build(int_payload, str_payload):
    for pload_type, pload in {"int": int_payload, "str": str_payload}.iteritems():
        payload.add_congresswise(pload)

