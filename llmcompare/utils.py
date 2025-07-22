import hashlib

import orjson


def hash_dict(d: dict) -> str:
    b = orjson.dumps(d, option=orjson.OPT_SORT_KEYS)
    return hashlib.blake2b(b).hexdigest()
