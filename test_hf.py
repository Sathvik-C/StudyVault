from services.semantic_search import _embed_via_hf_api
import sys
import logging
logging.basicConfig(level=logging.INFO)
res = _embed_via_hf_api(["hello world", "test"])
if res:
    print(f"Success, shape: {len(res)} x {len(res[0])}")
else:
    print("Failed")
