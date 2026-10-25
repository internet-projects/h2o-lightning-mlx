"""The facade's guided Box generation: the grammar accepts exactly H2O's Box JSON schema, and a refused schema
produces the 422 the shim maps to BOX_NEEDS_GUIDED (no model needed)."""
import json
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mlx_vllm_facade as F  # noqa: E402

MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")
GUIDED_UNAVAILABLE = re.compile(r"response_format|structured.?output|guided|grammar|json_schema|xgrammar|outlines", re.I)


class TestBoxGrammar(unittest.TestCase):
    def test_scan(self):
        k = "bbox_2d"
        for s, want in {'{"bbox_2d": [12, 30, 1000, 0]}': "complete", '{"bbox_2d":[0,0,0,0]}': "complete",
                        ' {\n "bbox_2d" : [ 5 ,6,7,8 ] } ': "complete", '{"bbox_2d": [100': "partial", '': "partial",
                        '{"bbox_2d": [01': "bad", '{"bbox_2d": [1001': "bad", '{"box': "bad", '{"bbox_2d": [-1': "bad",
                        '{"bbox_2d": [1,2,3]': "bad", '{"bbox_2d": [1,2,3,4]} x': "bad", '{"bbox_2d": [¹': "bad"}.items():
            self.assertEqual(F.box_scan(s, k), want, s)

    def test_schemas_of_this_release(self):
        sys.path.insert(0, MODEL)
        import box  # H2O's box.py, unmodified
        for conv, schema in box.SCHEMAS.items():
            key = F.box_schema_key({"type": "json_schema", "json_schema": {"name": "box", "strict": True, "schema": schema}})
            self.assertIn(key, json.dumps(schema), conv)

    def test_other_schemas_are_refused_the_way_the_shim_expects(self):
        with self.assertRaises(F.HTTPError) as e:
            F.box_schema_key({"type": "json_schema", "json_schema": {"schema": {"type": "object", "properties": {}}}})
        self.assertEqual(e.exception.status, 422)
        self.assertTrue(GUIDED_UNAVAILABLE.search(str(e.exception)))


if __name__ == "__main__":
    unittest.main()
