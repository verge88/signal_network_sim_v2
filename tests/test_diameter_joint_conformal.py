import math
import tempfile
import unittest
from pathlib import Path

from experiments.diameter_joint_conformal import (
    audit_captures, conformal_p, decision, require_resolution, split_development, wilson,
)

class JointConformalTests(unittest.TestCase):
    def test_exact_rank_and_ties(self):
        scores = list(range(19))
        self.assertEqual(conformal_p(scores, 19), .05)
        self.assertEqual(decision(scores, 19, .05), 1)
        self.assertEqual(decision(scores, 18, .05), 0)
        self.assertEqual(decision([1]*19, 1, .05), 0)

    def test_exchangeable_rank_bound(self):
        # Every rank plays the test role once, with no ties.
        for n in [19, 20, 39, 99]:
            values = list(range(n+1))
            rejected = sum(decision(values[:i]+values[i+1:], x, .05)
                           for i,x in enumerate(values))
            self.assertLessEqual(rejected/(n+1), .05)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            require_resolution(18, .05)
        require_resolution(19, .05)
        for scores, score in [([],1), ([1],math.nan), ([math.inf],0)]:
            with self.assertRaises(ValueError):
                conformal_p(scores, score)

    def test_archive_audit_checks_content_and_provenance(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cal_root, ev_root = root / "cal", root / "ev"
            cal_root.mkdir()
            ev_root.mkdir()
            cal = pd.DataFrame(
                [{"capture": name, "expected_label": "benign",
                  "scenario": scenario, "role": "calibration"}
                 for name, scenario in [
                     ("a.pcap", "attach"), ("b.pcap", "attach"),
                     ("c.pcap", "update"), ("d.pcap", "attach"),
                     ("e.pcap", "attach"), ("f.pcap", "update"),
                     ("missing.pcap", "attach"),
                 ]]
            )
            ev = pd.DataFrame([{"capture": "test.pcap", "expected_label": "benign",
                                "scenario": "attach"}])
            for name, data in {
                "a.pcap": b"one", "b.pcap": b"one", "c.pcap": b"",
                "d.pcap": b"shared", "e.pcap": b"conflict", "f.pcap": b"conflict",
            }.items():
                (cal_root / name).write_bytes(data)
            (ev_root / "test.pcap").write_bytes(b"shared")

            def decode(path):
                if not path.read_bytes():
                    raise RuntimeError(f"No Diameter PDUs decoded from {path}")
                return {"_decoded_pdus": 1}

            audit = audit_captures(cal, ev, cal_root, ev_root, decode)
            self.assertEqual(audit.set_index("capture").status.to_dict(), {
                "a.pcap": "retained", "b.pcap": "duplicate", "c.pcap": "undecodable",
                "d.pcap": "cross_dataset_overlap", "test.pcap": "cross_dataset_overlap",
                "e.pcap": "conflicting_metadata", "f.pcap": "conflicting_metadata",
                "missing.pcap": "missing",
            })

    def test_archive_split_keeps_inner_profiles_and_outer_resolution(self):
        import pandas as pd

        records = [
            {"capture": f"{scenario}_{role}_{i:02}.pcap",
             "expected_label": "benign", "scenario": scenario, "role": role}
            for scenario, count in [("attach", 14), ("reauth", 13), ("update", 13)]
            for role, size in [("reference", 8), ("calibration", count)]
            for i in range(size)
        ]
        inner, outer = split_development(pd.DataFrame(records[::-1]))
        self.assertEqual(len(inner), 42)
        self.assertEqual(len(outer), 22)
        self.assertTrue(set(inner.capture).isdisjoint(outer.capture))
        require_resolution(len(outer), .05)

    def test_wilson(self):
        self.assertGreater(wilson(0,8)[1], .30)
        self.assertLess(wilson(16,16)[0], .85)

if __name__ == "__main__":
    unittest.main()
