import unittest

import numpy as np
import pandas as pd

from diameter.diameter_report_recurrence import (
    PROFILE_COLUMNS, report_recurrence, validate_compromise_state,
)
from diameter.diameter_sim_v1 import AttackScenario, AttackType, DiameterSimulator, SimulationConfig


def reports(intervals):
    rows = len(intervals)
    ret = pd.DataFrame({"node_id": [1] * rows, "interval": intervals,
                        "cu_delivered": [1] * rows})
    for index, col in enumerate(PROFILE_COLUMNS):
        ret[col] = 0.125 if index < 4 else 0.0
    return ret


class ReportRecurrenceTests(unittest.TestCase):
    def test_history_is_causal_and_order_is_restored(self):
        frame = reports(np.arange(90))
        result = report_recurrence(frame)
        np.testing.assert_array_equal(result[:20], report_recurrence(frame.iloc[:20]))
        self.assertEqual(result[13], 0)
        self.assertAlmostEqual(result[14], 14 / 15)
        self.assertEqual(result[-1], 1)
        shuffled = frame.sample(frac=1, random_state=4)
        np.testing.assert_array_equal(report_recurrence(shuffled), result[shuffled.index])

    def test_stale_poll_node_boundary_and_interval_gap(self):
        first = reports(np.arange(75))
        first.loc[5:, "cu_delivered"] = 0
        later = reports(np.r_[np.arange(8), np.arange(70, 78)])
        later["node_id"] = 2
        self.assertFalse(report_recurrence(pd.concat([first, later])).any())

    def test_invalid_report_ratios_do_not_contribute(self):
        frame = reports(np.arange(60))
        frame[PROFILE_COLUMNS[0]] = float("nan")
        self.assertFalse(report_recurrence(frame).any())

    def test_simulator_stops_compromise_with_the_scenario(self):
        config = SimulationConfig(duration_hours=12 * 15 / 3600, interval_s=15.0,
                                  master_node_ids=[0, 1], seed=42)
        scenario = AttackScenario(AttackType.NODE_COMPROMISE, [7], 3, 7, intensity=5.0)
        sim = DiameterSimulator(config, [scenario], [])
        sim.run()
        frame = sim.to_dataframe()
        validate_compromise_state(frame)
        node = frame.loc[frame["node_id"].eq(7)].sort_values("interval")
        self.assertEqual(node.loc[node["is_compromised"].eq(1), "interval"].tolist(), [3, 4, 5, 6])
        self.assertFalse(sim.topology.nodes[7].is_compromised)
        self.assertIsNone(sim.topology.nodes[7].compromised_since)

    def test_inconsistent_historical_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent compromise state"):
            validate_compromise_state(pd.DataFrame(
                {"attack_type": ["none"], "is_compromised": [1]}))


if __name__ == "__main__":
    unittest.main()
