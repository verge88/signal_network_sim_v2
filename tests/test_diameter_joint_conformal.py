import math
import unittest
from experiments.diameter_joint_conformal import conformal_p, decision, require_resolution, wilson

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

    def test_wilson(self):
        self.assertGreater(wilson(0,8)[1], .30)
        self.assertLess(wilson(16,16)[0], .85)

if __name__ == "__main__":
    unittest.main()
