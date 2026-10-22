import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir_utils import (bootstrap_diff_ci, cohens_d, lexical_overlap, ndcg_at_k,
                      paired_bootstrap, permutation_test, recall_at_k,
                      reciprocal_rank, reciprocal_rank_fusion, simple_tokenize)


class TestTokenizerAndOverlap(unittest.TestCase):
    def test_tokenize(self):
        self.assertEqual(simple_tokenize("Do Statins-Cause Cancer? B12"),
                         ["do", "statins", "cause", "cancer", "b12"])

    def test_overlap(self):
        self.assertAlmostEqual(lexical_overlap("red meat cancer", "Meat and cancer risk"), 2 / 3)
        self.assertEqual(lexical_overlap("", "anything"), 0.0)


class TestRRF(unittest.TestCase):
    def test_formula(self):
        fused = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], c=60, k=3)
        # b: 1/62 + 1/61 ; a: 1/61 ; c: 1/62
        self.assertEqual(fused, ["b", "a", "c"])

    def test_truncation_and_union(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["d", "e", "f"]], k=4)
        self.assertEqual(len(fused), 4)
        self.assertEqual(set(fused[:2]), {"a", "d"})


class TestMetrics(unittest.TestCase):
    def test_recall(self):
        self.assertAlmostEqual(recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 3), 0.5)
        self.assertIsNone(recall_at_k(["a"], set(), 1))

    def test_rr(self):
        self.assertAlmostEqual(reciprocal_rank(["x", "y", "a"], {"a"}), 1 / 3)
        self.assertEqual(reciprocal_rank(["x", "y", "a"], {"a"}, k=2), 0.0)
        self.assertEqual(reciprocal_rank(["x"], {"a"}), 0.0)

    def test_ndcg_perfect_and_graded(self):
        rels = {"a": 2, "b": 1}
        self.assertAlmostEqual(ndcg_at_k(["a", "b", "z"], rels), 1.0)
        worse = ndcg_at_k(["b", "a"], rels)
        expected = (1 / 1 + 2 / math.log2(3)) / (2 / 1 + 1 / math.log2(3))
        self.assertAlmostEqual(worse, expected)
        self.assertEqual(ndcg_at_k(["z"], {"a": 1}), 0.0)


class TestStats(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.a = rng.normal(0.30, 0.15, 80)
        self.b = rng.normal(0.20, 0.15, 100)

    def test_permutation_detects_difference_and_is_deterministic(self):
        d1, p1 = permutation_test(self.a, self.b, n_perm=2000)
        d2, p2 = permutation_test(self.a, self.b, n_perm=2000)
        self.assertEqual((d1, p1), (d2, p2))
        self.assertLess(p1, 0.01)

    def test_permutation_null(self):
        rng = np.random.default_rng(1)
        _, p = permutation_test(rng.normal(0, 1, 60), rng.normal(0, 1, 60), n_perm=2000)
        self.assertGreater(p, 0.05)

    def test_bootstrap_ci_contains_true_diff(self):
        lo, hi = bootstrap_diff_ci(self.a, self.b, n_boot=2000)
        self.assertLess(lo, 0.10)
        self.assertGreater(hi, 0.10)
        self.assertGreater(lo, 0)

    def test_cohens_d_known(self):
        self.assertAlmostEqual(cohens_d([1, 2, 3], [2, 3, 4]), -1.0)

    def test_paired_bootstrap(self):
        rng = np.random.default_rng(2)
        y = rng.uniform(0, 1, 300)
        obs, (lo, hi), p = paired_bootstrap(y + 0.05 + rng.normal(0, 0.02, 300), y, n_boot=2000)
        self.assertGreater(lo, 0)
        self.assertLess(p, 0.01)
        obs0, (lo0, hi0), p0 = paired_bootstrap(y, y + rng.normal(0, 0.1, 300), n_boot=2000)
        self.assertLess(lo0, 0.02)
        self.assertGreater(hi0, -0.02)
        self.assertGreater(p0, 0.05)


if __name__ == "__main__":
    unittest.main()
