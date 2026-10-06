"""Validate tree-rule semantics, rather than the planted simulation outcomes."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
from sklearn.ensemble import RandomForestRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "frozen/structure"))
from rf_path_support import extract_rf_path_support


def forest_adapter(left, right, feature, threshold, p, copies=1):
    tree = SimpleNamespace(children_left=np.array(left), children_right=np.array(right),
                           feature=np.array(feature), threshold=np.array(threshold))
    return SimpleNamespace(n_features_in_=p,
                           estimators_=[SimpleNamespace(tree_=tree) for _ in range(copies)])


def balanced_tree(copies=1):
    return forest_adapter([1, 2, -1, -1, 5, -1, -1], [4, 3, -1, -1, 6, -1, -1],
                          [0, 1, -2, -2, 1, -2, -2], [.5, .5, -2, -2, .5, -2, -2],
                          2, copies)


class ForestPathSupportTests(unittest.TestCase):
    def setUp(self):
        self.X = np.repeat(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]), 2, axis=0)

    def test_signed_rules_and_projections_have_distinct_semantics(self):
        result = extract_rf_path_support(balanced_tree(2), self.X, 2, 2)
        self.assertEqual(result["positive_rules"], [[0], [0, 1]])
        self.assertEqual(result["positive_projection"], [[0], [1], [0, 1]])
        pure = {tuple(row["items"]): row for row in result["positive_rule_details"]}
        proj = {tuple(row["items"]): row for row in result["positive_projection_details"]}
        self.assertEqual(pure[(0,)], dict(items=[0], support_count=4, tree_count=2, path_count=2))
        # The mixed rule !x0 & x1 has support 2, whereas its projection x1 has 4.
        self.assertEqual(proj[(1,)]["support_count"], 4)
        self.assertEqual(proj[(1,)]["mixed_sign_path_count"], 2)
        self.assertEqual(proj[(0,)]["path_count"], 4)
        self.assertEqual(result["stats"]["unique_signed_prefixes"], 6)
        json.dumps(result, allow_nan=False)

    def test_internal_prefixes_support_and_length_filters(self):
        # The internal positive prefix x0 survives, even though longer leaves
        # are removed by length or support. Projection uses its own support.
        by_support = extract_rf_path_support(balanced_tree(), self.X, 3, 2)
        by_length = extract_rf_path_support(balanced_tree(), self.X, 1, 1)
        for result in (by_support, by_length):
            self.assertEqual(result["positive_rules"], [[0]])
            self.assertEqual(result["positive_projection"], [[0], [1]])
        self.assertEqual(extract_rf_path_support(balanced_tree(), self.X, 9, 2)["positive_rules"], [])

    def test_repeated_literals_contradictions_and_nonstandard_thresholds(self):
        # Root x0>0, then another split x0<=.9/x0>.9. The left child conflicts;
        # the right child repeats x0 and must not invent the itemset {0,0}.
        repeated = forest_adapter([1, -1, 3, -1, -1], [2, -1, 4, -1, -1],
                                  [0, -2, 0, -2, -2], [0, -2, .9, -2, -2], 2)
        result = extract_rf_path_support(repeated, self.X, 1, 2)
        self.assertEqual(result["positive_rules"], [[0]])
        self.assertEqual(result["positive_rule_details"][0]["path_count"], 2)
        self.assertEqual(result["stats"]["contradictory_branches"], 1)
        self.assertEqual(result["stats"]["repeated_literal_branches"], 1)
        for threshold in (-1, 1, np.inf, -np.inf):
            constant = forest_adapter([1, -1, -1], [2, -1, -1],
                                      [0, -2, -2], [threshold, -2, -2], 2)
            got = extract_rf_path_support(constant, self.X, 1, 2)
            self.assertEqual(got["positive_rules"], [])
            self.assertEqual(got["positive_projection"], [])
            self.assertEqual(got["stats"]["domain_empty_branches"], 1)

    def test_empty_fit_branch_and_constant_forest(self):
        X = np.zeros((4, 2), dtype=bool)
        result = extract_rf_path_support(balanced_tree(), X, 1, 2)
        self.assertEqual(result["positive_projection"], [])
        self.assertGreater(result["stats"]["fit_empty_branches"], 0)
        fitted = RandomForestRegressor(n_estimators=3, random_state=9).fit(X, np.ones(4))
        got = extract_rf_path_support(fitted, X, 1, 2)
        self.assertEqual(got["positive_rules"], [])
        self.assertEqual(got["stats"]["signed_prefix_occurrences"], 0)

    def test_real_forest_prefix_rules_match_node_decision_paths(self):
        X = np.array([[int((i >> j) & 1) for j in range(4)] for i in range(16)])
        y = 4 * X[:, 0] * X[:, 1] + 3 * X[:, 2] - X[:, 3]
        fitted = RandomForestRegressor(n_estimators=7, random_state=18,
                                       max_depth=4, bootstrap=True).fit(X, y)
        before = fitted.predict(X).copy()
        got = extract_rf_path_support(fitted, X, 1, 4)
        # Independent recursive oracle operates with explicit literal tuples,
        # and checks each rule against sklearn's actual decision_path rows.
        positive, projection = set(), set()
        for estimator in fitted.estimators_:
            decisions = estimator.decision_path(X).toarray().astype(bool)
            tree = estimator.tree_
            def visit(node, literals):
                if literals:
                    mask = np.ones(len(X), dtype=bool)
                    for feature, value in literals:
                        mask &= X[:, feature] == value
                    np.testing.assert_array_equal(mask, decisions[:, node])
                    pos = tuple(sorted({f for f, v in literals if v}))
                    if pos:
                        projection.add(pos)
                        if all(v for _, v in literals):
                            positive.add(pos)
                if tree.children_left[node] != -1:
                    feature = int(tree.feature[node])
                    visit(int(tree.children_left[node]), literals + [(feature, 0)])
                    visit(int(tree.children_right[node]), literals + [(feature, 1)])
            visit(0, [])
        self.assertEqual({tuple(v) for v in got["positive_rules"]}, positive)
        self.assertEqual({tuple(v) for v in got["positive_projection"]}, projection)
        for row in got["positive_projection_details"]:
            self.assertEqual(row["support_count"], int(X[:, row["items"]].all(axis=1).sum()))
        np.testing.assert_array_equal(before, fitted.predict(X))

    def test_invalid_input_rejected(self):
        fitted = balanced_tree()
        for X in (np.array([[0, .2]]), np.array([[0, np.nan]]), np.array([])):
            with self.assertRaises(ValueError):
                extract_rf_path_support(fitted, X, 1, 2)
        for minimum, maximum in ((0, 2), (1, 0), (True, 2), (1.5, 2)):
            with self.assertRaises(ValueError):
                extract_rf_path_support(fitted, self.X, minimum, maximum)
        with self.assertRaises(ValueError):
            extract_rf_path_support(RandomForestRegressor(), self.X, 1, 2)
        with self.assertRaises(ValueError):
            extract_rf_path_support(fitted, np.ones((3, 3)), 1, 2)


if __name__ == "__main__":
    unittest.main()
