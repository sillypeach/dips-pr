"""Small numerical truth checks; no manuscript-scale training."""
import importlib.util
import itertools
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run as release
sys.path.insert(0, str(HERE / "frozen/path"))
import model as m
import path_staged_solver as path


class ReleaseTests(unittest.TestCase):
    def test_manuscript_scope_and_original_bytes(self):
        self.assertEqual(len(release.scope_entries()), 20)
        self.assertEqual(len(release.scope_entries("pruning")), 35)
        self.assertEqual(release.verify_inputs()["datasets"], 45)

    def test_calibration_agrees_with_brute_closed_itemsets(self):
        tx = [[0, 1, 2], [0, 1, 2], [0, 1], [1], [2], []]
        y = np.array([11., 8., 4., 2., 1., 3.])
        for max_len in (1, 2, 3):
            correlations = []
            for length in range(1, max_len + 1):
                for pat in itertools.combinations(range(3), length):
                    tids = [i for i, row in enumerate(tx) if set(pat) <= set(row)]
                    if not tids:
                        continue
                    closure = set.intersection(*(set(tx[i]) for i in tids))
                    if closure == set(pat):
                        correlations.append(abs(float((y - y.mean())[tids].sum())))
            got = path.lambda_max_exact(tx, y, 1, max_len)
            self.assertTrue(got["exact"])
            self.assertAlmostEqual(got["lambda_max"], max(correlations, default=0.0), places=10)

    def test_screened_path_matches_explicit_full_dictionary_optimum(self):
        tx = [list(p) for p in itertools.chain.from_iterable(
            itertools.combinations(range(3), k) for k in range(4))] * 4
        y = np.array([3. + 5. * ({0, 1} <= set(t)) + (i % 3) for i, t in enumerate(tx)])
        tids, items = m.PoissonEN_LCM_SPP._transactions_to_item_tidsets(tx)
        patterns = list(m.LCMEnumerator(tids, items, len(tx), 2, 3, "closed").iter_patterns())
        X = m.build_X_from_patterns(len(tx), patterns)
        calibration = path.lambda_max_exact(tx, y, 2, 3)
        previous = None
        for tau in (1.0, 0.3, 0.1):
            cfg = dict(lam=tau * calibration["lambda_max"], kappa=.05, min_support=2,
                max_len=3, top_k_add=5, max_rounds=100, solver_max_iter=10000,
                solver_tol=1e-7, gap_tol=1e-7, pattern_space="closed", radius_mode="certified",
                dual_reference="ray", propagate_support=True)
            exact = m.PoissonElasticNetSolver(cfg["lam"], .05, 20000, 1e-9).fit(X, y)
            got = path.fit_path_point(tx, y, cfg, previous=previous, mode="PATH", calibration=calibration)
            self.assertTrue(got["fit"].converged)
            np.testing.assert_allclose(got["fit"].eta, exact.b + X @ exact.w, rtol=3e-5, atol=3e-5)
            self.assertLessEqual(got["fit"].gap / len(y), cfg["gap_tol"])
            if tau == 1:
                self.assertEqual(got["fit"].enum_stats_per_round, [])
                self.assertFalse(got["path_transfer"]["analytic_zero_solution_is_uv_pruning"])
            previous = got["state"]


class PreparedRunIntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="synthetic-integrity-")
        cls.output = Path(cls.temporary.name) / "run"
        release.prepare(cls.output)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_changed_run_inputs_and_sources_rejected_before_import(self):
        files = [
            "raw/source/extension_baseline_benchmark.py",
            "dips/source/regularization_path_experiment.py",
            "dips/audit/model.py",
            "structure/source/structure_baseline_benchmark.py",
            "raw/data/count_02_s2609271101.json",
            "dips/data/count_02_s2609271101.json",
            "raw/folds/count_02_s2609271101.json",
            "dips/folds/count_02_s2609271101.json",
        ]
        for relative in files:
            path = self.output / relative
            original = path.read_bytes()
            try:
                # A newline preserves valid Python/JSON; only a byte-level
                # registration check detects this harmless-looking change.
                path.write_bytes(original + b"\n")
                with self.subTest(file=relative), patch.object(release, "module",
                        side_effect=AssertionError("Output code imported before integrity check")):
                    with self.assertRaisesRegex(ValueError, "Changed or missing run file"):
                        release.aggregate(self.output)
            finally:
                path.write_bytes(original)

    def test_optional_lcm_absent_allows_empty_aggregation(self):
        self.assertFalse((self.output / "structure/vendor/lcm53.zip").exists())
        result = release.aggregate(self.output)
        self.assertEqual(result["rows"], 0)
        self.assertEqual(result["missing"], 200)
        self.assertFalse(result["complete"])

    def test_present_lcm_archive_must_match_pinned_hash(self):
        archive = self.output / "structure/vendor/lcm53.zip"
        archive.parent.mkdir(exist_ok=True)
        try:
            archive.write_bytes(b"not-the-pinned-archive")
            with patch.object(release, "module",
                    side_effect=AssertionError("Output code imported before integrity check")):
                with self.assertRaisesRegex(ValueError, "Changed or missing run file"):
                    release.aggregate(self.output)
        finally:
            archive.unlink()


if __name__ == "__main__":
    unittest.main()
