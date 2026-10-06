# Adapted from the original frozen study tests; only the source location changes.
"""Small, wholly synthetic on-disk certificates; never read experimental results."""
import importlib.util
import json
import math
from pathlib import Path
import shutil
import tempfile
import unittest


SOURCE = Path(__file__).resolve().parents[1] / "frozen/source/aggregate_repeats.py"
SPEC = importlib.util.spec_from_file_location("aggregate_repeats", SOURCE)
AGG = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGG)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def register_study(root, study):
    save(root / "study.json", study)
    save(root / "registered.json", {"study_sha256": AGG.sha256(root / "study.json")})


def fixture(root):
    """All 125 identities, with two synthetic path points instead of 101."""
    (root / "source").mkdir(parents=True)
    shutil.copy2(SOURCE, root / "source/aggregate_repeats.py")
    entries = []
    for endpoint in AGG.ENDPOINTS:
        for seed in AGG.SEEDS:
            original = "hiv_" + endpoint.replace("/", "_")
            dataset_id = original + f"__seed_{seed}"
            relative = f"repeats/seed_{seed}/endpoints/{original}"
            run = root / relative
            (run / "source").mkdir(parents=True)
            (run / "source/hiv_baseline.py").write_text("# Synthetic baseline source\n")
            data = {"y_train": [2., 4., 7., 8.], "y_final": [2., 4., 7., 8.], "y_test": [1., 2., 4.],
                    "train_indices": [0, 1, 2, 3], "validation_indices": [], "test_indices": [4, 5, 6]}
            save(run / "data.json", data)
            save(run / "folds.json", {"synthetic": True})
            protocol = {"state": "frozen", "endpoint": endpoint, "dataset_id": dataset_id,
                        "seed": 42, "model_seed": 42, "outer_split_seed": seed, "cv_seed": seed,
                        "n_train": 4, "n_final": 4, "n_test": 3, "taus": [1.0, 0.5],
                        "primary_selection": "CV-min", "paper_pr2_key": "PseudoR2_train_null",
                        "baselines": list(AGG.BASELINES),
                        "files": {name: AGG.sha256(run / name) for name in ("data.json", "folds.json", "source/hiv_baseline.py")}}
            save(run / "protocol.json", protocol)
            protocol_hash = AGG.sha256(run / "protocol.json")
            save(run / "registered.json", {"protocol_sha256": protocol_hash})
            entries.append({"dataset_id": dataset_id, "original_dataset_id": original, "endpoint": endpoint,
                            "root": relative, "outer_split_seed": seed, "protocol_sha256": protocol_hash})
            rows, baseline_hashes, point_rows = [], {}, {}
            for i, method in enumerate(["DIPS-PR (CV-min)", "DIPS-PR (1SE)"] + list(AGG.BASELINES)):
                delta = (seed - 42) * 0.03 + i * 0.07
                prediction = [1.1 + delta, 1.8 + delta, 4.2 + delta]
                metrics = AGG.prediction_metrics(data["y_test"], prediction, data["y_final"])
                # Deliberately distinct: a fallback to evaluation-mean PR2 must fail tests.
                metrics["PseudoR2"] = -999.0
                row = {"method": method, "metrics": metrics}
                if i < 2:
                    index = 1 - i
                    row.update(tau=protocol["taus"][index], lam=2 * protocol["taus"][index], NZ=2)
                    point_rows[index] = {"status": "complete", "verified": True, "protocol_sha256": protocol_hash,
                                         "split": "final", "point": index, "tau": row["tau"], "config": {"lam": row["lam"]},
                                         "prediction": prediction, "y_evaluation": data["y_test"], "metrics": metrics,
                                         "training_indices": data["train_indices"], "evaluation_indices": data["test_indices"],
                                         "independent": {"status": "passed", "objective_gap_consistent": True, "gap_per_sample": 0., "kkt": 0.}}
                else:
                    baseline = {"status": "complete", "method": method, "model": method, "endpoint": endpoint, "seed": 42,
                                "protocol_sha256": protocol_hash, "data_sha256": protocol["files"]["data.json"],
                                "folds_sha256": protocol["files"]["folds.json"], "source_sha256": protocol["files"]["source/hiv_baseline.py"],
                                "prediction": prediction, "metrics": metrics}
                    name = f"baselines/{method}/result.json"
                    save(run / name, baseline)
                    baseline_hashes[name] = AGG.sha256(run / name)
                rows.append(row)
            completions, sources = {}, {}
            for split in ("fold0", "fold1", "fold2", "final"):
                hashes = {}
                for index in (0, 1):
                    name = f"point_{index:03d}/result.json"
                    save(run / "dips" / split / name, {**point_rows[index], "split": split})
                    hashes[name] = AGG.sha256(run / "dips" / split / name)
                    if split != "final":
                        sources[f"{split}/{name}"] = hashes[name]
                completions[split] = {"status": "complete", "error": None, "endpoint": endpoint, "split": split,
                                      "protocol_sha256": protocol_hash, "expected_points": 2, "completed_points": 2,
                                      "verified_points": 2, "files": hashes}
            choice = {"protocol_sha256": protocol_hash, "min_index": 1, "one_se_index": 0,
                      "test_used": False, "truth_used": False, "sources": sources}
            save(run / "dips/selection.json", choice)
            selected = {"selection_sha256": AGG.sha256(run / "dips/selection.json"), "test_used_for_selection": False,
                        "cv_min": {"index": 1, "result": point_rows[1]}, "cv_one_se": {"index": 0, "result": point_rows[0]}}
            save(run / "dips/final/selected_results.json", selected)
            completions["final"]["files"]["selected_results.json"] = AGG.sha256(run / "dips/final/selected_results.json")
            for split, completion in completions.items():
                save(run / "dips" / split / "completion.json", completion)
            save(run / "analysis/summary.json", {"status": "independently_verified", "endpoint": endpoint,
                 "paper_pr2_key": "PseudoR2_train_null", "baseline_methods": 7, "path_points": 8,
                 "selection": choice, "rows": rows, "baseline_result_hashes": baseline_hashes})
    study = {"state": "frozen", "primary_selection": "CV-min", "paper_pr2_key": "PseudoR2_train_null",
             "datasets": entries, "files": {"source/aggregate_repeats.py": AGG.sha256(root / "source/aggregate_repeats.py")}}
    register_study(root, study)
    return study


class AggregateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="hiv-five-seed-fixture-")
        cls.root = Path(cls.temporary.name)
        cls.study = fixture(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_no_formal_statistics(self, result):
        self.assertEqual(result["status"], "incomplete")
        for row in result["primary"] + result["supplementary_1se"]:
            for metric in AGG.METRICS:
                self.assertIsNone(row["metrics"][metric]["mean"])
                self.assertIsNone(row["metrics"][metric]["sd"])
        self.assertEqual(result["tables"], [])
        for filename in AGG.TABLE_FILES:
            self.assertFalse((self.root / "analysis" / filename).exists())

    def test_01_complete_five_seed_output(self):
        result = AGG.aggregate(self.root)
        self.assertEqual(result["status"], "complete", result["failed_runs"][:1])
        self.assertEqual(len(result["primary"]), 200)
        self.assertEqual(len(result["supplementary_1se"]), 25)
        self.assertEqual(result["observed"]["main_seed_records_verified"], 1000)
        seeds = AGG.read_json(self.root / "analysis/seed_metrics.json")
        self.assertEqual(len(seeds["main_records"]), 1000)
        self.assertEqual(len(seeds["supplementary_1se_records"]), 125)
        row = result["primary"][0]
        self.assertEqual(row["method"], "DIPS-PR")
        self.assertNotEqual(row["metrics"]["R2"]["mean"], row["metrics"]["PseudoR2_train_null"]["mean"])
        self.assertNotEqual(row["metrics"]["PseudoR2_train_null"]["mean"], -999.)
        self.assertTrue(all(seed["n_train"] == 4 and seed["n_test"] == 3 for seed in row["seeds"]))
        table = (self.root / "analysis/nrti_main.tex").read_text()
        self.assertIn(r"\begin{tabular}{lcccccc}", table)
        self.assertNotIn("1SE", table)
        self.assertNotIn("CV-min", table)
        self.assertNotIn("CV-selected", table)
        self.assertIn(r"\pm", table)
        self.assertIn(r"\mathbf{", table)

    def test_02_ddof_one_and_metric_independence(self):
        score = AGG.sample_summary([1., 2., 3., 4., 5.])
        self.assertEqual(score["mean"], 3.)
        self.assertAlmostEqual(score["sd"], math.sqrt(2.5))
        self.assertEqual(score["ddof"], 1)
        with self.assertRaises(ValueError):
            AGG.sample_summary([1., 2., 3., 4., None])
        values = AGG.prediction_metrics([1., 2., 3.], [1.1, 2.1, 3.1], [8., 10.])
        self.assertNotEqual(values["R2"], values["PseudoR2_train_null"])

    def test_03_missing_seed_and_stale_table_removal(self):
        for name in AGG.TABLE_FILES:
            (self.root / "analysis" / name).write_text("stale official table")
        changed = dict(self.study, datasets=self.study["datasets"][:-1])
        try:
            register_study(self.root, changed)
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            self.assertEqual(result["missing_runs"], [{"endpoint": "CAI/LEN", "outer_split_seed": 46}])
        finally:
            register_study(self.root, self.study)

    def test_04_duplicate_seed(self):
        changed = dict(self.study, datasets=self.study["datasets"] + [self.study["datasets"][0]])
        try:
            register_study(self.root, changed)
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            self.assertEqual(result["duplicate_runs"][0]["count"], 2)
        finally:
            register_study(self.root, self.study)

    def test_05_null_never_falls_back_to_other_metric(self):
        path = self.root / self.study["datasets"][0]["root"] / "analysis/summary.json"
        original = path.read_text()
        try:
            summary = json.loads(original)
            summary["rows"][0]["metrics"]["PseudoR2_train_null"] = None
            save(path, summary)
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            seed = result["primary"][0]["seeds"][0]
            self.assertIsNotNone(seed["metrics"]["R2"])
            self.assertIsNone(seed["metrics"]["PseudoR2_train_null"])
            self.assertTrue(any("PseudoR2_train_null" in message for message in seed["issues"]))
        finally:
            path.write_text(original)

    def test_06_changed_baseline_hash(self):
        path = self.root / self.study["datasets"][0]["root"] / "baselines/Ridge/result.json"
        original = path.read_text()
        try:
            path.write_text(original + " ")
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            self.assertIn("SHA-256 mismatch", result["failed_runs"][0]["issues"][0])
        finally:
            path.write_text(original)

    def test_07_unverified_summary(self):
        path = self.root / self.study["datasets"][0]["root"] / "analysis/summary.json"
        original = path.read_text()
        try:
            summary = json.loads(original)
            summary["status"] = "complete"
            save(path, summary)
            self.assert_no_formal_statistics(AGG.aggregate(self.root))
        finally:
            path.write_text(original)

    def test_08_historical_seed42_mismatch_is_failure(self):
        previous = self.root / "previous"
        for entry in self.study["datasets"]:
            if entry["outer_split_seed"] == 42:
                source = self.root / entry["root"] / "analysis/summary.json"
                dest = previous / "endpoints" / entry["original_dataset_id"] / "analysis/summary.json"
                save(dest, AGG.read_json(source))
        study = dict(self.study, previous_root="previous")
        try:
            register_study(self.root, study)
            result = AGG.aggregate(self.root)
            self.assertEqual(result["status"], "complete")
            path = previous / "endpoints" / self.study["datasets"][0]["original_dataset_id"] / "analysis/summary.json"
            old = AGG.read_json(path)
            old["rows"][0]["metrics"]["R2"] += 0.01
            save(path, old)
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            self.assertIn("Historical seed42 replication mismatch", result["failed_runs"][0]["issues"][0])
        finally:
            register_study(self.root, self.study)

    def test_09_frozen_source_mismatch_is_failure(self):
        path = self.root / "source/aggregate_repeats.py"
        original = path.read_text()
        try:
            path.write_text(original + "\n")
            result = AGG.aggregate(self.root)
            self.assert_no_formal_statistics(result)
            self.assertIn("SHA-256 mismatch", result["issues"][0])
        finally:
            path.write_text(original)


if __name__ == "__main__":
    unittest.main()
