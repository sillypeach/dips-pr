"""Aggregate the frozen HIV five-split experiment; never train or select seeds.

Run ``python source/aggregate_repeats.py --root STUDY_ROOT``. Exit 0 means the
entire registered experiment is complete; exit 2 means incomplete. This checks
existing independent-verification evidence and hashes all 404 result JSONs per
run, but does not re-fit models or repeat the full-dictionary path audits.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


SEEDS = (42, 43, 44, 45, 46)
BASELINES = ("RandomForest", "Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR", "MLP")
MAIN_METHODS = ("DIPS-PR",) + BASELINES
SOURCE_METHODS = {"DIPS-PR": "DIPS-PR (CV-min)", **{x: x for x in BASELINES}}
SECONDARY = "DIPS-PR (1SE)"
METRICS = ("R2", "PseudoR2_train_null")
CLASSES = {
    "NRTI": ("3TC", "ABC", "AZT", "D4T", "DDI", "TDF"),
    "NNRTI": ("DOR", "EFV", "ETR", "NVP", "RPV"),
    "PI": ("ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"),
    "INI": ("BIC", "CAB", "DTG", "EVG", "RAL"),
    "CAI": ("LEN",),
}
ENDPOINTS = tuple(f"{group}/{drug}" for group, drugs in CLASSES.items() for drug in drugs)
TABLE_FILES = ("nrti_main.tex", "other_endpoints.tex", "one_se_supplement.tex")


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def safe_value(value):
    return float(value) if finite(value) else None


def contained(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Expected a nonempty relative path: {relative!r}")
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes its registered root: {relative}")
    return result


class Evidence:
    """Cache file hashes within one invocation, never across aggregation runs."""

    def __init__(self):
        self.hashes = {}

    def digest(self, path):
        path = Path(path).resolve()
        if path not in self.hashes:
            self.hashes[path] = sha256(path)
        return self.hashes[path]

    def check(self, root, relative, expected):
        path = contained(root, relative)
        if not isinstance(expected, str) or len(expected) != 64 or self.digest(path) != expected:
            raise ValueError(f"SHA-256 mismatch: {path}")
        return path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_metric(actual, saved):
    if actual is None:
        return saved is None
    return finite(saved) and math.isclose(actual, saved, rel_tol=1e-10, abs_tol=1e-8)


def prediction_metrics(truth, prediction, training):
    """Use raw predictions for R2 and the final-training mean for Poisson PR2."""
    require(isinstance(prediction, list) and len(truth) == len(prediction), "Prediction length mismatch")
    require(all(finite(x) for x in truth + prediction + training), "Nonfinite response or prediction")
    require(truth and training and min(truth + training) >= 0, "Invalid observed responses")

    def deviance(means):
        return max(0.0, 2 * math.fsum(
            y * math.log(y / max(mu, 1e-6)) - y + max(mu, 1e-6) if y > 0 else max(mu, 1e-6)
            for y, mu in zip(truth, means)))

    mean = statistics.fmean(truth)
    sst = math.fsum((y - mean) ** 2 for y in truth)
    null = deviance([statistics.fmean(training)] * len(truth))
    return {
        "R2": 1 - math.fsum((y - p) ** 2 for y, p in zip(truth, prediction)) / sst if sst else None,
        "PseudoR2_train_null": 1 - deviance(prediction) / null if null else None,
    }


def verify_run(study_root, entry, evidence):
    """Return raw rows even on failure, with all official statistics gated later."""
    result = {
        "dataset_id": entry.get("dataset_id"), "endpoint": entry.get("endpoint"),
        "outer_split_seed": entry.get("outer_split_seed"), "root": entry.get("root"),
        "status": "incomplete", "issues": [], "n_train": None, "n_final": None, "n_test": None,
        "rows": {}, "evidence": {},
    }
    try:
        root = contained(study_root, entry.get("root"))
        protocol = read_json(root / "protocol.json")
        for key in ("n_train", "n_final", "n_test"):
            count = protocol.get(key)
            result[key] = count if type(count) is int and count > 0 else None
        summary = read_json(root / "analysis/summary.json")
        rows = summary.get("rows")
        require(isinstance(rows, list), "summary.rows is missing or is not a list")
        row_names = [row.get("method") for row in rows]
        # Preserve values even when verification fails; never silently drop a seed.
        result["rows"] = {row["method"]: row for row in rows if isinstance(row, dict) and "method" in row}
        result["evidence"]["summary_sha256"] = evidence.digest(root / "analysis/summary.json")
        require(summary.get("status") == "independently_verified", "summary is not independently_verified")
        require(summary.get("endpoint") == entry["endpoint"], "summary endpoint mismatch")
        require(summary.get("paper_pr2_key") == METRICS[1], "summary PR2 null convention mismatch")
        require(Counter(row_names) == Counter(list(SOURCE_METHODS.values()) + [SECONDARY]),
                "summary must contain exactly CV-min, 1SE and the seven registered baselines, once each")
        require(summary.get("baseline_methods") == len(BASELINES), "summary baseline count mismatch")

        protocol_hash = evidence.digest(root / "protocol.json")
        result["evidence"]["protocol_sha256"] = protocol_hash
        require(entry.get("protocol_sha256") == protocol_hash, "study protocol hash mismatch")
        require(read_json(root / "registered.json").get("protocol_sha256") == protocol_hash,
                "registered protocol hash mismatch")
        require(protocol.get("state") == "frozen", "run protocol is not frozen")
        require(protocol.get("endpoint") == entry["endpoint"] and protocol.get("dataset_id") == entry["dataset_id"],
                "protocol identity mismatch")
        require(protocol.get("outer_split_seed") == entry["outer_split_seed"], "protocol outer seed mismatch")
        require(protocol.get("cv_seed") == entry["outer_split_seed"], "protocol CV seed mismatch")
        require(protocol.get("model_seed") == protocol.get("seed") == 42, "model seed must remain 42")
        require(protocol.get("paper_pr2_key") == METRICS[1], "protocol PR2 null convention mismatch")
        require(protocol.get("primary_selection") == "CV-min", "protocol primary selection must be CV-min")
        require(Counter(protocol.get("baselines", [])) == Counter(BASELINES), "protocol baseline inventory mismatch")
        files = protocol.get("files")
        require(isinstance(files, dict) and {"data.json", "folds.json", "source/hiv_baseline.py"} <= set(files),
                "run frozen-input/source manifest is incomplete")
        for relative, digest in files.items():
            evidence.check(root, relative, digest)

        data = read_json(root / "data.json")
        for split in ("train", "final", "test"):
            values = data["y_" + split]
            require(isinstance(values, list) and len(values) == protocol.get("n_" + split) and len(values) > 0,
                    f"{split} sample count mismatch")
        require(data["y_train"] == data["y_final"], "Final fitting responses must equal all training responses")
        truth, training = data["y_test"], data["y_final"]

        choice_path = root / "dips/selection.json"
        choice = read_json(choice_path)
        require(summary.get("selection") == choice, "summary selection differs from the saved CV selection")
        require(choice.get("protocol_sha256") == protocol_hash, "selection protocol hash mismatch")
        require(choice.get("test_used") is False and choice.get("truth_used") is False,
                "selection lacks an explicit no-test/no-truth certificate")
        taus = protocol.get("taus")
        require(isinstance(taus, list) and len(taus) > 0, "Missing registered DIPS path")
        npoints = len(taus)
        require(summary.get("path_points") == 4 * npoints, "summary path-point count mismatch")
        sources = choice.get("sources")
        required_sources = {f"fold{k}/point_{i:03d}/result.json" for k in range(3) for i in range(npoints)}
        require(isinstance(sources, dict) and set(sources) == required_sources, "CV selection source inventory mismatch")
        completions = {}
        for split in ("fold0", "fold1", "fold2", "final"):
            directory = root / "dips" / split
            completion = read_json(directory / "completion.json")
            completions[split] = completion
            require(completion.get("status") == "complete" and completion.get("error") is None,
                    f"Incomplete DIPS path: {split}")
            require(completion.get("endpoint") == entry["endpoint"] and completion.get("split") == split
                    and completion.get("protocol_sha256") == protocol_hash, f"DIPS completion identity mismatch: {split}")
            require(all(completion.get(k) == npoints for k in ("expected_points", "completed_points", "verified_points")),
                    f"DIPS completion count mismatch: {split}")
            hashes = completion.get("files", {})
            for index in range(npoints):
                relative = f"point_{index:03d}/result.json"
                require(relative in hashes, f"Missing DIPS result hash: {split}/{relative}")
                evidence.check(directory, relative, hashes[relative])
                if split != "final":
                    require(sources[f"{split}/{relative}"] == hashes[relative], f"CV source hash mismatch: {split}/{relative}")
        result["evidence"]["path_result_hashes_checked"] = 4 * npoints
        result["evidence"]["selection_sha256"] = evidence.digest(choice_path)
        final_dir = root / "dips/final"
        report_path = evidence.check(final_dir, "selected_results.json", completions["final"]["files"].get("selected_results.json"))
        report = read_json(report_path)
        require(report.get("selection_sha256") == evidence.digest(choice_path)
                and report.get("test_used_for_selection") is False, "Final selected-results selection certificate mismatch")
        for method, index_key, report_key in ((SOURCE_METHODS["DIPS-PR"], "min_index", "cv_min"),
                                              (SECONDARY, "one_se_index", "cv_one_se")):
            index = choice.get(index_key)
            require(type(index) is int and 0 <= index < npoints, "Invalid selected DIPS index")
            selected = read_json(final_dir / f"point_{index:03d}/result.json")
            require(report.get(report_key) == {"index": index, "result": selected}, "Selected result differs from final path result")
            require(selected.get("status") == "complete" and selected.get("verified") is True
                    and selected.get("protocol_sha256") == protocol_hash and selected.get("split") == "final"
                    and selected.get("point") == index and selected.get("tau") == taus[index], "Invalid selected DIPS certificate")
            audit = selected.get("independent", {})
            require(audit.get("status") == "passed" and audit.get("objective_gap_consistent") is True
                    and all(finite(audit.get(k)) and 0 <= audit[k] <= 1.01e-7 for k in ("gap_per_sample", "kkt")),
                    "Selected DIPS point lacks a passing full-dictionary certificate")
            require(selected.get("y_evaluation") == truth, "DIPS test responses differ from frozen data")
            require(selected.get("training_indices") == data["train_indices"] + data.get("validation_indices", [])
                    and selected.get("evaluation_indices") == data["test_indices"], "Selected DIPS row order mismatch")
            row = result["rows"][method]
            require(row.get("tau") == selected["tau"] and row.get("lam") == selected.get("config", {}).get("lam"),
                    "DIPS summary selected hyperparameter mismatch")
            score = prediction_metrics(truth, selected["prediction"], training)
            for metric in METRICS:
                require(same_metric(score[metric], selected.get("metrics", {}).get(metric))
                        and same_metric(score[metric], row.get("metrics", {}).get(metric)), f"DIPS metric mismatch: {method}/{metric}")

        baseline_hashes = summary.get("baseline_result_hashes")
        require(isinstance(baseline_hashes, dict)
                and set(baseline_hashes) == {f"baselines/{method}/result.json" for method in BASELINES},
                "Baseline result hash inventory mismatch")
        for method in BASELINES:
            relative = f"baselines/{method}/result.json"
            baseline = read_json(evidence.check(root, relative, baseline_hashes[relative]))
            require(baseline.get("status") == "complete" and baseline.get("model") == method
                    and baseline.get("method") == method and baseline.get("endpoint") == entry["endpoint"],
                    f"Invalid baseline result identity/status: {method}")
            require(baseline.get("protocol_sha256") == protocol_hash and baseline.get("seed") == 42,
                    f"Baseline protocol/seed mismatch: {method}")
            for key, filename in (("data_sha256", "data.json"), ("folds_sha256", "folds.json"),
                                  ("source_sha256", "source/hiv_baseline.py")):
                require(baseline.get(key) == evidence.digest(root / filename), f"Baseline {key} mismatch: {method}")
            score = prediction_metrics(truth, baseline["prediction"], training)
            for metric in METRICS:
                require(same_metric(score[metric], baseline.get("metrics", {}).get(metric))
                        and same_metric(score[metric], result["rows"][method].get("metrics", {}).get(metric)),
                        f"Baseline metric mismatch: {method}/{metric}")
        result["evidence"]["baseline_result_hashes"] = baseline_hashes
        result["status"] = "verified"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError, OverflowError) as error:
        result["issues"].append(f"{type(error).__name__}: {error}")
    return result


def verify_seed42_anchor(root, study, entries, runs, evidence):
    """An explicitly registered historical reference must be reachable and agree."""
    if "previous_root" not in study:
        return
    previous = Path(study["previous_root"])
    if not previous.is_absolute():
        previous = root / previous
    previous = previous.resolve()
    for entry, run in zip(entries, runs):
        if entry["outer_split_seed"] != 42 or run["status"] != "verified":
            continue
        try:
            original = entry.get("original_dataset_id", "hiv_" + entry["endpoint"].replace("/", "_"))
            path = contained(previous, f"endpoints/{original}/analysis/summary.json")
            old = read_json(path)
            require(old.get("status") == "independently_verified" and old.get("endpoint") == entry["endpoint"]
                    and old.get("paper_pr2_key") == METRICS[1], "Invalid historical seed42 summary")
            old_rows = {row["method"]: row for row in old["rows"]}
            for method in SOURCE_METHODS.values():
                require(method in old_rows, f"Missing historical method: {method}")
                for metric in METRICS:
                    value = run["rows"][method]["metrics"].get(metric)
                    prior = old_rows[method]["metrics"].get(metric)
                    equal = value is None and prior is None
                    equal = equal or (finite(value) and finite(prior) and math.isclose(value, prior, rel_tol=1e-8, abs_tol=1e-8))
                    require(equal, f"Historical seed42 replication mismatch: {method}/{metric}; new={value!r}, old={prior!r}")
            new_dips, old_dips = run["rows"][SOURCE_METHODS["DIPS-PR"]], old_rows[SOURCE_METHODS["DIPS-PR"]]
            require(new_dips.get("NZ") == old_dips.get("NZ"), "Historical seed42 DIPS model-size mismatch")
            for key in ("tau", "lam"):
                require(finite(new_dips.get(key)) and finite(old_dips.get(key))
                        and math.isclose(new_dips[key], old_dips[key], rel_tol=1e-8, abs_tol=1e-8),
                        f"Historical seed42 selected DIPS {key} mismatch")
            run["evidence"]["seed42_historical_anchor"] = {
                "status": "passed", "summary": str(path), "sha256": evidence.digest(path),
                "methods": list(MAIN_METHODS), "metrics": list(METRICS), "rtol": 1e-8, "atol": 1e-8,
            }
        except (OSError, ValueError, KeyError, TypeError) as error:
            run["status"] = "incomplete"
            run["issues"].append(f"Seed42 historical audit failure: {type(error).__name__}: {error}")


def sample_summary(values):
    require(len(values) == len(SEEDS) and all(finite(v) for v in values), "Five finite seed values are required")
    return {"mean": statistics.fmean(values), "sd": statistics.stdev(values), "ddof": 1, "n": len(values)}


def make_series(endpoint, method, source_method, by_pair, global_issues):
    seeds, issues = [], list(global_issues)
    for seed in SEEDS:
        candidates = by_pair.get((endpoint, seed), [])
        record = {"seed": seed, "status": "incomplete", "dataset_id": None,
                  "n_train": None, "n_final": None, "n_test": None, "metrics": {m: None for m in METRICS}}
        if len(candidates) != 1:
            record["issues"] = ["missing run" if not candidates else "duplicate endpoint/seed runs"]
        else:
            run = candidates[0]
            record.update({k: run[k] for k in ("dataset_id", "n_train", "n_final", "n_test")})
            row = run["rows"].get(source_method)
            record["issues"] = list(run["issues"])
            if row is None:
                record["issues"].append("missing method row")
            else:
                metrics = row.get("metrics", {})
                for metric in METRICS:
                    value = metrics.get(metric)
                    record["metrics"][metric] = safe_value(value)
                    if not finite(value):
                        record["issues"].append(f"{metric}: missing, null or nonfinite; no seed may be dropped")
                record["present"] = True
            if run["status"] == "verified" and not record["issues"]:
                record["status"] = "verified"
        issues.extend(f"seed {seed}: {issue}" for issue in record["issues"])
        seeds.append(record)
    return {"endpoint": endpoint, "method": method, "source_method": source_method,
            "status": "complete" if not issues else "incomplete", "issues": issues, "seeds": seeds,
            "metrics": {metric: {"mean": None, "sd": None, "ddof": 1,
                                  "n": sum(finite(row["metrics"][metric]) for row in seeds)} for metric in METRICS}}


def tex_escape(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def table_block(group, drugs, lookup, methods, label, secondary=False):
    endpoints = [f"{group}/{drug}" for drug in drugs]
    caption = (f"{group} test performance over five fixed SeqID-grouped splits (seeds 42--46). "
               "Entries are arithmetic mean $\\pm$ sample SD (ddof=1). "
               + ("DIPS-PR uses the prespecified 1SE selection; this is a separate supplementary analysis. " if secondary
                  else "Models are selected by training cross-validation. Bold marks the largest unrounded mean per drug and metric; it does not indicate significance. ")
               + "Poisson PR$^2$ uses each split's final-training response mean as its null.")
    lines = [r"\begin{table}[tbp]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}",
             r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\begin{tabular}{l" + "c" * len(drugs) + "}", r"\toprule",
             "Method & " + " & ".join(tex_escape(drug) for drug in drugs) + r" \\", r"\midrule"]
    for metric, title in (("R2", r"Test $R^2$"), ("PseudoR2_train_null", r"Test Poisson PR$^2$ (training-mean null)")):
        lines.append(r"\multicolumn{" + str(len(drugs) + 1) + r"}{l}{\textit{" + title + r"}} \\")
        maxima = {endpoint: max(lookup[(endpoint, method)]["metrics"][metric]["mean"] for method in methods) for endpoint in endpoints}
        for method in methods:
            cells = []
            for endpoint in endpoints:
                values = lookup[(endpoint, method)]["metrics"][metric]
                cell = f"{values['mean']:.3f} \\pm {values['sd']:.3f}"
                if not secondary and values["mean"] == maxima[endpoint]:
                    cell = r"\mathbf{" + cell + "}"
                cells.append("$" + cell + "$")
            lines.append(tex_escape(method) + " & " + " & ".join(cells) + r" \\")
        if metric == "R2":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def table_outputs(primary, secondary):
    lookup = {(row["endpoint"], row["method"]): row for row in primary}
    other, supplement = [], []
    header = "% Generated only for a complete verified study. Requires \\usepackage{booktabs}.\n"
    for group, drugs in CLASSES.items():
        # Eight PI endpoints are split evenly to keep each methods-as-rows table legible.
        chunks = (drugs[:4], drugs[4:]) if len(drugs) > 6 else (drugs,)
        for i, chunk in enumerate(chunks, 1):
            if group != "NRTI":
                other.append(table_block(group, chunk, lookup, MAIN_METHODS, f"tab:hiv-repeats-{group.lower()}-{i}"))
            secondary_lookup = {(row["endpoint"], row["method"]): row for row in secondary}
            supplement.append(table_block(group, chunk, secondary_lookup, (SECONDARY,), f"tab:hiv-repeats-1se-{group.lower()}-{i}", True))
    return {
        "nrti_main.tex": header + table_block("NRTI", CLASSES["NRTI"], lookup, MAIN_METHODS, "tab:hiv-repeats-nrti"),
        "other_endpoints.tex": header + "\n".join(other),
        "one_se_supplement.tex": header + "\n".join(supplement),
    }


def write_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def aggregate(root):
    root = Path(root).resolve()
    require(root.is_dir(), f"Study root does not exist: {root}")
    out = root / "analysis"
    out.mkdir(exist_ok=True)
    evidence, global_issues, entries = Evidence(), [], []
    study = {}
    try:
        study = read_json(root / "study.json")
        entries = study.get("datasets")
        require(isinstance(entries, list) and all(isinstance(x, dict) for x in entries), "study.datasets must be a list of objects")
        require(study.get("state") == "frozen", "Study is not frozen")
        require(read_json(root / "registered.json").get("study_sha256") == evidence.digest(root / "study.json"),
                "Registered study hash mismatch")
        require(study.get("paper_pr2_key") == METRICS[1] and study.get("primary_selection") == "CV-min",
                "Study metric/selection specification mismatch")
        files = study.get("files")
        require(isinstance(files, dict) and "source/aggregate_repeats.py" in files, "Aggregator is absent from frozen study files")
        for relative, digest in files.items():
            evidence.check(root, relative, digest)
        require(evidence.digest(Path(__file__)) == files["source/aggregate_repeats.py"], "Executing aggregator differs from frozen source")
    except (OSError, ValueError, TypeError, KeyError) as error:
        global_issues.append(f"{type(error).__name__}: {error}")
        if not isinstance(entries, list):
            entries = []
        entries = [entry for entry in entries if isinstance(entry, dict)]
    expected_pairs = {(endpoint, seed) for endpoint in ENDPOINTS for seed in SEEDS}
    pairs, identities, paths = Counter(), Counter(), Counter()
    valid_entries = []
    for i, entry in enumerate(entries):
        endpoint, seed = entry.get("endpoint"), entry.get("outer_split_seed")
        if not isinstance(endpoint, str) or type(seed) is not int or not isinstance(entry.get("dataset_id"), str) or not isinstance(entry.get("root"), str):
            global_issues.append(f"Invalid run identity at study.datasets[{i}]")
            continue
        pair = (endpoint, seed)
        pairs[pair] += 1
        identities[entry["dataset_id"]] += 1
        paths[entry["root"]] += 1
        if str(seed) not in entry["dataset_id"]:
            global_issues.append(f"dataset_id must include its seed: {entry['dataset_id']}")
        valid_entries.append(entry)
    missing = sorted(expected_pairs - set(pairs))
    unexpected = sorted(set(pairs) - expected_pairs)
    duplicates = sorted(pair for pair, n in pairs.items() if n > 1)
    if len(entries) != len(expected_pairs):
        global_issues.append(f"Expected {len(expected_pairs)} runs, found {len(entries)}")
    if missing:
        global_issues.append(f"Missing {len(missing)} endpoint/seed runs; see missing_runs")
    if unexpected:
        global_issues.append(f"Unexpected endpoint/seed runs: {unexpected}")
    if duplicates:
        global_issues.append(f"Duplicate endpoint/seed runs: {duplicates}")
    for label, counter in (("dataset_id", identities), ("root", paths)):
        repeated = sorted(key for key, count in counter.items() if count > 1)
        if repeated:
            global_issues.append(f"Duplicate {label}: {repeated}")
    runs = [verify_run(root, entry, evidence) for entry in valid_entries]
    try:
        verify_seed42_anchor(root, study, valid_entries, runs, evidence)
    except (OSError, ValueError, TypeError) as error:
        global_issues.append(f"Invalid historical anchor configuration: {error}")
    by_pair = defaultdict(list)
    for run in runs:
        by_pair[(run["endpoint"], run["outer_split_seed"])].append(run)
    primary = [make_series(endpoint, method, SOURCE_METHODS[method], by_pair, global_issues)
               for endpoint in ENDPOINTS for method in MAIN_METHODS]
    secondary = [make_series(endpoint, SECONDARY, SECONDARY, by_pair, global_issues) for endpoint in ENDPOINTS]
    complete = not global_issues and all(row["status"] == "complete" for row in primary + secondary)
    if complete:
        for row in primary + secondary:
            row["metrics"] = {metric: sample_summary([seed["metrics"][metric] for seed in row["seeds"]]) for metric in METRICS}
    else:
        # Even complete individual series have no formal means until the whole
        # fixed study is complete. This prevents selectively publishing subsets.
        for row in primary + secondary:
            if row["status"] == "complete":
                row["status"] = "withheld_until_study_complete"
        for filename in TABLE_FILES:
            (out / filename).unlink(missing_ok=True)
    main_records = [{"endpoint": row["endpoint"], "method": row["method"], "source_method": row["source_method"], **seed}
                    for row in primary for seed in row["seeds"]]
    secondary_records = [{"endpoint": row["endpoint"], "method": row["method"], **seed} for row in secondary for seed in row["seeds"]]
    expected = {"endpoints": len(ENDPOINTS), "runs": len(expected_pairs), "main_method_endpoint_combinations": len(ENDPOINTS) * len(MAIN_METHODS),
                "main_seed_records": len(expected_pairs) * len(MAIN_METHODS), "main_metric_values": len(expected_pairs) * len(MAIN_METHODS) * len(METRICS)}
    observed = {"run_entries": len(entries), "unique_endpoint_seed_pairs": len(pairs), "verified_runs": sum(run["status"] == "verified" for run in runs),
                "main_method_endpoint_combinations_with_all_records": sum(all(seed.get("present") for seed in row["seeds"]) for row in primary),
                "main_seed_records_present": sum(bool(record.get("present")) for record in main_records),
                "main_seed_records_verified": sum(record["status"] == "verified" for record in main_records)}
    result = {
        "schema_version": "hiv-five-seed-aggregate-v1", "status": "complete" if complete else "incomplete",
        "study_sha256": evidence.digest(root / "study.json") if (root / "study.json").is_file() else None,
        "aggregator_sha256": evidence.digest(Path(__file__)), "seeds": list(SEEDS), "primary_selection": "CV-min",
        "primary_method_display": "DIPS-PR", "supplementary_selection": "1SE", "paper_pr2_key": METRICS[1],
        "statistic": "Equal-weight arithmetic mean and sample standard deviation across the five test scores; ddof=1",
        "uncertainty": "Descriptive variation across overlapping grouped splits; not an independent-sample SE or a confidence interval",
        "integrity_scope": "Frozen inputs/sources, run identity, all path-result hashes, selected DIPS certificates, baseline-result hashes, and final test metric recomputation; existing independent full-path audits are reused",
        "expected": expected, "observed": observed, "issues": global_issues,
        "missing_runs": [{"endpoint": endpoint, "outer_split_seed": seed} for endpoint, seed in missing],
        "duplicate_runs": [{"endpoint": endpoint, "outer_split_seed": seed, "count": pairs[(endpoint, seed)]} for endpoint, seed in duplicates],
        "failed_runs": [{k: run[k] for k in ("dataset_id", "endpoint", "outer_split_seed", "root", "issues")} for run in runs if run["status"] != "verified"],
        "primary": primary, "supplementary_1se": secondary,
        "tables": list(TABLE_FILES) if complete else [],
    }
    seed_output = {"status": result["status"], "seeds": list(SEEDS), "paper_pr2_key": METRICS[1],
                   "main_records": main_records, "supplementary_1se_records": secondary_records,
                   "run_evidence": [{k: v for k, v in run.items() if k != "rows"} for run in runs]}
    write_text(out / "seed_metrics.json", json.dumps(seed_output, indent=2, allow_nan=False) + "\n")
    if complete:
        for filename, content in table_outputs(primary, secondary).items():
            write_text(out / filename, content)
    write_text(out / "aggregate_repeats.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    readme = ["HIV repeated grouped-split analysis", f"Status: {result['status']}", "",
              "Fixed outer/CV seeds: 42, 43, 44, 45, 46. Model seed: 42.",
              "The main analysis is CV-min, displayed as DIPS-PR, plus seven baselines.",
              "1SE is supplementary and never substitutes for a primary result.",
              "R2 uses raw predictions. Poisson PR2 is PseudoR2_train_null, using that split's final-training response mean.",
              "Means weight the five test scores equally. SD uses ddof=1; no seed is dropped, selected or weighted by test n.",
              "Splits may overlap: SD describes repeated-split variation, not independent-sample SE, CI or a significance test.",
              "Every seed's train/final/test n and both metrics are retained in seed_metrics.json.",
              "These data have previously been analyzed; this is not an external validation study.",
              result["integrity_scope"] + ".", "",
              f"Expected: {expected['endpoints']} endpoints, {expected['runs']} runs, {expected['main_method_endpoint_combinations']} main method/endpoint combinations, {expected['main_seed_records']} main seed records.",
              f"Observed: {observed['verified_runs']} verified runs; {observed['main_seed_records_verified']} verified main seed records.", ""]
    if complete:
        readme += ["All formal means and SDs use all five fixed seeds.",
                   "nrti_main.tex: methods as rows, six NRTI drugs as columns, R2 and PR2 panels.",
                   "other_endpoints.tex: class-specific tables; PI is split into two four-drug blocks.",
                   "one_se_supplement.tex: separate supplementary 1SE results.",
                   "LaTeX fragments require booktabs; entries have three decimals; bold compares unrounded means only."]
    else:
        readme += ["No formal means, SDs or publication tables are released while the fixed study is incomplete.",
                   "See aggregate_repeats.json issues, missing_runs, duplicate_runs, failed_runs and per-series issues."]
    write_text(out / "README.txt", "\n".join(readme) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = aggregate(args.root)
    print(json.dumps({"status": result["status"], "expected": result["expected"], "observed": result["observed"],
                      "issues": result["issues"], "failed_runs": len(result["failed_runs"]), "output": str(args.root.resolve() / "analysis")}, indent=2))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
