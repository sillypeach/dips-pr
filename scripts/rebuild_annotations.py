"""Recheck the archived seed-42 rule annotations from user-provided source data.

Descriptive metadata reconstruction only: no fitting, causal inference, or
mutation-specific coefficient estimation. Individual records are never exported.
"""
import argparse
import csv
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    adapter = module(ROOT / "experiments/hiv/run.py", "public_input_adapter")
    annotation_path = ROOT / "paper_snapshot/annotation_method.py"
    expected = json.loads((ROOT / "paper_snapshot/source_hashes.json").read_text())["source_sha256"]["annotation_method.py"]
    if hashlib.sha256(annotation_path.read_bytes()).hexdigest() != expected:
        raise ValueError("Changed annotation kernel")
    annotator = module(annotation_path, "frozen_rule_annotation")
    prepared, receipt = adapter.prepare_inputs(args.data_dir)
    raw = {}
    for group in adapter.CLASSES:
        rows, _ = adapter.read_class(args.data_dir / (group + "_DataSet.txt"))
        for row in rows:
            by_position = defaultdict(list)
            for token in row["CompMutList"].split(","):
                token = token.strip()
                match = annotator.GENERAL.fullmatch(token)
                if match:
                    by_position[int(match[2])].append(token)
            row["_mutations"] = dict(by_position)
        raw[group] = rows
    contexts = {}
    for item in prepared:
        e = item["expected"]
        if e["outer_split_seed"] != 42:
            continue
        endpoint = e["endpoint"]
        rows = raw[endpoint.split("/")[0]]
        contexts[endpoint] = {"rows": [rows[i] for i in item["data"]["train_indices"]]}
    reference_path = ROOT / "paper_snapshot/hiv_term_mutation_annotations.csv"
    with reference_path.open(encoding="utf-8", newline="") as f:
        reference = list(csv.DictReader(f))
    verified = []
    for row in reference:
        positions = [int(p) for p in row["positions"].split(";")]
        term = dict(endpoint=row["endpoint"], columns=[f"P{p}" for p in positions],
                    item_ids=[p - 1 for p in positions], training_support=int(row["training_support"]))
        got = annotator.annotate(term, contexts)
        checks = {
            "modal_fully_resolved_joint_profile": got["modal_profile"],
            "joint_count": str(got["joint_count"]),
            "joint_count_over_original_support": got["joint_count_over_original_support"],
            "resolved_rows": str(got["resolved_rows"]),
            "excluded_rows": str(got["excluded_rows"]),
            "excluded_reason_row_counts_nonexclusive": json.dumps(got["excluded_reason_row_counts_nonexclusive"], sort_keys=True),
            "distinct_resolved_joint_profiles": str(got["distinct_resolved_joint_profiles"]),
            "modal_tie_count": str(len(got["modal_ties"])),
            "tied_profiles": json.dumps(got["modal_ties"]),
            "tiebreak": got["tiebreak"], "annotation_status": got["annotation_status"],
            "n_fit": str(len(contexts[row["endpoint"]]["rows"])),
        }
        if any(row[key] != value for key, value in checks.items()):
            raise ValueError("Annotation differs from reference for " + row["endpoint"] + " " + row["positions"])
        verified.append({**row, **checks})
    if len(verified) != 1451:
        raise ValueError("Incomplete reference catalogue")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "verified_annotations.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(reference[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(verified)
    report = dict(status="verified", terms=len(verified), endpoints=len(contexts),
                  raw_input_hashes_verified=5, seed=42, fitting_performed=False,
                  scope="Observed metadata annotation of archived position rules, not mutation-specific effects.")
    (args.output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
