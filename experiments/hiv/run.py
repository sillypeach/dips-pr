#!/usr/bin/env python3
"""Reproduce the registered HIV grouped-repeat protocol from local source files.

This adapter creates new local registrations; frozen numerical workers are unchanged.
No command downloads data and no training runs without an explicit `run` command.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
FROZEN = HERE / "frozen"
FIXTURES = HERE / "fixtures"
SEEDS = (42, 43, 44, 45, 46)
BASELINES = ("RandomForest", "Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR", "MLP")
METHODS = ("DIPS-PR",) + BASELINES
CLASSES = {
    "NRTI": ("3TC", "ABC", "AZT", "D4T", "DDI", "TDF"),
    "NNRTI": ("EFV", "NVP", "ETR", "RPV", "DOR"),
    "PI": ("FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"),
    "INI": ("RAL", "EVG", "DTG", "BIC", "CAB"), "CAI": ("LEN",),
}
ENDPOINTS = tuple(f"{cls}/{drug}" for cls, drugs in CLASSES.items() for drug in drugs)
GRID_COUNTS = dict(RandomForest=18, Ridge=4, Lasso=3, ElasticNet=9,
                   PoissonGLM=4, **{"RBF-SVR": 12}, MLP=9)
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verify_files(root, files):
    root = Path(root).resolve()
    for relative, expected in files.items():
        target = (root / relative).resolve()
        check(not Path(relative).is_absolute() and target.is_relative_to(root), "Invalid file path: " + relative)
        check(sha(target) == expected, "SHA256 mismatch: " + str(target))


def frozen_preparer():
    verify_files(FROZEN, read(FROZEN / "MANIFEST.json")["files"])
    spec = importlib.util.spec_from_file_location("hiv_frozen_public_prepare", FROZEN / "prepare_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_class(path, expected=None):
    """Read TSV only; no data-dependent filtering other than finite endpoint responses."""
    if expected:
        check(sha(path) == expected["sha256"],
              f"Source snapshot differs: {Path(path).name}. Expected SHA256 {expected['sha256']}; "
              "updated upstream data cannot reproduce the frozen manuscript snapshot.")
    with Path(path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        fields, rows = reader.fieldnames, list(reader)
    check(fields and rows and all(None not in row and all(v is not None for v in row.values()) for row in rows),
          "Empty or malformed TSV: " + str(path))
    positions = [c for c in fields if c.startswith("P") and c[1:].isdigit()]
    check(positions == [f"P{i}" for i in range(1, len(positions) + 1)] and positions, "Invalid position columns")
    check("SeqID" in fields, "SeqID column is required")
    if expected:
        check(len(rows) == expected["n_rows"] and len(positions) == expected["n_items"], "Source dimensions differ")
    return rows, positions


def endpoint_records(rows, positions, drug):
    records = []
    for index, row in enumerate(rows):
        try:
            value = float(row[drug].strip())
        except ValueError:
            continue
        if not math.isfinite(value):
            continue
        check(value >= 0, "Negative phenotype")
        group = row["SeqID"].strip()
        check(bool(group), "Empty SeqID")
        items = [j for j, p in enumerate(positions) if row[p].strip() not in {"", "-", "."}]
        records.append([index, items, value, group])
    check(bool(records), "No finite endpoint responses")
    return records


def split_records(records, seed, preparer):
    # The frozen splitter restores original row order before each split.
    base = dict(description="Published filtered FC, fixed original position-presence encoding; SeqID-grouped outer split",
                group_scope="SeqID", train_indices=[r[0] for r in records], validation_indices=[], test_indices=[])
    for prefix, column in (("tx_", 1), ("y_", 2), ("groups_", 3)):
        base[prefix + "train"] = [r[column] for r in records]
        base[prefix + "final"] = [r[column] for r in records]
        base[prefix + "test"] = []
    return preparer.prepare_split(base, seed)


def prepare_inputs(data_dir):
    """Verify all raw snapshots and all125 hashes, without writing individual records."""
    preparer = frozen_preparer()
    expected_raw = read(FIXTURES / "raw_files.json")
    expected_splits = read(FIXTURES / "split_fingerprints.json")
    table_cache = {}
    for cls in CLASSES:
        name = cls + "_DataSet.txt"
        table_cache[cls] = read_class(Path(data_dir) / name, expected_raw[name])
    prepared = []
    for expected in expected_splits["datasets"]:
        cls, drug = expected["endpoint"].split("/")
        rows, positions = table_cache[cls]
        records = endpoint_records(rows, positions, drug)
        data, folds = split_records(records, expected["outer_split_seed"], preparer)
        actual = preparer.split_fingerprint(data, folds)
        check(all(expected.get(key) == value for key, value in actual.items()),
              "Frozen split fingerprint differs: " + expected["dataset_id"])
        prepared.append(dict(expected=expected, data=data, folds=folds, positions=positions))
    check(len(prepared) == 125, "Expected all125 endpoint/split combinations")
    receipt = dict(status="verified", raw_files=expected_raw, datasets=125,
        core_and_fold_hashes_matched=125, fingerprint_fixture_sha256=sha(FIXTURES / "split_fingerprints.json"),
        frozen_fingerprint_aggregate_sha256=expected_splits["aggregate_sha256"], individual_records_in_receipt=False)
    return prepared, receipt


def copy_workers(destination):
    for subdir in ("source", "audit"):
        for source in sorted((FROZEN / subdir).glob("*.py")):
            target = destination / subdir / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)


def write_endpoint(root, dataset, timestamp):
    expected, data, folds = dataset["expected"], dataset["data"], dataset["folds"]
    endpoint, seed = expected["endpoint"], expected["outer_split_seed"]
    cls, drug = endpoint.split("/")
    endpoint_id = "hiv_" + cls + "_" + drug
    eroot = root / "repeats" / f"seed_{seed}" / "endpoints" / endpoint_id
    eroot.mkdir(parents=True)
    copy_workers(eroot)
    positions = dataset["positions"]
    mapping = dict(drug_class=cls, n_items=len(positions),
        encoding="Fixed original position presence: stripped source token not in {'', '-', '.'}; no amino-acid identity expansion",
        items=[dict(item_id=j, column=p, source_column=p, protein_position=j+1) for j, p in enumerate(positions)])
    metadata = dict(case_id=expected["dataset_id"], dataset_id=expected["dataset_id"], endpoint=endpoint,
        drug_class=cls, drug=drug, seed=42, model_seed=42, outer_split_seed=seed, cv_seed=seed,
        n_items=len(positions), position_columns=positions, source_file=cls+"_DataSet.txt",
        source_sha256=read(FIXTURES / "raw_files.json")[cls+"_DataSet.txt"]["sha256"],
        encoding_mapping_file="feature_mapping.json", truth_available=False,
        group_scope="SeqID only; patient independence is not asserted")
    for name, value in (("data.json", data), ("folds.json", folds), ("metadata.json", metadata),
                        ("feature_mapping.json", mapping)):
        write_new(eroot / name, value)
    protocol = read(FIXTURES / "protocol_settings.json")
    protocol.update(version="hiv-seqid-grouped-repeated-split-v1", state="frozen",
        dataset_id=expected["dataset_id"], endpoint=endpoint, seed=42, model_seed=42,
        outer_split_seed=seed, cv_seed=seed, n_items=len(positions),
        n_train=len(data["y_train"]), n_final=len(data["y_final"]), n_test=len(data["y_test"]),
        baselines=list(BASELINES), baseline_grid_counts=GRID_COUNTS,
        split=f"GroupShuffleSplit(n_splits=1,test_size=.1,random_state={seed}) on SeqID",
        cv=f"KFold(3,shuffle=True,random_state={seed}) on sorted unique training SeqIDs, then map groups to rows; mean fold RMSE; largest tau resolves exact ties",
        final="Use every outer-training row; tx_train == tx_final; no unused validation layer",
        baseline_grid="Original seven grids, 59 candidates; minimum mean three-fold RMSE; first ParameterGrid index resolves ties",
        registered_at=timestamp, resources=dict(memory_gib=8, threads=1),
        release_adapter="New portable registration with identical frozen numerical workers and verified data/fold cores",
        reconstruction_fingerprint=expected)
    protocol["files"] = {str(p.relative_to(eroot)): sha(p) for p in sorted(eroot.rglob("*")) if p.is_file()}
    write_new(eroot / "protocol.json", protocol)
    digest = sha(eroot / "protocol.json")
    write_new(eroot / "registered.json", dict(protocol_sha256=digest, at=timestamp))
    return dict(dataset_id=expected["dataset_id"], endpoint=endpoint, endpoint_id=endpoint_id,
        root=str(eroot.relative_to(root)), outer_split_seed=seed, cv_seed=seed, model_seed=42,
        protocol_sha256=digest, baselines=list(BASELINES))


def prepare(data_dir, output):
    output = Path(output).resolve()
    check(not output.exists(), "Output already exists; choose a new directory: " + str(output))
    prepared, receipt = prepare_inputs(data_dir)
    output.mkdir(parents=True)
    copy_workers(output)
    timestamp = datetime.now(timezone.utc).isoformat()
    entries = [write_endpoint(output, dataset, timestamp) for dataset in prepared]
    write_new(output / "preparation_receipt.json", receipt)
    study = dict(version="hiv-repeated-split-study-v1", state="frozen", registered_at=timestamp,
        datasets=entries, outer_split_seeds=list(SEEDS), model_seed=42, baseline_grid_counts=GRID_COUNTS,
        primary_selection="CV-min", secondary_selection="CV-1SE", paper_pr2_key="PseudoR2_train_null",
        expected=dict(endpoints=25, repeats=5, datasets=125, dips_paths=500, dips_path_points=50500,
            baseline_cv_fits=22125, baseline_final_fits=875, total_models=1000, worker_processes=1375),
        resources=dict(memory_gib=8, threads=1), no_automatic_retry=True,
        release_adapter="Portable sequential runner; fresh registrations are intentionally distinct from original server receipts")
    study["files"] = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob("*")) if p.is_file()}
    write_new(output / "study.json", study)
    write_new(output / "registered.json", dict(study_sha256=sha(output / "study.json"), at=timestamp))
    print(json.dumps(dict(receipt, output=str(output)), indent=2))


def verify_study(root):
    root = Path(root).resolve()
    study = read(root / "study.json")
    check(sha(root / "study.json") == read(root / "registered.json")["study_sha256"], "Study registration differs")
    verify_files(root, study["files"])
    return study


def call_worker(path, args):
    env = dict(os.environ, **THREAD_ENV)
    subprocess.run([sys.executable, str(path), *map(str, args)], env=env, check=True)


def targets(study, endpoint, seed, all_runs):
    if all_runs:
        check(endpoint is None and seed is None, "--all cannot be combined with --endpoint or --seed")
        return study["datasets"]
    check(endpoint in ENDPOINTS and seed in SEEDS, "Specify --endpoint and --seed, or explicitly request --all")
    selected = [e for e in study["datasets"] if e["endpoint"] == endpoint and e["outer_split_seed"] == seed]
    check(len(selected) == 1, "Requested endpoint/split is not registered")
    return selected


def run(root, endpoint=None, seed=None, method=None, all_runs=False, cpu=None):
    root = Path(root).resolve()
    study = verify_study(root)
    entries = targets(study, endpoint, seed, all_runs)
    check(method in METHODS + ("all",) or (all_runs and method is None),
          "Specify --method (or --method all); --all explicitly runs the full registered study")
    check(not all_runs or method in (None, "all"), "--all runs all methods; use an endpoint/split for one method")
    selected_methods = METHODS if method in (None, "all") else (method,)
    # Preflight every selected attempt before starting a potentially long run.
    for entry in entries:
        eroot = root / entry["root"]
        for selected in selected_methods:
            if selected == "DIPS-PR":
                attempts = list((eroot / "dips").glob("*/started.json")) if (eroot / "dips").exists() else []
            else:
                attempts = [p for p in ((eroot / "baselines" / selected / "request.json"),
                                       (eroot / "baselines" / selected / "result.json")) if p.exists()]
            check(not attempts, f"A previous {selected} attempt exists for {entry['dataset_id']}; no automatic retry or overwrite. Use verify for completed results or prepare a new output directory.")
    resource_args = ["--memory-gib", "8"] + (["--cpu", str(cpu)] if cpu is not None else [])
    for entry in entries:
        eroot = root / entry["root"]
        for selected in selected_methods:
            print(f"Running {entry['dataset_id']} {selected}", flush=True)
            if selected == "DIPS-PR":
                for split in ("fold0", "fold1", "fold2", "final"):
                    call_worker(eroot / "source/hiv_dips.py", ["--root", eroot, "--split", split, *resource_args])
            else:
                call_worker(eroot / "source/hiv_baseline.py", ["--root", eroot, "--method", selected, *resource_args])
        complete = (eroot / "dips/final/completion.json").exists() and all(
            (eroot / "baselines" / name / "result.json").exists() for name in BASELINES)
        if complete:
            call_worker(eroot / "source/analyze_endpoint.py", ["--root", eroot])
    if all_runs:
        call_worker(root / "source/aggregate_repeats.py", ["--root", root])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-inputs", help="Check five source snapshots and all125 split fingerprints; do not fit or save observations")
    validate.add_argument("--data-dir", required=True, type=Path)
    prep = commands.add_parser("prepare", help="Create all125 local registered datasets after exact hash validation; no fitting")
    prep.add_argument("--data-dir", required=True, type=Path)
    prep.add_argument("--output", required=True, type=Path)
    fit = commands.add_parser("run", help="Explicitly fit one endpoint/split/method, or --all for the full study")
    fit.add_argument("--root", required=True, type=Path)
    fit.add_argument("--endpoint", choices=ENDPOINTS)
    fit.add_argument("--seed", type=int, choices=SEEDS)
    fit.add_argument("--method", choices=METHODS + ("all",))
    fit.add_argument("--all", action="store_true", dest="all_runs")
    fit.add_argument("--cpu", type=int, help="Optional explicit CPU affinity; otherwise leave process unpinned")
    verify = commands.add_parser("verify", help="Independently audit saved endpoint results without fitting")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--endpoint", required=True, choices=ENDPOINTS)
    verify.add_argument("--seed", required=True, type=int, choices=SEEDS)
    agg = commands.add_parser("aggregate", help="Audit all125 completed runs and generate mean/sample-SD tables; no fitting")
    agg.add_argument("--root", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate-inputs":
            print(json.dumps(prepare_inputs(args.data_dir)[1], indent=2))
        elif args.command == "prepare":
            prepare(args.data_dir, args.output)
        elif args.command == "run":
            run(args.root, args.endpoint, args.seed, args.method, args.all_runs, args.cpu)
        elif args.command == "verify":
            root = args.root.resolve()
            entry = targets(verify_study(root), args.endpoint, args.seed, False)[0]
            eroot = root / entry["root"]
            call_worker(eroot / "source/analyze_endpoint.py", ["--root", eroot])
        elif args.command == "aggregate":
            root = args.root.resolve()
            verify_study(root)
            call_worker(root / "source/aggregate_repeats.py", ["--root", root])
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        parser.exit(2, f"{type(error).__name__}: {error}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
