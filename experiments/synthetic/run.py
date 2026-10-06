#!/usr/bin/env python3
"""Portable orchestration around byte-exact frozen synthetic experiment code."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile
import traceback

HERE = Path(__file__).resolve().parent
THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREADS)
RAW = ["Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR", "RandomForest", "MLP"]
LCM_SHA = "e294c1a015a45897cf0f173ee3d1b27c984ad656a7992a4c962c3c0942e7c0f7"
LCM_URL = "https://research.nii.ac.jp/~uno/code/lcm53.zip"


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


def entries():
    return load(HERE / "inputs/manifest.json")["entries"]


def scope_entries(scope="matched", requested=None):
    rows = entries()
    if requested:
        wanted = set(requested)
        if not wanted <= {e["dataset_id"] for e in rows}:
            raise ValueError("Unknown dataset ID")
        return [e for e in rows if e["dataset_id"] in wanted]
    conditions = {("count", 2), ("count", 4), ("count", 8), ("length", 5)}
    if scope == "pruning":
        conditions |= {("count", 12), ("length", 6), ("shared", 30)}
    return rows if scope == "all" else [e for e in rows if (e["family"], e["level"]) in conditions]


def verify_inputs():
    for relative, expected in load(HERE / "frozen/SHA256.json").items():
        if sha(HERE / relative) != expected:
            raise ValueError("Changed frozen source: " + relative)
    for row in entries():
        for key, digest in (("data", "sha256"), ("folds", "folds_sha256")):
            if sha(HERE / "inputs" / row[key]) != row[digest]:
                raise ValueError("Changed input: " + row[key])
    return {"datasets": len(entries()), "input_and_source_hashes": "passed"}


def verify_generator():
    import numpy as np
    verify_inputs()
    sys.path.insert(0, str(HERE / "frozen/generators"))
    import pattern_count_extension as count
    import length_ramp_recovery as length
    baseline = module(HERE / "frozen/raw/extension_baseline_benchmark.py", "_gen_raw")
    largest = 0.0
    checked = []
    # These generator families cover all count/length inputs. Shared S30 is
    # supplied as exact original data; it is outside Tables 1/2.
    for e in entries():
        if e["family"] not in ("count", "length"):
            continue
        generator = count if e["family"] == "count" else length
        new = generator.generate(generator.case_for(e["level"]), e["seed"])
        old = load(HERE / "inputs" / e["data"])
        if new.keys() != old.keys():
            raise ValueError("Generator fields changed")
        for k in new:
            if k != "true_mean" and new[k] != old[k]:
                raise ValueError(f"Generator mismatch: {e['dataset_id']}/{k}")
        np.testing.assert_allclose(new["true_mean"], old["true_mean"], rtol=1e-14, atol=1e-12)
        largest = max(largest, float(np.max(np.abs(np.array(new["true_mean"]) - old["true_mean"]))))
        if baseline.jsonable(baseline.folds_for(1400, e["seed"])) != load(HERE / "inputs" / e["folds"])["folds"]:
            raise ValueError("Regenerated fold mismatch")
        checked.append(e["dataset_id"])
    return {"status": "passed", "regenerated_datasets": len(checked), "dataset_ids": checked,
            "observations_truth_and_splits": "exact", "true_mean_max_absolute_roundoff": largest,
            "shared_S30": "original byte hashes verified; generator not included"}


def prepare(output):
    verify_inputs()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Use a fresh output directory; original or partial runs are never overwritten")
    raw = output / "raw"
    raw.mkdir(parents=True)
    shutil.copytree(HERE / "inputs/data", raw / "data")
    save(raw / "datasets_registered.json", {"entries": entries(), "version": "public-synthetic-inputs-v1"})
    baseline = module(HERE / "frozen/raw/extension_baseline_benchmark.py", "_prepare_raw")
    baseline.prepare(raw)
    for e in entries():
        if sha(raw / e["folds"]) != e["folds_sha256"]:
            raise ValueError("Generated fold bytes differ from archived folds")
    path = module(HERE / "frozen/path/regularization_path_experiment.py", "_prepare_path")
    path.prepare(argparse.Namespace(root=output / "dips", data_root=raw,
        model=HERE / "frozen/path/model.py", audit_model=HERE / "frozen/audit/model.py"))
    # A new public-run registration uses the historical numerical settings.
    # Its hashes intentionally differ from registrations containing old host paths.
    structure = output / "structure"
    shutil.copytree(HERE / "frozen/structure", structure / "source")
    (structure / "reference").symlink_to(Path("../raw"), target_is_directory=True)
    plan = load(HERE / "structure_settings.json")
    plan.update(datasets=entries(), methods=["LCM-Lasso", "RF-paths"],
        parent_baseline_protocol_sha256=sha(raw / "protocol.json"),
        parent_path_protocol_sha256=sha(output / "dips/protocol.json"),
        files={"source/" + p.name: sha(p) for p in (structure / "source").glob("*.py")})
    plan["files"]["vendor/lcm53.zip"] = LCM_SHA
    save(structure / "protocol.json", plan)
    save(structure / "registered.json", {"version": plan["version"], "protocol_sha256": sha(structure / "protocol.json")})
    save(output / "release_run.json", {"version": "synthetic-public-run-v1", "default_scope": "matched",
        "default_datasets": [e["dataset_id"] for e in scope_entries()],
        "input_manifest_sha256": sha(HERE / "inputs/manifest.json"),
        "protocols": {key: sha(output / key / "protocol.json") for key in ("raw", "dips", "structure")}})
    return {"prepared": str(output), "available_datasets": 45, "default_run_datasets": 20}


def check_run(output):
    verify_inputs()
    receipt = load(output / "release_run.json")
    if receipt["input_manifest_sha256"] != sha(HERE / "inputs/manifest.json"):
        raise ValueError("Input manifest changed")
    for key, expected in receipt["protocols"].items():
        if sha(output / key / "protocol.json") != expected:
            raise ValueError("Changed run protocol: " + key)
    # Validate the copies that workers and aggregators actually read, before
    # importing any code from the output tree. Verifying the bundled originals
    # alone cannot detect a changed file in a prepared run.
    raw = output / "raw"
    raw_plan = load(raw / "protocol.json")
    raw_registration = load(raw / "registered.json")
    if (raw_registration["protocol_sha256"] != sha(raw / "protocol.json")
            or raw_registration["manifest_sha256"] != raw_plan["manifest_sha256"]
            or raw_registration["source_sha256"] != raw_plan["source_sha256"]):
        raise ValueError("Changed raw registration")
    raw_files = {raw_plan["source"]: raw_plan["source_sha256"],
                 raw_plan["manifest"]: raw_plan["manifest_sha256"]}
    for entry in raw_plan["datasets"]:
        raw_files[entry["data"]] = entry["sha256"]
        raw_files[entry["folds"]] = entry["folds_sha256"]
    verify_run_files(raw, raw_files)
    dips = output / "dips"
    if load(dips / "registered.json")["protocol_sha256"] != sha(dips / "protocol.json"):
        raise ValueError("Changed DIPS registration")
    verify_run_files(dips, load(dips / "protocol.json")["hashes"])
    structure = output / "structure"
    if load(structure / "registered.json")["protocol_sha256"] != sha(structure / "protocol.json"):
        raise ValueError("Changed structure registration")
    verify_run_files(structure, load(structure / "protocol.json")["files"],
                     optional={"vendor/lcm53.zip"})
    # The usual reference is a relative symlink to raw/. If a user replaces it
    # with a copied directory, validate that copy too rather than trusting it.
    reference = structure / "reference"
    if reference.resolve() != raw.resolve():
        verify_run_files(reference, dict(raw_files,
            **{"protocol.json": sha(raw / "protocol.json"),
               "registered.json": sha(raw / "registered.json")}))


def verify_run_files(root, expected_files, optional=()):
    root = Path(root).resolve()
    for relative, expected in expected_files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Run file escapes registered root: " + relative)
        if relative in optional and not path.exists():
            continue
        if not path.is_file() or sha(path) != expected:
            raise ValueError("Changed or missing run file: " + str(path))


def install_lcm(output, archive=None, fetch=False):
    check_run(output)
    structure = output / "structure"
    destination = structure / "vendor/lcm53.zip"
    if destination.exists():
        if sha(destination) != LCM_SHA:
            raise ValueError("Existing LCM archive hash mismatch")
    else:
        destination.parent.mkdir(exist_ok=True)
        if archive:
            if sha(archive) != LCM_SHA:
                raise ValueError("LCM archive hash mismatch")
            shutil.copyfile(archive, destination)
        elif fetch:
            import urllib.request
            # Network is used only by the explicit fetch-lcm command.
            payload = urllib.request.urlopen(LCM_URL, timeout=60).read()
            if hashlib.sha256(payload).hexdigest() != LCM_SHA:
                raise ValueError("Downloaded LCM archive hash mismatch; nothing installed")
            destination.write_bytes(payload)
        else:
            raise ValueError("Provide --lcm-archive or explicitly run fetch-lcm")
    worker = module(structure / "source/structure_baseline_benchmark.py", "_lcm_build")
    binary = structure / "runtime/lcm"
    if binary.exists():
        worker.check_binary(structure, load(structure / "protocol.json"), binary)
        return {"binary": str(binary), "already_built": True}
    return worker.build_miner(structure)


def structure_worker(output, dataset_id, method, memory_gib):
    """Call unchanged numerical routines; optional LCM is unnecessary for RF."""
    check_run(output)
    root = output / "structure"
    sys.path.insert(0, str(root / "source"))
    worker = module(root / "source/structure_baseline_benchmark.py", "_structure")
    plan = load(root / "protocol.json")
    for relative, expected in plan["files"].items():
        if method == "RF-paths" and relative.startswith("vendor/"):
            continue
        if sha(root / relative) != expected:
            raise ValueError("Changed structure dependency: " + relative)
    entry = next(e for e in plan["datasets"] if e["dataset_id"] == dataset_id)
    directory = root / "jobs" / dataset_id / method
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        result = worker.verify(root, plan, entry, method)
        if result["status"] != "complete":
            raise ValueError("Prior failure retained; use a fresh output directory")
        return result
    if (directory / "started.json").exists():
        raise ValueError("Prior incomplete attempt retained; use a fresh output directory")
    worker.limits(None, memory_gib)
    worker.save(directory / "started.json", {"at": worker.now(), "pid": os.getpid()}, True)
    result = {"dataset_id": dataset_id, "method": method, "family": entry["family"],
        "level": entry["level"], "seed": entry["seed"], "protocol_sha256": sha(root / "protocol.json")}
    try:
        env = worker.common(root).runtime_environment()
        if any(env[k] != v for k, v in plan["versions"].items()):
            raise ValueError("Install the pinned numerical package versions")
        result.update(worker.run_lasso(root, plan, entry, root / "runtime/lcm", directory)
            if method == "LCM-Lasso" else worker.run_rf(root, plan, entry, directory))
        result.update(status="complete", environment=env)
    except Exception:
        result.update(status="failed", traceback=traceback.format_exc())
    worker.save(directory / "result.json", result, True)
    worker.save(directory / "integrity.json", {"sha256": sha(directory / "result.json")}, True)
    result = worker.verify(root, plan, entry, method)
    if result["status"] != "complete":
        raise RuntimeError(result["traceback"])
    return {"dataset_id": dataset_id, "method": method, "status": "complete"}


def run(args):
    output = args.output.resolve()
    check_run(output)
    selected = scope_entries(args.scope, args.datasets)
    ids = [e["dataset_id"] for e in selected]
    methods = args.methods.split(",")
    if not set(methods) <= {"raw", "dips", "lcm", "rf"}:
        raise ValueError("--methods accepts raw,dips,lcm,rf")
    if "lcm" in methods:
        install_lcm(output, args.lcm_archive)
    if "raw" in methods:
        for did in ids:
            for method in RAW:
                subprocess.run([sys.executable, str(output / "raw/source/extension_baseline_benchmark.py"),
                    "worker", "--root", str(output / "raw"), "--dataset-id", did, "--method", method,
                    "--memory-gib", str(args.memory_gib)], check=True)
    if "dips" in methods:
        cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
        if args.workers < 1 or args.workers > len(cpus):
            raise ValueError("Invalid number of workers")
        subprocess.run([sys.executable, str(output / "dips/source/regularization_path_experiment.py"), "run",
            "--root", str(output / "dips"), "--mode", "full", "--workers", str(args.workers),
            "--memory-gib", str(min(args.memory_gib, 8)), "--timeout", "0", "--prepare-timeout", "0",
            "--cpus", *map(str, cpus[:args.workers]), "--datasets", *ids], check=True)
    for group, method in (("lcm", "LCM-Lasso"), ("rf", "RF-paths")):
        if group in methods:
            for did in ids:
                subprocess.run([sys.executable, str(HERE / "run.py"), "_structure", "--output", str(output),
                    "--dataset", did, "--method", method, "--memory-gib", str(args.memory_gib)], check=True)
    return aggregate(output, args.scope, args.datasets,
                     require_complete=set(methods) == {"raw", "dips", "lcm", "rf"})


def verified_path_rows(root, dataset_id, split):
    """Reject changed/incomplete native worker artifacts before aggregation."""
    directory = root / "jobs/full" / dataset_id / split / "PATH"
    process = load(directory / "process.json")
    if process.get("status") != "exited" or process.get("returncode") != 0:
        raise ValueError("PATH worker did not exit successfully")
    for relative, expected in process["output_hashes"].items():
        candidate = (directory / relative).resolve()
        if not candidate.is_relative_to(directory.resolve()) or sha(candidate) != expected:
            raise ValueError("Changed PATH worker output: " + relative)
    if not load(directory / "completion.json").get("all_verified"):
        raise ValueError("PATH independent numerical checks did not pass")
    rows = load(directory / "rows.json")
    if len(rows) != 101 or not all(r.get("verified") for r in rows):
        raise ValueError("PATH has fewer than101 verified points")
    for index, row in enumerate(rows):
        if row != load(directory / f"point_{index:02d}/result.json"):
            raise ValueError("PATH summary differs from saved point")
    return rows


def aggregate(output, scope="matched", requested=None, require_complete=False):
    """Public schema: one row per dataset/method/selection with native metrics."""
    check_run(output)
    baseline = module(output / "raw/source/extension_baseline_benchmark.py", "_aggregate_raw")
    path = module(output / "dips/source/regularization_path_experiment.py", "_aggregate_path")
    structure = module(output / "structure/source/structure_baseline_benchmark.py", "_aggregate_structure")
    plan = load(output / "dips/protocol.json")
    rows, missing = [], []
    for e in scope_entries(scope, requested):
        did = e["dataset_id"]
        common = {k: e[k] for k in ("dataset_id", "family", "level", "seed")}
        for method in RAW:
            source = output / "raw/jobs" / did / method / "result.json"
            if not source.exists():
                missing.append({"dataset_id": did, "method": method}); continue
            r = baseline.verify_result(output / "raw", e, method)
            if r["status"] != "complete":
                raise ValueError("Failed raw result: " + str(source))
            rows.append(dict(common, method=method, selection="CV-RMSE", metrics=r["metrics"],
                recovery=None, source=str(source.relative_to(output)), source_sha256=sha(source)))
        source = output / "dips/jobs/full" / did / "train/PATH/rows.json"
        if source.exists():
            rs = verified_path_rows(output / "dips", did, "train")
            for fold in range(3):
                verified_path_rows(output / "dips", did, f"fold{fold}")
            choice = path.cv_choice(output / "dips", plan, e)
            if choice is None:
                raise ValueError("Incomplete verified DIPS CV")
            for method, selected in [("DIPS-PATH", rs[choice["selected_point"]]),
                    ("DIPS-ORACLE-F1", max(rs, key=lambda r: (r["f1"], r["precision"], r["tau"])) )]:
                rec = {k: selected[k] for k in ("TP", "FP", "NZ", "precision", "recall", "f1")}
                rec["FN"] = e["truths"] - rec["TP"]
                rows.append(dict(common, method=method, selection="oracle-F1" if method.endswith("F1") else "CV-RMSE",
                    metrics=selected["metrics"], recovery=rec, point=selected["point"], tau=selected["tau"],
                    source=str(source.relative_to(output)), source_sha256=sha(source)))
        else:
            missing.append({"dataset_id": did, "method": "DIPS-PATH/ORACLE"})
        for method in ("LCM-Lasso", "RF-paths"):
            source = output / "structure/jobs" / did / method / "result.json"
            if not source.exists():
                missing.append({"dataset_id": did, "method": method}); continue
            r = structure.verify(output / "structure", load(output / "structure/protocol.json"), e, method)
            if r["status"] != "complete":
                raise ValueError("Failed structure result: " + str(source))
            rows.append(dict(common, method=method, selection="CV-RMSE", metrics=r["metrics"], recovery=r["recovery"],
                source=str(source.relative_to(output)), source_sha256=sha(source)))
    groups = []
    for family, level, method in sorted({(r["family"], r["level"], r["method"]) for r in rows}):
        group = [r for r in rows if (r["family"], r["level"], r["method"]) == (family, level, method)]
        if len(group) != 5:
            continue  # Partial conditions never become published means.
        values = {}
        for section in ("metrics", "recovery"):
            if group[0][section] is None:
                continue
            values[section] = {key: {"mean": statistics.mean(r[section][key] for r in group),
                "sample_sd": statistics.stdev(r[section][key] for r in group)}
                for key in group[0][section] if isinstance(group[0][section][key], (int, float))}
        groups.append(dict(family=family, level=level, method=method, n=5, **values))
    result = {"schema": "synthetic-seed-metrics-v1", "scope": scope, "rows": rows, "groups": groups,
        "complete": not missing, "missing": missing,
        "table1_methods": ["DIPS-PATH", "LCM-Lasso", *RAW],
        "table2_methods": ["DIPS-ORACLE-F1", "LCM-Lasso", "RF-paths"]}
    save(output / "aggregate.json", result)
    if require_complete and missing:
        raise ValueError(f"{len(missing)} required method/dataset outputs missing; aggregate.json records them")
    return {"aggregate": str(output / "aggregate.json"), "rows": len(rows), "complete": not missing, "missing": len(missing)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "verify-inputs", "verify-generator", "run", "aggregate", "fetch-lcm", "build-lcm", "_structure"])
    parser.add_argument("--output", type=Path, default=Path("runs/synthetic"))
    parser.add_argument("--scope", choices=["matched", "pruning", "all"], default="matched")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--methods", default="raw,dips,lcm,rf")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--lcm-archive", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--method", choices=["LCM-Lasso", "RF-paths"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.command == "verify-inputs": result = verify_inputs()
    elif args.command == "verify-generator": result = verify_generator()
    elif args.command == "prepare": result = prepare(args.output)
    elif args.command == "run": result = run(args)
    elif args.command == "aggregate": result = aggregate(args.output, args.scope, args.datasets, args.require_complete)
    elif args.command in ("fetch-lcm", "build-lcm"):
        result = install_lcm(args.output, args.lcm_archive, args.command == "fetch-lcm")
    else: result = structure_worker(args.output, args.dataset, args.method, args.memory_gib)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
