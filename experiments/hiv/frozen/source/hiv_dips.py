"""Frozen-source HIV Poisson PATH worker: three CV folds, then full final path.

No planted-support field is used. The final path starts only after an immutable
training-CV min/one-SE choice exists; test responses never choose a path point.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
import warnings

from hiv_shared import registered, validate_data, prediction_metrics, warning_rows, configure_resources

VERSION = "hiv-grouped-training-cv-path-v1"
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREAD_ENV)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def save(path, value, exclusive=False):
    def default(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        raise TypeError(type(x).__name__)
    encoded = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=default) + "\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(encoded)
        return
    handle, name = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(encoded)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def check(root):
    plan = registered(root, __file__, {"source/model.py", "source/uv_only_solver.py",
        "source/path_staged_solver.py", "source/regularization_path_experiment.py", "audit/model.py"})
    taus = plan["taus"]
    if len(taus) != 101 or any(not math.isclose(tau, 10**(-2*k/100), rel_tol=1e-13, abs_tol=0)
                              for k, tau in enumerate(taus)):
        raise ValueError("Expected the frozen 101-point descending tau path")
    cfg = plan["config"]
    expected = dict(kappa=.05, max_len=4, pattern_space="closed", support_fraction=.02,
                    support_minimum=2, solver_tol=1e-7, gap_tol=1e-7, radius_mode="certified",
                    top_k_add=200, max_rounds=100, solver_max_iter=10000,
                    propagate_support=True, dual_reference="ray")
    if any(cfg.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected HIV objective/dictionary/certificate settings")
    if plan["staged_options"] != dict(initial_tol=1e-3, tolerance_factor=.1, ready_fraction=.8):
        raise ValueError("Unexpected staged solver settings")
    return plan


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def backends(root):
    """Load only already hash-checked frozen modules, with explicit names."""
    _module(root / "source/model.py", "model")
    _module(root / "source/uv_only_solver.py", "uv_only_solver")
    path = _module(root / "source/path_staged_solver.py", "path_staged_solver")
    helpers = _module(root / "source/regularization_path_experiment.py", "_hiv_path_helpers")
    audit = _module(root / "audit/model.py", "_hiv_independent_audit")
    return path, helpers, audit


def split_data(data, folds, plan, split):
    import numpy as np
    validate_data(data, folds, plan)
    if split == "final":
        tx, y, xt, yt = data["tx_final"], data["y_final"], data["tx_test"], data["y_test"]
        tr, te = data["train_indices"] + data["validation_indices"], data["test_indices"]
    elif split in ("fold0", "fold1", "fold2"):
        fit, valid = folds[int(split[-1])]
        tx, xt = [[data["tx_train"][i] for i in ids] for ids in (fit, valid)]
        y, yt = [[data["y_train"][i] for i in ids] for ids in (fit, valid)]
        tr, te = [[data["train_indices"][i] for i in ids] for ids in (fit, valid)]
    else:
        raise ValueError("Unknown split")
    cfg = dict(plan["config"])
    fraction, minimum = cfg.pop("support_fraction"), cfg.pop("support_minimum")
    cfg["min_support"] = max(minimum, math.floor(fraction * len(y)))
    return tx, np.asarray(y, float), xt, np.asarray(yt, float), tr, te, cfg


def select_scores(scores, taus):
    """Largest-tau min tie; one SE at minimum, then largest eligible tau."""
    import numpy as np
    scores = np.asarray(scores, float)
    if scores.shape != (3, len(taus)) or not np.isfinite(scores).all():
        raise ValueError("Every candidate needs three finite validation RMSEs")
    means = scores.mean(axis=0)
    standard_errors = scores.std(axis=0, ddof=1) / math.sqrt(3)
    best = int(np.argmin(means))
    threshold = float(means[best] + standard_errors[best])
    one_se = int(np.flatnonzero(means <= threshold)[0])
    return dict(min_index=best, one_se_index=one_se, min_tau=taus[best], one_se_tau=taus[one_se],
                mean_rmse=means.tolist(), se_rmse=standard_errors.tolist(),
                one_se_threshold=threshold, se_definition="sample SD of three fold RMSEs divided by sqrt(3)")


def verify_split(root, plan, split):
    import numpy as np
    directory = root / "dips" / split
    completion = load(directory / "completion.json")
    if (completion.get("protocol_sha256") != sha(root / "protocol.json")
            or completion.get("status") != "complete" or completion.get("verified_points") != 101
            or completion.get("completed_points") != 101 or completion.get("expected_points") != 101):
        raise ValueError("Incomplete or mismatched path: " + split)
    required = {"lambda_max.json", "rows.json", "started.json"}
    required.update(f"point_{k:03d}/{name}" for k in range(101)
                    for name in ("result.json", "fit.json", "state.json", "fit.log", "warnings.json"))
    if split == "final":
        required.add("selected_results.json")
    if not required <= set(completion["files"]):
        raise ValueError("Incomplete path evidence manifest")
    for relative, digest in completion["files"].items():
        path = (directory / relative).resolve()
        if not path.is_relative_to(directory.resolve()) or sha(path) != digest:
            raise ValueError("Changed path evidence: " + relative)
    tx, y, xt, yt, tr, te, cfg = split_data(load(root / "data.json"),
        load(root / "folds.json")["folds"], plan, split)
    calibration = load(directory / "lambda_max.json")
    rows = []
    for k, tau in enumerate(plan["taus"]):
        row = load(directory / f"point_{k:03d}/result.json")
        audit = row.get("independent", {})
        if (row.get("verified") is not True or row.get("status") != "complete"
                or row["split"] != split or row["point"] != k or row["tau"] != tau
                or row["protocol_sha256"] != completion["protocol_sha256"]
                or row["training_indices"] != tr or row["evaluation_indices"] != te
                or row["config"] != dict(cfg, lam=tau * calibration["lambda_max"])
                or audit.get("status") != "passed" or audit["gap_per_sample"] > 1.01e-7
                or audit["kkt"] > 1.01e-7 or not audit.get("objective_gap_consistent")
                or not math.isfinite(audit["gap_per_sample"]) or not math.isfinite(audit["kkt"])):
            raise ValueError("Path point lacks a valid full-dictionary certificate")
        if row["warnings"] != load(directory / f"point_{k:03d}/warnings.json"):
            raise ValueError("Point warning evidence differs")
        observed = np.asarray(row["y_evaluation"], float)
        pred = np.asarray(row["prediction"], float)
        if not np.array_equal(observed, yt):
            raise ValueError("Point evaluation labels do not match the frozen split")
        for key, value in prediction_metrics(observed, pred, y).items():
            saved = row["metrics"][key]
            if (value is None and saved is not None) or (value is not None and
                    (saved is None or not np.isclose(value, saved, rtol=1e-12, atol=1e-12))):
                raise ValueError("Point metric cannot be reproduced: " + key)
        if any(key in row for key in ("TP", "FP", "precision", "recall", "f1")):
            raise ValueError("Real HIV data cannot have planted-support recovery metrics")
        rows.append(row)
    return rows


def selection(root, plan):
    rows = [verify_split(root, plan, f"fold{k}") for k in range(3)]
    choice = dict(select_scores([[r["metrics"]["RMSE"] for r in fold] for fold in rows], plan["taus"]),
        protocol_sha256=sha(root / "protocol.json"), test_used=False, truth_used=False,
        sources={f"fold{k}/point_{i:03d}/result.json": sha(root / "dips" / f"fold{k}/point_{i:03d}/result.json")
                 for k in range(3) for i in range(101)})
    destination = root / "dips/selection.json"
    if destination.exists():
        if load(destination) != choice:
            raise ValueError("Saved training-CV choice differs from verified evidence")
    else:
        save(destination, choice, True)
    return choice


def worker(root, split, cpu=None, memory_gib=8):
    configure_resources(cpu, memory_gib)
    root = Path(root).resolve()
    plan = check(root)
    if split not in ("fold0", "fold1", "fold2", "final"):
        raise ValueError("Unknown split")
    directory = root / "dips" / split
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "completion.json").exists():
            verify_split(root, plan, split)
            return load(directory / "completion.json")
        if (directory / "started.json").exists():
            raise ValueError("Incomplete previous attempt; no automatic retry")
        choice = selection(root, plan) if split == "final" else None
        save(directory / "started.json", dict(version=VERSION, endpoint=plan["endpoint"], at=now(), pid=os.getpid(),
            cpu=cpu, memory_gib=memory_gib, protocol_sha256=sha(root / "protocol.json"),
            selection_sha256=sha(root / "dips/selection.json") if choice else None), True)
        rows, files, previous = [], {}, None
        started = time.perf_counter()

        def remember(path):
            files[str(path.relative_to(directory))] = sha(path)

        remember(directory / "started.json")
        try:
            path, helpers, audit = backends(root)
            data = load(root / "data.json")
            tx, y, xt, yt, tr, te, cfg = split_data(data, load(root / "folds.json")["folds"], plan, split)
            save(directory / "progress.json", dict(at=now(), phase="lambda_max", completed_points=0))
            calibration = path.lambda_max_exact(tx, y, cfg["min_support"], cfg["max_len"], cfg["pattern_space"])
            if not calibration["exact"] or not math.isfinite(calibration["lambda_max"]) or calibration["lambda_max"] <= 0:
                raise ValueError("Expected a positive exact training-only lambda_max")
            save(directory / "lambda_max.json", dict(calibration, config=cfg, training_indices=tr,
                data_sha256=plan["files"]["data.json"], folds_sha256=plan["files"]["folds.json"]), True)
            remember(directory / "lambda_max.json")
            for k, tau in enumerate(plan["taus"]):
                pointdir = directory / f"point_{k:03d}"
                pointdir.mkdir()
                config = dict(cfg, lam=tau * calibration["lambda_max"])
                identity = dict(endpoint=plan["endpoint"], point=k, tau=tau, lambda_max=calibration["lambda_max"], split=split,
                    training_indices=tr, evaluation_indices=te, arm="PATH", config=config,
                    protocol_sha256=sha(root / "protocol.json"))
                save(directory / "progress.json", dict(at=now(), phase="fit", point=k, tau=tau,
                    completed_points=len(rows), verified_points=sum(r.get("verified", False) for r in rows)))
                print(now(), split, "POINT_START", k, tau, flush=True)
                captured = []
                try:
                    with warnings.catch_warnings(record=True) as captured:
                        warnings.simplefilter("always")
                        with (pointdir / "fit.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                            tick = time.perf_counter()
                            fitted = path.fit_path_point(tx, y, config, previous=previous, mode="PATH",
                                verbose=True, calibration=calibration, **plan["staged_options"])
                            fit_seconds = time.perf_counter() - tick
                        # The frozen serializer only enters its truth branch for
                        # split='train'. HIV always uses a different name and None.
                        row = dict(identity, **helpers.serialize(fitted, config, xt, yt, y, None, "hiv_" + split))
                        row["metrics"] = prediction_metrics(yt, row["prediction"], y)
                        row.update(fit_seconds=fit_seconds, evaluation_scope=("held-out test; min/one-SE choice already frozen"
                            if split == "final" else "training-only CV validation"),
                            truth_available=False, coefficient_threshold=1e-8,
                            tau_one_note="Analytic intercept-only optimum; not counted as u/v pruning" if k == 0 else None)
                        save(pointdir / "fit.json", row, True)
                        save(pointdir / "state.json", fitted["state"], True)
                        save(directory / "progress.json", dict(at=now(), phase="independent_audit", point=k,
                            completed_points=len(rows), tau=tau))
                        row["independent"] = helpers.audit_fit(audit, tx, y, row)
                        row["verified"] = row["status"] == "complete" and row["independent"]["status"] == "passed"
                        # Audit never repairs the model and no failed state advances.
                        if row["verified"]:
                            previous = fitted["state"]
                except Exception:
                    row = dict(identity, status="failed", verified=False, error=traceback.format_exc())
                row["warnings"] = warning_rows(captured)
                save(pointdir / "warnings.json", row["warnings"], True)
                save(pointdir / "result.json", row, True)
                for artifact in pointdir.iterdir():
                    if artifact.is_file():
                        remember(artifact)
                rows.append(row)
                save(directory / "rows.json", rows)
                print(now(), split, "POINT_END", k, row["status"], row["verified"], flush=True)
                if not row["verified"]:
                    raise RuntimeError(f"Path stopped at point {k}; preserve failed evidence")
            if split == "final":
                report = dict(selection_sha256=sha(root / "dips/selection.json"), test_used_for_selection=False,
                    cv_min=dict(index=choice["min_index"], result=rows[choice["min_index"]]),
                    cv_one_se=dict(index=choice["one_se_index"], result=rows[choice["one_se_index"]]))
                save(directory / "selected_results.json", report, True)
                remember(directory / "selected_results.json")
            check(root)
            status, error = "complete", None
        except Exception:
            status, error = "failed", traceback.format_exc()
            print(error, file=sys.stderr, flush=True)
        if (directory / "rows.json").exists():
            remember(directory / "rows.json")
        completion = dict(version=VERSION, endpoint=plan["endpoint"], split=split, status=status, error=error, completed_at=now(),
            protocol_sha256=sha(root / "protocol.json"), expected_points=101, completed_points=len(rows),
            verified_points=sum(r.get("verified", False) for r in rows), files=files,
            warnings_count=sum(len(r.get("warnings", [])) for r in rows),
            fit_seconds=sum(r.get("fit_seconds", 0) for r in rows),
            audit_seconds=sum(r.get("independent", {}).get("seconds", 0) for r in rows),
            wall_seconds=time.perf_counter()-started)
        save(directory / "completion.json", completion, True)
        save(directory / "progress.json", dict(at=now(), phase=status, completed_points=len(rows),
                                               verified_points=completion["verified_points"]))
        if status == "complete":
            verify_split(root, plan, split)
        return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--split", choices=("fold0", "fold1", "fold2", "final"), required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--memory-gib", type=int, default=8)
    args = parser.parse_args()
    result = worker(args.root, args.split, args.cpu, args.memory_gib)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return int(result["status"] != "complete")


if __name__ == "__main__":
    sys.exit(main())
