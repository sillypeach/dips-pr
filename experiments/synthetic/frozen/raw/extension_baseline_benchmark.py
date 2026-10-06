"""Frozen matched-data baseline benchmark, without fitting or importing truth patterns.

prepare registers existing data; run dispatches separate bounded subprocesses;
worker fits one registered method; aggregate/check independently recompute metrics.
The estimator factories and grids reproduce new_dgp_benchmark.py (20260923).
Only Python stdlib is imported before worker resource limits are applied.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

VERSION = "matched-extension-baselines-v1"
METHODS = ["Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR",
           "RandomForest", "MLP", "MLP-scaled-y-sensitivity"]
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
              "BLIS_NUM_THREADS")}
os.environ.update(THREAD_ENV)


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    with Path(path).open() as stream:
        return json.load(stream)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def encoded(value):
    return (json.dumps(jsonable(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    """Atomic write; only run products use replace. Frozen files use exclusive()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(encoded(value))


def relative_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Registered path escapes benchmark root")
    return path


def baseline_specs(seed):
    from sklearn.base import clone
    from sklearn.compose import TransformedTargetRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import ElasticNet, Lasso, PoissonRegressor, Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    def scaled(estimator):
        return Pipeline([("scale", StandardScaler()), ("model", estimator)])

    specs = {
        "Ridge": (scaled(Ridge()), {"model__alpha": [.01, .1, 1., 10., 100., 1000.]}),
        "Lasso": (scaled(Lasso(max_iter=10000)),
                  {"model__alpha": [.001, .01, .1, 1., 10.]}),
        "ElasticNet": (scaled(ElasticNet(max_iter=10000)),
                       {"model__alpha": [.001, .01, .1, 1., 10.],
                        "model__l1_ratio": [.2, .5, .8]}),
        "PoissonGLM": (scaled(PoissonRegressor(max_iter=5000)),
                       {"model__alpha": [.001, .01, .1, 1.]}),
        "RBF-SVR": (scaled(SVR(kernel="rbf")),
                    {"model__C": [.1, 1., 10., 100., 1000.],
                     "model__epsilon": [.01, .1], "model__gamma": ["scale", .01, .1]}),
        "RandomForest": (Pipeline([("model", RandomForestRegressor(
            n_estimators=300, random_state=seed, n_jobs=1))]),
            {"model__max_depth": [None, 12], "model__min_samples_leaf": [1, 5],
             "model__max_features": [1., "sqrt"]}),
        "MLP": (scaled(MLPRegressor(random_state=seed, early_stopping=True,
            validation_fraction=.15, max_iter=1000, n_iter_no_change=30)),
            {"model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
             "model__alpha": [1e-4, 1e-3]}),
    }
    mlp, grid = specs["MLP"]
    specs["MLP-scaled-y-sensitivity"] = (
        TransformedTargetRegressor(regressor=clone(mlp), transformer=StandardScaler()),
        {"regressor__" + key: list(value) for key, value in grid.items()})
    return specs


def folds_for(n, seed):
    import numpy as np
    folds = np.array_split(np.random.default_rng(seed).permutation(n), 3)
    return [(np.concatenate([folds[j] for j in range(3) if j != k]), val)
            for k, val in enumerate(folds)]


def validate_data(data, entry):
    """Validate only observable data/splits, never construct truth features."""
    import numpy as np
    y = np.asarray(data["y"], dtype=float)
    n = len(y)
    p = int(data.get("n_items", data.get("case", {}).get("p", entry.get("p", 0))))
    if y.ndim != 1 or not np.isfinite(y).all() or np.any(y < 0) or p <= 0:
        raise ValueError("Invalid observed response or item dimension")
    if int(data["seed"]) != int(entry["seed"]) or len(data["transactions"]) != n:
        raise ValueError("Dataset identity or observation count mismatch")
    tr, te = (np.asarray(data[key]) for key in ("train_indices", "test_indices"))
    if tr.dtype.kind not in "iu" or te.dtype.kind not in "iu" or min(len(tr), len(te)) < 3:
        raise ValueError("Invalid split index type/size")
    if sorted(tr.tolist() + te.tolist()) != list(range(n)):
        raise ValueError("Splits must partition observations exactly")
    for tx in data["transactions"]:
        if len(tx) != len(set(tx)) or any(type(i) is not int or not 0 <= i < p for i in tx):
            raise ValueError("Invalid transaction item")
    return n, p, tr, te


def validate_folds(folds, n):
    coverage = []
    for tr, val in folds:
        if sorted(list(tr) + list(val)) != list(range(n)):
            raise ValueError("Invalid train-only CV fold")
        coverage.extend(val)
    if len(folds) != 3 or sorted(coverage) != list(range(n)):
        raise ValueError("CV validation folds must partition training observations")


def runtime_environment():
    import numpy, scipy, sklearn
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                host=platform.node(), numpy=numpy.__version__, scipy=scipy.__version__,
                sklearn=sklearn.__version__, threads=dict(THREAD_ENV),
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def prepare(root, expected_datasets=45):
    """Register immutable existing data and this self-contained source before launch."""
    root = Path(root).resolve()
    if (root / "registered.json").exists():
        check_registration(root)
        return load(root / "protocol.json")
    if any((root / name).exists() for name in ("protocol.json", "source", "folds", "jobs")):
        raise ValueError("Prepare requires only input data/manifest; partial registration needs inspection")
    manifest_path = root / "datasets_registered.json"
    manifest = load(manifest_path)
    entries = manifest["entries"]
    if len(entries) != expected_datasets:
        raise ValueError(f"Expected {expected_datasets} datasets, found {len(entries)}")
    ids = [e["dataset_id"] for e in entries]
    if len(ids) != len(set(ids)) or any(not re.fullmatch(r"[A-Za-z0-9_-]+", s) for s in ids):
        raise ValueError("Unsafe or duplicated dataset_id")
    prepared = []
    for e in entries:
        path = relative_path(root, e["data"])
        if sha(path) != e["sha256"]:
            raise ValueError(f"Dataset hash mismatch: {e['dataset_id']}")
        data = load(path)
        n, p, tr, te = validate_data(data, e)
        folds = folds_for(len(tr), e["seed"])
        validate_folds(folds, len(tr))
        prepared.append((e, n, p, len(tr), len(te), jsonable(folds)))
    from sklearn.model_selection import ParameterGrid
    specs = baseline_specs(entries[0]["seed"])
    source_dir = root / "source"
    source_dir.mkdir()
    source_path = source_dir / Path(__file__).name
    shutil.copyfile(Path(__file__).resolve(), source_path)
    datasets = []
    for e, n, p, ntr, nte, folds in prepared:
        foldpath = "folds/" + e["dataset_id"] + ".json"
        exclusive(root / foldpath, dict(dataset_id=e["dataset_id"], seed=e["seed"],
            data_sha256=e["sha256"], index_space="relative to saved train_indices order",
            algorithm="default_rng(seed).permutation(ntrain); array_split(3); concatenate other folds",
            folds=folds))
        datasets.append(dict(e, n=n, p=p, train_size=ntr, test_size=nte,
                             folds=foldpath, folds_sha256=sha(root / foldpath)))
    protocol = dict(version=VERSION, registered_at=now(), methods=METHODS,
        source="source/" + Path(__file__).name, source_sha256=sha(source_path),
        manifest="datasets_registered.json", manifest_sha256=sha(manifest_path),
        datasets=datasets, expected_jobs=len(datasets) * len(METHODS),
        baseline_grids={k: jsonable(v[1]) for k, v in specs.items()},
        baseline_grid_sizes={k: len(ParameterGrid(v[1])) for k, v in specs.items()},
        estimator_reference="new_dgp_benchmark_20260923: expanded baseline_specs; exact factories copied",
        reference_library_versions=dict(numpy="2.2.6", scipy="1.15.3", sklearn="1.7.2"),
        factory_details=dict(RF="300 trees; n_jobs=1; random_state=dataset seed",
            MLP="max_iter=1000; early_stopping=True; validation_fraction=.15; n_iter_no_change=30; random_state=dataset seed",
            Lasso_EN="max_iter=10000", PoissonGLM="max_iter=5000",
            feature_scaling="train-only StandardScaler in every pipeline except RandomForest",
            scaled_MLP="TransformedTargetRegressor(MLP pipeline, StandardScaler target)",
            other_parameters="scikit-learn defaults, execution package versions recorded per job"),
        cv="Saved three folds of outer training only; dataset seed; GridSearchCV negative RMSE, refit=False, error_score=raise, n_jobs=1. First grid candidate breaks exact score ties.",
        final_fit="clone selected estimator; fit all outer training; exactly one untouched test prediction; no truth patterns/weights used in estimator or selection",
        metrics=dict(R2="1-SSE_test/SST_about_test_mean using raw predictions",
            PseudoR2="1-D(y_test,max(pred,1e-6))/max(D(y_test,max(mean(y_train),1e-6)),1e-12)",
            Deviance="2*sum(y*log(y/mu)-y+mu), y=0 log term zero",
            RMSE="sqrt(mean squared raw prediction error)", MAE="mean absolute raw prediction error",
            recovery="precision/recall/F1 are null: these estimators do not output conjunction itemsets"),
        warning_policy="Retain all CV/final warnings; metric_verified does not certify optimizer convergence. Failed jobs are recorded, never automatically retried or replaced.",
        execution=dict(max_workers=16, cpus=list(range(16, 32)), memory_gib_per_worker=8,
            memory_limit="Linux RLIMIT_AS", threads=THREAD_ENV, fit_wall_time_limit=None,
            resume="skip only hash-matched independently metric-verified completion; failures/incomplete previous attempts do not retry"),
        prepare_environment=runtime_environment(), test_only=expected_datasets != 45)
    exclusive(root / "protocol.json", protocol)
    exclusive(root / "registered.json", dict(version=VERSION, at=now(),
        protocol_sha256=sha(root / "protocol.json"), manifest_sha256=sha(manifest_path),
        source_sha256=sha(source_path), jobs=protocol["expected_jobs"]))
    check_registration(root)
    return protocol


def check_registration(root):
    root = Path(root).resolve()
    reg = load(root / "registered.json")
    plan = load(root / "protocol.json")
    if reg["protocol_sha256"] != sha(root / "protocol.json") or plan["version"] != VERSION:
        raise ValueError("Frozen protocol changed")
    if sha(root / plan["source"]) != plan["source_sha256"] or sha(Path(__file__)) != plan["source_sha256"]:
        raise ValueError("Frozen/executing source mismatch")
    if sha(root / plan["manifest"]) != plan["manifest_sha256"]:
        raise ValueError("Dataset manifest changed")
    if reg["manifest_sha256"] != plan["manifest_sha256"] or reg["source_sha256"] != plan["source_sha256"]:
        raise ValueError("Registration mismatch")
    for e in plan["datasets"]:
        if sha(relative_path(root, e["data"])) != e["sha256"]:
            raise ValueError(f"Dataset changed: {e['dataset_id']}")
        if sha(relative_path(root, e["folds"])) != e["folds_sha256"]:
            raise ValueError(f"CV folds changed: {e['dataset_id']}")
    return plan


def job_directory(root, entry, method):
    return Path(root) / "jobs" / entry["dataset_id"] / method


def signature(root, entry, method):
    return digest(dict(protocol_sha256=sha(Path(root) / "protocol.json"), dataset=entry,
                       method=method))


def metrics(y, prediction, train_mean):
    import numpy as np
    from scipy.special import xlogy
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    if y.shape != prediction.shape or y.ndim != 1 or not np.isfinite(prediction).all():
        raise ValueError("Invalid/nonfinite prediction")
    def deviance(mu):
        mu = np.maximum(np.broadcast_to(mu, y.shape), 1e-6)
        # Match archived scoring's protection against negative roundoff terms.
        terms = 2 * (xlogy(y, y) - xlogy(y, mu) - y + mu)
        return float(np.maximum(terms, 0.).sum())
    sse = float(np.sum((y - prediction) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    if sst <= 0:
        raise ValueError("Test response has zero variance")
    dev, null_dev = deviance(prediction), deviance(max(float(train_mean), 1e-6))
    result = dict(R2=1 - sse / sst, PseudoR2=1 - dev / max(null_dev, 1e-12),
                  RMSE=float(np.sqrt(np.mean((y - prediction) ** 2))),
                  MAE=float(np.mean(np.abs(y - prediction))), Deviance=dev,
                  NullDeviance=null_dev, SSE=sse, SST=sst)
    if not all(np.isfinite(v) for v in result.values()):
        raise FloatingPointError("Nonfinite metric")
    return result


def warning_records(caught, stage):
    counts = Counter((w.category.__name__, str(w.message), w.filename, w.lineno) for w in caught)
    return [dict(stage=stage, category=k[0], message=k[1], filename=k[2], lineno=k[3], count=n)
            for k, n in sorted(counts.items())]


def apply_limits(cpu, memory_gib):
    if cpu is not None:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU binding requires Linux")
        os.sched_setaffinity(0, {cpu})
    if sys.platform.startswith("linux"):
        import resource
        limit = int(memory_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    elif cpu is not None:
        raise RuntimeError("Formal resource limit requires Linux")


def commit_result(directory, result):
    save(directory / "result.json", result)
    save(directory / "result_integrity.json", dict(result_sha256=sha(directory / "result.json"),
        signature=result["signature"], status=result["status"]))


def verify_result(root, entry, method, result=None):
    import numpy as np
    directory = job_directory(root, entry, method)
    result = load(directory / "result.json") if result is None else result
    if result["signature"] != signature(root, entry, method):
        raise ValueError("Existing result signature mismatch")
    integrity = load(directory / "result_integrity.json")
    if integrity != dict(result_sha256=sha(directory / "result.json"),
                         signature=result["signature"], status=result["status"]):
        raise ValueError("Existing result integrity mismatch")
    if result["status"] == "failed":
        return result
    if result["status"] != "complete":
        raise ValueError("Unknown result status")
    data = load(relative_path(Path(root), entry["data"]))
    _, _, tr, te = validate_data(data, entry)
    y = np.asarray(data["y"], dtype=float)
    prediction = np.asarray(result["prediction"], dtype=float)
    if result["test_indices"] != te.tolist() or result["y_test"] != y[te].tolist():
        raise ValueError("Saved test targets/indices mismatch")
    expected = metrics(y[te], prediction, float(y[tr].mean()))
    if set(expected) != set(result["metrics"]) or not all(
        np.isclose(v, result["metrics"][k], rtol=1e-11, atol=1e-10) for k, v in expected.items()):
        raise ValueError("Saved metric recomputation mismatch")
    if result["negative_prediction_count"] != int(np.sum(prediction < 0)) or \
       result["deviance_clipped_count"] != int(np.sum(prediction < 1e-6)):
        raise ValueError("Prediction floor counts mismatch")
    if result["choice_sha256"] != sha(directory / "choice_before_test.json"):
        raise ValueError("Saved CV choice changed")
    choice = load(directory / "choice_before_test.json")
    if choice["signature"] != result["signature"] or choice["best_params"] != result["best_params"] \
       or choice["test_used"] or choice["truth_used"]:
        raise ValueError("Invalid CV choice identity")
    if not result["metric_verified"] or any(result[k] is not None for k in ("precision", "recall", "f1")):
        raise ValueError("Invalid metric/recovery status")
    return result


def worker(root, dataset_id, method, cpu=None, memory_gib=8):
    apply_limits(cpu, memory_gib)
    root = Path(root).resolve()
    plan = check_registration(root)
    if method not in plan["methods"]:
        raise ValueError("Unregistered method")
    entry = next(e for e in plan["datasets"] if e["dataset_id"] == dataset_id)
    directory = job_directory(root, entry, method)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "result.json").exists():
            return verify_result(root, entry, method)
        sig = signature(root, entry, method)
        if (directory / "started.json").exists():
            result = dict(status="failed", signature=sig, dataset_id=dataset_id, method=method,
                failed_at=now(), error="Previous attempt incomplete; automatic retry prohibited", metric_verified=False)
            commit_result(directory, result)
            return result
        save(directory / "started.json", dict(at=now(), pid=os.getpid(), signature=sig,
                                               cpu=cpu, memory_gib=memory_gib))
        started = time.perf_counter()
        records = []
        try:
            import numpy as np
            from sklearn.base import clone
            from sklearn.model_selection import GridSearchCV
            environment = runtime_environment()
            if any(environment[k] != v for k, v in plan["reference_library_versions"].items()):
                raise RuntimeError("Execution libraries differ from frozen reference versions")
            data = load(relative_path(root, entry["data"]))
            n, p, tr, te = validate_data(data, entry)
            x = np.zeros((n, p), dtype=float)
            for i, tx in enumerate(data["transactions"]):
                x[i, tx] = 1.
            y = np.asarray(data["y"], dtype=float)
            folds = load(root / entry["folds"])["folds"]
            validate_folds(folds, len(tr))
            folds = [(np.asarray(a), np.asarray(b)) for a, b in folds]
            estimator, grid = baseline_specs(entry["seed"])[method]
            if jsonable(grid) != plan["baseline_grids"][method]:
                raise ValueError("Frozen estimator grid mismatch")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    search = GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error",
                        cv=folds, n_jobs=1, refit=False, error_score="raise")
                    search.fit(x[tr], y[tr])
                finally:
                    records.extend(warning_records(caught, "CV"))
            if not np.isfinite(search.best_score_):
                raise FloatingPointError("Nonfinite selected CV score")
            save(directory / "choice_before_test.json", dict(signature=sig, selected_at=now(),
                best_params=search.best_params_, cv_rmse=-search.best_score_, test_used=False, truth_used=False,
                cv_results={k: v for k, v in search.cv_results_.items()
                    if k in ("params", "mean_test_score", "std_test_score", "rank_test_score")
                    or k.startswith("split")}))
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    selected = clone(estimator).set_params(**search.best_params_).fit(x[tr], y[tr])
                    prediction = selected.predict(x[te])
                finally:
                    records.extend(warning_records(caught, "final_fit_and_prediction"))
            result = dict(status="complete", signature=sig, dataset_id=dataset_id,
                family=entry["family"], level=entry["level"], seed=entry["seed"], method=method,
                data_sha256=entry["sha256"], folds_sha256=entry["folds_sha256"],
                protocol_sha256=sha(root / "protocol.json"), completed_at=now(),
                best_params=search.best_params_, cv_rmse=float(-search.best_score_),
                metrics=metrics(y[te], prediction, float(y[tr].mean())),
                prediction=prediction.tolist(), y_test=y[te].tolist(), test_indices=te.tolist(),
                train_response_mean=float(y[tr].mean()), test_response_mean=float(y[te].mean()),
                negative_prediction_count=int(np.sum(prediction < 0)),
                deviance_clipped_count=int(np.sum(prediction < 1e-6)),
                warnings=records, optimizer_warning=any(w["category"] == "ConvergenceWarning" for w in records),
                metric_verified=True, optimizer_convergence_certified=False,
                precision=None, recall=None, f1=None,
                recovery_status="inapplicable: no conjunction itemset output",
                choice_sha256=sha(directory / "choice_before_test.json"),
                environment=environment, seconds=time.perf_counter() - started)
            commit_result(directory, result)
            verify_result(root, entry, method)
        except BaseException as exc:
            trace = traceback.format_exc()
            print(trace, file=sys.stderr, flush=True)
            result = dict(status="failed", signature=sig, dataset_id=dataset_id,
                family=entry["family"], level=entry["level"], seed=entry["seed"], method=method,
                failed_at=now(), error=f"{type(exc).__name__}: {exc}", traceback=trace,
                warnings=records, metric_verified=False, seconds=time.perf_counter() - started)
            commit_result(directory, result)
        return result


def mark_subprocess_failure(root, entry, method, returncode):
    directory = job_directory(root, entry, method)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "result.json").exists():
            return verify_result(root, entry, method)
        result = dict(status="failed", signature=signature(root, entry, method),
            dataset_id=entry["dataset_id"], method=method, failed_at=now(),
            error=f"Worker exited without complete result (returncode {returncode}); no automatic retry",
            returncode=returncode, metric_verified=False)
        commit_result(directory, result)
        return result


def aggregate(root, require_complete=False):
    root = Path(root).resolve()
    plan = check_registration(root)
    rows, pending, invalid = [], [], []
    for e in plan["datasets"]:
        for method in plan["methods"]:
            directory = job_directory(root, e, method)
            if not (directory / "result.json").exists():
                pending.append(dict(dataset_id=e["dataset_id"], method=method))
                continue
            try:
                rows.append(verify_result(root, e, method))
            except Exception as exc:
                invalid.append(dict(dataset_id=e["dataset_id"], method=method, error=str(exc)))
    completed = [r for r in rows if r["status"] == "complete"]
    failed = [r for r in rows if r["status"] == "failed"]
    status = dict(at=now(), expected=plan["expected_jobs"], completed=len(completed),
        failed=len(failed), pending=len(pending), invalid=len(invalid),
        metric_verified=len(completed), optimizer_warning_jobs=sum(r["optimizer_warning"] for r in completed),
        all_jobs_accounted_for=len(rows) == plan["expected_jobs"] and not invalid,
        all_completed_and_verified=len(completed) == plan["expected_jobs"] and not invalid,
        protocol_sha256=sha(root / "protocol.json"),
        failed_jobs=[dict(dataset_id=r["dataset_id"], method=r["method"], error=r["error"]) for r in failed],
        pending_jobs=pending, invalid_jobs=invalid)
    save(root / "all_results.json", rows)
    save(root / "completion.json", status)
    if require_complete and not status["all_completed_and_verified"]:
        raise ValueError(f"Not all jobs completed and verified: {len(completed)}/{plan['expected_jobs']}")
    return status


def run(root, workers=16, cpus=None, memory_gib=8):
    root = Path(root).resolve()
    cpus = list(range(16, 32)) if cpus is None else cpus
    if not 1 <= workers <= 16 or len(set(cpus)) != len(cpus) or workers > len(cpus):
        raise ValueError("Use 1..16 workers and at least one distinct allowed CPU per worker")
    if not hasattr(os, "sched_getaffinity") or not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Requested CPUs are not available on this Linux host")
    if not 0 < memory_gib <= 8:
        raise ValueError("Worker memory must be at most registered 8 GiB")
    plan = check_registration(root)
    if not set(cpus) <= set(plan["execution"]["cpus"]):
        raise ValueError("CPUs outside registered 16..31 range")
    with (root / ".controller.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another controller already holds this root")
        # Import NumPy only in controller verification; subprocesses start fresh with limits.
        pending = []
        for entry in plan["datasets"]:
            for method in plan["methods"]:
                path = job_directory(root, entry, method) / "result.json"
                if path.exists():
                    verify_result(root, entry, method)  # Includes failures, deliberately never retries.
                else:
                    pending.append((entry, method))
        save(root / "controller_status.json", dict(status="running", at=now(), pid=os.getpid(),
            workers=workers, cpus=cpus[:workers], memory_gib=memory_gib,
            scheduled=len(pending), prior_results=plan["expected_jobs"] - len(pending)))
        # Each lane owns a CPU and takes the next job, so slow MLP fits do not
        # leave fifteen CPUs idle after their short-method lanes finish.
        pending_queue = queue.Queue()
        for job in pending:
            pending_queue.put(job)
        def lane(index):
            while True:
                try:
                    entry, method = pending_queue.get_nowait()
                except queue.Empty:
                    return
                directory = job_directory(root, entry, method)
                directory.mkdir(parents=True, exist_ok=True)
                command = [sys.executable, str(root / plan["source"]), "worker", "--root", str(root),
                    "--dataset-id", entry["dataset_id"], "--method", method,
                    "--cpu", str(cpus[index]), "--memory-gib", str(memory_gib)]
                with (directory / "run.log").open("a", buffering=1) as log:
                    log.write(f"{now()} dispatch {json.dumps(command)}\n")
                    process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=dict(os.environ, **THREAD_ENV), check=False)
                row = mark_subprocess_failure(root, entry, method, process.returncode)
                print(json.dumps(dict(at=now(), dataset_id=entry["dataset_id"], method=method,
                                      status=row["status"])), flush=True)
        errors = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(lane, i) for i in range(workers)]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    traceback.print_exc()
        status = aggregate(root)
        save(root / "controller_status.json", dict(status="finished", at=now(), pid=os.getpid(),
             execution_errors=errors, **{k: v for k, v in status.items() if k != "at"}))
        return status


def parse_cpus(value):
    values = []
    for part in value.split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            values.extend(range(a, b + 1))
        else:
            values.append(int(part))
    return values


def self_test():
    """Small artificial-data implementation checks; no registered study is fitted."""
    import numpy as np
    from sklearn.model_selection import ParameterGrid
    checks = []
    sizes = {k: len(ParameterGrid(v[1])) for k, v in baseline_specs(2609271101).items()}
    assert list(sizes.values()) == [6, 5, 15, 4, 30, 8, 6, 6]
    checks.append("all eight original expanded grid sizes")
    folds = folds_for(35, 2609271101)
    validate_folds(folds, 35)
    assert jsonable(folds) == jsonable(folds_for(35, 2609271101))
    checks.append("deterministic training-only fold partition")
    y = np.asarray([0., 2., 4.]); prediction = np.asarray([-1., 2., 3.])
    m = metrics(y, prediction, 10.)
    assert np.isclose(m["R2"], .75) and np.isclose(m["RMSE"], np.sqrt(2 / 3))
    assert m["PseudoR2"] != metrics(y, prediction, y.mean())["PseudoR2"]
    checks.append("raw prediction R2 and training-mean-null deviance")
    with tempfile.TemporaryDirectory(prefix="extension_baseline_test_") as tmp:
        root = Path(tmp)
        rng = np.random.default_rng(123)
        x = rng.binomial(1, .4, (50, 8))
        data = dict(seed=2609271101, case=dict(p=8), transactions=[np.flatnonzero(a).tolist() for a in x],
            y=rng.poisson(np.exp(.7 + .6 * x[:, 0])).tolist(),
            train_indices=list(range(35)), test_indices=list(range(35, 50)))
        exclusive(root / "data/toy.json", data)
        exclusive(root / "datasets_registered.json", dict(entries=[dict(dataset_id="toy", family="test",
            level=0, seed=data["seed"], data="data/toy.json", sha256=sha(root / "data/toy.json"))]))
        plan = prepare(root, expected_datasets=1)
        check_registration(root)
        checks.append("registration source/data/fold hashes")
        result = worker(root, "toy", "Ridge", memory_gib=8)
        assert result["status"] == "complete", result
        assert worker(root, "toy", "Ridge")["signature"] == result["signature"]
        checks.append("actual Ridge train-CV-test fit and verified resume")
        e = plan["datasets"][0]
        d = job_directory(root, e, "Lasso")
        exclusive(d / "started.json", dict(old_attempt=True))
        failed = worker(root, "toy", "Lasso")
        assert failed["status"] == "failed" and worker(root, "toy", "Lasso") == failed
        checks.append("interrupted/failed attempts are never silently retried")
        path = job_directory(root, e, "Ridge") / "result.json"
        changed = load(path); changed["prediction"][0] += 1; save(path, changed)
        try:
            verify_result(root, e, "Ridge")
        except ValueError:
            checks.append("tampered completion rejected")
        else:
            raise AssertionError("Tampered result accepted")
    return dict(status="passed", checks=checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "run", "worker", "aggregate", "check", "self-test"])
    parser.add_argument("--root", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--cpus", type=parse_cpus, default=list(range(16, 32)))
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--dataset-id")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.command == "self-test":
        answer = self_test()
    else:
        if args.root is None:
            parser.error("--root is required")
        if args.command == "prepare":
            plan = prepare(args.root)
            answer = dict(status="registered", datasets=len(plan["datasets"]), jobs=plan["expected_jobs"],
                          protocol=str(args.root / "protocol.json"))
        elif args.command == "worker":
            if not args.dataset_id or not args.method:
                parser.error("worker needs --dataset-id and --method")
            row = worker(args.root, args.dataset_id, args.method, args.cpu, args.memory_gib)
            answer = {k: row[k] for k in ("status", "dataset_id", "method")}
        elif args.command == "run":
            answer = run(args.root, args.workers, args.cpus, args.memory_gib)
        else:
            answer = aggregate(args.root, require_complete=args.command == "check")
    print(json.dumps(jsonable(answer), indent=2, allow_nan=False), flush=True)
    if args.command in ("run", "check") and not answer.get("all_completed_and_verified", False):
        raise SystemExit(1)
    if args.command == "worker" and answer["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
