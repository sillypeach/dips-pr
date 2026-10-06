"""Fresh HIV baseline CV/refit jobs using the registered, shared folds.

Run: python source/hiv_baseline.py --root EXPERIMENT --method RandomForest
No previous model, prediction or CV result is reused. Invalid CV candidates are
recorded and excluded; finite completed fits remain eligible despite warnings.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
import time
import traceback
import warnings

from hiv_shared import (THREAD_ENV, registered, validate_data, prediction_metrics, configure_resources)
os.environ.update(THREAD_ENV)

import numpy as np
import sklearn
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import r2_score
from sklearn.model_selection import ParameterGrid
from sklearn.tree import DecisionTreeRegressor

METHODS = ("Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR", "RandomForest", "MLP", "DecisionTree")
COUNTS = dict(zip(METHODS, (4, 3, 9, 4, 12, 18, 9, 12)))
VERSION = "hiv-grouped-common-fold-baselines-v1"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonable(value):
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, BaseEstimator):
        return {"class": type(value).__module__ + "." + type(value).__name__,
                "params": jsonable(value.get_params(deep=False))}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(jsonable(payload), stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_registered(root):
    protocol = registered(root, __file__, {"source/hiv_nonlinear_baselines.py",
        "source/common.py", "source/model.py", "source/utils.py"})
    return protocol, sha256(Path(root)/"protocol.json")

def load_frozen_specs(root, seed):
    source = (Path(root) / "source").resolve()
    # A fresh CLI process is expected. Never silently import project-head deps.
    for name in ("common", "model", "utils"):
        module = sys.modules.get(name)
        if module is not None and Path(module.__file__).resolve().parent != source:
            raise ValueError("Already-imported non-frozen dependency: " + name)
    sys.path.insert(0, str(source))
    try:
        spec = importlib.util.spec_from_file_location("hiv_frozen_hiv_factories", source / "hiv_nonlinear_baselines.py")
        module = importlib.util.module_from_spec(spec)
        # The historic file sets a warning filter. Do not let it leak into fits.
        with warnings.catch_warnings():
            spec.loader.exec_module(module)
        for name in ("common", "model", "utils"):
            loaded = sys.modules.get(name)
            if loaded is None or Path(loaded.__file__).resolve().parent != source:
                raise ValueError("Factory imported a non-frozen dependency: " + name)
        specs = module.model_specs(seed)
        if set(specs) != set(METHODS)-{"DecisionTree"}:
            raise ValueError("Frozen factory does not contain exactly the original seven models")
        for name, (_, grid) in specs.items():
            if len(ParameterGrid(grid)) != COUNTS[name]:
                raise ValueError("Unexpected original grid size: " + name)
        specs["DecisionTree"] = (DecisionTreeRegressor(random_state=seed),
            {"max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 5, 10]})
        if len(ParameterGrid(specs["DecisionTree"][1])) != 12:
            raise ValueError("DecisionTree grid changed")
        return specs
    finally:
        sys.path.pop(0)


def validate_data_folds(data, fold_payload, plan=None):
    folds = fold_payload["folds"]
    nfeatures = validate_data(data, folds, plan)
    arrays = {}
    for split in ("train", "final", "test"):
        arrays["y_"+split] = np.asarray(data["y_"+split],float)
        x = np.zeros((len(data["tx_"+split]), nfeatures),dtype=float)
        for i, row in enumerate(data["tx_"+split]):
            x[i,row] = 1.
        arrays["X_"+split] = x
    return arrays, [(np.asarray(tr,dtype=int),np.asarray(va,dtype=int)) for tr,va in folds]

def warning_rows(captured):
    return [{"category": w.category.__name__, "message": str(w.message), "filename": w.filename, "lineno": w.lineno} for w in captured]


def fitted_diagnostics(estimator):
    steps = getattr(estimator, "named_steps", {"estimator": estimator})
    result = {}
    for name, step in steps.items():
        result[name] = {key: jsonable(getattr(step, key)) for key in ("n_iter_", "dual_gap_", "loss_", "best_validation_score_", "loss_curve_", "validation_scores_", "n_features_in_") if hasattr(step, key)}
    return result


def fit_predict_record(estimator, params, xfit, yfit, xeval):
    started = time.perf_counter()
    record = dict(status="failed", valid=False, params=params, n_fit=len(yfit), n_predict=len(xeval), warnings=[])
    captured = []
    fitted = None
    try:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            fitted = clone(estimator).set_params(**params)
            record["fit_params"] = jsonable(fitted.get_params(deep=True))
            fitted.fit(xfit, yfit)
            prediction = np.asarray(fitted.predict(xeval), dtype=float)
            if prediction.ndim != 1 or prediction.shape != (len(xeval),) or not np.all(np.isfinite(prediction)):
                raise ValueError("Prediction must be a finite one-dimensional vector of the expected length")
            record.update(status="complete", valid=True, prediction=prediction.tolist(), fitted_diagnostics=fitted_diagnostics(fitted))
    except Exception as error:
        record["error"] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    finally:
        record["warnings"] = warning_rows(captured)
        record["convergence_warning_count"] = sum(w["category"] == "ConvergenceWarning" for w in record["warnings"])
        record["elapsed_seconds"] = time.perf_counter() - started
    return record


def score_prediction(y, prediction, final_train_y):
    metrics = prediction_metrics(y,prediction,final_train_y)
    prediction = np.asarray(prediction,float)
    details = dict(deviance_prediction_floor=1e-6, undefined_zero_denominator_metrics=None,
        test_mean=metrics["NullMean_test"], final_train_mean=metrics["NullMean_train"],
        test_null_deviance=metrics["NullDeviance"], final_train_null_deviance=metrics["NullDeviance_train"],
        train_null_scope="y_final (final fitting observations)",
        negative_raw_predictions=int(np.count_nonzero(prediction<0)),
        raw_predictions_below_deviance_floor=int(np.count_nonzero(prediction<1e-6)))
    return metrics,details


@contextmanager
def job_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def run(root, method, cpu=None, memory_gib=None):
    configure_resources(cpu,memory_gib)
    root = Path(root).resolve()
    if method not in METHODS:
        raise ValueError("Unknown original baseline: " + method)
    destination = root / "baselines" / method
    with job_lock(destination / ".lock"):
        if (destination / "request.json").exists() or (destination / "result.json").exists():
            raise FileExistsError("Job already has an attempt; refusing overwrite or prediction reuse: " + str(destination))
        protocol, protocol_hash = verify_registered(root)
        specs = load_frozen_specs(root, int(protocol["model_seed"]))
        data = read_json(root / "data.json")
        arrays, folds = validate_data_folds(data, read_json(root / "folds.json"), protocol)
        for split in ("train", "final", "test"):
            if "n_" + split in protocol and len(arrays["y_" + split]) != protocol["n_" + split]:
                raise ValueError("Registered sample count mismatch: " + split)
        if "n_items" in protocol and arrays["X_train"].shape[1] != protocol["n_items"]:
            raise ValueError("Registered feature dimension mismatch")
        estimator, grid = specs[method]
        candidates = list(ParameterGrid(grid))
        metadata = dict(version=VERSION, endpoint=protocol["endpoint"], method=method, protocol_sha256=protocol_hash,
                        data_sha256=sha256(root / "data.json"), folds_sha256=sha256(root / "folds.json"),
                        source_sha256=sha256(root / "source/hiv_baseline.py"), seed=int(protocol["model_seed"]),
                        packages=dict(python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__),
                        n_candidates=len(candidates), n_folds=3, candidate_order="sklearn.model_selection.ParameterGrid",
                        warning_policy="Finite completed fits remain eligible; warnings including ConvergenceWarning retained",
                        invalid_candidate_policy="Exclude candidate if any fold fails or has nonfinite predictions/scores",
                        selector="Minimum arithmetic mean of three validation RMSEs; exact ties choose first ParameterGrid index",
                        final_fit_scope="tx_final/y_final; test used only after CV selection", prediction_policy="raw, unclipped")
        write_json(destination / "request.json", metadata)
        started = time.perf_counter()
        candidate_rows = []
        try:
            for index, params in enumerate(candidates):
                fold_rows = []
                for k, (tr, va) in enumerate(folds):
                    record = fit_predict_record(estimator, params, arrays["X_train"][tr], arrays["y_train"][tr], arrays["X_train"][va])
                    record.update(candidate_index=index, fold_index=k, train_indices=tr.tolist(), validation_indices=va.tolist())
                    if record["valid"]:
                        with np.errstate(over="ignore", invalid="ignore"):
                            rmse = float(np.sqrt(np.mean((np.asarray(record["prediction"])-arrays["y_train"][va])**2)))
                        if np.isfinite(rmse):
                            record["rmse"] = rmse
                        else:
                            record.update(status="failed", valid=False, error=dict(type="ValueError", message="Nonfinite validation RMSE"))
                    write_json(destination / f"candidate_{index:03d}" / f"fold_{k}.json", record)
                    fold_rows.append(record)
                    write_json(destination / "progress.json", dict(status="running", stage="cv", candidate_index=index, fold_index=k,
                               completed_cv_fits=3*index+k+1, planned_cv_fits=3*len(candidates), protocol_sha256=protocol_hash))
                candidate = dict(candidate_index=index, params=params, valid=all(r["valid"] for r in fold_rows),
                                 fold_files=[f"candidate_{index:03d}/fold_{k}.json" for k in range(3)],
                                 folds=[{key: r[key] for key in ("fold_index", "valid", "status", "rmse", "warnings", "convergence_warning_count", "elapsed_seconds", "error") if key in r} for r in fold_rows])
                if candidate["valid"]:
                    losses = np.array([r["rmse"] for r in fold_rows])
                    candidate.update(mean_rmse=float(losses.mean()), std_rmse=float(losses.std(ddof=0)), se_rmse=float(losses.std(ddof=1)/np.sqrt(3)))
                candidate_rows.append(candidate)
                write_json(destination / "cv_results.json", dict(metadata=metadata, candidates=candidate_rows))
            valid = [row for row in candidate_rows if row["valid"]]
            for row in valid:
                row["rank_test_rmse"] = 1 + sum(other["mean_rmse"] < row["mean_rmse"] for other in valid)
            write_json(destination / "cv_results.json", dict(metadata=metadata, candidates=candidate_rows,
                       valid_candidate_count=len(valid), invalid_candidate_count=len(candidates)-len(valid)))
            if not valid:
                raise RuntimeError("No valid candidate completed all three folds")
            best = min(valid, key=lambda row: (row["mean_rmse"], row["candidate_index"]))
            write_json(destination / "selection.json", dict(metadata=metadata, best_candidate_index=best["candidate_index"],
                       best_params=best["params"], cv_rmse=best["mean_rmse"], valid_candidate_count=len(valid)))
            write_json(destination / "progress.json", dict(status="running", stage="final_fit", protocol_sha256=protocol_hash))
            final = fit_predict_record(estimator, best["params"], arrays["X_final"], arrays["y_final"], arrays["X_test"])
            write_json(destination / "final_fit.json", final)
            if not final["valid"]:
                raise RuntimeError("Selected candidate failed final fit/prediction; see final_fit.json")
            metrics, metric_details = score_prediction(arrays["y_test"], final["prediction"], arrays["y_final"])
            _, end_hash = verify_registered(root)
            if end_hash != protocol_hash:
                raise ValueError("Protocol changed during execution")
            result = dict(metadata, status="complete", model=method, best_candidate_index=best["candidate_index"],
                          best_params=best["params"], cv_rmse=best["mean_rmse"], valid_candidate_count=len(valid),
                          invalid_candidate_count=len(candidates)-len(valid), cv_results=candidate_rows,
                          prediction=final["prediction"], metrics=metrics, metric_definitions=metric_details,
                          warnings=final["warnings"], final_fit_file="final_fit.json", elapsed_seconds=time.perf_counter()-started)
            write_json(destination / "result.json", result)
            write_json(destination / "progress.json", dict(status="complete", stage="complete", protocol_sha256=protocol_hash))
            return result
        except BaseException as error:
            write_json(destination / "failure.json", dict(metadata, status="failed", error=dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()),
                       completed_candidate_count=len(candidate_rows), elapsed_seconds=time.perf_counter()-started))
            write_json(destination / "progress.json", dict(status="failed", protocol_sha256=protocol_hash))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--memory-gib", type=float)
    args = parser.parse_args()
    result = run(args.root, args.method, args.cpu, args.memory_gib)
    print(json.dumps({key: result[key] for key in ("status", "model", "best_params", "cv_rmse", "metrics", "valid_candidate_count")}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
