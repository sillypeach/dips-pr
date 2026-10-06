"""Shared integrity, SeqID grouping and metrics for registered HIV endpoints."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import sys

VERSION = "hiv-grouped-endpoint-workers-v1"
THREAD_ENV = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def registered(root, executing_path, required):
    root = Path(root).resolve()
    protocol = json.loads((root/"protocol.json").read_text())
    if sha(root/"protocol.json") != json.loads((root/"registered.json").read_text())["protocol_sha256"]:
        raise ValueError("Changed registered protocol")
    own = "source/" + Path(executing_path).name
    required = set(required) | {own, "source/hiv_shared.py", "data.json", "folds.json", "metadata.json"}
    files = protocol.get("files")
    if not isinstance(files, dict) or not required <= set(files):
        raise ValueError("Missing required frozen input/source hashes")
    for relative, expected in files.items():
        file = (root/relative).resolve()
        if Path(relative).is_absolute() or not file.is_relative_to(root) or sha(file) != expected:
            raise ValueError("Changed or invalid registered file: " + relative)
    if sha(executing_path) != files[own] or sha(__file__) != files["source/hiv_shared.py"]:
        raise ValueError("Executing worker/shared helper differs from frozen source")
    if protocol.get("seed") != 42:
        raise ValueError("Expected the registered HIV seed 42")
    if protocol.get("model_seed") != 42 or protocol.get("model_seed") != protocol["seed"]:
        raise ValueError("Legacy seed must equal the frozen model seed42")
    if protocol.get("outer_split_seed") not in (42,43,44,45,46) or protocol.get("cv_seed") != protocol["outer_split_seed"]:
        raise ValueError("Unregistered outer/CV split seed")
    if json.loads((root/"folds.json").read_text()).get("seed") != protocol["cv_seed"]:
        raise ValueError("CV seed metadata differs from protocol")
    metadata=json.loads((root/"metadata.json").read_text())
    if metadata.get("outer_split_seed") != protocol["outer_split_seed"] or metadata.get("model_seed") != protocol["model_seed"]:
        raise ValueError("Split/model seed metadata differs from protocol")
    if not protocol.get("endpoint"):
        raise ValueError("Endpoint identity is required")
    return protocol


def validate_data(data, folds, plan=None):
    """Validate saved observations and SeqID group splits, without resplitting.

    Final fit may equal the entire CV training data. An optional independent
    validation tail is supported for compatibility, but is never constructed.
    Group identity is SeqID, and is not asserted to identify unique patients.
    """
    import numpy as np
    for suffix in ("train", "final", "test"):
        tx, y = data["tx_"+suffix], np.asarray(data["y_"+suffix],float)
        groups = data["groups_"+suffix]
        if (y.ndim != 1 or not len(y) or len(tx) != len(y) or len(groups) != len(y)
                or not np.isfinite(y).all() or np.any(y < 0)):
            raise ValueError("Invalid observed response/transaction/group dimensions: " + suffix)
        if suffix != "test" and float(y.sum()) <= 0:
            raise ValueError("Training responses must have a positive sum")
        if any(not isinstance(g,str) or not g for g in groups):
            raise ValueError("SeqID group identities must be nonempty strings")
        for row in tx:
            if not isinstance(row,list) or any(type(i) is not int or i < 0 for i in row) or len(set(row)) != len(row):
                raise ValueError("Positions must be distinct nonnegative integers")
        if plan and "n_"+suffix in plan and len(y) != plan["n_"+suffix]:
            raise ValueError("Registered sample dimension mismatch: " + suffix)
    n = len(data["y_train"])
    for prefix in ("tx_", "y_", "groups_"):
        if data[prefix+"final"][:n] != data[prefix+"train"]:
            raise ValueError("CV rows/groups must be preserved as the prefix of final training")
    train, validation, test = [data[key] for key in ("train_indices", "validation_indices", "test_indices")]
    ids = train+validation+test
    if (any(type(i) is not int or i < 0 for i in ids) or len(train) != n
            or len(train+validation) != len(data["y_final"]) or len(test) != len(data["y_test"])
            or len(set(ids)) != len(ids)):
        raise ValueError("Original row identities overlap or do not align")
    groups_train = set(data["groups_train"])
    groups_validation = set(data["groups_final"][n:])
    groups_test = set(data["groups_test"])
    if (groups_train & groups_validation) or (groups_train & groups_test) or (groups_validation & groups_test):
        raise ValueError("SeqID groups overlap across outer train/validation/test")
    if len(folds) != 3:
        raise ValueError("Exactly three saved grouped CV folds are required")
    coverage = []
    for pair in folds:
        if len(pair) != 2:
            raise ValueError("Expected fit/validation index pair")
        fit, valid = pair
        if (not fit or not valid or any(type(i) is not int for i in fit+valid)
                or sorted(fit+valid) != list(range(n))):
            raise ValueError("CV fit/validation must partition the tuning rows")
        fitgroups = {data["groups_train"][i] for i in fit}
        validgroups = {data["groups_train"][i] for i in valid}
        if fitgroups & validgroups:
            raise ValueError("SeqID group crosses CV fit/validation boundary")
        if sum(data["y_train"][i] for i in fit) <= 0:
            raise ValueError("Each fitting fold needs positive response sum")
        coverage.extend(valid)
    if sorted(coverage) != list(range(n)):
        raise ValueError("CV validations must cover every tuning row exactly once")
    maximum = max((max(tx,default=-1) for key in ("tx_train","tx_final","tx_test") for tx in data[key]),default=-1)
    if plan and "n_items" in plan:
        if type(plan["n_items"]) is not int or plan["n_items"] <= maximum or plan["n_items"] <= 0:
            raise ValueError("Registered ambient position dimension excludes observed items")
        return plan["n_items"]
    if maximum < 0:
        raise ValueError("No position dimension available")
    return maximum+1


def prediction_metrics(observed, prediction, y_fit):
    """Raw regression metrics, test-mean PR² and fitting-mean PR².

    Only deviance predictions/null means have a numerical floor of 1e-6.
    Undefined R²/PR² (zero denominator) is represented as JSON null.
    """
    import numpy as np
    from scipy.special import xlogy
    observed, prediction, y_fit = [np.asarray(v,float) for v in (observed,prediction,y_fit)]
    if (observed.ndim != 1 or prediction.shape != observed.shape or not len(observed)
            or not np.isfinite(observed).all() or not np.isfinite(prediction).all()
            or np.any(observed < 0) or y_fit.ndim != 1 or not len(y_fit)
            or not np.isfinite(y_fit).all() or np.any(y_fit < 0)):
        raise ValueError("Invalid scoring response/prediction vector")
    error = observed-prediction
    sse = float(error@error)
    sst = float(np.sum((observed-observed.mean())**2))
    mean_eval, mean_train = float(observed.mean()), float(y_fit.mean())
    def deviance(mu):
        mu = np.maximum(mu,1e-6)
        return max(0.,float(2*np.sum(xlogy(observed,observed/mu)-observed+mu)))
    dev = deviance(prediction)
    null_eval, null_train = deviance(mean_eval), deviance(mean_train)
    result = dict(R2=1-sse/sst if sst else None, RMSE=math.sqrt(sse/len(observed)),
        MAE=float(np.mean(np.abs(error))), Deviance=dev,
        PseudoR2=1-dev/null_eval if null_eval else None,
        PseudoR2_train_null=1-dev/null_train if null_train else None,
        NullDeviance=null_eval, NullDeviance_train=null_train,
        NullMean_test=mean_eval, NullMean_train=mean_train)
    if any(value is not None and not math.isfinite(value) for value in result.values()):
        raise ValueError("Nonfinite regression/deviance metric")
    return result


def warning_rows(captured):
    return [dict(category=w.category.__name__, message=str(w.message), filename=w.filename, lineno=w.lineno) for w in captured]


def configure_resources(cpu=None, memory_gib=None):
    if cpu is not None:
        if type(cpu) is not int or cpu < 0:
            raise ValueError("CPU must be a nonnegative integer")
        if hasattr(os,"sched_setaffinity"):
            os.sched_setaffinity(0,{cpu})
    if memory_gib is not None:
        if not math.isfinite(memory_gib) or memory_gib <= 0:
            raise ValueError("Memory limit must be positive")
        if sys.platform == "linux":
            import resource
            resource.setrlimit(resource.RLIMIT_AS,(int(memory_gib*1024**3),)*2)
