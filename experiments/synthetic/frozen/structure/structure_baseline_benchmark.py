"""Matched itemset-recovery baselines: official LCM + Lasso, raw RF paths.

Training never consumes planted itemsets. Recovery is evaluated after the
train-only CV choice and final support have been saved. Existing RF CV evidence
is reused byte-for-byte and the reconstructed forest must reproduce predictions.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import warnings

THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREADS)
VERSION = "matched-structure-baselines-v1"
METHODS = ["LCM-Lasso", "RF-paths"]


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def encode(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as f:
            f.write(encode(value))
        return
    fd, tmp = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(encode(value))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def common(root):
    return module(root / "reference/source/extension_baseline_benchmark.py", "_structure_common")


def check_binary(root, plan, binary):
    receipt = load(root / "runtime/build.json")
    if receipt["archive_sha256"] != plan["files"]["vendor/lcm53.zip"] or receipt["binary_sha256"] != sha(binary):
        raise ValueError("Miner binary does not match registered official-source build")
    if receipt["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Miner build belongs to a different protocol")


def build_miner(root):
    import zipfile
    import stat
    plan = check(root)
    runtime = root / "runtime"
    runtime.mkdir(exist_ok=False)
    source = runtime / "lcm53"
    source.mkdir()
    with zipfile.ZipFile(root / "vendor/lcm53.zip") as archive:
        for info in archive.infolist():
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts or stat.S_ISLNK(info.external_attr >> 16):
                raise ValueError("Unsafe LCM archive entry")
        archive.extractall(source)
    cfiles = list(source.rglob("lcm.c"))
    if len(cfiles) != 1:
        raise ValueError("Ambiguous LCM source layout")
    binary = runtime / "lcm"
    command = ["cc", "-O3", "-o", str(binary), "lcm.c"]
    process = subprocess.run(command, cwd=cfiles[0].parent, capture_output=True, text=True, check=True)
    save(runtime / "build.json", dict(at=now(), command=command,
        compiler=subprocess.check_output(["cc", "--version"], text=True),
        compiler_stderr=process.stderr, archive_sha256=sha(root / "vendor/lcm53.zip"),
        source_hashes={str(p.relative_to(source)):sha(p) for p in source.rglob("*") if p.is_file()},
        binary_sha256=sha(binary), protocol_sha256=sha(root / "protocol.json")), True)
    check_binary(root, plan, binary)
    return dict(binary=str(binary), binary_sha256=sha(binary))


def verify_rf_cv(reference, entry, result):
    import numpy as np
    choice = load(reference / "jobs" / entry["dataset_id"] / "RandomForest/choice_before_test.json")
    cv = choice["cv_results"]
    splits = np.array([cv[f"split{k}_test_score"] for k in range(3)])
    if splits.shape != (3, 8) or not np.isfinite(splits).all():
        raise ValueError("RF CV must contain all eight candidates and three folds")
    mean, std = splits.mean(axis=0), splits.std(axis=0)
    ranks = [1+int(np.sum(mean>value)) for value in mean]
    if not np.allclose(mean, cv["mean_test_score"], rtol=1e-13, atol=1e-12) or not np.allclose(std, cv["std_test_score"], rtol=1e-13, atol=1e-12) or ranks != cv["rank_test_score"]:
        raise ValueError("RF CV mean/std/rank mismatch")
    best = int(np.argmax(mean))
    if cv["params"][best] != result["best_params"] or cv["params"][best] != choice["best_params"] or not np.isclose(-mean[best], result["cv_rmse"], rtol=1e-13, atol=1e-12):
        raise ValueError("RF CV selection mismatch")


def prepare(root, reference, path_reference):
    root, reference, path_reference = map(lambda p: Path(p).resolve(), (root, reference, path_reference))
    if (root / "protocol.json").exists():
        raise ValueError("Registration already exists; never overwrite frozen study")
    c = module(reference / "source/extension_baseline_benchmark.py", "_structure_reference")
    ref = c.check_registration(reference)
    path_plan = load(path_reference / "protocol.json")
    if sha(path_reference / "protocol.json") != load(path_reference / "registered.json")["protocol_sha256"]:
        raise ValueError("PATH reference registration mismatch")
    path_entries = {e["dataset_id"]: e for e in path_plan["datasets"]}
    if len(ref["datasets"]) != 45 or set(path_entries) != {e["dataset_id"] for e in ref["datasets"]}:
        raise ValueError("Must reuse all 45 matched datasets")
    copied = ["protocol.json", "registered.json", ref["manifest"], ref["source"]]
    for e in ref["datasets"]:
        pe = path_entries[e["dataset_id"]]
        for k in ("sha256", "folds_sha256", "maximum_true_length"):
            if pe[k] != e[k]:
                raise ValueError("Different PATH input/config " + e["dataset_id"])
        rf = c.verify_result(reference, e, "RandomForest")
        verify_rf_cv(reference, e, rf)
        if rf["status"] != "complete" or rf["optimizer_warning"]:
            raise ValueError("Cannot reuse incomplete/warning RF " + e["dataset_id"])
        copied.extend([e["data"], e["folds"]])
        copied.extend(f"jobs/{e['dataset_id']}/RandomForest/{name}" for name in
                      ("result.json", "result_integrity.json", "choice_before_test.json"))
    for relative in copied:
        dst = root / "reference" / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            raise ValueError("Existing reference file " + str(dst))
        shutil.copyfile(reference / relative, dst)
    source_names = [Path(__file__).name, "lcm_dictionary.py", "rf_path_support.py"]
    for name in source_names:
        dst = root / "source" / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(__file__).parent / name, dst)
    frozen = {str(p.relative_to(root)): sha(p) for p in (root / "reference").rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    frozen.update({"source/" + name: sha(root / "source" / name) for name in source_names})
    frozen["vendor/lcm53.zip"] = sha(root / "vendor/lcm53.zip")
    plan = dict(version=VERSION, registered_at=now(), datasets=ref["datasets"], methods=METHODS,
        expected_jobs=90, files=frozen, parent_baseline_protocol_sha256=sha(reference / "protocol.json"),
        parent_path_protocol_sha256=sha(path_reference / "protocol.json"),
        dictionary=dict(engine="Uno official LCM 5.3", pattern_space="closed", lower_length=1,
            maximum_length="matched registered dataset maximum_true_length", support_fraction=.015,
            support_minimum=3, scope="Each CV fit subset separately; final dictionary uses all outer train X only",
            scaling="Unscaled binary conjunction columns, matching DIPS original scale",
            truncation="none; resource failures are reported, never silently truncate"),
        lasso=dict(loss="squared error / (2*n) + alpha * L1", fit_intercept=True, positive=False,
            l2=0, tau=[10 ** (-2 * k / 100) for k in range(101)],
            alpha_max="max(abs(X.T @ (y-mean(y)))) / n independently on each fit subset",
            cv="Mean of the same three validation RMSEs; exact ties select earliest/largest tau",
            tol=1e-8, max_iter=20000, coefficient_threshold=1e-8,
            convergence="No ConvergenceWarning; dual_gap <= 1.01*tol*var(y); independent normalized KKT <= 1e-6",
            final="Warm start from tau=1 down to CV-selected tau, never select on truth or test"),
        rf=dict(training="Raw 64 binary item columns; reconstruct existing 300-tree RF with saved seed and CV parameters",
            reuse="Original 3-fold RMSE selection is frozen; require same sklearn/numpy/scipy and matching saved predictions",
            primary="RF positive-path: all non-root pure positive root-to-node prefixes; unique itemsets; fit support and length filters",
            supplementary="RF positive projection: drop negative literals of every prefix; NOT a logically equivalent rule; reported separately without choosing the better definition",
            selection="No planted-count cutoff, no truth-based threshold, no new response-dependent rule filtering"),
        recovery="Exact sorted itemset equality to generating log-mean support; TP,FP,FN,NZ,precision,recall,F1; empty precision/F1=0",
        prediction="R2 on raw test predictions; Poisson PR2 uses max(pred,1e-6) and outer-training-mean null",
        aggregation="All five seeds per condition, mean and sample SD; failed/incomplete jobs disclosed separately",
        versions=ref["reference_library_versions"],
        execution=dict(threads=THREADS, workers="Assigned only idle Jupiter CPUs, explicit launch receipt",
            memory_gib_per_worker=16, resume="Verify completed jobs; never automatically retry failures/incomplete attempts"),
        interpretation="Classical Lasso has identity link whereas planted support defines log Poisson mean; RF positive-rule family differs from closed itemsets. This is a method comparison, not a pruning-only ablation.")
    save(root / "protocol.json", plan, True)
    save(root / "registered.json", dict(version=VERSION, protocol_sha256=sha(root / "protocol.json")), True)
    return check(root)


def check(root):
    root = Path(root).resolve()
    plan = load(root / "protocol.json")
    if plan["version"] != VERSION or load(root / "registered.json")["protocol_sha256"] != sha(root / "protocol.json"):
        raise ValueError("Protocol mismatch")
    for name, value in plan["files"].items():
        if sha(root / name) != value:
            raise ValueError("Changed registered file " + name)
    if sha(Path(__file__)) != plan["files"]["source/" + Path(__file__).name]:
        raise ValueError("Executing source differs from frozen runner")
    return plan


def arrays(root, entry, split):
    import numpy as np
    data = load(root / "reference" / entry["data"])
    c = common(root)
    n, p, tr, te = c.validate_data(data, entry)
    folds = load(root / "reference" / entry["folds"])["folds"]
    c.validate_folds(folds, len(tr))
    if split != "train":
        a, b = folds[int(split[-1])]
        tr, te = tr[a], tr[b]
    tx, xt = [[data["transactions"][i] for i in ids] for ids in (tr, te)]
    yy = np.asarray(data["y"], float)
    # No truth/true_mean/true_weights escapes into any fitting routine.
    return tx, yy[tr], xt, yy[te], tr.tolist(), te.tolist()


def recovery(active, truth):
    selected = {tuple(sorted(p)) for p in active}
    actual = {tuple(sorted(p)) for p in truth}
    tp = len(selected & actual)
    return dict(TP=tp, FP=len(selected)-tp, FN=len(actual)-tp, NZ=len(selected),
                precision=tp/max(1, len(selected)), recall=tp/max(1, len(actual)),
                f1=2*tp/max(1, len(selected)+len(actual)))


def mine(root, plan, entry, split, binary, tx, tr):
    """Persist sparse dictionaries bound to exact fit rows and miner binary."""
    from lcm_dictionary import mine_closed, load_dictionary
    check_binary(root, plan, binary)
    cache = root / "dictionaries" / entry["dataset_id"] / split
    signature = dict(data_sha256=entry["sha256"], folds_sha256=entry["folds_sha256"],
        training_indices=tr, min_support=max(3, math.ceil(.015*len(tr))),
        max_len=entry["maximum_true_length"], binary_sha256=sha(binary),
        module_sha256=plan["files"]["source/lcm_dictionary.py"])
    seal = cache / "cache_seal.json"
    if seal.exists():
        evidence = load(seal)
        if evidence["signature"] != signature:
            raise ValueError("Dictionary identity mismatch")
        if sha(cache / "miner/receipt.json") != evidence["miner_receipt_sha256"]:
            raise ValueError("Changed dictionary receipt")
        got = load_dictionary(cache / "miner", mmap_mode="c")
        return got["patterns"], got["matrix"], evidence["receipt"]
    if cache.exists():
        raise ValueError("Incomplete dictionary cache; no automatic retry: " + str(cache))
    cache.mkdir(parents=True)
    got = mine_closed(tx, signature["min_support"], signature["max_len"], binary, cache / "miner")
    patterns, X = got["patterns"], got["matrix"]
    receipt = dict(got["receipt"], M=int(X.shape[1]), nnz=int(X.nnz),
                   sparse_bytes=int(X.data.nbytes + X.indices.nbytes + X.indptr.nbytes))
    save(seal, dict(signature=signature, receipt=receipt,
        miner_receipt_sha256=sha(cache / "miner/receipt.json")), True)
    return patterns, X, receipt


def predict_patterns(tx, patterns, coefficients, intercept):
    """Evaluate only nonzero terms, without mapping a huge test dictionary."""
    import numpy as np
    masks = [sum(1 << int(i) for i in t) for t in tx]
    pred = np.full(len(tx), float(intercept))
    for pat, value in zip(patterns, coefficients):
        bits = sum(1 << int(i) for i in pat)
        pred += float(value) * np.fromiter(((row & bits) == bits for row in masks), dtype=float, count=len(tx))
    return pred


def lasso_path(root, plan, entry, split, binary, directory, last=100):
    import numpy as np
    from sklearn.linear_model import Lasso
    tx, y, xt, yt, tr, te = arrays(root, entry, split)
    patterns, X, receipt = mine(root, plan, entry, split, binary, tx, tr)
    settings = plan["lasso"]
    amax = float(np.max(np.abs(np.asarray(X.T @ (y-y.mean())).ravel())) / len(y))
    if amax <= 0 or not math.isfinite(amax):
        raise ValueError("Degenerate Lasso alpha_max")
    estimator = Lasso(fit_intercept=True, positive=False, precompute=False,
        warm_start=True, copy_X=False, tol=settings["tol"], max_iter=settings["max_iter"], selection="cyclic")
    rows, threshold = [], settings["coefficient_threshold"]
    directory.mkdir(parents=True, exist_ok=True)
    for k, tau in enumerate(settings["tau"][:last+1]):
        t0 = time.perf_counter()
        alpha = amax * tau
        estimator.set_params(alpha=alpha)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            estimator.fit(X, y)
        beta = estimator.coef_
        residual = y - estimator.predict(X)
        gradient = np.asarray(X.T @ residual).ravel() / len(y)
        nz = np.flatnonzero(np.abs(beta) > threshold)
        used = [patterns[j] for j in nz]
        violation = np.maximum(np.abs(gradient)-alpha, 0)
        nonzero = beta != 0
        violation[nonzero] = np.abs(gradient[nonzero]-alpha*np.sign(beta[nonzero]))
        kkt = max(float(violation.max()), abs(float(residual.mean()))) / max(1., amax)
        gap = float(estimator.dual_gap_)
        gap_bound = 1.01 * settings["tol"] * float(np.var(y))
        warn = [dict(category=w.category.__name__, message=str(w.message)) for w in caught]
        converged = not any(w["category"] == "ConvergenceWarning" for w in warn) and gap <= gap_bound and kkt <= 1e-6
        row = dict(k=k, tau=tau, alpha=alpha, alpha_max=amax, M=len(patterns),
            NZ=len(nz), dual_gap=gap, dual_gap_bound=gap_bound, normalized_kkt=kkt,
            n_iter=int(estimator.n_iter_), warnings=warn, converged=converged,
            seconds=time.perf_counter()-t0)
        # CV is evaluated without accessing held-out outer-test observations.
        if split != "train":
            # Include every fitted coefficient in predictions; support threshold is reporting only.
            pred_indices = np.flatnonzero(beta != 0)
            pred = predict_patterns(xt, [patterns[j] for j in pred_indices], beta[pred_indices], estimator.intercept_)
            row["RMSE"] = float(np.sqrt(np.mean((yt-pred)**2)))
        save(directory / f"point_{k:03d}.json", row, True)
        rows.append(row)
        save(directory / "progress.json", dict(at=now(), completed_points=len(rows), last=row))
        if not converged:
            raise RuntimeError(f"Unconverged Lasso at {split} tau={tau}: gap={gap}, kkt={kkt}")
    pred_indices = np.flatnonzero(beta != 0)
    final_model = dict(patterns=[patterns[j] for j in pred_indices], coefficients=beta[pred_indices].tolist(),
        intercept=float(estimator.intercept_), selected_patterns=used, tau=settings["tau"][last],
        alpha=float(estimator.alpha), alpha_max=amax, selected_point=last, dictionary=receipt,
        training_indices=tr, evaluation_indices=te,
        point_sha256={f"point_{k:03d}.json":sha(directory/f"point_{k:03d}.json") for k in range(last+1)})
    save(directory / "fit.json", final_model, True)
    return rows, final_model


def run_lasso(root, plan, entry, binary, directory):
    import numpy as np
    paths = []
    for fold in range(3):
        rows, _ = lasso_path(root, plan, entry, f"fold{fold}", binary, directory / f"fold{fold}")
        paths.append(rows)
        gc.collect()
    scores = np.mean([[row["RMSE"] for row in rows] for rows in paths], axis=0)
    chosen = int(np.argmin(scores))
    save(directory / "choice_before_test.json", dict(selected_point=chosen,
        tau=plan["lasso"]["tau"][chosen], mean_cv_rmse=scores.tolist(),
        cv_sources={f"fold{fold}/point_{k:03d}.json": sha(directory/f"fold{fold}/point_{k:03d}.json")
                    for fold in range(3) for k in range(101)}, test_used=False, truth_used=False), True)
    _, fitted = lasso_path(root, plan, entry, "train", binary, directory / "train", chosen)
    tx, y, xt, yt, tr, te = arrays(root, entry, "train")
    pred = predict_patterns(xt, fitted["patterns"], fitted["coefficients"], fitted["intercept"])
    # The complete fitted support is saved above before reading evaluation truth.
    data = load(root / "reference" / entry["data"])
    truth = data["true_patterns"]
    patfile = root / "dictionaries" / entry["dataset_id"] / "train/miner/patterns.txt"
    all_patterns = {tuple(map(int, line.split())) for line in patfile.read_text().splitlines()}
    return dict(metrics=common(root).metrics(yt, pred, float(y.mean())), prediction=pred.tolist(),
        recovery=recovery(fitted["selected_patterns"], truth),
        selected_patterns=fitted["selected_patterns"], tau=fitted["tau"], alpha=fitted["alpha"],
        cv_rmse=float(scores[chosen]), chosen_on_boundary=chosen in (0, 100),
        candidate_truth_count=sum(tuple(sorted(p)) in all_patterns for p in truth),
        candidate_count=len(all_patterns), final_fit_sha256=sha(directory / "train/fit.json"),
        choice_sha256=sha(directory / "choice_before_test.json"))


def run_rf(root, plan, entry, directory):
    import numpy as np
    from sklearn.base import clone
    from rf_path_support import extract_rf_path_support
    c = common(root)
    old = c.verify_result(root / "reference", entry, "RandomForest")
    verify_rf_cv(root / "reference", entry, old)
    tx, y, xt, yt, tr, te = arrays(root, entry, "train")
    def binary(items):
        X = np.zeros((len(items), entry["p"]), float)
        for i, t in enumerate(items):
            X[i, t] = 1
        return X
    X, Xt = binary(tx), binary(xt)
    model = clone(c.baseline_specs(entry["seed"])["RandomForest"][0]).set_params(**old["best_params"])
    model.fit(X, y)
    pred = model.predict(Xt)
    if not np.allclose(pred, old["prediction"], rtol=1e-12, atol=1e-10):
        raise ValueError("Reconstructed RF differs from original predictions; refuse reuse")
    support = extract_rf_path_support(model.named_steps["model"], X,
        min_support=max(3, math.ceil(.015*len(X))), max_len=entry["maximum_true_length"])
    save(directory / "extracted_before_truth.json", support, True)
    data = load(root / "reference" / entry["data"])
    truth = data["true_patterns"]
    return dict(metrics=c.metrics(yt, pred, float(y.mean())), prediction=pred.tolist(),
        recovery=recovery(support["positive_rules"], truth), selected_patterns=support["positive_rules"],
        projection_recovery=recovery(support["positive_projection"], truth),
        projection_patterns=support["positive_projection"],
        best_params=old["best_params"], cv_rmse=old["cv_rmse"],
        reconstructed_prediction_max_abs_error=float(np.max(np.abs(pred-np.asarray(old["prediction"])))),
        parent_result_sha256=sha(root/"reference/jobs"/entry["dataset_id"]/"RandomForest/result.json"),
        extraction_sha256=sha(directory / "extracted_before_truth.json"))


def verify(root, plan, entry, method):
    import numpy as np
    directory = root / "jobs" / entry["dataset_id"] / method
    result = load(directory / "result.json")
    if load(directory / "integrity.json")["sha256"] != sha(directory / "result.json"):
        raise ValueError("Result checksum mismatch")
    if result["protocol_sha256"] != sha(root / "protocol.json") or result["dataset_id"] != entry["dataset_id"] or result["method"] != method:
        raise ValueError("Result identity mismatch")
    if result["status"] != "complete":
        return result
    data = load(root / "reference" / entry["data"])
    y = np.asarray(data["y"])
    expected = common(root).metrics(y[data["test_indices"]], np.asarray(result["prediction"]), float(y[data["train_indices"]].mean()))
    if any(not np.isclose(v, result["metrics"][k], atol=1e-10, rtol=1e-11) for k,v in expected.items()):
        raise ValueError("Prediction metric mismatch")
    if recovery(result["selected_patterns"], data["true_patterns"]) != result["recovery"]:
        raise ValueError("Recovery mismatch")
    if method == "RF-paths":
        if recovery(result["projection_patterns"], data["true_patterns"]) != result["projection_recovery"]:
            raise ValueError("Projection recovery mismatch")
        if sha(directory / "extracted_before_truth.json") != result["extraction_sha256"]:
            raise ValueError("Extraction changed")
        extracted = load(directory / "extracted_before_truth.json")
        if extracted["positive_rules"] != result["selected_patterns"] or extracted["positive_projection"] != result["projection_patterns"]:
            raise ValueError("Result differs from original extracted support")
        parent = root/"reference/jobs"/entry["dataset_id"]/"RandomForest/result.json"
        if sha(parent) != result["parent_result_sha256"] or not np.allclose(result["prediction"], load(parent)["prediction"], rtol=1e-12, atol=1e-10):
            raise ValueError("RF result no longer matches parent evidence")
    else:
        if sha(directory / "train/fit.json") != result["final_fit_sha256"] or sha(directory / "choice_before_test.json") != result["choice_sha256"]:
            raise ValueError("Lasso fit/choice changed")
        choice = load(directory / "choice_before_test.json")
        if choice["test_used"] or choice["truth_used"]:
            raise ValueError("Invalid CV choice")
        all_rmse = [[], [], []]
        for rel, value in choice["cv_sources"].items():
            if sha(directory/rel) != value:
                raise ValueError("CV evidence changed")
            row = load(directory/rel)
            if not row["converged"]:
                raise ValueError("CV includes unconverged fit")
            all_rmse[int(rel[4])].append((row["k"], row["RMSE"]))
        if any(len(rows) != 101 or [k for k,_ in sorted(rows)] != list(range(101)) for rows in all_rmse):
            raise ValueError("Incomplete CV path")
        scores = np.mean([[v for _,v in sorted(rows)] for rows in all_rmse], axis=0)
        if choice["selected_point"] != int(np.argmin(scores)) or not np.allclose(scores, choice["mean_cv_rmse"]):
            raise ValueError("CV selection mismatch")
        fitted = load(directory / "train/fit.json")
        if fitted["selected_patterns"] != result["selected_patterns"] or fitted["selected_point"] != choice["selected_point"]:
            raise ValueError("Final fit support/selection mismatch")
        inferred = [p for p,w in zip(fitted["patterns"],fitted["coefficients"]) if abs(w)>plan["lasso"]["coefficient_threshold"]]
        if inferred != result["selected_patterns"]:
            raise ValueError("Reported support differs from fitted coefficients")
        last = choice["selected_point"]
        if set(fitted["point_sha256"]) != {f"point_{k:03d}.json" for k in range(last+1)}:
            raise ValueError("Incomplete final training path evidence")
        for rel, value in fitted["point_sha256"].items():
            if sha(directory/"train"/rel) != value:
                raise ValueError("Final path evidence changed")
            row = load(directory/"train"/rel)
            if (not row["converged"] or not all(math.isfinite(row[k]) for k in ("dual_gap","normalized_kkt","alpha","tau"))
                or row["dual_gap"] > row["dual_gap_bound"] or row["normalized_kkt"] > 1e-6
                or row["tau"] != plan["lasso"]["tau"][row["k"]]
                or not np.isclose(row["alpha"], fitted["alpha_max"]*row["tau"], rtol=1e-13,atol=1e-13)):
                raise ValueError("Invalid final fit convergence evidence")
        xt = [data["transactions"][i] for i in data["test_indices"]]
        predicted = predict_patterns(xt, fitted["patterns"], fitted["coefficients"], fitted["intercept"])
        if not np.allclose(predicted, result["prediction"], rtol=1e-12, atol=1e-10):
            raise ValueError("Final model prediction mismatch")
    return result


def limits(cpu, memory_gib):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    if sys.platform == "linux":
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (int(memory_gib*1024**3),)*2)


def worker(root, dataset_id, method, binary, cpu, memory_gib):
    limits(cpu, memory_gib)
    root = Path(root).resolve()
    plan = check(root)
    entry = next(e for e in plan["datasets"] if e["dataset_id"] == dataset_id)
    directory = root / "jobs" / dataset_id / method
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (directory / "result.json").exists():
            return verify(root, plan, entry, method)
        if (directory / "started.json").exists():
            raise ValueError("Incomplete previous attempt; no automatic retry")
        save(directory / "started.json", dict(at=now(), pid=os.getpid(), cpu=cpu, memory_gib=memory_gib), True)
        started = time.perf_counter()
        result = dict(dataset_id=dataset_id, method=method, family=entry["family"], level=entry["level"],
            seed=entry["seed"], protocol_sha256=sha(root / "protocol.json"))
        try:
            env = common(root).runtime_environment()
            if any(env[k] != v for k,v in plan["versions"].items()):
                raise ValueError("Reference library versions differ")
            result.update(run_lasso(root, plan, entry, binary, directory) if method == "LCM-Lasso"
                          else run_rf(root, plan, entry, directory))
            result["negative_prediction_count"] = sum(p < 0 for p in result["prediction"])
            result["deviance_clipped_count"] = sum(p < 1e-6 for p in result["prediction"])
            result.update(status="complete", environment=env)
        except Exception as exc:
            result.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
            print(result["traceback"], file=sys.stderr, flush=True)
        result.update(completed_at=now(), seconds=time.perf_counter()-started)
        save(directory / "result.json", result, True)
        save(directory / "integrity.json", dict(sha256=sha(directory / "result.json")), True)
        return verify(root, plan, entry, method)


def aggregate(root):
    import numpy as np
    root = Path(root).resolve()
    plan = check(root)
    complete, failed, pending = [], [], []
    for entry in plan["datasets"]:
        for method in METHODS:
            if not (root / "jobs" / entry["dataset_id"] / method / "result.json").exists():
                pending.append(dict(dataset_id=entry["dataset_id"], method=method))
                continue
            row = verify(root, plan, entry, method)
            (complete if row["status"] == "complete" else failed).append(row)
    groups = []
    for family, level, method in sorted({(r["family"],r["level"],r["method"]) for r in complete}):
        rows = [r for r in complete if (r["family"],r["level"],r["method"]) == (family,level,method)]
        variants = [(method, "recovery")]
        if method == "RF-paths":
            variants.append(("RF-paths-positive-projection-supplement", "projection_recovery"))
        for name, key in variants:
            stats = {}
            for metric in ("R2", "PseudoR2", "precision", "recall", "f1", "NZ", "TP", "FP", "FN"):
                vals = [r["metrics"][metric] if metric in ("R2", "PseudoR2") else r[key][metric] for r in rows]
                stats[metric] = dict(mean=float(np.mean(vals)), sample_sd=float(np.std(vals, ddof=1)) if len(vals)>1 else None)
            groups.append(dict(family=family, level=level, method=name, completed_seeds=len(rows), expected_seeds=5,
                               complete_condition=len(rows)==5, statistics=stats))
    summary = dict(at=now(), complete_jobs=len(complete), expected_jobs=90, groups=groups,
                   failed=failed, pending=pending, protocol_sha256=sha(root / "protocol.json"))
    save(root / "summary.json", summary)
    return summary


def run(root, binary, cpus, methods, dataset_ids):
    root = Path(root).resolve()
    plan = check(root)
    if len(cpus) != len(set(cpus)) or not cpus or not set(methods) <= set(METHODS):
        raise ValueError("Invalid CPU/method assignment")
    jobs = [(e["dataset_id"], m) for e in plan["datasets"] if not dataset_ids or e["dataset_id"] in dataset_ids for m in methods]
    pool = queue.Queue()
    for cpu in cpus:
        pool.put(cpu)
    def dispatch(job):
        cpu = pool.get()
        try:
            did, method = job
            directory = root/"jobs"/did/method
            directory.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(root/"source"/Path(__file__).name), "worker", "--root", str(root),
                       "--binary", str(binary), "--dataset", did, "--method", method, "--cpu", str(cpu)]
            with (directory/"process.log").open("ab") as log:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ, **THREADS))
                save(directory/"process_started.json", dict(at=now(), pid=process.pid, cpu=cpu, command=command))
                rc = process.wait()
                save(directory/"process.json", dict(at=now(), returncode=rc))
            return dict(dataset_id=did, method=method, returncode=rc)
        finally:
            pool.put(cpu)
    save(root/"launches"/(datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+f"_{os.getpid()}.json"),
         dict(at=now(), pid=os.getpid(), cpus=cpus, methods=methods, jobs=jobs), True)
    with ThreadPoolExecutor(max_workers=len(cpus)) as executor:
        for row in executor.map(dispatch, jobs):
            print(json.dumps(row), flush=True)
    return aggregate(root)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "check", "build-lcm", "worker", "run", "aggregate", "preflight"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--reference")
    parser.add_argument("--path-reference")
    parser.add_argument("--binary")
    parser.add_argument("--dataset")
    parser.add_argument("--split", default="train", choices=["train", "fold0", "fold1", "fold2"])
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--datasets", default="")
    parser.add_argument("--cpu", type=int)
    parser.add_argument("--cpus", default="19,20,21,23,24,25,26,27")
    parser.add_argument("--memory-gib", type=int, default=16)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.command == "prepare":
        result = prepare(root, args.reference, args.path_reference)
        print(json.dumps(dict(registered=True, datasets=len(result["datasets"]), protocol_sha256=sha(root/"protocol.json"))))
    elif args.command == "check":
        print(json.dumps(dict(valid=True, datasets=len(check(root)["datasets"]))))
    elif args.command == "build-lcm":
        print(json.dumps(build_miner(root)))
    elif args.command == "worker":
        result = worker(root, args.dataset, args.method, args.binary, args.cpu, args.memory_gib)
        print(json.dumps({k:result[k] for k in ("dataset_id", "method", "status", "seconds")}))
        if result["status"] != "complete":
            sys.exit(1)
    elif args.command == "preflight":
        limits(args.cpu, args.memory_gib)
        plan = check(root)
        entry = next(e for e in plan["datasets"] if e["dataset_id"] == args.dataset)
        tx, _, _, _, tr, _ = arrays(root, entry, args.split)
        pats, X, receipt = mine(root, plan, entry, args.split, args.binary, tx, tr)
        print(json.dumps(dict(dataset_id=args.dataset, split=args.split, receipt=receipt)))
    elif args.command == "aggregate":
        result = aggregate(root)
        print(json.dumps(dict(complete=result["complete_jobs"], expected=90, failed=len(result["failed"]))))
    else:
        result = run(root, args.binary, [int(i) for i in args.cpus.split(",")], args.methods.split(","), set(filter(None,args.datasets.split(","))))
        print(json.dumps(dict(complete=result["complete_jobs"], expected=90, failed=len(result["failed"]))))


if __name__ == "__main__":
    main()
