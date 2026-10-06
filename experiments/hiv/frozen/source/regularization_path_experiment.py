"""Registered regularization-path experiment on 45 existing synthetic datasets.

Only new output directories are prepared. Fits, CV choices, independent audits,
lambda-max construction and process failures are separate durable records.
No truth information enters fitting or CV selection. Only PATH is executed.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from collections import deque
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

THREAD_ENV = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")}
os.environ.update(THREAD_ENV)
VERSION = "regularization-path-frozen-data-v3"
ARMS = ("PATH",)
SOURCE_NAMES = ("regularization_path_experiment.py", "path_staged_solver.py",
                "uv_only_solver.py")


def now():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def encode(value):
    def default(x):
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "item"):
            return x.item()
        raise TypeError(type(x).__name__)
    return (json.dumps(value, sort_keys=True, indent=2, default=default,
                       allow_nan=False) + "\n").encode()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value, exclusive=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive:
        with path.open("xb") as stream:
            stream.write(encode(value))
        return
    fd, temporary = tempfile.mkstemp(prefix="." + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encode(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def safe_path(root, relative):
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Path escapes registered root")
    return result


def prepare(args):
    root, parent = Path(args.root).resolve(), Path(args.data_root).resolve()
    if root.exists():
        raise ValueError("Prepare requires a fresh output directory; no overwrites")
    old = load(parent / "protocol.json")
    entries = old["datasets"]
    if len(entries) != 45:
        raise ValueError("Expected all 45 frozen datasets")
    files = {}
    for entry in entries:
        for key, digest_key in (("data", "sha256"), ("folds", "folds_sha256")):
            origin = safe_path(parent, entry[key])
            if sha(origin) != entry[digest_key]:
                raise ValueError("Changed frozen input: " + str(origin))
            files[entry[key]] = origin
    source = Path(__file__).resolve().parent
    for name in SOURCE_NAMES:
        files["source/" + name] = source / name
    files["source/model.py"] = Path(args.model).resolve()
    files["audit/model.py"] = Path(args.audit_model).resolve()
    files["input_datasets_protocol.json"] = parent / "protocol.json"
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    pilot = [next(e["dataset_id"] for e in entries if e["family"] == family
                  and e["level"] == level) for family, level in
             (("count", 4), ("length", 8), ("shared", 30))]
    root.mkdir()
    for relative, origin in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, target)
    plan = dict(version=VERSION, registered_at=now(), datasets=entries,
        arms=list(ARMS), taus=[10 ** (-2 * i / 100) for i in range(101)],
        pilot_taus=[10 ** (-.02*i) for i in range(6)], pilot_datasets=pilot,
        path_density_basis="Training-only runtime preflight on the existing first L8 seed: finer 0.02-log10 steps completed the prefix to tau=10^-0.1, whereas the prior coarse step timed out. No test, CV, or truth recovery score selects density; this is a computational pilot, not fresh predictive confirmation.",
        lambda_max="Exact maximum absolute closed-pattern correlation with y-mean(y), on each training subset; branch-and-bound mass upper bound; summed loss, no division by n",
        config=dict(kappa=.05, support_fraction=.015, support_minimum=3,
            top_k_add=200, max_rounds=100, solver_max_iter=10000,
            solver_tol=1e-7, gap_tol=1e-7, radius_mode="certified",
            pattern_space="closed", propagate_support=True, dual_reference="ray"),
        support_rule="max(3,ceil(.015*n_subset)); recomputed on every training subset",
        max_len_rule="Frozen experimental design metadata maximum_true_length; never inspect truth itemsets for fitting",
        staged_options=dict(initial_tol=1e-3, tolerance_factor=.1, ready_fraction=.8),
        cv="Three saved training-only folds; PATH only; unweighted mean of three validation RMSE values; require all folds at every path point complete and independently verified; ties prefer larger tau; no truth/test selection",
        final="After this dataset's three CV paths finish and its choice is saved, immediately prioritize its PATH outer-training path on the next free CPU. Outer-training lambda_max and path state are independent of the fold fits. Report the preselected tau; no BASE, LOCAL, or baseline job is permitted.",
        scheduling="Compute training-only lambda_max for all requested subsets first; then a shared CPU pool runs fold paths. Each dataset's outer path is queued with priority as soon as its three folds are complete and independently verified. Do not wait for unrelated datasets' folds.",
        initial_reference="Exact lambda-max calibration supplies alpha0=y-mean(y). Without a preceding-lambda dual reference, scale alpha0 by min(1,lambda/(lambda_max+roundoff_pad)) so all dictionary correlations are at most lambda and the EN conjugate sum is zero. PATH subsequently uses its recertified preceding-lambda reference, with radius from the current target primal objective.",
        null_point_accounting="At tau=1 the exact lambda-max certificate gives the analytic intercept-only solution. This point has zero u/v training traversals and is not evidence of screening. Common lambda-max construction and independent auditing remain separately recorded.",
        initial_traversal_record="Each point preserves path_transfer.initial_reference_source/global_bound/scale and first_traversal mode, nodes, u/v counts, radius and center norm. At the analytic null solution first_traversal is null and analytic_zero_solution_is_uv_pruning is false. A valid initial reference permits screening but does not promise positive pruning counts.",
        modes=dict(PATH="Own primal warm start; calibrated cold reference at the first nontrivial target without prior state, otherwise recertified preceding-lambda reference; within-point u/v screening"),
        independent_audit="Post-fit full-dictionary gap/KKT; no repair or candidate feedback; separate seconds and visits; failed audit stops that path and records remaining points as not run",
        timeout="Per-process budget includes fit and post-fit audit; saved phase distinguishes where timeout occurs. Pilot default 600 s/path; lambda-max preparation has its own default 600 s budget. Full run defaults to no wall cap. All failures retained; no automatic retries.",
        resources=dict(memory_gib_per_worker=8, threads=THREAD_ENV,
            recommended_cpus=list(range(30)), reserved_system_cpus=[30, 31],
            prior_study_status="Prior workers suspended by user authorization before v2; no prior process is managed by this runner"),
        expected=dict(datasets=45, pilot_paths=3, pilot_fits=18,
            cv_paths=135, cv_fits=13635, outer_paths=45, outer_fits=4545,
            total_formal_paths=180, total_formal_points=18180),
        controls="PATH only; no comparator fitting or baseline analysis in this run",
        input_origin=str(parent), hashes={rel: sha(root / rel) for rel in files},
        origin_hashes={str(origin): sha(origin) for origin in files.values()})
    save(root / "protocol.json", plan, exclusive=True)
    save(root / "registered.json", dict(at=now(), version=VERSION,
        protocol_sha256=sha(root / "protocol.json")), exclusive=True)
    print(json.dumps(dict(prepared=str(root), pilot_datasets=pilot,
                         registered_files=len(files)), indent=2))
    return plan


def check(root, all_files=False):
    plan = load(root / "protocol.json")
    if sha(root / "protocol.json") != load(root / "registered.json")["protocol_sha256"]:
        raise ValueError("Changed protocol")
    for relative, expected in plan["hashes"].items():
        if all_files or relative.startswith(("source/", "audit/")):
            if sha(safe_path(root, relative)) != expected:
                raise ValueError("Changed registered source/input: " + relative)
    return plan


def limits(cpu, memory_gib):
    if cpu is not None and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {cpu})
    if sys.platform == "linux":
        import resource
        limit = int(memory_gib * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))


def environment():
    import numpy, scipy, sklearn
    return dict(at=now(), executable=sys.executable, python=sys.version,
        numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__,
        host=platform.node(), platform=platform.platform(), threads=THREAD_ENV,
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None)


def peak_rss_mib():
    import resource
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == "darwin" else 1024)


def dataset(root, plan, dataset_id, split):
    import numpy as np
    entry = next(e for e in plan["datasets"] if e["dataset_id"] == dataset_id)
    for key, digest in (("data", "sha256"), ("folds", "folds_sha256")):
        if sha(root / entry[key]) != entry[digest]:
            raise ValueError("Changed dataset/folds")
    data = load(root / entry["data"])
    tr, te = data["train_indices"], data["test_indices"]
    if sorted(tr + te) != list(range(len(data["y"]))):
        raise ValueError("Outer indices must partition all observations")
    if split != "train":
        fold = int(split.removeprefix("fold"))
        inner, valid = load(root / entry["folds"])["folds"][fold]
        if sorted(inner + valid) != list(range(len(tr))):
            raise ValueError("Invalid saved training fold")
        tr, te = [tr[i] for i in inner], [tr[i] for i in valid]
    yy = np.asarray(data["y"], float)
    tx, xt = [[data["transactions"][i] for i in ids] for ids in (tr, te)]
    cfg = dict(plan["config"])
    fraction = cfg.pop("support_fraction")
    minimum = cfg.pop("support_minimum")
    cfg.update(min_support=max(minimum, math.ceil(fraction * len(tr))),
               max_len=int(entry["maximum_true_length"]))
    return entry, data, tx, yy[tr], xt, yy[te], tr, te, cfg


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def audit_fit(audit, tx, y, row):
    import numpy as np
    started = time.perf_counter()
    cfg = row["config"]
    tids, items = audit.PoissonEN_LCM_SPP._transactions_to_item_tidsets(tx)
    patterns = [audit.Pattern(tuple(p), ()) for p in row["patterns"]]
    X = audit.build_X_from_patterns_on_transactions(tx, patterns)
    w, alpha = np.asarray(row["w"]), np.asarray(row["alpha"])
    eta = row["b"] + X @ w
    calc = audit.PoissonENGap(cfg["lam"], cfg["kappa"])
    e1 = audit.LCMEnumerator(tids, items, len(y), cfg["min_support"], cfg["max_len"], cfg["pattern_space"])
    scan = e1.dual_scan(alpha, cfg["lam"], cfg["kappa"], top_k=0)
    primal, _, gap = calc.gap(y, eta, w, alpha, full_penalty=scan["penalty"])
    e2 = audit.LCMEnumerator(tids, items, len(y), cfg["min_support"], cfg["max_len"], cfg["pattern_space"])
    residual = calc.build_dual_feasible(y, eta)
    scan2 = e2.dual_scan(residual, cfg["lam"], cfg["kappa"], already_have=map(tuple, row["patterns"]), top_k=0)
    kkt = max(scan2["outside_violation"] / len(y), audit.kkt_residual(
        X, y, audit.ENParams(row["b"], w), cfg["lam"], cfg["kappa"]))
    consistent = bool(np.isclose(primal, row["objective"], rtol=1e-12, atol=1e-7)
                      and np.isclose(gap, row["gap"], rtol=1e-4, atol=1e-7))
    passed = consistent and gap / len(y) <= 1.01 * cfg["gap_tol"] and kkt <= 1.01 * cfg["solver_tol"]
    return dict(status="passed" if passed else "failed", gap_per_sample=gap/len(y),
        kkt=kkt, objective_gap_consistent=consistent, seconds=time.perf_counter()-started,
        scans=[asdict(e1.stats), asdict(e2.stats)],
        nodes_visited=e1.stats.nodes_visited + e2.stats.nodes_visited,
        role="post-fit verification only; no repair or training feedback")


def serialize(fitted, config, xt, yt, y, data, split):
    import numpy as np
    from scipy.special import xlogy
    import model as m
    fit = fitted["fit"]
    prediction = np.exp(fit.params.b + m.build_X_from_patterns_on_transactions(xt, fit.patterns) @ fit.params.w)
    if not np.isfinite(prediction).all():
        raise FloatingPointError("Nonfinite predictions")
    active = {p.itemset for p, w in zip(fit.patterns, fit.params.w) if abs(w) > 1e-8}
    sse = float(np.sum((yt-prediction)**2))
    sst = float(np.sum((yt-yt.mean())**2))
    deviance = float(2*np.sum(xlogy(yt, yt/prediction)-yt+prediction))
    null = float(2*np.sum(xlogy(yt, yt/y.mean())-yt+y.mean()))
    summary = m.summarize_pruning(fit)
    summary["full_kkt"] = None  # Full-dictionary KKT is measured only by the independent audit.
    row = dict(status="complete" if fit.converged else "nonconverged",
        termination_reason=fit.termination_reason, config=config,
        objective=m.poisson_objective(y, fit.eta, fit.params.w, config["lam"], config["kappa"]),
        gap=float(fit.gap), patterns=[list(p.itemset) for p in fit.patterns],
        w=fit.params.w.tolist(), b=float(fit.params.b), alpha=fit.alpha.tolist(),
        prediction=prediction.tolist(), y_evaluation=yt.tolist(),
        metrics=dict(RMSE=math.sqrt(sse/len(yt)), R2=1-sse/sst if sst else None,
            PseudoR2=1-deviance/null if null else None, Deviance=deviance,
            NullDeviance=null, null_mean=float(y.mean())),
        evaluation_scope="held-out test; never selection" if split == "train" else "training-only CV validation",
        NZ=len(active), summary=summary,
        rounds=fit.round_times, search=[asdict(s) for s in fit.enum_stats_per_round],
        path_transfer=fitted["path_transfer"], transfer_certificate=fitted["transfer_certificate"],
        implementation_version=fit.implementation_version)
    if split == "train":
        truth = {tuple(p) for p in data["true_patterns"]}
        tp = len(active & truth)
        row.update(TP=tp, FP=len(active)-tp, precision=tp/max(1,len(active)),
            recall=tp/len(truth), f1=2*tp/max(1,len(active)+len(truth)))
    return row


def jobdir(root, request):
    if request["kind"] == "lambda":
        return root / "lambda_max" / request["dataset_id"] / request["split"]
    return root / "jobs" / request["stage"] / request["dataset_id"] / request["split"] / request["arm"]


def worker(args):
    limits(args.cpu, args.memory_gib)
    root = Path(args.root).resolve()
    plan, request = check(root), load(args.request)
    if plan.get("version") != VERSION or plan.get("arms") != ["PATH"]:
        raise ValueError("This v3 worker accepts only the registered PATH-only plan")
    if request["kind"] == "path" and request.get("arm") != "PATH":
        raise ValueError("BASE/LOCAL/control jobs are prohibited in v3")
    directory = jobdir(root, request)
    if sha(root / "protocol.json") != request["protocol_sha256"]:
        raise ValueError("Request protocol mismatch")
    save(directory / "started.json", dict(at=now(), pid=os.getpid(), cpu=args.cpu,
        memory_gib=args.memory_gib, environment=environment()), exclusive=True)
    from path_staged_solver import fit_path_point, lambda_max_exact
    entry, data, tx, y, xt, yt, tr, te, cfg = dataset(root, plan, request["dataset_id"], request["split"])
    phase = dict(at=now(), stage="loading", dataset_id=entry["dataset_id"], split=request["split"])
    def progress(stage, **extra):
        phase.update(at=now(), stage=stage, **extra)
        save(directory / "progress.json", phase)
    if request["kind"] == "lambda":
        progress("lambda_max_exact")
        result = lambda_max_exact(tx, y, cfg["min_support"], cfg["max_len"], cfg["pattern_space"])
        if not result["exact"] or not math.isfinite(result["lambda_max"]) or result["lambda_max"] <= 0:
            raise ValueError("No positive exact lambda_max; retain as failed degenerate design")
        result.update(status="complete", dataset_id=entry["dataset_id"], split=request["split"],
            config=cfg, training_indices=tr, data_sha256=entry["sha256"],
            folds_sha256=entry["folds_sha256"], completed_at=now())
        save(directory / "result.json", result, exclusive=True)
        progress("complete")
        return
    ldir = root / "lambda_max" / entry["dataset_id"] / request["split"]
    lm = load(ldir / "result.json")
    if lm["status"] != "complete" or not lm["exact"]:
        raise ValueError("Missing exact lambda_max")
    audit = module(root / "audit/model.py", "_path_independent_audit")
    if request["stage"] == "full" and request["split"] == "train":
        choice = load(root/"selection"/(entry["dataset_id"]+".json"))
        if choice["dataset_id"] != entry["dataset_id"] or choice.get("test_used") is not False:
            raise ValueError("Outer fit requires this dataset's saved training-only CV choice")
        for relative, digest in choice["source_hashes"].items():
            if sha(root/relative) != digest:
                raise ValueError("Changed CV evidence before outer fit")
    previous, points, stopped = None, [], False
    for index, tau in enumerate(request["taus"]):
        pointdir = directory / f"point_{index:02d}"
        pointdir.mkdir()
        config = dict(cfg, lam=tau*lm["lambda_max"])
        identity = dict(point=index, tau=tau, arm=request["arm"], stage=request["stage"],
            dataset_id=entry["dataset_id"], split=request["split"], config=config,
            lambda_max=lm["lambda_max"], data_sha256=entry["sha256"],
            folds_sha256=entry["folds_sha256"], training_indices=tr, evaluation_indices=te)
        if stopped:
            row = dict(identity, status="not_run", reason="Earlier path point did not converge or pass independent audit")
            save(pointdir / "result.json", row, exclusive=True)
            points.append(row)
            save(directory / "rows.json", points)
            continue
        progress("fit", point=index, points_total=len(request["taus"]),
                 verified_points=sum(bool(r.get("verified")) for r in points),
                 tau=tau, lambda_value=config["lam"])
        print(now(), "POINT_START", entry["dataset_id"], request["split"],
              f"{index+1}/{len(request['taus'])}", "tau", tau, flush=True)
        try:
            with (pointdir / "fit.log").open("x", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
                tick = time.perf_counter()
                fitted = fit_path_point(tx, y, config, previous=previous,
                    mode=request["arm"], verbose=True, calibration=lm, **plan["staged_options"])
                seconds = time.perf_counter()-tick
            row = dict(identity, **serialize(fitted, config, xt, yt, y, data, request["split"]))
            row.update(fit_seconds=seconds, completed_fit_at=now(),
                process_peak_rss_mib_before_current_audit=peak_rss_mib(),
                memory_scope="Process high-water mark, includes previous path points and audits; not isolated fit peak")
            # Durable fit before independent auditing; never modify this fit record.
            save(pointdir / "fit.json", row, exclusive=True)
            save(pointdir / "state.json", fitted["state"], exclusive=True)
            progress("independent_audit", point=index, tau=tau)
            row["independent"] = audit_fit(audit, tx, y, row)
            row["process_peak_rss_mib_after_current_audit"] = peak_rss_mib()
            row["verified"] = row["status"] == "complete" and row["independent"]["status"] == "passed"
            # Audit only validates: never changes parameters, candidates or certificate.
            previous = fitted["state"]
            stopped = not row["verified"]
        except Exception:
            row = dict(identity, status="failed", error=traceback.format_exc(), failed_at=now(), phase=dict(phase))
            stopped = True
        save(pointdir / "result.json", row, exclusive=True)
        points.append(row)
        save(directory / "rows.json", points)
        print(now(), "POINT_END", entry["dataset_id"], request["split"],
              f"{index+1}/{len(request['taus'])}", row["status"],
              "verified", row.get("verified",False), flush=True)
    save(directory / "completion.json", dict(at=now(), completed=len(points),
        expected=len(request["taus"]), all_verified=all(r.get("verified",False) for r in points),
        fit_seconds=sum(r.get("fit_seconds",0) for r in points),
        audit_seconds=sum(r.get("independent",{}).get("seconds",0) for r in points)), exclusive=True)
    progress("complete" if not stopped else "failed", points_total=len(request["taus"]),
             verified_points=sum(bool(r.get("verified")) for r in points))


def dispatch(root, plan, spec, cpu, memory_gib, timeout):
    if spec["kind"] == "path" and spec.get("arm") != "PATH":
        raise ValueError("v3 never dispatches BASE/LOCAL/control jobs")
    request = dict(spec, protocol_sha256=sha(root / "protocol.json"))
    directory = jobdir(root, request)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (directory / "request.json").exists():
            if load(directory / "request.json") != request:
                raise ValueError("Changed resumed job definition")
        else:
            save(directory / "request.json", request, exclusive=True)
        if (directory / "process.json").exists():
            receipt = load(directory / "process.json")
            for rel, digest in receipt.get("output_hashes", {}).items():
                if sha(directory / rel) != digest:
                    raise ValueError("Changed completed output")
            return receipt
        if (directory / "process_started.json").exists() or (directory / "started.json").exists():
            return dict(status="incomplete_previous_attempt", job=str(directory),
                        reason="Preserved; automatic retry prohibited")
        command = [sys.executable, str(root / "source/regularization_path_experiment.py"),
            "worker", "--root", str(root), "--request", str(directory / "request.json"),
            "--cpu", str(cpu), "--memory-gib", str(memory_gib)]
        tick = time.perf_counter()
        with (directory / "worker.log").open("x", buffering=1) as log:
            proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, **THREAD_ENV), start_new_session=True)
            save(directory / "process_started.json", dict(at=now(), pid=proc.pid,
                command=command, cpu=cpu, memory_gib=memory_gib,
                timeout_seconds=timeout or None), exclusive=True)
            status = "exited"
            try:
                code = proc.wait(timeout=timeout or None)
                if code != 0:
                    status = "worker_failed"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                code, status = proc.returncode, "timeout"
        outcome = dict(at=now(), status=status, returncode=code, job=str(directory.relative_to(root)),
            seconds=time.perf_counter()-tick, timeout_seconds=timeout or None,
            last_progress=load(directory / "progress.json") if (directory / "progress.json").exists() else None,
            output_hashes={str(p.relative_to(directory)):sha(p) for p in directory.rglob("*.json")
                           if p.name != "process.json"})
        save(directory / "process.json", outcome, exclusive=True)
        return outcome


def jobs_run(root, plan, specs, args, timeout):
    cpus = args.cpus[:args.workers]
    available = queue.Queue()
    for cpu in cpus:
        available.put(cpu)
    def one(spec):
        cpu = available.get()
        try:
            result = dispatch(root, plan, spec, cpu, args.memory_gib, timeout)
            print(now(), spec["kind"], spec["dataset_id"], spec["split"], spec.get("arm"), result["status"], flush=True)
            return result
        finally:
            available.put(cpu)
    outcomes = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for future in as_completed([pool.submit(one, spec) for spec in specs]):
            outcomes.append(future.result())
    return outcomes


def cv_choice(root, plan, entry):
    import numpy as np
    did = entry["dataset_id"]
    destination = root / "selection" / (did + ".json")
    values, inputs = [], {}
    for fold in range(3):
        directory = root / "jobs/full" / did / f"fold{fold}" / "PATH"
        rows_path = directory / "rows.json"
        if not rows_path.exists():
            return None
        rows = load(rows_path)
        if len(rows) != len(plan["taus"]) or not all(r.get("verified",False) for r in rows):
            return None
        inputs[str(rows_path.relative_to(root))] = sha(rows_path)
        values.append([r["metrics"]["RMSE"] for r in rows])
    mean = np.mean(values, axis=0)
    index = min(range(len(mean)), key=lambda i: (float(mean[i]), -plan["taus"][i]))
    choice = dict(dataset_id=did, selected_point=index, selected_tau=plan["taus"][index],
        mean_rmse=mean.tolist(), fold_rmse=values, rule="min mean validation RMSE, exact ties prefer larger tau",
        source_hashes=inputs, test_used=False, truth_used=False)
    if destination.exists():
        if load(destination) != choice:
            raise ValueError("Changed existing CV choice")
    else:
        save(destination, choice, exclusive=True)
    return choice


def full_paths_by_dataset(root, plan, entries, valid, args, timeout):
    """Prioritize a dataset's final path immediately after its own three folds.

    Submit only as many processes as CPUs. Keeping later folds in this local
    deque allows a newly eligible outer path to take the next free CPU instead
    of sitting behind every remaining dataset's already-submitted folds.
    """
    todo, outcomes, pending = deque(), [], {}
    available = deque(args.cpus[:args.workers])
    by_id = {e["dataset_id"]:e for e in entries}
    states = {}
    for entry in entries:
        did = entry["dataset_id"]
        missing = [s for s in ("train","fold0","fold1","fold2") if (did,s) not in valid]
        states[did] = dict(phase="blocked_calibration" if missing else "cv_pending",
                           missing_calibrations=missing, completed_folds=[])
        if not missing:
            todo.extend(dict(kind="path",stage="full",dataset_id=did,split=f"fold{k}",
                             arm="PATH",taus=plan["taus"]) for k in range(3))

    def progress():
        save(root/"full_dependency_progress.json", dict(at=now(),
            stage="PATH-only CV followed by each eligible dataset's outer path",
            completed_processes=len(outcomes), queued_paths=len(todo),
            active=[dict(dataset_id=spec["dataset_id"],split=spec["split"],cpu=cpu)
                    for spec,cpu in pending.values()], datasets=states))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while todo or pending:
            while todo and available:
                spec, cpu = todo.popleft(), available.popleft()
                states[spec["dataset_id"]]["phase"] = "outer_running" if spec["split"] == "train" else "cv_running"
                future = pool.submit(dispatch,root,plan,spec,cpu,args.memory_gib,timeout)
                pending[future] = (spec,cpu)
            progress()
            completed, _ = wait(pending,timeout=30,return_when=FIRST_COMPLETED)
            for future in completed:
                spec,cpu = pending.pop(future)
                receipt = future.result()  # A changed frozen hash is fatal, never silently retried.
                outcomes.append(receipt)
                available.append(cpu)
                did = spec["dataset_id"]
                print(now(),"PATH_PROCESS",did,spec["split"],receipt["status"],flush=True)
                if spec["split"] != "train":
                    states[did]["completed_folds"].append(spec["split"])
                    if len(states[did]["completed_folds"]) == 3:
                        choice = cv_choice(root,plan,by_id[did])
                        if choice is None:
                            states[did]["phase"] = "blocked_cv_not_fully_verified"
                        else:
                            states[did].update(phase="outer_queued",selected_tau=choice["selected_tau"])
                            todo.appendleft(dict(kind="path",stage="full",dataset_id=did,split="train",
                                                 arm="PATH",taus=plan["taus"]))
                            print(now(),"CV_SELECTED",did,"tau",choice["selected_tau"],
                                  "outer path prioritized",flush=True)
                else:
                    completion = jobdir(root,spec)/"completion.json"
                    verified = completion.exists() and load(completion).get("all_verified") is True
                    states[did]["phase"] = "complete" if verified else "outer_incomplete_or_failed"
                    states[did]["outer_process_status"] = receipt["status"]
            progress()
    progress()
    return outcomes


def aggregate(root, stage):
    import numpy as np
    rows = []
    for path in sorted((root / "jobs" / stage).glob("*/*/*/point_*/result.json")):
        row = load(path)
        row["result_path"] = str(path.relative_to(root))
        rows.append(row)
    bykey = {(r["dataset_id"],r["point"]):r for r in rows
             if r["split"] == "train" and r["arm"] == "BASE" and r.get("verified")}
    for row in rows:
        base = bykey.get((row["dataset_id"],row["point"]))
        if not base or row["split"] != "train" or not row.get("verified"):
            continue
        active = lambda r: {tuple(p) for p,w in zip(r["patterns"],r["w"]) if abs(w)>1e-8}
        row["agreement"] = dict(same_active=active(row)==active(base),
            predictions_close=bool(np.allclose(row["prediction"],base["prediction"],rtol=1e-5,atol=1e-5)),
            maximum_prediction_difference=float(np.max(np.abs(np.asarray(row["prediction"])-base["prediction"]))))
    save(root / (stage + "_all_results.json"), rows)
    counts = {s:sum(r["status"]==s for r in rows) for s in sorted({r["status"] for r in rows})}
    paths = []
    for directory in sorted((root / "jobs" / stage).glob("*/*/*")):
        request_path = directory / "request.json"
        if not request_path.exists():
            continue
        request = load(request_path)
        selected = [r for r in rows if r["dataset_id"] == request["dataset_id"]
                    and r["split"] == request["split"] and r["arm"] == request["arm"]]
        complete = len(selected) == len(request["taus"]) and all(r.get("verified",False) for r in selected)
        calibration_path = root / "lambda_max" / request["dataset_id"] / request["split"] / "result.json"
        calibration = load(calibration_path) if calibration_path.exists() else {}
        process = load(directory / "process.json") if (directory / "process.json").exists() else {}
        fit_seconds = sum(r.get("fit_seconds",0) for r in selected)
        nodes = sum(r.get("summary",{}).get("nodes_visited",0) for r in selected)
        paths.append(dict(dataset_id=request["dataset_id"], split=request["split"], arm=request["arm"],
            all_verified=complete, completed_points=sum(bool(r.get("verified")) for r in selected),
            expected_points=len(request["taus"]), process_status=process.get("status","not_finished"),
            process_seconds=process.get("seconds"), last_progress=process.get("last_progress"),
            completed_fit_seconds=fit_seconds, completed_fit_nodes=nodes,
            independent_audit_seconds=sum(r.get("independent",{}).get("seconds",0) for r in selected),
            common_calibration_seconds=calibration.get("seconds"),
            common_calibration_nodes=calibration.get("stats",{}).get("nodes_visited"),
            complete_path_seconds_with_common_calibration=fit_seconds+calibration["seconds"] if complete else None,
            complete_path_nodes_with_common_calibration=nodes+calibration["stats"]["nodes_visited"] if complete else None,
            peak_process_rss_mib=max((r.get("process_peak_rss_mib_after_current_audit",0) for r in selected),default=0),
            cost_scope="Completed fit subtotal is incomplete when all_verified=false; common calibration charged equally; audits separate"))
    report = dict(at=now(), stage=stage, rows=len(rows), statuses=counts, paths=paths,
        verified=sum(bool(r.get("verified")) for r in rows),
        agreement_failures=[{k:r[k] for k in ("dataset_id","point","arm","agreement")}
            for r in rows if "agreement" in r and not all(r["agreement"][k] for k in ("same_active","predictions_close"))])
    save(root / (stage + "_summary.json"), report)
    return report


def run(args):
    root = Path(args.root).resolve()
    plan = check(root, all_files=True)
    if plan.get("version") != VERSION or plan.get("arms") != ["PATH"]:
        raise ValueError("This v3 controller accepts only its registered PATH-only plan")
    if args.workers < 1 or len(args.cpus) < args.workers or len(set(args.cpus)) != len(args.cpus):
        raise ValueError("Need distinct CPU ids for every worker")
    if not 0 < args.memory_gib <= 8:
        raise ValueError("At most 8 GiB per registered worker")
    with (root / ("." + args.mode + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        allowed = plan["pilot_datasets"] if args.mode == "pilot" else [e["dataset_id"] for e in plan["datasets"]]
        selected = args.datasets or allowed
        if not set(selected) <= set(allowed):
            raise ValueError("Dataset outside this registered stage")
        entries = [e for e in plan["datasets"] if e["dataset_id"] in selected]
        splits = ["train"] if args.mode == "pilot" else ["train","fold0","fold1","fold2"]
        prep = [dict(kind="lambda", dataset_id=e["dataset_id"], split=s) for e in entries for s in splits]
        timeout = args.timeout if args.timeout is not None else (600 if args.mode == "pilot" else 0)
        outcomes = jobs_run(root, plan, prep, args, args.prepare_timeout)
        valid = set()
        for entry in entries:
            for split in splits:
                result_path = root/"lambda_max"/entry["dataset_id"]/split/"result.json"
                if result_path.exists():
                    calibration = load(result_path)
                    if calibration.get("status") == "complete" and calibration.get("exact") is True:
                        valid.add((entry["dataset_id"],split))
        def specs(es, split_names, arms, taus):
            return [dict(kind="path",stage=args.mode,dataset_id=e["dataset_id"],split=s,arm=a,taus=taus)
                    for e in es for s in split_names for a in arms if (e["dataset_id"],s) in valid]
        if args.mode == "full":
            outcomes += full_paths_by_dataset(root,plan,entries,valid,args,timeout)
        else:
            outcomes += jobs_run(root,plan,specs(entries,["train"],ARMS,plan["pilot_taus"]),args,timeout)
        result = aggregate(root,args.mode)
        save(root / (args.mode + "_controller.json"),dict(at=now(),datasets=selected,
            process_outcomes=outcomes,summary=result,timeout_seconds=timeout or None))
        print(json.dumps(result,indent=2))


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare","run","worker","check","aggregate"))
    parser.add_argument("--root",required=True)
    parser.add_argument("--data-root",default=str(repo/"outputs/extension_baselines_20260928"))
    parser.add_argument("--model",default=str(repo/"model.py"))
    parser.add_argument("--audit-model",default=str(repo/"outputs/experiment_revision_20260927/length_snapshot_20260928/audit/model.py"))
    parser.add_argument("--mode",choices=("pilot","full"),default="pilot")
    parser.add_argument("--datasets",nargs="+")
    parser.add_argument("--workers",type=int,default=3)
    parser.add_argument("--cpus",nargs="+",type=int,default=list(range(30)))
    parser.add_argument("--memory-gib",type=float,default=8)
    parser.add_argument("--timeout",type=float,default=None,help="Seconds per path; 0 disables cap")
    parser.add_argument("--prepare-timeout",type=float,default=600,help="Separate lambda-max worker budget; 0 disables")
    parser.add_argument("--request")
    parser.add_argument("--cpu",type=int)
    args = parser.parse_args()
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        run(args)
    elif args.action == "worker":
        worker(args)
    elif args.action == "check":
        plan = check(Path(args.root).resolve(),all_files=True)
        print(json.dumps(dict(ok=True,files=len(plan["hashes"]),datasets=len(plan["datasets"]))))
    else:
        print(json.dumps(aggregate(Path(args.root).resolve(),args.mode),indent=2))


if __name__ == "__main__":
    main()
