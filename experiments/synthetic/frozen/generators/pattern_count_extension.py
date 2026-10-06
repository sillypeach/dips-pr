"""Predeclared pattern-count sensitivity test using an archived, unchanged solver."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import queue
import shutil
import statistics
import sys

import numpy as np


COUNTS = [2, 4, 8, 12]
SEEDS = list(range(2609271101, 2609271106))
ARMS = ["BASE", "Full", "KKT-priced"]
WEIGHTS = [.9, -.8, .75, 1.1, -.85, .95, .8, -1., .9, -.8, .75, 1.1]
SOURCES = ["model.py", "pruning_regime_search.py", "paper_pruning_usefulness.py",
           "uv_staged_solver.py", "uv_only_solver.py", "uv_only_benchmark.py",
           "core_screening_validation.py", "dual_reference_repair.py"]


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_for(count):
    return dict(id=f"truths_{count:02d}", n=2000, p=64, signal=.65, noise=.20,
                baseline=30., support=.015, truths=count, weight_scale=1.)


def generate(case, seed):
    n, p, count = case["n"], case["p"], case["truths"]
    if not 1 <= count <= len(WEIGHTS) or not 4*count <= p <= 96 or n < 20:
        raise ValueError("Insufficient samples/items or undefined coefficients")
    streams = np.random.SeedSequence([seed, 202609244]).spawn(4)
    sr, nr, yr, pr = [np.random.default_rng(s) for s in streams]
    signals = sr.random((n, count, 4)) < case["signal"]
    noise = nr.random((n, 96))[:, :p-4*count] < case["noise"]
    weights = np.asarray(WEIGHTS[:count])*case["weight_scale"]
    mu = np.exp(np.log(case["baseline"])+signals.all(axis=2)@weights)
    y = yr.poisson(mu)
    permutation = pr.permutation(p)
    X = np.column_stack((signals.reshape(n, 4*count), noise))
    tx = [sorted(permutation[np.flatnonzero(row)].tolist()) for row in X]
    truth = [sorted(permutation[4*j:4*j+4].tolist()) for j in range(count)]
    ntrain = int(.7*n)
    return dict(transactions=tx, y=y.tolist(), true_mean=mu.tolist(),
                true_patterns=truth, true_weights=weights.tolist(),
                train_indices=list(range(ntrain)), test_indices=list(range(ntrain, n)),
                case=case, seed=seed, permutation=permutation.tolist())


def dependencies(source):
    sys.path.insert(0, str(source))
    import pruning_regime_search as helper
    return helper


def prepare(args):
    root, parent = Path(args.output).resolve(), Path(args.parent).resolve()
    if root.exists():
        raise ValueError("Use a fresh directory; previous attempts must remain intact")
    source = root/"source"
    source.mkdir(parents=True)
    for name in SOURCES:
        shutil.copyfile(parent/"source"/name, source/name)
    shutil.copyfile(__file__, source/Path(__file__).name)
    (root/"audit").mkdir()
    shutil.copyfile(parent/"audit/model.py", root/"audit/model.py")
    parent_plan = load(parent/"protocol.json")
    for name in SOURCES:
        if sha(source/name) != parent_plan["source_sha256"]["source/"+name]:
            raise ValueError("Parent source does not match its registered hash")
    plan = dict(version="pattern-count-frozen-extension-v1", at=now(),
        parent_protocol_sha256=sha(parent/"protocol.json"), counts=COUNTS, seeds=SEEDS,
        cases=[case_for(c) for c in COUNTS], arms=ARMS, weights=WEIGHTS,
        ratio=.03, policy=parent_plan["policy"], timeout_seconds=180,
        planned_fits=len(COUNTS)*len(SEEDS)*len(ARMS), repetitions=1,
        source_sha256={str(p.relative_to(root)):sha(p)
            for folder in (source, root/"audit") for p in folder.glob("*.py")},
        notes=[
            "All counts and five new seed IDs fixed before generation. No outcome-based seed selection, retries or retuning. This extends a previously development-selected favorable generator; it is not a universal benchmark.",
            "Same 2000 samples, 64 items, 70/30 train/test split, independent four-item AND truths, signal marginal 0.65, nuisance marginal 0.20, baseline 30, native Poisson sampling.",
            "For <=8 truths the generator and coefficient prefix exactly reproduce the archived generator. For 12, repeat the original first four coefficients after the original eight. No coefficient optimization.",
            "Holding p fixed means more truths also changes signal/nuisance item counts and total signal/mean distribution. This is a count-extension stress test, not an isolated causal effect of count.",
            "Lambda=0.03*sum(training y), kappa=0.05, support=max(3,floor(0.015*n_train)), max pattern length 4, top-K expansion 200. Formula fixed, numerical lambda varies with training responses. Rule was selected on earlier development data.",
            "True patterns used ONLY for recovery reporting, not solver stopping, model-size constraints or tuning. Exact equality, coefficient cutoff 1e-8. No second-stage screening/refit.",
            "Full uses dynamic working sets with sphere u/v and the previous-outer-round second reference, no independent KKT candidate rescue. BASE shares the driver with screening/removal disabled. KKT-priced is a separate comparator.",
            "Every completed fit independently audited for full-dictionary gap and KKT; audit never repairs output. Failed/nonconverged/timed-out cases retained, never counted as converged results.",
            "Four pinned physical-core jobs may run concurrently; arms within each dataset are serial in rotated order. Single BLAS thread. Wall-clock timings are diagnostic, not clean speedup evidence. Node comparisons use cumulative whole-fit visits.",
            "This is a recovery-count experiment, not a replacement for the full u/v/2ref ablation. No manuscript edits."])
    save(root/"protocol.json", plan)
    save(root/"registered.json", dict(at=now(), protocol_sha256=sha(root/"protocol.json")))
    print("PREPARED", root, plan["planned_fits"], "fits", flush=True)


def summarize(root, rows, plan, helper):
    helper.pair(rows)
    groups = []
    for count in plan["counts"]:
        for arm in plan["arms"]:
            rs = [r for r in rows if r["setting"] == case_for(count)["id"] and r["arm"] == arm]
            valid = [r for r in rs if r.get("verified")]
            group = dict(truths=count, arm=arm, attempted=len(rs), planned=len(plan["seeds"]),
                         verified=len(valid), complete=len(valid) == len(plan["seeds"]))
            for key in ("NZ", "TP", "FP", "FN", "precision", "recall", "f1", "test_r2", "test_pr2"):
                if valid:
                    group[key] = dict(mean=statistics.mean(r[key] for r in valid),
                                     std=statistics.stdev(r[key] for r in valid) if len(valid)>1 else None)
            comparisons = [r["comparison"] for r in valid if "comparison" in r]
            group["matched_base_pairs"] = len(comparisons)
            if comparisons:
                group["mean_node_reduction"] = statistics.mean(c["node_reduction"] for c in comparisons)
                group["all_same_active"] = all(c["same_active"] for c in comparisons)
                group["all_predictions_close"] = all(c["predictions_close"] for c in comparisons)
            groups.append(group)
    save(root/"all_results.json", rows)
    save(root/"summary.json", dict(at=now(), groups=groups, attempted=len(rows),
        verified=sum(bool(r.get("verified")) for r in rows), planned=plan["planned_fits"]))
    save(root/"progress.json", dict(at=now(), attempted=len(rows), planned=plan["planned_fits"]))
    lines = ["# Pattern-count extension", "", "Fixed 2/4/8/12 truths x five seeds x three arms. No seed selection or retuning.",
        "Means below use independently verified fits; incomplete groups are not complete-study estimates. Standard deviations are stored in summary.json.",
        "No postselection/refit. Exact itemset match; NZ uses |coefficient| > 1e-8. Test R2 is held-out prediction; recovery concerns training-selected patterns.", "",
        "|Truths|Arm|Verified/planned|Mean NZ|Precision|Recall|F1|Test R2|Test PR2|BASE pairs|Node reduction|",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        metrics = "|".join(f"{g[k]['mean']:.4f}" if k in g else "NA" for k in
                           ("NZ", "precision", "recall", "f1", "test_r2", "test_pr2"))
        reduction = f"{g['mean_node_reduction']:.2%}" if "mean_node_reduction" in g else "NA"
        lines.append(f"|{g['truths']}|{g['arm']}|{g['verified']}/{g['planned']}|{metrics}|{g['matched_base_pairs']}|{reduction}|")
    lines += ["", "## All attempts", "", "|Truths|Seed|Arm|Process|Fit status|Verified|TP|FP|FN|NZ|P|R|F1|R2|", "|---:|---:|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r:(r["truths"], r["seed"], r["arm"])):
        metrics = "|".join(f"{r[k]:.4f}" if k in r else "NA" for k in
                           ("TP", "FP", "FN", "NZ", "precision", "recall", "f1", "test_r2"))
        lines.append(f"|{r['truths']}|{r['seed']}|{r['arm']}|{r['process_status']}|{r['status']}|{r.get('verified',False)}|{metrics}|")
    lines += ["", "## Interpretation limits", ""] + ["- "+note for note in plan["notes"]]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n")


def run(args):
    root = Path(args.output).resolve()
    helper = dependencies(root/"source")
    plan = helper.check_sources(root)
    if (root/"started.json").exists():
        raise ValueError("Already started; refusing to overwrite or silently retry")
    cpus = args.cpus
    if len(set(cpus)) != len(cpus):
        raise ValueError("Duplicate CPU IDs")
    if hasattr(os, "sched_getaffinity") and not set(cpus) <= os.sched_getaffinity(0):
        raise ValueError("Requested CPU outside allowed affinity")
    save(root/"started.json", dict(at=now(), host=platform.node(), python=sys.version,
        executable=sys.executable, cpus=cpus, numpy=np.__version__, parent_pid=os.getpid()))
    blocks = []
    for i, seed in enumerate(plan["seeds"]):
        for j, case in enumerate(plan["cases"]):
            data = generate(case, seed)
            path = root/"data/extension"/f"{case['id']}_s{seed}.json"
            save(path, data)
            blocks.append((i+j, data, sha(path)))
    save(root/"datasets_registered.json", dict(at=now(), hashes={str(p.relative_to(root)):sha(p)
        for p in (root/"data/extension").glob("*.json")}))
    free_cpus = queue.Queue()
    for cpu in cpus:
        free_cpus.put(cpu)

    def block_work(block):
        offset, data, data_hash = block
        cpu = free_cpus.get()
        try:
            case, seed = data["case"], data["seed"]
            offset %= len(plan["arms"])
            arms = plan["arms"][offset:] + plan["arms"][:offset]
            results = []
            for arm in arms:
                row = helper.dispatch(root, plan, data, "extension", case["id"], plan["ratio"], seed, 0, arm, cpu)
                row.update(truths=case["truths"], dataset_sha256=data_hash)
                if "patterns" in row:
                    selected = helper.active_set(row["patterns"], row["w"])
                    truth = {tuple(p) for p in data["true_patterns"]}
                    row.update(TP=len(selected & truth), FP=len(selected-truth), FN=len(truth-selected))
                results.append(row)
            save(root/"blocks"/f"{case['id']}_s{seed}.json", results)
            return results
        finally:
            free_cpus.put(cpu)

    rows = []
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        futures = [pool.submit(block_work, block) for block in blocks]
        for future in as_completed(futures):
            rows.extend(future.result())
            summarize(root, rows, plan, helper)
    save(root/"completion.json", dict(at=now(), attempted=len(rows), planned=plan["planned_fits"],
        all_attempted=len(rows)==plan["planned_fits"], all_verified=all(r.get("verified",False) for r in rows)))


def verify(args):
    root = Path(args.output).resolve()
    helper = dependencies(root/"source")
    plan = helper.check_sources(root)
    rows = load(root/"all_results.json")
    identities = {(r["truths"], r["seed"], r["arm"]) for r in rows}
    expected = {(c,s,a) for c in plan["counts"] for s in plan["seeds"] for a in plan["arms"]}
    if identities != expected or len(rows) != len(expected):
        raise ValueError("Missing or duplicate planned attempts")
    data_hashes = load(root/"datasets_registered.json")["hashes"]
    for name, digest in data_hashes.items():
        data = load(root/name)
        if sha(root/name) != digest or data != generate(data["case"], data["seed"]):
            raise ValueError("Dataset hash/reproduction mismatch")
    for r in rows:
        job = root/"jobs/extension"/r["setting"]/f"s{r['seed']}"/"r0"/r["arm"]
        req, proc = load(job/"request.json"), load(job/"process.json")
        if req["data_sha256"] != r["dataset_sha256"] or sha(root/req["data"]) != req["data_sha256"]:
            raise ValueError("Dataset routing mismatch")
        data = load(root/req["data"])
        if req["config"] != helper.arm_config(helper.config(data, plan["ratio"]), r["arm"]):
            raise ValueError("Config does not match frozen rule")
        if not (job/"process.log").exists():
            raise ValueError("Missing process log")
        if "patterns" not in r:
            continue
        selected = helper.active_set(r["patterns"], r["w"])
        truth = {tuple(p) for p in data["true_patterns"]}
        tp, nz = len(selected & truth), len(selected)
        calculated = dict(TP=tp, FP=nz-tp, FN=len(truth)-tp, NZ=nz,
            precision=tp/max(1,nz), recall=tp/len(truth), f1=2*tp/max(1,nz+len(truth)))
        y = np.asarray(data["y"],float)[data["test_indices"]]
        pred = np.asarray(r["prediction"])
        calculated["test_r2"] = 1-float(np.sum((y-pred)**2)/np.sum((y-y.mean())**2))
        train = np.asarray(data["y"],float)[data["train_indices"]]
        dev = lambda mu: float(2*np.sum(helper.xlogy(y,y/mu)-y+mu))
        calculated["test_pr2"] = 1-dev(pred)/dev(np.full(len(y),train.mean()))
        if not all(np.isclose(r[k],v,rtol=1e-10,atol=1e-10) for k,v in calculated.items()):
            raise ValueError("Metric mismatch")
        if r.get("verified") and not (proc["returncode"]==0 and r["status"]=="complete"
                and r["independent"]["status"]=="passed"
                and r["independent"]["gap_per_sample"]<=req["config"]["gap_tol"]
                and r["independent"]["kkt"]<=req["config"]["solver_tol"]):
            raise ValueError("Unjustified verified flag")
    summarize(root, rows, plan, helper)
    record = dict(at=now(), integrity_checks_passed=True, attempts=len(rows),
        datasets=len(data_hashes), verified=sum(bool(r.get("verified")) for r in rows),
        sources=len(plan["source_sha256"]), role="Recomputed saved metrics and checked logged audits; no fitting")
    save(root/"verification.json",record)
    print(json.dumps(record,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--output",required=True)
    p.add_argument("--parent",required=True)
    p = sub.add_parser("run")
    p.add_argument("--output",required=True)
    p.add_argument("--cpus",type=int,nargs="+",default=[0])
    p = sub.add_parser("verify")
    p.add_argument("--output",required=True)
    args = parser.parse_args()
    globals()[args.command](args)
