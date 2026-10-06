"""Frozen-solver, all-seed mixed-order length stress test with registered stopping."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import mixed_order_recovery as prior

save, load, sha, now = prior.save, prior.load, prior.sha, prior.now
LEVELS = list(range(5, 11))
SEEDS = list(range(2609272301, 2609272306))
ARMS = ["Full", "KKT-priced"]


def case_for(level):
    if level not in LEVELS:
        raise ValueError("Unregistered maximum order")
    return dict(prior.case_for(4, "mixed_equal_support"),
                id=f"lengths_{level-3}_{level}", maximum_order=level)


def generate(case, seed):
    level = case["maximum_order"]
    if level not in LEVELS or case["truths"] != 4 or case["p"] < 4*level-6:
        raise ValueError("Invalid length-ramp case")
    n, p = case["n"], case["p"]
    sr, nr, yr, pr, lr = [np.random.default_rng(s) for s in
        np.random.SeedSequence([seed, 202609244]).spawn(5)]
    lengths = np.arange(level-3, level+1)
    lr.shuffle(lengths)
    probs = case["signal"]**(4/lengths)
    signals = sr.random((n, int(lengths.sum()))) < np.repeat(probs, lengths)
    edges = np.r_[0, np.cumsum(lengths)]
    truth_x = np.column_stack([signals[:,edges[j]:edges[j+1]].all(axis=1) for j in range(4)])
    noise = nr.random((n,96))[:,:p-int(lengths.sum())] < case["noise"]
    weights = np.asarray(prior.base.WEIGHTS[:4])*case["weight_scale"]
    mu = np.exp(np.log(case["baseline"])+truth_x@weights)
    y = yr.poisson(mu)
    permutation = pr.permutation(p)
    X = np.column_stack((signals, noise))
    tx = [sorted(permutation[np.flatnonzero(row)].tolist()) for row in X]
    truth = [sorted(permutation[edges[j]:edges[j+1]].tolist()) for j in range(4)]
    cut = int(.7*n)
    return dict(transactions=tx, y=y.tolist(), true_mean=mu.tolist(), true_patterns=truth,
        true_weights=weights.tolist(), train_indices=list(range(cut)), test_indices=list(range(cut,n)),
        case=case, seed=seed, permutation=permutation.tolist(), true_orders=lengths.tolist(),
        signal_probabilities=probs.tolist())


def config(data, plan, helper):
    return dict(helper.config(data, plan["ratio"]), max_len=data["case"]["maximum_order"])


def recovery(row, data, helper):
    selected = helper.active_set(row["patterns"], row["w"])
    truth = {tuple(p) for p in data["true_patterns"]}
    tp, nz = len(selected & truth), len(selected)
    out = dict(TP=tp, FP=nz-tp, FN=len(truth)-tp, NZ=nz,
        precision=tp/max(1,nz), recall=tp/len(truth), f1=2*tp/max(1,nz+len(truth)))
    out["by_order"] = {}
    for order in range(1, data["case"]["maximum_order"]+1):
        t, s = {p for p in truth if len(p)==order}, {p for p in selected if len(p)==order}
        hit = len(t & s)
        out["by_order"][str(order)] = dict(truth=len(t), NZ=len(s), TP=hit, FP=len(s)-hit,
            FN=len(t)-hit, recall=hit/len(t) if t else None,
            precision=hit/len(s) if s else None, f1=2*hit/(len(t)+len(s)) if t or s else None)
    return out


def prepare(args):
    root, parent = Path(args.output).resolve(), Path(args.parent).resolve()
    if root.exists():
        raise ValueError("Use a fresh directory")
    old = load(parent/"protocol.json")
    if sha(parent/"protocol.json") != load(parent/"registered.json")["protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    for name, digest in old["source_sha256"].items():
        if sha(parent/name) != digest:
            raise ValueError("Parent source changed: "+name)
        (root/name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(parent/name, root/name)
    shutil.copyfile(__file__, root/"source"/Path(__file__).name)
    plan = dict(version="mixed-order-length-ramp-v1", at=now(),
        parent_protocol_sha256=sha(parent/"protocol.json"), levels=LEVELS, seeds=SEEDS,
        cases=[case_for(level) for level in LEVELS], arms=ARMS, ratio=.03, policy=old["policy"],
        timeout_seconds=900, address_space_gib=8, max_attempts=60,
        stop_rule=dict(min_verified=3, f1_drop=.15, per_seed_f1_drop=.10, worse_seeds=4,
                       precision_floor=.6, recall_floor=.75, f1_floor=.7, consecutive=2),
        source_sha256={str(p.relative_to(root)):sha(p) for folder in (root/"source",root/"audit") for p in folder.glob("*.py")},
        notes=[
            "Fixed four disjoint truths. Increase their lengths together: [2,3,4,5], [3,4,5,6], ..., [7,8,9,10]. All five NEW seed IDs are paired across levels; retain every seed, including failed fits. No hyperparameter tuning or postselection.",
            "Same n=2000,p=64,70/30 split, beta=[0.9,-0.8,0.75,1.1], baseline mean30, nuisance item probability0.20. Length-to-coefficient assignment shuffled by seed. Conditional Poisson responses. Level5 reproduces the prior equal-support generator for the same seed.",
            "Signal item probability=0.65**(4/L), hence population truth support probability=0.65**4 at every length. Actual supports not forced. Holding p fixed changes nuisance count, marginal frequencies and correlations as length grows; this is not a one-factor causal experiment.",
            "Search max_len grows with the largest embedded length; all shorter closed frequent patterns remain eligible. The maximum order is a declared experimental dictionary parameter, not inferred from observed test outcomes. Truth count and individual truths are never passed to the solver.",
            "Previously development-selected rule lambda=0.03*sum(training y) remains fixed, kappa0.05, support21, top_k200, maximum100 rounds, and all frozen staged solver settings. No second-stage screening, coefficient cutoff changes, refit or KKT candidate rescue in Full.",
            "Full uses u/v/previous-round2ref; KKT-priced is a separately fitted comparator. Each completed fit has a full allowed-dictionary post-fit gap/KKT audit that never repairs it. Recovery uses exact equality and |coef|>1e-8. Subsets and supersets remain FPs.",
            "At each level run all five seeds and both arms before making a level-wide stop decision. Strong recovery deterioration is defined descriptively, NOT as statistical significance: complete Full results have mean P<0.6, R<0.75 or F1<0.7; OR mean F1 decreases >=0.15 from level5 with >=4/5 paired drops >=0.10. Stop after two consecutive qualifying levels, so the first deterioration gets a next-level check.",
            "Separately stop if fewer than3/5 Full fits pass independent verification at a level. This is an optimization/resource boundary, not evidence of low statistical precision. Verified comparator output may distinguish objective recovery from Full computational failure. If neither boundary occurs, stop at maximum order10 and report no collapse found within tested range.",
            "900 seconds per arm, including independent audit; Linux address-space cap8GiB per worker. Keep timeouts, numerical failures and nonconvergence. No automatic retries or silent budgets changes. Means from incomplete groups are conditional and not complete-study results.",
            "Up to five single-BLAS-thread jobs on distinct pinned cores. Arms run serially per dataset in alternating order. All time costs logged; concurrent timings diagnostic, not clean speedup claims. This is not a pruning ablation.",
            "All30 datasets generated and hashed before fitting. Protocol, code hashes, per-job requests/logs, predictions and active itemsets retained. Stop decisions replayable. No manuscript edits."])
    save(root/"protocol.json", plan)
    save(root/"registered.json", dict(at=now(), protocol_sha256=sha(root/"protocol.json")))
    print("PREPARED", root, "up to 60 fits", flush=True)


def level_stats(rows, level, arm, plan):
    rs = [r for r in rows if r["level"]==level and r["arm"]==arm]
    ok = [r for r in rs if r.get("verified")]
    out = dict(level=level, setting=case_for(level)["id"], arm=arm, attempted=len(rs),
               verified=len(ok), planned=len(plan["seeds"]), complete=len(ok)==len(plan["seeds"]))
    out["timeouts"] = sum(r["process_status"]=="timeout" for r in rs)
    for key in ("NZ", "precision", "recall", "f1", "test_r2", "test_pr2", "seconds"):
        if ok:
            vals = [r[key] for r in ok]
            out[key] = dict(mean=statistics.mean(vals), std=statistics.stdev(vals) if len(vals)>1 else None)
    return out


def assessment(rows, level, plan):
    rule = plan["stop_rule"]
    group = level_stats(rows, level, "Full", plan)
    resource = group["verified"] < rule["min_verified"]
    recovery_bad, paired_drop, worse = False, None, None
    if group["complete"]:
        recovery_bad = (group["precision"]["mean"]<rule["precision_floor"] or
                        group["recall"]["mean"]<rule["recall_floor"] or group["f1"]["mean"]<rule["f1_floor"])
        baseline = {r["seed"]:r for r in rows if r["level"]==5 and r["arm"]=="Full" and r.get("verified")}
        if len(baseline)==len(plan["seeds"]):
            current = [r for r in rows if r["level"]==level and r["arm"]=="Full"]
            differences = [baseline[r["seed"]]["f1"]-r["f1"] for r in current]
            paired_drop = statistics.mean(differences)
            worse = sum(d>=rule["per_seed_f1_drop"] for d in differences)
            recovery_bad |= paired_drop>=rule["f1_drop"] and worse>=rule["worse_seeds"]
    return dict(level=level, full_verified=group["verified"], full_complete=group["complete"],
                resource_boundary=resource, recovery_deterioration=bool(recovery_bad),
                paired_f1_drop=paired_drop, paired_worse_seeds=worse)


def decide(rows, level, plan):
    result = assessment(rows, level, plan)
    reason = None
    if result["resource_boundary"]:
        reason = "optimization_or_resource_boundary"
    elif level>5 and result["recovery_deterioration"] and assessment(rows,level-1,plan)["recovery_deterioration"]:
        reason = "recovery_deterioration_two_consecutive_levels"
    elif level==max(plan["levels"]):
        reason = "maximum_registered_length"
    return dict(result, stop=reason is not None, reason=reason)


def report(root, rows, plan):
    prior.compare(rows)
    levels = sorted({r["level"] for r in rows})
    groups = [level_stats(rows,level,arm,plan) for level in levels for arm in ARMS]
    save(root/"all_results.json", rows)
    save(root/"summary.json", dict(at=now(), attempts=len(rows), groups=groups))
    lines = ["# Mixed-order length stress test", "",
        "Exact recovery, no refit. Five fixed NEW seeds per level, same four truths and fixed lambda rule.",
        "Only verified fits enter averages; incomplete groups are NOT estimates over all five seeds. Timeouts are not assigned zero precision.", "",
        "|Lengths|Arm|Verified/planned|Timeouts|NZ|Precision|Recall|F1|Test R2|Test PR2|Train seconds|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        values = "|".join(f"{g[k]['mean']:.4f}" if k in g else "NA" for k in ("NZ","precision","recall","f1","test_r2","test_pr2","seconds"))
        lines.append(f"|{g['level']-3}--{g['level']}|{g['arm']}|{g['verified']}/{g['planned']}|{g['timeouts']}|{values}|")
    lines += ["", "## Every attempt", "", "|Lengths|Seed|Arm|Process|Fit|Verified|NZ|P|R|F1|R2|", "|---|---:|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows,key=lambda r:(r["level"],r["seed"],r["arm"])):
        values = "|".join(f"{r[k]:.4f}" if k in r else "NA" for k in ("NZ","precision","recall","f1","test_r2"))
        lines.append(f"|{r['level']-3}--{r['level']}|{r['seed']}|{r['arm']}|{r['process_status']}|{r['status']}|{r.get('verified',False)}|{values}|")
    lines += ["", "## Registered interpretation and limits", ""]+["- "+note for note in plan["notes"]]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n")


def dispatch(root, plan, data, arm, cpu, helper):
    case, seed = data["case"], data["seed"]
    path = root/"data/length_ramp"/f"{case['id']}_s{seed}.json"
    job = root/"jobs"/case["id"]/f"s{seed}"/arm
    req = dict(stage="length_ramp",setting=case["id"],level=case["maximum_order"],seed=seed,arm=arm,cpu=cpu,
        data=str(path.relative_to(root)),data_sha256=sha(path),config=helper.arm_config(config(data,plan,helper),arm))
    save(job/"request.json",req)
    def event(message):
        line=f"{now()} {message} {case['id']} {seed} {arm}"
        print(line,flush=True)
        with (root/"controller.log").open("a") as log:
            log.write(line+"\n")
    event("START")
    tick=time.perf_counter()
    try:
        with (job/"process.log").open("w") as log:
            proc=subprocess.run([sys.executable,str(root/"source"/Path(__file__).name),"worker","--output",str(root),"--request",str(job/"request.json")],stdout=log,stderr=log,timeout=plan["timeout_seconds"])
        status,code="exited",proc.returncode
    except subprocess.TimeoutExpired:
        status,code="timeout",None
    save(job/"process.json",dict(at=now(),status=status,returncode=code,seconds=time.perf_counter()-tick))
    row=load(job/"result.json") if (job/"result.json").exists() else dict(status="no_completed_fit")
    row.update({k:req[k] for k in ("stage","setting","level","seed","arm")})
    row.update(process_status=status,dataset_sha256=req["data_sha256"],process_seconds=time.perf_counter()-tick)
    if code!=0:
        row["verified"]=False
    if "patterns" in row:
        row.update(recovery(row,data,helper))
    save(job/"evaluated.json",row)
    event("END verified="+str(row.get("verified",False)))
    return row


def run(args):
    root=Path(args.output).resolve()
    helper=prior.base.dependencies(root/"source")
    plan=helper.check_sources(root)
    cpus=args.cpus
    if (root/"started.json").exists():
        raise ValueError("Already started; no silent overwrite or retry")
    if len(set(cpus))!=len(cpus) or (hasattr(os,"sched_getaffinity") and not set(cpus)<=os.sched_getaffinity(0)):
        raise ValueError("Invalid CPU allocation")
    save(root/"started.json",dict(at=now(),host=platform.node(),python=sys.version,executable=sys.executable,cpus=cpus))
    for case in plan["cases"]:
        for seed in plan["seeds"]:
            save(root/"data/length_ramp"/f"{case['id']}_s{seed}.json",generate(case,seed))
    save(root/"datasets_registered.json",dict(at=now(),hashes={str(p.relative_to(root)):sha(p) for p in (root/"data/length_ramp").glob("*.json")}))
    rows=[]
    for level in plan["levels"]:
        case=case_for(level)
        def work(index_seed):
            index,seed=index_seed
            data=load(root/"data/length_ramp"/f"{case['id']}_s{seed}.json")
            offset=(level+index)%len(ARMS)
            arms=ARMS[offset:]+ARMS[:offset]
            return [dispatch(root,plan,data,arm,cpus[index],helper) for arm in arms]
        if len(cpus)<len(plan["seeds"]):
            raise ValueError("Supply one CPU per seed")
        with ThreadPoolExecutor(max_workers=len(plan["seeds"])) as pool:
            for future in as_completed([pool.submit(work,pair) for pair in enumerate(plan["seeds"])]):
                rows.extend(future.result())
                report(root,rows,plan)
        decision=decide(rows,level,plan)
        save(root/"decisions"/f"level_{level}.json",dict(at=now(),**decision))
        print("LEVEL",level,decision,flush=True)
        if decision["stop"]:
            break
    report(root,rows,plan)
    attempted_levels=sorted({r["level"] for r in rows})
    save(root/"completion.json",dict(at=now(),attempts=len(rows),attempted_levels=attempted_levels,
        skipped_levels=[n for n in LEVELS if n not in attempted_levels],last_decision=decision,
        all_required_attempts=len(rows)==len(attempted_levels)*len(SEEDS)*len(ARMS),
        all_verified=all(r.get("verified",False) for r in rows)))


def worker(args):
    root=Path(args.output).resolve()
    plan=load(root/"protocol.json")
    if sys.platform.startswith("linux"):
        import resource
        limit=plan["address_space_gib"]*1024**3
        resource.setrlimit(resource.RLIMIT_AS,(limit,limit))
    helper=prior.base.dependencies(root/"source")
    helper.worker(args)


def verify(args):
    root=Path(args.output).resolve()
    helper=prior.base.dependencies(root/"source")
    plan=helper.check_sources(root)
    rows=load(root/"all_results.json")
    completion=load(root/"completion.json")
    manifest=load(root/"datasets_registered.json")
    for name,digest in manifest["hashes"].items():
        data=load(root/name)
        if sha(root/name)!=digest or data!=generate(data["case"],data["seed"]):
            raise ValueError("Dataset integrity failed")
    levels=completion["attempted_levels"]
    if levels!=list(range(5,max(levels)+1)):
        raise ValueError("Skipped intermediate level")
    expected={(level,seed,arm) for level in levels for seed in SEEDS for arm in ARMS}
    if len(rows)!=len(expected) or {(r["level"],r["seed"],r["arm"]) for r in rows}!=expected:
        raise ValueError("Missing or duplicate attempts")
    for level in levels:
        d=decide(rows,level,plan)
        saved=load(root/"decisions"/f"level_{level}.json")
        if any(saved[k]!=v for k,v in d.items()) or (level<max(levels) and d["stop"]):
            raise ValueError("Stopping rule mismatch")
    if not decide(rows,max(levels),plan)["stop"]:
        raise ValueError("Stopped early")
    for r in rows:
        job=root/"jobs"/r["setting"]/f"s{r['seed']}"/r["arm"]
        req,proc=load(job/"request.json"),load(job/"process.json")
        data=load(root/req["data"])
        if sha(root/req["data"])!=req["data_sha256"] or req["data_sha256"]!=r["dataset_sha256"]:
            raise ValueError("Data routing mismatch")
        if req["config"]!=helper.arm_config(config(data,plan,helper),r["arm"]) or not (job/"process.log").exists():
            raise ValueError("Parameter or log mismatch")
        if "patterns" in r:
            calc=recovery(r,data,helper)
            if any(r[k]!=v for k,v in calc.items()):
                raise ValueError("Recovery mismatch")
            yt=np.asarray(data["y"],float)[data["test_indices"]]
            pred=np.asarray(r["prediction"])
            train=np.asarray(data["y"],float)[data["train_indices"]]
            dev=lambda mu:float(2*np.sum(helper.xlogy(yt,yt/mu)-yt+mu))
            r2=1-float(np.sum((yt-pred)**2)/np.sum((yt-yt.mean())**2))
            pr2=1-dev(pred)/dev(np.full(len(yt),train.mean()))
            if not np.allclose([r2,pr2],[r["test_r2"],r["test_pr2"]],rtol=1e-10,atol=1e-10):
                raise ValueError("Predictive metric mismatch")
        if r.get("verified") and not (proc["returncode"]==0 and r["status"]=="complete" and r["independent"]["status"]=="passed" and r["independent"]["gap_per_sample"]<=req["config"]["gap_tol"] and r["independent"]["kkt"]<=req["config"]["solver_tol"]):
            raise ValueError("Unjustified verification label")
    report(root,rows,plan)
    record=dict(at=now(),integrity_checks_passed=True,datasets=len(manifest["hashes"]),attempts=len(rows),
        verified=sum(bool(r.get("verified")) for r in rows),stop_rule_reproduced=True)
    save(root/"verification.json",record)
    print(record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","run","worker","verify"):
        p=sub.add_parser(command)
        p.add_argument("--output",required=True)
        if command=="prepare":
            p.add_argument("--parent",required=True)
        if command=="run":
            p.add_argument("--cpus",type=int,nargs="+",required=True)
        if command=="worker":
            p.add_argument("--request",required=True)
    args=parser.parse_args()
    globals()[args.command](args)
