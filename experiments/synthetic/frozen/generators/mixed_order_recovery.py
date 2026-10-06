"""Disclosed mixed-order development study and fresh-seed confirmation."""
import os
for _key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import queue
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np
import pattern_count_extension as base

save, load, sha, now = base.save, base.load, base.sha, base.now
DEV_SEEDS = [2609272101, 2609272102]
CONFIRM_SEEDS = list(range(2609272201, 2609272206))
ARMS = ["Full", "KKT-priced"]
REGIMES = ["fixed4", "mixed_natural", "mixed_equal_support"]


def case_for(count, regime):
    if count not in (4,8) or regime not in REGIMES:
        raise ValueError("Unregistered count or regime")
    return dict(base.case_for(count), id=f"{regime}_k{count}", regime=regime)


def generate(case, seed):
    n,p,k = case["n"],case["p"],case["truths"]
    if k not in (4,8) or n<20 or case["regime"] not in REGIMES:
        raise ValueError("Unsupported generator setting")
    sr,nr,yr,pr,lr = [np.random.default_rng(s) for s in
                      np.random.SeedSequence([seed,202609244]).spawn(5)]
    lengths = np.full(k,4,dtype=int) if case["regime"]=="fixed4" else np.tile([2,3,4,5],k//4)
    if case["regime"]!="fixed4":
        lr.shuffle(lengths)
    if not lengths.sum()<=p<=96:
        raise ValueError("Insufficient item columns")
    probs = (case["signal"]**(4/lengths) if case["regime"]=="mixed_equal_support"
             else np.full(k,case["signal"]))
    signals = sr.random((n,int(lengths.sum()))) < np.repeat(probs,lengths)
    edges = np.r_[0,np.cumsum(lengths)]
    truth_x = np.column_stack([signals[:,edges[j]:edges[j+1]].all(axis=1) for j in range(k)])
    noise = nr.random((n,96))[:,:p-int(lengths.sum())] < case["noise"]
    weights = np.asarray(base.WEIGHTS[:k])*case["weight_scale"]
    mu = np.exp(np.log(case["baseline"])+truth_x@weights)
    y = yr.poisson(mu)
    permutation = pr.permutation(p)
    X = np.column_stack((signals,noise))
    tx = [sorted(permutation[np.flatnonzero(row)].tolist()) for row in X]
    truth = [sorted(permutation[edges[j]:edges[j+1]].tolist()) for j in range(k)]
    cut = int(.7*n)
    return dict(transactions=tx,y=y.tolist(),true_mean=mu.tolist(),true_patterns=truth,
        true_weights=weights.tolist(),train_indices=list(range(cut)),test_indices=list(range(cut,n)),
        case=case,seed=seed,permutation=permutation.tolist(),true_orders=lengths.tolist(),
        signal_probabilities=probs.tolist())


def config(data, ratio, helper):
    return dict(helper.config(data,ratio),max_len=5)


def recovery(row, data, helper):
    selected = helper.active_set(row["patterns"],row["w"])
    truth = {tuple(p) for p in data["true_patterns"]}
    tp,nz = len(selected & truth),len(selected)
    out = dict(TP=tp,FP=nz-tp,FN=len(truth)-tp,NZ=nz,precision=tp/max(1,nz),
               recall=tp/len(truth),f1=2*tp/max(1,nz+len(truth)))
    orders={}
    for order in range(1,6):
        t={p for p in truth if len(p)==order}
        s={p for p in selected if len(p)==order}
        hit=len(s&t)
        orders[str(order)]=dict(truth=len(t),NZ=len(s),TP=hit,FP=len(s)-hit,FN=len(t)-hit,
            precision=hit/len(s) if s else None,recall=hit/len(t) if t else None,
            f1=2*hit/(len(s)+len(t)) if s or t else None)
    out["by_order"]=orders
    return out


def prepare(args):
    root,parent=Path(args.output).resolve(),Path(args.parent).resolve()
    if root.exists():
        raise ValueError("Use a fresh output directory")
    old=load(parent/"protocol.json")
    if sha(parent/"protocol.json")!=load(parent/"registered.json")["protocol_sha256"]:
        raise ValueError("Parent protocol changed")
    source=root/"source"
    source.mkdir(parents=True)
    for name in base.SOURCES+["pattern_count_extension.py"]:
        if sha(parent/"source"/name)!=old["source_sha256"]["source/"+name]:
            raise ValueError("Parent source changed: "+name)
        shutil.copyfile(parent/"source"/name,source/name)
    shutil.copyfile(__file__,source/Path(__file__).name)
    (root/"audit").mkdir()
    if sha(parent/"audit/model.py")!=old["source_sha256"]["audit/model.py"]:
        raise ValueError("Parent audit source changed")
    shutil.copyfile(parent/"audit/model.py",root/"audit/model.py")
    plan=dict(version="mixed-order-development-confirmation-v1",at=now(),
        parent_protocol_sha256=sha(parent/"protocol.json"),
        cases=[case_for(k,r) for k in (4,8) for r in REGIMES],arms=ARMS,
        development_seeds=DEV_SEEDS,confirmation_seeds=CONFIRM_SEEDS,
        ratio=.03,policy=old["policy"],timeout_seconds=900,
        planned_development_fits=24,planned_confirmation_fits_if_selected=20,
        targets=dict(precision=.8,recall=.8,f1=.8,r2=.7,each_order_recall=.8),
        selection_rule=[
            "Candidate settings are the four mixed-order regimes, not the fixed-four control.",
            "Require both development seeds and both arms independently verified, with identical active sets and predictions agreeing at rtol=atol=1e-5.",
            "Eligible if development means satisfy precision/recall/F1>=0.8, validation R2>=0.7, and recall>=0.8 at EACH of orders 2,3,4,5.",
            "Among eligible settings prefer eight truths, then greater mean F1, precision, validation R2, then lexicographic setting ID.",
            "If none eligible, choose the valid mixed setting maximizing minimum(mean precision,mean recall), then F1, then ID. Clearly label fallback, not success. If none valid, skip confirmation.",
            "After saving selection, generate all five new confirmation seeds for the selected regime and its same-count fixed-four control; run both arms. No further selection or tuning."],
        source_sha256={str(p.relative_to(root)):sha(p) for folder in (source,root/"audit") for p in folder.glob("*.py")},
        notes=[
            "This is a disclosed favorable-regime search. Development truth and validation metrics ARE used for selection. Development holdouts are not independent test results.",
            "Confirmation seeds are fixed here but generated only after the selection file is registered. Report every new seed including failures; no selection among confirmation seeds.",
            "Same 2000 samples, 64 observed items, 70/30 split, baseline mean 30, Poisson responses and coefficient prefix as the previous experiment. Truths are disjoint AND patterns, with either all lengths 4 or balanced lengths 2/3/4/5.",
            "Length-to-coefficient assignment is shuffled by an independent seed stream, not selected to improve recovery. Coefficients remain [0.9,-0.8,0.75,1.1,-0.85,0.95,0.8,-1.0] (prefix for four truths).",
            "Natural mixed: every signal item has marginal probability 0.65. Equal-support mixed: a length-L signal item has marginal probability 0.65**(4/L), so every embedded conjunction has population probability 0.65**4. This is a deliberately constructed regime, not empirical evidence of universal performance.",
            "Nuisance items have probability 0.20. Fixed p means the nuisance-item count changes with total true-pattern lengths. Signal dependence/exposure and mean response distributions change across regimes; do not claim a one-factor causal experiment.",
            "All regimes, including fixed-four controls, SEARCH lengths 1--5 with max_len=5, min_support=21, lambda=0.03*sum(training y), kappa=0.05, top_k_add=200, max_rounds=100 and unchanged tolerances. No new lambda tuning.",
            "The optimizer never receives a constraint on truth count or length proportions. Exact itemset equality defines TP; proper subsets and supersets remain FPs. Active means |coefficient|>1e-8. No second-stage screening or refitting.",
            "Full is frozen dynamic sphere u/v/previous-round-2ref, without independent KKT candidate rescue. KKT-priced is separately fitted. Both use the same data and objective. This is not a component ablation or a node-reduction study.",
            "Each fit has a 900-second process budget INCLUDING post-fit audit. The independent full-dictionary audit never repairs a fit. Keep timeouts and nonconvergence; no automatic retries.",
            "Up to six physical-core jobs may run concurrently, with one BLAS thread and serial rotated arm order within each dataset. Timings are diagnostic, not publication-quality speed comparisons.",
            "No manuscript edits. All source/data hashes, per-fit requests, logs, predictions and itemsets retained."])
    save(root/"protocol.json",plan)
    save(root/"registered.json",dict(at=now(),protocol_sha256=sha(root/"protocol.json")))
    print("PREPARED",root,"24 development fits + up to 20 confirmation fits",flush=True)


def compare(rows):
    refs={(r["stage"],r["setting"],r["seed"]):r for r in rows if r["arm"]=="KKT-priced" and r.get("verified")}
    for r in rows:
        b=refs.get((r["stage"],r["setting"],r["seed"]))
        if b is not None and r.get("verified"):
            active=lambda x:{tuple(p) for p,w in zip(x["patterns"],x["w"]) if abs(w)>1e-8}
            r["kkt_comparison"]=dict(same_active=active(r)==active(b),
                predictions_close=bool(np.allclose(r["prediction"],b["prediction"],rtol=1e-5,atol=1e-5)))


def choose(rows,plan):
    compare(rows)
    candidates=[]
    for case in plan["cases"]:
        if case["regime"]=="fixed4":
            continue
        rs=[r for r in rows if r["stage"]=="development" and r["setting"]==case["id"]]
        valid=len(rs)==4 and all(r.get("verified") and all(r.get("kkt_comparison",{}).values())
            and len(r.get("kkt_comparison",{}))==2 for r in rs)
        entry=dict(case=case,valid=bool(valid),eligible=False)
        if valid:
            full=[r for r in rs if r["arm"]=="Full"]
            for key in ("precision","recall","f1","test_r2"):
                entry[key]=statistics.mean(r[key] for r in full)
            entry["order_recall"]={str(o):statistics.mean(r["by_order"][str(o)]["recall"] for r in full) for o in range(2,6)}
            entry["eligible"]=all(entry[k]>=plan["targets"][k] for k in ("precision","recall","f1")) and entry["test_r2"]>=plan["targets"]["r2"] and min(entry["order_recall"].values())>=plan["targets"]["each_order_recall"]
        candidates.append(entry)
    eligible=[c for c in candidates if c["eligible"]]
    valid=[c for c in candidates if c["valid"]]
    if eligible:
        selected=sorted(eligible,key=lambda c:(-c["case"]["truths"],-c["f1"],-c["precision"],-c["test_r2"],c["case"]["id"]))[0]
    elif valid:
        selected=sorted(valid,key=lambda c:(-min(c["precision"],c["recall"]),-c["f1"],c["case"]["id"]))[0]
    else:
        selected=None
    return dict(at=now(),candidates=candidates,selected=selected,fallback=not bool(eligible),
                selection_uses_development_truth=True,confirmation_generated=False)


def report(root,rows,plan):
    compare(rows)
    groups=[]
    for stage in ("development","confirmation"):
        for case in plan["cases"]:
            for arm in plan["arms"]:
                rs=[r for r in rows if (r["stage"],r["setting"],r["arm"])==(stage,case["id"],arm)]
                if not rs:
                    continue
                ok=[r for r in rs if r.get("verified")]
                g=dict(stage=stage,setting=case["id"],truths=case["truths"],arm=arm,
                    attempted=len(rs),verified=len(ok),planned=len(plan[stage+"_seeds"]))
                for key in ("NZ","TP","FP","FN","precision","recall","f1","test_r2","test_pr2"):
                    if ok:
                        g[key]=dict(mean=statistics.mean(r[key] for r in ok),std=statistics.stdev(r[key] for r in ok) if len(ok)>1 else None)
                g["by_order"]={}
                for o in range(1,6):
                    g["by_order"][str(o)]={}
                    for key in ("truth","NZ","TP","FP","FN","precision","recall","f1"):
                        vals=[r["by_order"][str(o)][key] for r in ok if r["by_order"][str(o)][key] is not None]
                        g["by_order"][str(o)][key]=statistics.mean(vals) if vals else None
                groups.append(g)
    save(root/"all_results.json",rows)
    save(root/"summary.json",dict(at=now(),groups=groups,attempts=len(rows),verified=sum(bool(r.get("verified")) for r in rows)))
    save(root/"progress.json",dict(at=now(),attempts=len(rows),verified=sum(bool(r.get("verified")) for r in rows)))
    lines=["# Mixed-order recovery","",
        "Development and fresh-seed confirmation are separate. All algorithms search lengths 1--5; all recovery scores require exact equality.",
        "Means use verified fits only. Incomplete groups are not complete-study estimates. See protocol.json for the predeclared selection rule.","",
        "|Stage|Generator|Arm|Verified/planned|NZ|P|R|F1|R2|PR2|",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        values="|".join(f"{g[k]['mean']:.4f}" if k in g else "NA" for k in ("NZ","precision","recall","f1","test_r2","test_pr2"))
        lines.append(f"|{g['stage']}|{g['setting']}|{g['arm']}|{g['verified']}/{g['planned']}|{values}|")
    lines += ["","## Full-model per-order recovery","",
        "|Stage|Generator|Order|Mean truth|Mean selected|Mean TP|Mean FP|Mean FN|Recall|",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for g in groups:
        if g["arm"]!="Full":
            continue
        for order,values in g["by_order"].items():
            fields="|".join(f"{values[k]:.4f}" if values[k] is not None else "NA" for k in ("truth","NZ","TP","FP","FN","recall"))
            lines.append(f"|{g['stage']}|{g['setting']}|{order}|{fields}|")
    lines += ["","## Every attempt","",
        "|Stage|Generator|Seed|Arm|Process|Fit status|Verified|NZ|P|R|F1|R2|",
        "|---|---|---:|---|---|---|---|---:|---:|---:|---:|---:|"]
    for r in sorted(rows,key=lambda r:(r["stage"],r["setting"],r["seed"],r["arm"])):
        values="|".join(f"{r[k]:.4f}" if k in r else "NA" for k in ("NZ","precision","recall","f1","test_r2"))
        lines.append(f"|{r['stage']}|{r['setting']}|{r['seed']}|{r['arm']}|{r['process_status']}|{r['status']}|{r.get('verified',False)}|{values}|")
    lines += ["","## Scope",""]+["- "+n for n in plan["notes"]]
    (root/"RESULTS.md").write_text("\n".join(lines)+"\n")


def dispatch(root,plan,data,stage,arm,cpu,helper):
    case,seed=data["case"],data["seed"]
    datafile=root/"data"/stage/f"{case['id']}_s{seed}.json"
    job=root/"jobs"/stage/case["id"]/f"s{seed}"/arm
    req=dict(stage=stage,setting=case["id"],seed=seed,arm=arm,cpu=cpu,
        data=str(datafile.relative_to(root)),data_sha256=sha(datafile),
        config=helper.arm_config(config(data,plan["ratio"],helper),arm))
    save(job/"request.json",req)
    tick=time.perf_counter()
    print(now(),"START",stage,case["id"],seed,arm,flush=True)
    with (root/"controller.log").open("a") as log:
        log.write(f"{now()} START {stage} {case['id']} {seed} {arm}\n")
    try:
        with (job/"process.log").open("w") as log:
            proc=subprocess.run([sys.executable,str(root/"source"/Path(__file__).name),"worker",
                "--output",str(root),"--request",str(job/"request.json")],stdout=log,stderr=log,timeout=plan["timeout_seconds"])
        status,code="exited",proc.returncode
    except subprocess.TimeoutExpired:
        status,code="timeout",None
    save(job/"process.json",dict(at=now(),status=status,returncode=code,seconds=time.perf_counter()-tick))
    row=load(job/"result.json") if (job/"result.json").exists() else dict(status="failed")
    row.update({k:req[k] for k in ("stage","setting","seed","arm")})
    row.update(process_status=status,dataset_sha256=req["data_sha256"],truths=case["truths"])
    if code!=0:
        row["verified"]=False
    if "patterns" in row:
        row.update(recovery(row,data,helper))
    save(job/"evaluated.json",row)
    print(now(),"END",stage,case["id"],seed,arm,"verified",row.get("verified",False),flush=True)
    with (root/"controller.log").open("a") as log:
        log.write(f"{now()} END {stage} {case['id']} {seed} {arm} verified={row.get('verified',False)}\n")
    return row


def run_stage(root,plan,stage,cases,seeds,cpus,helper,rows):
    blocks=[]
    for i,seed in enumerate(seeds):
        for j,case in enumerate(cases):
            data=generate(case,seed)
            path=root/"data"/stage/f"{case['id']}_s{seed}.json"
            save(path,data)
            blocks.append((i+j,data))
    save(root/f"{stage}_datasets_registered.json",dict(at=now(),hashes={str(p.relative_to(root)):sha(p) for p in (root/"data"/stage).glob("*.json")}))
    free=queue.Queue()
    for cpu in cpus:
        free.put(cpu)
    def work(block):
        i,data=block
        cpu=free.get()
        try:
            arms=plan["arms"][i%len(ARMS):]+plan["arms"][:i%len(ARMS)]
            results=[dispatch(root,plan,data,stage,a,cpu,helper) for a in arms]
            save(root/"blocks"/stage/f"{data['case']['id']}_s{data['seed']}.json",results)
            return results
        finally:
            free.put(cpu)
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        for future in as_completed([pool.submit(work,b) for b in blocks]):
            rows.extend(future.result())
            report(root,rows,plan)


def run(args):
    root=Path(args.output).resolve()
    helper=base.dependencies(root/"source")
    plan=helper.check_sources(root)
    if (root/"started.json").exists():
        raise ValueError("Already started; no overwrite or automatic retry")
    if len(set(args.cpus))!=len(args.cpus) or (hasattr(os,"sched_getaffinity") and not set(args.cpus)<=os.sched_getaffinity(0)):
        raise ValueError("Invalid CPU allocation")
    save(root/"started.json",dict(at=now(),host=platform.node(),python=sys.version,executable=sys.executable,cpus=args.cpus))
    rows=[]
    run_stage(root,plan,"development",plan["cases"],plan["development_seeds"],args.cpus,helper,rows)
    selection=choose(rows,plan)
    save(root/"selection.json",selection)
    save(root/"selection_registered.json",dict(at=now(),sha256=sha(root/"selection.json")))
    if selection["selected"] is not None:
        selected=selection["selected"]["case"]
        cases=[selected,case_for(selected["truths"],"fixed4")]
        run_stage(root,plan,"confirmation",cases,plan["confirmation_seeds"],args.cpus,helper,rows)
    expected=plan["planned_development_fits"]+(plan["planned_confirmation_fits_if_selected"] if selection["selected"] is not None else 0)
    report(root,rows,plan)
    save(root/"completion.json",dict(at=now(),attempts=len(rows),expected=expected,
        all_attempted=len(rows)==expected,all_verified=all(r.get("verified",False) for r in rows)))


def worker(args):
    helper=base.dependencies(Path(args.output).resolve()/"source")
    helper.worker(args)


def verify(args):
    root=Path(args.output).resolve()
    helper=base.dependencies(root/"source")
    plan=helper.check_sources(root)
    rows=load(root/"all_results.json")
    selection=load(root/"selection.json")
    if sha(root/"selection.json")!=load(root/"selection_registered.json")["sha256"]:
        raise ValueError("Selection changed")
    recalculated=choose([r for r in rows if r["stage"]=="development"],plan)
    if any(selection[k]!=recalculated[k] for k in ("candidates","selected","fallback")):
        raise ValueError("Selection rule mismatch")
    expected={("development",c["id"],s,a) for c in plan["cases"] for s in plan["development_seeds"] for a in ARMS}
    stages=["development"]
    if selection["selected"] is not None:
        c=selection["selected"]["case"]
        cases=[c,case_for(c["truths"],"fixed4")]
        expected|={("confirmation",c["id"],s,a) for c in cases for s in plan["confirmation_seeds"] for a in ARMS}
        stages.append("confirmation")
    keys={(r["stage"],r["setting"],r["seed"],r["arm"]) for r in rows}
    if keys!=expected or len(rows)!=len(expected):
        raise ValueError("Missing or duplicate attempts")
    datasets=0
    for stage in stages:
        manifest=load(root/f"{stage}_datasets_registered.json")
        if stage=="confirmation" and manifest["at"]<=load(root/"selection_registered.json")["at"]:
            raise ValueError("Confirmation generated before selection registration")
        for name,digest in manifest["hashes"].items():
            data=load(root/name)
            if sha(root/name)!=digest or data!=generate(data["case"],data["seed"]):
                raise ValueError("Data integrity/reproduction failed")
            datasets+=1
    for r in rows:
        job=root/"jobs"/r["stage"]/r["setting"]/f"s{r['seed']}"/r["arm"]
        req,proc=load(job/"request.json"),load(job/"process.json")
        data=load(root/req["data"])
        if req["data_sha256"]!=r["dataset_sha256"] or sha(root/req["data"])!=req["data_sha256"]:
            raise ValueError("Data routing mismatch")
        if req["config"]!=helper.arm_config(config(data,plan["ratio"],helper),r["arm"]):
            raise ValueError("Parameter changes")
        if not (job/"process.log").exists():
            raise ValueError("Missing log")
        if "patterns" not in r:
            continue
        calc=recovery(r,data,helper)
        if any(r[k]!=v for k,v in calc.items()):
            raise ValueError("Recovery metric mismatch")
        yt=np.asarray(data["y"],float)[data["test_indices"]]
        pred=np.asarray(r["prediction"])
        train=np.asarray(data["y"],float)[data["train_indices"]]
        dev=lambda mu:float(2*np.sum(helper.xlogy(yt,yt/mu)-yt+mu))
        r2=1-float(np.sum((yt-pred)**2)/np.sum((yt-yt.mean())**2))
        pr2=1-dev(pred)/dev(np.full(len(yt),train.mean()))
        if not np.allclose([r2,pr2],[r["test_r2"],r["test_pr2"]],rtol=1e-10,atol=1e-10):
            raise ValueError("Predictive metric mismatch")
        if r.get("verified") and not (proc["returncode"]==0 and r["status"]=="complete" and r["independent"]["status"]=="passed" and r["independent"]["kkt"]<=req["config"]["solver_tol"] and r["independent"]["gap_per_sample"]<=req["config"]["gap_tol"]):
            raise ValueError("Unjustified convergence label")
    report(root,rows,plan)
    record=dict(at=now(),integrity_checks_passed=True,attempts=len(rows),datasets=datasets,
        verified=sum(bool(r.get("verified")) for r in rows),sources=len(plan["source_sha256"]),
        selection_rule_reproduced=True,confirmation_generated_after_selection=True)
    save(root/"verification.json",record)
    print(json.dumps(record,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="command",required=True)
    for command in ("prepare","run","worker","verify"):
        p=sub.add_parser(command)
        p.add_argument("--output",required=True)
        if command=="prepare":
            p.add_argument("--parent",required=True)
        if command=="run":
            p.add_argument("--cpus",type=int,nargs="+",default=[0])
        if command=="worker":
            p.add_argument("--request",required=True)
    args=parser.parse_args()
    globals()[args.command](args)
