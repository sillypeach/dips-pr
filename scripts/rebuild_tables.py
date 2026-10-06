"""Recompute manuscript summary cells from archived seed-level evidence.

This is an arithmetic reproduction of the reference results, not model fitting.
Fresh-run summaries are produced by the experiment-specific aggregate commands.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT = ROOT / "paper_snapshot"
CONDITIONS = ("C2", "C4", "C8", "L2-5")
PREDICTION = ("DIPS-PATH", "LCM-Lasso", "Ridge", "Lasso", "ElasticNet",
              "PoissonGLM", "RBF-SVR", "RandomForest", "MLP")
RECOVERY = ("DIPS-ORACLE-F1", "LCM-Lasso", "RF-paths")


def load(name):
    return json.loads((SNAPSHOT / name).read_text())


def five(values, reference, sd_key="sample_sd"):
    if len(values) != 5 or not all(math.isfinite(x) for x in values):
        raise ValueError("A manuscript cell requires all five finite seed values")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    if reference.get("n", 5) != 5:
        raise ValueError("Reference does not contain five seeds")
    for observed, expected in ((mean, reference["mean"]), (sd, reference[sd_key])):
        if not math.isclose(observed, expected, rel_tol=2e-12, abs_tol=2e-12):
            raise ValueError(f"Summary mismatch: {observed} != {expected}")
    return mean, sd


def cell(table, condition, method, metric, values, reference, sd_key="sample_sd"):
    mean, sd = five(values, reference, sd_key)
    return dict(table=table, condition=condition, method=method, metric=metric,
                n=5, mean=mean, sample_sd=sd)


def condition_name(family, level):
    return {("count", 2): "C2", ("count", 4): "C4", ("count", 8): "C8",
            ("count", 12): "C12", ("length", 5): "L2-5",
            ("length", 6): "L3-6", ("shared", 30): "S30"}[(family, level)]


def rebuild():
    rows = []
    synthetic = load("paper_table_values_synthetic.json")
    for table, methods, metrics in (("synthetic_prediction", PREDICTION, ("R2", "PseudoR2")),
                                    ("synthetic_recovery", RECOVERY, ("precision", "recall", "f1", "NZ"))):
        for condition in CONDITIONS:
            for method in methods:
                for metric in metrics:
                    found = [x for x in synthetic if (x["table"], x["condition"], x.get("method"), x["metric"])
                             == (table, condition, method, metric)]
                    if len(found) != 1:
                        raise ValueError("Missing or duplicate synthetic reference cell")
                    x = found[0]
                    rows.append(cell(table, condition, method, metric, x["values"], x))

    pruning = load("pruning_complete_conditions.json")
    if len(pruning["conditions"]) != 7:
        raise ValueError("The manuscript includes exactly seven pruning conditions")
    for c in pruning["conditions"]:
        condition = condition_name(c["family"], c["level"])
        if len(c["seeds"]) != 5 or len({s["seed"] for s in c["seeds"]}) != 5:
            raise ValueError("Incomplete or duplicate pruning seeds")
        rates = []
        for seed in c["seeds"]:
            if seed["M"] <= 0 or seed["traversals"] <= 0:
                raise ValueError("A complete nontrivial path must have a positive denominator")
            rate = 1 - seed["remaining"] / (seed["M"] * seed["traversals"])
            if not 0 <= rate <= 1 or not math.isclose(rate, seed["rate"], abs_tol=1e-12):
                raise ValueError("Invalid full-dictionary pruning ratio")
            rates.append(rate)
        rows.append(cell("pruning", condition, "DIPS-PATH", "exclusion_rate", rates, c["whole_path_rate"]))
        rows.append(cell("pruning", condition, "DIPS-PATH", "dictionary_size",
                         [s["M"] for s in c["seeds"]], c["dictionary_size"]))

    ablation = load("component_ablation_summary.json")["component"]["conditions"]
    if len(ablation) != 3:
        raise ValueError("The manuscript contains three ablation conditions")
    for c in ablation:
        condition = condition_name(c["family"], c["level"])
        for arm in ("v_only", "u_only", "vu_two"):
            a = next(a for a in c["arm_statistics"] if a["arm"] == arm)
            for metric in ("full_dictionary_exclusion_rate", "nodes", "fit_seconds"):
                stat = a["metrics"][metric]
                rows.append(cell("component_ablation", condition, arm, metric, stat["values"], stat))

    hiv = load("hiv_repeated_split_summary.json")
    if len(hiv["primary"]) != 200:
        raise ValueError("Expected 25 endpoints times eight methods")
    refs = {(r["endpoint"], r["method"], r["metric"]): r for r in load("paper_table_values_hiv.json")}
    seen = set()
    for e in hiv["primary"]:
        if e["status"] != "complete" or sorted(s["seed"] for s in e["seeds"]) != list(range(42, 47)):
            raise ValueError("Incomplete HIV repeats")
        if (e["endpoint"], e["method"]) in seen:
            raise ValueError("Duplicate HIV endpoint/method")
        seen.add((e["endpoint"], e["method"]))
        for metric in ("R2", "PseudoR2_train_null"):
            values = [s["metrics"][metric] for s in e["seeds"]]
            five(values, refs[(e["endpoint"], e["method"], metric)], "sd")
            table = "hiv_nrti" if e["endpoint"].startswith("NRTI/") else "hiv_appendix"
            rows.append(cell(table, e["endpoint"], e["method"], metric, values, e["metrics"][metric], "sd"))
    return rows


def tex_escape(value):
    return value.replace("_", r"\_").replace("%", r"\%")


def export(output):
    rows = rebuild()
    output.mkdir(parents=True, exist_ok=True)
    tables = sorted({r["table"] for r in rows})
    for table in tables:
        selected = [r for r in rows if r["table"] == table]
        with (output / (table + ".csv")).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(selected[0]))
            writer.writeheader()
            writer.writerows(selected)
        lines = ["% Recomputed archived values, not fresh training results.",
                 r"\begin{tabular}{lllr}", r"\hline",
                 r"Condition & Method & Metric & Mean $\pm$ sample SD \\", r"\hline"]
        for row in selected:
            lines.append(" & ".join(tex_escape(str(row[k])) for k in ("condition", "method", "metric"))
                         + f" & ${row['mean']:.6g} \\pm {row['sample_sd']:.6g}$" + r" \\")
        lines += [r"\hline", r"\end{tabular}"]
        (output / (table + ".tex")).write_text("\n".join(lines) + "\n")
    report = dict(status="verified", summary_cells=len(rows), tables=tables,
                  interpretation="Archived seed-level arithmetic reconstruction; not retraining.")
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/paper_tables")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        print(json.dumps({"status": "verified", "summary_cells": len(rebuild())}))
    else:
        export(args.output)


if __name__ == "__main__":
    main()
