from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def read_json(path: str | Path):
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def aggregate_synthetic(root: Path) -> None:
    rows = []
    recovery_rows = []
    for path in glob.glob(str(root / "synthetic" / "seed_*" / "*.json")):
        if path.endswith("manifest.json") or path.endswith("error.json"):
            continue
        result = read_json(path)
        for model in result["models"]:
            row = {
                "seed": result["seed"],
                "size": result["size"],
                "weight": result["weight"],
                "n_total": result["n_total"],
                "model": model["model"],
                "elapsed_seconds": model.get("elapsed_seconds"),
            }
            row.update(model["metrics"])
            rows.append(row)
            if model["model"] == "DIPS-PR":
                recovery = model["recovery"]
                recovery_rows.append(
                    {
                        "seed": result["seed"],
                        "size": result["size"],
                        "weight": result["weight"],
                        "precision": recovery.get("exact_precision"),
                        "recall": recovery.get("exact_recall"),
                        "f1": recovery.get("exact_f1"),
                        "n_selected": recovery.get("num_selected_nz"),
                        "n_hits": recovery.get("num_hits_exact"),
                        "precision_at_8": recovery.get("precision_at_8"),
                        "recall_at_8": recovery.get("recall_at_8"),
                        "recall_at_20": recovery.get("recall_at_20"),
                        "recall_at_50": recovery.get("recall_at_50"),
                        "mean_true_pattern_rank": recovery.get(
                            "mean_true_pattern_rank"
                        ),
                    }
                )
    if not rows:
        return
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "synthetic_all_runs.csv", index=False)
    summary = (
        frame.groupby(["size", "weight", "model"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            r2_mean=("R2", "mean"),
            r2_std=("R2", "std"),
            pr2_mean=("PseudoR2", "mean"),
            pr2_std=("PseudoR2", "std"),
            time_mean=("elapsed_seconds", "mean"),
            time_std=("elapsed_seconds", "std"),
        )
        .sort_values(["size", "weight", "model"])
    )
    summary.to_csv(root / "synthetic_mean_std.csv", index=False)
    recovery_frame = pd.DataFrame(recovery_rows)
    recovery_frame.to_csv(root / "synthetic_recovery_all_runs.csv", index=False)
    recovery_summary = (
        recovery_frame.groupby(["size", "weight"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            n_selected_mean=("n_selected", "mean"),
            n_selected_std=("n_selected", "std"),
            precision_at_8_mean=("precision_at_8", "mean"),
            recall_at_20_mean=("recall_at_20", "mean"),
            recall_at_50_mean=("recall_at_50", "mean"),
            mean_true_pattern_rank=("mean_true_pattern_rank", "mean"),
        )
        .sort_values(["size", "weight"])
    )
    recovery_summary.to_csv(root / "synthetic_recovery_mean_std.csv", index=False)


def aggregate_hiv(root: Path) -> None:
    rows = []
    for path in glob.glob(str(root / "hiv" / "*" / "*.json")):
        result = read_json(path)
        for model in result["models"]:
            row = {
                "dataset": result["dataset"],
                "drug": result["drug"],
                "n_train": result["n_train"],
                "n_validation": result["n_validation"],
                "n_test": result["n_test"],
                "model": model["model"],
                "cv_rmse": model["cv_rmse"],
                "best_params": json.dumps(model["best_params"], sort_keys=True),
                "elapsed_seconds": model["elapsed_seconds"],
            }
            row.update(model["metrics"])
            intervals = result.get("confidence_intervals", {}).get(
                model["model"], {}
            )
            for metric in ("R2", "PseudoR2", "RMSE"):
                row[f"{metric}_ci_lower"] = intervals.get(metric, {}).get("lower")
                row[f"{metric}_ci_upper"] = intervals.get(metric, {}).get("upper")
            rows.append(row)
    if not rows:
        return
    frame = pd.DataFrame(rows).sort_values(["dataset", "drug", "model"])
    frame.to_csv(root / "hiv_nonlinear_all_endpoints.csv", index=False)
    winner_rows = []
    for (dataset, drug), group in frame.groupby(["dataset", "drug"]):
        for metric in ("R2", "PseudoR2"):
            winner = group.loc[group[metric].idxmax()]
            winner_rows.append(
                {
                    "dataset": dataset,
                    "drug": drug,
                    "metric": metric,
                    "winner": winner["model"],
                    "value": winner[metric],
                }
            )
    pd.DataFrame(winner_rows).to_csv(
        root / "hiv_nonlinear_winners_by_class.csv",
        index=False,
    )
    paired_rows = []
    for path in glob.glob(str(root / "hiv" / "*" / "*.json")):
        result = read_json(path)
        for baseline, metrics in result.get(
            "paired_differences_vs_dips", {}
        ).items():
            row = {
                "dataset": result["dataset"],
                "drug": result["drug"],
                "baseline": baseline,
            }
            for metric, values in metrics.items():
                row[f"{metric}_difference_ci_lower"] = values["lower"]
                row[f"{metric}_difference_ci_upper"] = values["upper"]
                row[f"{metric}_dips_better_fraction"] = values[
                    "positive_fraction"
                ]
            paired_rows.append(row)
    pd.DataFrame(paired_rows).to_csv(
        root / "hiv_paired_bootstrap_vs_dips.csv",
        index=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    aggregate_synthetic(args.results_dir)
    aggregate_hiv(args.results_dir)
    print(f"Aggregated results under {args.results_dir}")


if __name__ == "__main__":
    main()
