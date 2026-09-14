from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


SIZE_ORDER = ["n500", "n1500", "n3000"]
WEIGHT_ORDER = ["small", "medium", "large"]
MODEL_ORDER = [
    "Ridge",
    "Lasso",
    "ElasticNet",
    "PoissonGLM",
    "RBF-SVR",
    "RandomForest",
    "MLP",
    "DIPS-PR",
]
HIV_ORDER = {
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["DOR", "EFV", "ETR", "NVP", "RPV"],
    "PI": ["ATV", "DRV", "FPV", "IDV", "LPV", "NFV", "SQV", "TPV"],
    "INI": ["BIC", "CAB", "DTG", "EVG", "RAL"],
    "CAI": ["LEN"],
}
HIV_MODEL_ORDER = [
    "DIPS-PR",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "PoissonGLM",
    "RBF-SVR",
    "RandomForest",
    "MLP",
]
DISPLAY_MODEL = {
    "DIPS-PR": r"\method{}",
    "ElasticNet": "Elastic Net",
    "PoissonGLM": "Poisson GLM",
    "RBF-SVR": "RBF-SVR",
    "RandomForest": "RF",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))


def mean_std(values: list[float]) -> tuple[float, float]:
    return sum(values) / len(values), sample_std(values)


def fmt_mean_std(mean: float, std: float, bold: bool = False) -> str:
    value = f"{mean:.3f} $\\pm$ {std:.3f}"
    return rf"\textbf{{{value}}}" if bold else value


def fmt_point(value: float, bold: bool = False) -> str:
    formatted = f"{value:.4f}"
    return rf"\textbf{{{formatted}}}" if bold else formatted


def synthetic_rows(results: Path, output: Path) -> None:
    rows = read_csv(results / "synthetic_mean_std.csv")
    lookup = {(r["size"], r["weight"], r["model"]): r for r in rows}
    for metric in ("r2", "pr2"):
        winners = {}
        for size in SIZE_ORDER:
            for weight in WEIGHT_ORDER:
                winners[(size, weight)] = max(
                    MODEL_ORDER,
                    key=lambda model: float(lookup[(size, weight, model)][f"{metric}_mean"]),
                )
        lines = []
        for model in MODEL_ORDER:
            values = []
            for size in SIZE_ORDER:
                for weight in WEIGHT_ORDER:
                    row = lookup[(size, weight, model)]
                    values.append(
                        fmt_mean_std(
                            float(row[f"{metric}_mean"]),
                            float(row[f"{metric}_std"]),
                            winners[(size, weight)] == model,
                        )
                    )
            label = DISPLAY_MODEL.get(model, model)
            lines.append(label + " & " + " & ".join(values) + r" \\")
        (output / f"synthetic_{metric}_rows.tex").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )


def recovery_rows(results: Path, output: Path) -> None:
    rows = read_csv(results / "synthetic_recovery_all_runs.csv")
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["size"], row["weight"])].append(row)
    lines = []
    summary_rows = []
    for size in SIZE_ORDER:
        for weight in WEIGHT_ORDER:
            group = grouped[(size, weight)]
            stats = {}
            for field in ("n_selected", "precision", "recall", "f1", "recall_at_50"):
                stats[field] = mean_std([float(row[field]) for row in group])
            size_value = size.removeprefix("n")
            label = rf"$m={size_value}$, {weight} weight"
            lines.append(
                label
                + " & "
                + " & ".join(
                    [
                        f"{stats['n_selected'][0]:.1f} $\\pm$ {stats['n_selected'][1]:.1f}",
                        fmt_mean_std(*stats["precision"]),
                        fmt_mean_std(*stats["recall"]),
                        fmt_mean_std(*stats["f1"]),
                        fmt_mean_std(*stats["recall_at_50"]),
                    ]
                )
                + r" \\"
            )
            summary_rows.append(
                {
                    "size": size,
                    "weight": weight,
                    **{
                        f"{field}_{part}": value
                        for field, values in stats.items()
                        for part, value in zip(("mean", "std"), values)
                    },
                }
            )
    (output / "synthetic_recovery_rows.tex").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )
    write_csv(output / "synthetic_recovery_summary.csv", summary_rows)


def hiv_rows(results: Path, output: Path) -> None:
    rows = read_csv(results / "hiv_nonlinear_all_endpoints.csv")
    lookup = {
        (row["dataset"], row["drug"], row["model"]): row
        for row in rows
    }
    for metric, suffix in (("R2", "r2"), ("PseudoR2", "pr2")):
        lines = []
        for dataset, drugs in HIV_ORDER.items():
            for drug in drugs:
                values = {
                    model: float(lookup[(dataset, drug, model)][metric])
                    for model in HIV_MODEL_ORDER
                }
                winner = max(values, key=values.get)
                n_test = int(lookup[(dataset, drug, "DIPS-PR")]["n_test"])
                formatted = [
                    fmt_point(values[model], model == winner)
                    for model in HIV_MODEL_ORDER
                ]
                lines.append(
                    f"{dataset} & {drug} & {n_test} & "
                    + " & ".join(formatted)
                    + r" \\"
                )
            if dataset != "CAI":
                lines.append(r"\midrule")
        (output / f"hiv_{suffix}_rows.tex").write_text(
            "\n".join(lines) + "\n", encoding="ascii"
        )

    paired = read_csv(results / "hiv_paired_bootstrap_vs_dips.csv")
    paired_lookup = {
        (row["dataset"], row["drug"], row["baseline"]): row
        for row in paired
    }
    highlight_lines = []
    highlight_summary = []
    for dataset, drug in (("NRTI", "DDI"), ("NRTI", "TDF"), ("PI", "TPV")):
        dips = lookup[(dataset, drug, "DIPS-PR")]
        baselines = [model for model in HIV_MODEL_ORDER if model != "DIPS-PR"]
        best_r2 = max(baselines, key=lambda model: float(lookup[(dataset, drug, model)]["R2"]))
        best_pr2 = max(
            baselines,
            key=lambda model: float(lookup[(dataset, drug, model)]["PseudoR2"]),
        )
        pair_r2 = paired_lookup[(dataset, drug, best_r2)]
        pair_pr2 = paired_lookup[(dataset, drug, best_pr2)]
        n_test = int(dips["n_test"])
        train = int(dips["n_train"])
        validation = int(dips["n_validation"])
        r2_ci = (
            float(pair_r2["R2_difference_ci_lower"]),
            float(pair_r2["R2_difference_ci_upper"]),
        )
        pr2_ci = (
            float(pair_pr2["PseudoR2_difference_ci_lower"]),
            float(pair_pr2["PseudoR2_difference_ci_upper"]),
        )
        highlight_lines.append(
            f"HIVDB-{dataset} & {drug} & {train} / {validation} / {n_test} & "
            f"{float(dips['R2']):.4f} & "
            f"{DISPLAY_MODEL.get(best_r2, best_r2)} ({float(lookup[(dataset, drug, best_r2)]['R2']):.4f}) & "
            f"[{r2_ci[0]:.4f}, {r2_ci[1]:.4f}] & "
            f"{float(dips['PseudoR2']):.4f} & "
            f"{DISPLAY_MODEL.get(best_pr2, best_pr2)} ({float(lookup[(dataset, drug, best_pr2)]['PseudoR2']):.4f}) & "
            f"[{pr2_ci[0]:.4f}, {pr2_ci[1]:.4f}]"
            + r" \\"
        )
        highlight_summary.append(
            {
                "dataset": dataset,
                "drug": drug,
                "n_test": n_test,
                "dips_r2": float(dips["R2"]),
                "best_r2_baseline": best_r2,
                "best_r2_baseline_value": float(lookup[(dataset, drug, best_r2)]["R2"]),
                "r2_difference_ci_lower": r2_ci[0],
                "r2_difference_ci_upper": r2_ci[1],
                "dips_pr2": float(dips["PseudoR2"]),
                "best_pr2_baseline": best_pr2,
                "best_pr2_baseline_value": float(
                    lookup[(dataset, drug, best_pr2)]["PseudoR2"]
                ),
                "pr2_difference_ci_lower": pr2_ci[0],
                "pr2_difference_ci_upper": pr2_ci[1],
            }
        )
    (output / "hiv_highlight_rows.tex").write_text(
        "\n".join(highlight_lines) + "\n", encoding="ascii"
    )
    write_csv(output / "hiv_highlight_summary.csv", highlight_summary)


def agreement_summary(results: Path, output: Path) -> None:
    files = sorted((results / "agreement_multiseed").glob("seed_*.json"))
    settings = []
    seeds = []
    for path in files:
        payload = json.loads(path.read_text())
        seeds.append(int(payload["seed"]))
        settings.extend(payload["settings"])
    by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
    for setting in settings:
        by_config[str(setting["config"])].append(setting)
    rows = []
    for config in ("D1", "D2", "D3", "D4", "D5"):
        group = by_config[config]
        rows.append(
            {
                "config": config,
                "seeds": len(group),
                "same_selected_set": sum(
                    bool(item["comparison"]["same_selected_set"]) for item in group
                ),
                "same_exact_recall": sum(
                    bool(item["comparison"]["same_exact_recall"]) for item in group
                ),
                "same_superset_recall": sum(
                    bool(item["comparison"]["same_superset_recall"]) for item in group
                ),
                "certified_only_true_related_runs": sum(
                    bool(item["comparison"]["certified_only_true_related"])
                    for item in group
                ),
                "selected_jaccard_mean": sum(
                    float(item["comparison"]["selected_jaccard"]) for item in group
                )
                / len(group),
                "selected_jaccard_min": min(
                    float(item["comparison"]["selected_jaccard"]) for item in group
                ),
                "heuristic_nodes_mean": sum(
                    int(item["heuristic"]["nodes_visited"]) for item in group
                )
                / len(group),
                "certified_nodes_mean": sum(
                    int(item["certified"]["nodes_visited"]) for item in group
                )
                / len(group),
                "heuristic_seconds_mean": sum(
                    float(item["heuristic"]["elapsed_seconds"]) for item in group
                )
                / len(group),
                "certified_seconds_mean": sum(
                    float(item["certified"]["elapsed_seconds"]) for item in group
                )
                / len(group),
            }
        )
    write_csv(output / "agreement_summary.csv", rows)
    overall = {
        "seed_count": len(set(seeds)),
        "seeds": sorted(set(seeds)),
        "comparison_count": len(settings),
        "same_selected_set": sum(
            bool(item["comparison"]["same_selected_set"]) for item in settings
        ),
        "same_exact_recall": sum(
            bool(item["comparison"]["same_exact_recall"]) for item in settings
        ),
        "same_superset_recall": sum(
            bool(item["comparison"]["same_superset_recall"]) for item in settings
        ),
        "certified_only_true_related_runs": sum(
            bool(item["comparison"]["certified_only_true_related"])
            for item in settings
        ),
        "selected_jaccard_mean": sum(
            float(item["comparison"]["selected_jaccard"]) for item in settings
        )
        / len(settings),
        "selected_jaccard_min": min(
            float(item["comparison"]["selected_jaccard"]) for item in settings
        ),
        "maximum_common_coefficient_difference": max(
            float(item["comparison"]["maximum_common_coefficient_difference"])
            for item in settings
        ),
    }
    (output / "agreement_overall.json").write_text(
        json.dumps(overall, indent=2) + "\n", encoding="ascii"
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("revision_tables"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    synthetic_rows(args.results, args.output)
    recovery_rows(args.results, args.output)
    hiv_rows(args.results, args.output)
    agreement_summary(args.results, args.output)
    print(f"wrote revision summaries to {args.output}")


if __name__ == "__main__":
    main()
