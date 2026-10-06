from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


CATEGORIES = (
    "exact",
    "strict_subset",
    "strict_superset",
    "partial_overlap",
    "disjoint",
)


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate structural itemset audits.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-runs", type=int, default=45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("seed_*/n*.json"))
    if len(paths) != args.expected_runs:
        raise RuntimeError(f"expected {args.expected_runs} audits, found {len(paths)}")

    rows: list[dict[str, object]] = []
    for path in paths:
        payload = load_json(path)
        checks = payload["reproduction_checks"]
        if not checks["matches_original"]:
            raise RuntimeError(f"refit mismatch: {path}: {checks}")
        audit = payload["structural_pattern_audit"]
        counts = audit["category_counts"]
        nz = int(audit["num_active_patterns"])
        if sum(int(counts[name]) for name in CATEGORIES) != nz:
            raise RuntimeError(f"category counts do not sum to NZ: {path}")
        exact = int(counts["exact"])
        row: dict[str, object] = {
            "seed": int(payload["seed"]),
            "size": str(payload["size"]),
            "weight": str(payload["weight"]),
            "nz": nz,
            "exact_precision": exact / max(1, nz),
            "exact_recall": exact / 8.0,
            "r2": float(payload["metrics"]["R2"]),
            "pseudo_r2": float(payload["metrics"]["PseudoR2"]),
        }
        for category in CATEGORIES:
            row[category] = int(counts[category])
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_fields = [
        "seed",
        "size",
        "weight",
        "nz",
        *CATEGORIES,
        "exact_precision",
        "exact_recall",
        "r2",
        "pseudo_r2",
    ]
    with (args.output_dir / "structural_audit_all_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=all_fields)
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, object]] = []
    for size in ("n500", "n1500", "n3000"):
        for weight in ("small", "medium", "large"):
            group = [row for row in rows if row["size"] == size and row["weight"] == weight]
            if len(group) != 5:
                raise RuntimeError(f"expected five seeds for {size}/{weight}, found {len(group)}")
            summary: dict[str, object] = {"size": size, "weight": weight, "runs": 5}
            for metric in ("nz", *CATEGORIES, "exact_precision", "exact_recall"):
                mean, sd = mean_sd([float(row[metric]) for row in group])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_sd"] = sd
            summary_rows.append(summary)

    summary_fields = list(summary_rows[0])
    with (args.output_dir / "structural_audit_mean_sd.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summary_rows)

    size_labels = {"n500": "$m=500$", "n1500": "$m=1500$", "n3000": "$m=3000$"}
    with (args.output_dir / "structural_audit_rows.tex").open(
        "w", encoding="utf-8"
    ) as stream:
        for row in summary_rows:
            values = [
                f"{float(row['nz_mean']):.1f} $\\pm$ {float(row['nz_sd']):.1f}",
                f"{float(row['exact_mean']):.1f} $\\pm$ {float(row['exact_sd']):.1f}",
                f"{float(row['strict_subset_mean']):.1f} $\\pm$ {float(row['strict_subset_sd']):.1f}",
                f"{float(row['strict_superset_mean']):.1f} $\\pm$ {float(row['strict_superset_sd']):.1f}",
                f"{float(row['partial_overlap_mean']):.1f} $\\pm$ {float(row['partial_overlap_sd']):.1f}",
                f"{float(row['disjoint_mean']):.1f} $\\pm$ {float(row['disjoint_sd']):.1f}",
                f"{float(row['exact_precision_mean']):.3f} $\\pm$ {float(row['exact_precision_sd']):.3f}",
                f"{float(row['exact_recall_mean']):.3f} $\\pm$ {float(row['exact_recall_sd']):.3f}",
            ]
            stream.write(
                f"{size_labels[str(row['size'])]}, {str(row['weight'])} weight & "
                + " & ".join(values)
                + r" \\"
                + "\n"
            )
        stream.write("\\bottomrule\n")

    total_selected = sum(int(row["nz"]) for row in rows)
    total_counts = {
        category: sum(int(row[category]) for row in rows) for category in CATEGORIES
    }
    overall = {
        "runs": len(rows),
        "total_selected_patterns": total_selected,
        "category_counts": total_counts,
        "category_fractions": {
            category: count / total_selected for category, count in total_counts.items()
        },
        "non_disjoint_fraction": 1.0 - total_counts["disjoint"] / total_selected,
    }
    with (args.output_dir / "structural_audit_overall.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(overall, stream, indent=2, sort_keys=True)
        stream.write("\n")

    verification = {
        "audited_runs": len(rows),
        "all_refits_match_original": True,
        "maximum_r2_absolute_difference": max(
            float(load_json(path)["reproduction_checks"]["r2_absolute_difference"])
            for path in paths
        ),
        "maximum_pseudo_r2_absolute_difference": max(
            float(load_json(path)["reproduction_checks"]["pseudo_r2_absolute_difference"])
            for path in paths
        ),
        "maximum_absolute_nz_difference": max(
            abs(int(load_json(path)["reproduction_checks"]["nz_difference"]))
            for path in paths
        ),
    }
    with (args.output_dir / "reproduction_verification.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(verification, stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
