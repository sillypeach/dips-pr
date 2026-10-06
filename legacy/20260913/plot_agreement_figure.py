from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CONFIGS = ["D1", "D2", "D3", "D4", "D5"]
SPP_SPEEDUP = [10.3, 2.2, 8.6, 4.8, 4.5]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.summary.open(newline="") as handle:
        rows = {row["config"]: row for row in csv.DictReader(handle)}

    recall_rates = [
        100.0 * float(rows[config]["same_exact_recall"]) / float(rows[config]["seeds"])
        for config in CONFIGS
    ]
    selected_rates = [
        100.0 * float(rows[config]["same_selected_set"]) / float(rows[config]["seeds"])
        for config in CONFIGS
    ]
    plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 10.5})
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))

    x = np.arange(len(CONFIGS))
    width = 0.32
    axes[0].bar(x - width / 2, recall_rates, width, label="Embedded recall")
    axes[0].bar(x + width / 2, selected_rates, width, label="Selected set")
    axes[0].set_title("(a) Agreement over random seeds")
    axes[0].set_ylabel("Agreement rate (%)")
    axes[0].set_xticks(x, CONFIGS)
    axes[0].set_ylim(0, 112)
    axes[0].grid(axis="y", alpha=0.35)
    axes[0].set_axisbelow(True)
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    bars = axes[1].bar(CONFIGS, SPP_SPEEDUP, color="tab:green")
    axes[1].set_title("(b) Full pipeline gain over ordinary SPP")
    axes[1].set_ylabel("Speedup over ordinary SPP (x)")
    axes[1].set_ylim(0, 12.5)
    axes[1].grid(axis="y", alpha=0.35)
    axes[1].set_axisbelow(True)
    for bar, speedup in zip(bars, SPP_SPEEDUP):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            speedup + 0.25,
            f"{speedup:.1f}x",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

    fig.tight_layout(w_pad=1.8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
