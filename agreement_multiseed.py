from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import atomic_write_json
from model import PoissonEN_LCM_SPP
from utils import eval_feature_recovery_extended


TRUE_PATTERNS = [
    (3, 7),
    (10, 11, 12),
    (25, 26),
    (1, 5, 9),
    (2, 14),
    (6, 18, 33),
    (4, 21),
    (8, 16),
]
PATTERN_PROBABILITIES = [0.35, 0.30, 0.40, 0.28, 0.32, 0.25, 0.30, 0.27]
N_ITEMS = 80
DATA_CONFIGS = {
    "D1": {
        "n": 1500,
        "poisson": True,
        "weights": [0.35, -0.30, 0.25, 0.40, -0.30, 0.35, 0.30, -0.35],
        "lambda": 15,
        "eta_clip": 6.0,
    },
    "D2": {
        "n": 1500,
        "poisson": True,
        "weights": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
        "lambda": 30,
        "eta_clip": 6.0,
    },
    "D3": {
        "n": 1500,
        "poisson": True,
        "weights": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
        "lambda": 60,
        "eta_clip": 6.0,
    },
    "D4": {
        "n": 1500,
        "poisson": False,
        "weights": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
        "lambda": 15,
        "eta_clip": 12.0,
    },
    "D5": {
        "n": 3000,
        "poisson": True,
        "weights": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
        "lambda": 60,
        "eta_clip": 6.0,
    },
}


def make_transactions(n_samples: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    pattern_sets = [set(pattern) for pattern in TRUE_PATTERNS]
    pattern_items = set().union(*pattern_sets)
    noise_items = [item for item in range(N_ITEMS) if item not in pattern_items]
    transactions = []
    for _ in range(n_samples):
        items: set[int] = set()
        for pattern, probability in zip(pattern_sets, PATTERN_PROBABILITIES):
            if rng.random() < probability:
                items.update(pattern)
        for item in noise_items:
            if rng.random() < 0.05:
                items.add(item)
        transactions.append(sorted(items))
    return transactions


def oracle_matrix(transactions: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(transactions), len(TRUE_PATTERNS)), dtype=float)
    for row, transaction in enumerate(transactions):
        items = set(transaction)
        for column, pattern in enumerate(TRUE_PATTERNS):
            matrix[row, column] = float(set(pattern).issubset(items))
    return matrix


def make_response(
    design: np.ndarray,
    weights: list[float],
    poisson: bool,
    eta_clip: float,
    seed: int,
) -> np.ndarray:
    eta = -0.2 + design @ np.asarray(weights, dtype=float)
    if poisson:
        rng = np.random.default_rng(seed + 1000)
        return rng.poisson(np.exp(np.clip(eta, -eta_clip, eta_clip))).astype(float)
    response = design @ np.asarray(weights, dtype=float)
    return response - float(np.min(response)) + 1.0


def fit_mode(
    transactions: list[list[int]],
    response: np.ndarray,
    config: dict[str, object],
    seed: int,
    radius_mode: str,
):
    model = PoissonEN_LCM_SPP(
        lam=float(config["lambda"]),
        kappa=0.05,
        min_support=2,
        max_len=4,
        top_k_add=200,
        max_rounds=10,
        eta_clip=float(config["eta_clip"]),
        seed=seed,
        do_ws_screen=True,
        enum_mode_single="both",
        enum_mode_two="both",
        radius_mode=radius_mode,
    )
    started = time.perf_counter()
    result = model.fit(transactions, response, verbose=False)
    elapsed = time.perf_counter() - started
    selected = {
        tuple(pattern.itemset): float(result.params.w[index])
        for index, pattern in enumerate(result.patterns)
        if abs(float(result.params.w[index])) > 1e-8
    }
    recovery = eval_feature_recovery_extended(
        result,
        TRUE_PATTERNS,
        transactions,
    )
    nodes = sum(stat.nodes_visited for stat in result.enum_stats_per_round)
    return {
        "selected": selected,
        "recovery": recovery,
        "nodes_visited": int(nodes),
        "elapsed_seconds": float(elapsed),
    }


def compare_modes(heuristic: dict[str, object], certified: dict[str, object]):
    selected_h = heuristic["selected"]
    selected_c = certified["selected"]
    patterns_h = set(selected_h)
    patterns_c = set(selected_c)
    union = patterns_h | patterns_c
    common = patterns_h & patterns_c
    truth = [set(pattern) for pattern in TRUE_PATTERNS]
    certified_only = patterns_c - patterns_h
    certified_only_true_related = [
        list(pattern)
        for pattern in certified_only
        if any(set(pattern).issuperset(true_pattern) for true_pattern in truth)
    ]
    maximum_coefficient_difference = max(
        (
            abs(float(selected_h[pattern]) - float(selected_c[pattern]))
            for pattern in common
        ),
        default=0.0,
    )
    recovery_h = heuristic["recovery"]
    recovery_c = certified["recovery"]
    return {
        "same_selected_set": patterns_h == patterns_c,
        "selected_jaccard": float(len(common) / max(1, len(union))),
        "n_heuristic": len(patterns_h),
        "n_certified": len(patterns_c),
        "n_only_heuristic": len(patterns_h - patterns_c),
        "n_only_certified": len(certified_only),
        "maximum_common_coefficient_difference": float(
            maximum_coefficient_difference
        ),
        "exact_recall_heuristic": recovery_h["exact_recall"],
        "exact_recall_certified": recovery_c["exact_recall"],
        "superset_recall_heuristic": recovery_h["superset_recall"],
        "superset_recall_certified": recovery_c["superset_recall"],
        "same_exact_recall": bool(
            abs(recovery_h["exact_recall"] - recovery_c["exact_recall"]) < 1e-12
        ),
        "same_superset_recall": bool(
            abs(
                recovery_h["superset_recall"]
                - recovery_c["superset_recall"]
            )
            < 1e-12
        ),
        "certified_only_true_related": certified_only_true_related,
    }


def run_seed(seed: int) -> dict[str, object]:
    settings = []
    for name, config in DATA_CONFIGS.items():
        print(f"seed={seed} config={name}", flush=True)
        transactions = make_transactions(int(config["n"]), seed)
        response = make_response(
            oracle_matrix(transactions),
            config["weights"],
            bool(config["poisson"]),
            float(config["eta_clip"]),
            seed,
        )
        heuristic = fit_mode(transactions, response, config, seed, "heuristic")
        certified = fit_mode(transactions, response, config, seed, "certified")
        settings.append(
            {
                "config": name,
                "n": config["n"],
                "lambda": config["lambda"],
                "heuristic": {
                    key: value
                    for key, value in heuristic.items()
                    if key != "selected"
                },
                "certified": {
                    key: value
                    for key, value in certified.items()
                    if key != "selected"
                },
                "comparison": compare_modes(heuristic, certified),
            }
        )
    return {"seed": seed, "settings": settings}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/agreement_multiseed"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        output_path = args.output_dir / f"seed_{seed}.json"
        if args.resume and output_path.exists():
            print(f"skip seed={seed}", flush=True)
            continue
        try:
            atomic_write_json(output_path, run_seed(seed))
            print(f"saved={output_path}", flush=True)
        except Exception as error:
            atomic_write_json(
                args.output_dir / f"seed_{seed}.error.json",
                {"seed": seed, "error": repr(error)},
            )
            raise


if __name__ == "__main__":
    main()
