from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model import PoissonEN_LCM_SPP, summarize_pruning
from utils import eval_feature_recovery_extended


TRUE_PATTERNS = [(1, 3), (2, 5, 7)]
PATTERN_PROBS = [0.35, 0.28]
N_ITEMS = 20


def make_transactions(n: int = 240, seed: int = 7) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    pattern_sets = [set(p) for p in TRUE_PATTERNS]
    pattern_items = set().union(*pattern_sets)
    noise_items = [i for i in range(N_ITEMS) if i not in pattern_items]

    transactions: list[list[int]] = []
    for _ in range(n):
        items: set[int] = set()
        for pset, prob in zip(pattern_sets, PATTERN_PROBS):
            if rng.random() < prob:
                items.update(pset)
        for item in pattern_items:
            if item not in items and rng.random() < 0.08:
                items.add(item)
        for item in noise_items:
            if rng.random() < 0.04:
                items.add(item)
        transactions.append(sorted(items))
    return transactions


def build_oracle_matrix(transactions: list[list[int]]) -> np.ndarray:
    X = np.zeros((len(transactions), len(TRUE_PATTERNS)), dtype=float)
    for i, transaction in enumerate(transactions):
        itemset = set(transaction)
        for j, pattern in enumerate(TRUE_PATTERNS):
            X[i, j] = float(set(pattern).issubset(itemset))
    return X


def make_response(X: np.ndarray, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    weights = np.array([1.0, -0.8])
    eta = -0.1 + X @ weights
    return rng.poisson(np.exp(np.clip(eta, -5.0, 5.0))).astype(float)


def main() -> None:
    transactions = make_transactions()
    X_oracle = build_oracle_matrix(transactions)
    y = make_response(X_oracle)

    model = PoissonEN_LCM_SPP(
        lam=3.0,
        kappa=0.05,
        min_support=5,
        max_len=3,
        top_k_add=50,
        max_rounds=5,
        eta_clip=6.0,
        do_ws_screen=True,
        enum_mode_single="both",
        enum_mode_two="both",
        radius_mode="heuristic",
    )
    result = model.fit(transactions, y, verbose=False)

    summary = summarize_pruning(result)
    recovery = eval_feature_recovery_extended(result, TRUE_PATTERNS, transactions)

    selected = [
        (result.patterns[j].itemset, float(result.params.w[j]))
        for j in range(len(result.patterns))
        if abs(result.params.w[j]) > 1e-8
    ]

    print("Selected patterns:")
    for itemset, weight in selected[:10]:
        print(f"  {itemset}: {weight:+.4f}")
    print()
    print("Pruning summary:")
    for key in ["nodes_visited", "v_prune_rate", "u_screen_rate", "time_total_sec"]:
        print(f"  {key}: {summary[key]}")
    print()
    print("Recovery:")
    for key in ["exact_recall", "superset_recall", "num_selected_nz"]:
        print(f"  {key}: {recovery[key]}")


if __name__ == "__main__":
    main()

