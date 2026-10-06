from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Lasso, PoissonRegressor, Ridge
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from common import atomic_write_json, load_json, regression_metrics
from model import PoissonEN_LCM_SPP, fit_cv
from utils import build_X_trans, eval_feature_recovery_extended


warnings.filterwarnings("ignore", category=ConvergenceWarning)

TRUE_PATTERNS = [
    (3, 7, 15, 22),
    (10, 11, 12, 19),
    (25, 26, 37, 48),
    (1, 5, 9, 17),
    (2, 14, 31, 45),
    (6, 18, 33, 55),
    (4, 21, 38, 52),
    (8, 16, 29, 43),
]
PATTERN_PROBABILITIES = [0.35, 0.30, 0.38, 0.28, 0.32, 0.25, 0.30, 0.27]
WEIGHT_CONFIGS = {
    "small": [0.55, -0.50, 0.50, 0.65, -0.50, 0.55, 0.50, -0.55],
    "medium": [0.90, -0.80, 0.75, 1.10, -0.85, 0.95, 0.80, -1.00],
    "large": [1.35, -1.20, 1.125, 1.65, -1.275, 1.425, 1.20, -1.50],
}
SIZE_CONFIGS = {"n500": 500, "n1500": 1500, "n3000": 3000}
N_ITEMS = 150
INDIVIDUAL_NOISE_PROBABILITY = 0.10
TRAIN_FRACTION = 0.70
LAMBDA_CANDIDATES = [200, 100, 50, 30, 15, 10, 7, 5, 3, 2]


def make_transactions(n_samples: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    pattern_sets = [set(pattern) for pattern in TRUE_PATTERNS]
    pattern_items = set().union(*pattern_sets)
    noise_items = [item for item in range(N_ITEMS) if item not in pattern_items]
    transactions: list[list[int]] = []
    for _ in range(n_samples):
        items: set[int] = set()
        for pattern, probability in zip(pattern_sets, PATTERN_PROBABILITIES):
            if rng.random() < probability:
                items.update(pattern)
        for item in pattern_items:
            if item not in items and rng.random() < INDIVIDUAL_NOISE_PROBABILITY:
                items.add(item)
        for item in noise_items:
            if rng.random() < 0.05:
                items.add(item)
        transactions.append(sorted(items))
    return transactions


def build_item_matrix(transactions: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(transactions), N_ITEMS), dtype=float)
    for row, transaction in enumerate(transactions):
        matrix[row, transaction] = 1.0
    return matrix


def build_oracle_matrix(transactions: list[list[int]]) -> np.ndarray:
    matrix = np.zeros((len(transactions), len(TRUE_PATTERNS)), dtype=float)
    pattern_sets = [set(pattern) for pattern in TRUE_PATTERNS]
    for row, transaction in enumerate(transactions):
        item_set = set(transaction)
        for column, pattern in enumerate(pattern_sets):
            matrix[row, column] = float(pattern.issubset(item_set))
    return matrix


def make_response(
    oracle_matrix: np.ndarray,
    weights: list[float],
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    eta = -0.2 + oracle_matrix @ np.asarray(weights, dtype=float)
    return rng.poisson(np.exp(np.clip(eta, -5.0, 5.0))).astype(float)


def baseline_specs(seed: int) -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    return {
        "Ridge": (
            Pipeline([("scale", StandardScaler()), ("model", Ridge())]),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0]},
        ),
        "Lasso": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Lasso(max_iter=10000)),
                ]
            ),
            {"model__alpha": [0.001, 0.01, 0.1]},
        ),
        "ElasticNet": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", ElasticNet(max_iter=10000)),
                ]
            ),
            {
                "model__alpha": [0.001, 0.01, 0.1],
                "model__l1_ratio": [0.2, 0.5, 0.8],
            },
        ),
        "PoissonGLM": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", PoissonRegressor(max_iter=5000)),
                ]
            ),
            {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
        ),
        "RBF-SVR": (
            Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf"))]),
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__epsilon": [0.01, 0.1],
            },
        ),
        "RandomForest": (
            Pipeline(
                [
                    (
                        "model",
                        RandomForestRegressor(
                            n_estimators=300,
                            random_state=seed,
                            n_jobs=1,
                        ),
                    )
                ]
            ),
            {
                "model__max_depth": [None, 12],
                "model__min_samples_leaf": [1, 5],
                "model__max_features": [1.0, "sqrt"],
            },
        ),
        "MLP": (
            Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        MLPRegressor(
                            random_state=seed,
                            early_stopping=True,
                            validation_fraction=0.15,
                            max_iter=1000,
                            n_iter_no_change=30,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__alpha": [1e-4, 1e-3],
            },
        ),
    }


def run_baselines(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    jobs: int,
) -> list[dict[str, object]]:
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    rows: list[dict[str, object]] = []
    for name, (estimator, parameters) in baseline_specs(seed).items():
        print(f"    baseline={name} tuning", flush=True)
        started = time.perf_counter()
        search = GridSearchCV(
            estimator=clone(estimator),
            param_grid=parameters,
            scoring="neg_root_mean_squared_error",
            cv=splitter,
            n_jobs=jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        prediction = search.predict(x_test)
        rows.append(
            {
                "model": name,
                "metrics": regression_metrics(y_test, prediction),
                "cv_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
    return rows


def dips_factory(n_train: int, seed: int):
    scale = max(1.0, n_train / 350.0)
    minimum_support = max(3, int(n_train * 0.04))

    def factory(lambda_base: float) -> PoissonEN_LCM_SPP:
        return PoissonEN_LCM_SPP(
            lam=float(lambda_base) * scale,
            kappa=0.05,
            min_support=minimum_support,
            max_len=4,
            top_k_add=200,
            max_rounds=10,
            solver_max_iter=800,
            solver_tol=1e-7,
            eta_clip=6.0,
            seed=seed,
            do_ws_screen=True,
            enum_mode_single="both",
            enum_mode_two="both",
        )

    return factory, scale, minimum_support


def exact_rank_metrics(result) -> dict[str, object]:
    true_set = {tuple(pattern) for pattern in TRUE_PATTERNS}
    nonzero = [
        index
        for index, weight in enumerate(result.params.w)
        if abs(float(weight)) > 1e-8
    ]
    ordered = sorted(nonzero, key=lambda index: -abs(float(result.params.w[index])))
    ranks = {
        tuple(result.patterns[index].itemset): rank
        for rank, index in enumerate(ordered, start=1)
    }
    true_ranks = [ranks.get(pattern) for pattern in sorted(true_set)]
    finite_ranks = [rank for rank in true_ranks if rank is not None]
    payload: dict[str, object] = {
        "true_pattern_ranks": true_ranks,
        "mean_true_pattern_rank": (
            float(np.mean(finite_ranks)) if finite_ranks else None
        ),
    }
    for cutoff in (8, 10, 20, 50):
        top_patterns = {
            tuple(result.patterns[index].itemset)
            for index in ordered[:cutoff]
        }
        hits = len(top_patterns & true_set)
        payload[f"precision_at_{cutoff}"] = float(
            hits / max(1, min(cutoff, len(ordered)))
        )
        payload[f"recall_at_{cutoff}"] = float(hits / len(true_set))
    return payload


def structural_pattern_audit(result) -> dict[str, object]:
    """Classify every active itemset under the strict exact-recovery target."""
    truth_sets = [set(pattern) for pattern in TRUE_PATTERNS]
    active = [
        index
        for index, weight in enumerate(result.params.w)
        if abs(float(weight)) > 1e-8
    ]
    ordered = sorted(active, key=lambda index: -abs(float(result.params.w[index])))
    counts = {
        "exact": 0,
        "strict_subset": 0,
        "strict_superset": 0,
        "partial_overlap": 0,
        "disjoint": 0,
    }
    patterns: list[dict[str, object]] = []

    for rank, index in enumerate(ordered, start=1):
        selected = set(result.patterns[index].itemset)
        exact_matches = [i + 1 for i, truth in enumerate(truth_sets) if selected == truth]
        subset_matches = [i + 1 for i, truth in enumerate(truth_sets) if selected < truth]
        superset_matches = [i + 1 for i, truth in enumerate(truth_sets) if selected > truth]
        overlap_matches = [i + 1 for i, truth in enumerate(truth_sets) if selected & truth]

        if exact_matches:
            category = "exact"
            related_truths = exact_matches
        elif subset_matches:
            category = "strict_subset"
            related_truths = subset_matches
        elif superset_matches:
            category = "strict_superset"
            related_truths = superset_matches
        elif overlap_matches:
            category = "partial_overlap"
            related_truths = overlap_matches
        else:
            category = "disjoint"
            related_truths = []

        counts[category] += 1
        pattern = result.patterns[index]
        patterns.append(
            {
                "rank": rank,
                "itemset": list(pattern.itemset),
                "coefficient": float(result.params.w[index]),
                "support": int(pattern.support()),
                "category": category,
                "related_truth_indices": related_truths,
            }
        )

    denominator = max(1, len(ordered))
    return {
        "classification_priority": [
            "exact",
            "strict_subset",
            "strict_superset",
            "partial_overlap",
            "disjoint",
        ],
        "category_counts": counts,
        "category_fractions": {
            category: float(count / denominator)
            for category, count in counts.items()
        },
        "num_active_patterns": len(ordered),
        "active_patterns": patterns,
    }


def run_dips(
    transactions_train: list[list[int]],
    y_train: np.ndarray,
    transactions_test: list[list[int]],
    y_test: np.ndarray,
    seed: int,
) -> dict[str, object]:
    factory, lambda_scale, minimum_support = dips_factory(len(y_train), seed)
    started = time.perf_counter()
    best_lambda, cv_metrics = fit_cv(
        factory,
        transactions_train,
        y_train,
        lam_list=LAMBDA_CANDIDATES,
        K=3,
        seed=seed,
        use_two_ref=True,
    )
    model = factory(best_lambda)
    result = model.fit(transactions_train, y_train, verbose=False)
    design_test = (
        build_X_trans(transactions_test, result.patterns)
        if result.patterns
        else np.zeros((len(transactions_test), 0), dtype=float)
    )
    eta = np.full(len(transactions_test), result.params.b, dtype=float)
    if design_test.shape[1]:
        eta = eta + design_test @ result.params.w
    prediction = np.exp(np.clip(eta, -model.eta_clip, model.eta_clip))
    recovery = eval_feature_recovery_extended(
        result,
        TRUE_PATTERNS,
        transactions_train,
    )
    recovery.update(exact_rank_metrics(result))
    nodes = sum(stat.nodes_visited for stat in result.enum_stats_per_round)
    return {
        "model": "DIPS-PR",
        "metrics": regression_metrics(y_test, prediction),
        "best_lambda_base": float(best_lambda),
        "best_lambda_effective": float(best_lambda * lambda_scale),
        "minimum_support": int(minimum_support),
        "cv_metrics": cv_metrics,
        "recovery": recovery,
        "structural_pattern_audit": structural_pattern_audit(result),
        "nodes_visited": int(nodes),
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def run_setting(
    seed: int,
    size_label: str,
    n_samples: int,
    weight_label: str,
    weights: list[float],
    baseline_jobs: int,
) -> dict[str, object]:
    transactions = make_transactions(n_samples, seed)
    oracle_matrix = build_oracle_matrix(transactions)
    response = make_response(oracle_matrix, weights, 10000 + seed)
    n_train = int(n_samples * TRAIN_FRACTION)
    transactions_train = transactions[:n_train]
    transactions_test = transactions[n_train:]
    y_train = response[:n_train]
    y_test = response[n_train:]
    item_matrix = build_item_matrix(transactions)
    x_train = item_matrix[:n_train]
    x_test = item_matrix[n_train:]

    print(
        f"  setting={size_label}/{weight_label} "
        f"train={n_train} test={len(y_test)}",
        flush=True,
    )
    baselines = run_baselines(
        x_train,
        y_train,
        x_test,
        y_test,
        seed,
        baseline_jobs,
    )
    print("    model=DIPS-PR fitting", flush=True)
    dips = run_dips(
        transactions_train,
        y_train,
        transactions_test,
        y_test,
        seed,
    )
    return {
        "seed": seed,
        "transaction_seed": seed,
        "response_seed": 10000 + seed,
        "size": size_label,
        "weight": weight_label,
        "n_total": n_samples,
        "n_train": n_train,
        "n_test": len(y_test),
        "models": baselines + [dips],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("results/synthetic"))
    parser.add_argument("--baseline-jobs", type=int, default=2)
    parser.add_argument("--sizes", nargs="*", choices=SIZE_CONFIGS, default=list(SIZE_CONFIGS))
    parser.add_argument(
        "--weights",
        nargs="*",
        choices=WEIGHT_CONFIGS,
        default=list(WEIGHT_CONFIGS),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_directory = args.output_dir / f"seed_{args.seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = seed_directory / "manifest.json"
    completed: dict[str, object] = {}
    if args.resume and manifest_path.exists():
        completed = load_json(manifest_path).get("completed", {})

    for size_label in args.sizes:
        for weight_label in args.weights:
            key = f"{size_label}__{weight_label}"
            result_path = seed_directory / f"{key}.json"
            if args.resume and result_path.exists():
                print(f"skip completed {key}", flush=True)
                completed[key] = str(result_path)
                continue
            try:
                result = run_setting(
                    args.seed,
                    size_label,
                    SIZE_CONFIGS[size_label],
                    weight_label,
                    WEIGHT_CONFIGS[weight_label],
                    args.baseline_jobs,
                )
                atomic_write_json(result_path, result)
                completed[key] = str(result_path)
                atomic_write_json(
                    manifest_path,
                    {"seed": args.seed, "completed": completed},
                )
                print(f"  saved={result_path}", flush=True)
            except Exception as error:
                atomic_write_json(
                    seed_directory / f"{key}.error.json",
                    {"seed": args.seed, "setting": key, "error": repr(error)},
                )
                raise


if __name__ == "__main__":
    main()
