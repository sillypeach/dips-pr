from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import KFold

from common import atomic_write_json, load_json, regression_metrics
from synthetic_multiseed import (
    SIZE_CONFIGS,
    TRAIN_FRACTION,
    TRUE_PATTERNS,
    WEIGHT_CONFIGS,
    build_oracle_matrix,
    dips_factory,
    make_response,
    make_transactions,
)
from utils import build_X_trans


TOP_K_CANDIDATES = (8, 20, 50, 100)
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)


def recovery_metrics(patterns) -> dict[str, float | int]:
    selected = {tuple(pattern.itemset) for pattern in patterns}
    truth = {tuple(pattern) for pattern in TRUE_PATTERNS}
    hits = len(selected & truth)
    precision = hits / max(1, len(selected))
    recall = hits / len(truth)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "n_selected": len(selected),
        "n_exact_hits": hits,
        "exact_precision": float(precision),
        "exact_recall": float(recall),
        "exact_f1": float(f1),
    }


def refit_model(alpha: float) -> PoissonRegressor:
    return PoissonRegressor(
        alpha=alpha,
        fit_intercept=True,
        max_iter=10000,
        tol=1e-9,
    )


def select_compact_size(
    design: np.ndarray,
    response: np.ndarray,
    candidates: list[int],
    seed: int,
) -> tuple[int, float, dict[str, dict[str, object]]]:
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    rows: dict[str, dict[str, object]] = {}
    fold_scores = {
        (candidate, alpha): []
        for candidate in candidates
        for alpha in RIDGE_ALPHAS
    }
    for train_indices, validation_indices in splitter.split(design):
        for candidate in candidates:
            for alpha in RIDGE_ALPHAS:
                model = refit_model(alpha)
                model.fit(
                    design[train_indices, :candidate],
                    response[train_indices],
                )
                prediction = model.predict(design[validation_indices, :candidate])
                rmse = float(
                    np.sqrt(
                        np.mean((response[validation_indices] - prediction) ** 2)
                    )
                )
                fold_scores[(candidate, alpha)].append(rmse)

    for candidate in candidates:
        alpha_rows = {}
        for alpha in RIDGE_ALPHAS:
            scores = np.asarray(fold_scores[(candidate, alpha)], dtype=float)
            alpha_rows[str(alpha)] = {
                "rmse_mean": float(np.mean(scores)),
                "rmse_std": float(np.std(scores, ddof=1)),
                "rmse_se": float(np.std(scores, ddof=1) / np.sqrt(len(scores))),
            }
        best_alpha = min(
            RIDGE_ALPHAS,
            key=lambda alpha: alpha_rows[str(alpha)]["rmse_mean"],
        )
        rows[str(candidate)] = {
            "best_alpha": best_alpha,
            "rmse_mean": alpha_rows[str(best_alpha)]["rmse_mean"],
            "rmse_std": alpha_rows[str(best_alpha)]["rmse_std"],
            "rmse_se": alpha_rows[str(best_alpha)]["rmse_se"],
            "alphas": alpha_rows,
        }
    best = min(candidates, key=lambda candidate: rows[str(candidate)]["rmse_mean"])
    threshold = rows[str(best)]["rmse_mean"] + rows[str(best)]["rmse_se"]
    selected = min(
        candidate
        for candidate in candidates
        if rows[str(candidate)]["rmse_mean"] <= threshold
    )
    for candidate in candidates:
        rows[str(candidate)]["within_one_se"] = bool(
            rows[str(candidate)]["rmse_mean"] <= threshold
        )
    return selected, float(rows[str(selected)]["best_alpha"]), rows


def run_setting(
    seed: int,
    size_label: str,
    weight_label: str,
    source_results: Path,
) -> dict[str, object]:
    source_path = (
        source_results
        / f"seed_{seed}"
        / f"{size_label}__{weight_label}.json"
    )
    source = load_json(source_path)
    n_samples = SIZE_CONFIGS[size_label]
    transactions = make_transactions(n_samples, seed)
    response = make_response(
        build_oracle_matrix(transactions),
        WEIGHT_CONFIGS[weight_label],
        10000 + seed,
    )
    n_train = int(n_samples * TRAIN_FRACTION)
    transactions_train = transactions[:n_train]
    transactions_test = transactions[n_train:]
    y_train = response[:n_train]
    y_test = response[n_train:]

    dips_source = next(
        model for model in source["models"] if model["model"] == "DIPS-PR"
    )
    selected_lambda = float(dips_source["best_lambda_base"])
    factory, _, _ = dips_factory(n_train, seed)
    started = time.perf_counter()
    result = factory(selected_lambda).fit(
        transactions_train,
        y_train,
        verbose=False,
    )
    active = [
        index
        for index, weight in enumerate(result.params.w)
        if abs(float(weight)) > 1e-8
    ]
    ordered = sorted(active, key=lambda index: -abs(float(result.params.w[index])))
    if not ordered:
        raise RuntimeError("DIPS-PR returned no active patterns")
    maximum = min(max(TOP_K_CANDIDATES), len(ordered))
    ordered = ordered[:maximum]
    ranked_patterns = [result.patterns[index] for index in ordered]
    design_train = build_X_trans(transactions_train, ranked_patterns)
    design_test = build_X_trans(transactions_test, ranked_patterns)
    candidates = sorted(
        {min(candidate, len(ranked_patterns)) for candidate in TOP_K_CANDIDATES}
    )
    selected_k, selected_alpha, cv_rows = select_compact_size(
        design_train,
        y_train,
        candidates,
        seed,
    )

    variants = []
    for candidate in candidates:
        alpha = float(cv_rows[str(candidate)]["best_alpha"])
        model = refit_model(alpha)
        model.fit(design_train[:, :candidate], y_train)
        prediction = model.predict(design_test[:, :candidate])
        variants.append(
            {
                "top_k": candidate,
                "selected_by_one_se": candidate == selected_k,
                "ridge_alpha": alpha,
                "metrics": regression_metrics(y_test, prediction),
                "recovery": recovery_metrics(ranked_patterns[:candidate]),
                "intercept": float(model.intercept_),
                "coefficients": model.coef_,
                "patterns": [
                    list(pattern.itemset) for pattern in ranked_patterns[:candidate]
                ],
            }
        )
    return {
        "seed": seed,
        "size": size_label,
        "weight": weight_label,
        "n_total": n_samples,
        "n_train": n_train,
        "n_test": len(y_test),
        "source_result": str(source_path),
        "best_lambda_base": selected_lambda,
        "full_dips_metrics": dips_source["metrics"],
        "full_dips_recovery": dips_source["recovery"],
        "full_active_patterns": len(active),
        "selection_rule": "smallest_k_within_one_se_of_best_training_cv_rmse",
        "refit": "ridge_stabilized_PoissonRegressor",
        "selected_k": selected_k,
        "selected_alpha": selected_alpha,
        "cv": cv_rows,
        "variants": variants,
        "elapsed_seconds": float(time.perf_counter() - started),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--source-results",
        type=Path,
        default=Path("results/synthetic"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/synthetic_compact_refit"),
    )
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
    for size_label in args.sizes:
        for weight_label in args.weights:
            output_path = seed_directory / f"{size_label}__{weight_label}.json"
            if args.resume and output_path.exists():
                print(f"skip completed {size_label}/{weight_label}", flush=True)
                continue
            print(
                f"seed={args.seed} setting={size_label}/{weight_label}",
                flush=True,
            )
            try:
                payload = run_setting(
                    args.seed,
                    size_label,
                    weight_label,
                    args.source_results,
                )
                atomic_write_json(output_path, payload)
                print(
                    f"  selected_k={payload['selected_k']} "
                    f"saved={output_path}",
                    flush=True,
                )
            except Exception as error:
                atomic_write_json(
                    output_path.with_suffix(".error.json"),
                    {
                        "seed": args.seed,
                        "size": size_label,
                        "weight": weight_label,
                        "error": repr(error),
                    },
                )
                raise


if __name__ == "__main__":
    main()
