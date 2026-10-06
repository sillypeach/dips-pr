from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from common import atomic_write_json, load_json, regression_metrics
from synthetic_multiseed import (
    SIZE_CONFIGS,
    TRAIN_FRACTION,
    TRUE_PATTERNS,
    WEIGHT_CONFIGS,
    build_oracle_matrix,
    dips_factory,
    exact_rank_metrics,
    make_response,
    make_transactions,
)
from utils import build_X_trans, eval_feature_recovery_extended


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
    source_dips = next(
        model for model in source["models"] if model["model"] == "DIPS-PR"
    )
    cv = source_dips["cv_metrics"]
    rmse_key = min(cv, key=lambda key: cv[key]["RMSE"])
    rmse_lambda = float(rmse_key)

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
    factory, lambda_scale, minimum_support = dips_factory(n_train, seed)

    started = time.perf_counter()
    model = factory(rmse_lambda)
    result = model.fit(transactions_train, y_train, verbose=False)
    design_test = (
        build_X_trans(transactions_test, result.patterns)
        if result.patterns
        else np.zeros((len(transactions_test), 0), dtype=float)
    )
    eta = np.full(len(y_test), result.params.b, dtype=float)
    if design_test.shape[1]:
        eta += design_test @ result.params.w
    prediction = np.exp(np.clip(eta, -model.eta_clip, model.eta_clip))
    recovery = eval_feature_recovery_extended(
        result,
        TRUE_PATTERNS,
        transactions_train,
    )
    recovery.update(exact_rank_metrics(result))
    nodes = sum(stat.nodes_visited for stat in result.enum_stats_per_round)
    return {
        "seed": seed,
        "size": size_label,
        "weight": weight_label,
        "n_total": n_samples,
        "n_train": n_train,
        "n_test": len(y_test),
        "selection_metric": "training_3fold_CV_RMSE",
        "lambda_base": rmse_lambda,
        "lambda_effective": rmse_lambda * lambda_scale,
        "minimum_support": minimum_support,
        "cv_selected_metrics": cv[rmse_key],
        "metrics": regression_metrics(y_test, prediction),
        "recovery": recovery,
        "nodes_visited": int(nodes),
        "elapsed_seconds": float(time.perf_counter() - started),
        "source_nll_selected": {
            "lambda_base": source_dips["best_lambda_base"],
            "metrics": source_dips["metrics"],
            "recovery": source_dips["recovery"],
            "nodes_visited": source_dips["nodes_visited"],
        },
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
        default=Path("results/synthetic_rmse_lambda"),
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
                    f"  lambda={payload['lambda_base']:g} "
                    f"r2={payload['metrics']['R2']:.4f} saved={output_path}",
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
