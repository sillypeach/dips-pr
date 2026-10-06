from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from common import atomic_write_json, regression_metrics
from synthetic_multiseed import (
    SIZE_CONFIGS,
    TRAIN_FRACTION,
    WEIGHT_CONFIGS,
    build_item_matrix,
    build_oracle_matrix,
    dips_factory,
    make_response,
    make_transactions,
    structural_pattern_audit,
)
from utils import build_X_trans, eval_feature_recovery_extended
from synthetic_multiseed import TRUE_PATTERNS, exact_rank_metrics


def load_original(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def original_dips(payload: dict[str, object]) -> dict[str, object]:
    for model in payload["models"]:
        if model["model"] == "DIPS-PR":
            return model
    raise ValueError("DIPS-PR result is missing from the original checkpoint")


def run_refit(seed: int, size: str, weight: str, original_path: Path) -> dict[str, object]:
    original = load_original(original_path)
    old_dips = original_dips(original)
    transactions = make_transactions(SIZE_CONFIGS[size], seed)
    response = make_response(
        build_oracle_matrix(transactions),
        WEIGHT_CONFIGS[weight],
        10000 + seed,
    )
    n_train = int(len(response) * TRAIN_FRACTION)
    transactions_train = transactions[:n_train]
    transactions_test = transactions[n_train:]
    y_train = response[:n_train]
    y_test = response[n_train:]

    factory, lambda_scale, minimum_support = dips_factory(n_train, seed)
    lambda_base = float(old_dips["best_lambda_base"])
    started = time.perf_counter()
    model = factory(lambda_base)
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
    metrics = regression_metrics(y_test, prediction)
    recovery = eval_feature_recovery_extended(result, TRUE_PATTERNS, transactions_train)
    recovery.update(exact_rank_metrics(result))
    audit = structural_pattern_audit(result)

    old_metrics = old_dips["metrics"]
    old_nz = int(round(float(old_dips["recovery"]["num_selected_nz"])))
    checks = {
        "r2_absolute_difference": abs(float(metrics["R2"]) - float(old_metrics["R2"])),
        "pseudo_r2_absolute_difference": abs(
            float(metrics["PseudoR2"]) - float(old_metrics["PseudoR2"])
        ),
        "nz_difference": int(audit["num_active_patterns"]) - old_nz,
    }
    checks["matches_original"] = bool(
        checks["r2_absolute_difference"] <= 1e-10
        and checks["pseudo_r2_absolute_difference"] <= 1e-10
        and checks["nz_difference"] == 0
    )

    return {
        "seed": seed,
        "size": size,
        "weight": weight,
        "n_total": len(response),
        "n_train": n_train,
        "n_test": len(y_test),
        "best_lambda_base": lambda_base,
        "best_lambda_effective": lambda_base * lambda_scale,
        "minimum_support": minimum_support,
        "metrics": metrics,
        "recovery": recovery,
        "structural_pattern_audit": audit,
        "original_checkpoint": str(original_path),
        "reproduction_checks": checks,
        "elapsed_seconds": time.perf_counter() - started,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refit final DIPS-PR models and audit active itemset structure."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--original-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="*", choices=SIZE_CONFIGS, default=list(SIZE_CONFIGS))
    parser.add_argument(
        "--weights", nargs="*", choices=WEIGHT_CONFIGS, default=list(WEIGHT_CONFIGS)
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_directory = args.output_dir / f"seed_{args.seed}"
    seed_directory.mkdir(parents=True, exist_ok=True)
    for size in args.sizes:
        for weight in args.weights:
            key = f"{size}__{weight}"
            output_path = seed_directory / f"{key}.json"
            if args.resume and output_path.exists():
                print(f"skip completed {key}", flush=True)
                continue
            original_path = args.original_dir / f"seed_{args.seed}" / f"{key}.json"
            print(f"refit seed={args.seed} setting={key}", flush=True)
            payload = run_refit(args.seed, size, weight, original_path)
            atomic_write_json(output_path, payload)
            print(
                f"saved={output_path} matched={payload['reproduction_checks']['matches_original']}",
                flush=True,
            )


if __name__ == "__main__":
    main()
