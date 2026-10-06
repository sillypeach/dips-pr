from __future__ import annotations

import argparse
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

from common import atomic_write_json, regression_metrics
from model import PoissonEN_LCM_SPP
from utils import build_X_trans


warnings.filterwarnings("ignore", category=ConvergenceWarning)

DRUG_CLASSES = {
    "NRTI": ["3TC", "ABC", "AZT", "D4T", "DDI", "TDF"],
    "NNRTI": ["EFV", "NVP", "ETR", "RPV", "DOR"],
    "PI": ["FPV", "ATV", "IDV", "LPV", "NFV", "SQV", "TPV", "DRV"],
    "INI": ["RAL", "EVG", "DTG", "BIC", "CAB"],
    "CAI": ["LEN"],
}

# Selected by the original three-fold training-set CV runs. Reusing these
# values keeps the test split untouched while making the refits reproducible.
DIPS_LAMBDAS = {
    "CAI/LEN": 0.5,
    "INI/BIC": 0.5,
    "INI/CAB": 10.0,
    "INI/DTG": 5.0,
    "INI/EVG": 10.0,
    "INI/RAL": 10.0,
    "NNRTI/DOR": 30.0,
    "NNRTI/EFV": 1000.0,
    "NNRTI/ETR": 200.0,
    "NNRTI/NVP": 1000.0,
    "NNRTI/RPV": 50.0,
    "NRTI/3TC": 500.0,
    "NRTI/ABC": 15.0,
    "NRTI/AZT": 300.0,
    "NRTI/D4T": 15.0,
    "NRTI/DDI": 12.0,
    "NRTI/TDF": 15.0,
    "PI/ATV": 500.0,
    "PI/DRV": 500.0,
    "PI/FPV": 500.0,
    "PI/IDV": 300.0,
    "PI/LPV": 300.0,
    "PI/NFV": 300.0,
    "PI/SQV": 700.0,
    "PI/TPV": 500.0,
}


def parse_dataset(path: Path):
    with path.open("r", encoding="utf-8") as stream:
        lines = stream.readlines()
    header = lines[0].rstrip("\n").split("\t")
    position_columns = [
        index
        for index, name in enumerate(header)
        if name.startswith("P") and name[1:].isdigit()
    ]
    if not position_columns:
        raise ValueError(f"No position columns in {path}")
    first_position = position_columns[0]
    drug_names = header[1:first_position]
    response_rows: list[list[float]] = []
    transactions: list[list[int]] = []
    for line in lines[1:]:
        columns = line.rstrip("\n").split("\t")
        if len(columns) <= position_columns[-1]:
            continue
        responses: list[float] = []
        for index in range(1, first_position):
            value = columns[index].strip()
            try:
                responses.append(float(value))
            except ValueError:
                responses.append(float("nan"))
        items = [
            item_index
            for item_index, column_index in enumerate(position_columns)
            if columns[column_index].strip() not in {"", "-", "."}
        ]
        response_rows.append(responses)
        transactions.append(items)
    return drug_names, np.asarray(response_rows, dtype=float), transactions


def split_endpoint(
    response_matrix: np.ndarray,
    transactions: list[list[int]],
    drug_index: int,
    seed: int,
):
    valid_indices = np.where(~np.isnan(response_matrix[:, drug_index]))[0]
    rng = np.random.RandomState(seed)
    shuffled = valid_indices[rng.permutation(len(valid_indices))]
    n_validation = max(1, int(len(shuffled) * 0.10))
    n_test = max(1, int(len(shuffled) * 0.10))
    n_train = len(shuffled) - n_validation - n_test
    groups = {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train : n_train + n_validation],
        "test": shuffled[n_train + n_validation :],
    }
    response = response_matrix[:, drug_index]
    return {
        name: (
            [transactions[index] for index in indices],
            response[indices].astype(float),
        )
        for name, indices in groups.items()
    }


def item_matrix(transactions: list[list[int]], n_items: int) -> np.ndarray:
    matrix = np.zeros((len(transactions), n_items), dtype=float)
    for row, transaction in enumerate(transactions):
        matrix[row, transaction] = 1.0
    return matrix


def model_specs(seed: int):
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
                "model__C": [0.1, 1.0, 10.0, 100.0],
                "model__epsilon": [0.01, 0.1, 1.0],
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
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1, 2, 5],
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
                            max_iter=1200,
                            n_iter_no_change=40,
                        ),
                    ),
                ]
            ),
            {
                "model__hidden_layer_sizes": [(64,), (128,), (64, 32)],
                "model__alpha": [1e-4, 1e-3, 1e-2],
            },
        ),
    }


def fit_dips(
    drug_class: str,
    drug: str,
    transactions_train_validation: list[list[int]],
    response_train_validation: np.ndarray,
    transactions_test: list[list[int]],
    response_test: np.ndarray,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    selected_lambda = DIPS_LAMBDAS[f"{drug_class}/{drug}"]
    minimum_support = max(2, int(len(response_train_validation) * 0.02))
    model = PoissonEN_LCM_SPP(
        lam=selected_lambda,
        kappa=0.05,
        min_support=minimum_support,
        max_len=4,
        top_k_add=200,
        max_rounds=12,
        solver_max_iter=800,
        solver_tol=1e-7,
        eta_clip=8.0,
        seed=seed,
        do_ws_screen=True,
        enum_mode_single="both",
        enum_mode_two="both",
    )
    print(
        f"  {drug_class}/{drug} model=DIPS-PR fitting "
        f"lambda={selected_lambda:g}",
        flush=True,
    )
    started = time.perf_counter()
    result = model.fit(
        transactions_train_validation,
        response_train_validation,
        verbose=False,
    )
    design_test = (
        build_X_trans(transactions_test, result.patterns)
        if result.patterns
        else np.zeros((len(transactions_test), 0), dtype=float)
    )
    eta = np.full(len(response_test), result.params.b, dtype=float)
    if design_test.shape[1]:
        eta += design_test @ result.params.w
    prediction = np.exp(np.clip(eta, -model.eta_clip, model.eta_clip))
    nonzero = [
        index
        for index, weight in enumerate(result.params.w)
        if abs(float(weight)) > 1e-8
    ]
    payload = {
        "model": "DIPS-PR",
        "cv_rmse": None,
        "best_params": {
            "lambda": selected_lambda,
            "kappa": 0.05,
            "min_support": minimum_support,
            "max_pattern_length": 4,
        },
        "metrics": regression_metrics(response_test, prediction),
        "prediction": prediction,
        "n_patterns_total": len(result.patterns),
        "n_patterns_nonzero": len(nonzero),
        "n_patterns_multi_nonzero": sum(
            len(result.patterns[index].itemset) > 1 for index in nonzero
        ),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    return payload, prediction


def bootstrap_intervals(
    response: np.ndarray,
    predictions: dict[str, np.ndarray],
    seed: int,
    repetitions: int,
) -> tuple[dict[str, object], dict[str, object]]:
    rng = np.random.default_rng(seed)
    names = list(predictions)
    sampled_metrics = {
        name: {metric: [] for metric in ("R2", "PseudoR2", "RMSE")}
        for name in names
    }
    paired = {
        name: {metric: [] for metric in ("R2", "PseudoR2", "RMSE")}
        for name in names
        if name != "DIPS-PR"
    }
    for _ in range(repetitions):
        indices = rng.integers(0, len(response), size=len(response))
        y_sample = response[indices]
        values = {
            name: regression_metrics(y_sample, prediction[indices])
            for name, prediction in predictions.items()
        }
        for name in names:
            for metric in sampled_metrics[name]:
                sampled_metrics[name][metric].append(values[name][metric])
        dips_values = values["DIPS-PR"]
        for name in paired:
            for metric in paired[name]:
                difference = (
                    values[name][metric] - dips_values[metric]
                    if metric == "RMSE"
                    else dips_values[metric] - values[name][metric]
                )
                paired[name][metric].append(difference)

    intervals: dict[str, object] = {}
    for name, metrics in sampled_metrics.items():
        intervals[name] = {
            metric: {
                "lower": float(np.nanpercentile(values, 2.5)),
                "upper": float(np.nanpercentile(values, 97.5)),
            }
            for metric, values in metrics.items()
        }
    paired_intervals: dict[str, object] = {}
    for name, metrics in paired.items():
        paired_intervals[name] = {
            metric: {
                "lower": float(np.nanpercentile(values, 2.5)),
                "upper": float(np.nanpercentile(values, 97.5)),
                "positive_fraction": float(np.mean(np.asarray(values) > 0.0)),
            }
            for metric, values in metrics.items()
        }
    return intervals, paired_intervals


def run_endpoint(
    drug_class: str,
    drug: str,
    response_matrix: np.ndarray,
    transactions: list[list[int]],
    drug_index: int,
    seed: int,
    jobs: int,
    bootstrap_repetitions: int,
) -> dict[str, object]:
    splits = split_endpoint(response_matrix, transactions, drug_index, seed)
    all_items = [item for transaction in transactions for item in transaction]
    n_items = max(all_items) + 1 if all_items else 1
    matrices = {
        name: item_matrix(split_transactions, n_items)
        for name, (split_transactions, _) in splits.items()
    }
    y_train = splits["train"][1]
    y_validation = splits["validation"][1]
    y_test = splits["test"][1]
    x_train = matrices["train"]
    x_validation = matrices["validation"]
    x_test = matrices["test"]
    x_train_validation = np.vstack([x_train, x_validation])
    y_train_validation = np.concatenate([y_train, y_validation])
    splitter = KFold(n_splits=3, shuffle=True, random_state=seed)
    models: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    for name, (estimator, grid) in model_specs(seed).items():
        print(f"  {drug_class}/{drug} model={name} tuning", flush=True)
        started = time.perf_counter()
        search = GridSearchCV(
            clone(estimator),
            grid,
            scoring="neg_root_mean_squared_error",
            cv=splitter,
            n_jobs=jobs,
            refit=True,
            error_score="raise",
        )
        search.fit(x_train, y_train)
        selected = clone(estimator).set_params(**search.best_params_)
        selected.fit(x_train_validation, y_train_validation)
        prediction = selected.predict(x_test)
        models.append(
            {
                "model": name,
                "cv_rmse": float(-search.best_score_),
                "best_params": search.best_params_,
                "metrics": regression_metrics(y_test, prediction),
                "prediction": prediction,
                "elapsed_seconds": float(time.perf_counter() - started),
            }
        )
        predictions[name] = prediction

    dips_payload, dips_prediction = fit_dips(
        drug_class,
        drug,
        splits["train"][0] + splits["validation"][0],
        y_train_validation,
        splits["test"][0],
        y_test,
        seed,
    )
    models.append(dips_payload)
    predictions["DIPS-PR"] = dips_prediction
    intervals, paired_intervals = bootstrap_intervals(
        y_test,
        predictions,
        seed + 100000,
        bootstrap_repetitions,
    )

    return {
        "dataset": drug_class,
        "drug": drug,
        "seed": seed,
        "split": "80/10/10",
        "n_train": len(y_train),
        "n_validation": len(y_validation),
        "n_test": len(y_test),
        "response_transform": "none",
        "response": "original_positive_fold_change",
        "bootstrap_repetitions": bootstrap_repetitions,
        "bootstrap_scope": "held_out_test_samples_conditional_on_fitted_models",
        "confidence_intervals": intervals,
        "paired_differences_vs_dips": paired_intervals,
        "y_test": y_test,
        "models": models,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--drug-class", choices=DRUG_CLASSES, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/hiv"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.data_dir / f"{args.drug_class}_DataSet.txt"
    drug_names, response_matrix, transactions = parse_dataset(dataset_path)
    class_directory = args.output_dir / args.drug_class
    class_directory.mkdir(parents=True, exist_ok=True)
    for drug in DRUG_CLASSES[args.drug_class]:
        output_path = class_directory / f"{drug}.json"
        if args.resume and output_path.exists():
            print(f"skip completed {args.drug_class}/{drug}", flush=True)
            continue
        if drug not in drug_names:
            print(f"skip missing {args.drug_class}/{drug}", flush=True)
            continue
        drug_index = drug_names.index(drug)
        valid_count = int(np.sum(~np.isnan(response_matrix[:, drug_index])))
        if valid_count < 30:
            print(f"skip {args.drug_class}/{drug}: n={valid_count}", flush=True)
            continue
        result = run_endpoint(
            args.drug_class,
            drug,
            response_matrix,
            transactions,
            drug_index,
            args.seed,
            args.jobs,
            args.bootstrap_repetitions,
        )
        atomic_write_json(output_path, result)
        print(f"  saved={output_path}", flush=True)


if __name__ == "__main__":
    main()
