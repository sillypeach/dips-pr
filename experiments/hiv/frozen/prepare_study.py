"""Freeze five predeclared SeqID splits from the 25 previous frozen datasets.

No estimators are imported or fitted, and no previous fitted result is read.
Only the new study root is written; an existing registration is never replaced.
The caller must finish installing source/*.py and audit/model.py before freezing.
"""
from __future__ import annotations

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

import numpy as np
import sklearn
from sklearn.model_selection import GroupShuffleSplit, KFold


OUTER_SPLIT_SEEDS = (42, 43, 44, 45, 46)
MODEL_SEED = 42
BASELINES = ("RandomForest", "Ridge", "Lasso", "ElasticNet", "PoissonGLM", "RBF-SVR", "MLP")
GRID_COUNTS = dict(RandomForest=18, Ridge=4, Lasso=3, ElasticNet=9,
                   PoissonGLM=4, **{"RBF-SVR": 12}, MLP=9)
NRTI_ORDER = {drug: i for i, drug in enumerate(("AZT", "3TC", "ABC", "D4T", "DDI", "TDF"))}
CORE_DATA_KEYS = ("train_indices", "validation_indices", "test_indices",
                  "tx_train", "tx_final", "tx_test", "y_train", "y_final", "y_test",
                  "groups_train", "groups_final", "groups_test")
CORE_FOLD_KEYS = ("folds", "source_row_folds", "group_folds")


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def value_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def verified_files(root, files):
    root = Path(root).resolve()
    for relative, expected in files.items():
        path = (root / relative).resolve()
        check(not Path(relative).is_absolute() and path.is_relative_to(root),
              "Invalid registered path: " + relative)
        check(sha(path) == expected, "Changed registered file: " + str(path))


def baseline_grid_counts(source_path):
    """Read only literal grids from model_specs; do not execute training imports."""
    tree = ast.parse(Path(source_path).read_text())
    func = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                and node.name == "model_specs")
    returns = [node for node in func.body if isinstance(node, ast.Return)]
    check(len(returns) == 1 and isinstance(returns[0].value, ast.Dict),
          "Expected an explicit model_specs dictionary")
    result = {}
    for key, value in zip(returns[0].value.keys, returns[0].value.values):
        name = ast.literal_eval(key)
        grid = ast.literal_eval(value.elts[1])
        result[name] = math.prod(len(values) for values in grid.values())
    check(result == GRID_COUNTS and sum(result.values()) == 59,
          "The seven original baseline grids must contain 59 candidates")
    return result


def restore_rows(data):
    """Return unique source rows in original order, preserving saved values."""
    check(not data["validation_indices"], "Previous dataset has an unused validation layer")
    for prefix in ("tx_", "y_", "groups_"):
        check(data[prefix + "train"] == data[prefix + "final"],
              "Previous train and final arrays differ: " + prefix)
    rows = []
    for split, indices in (("final", data["train_indices"]), ("test", data["test_indices"])):
        columns = [indices, data["tx_" + split], data["y_" + split], data["groups_" + split]]
        check(len({len(column) for column in columns}) == 1, "Misaligned previous rows")
        rows.extend([row_id, tx, y, group] for row_id, tx, y, group in zip(*columns))
    check(all(type(row[0]) is int and row[0] >= 0 for row in rows), "Invalid source row id")
    check(len({row[0] for row in rows}) == len(rows), "Duplicate source row id")
    check(all(isinstance(row[3], str) and row[3] for row in rows), "Empty or non-string SeqID")
    return sorted(rows, key=lambda row: row[0])


def grouped_split(groups, seed):
    check(seed in OUTER_SPLIT_SEEDS, "Outer seed was not predeclared")
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    check(len(unique) >= 4, "At least four SeqID groups are required")
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=.1, random_state=seed)
                       .split(np.empty((len(groups), 0)), groups=groups))
    expected = set(unique[np.random.RandomState(seed).permutation(len(unique))
                          [:math.ceil(.1 * len(unique))]])
    check(set(groups[test]) == expected, "Unexpected group-selection implementation")
    check(not set(groups[train]) & set(groups[test]), "Outer SeqID overlap")
    return train.tolist(), test.tolist()


def grouped_folds(groups, seed):
    groups = np.asarray(groups, dtype=str)
    unique = np.unique(groups)
    check(len(unique) >= 3, "Three-fold CV needs at least three training groups")
    folds = []
    for fit, valid in KFold(3, shuffle=True, random_state=seed).split(unique):
        fit_rows = np.flatnonzero(np.isin(groups, unique[fit])).tolist()
        valid_rows = np.flatnonzero(np.isin(groups, unique[valid])).tolist()
        check(not set(groups[fit_rows]) & set(groups[valid_rows]), "CV SeqID overlap")
        check(sorted(fit_rows + valid_rows) == list(range(len(groups))), "Incomplete CV partition")
        folds.append([fit_rows, valid_rows])
    check(sorted(i for _, valid in folds for i in valid) == list(range(len(groups))),
          "CV validation coverage differs from training rows")
    return folds


def prepare_split(previous_data, seed):
    """Pure data/fold reconstruction, suitable for all 125 local checks."""
    rows = restore_rows(previous_data)
    train, test = grouped_split([row[3] for row in rows], seed)
    data = copy.deepcopy(previous_data)
    data.update(train_indices=[rows[i][0] for i in train], validation_indices=[],
                test_indices=[rows[i][0] for i in test])
    for split, indices in (("train", train), ("final", train), ("test", test)):
        for column, prefix in ((1, "tx_"), (2, "y_"), (3, "groups_")):
            data[prefix + split] = [rows[i][column] for i in indices]
    folds = grouped_folds(data["groups_train"], seed)
    fold_payload = dict(seed=seed, outer_split_seed=seed, cv_seed=seed, model_seed=MODEL_SEED,
        index_scope="relative to data.tx_train / y_train / groups_train",
        method=f"KFold on sorted unique training SeqIDs, shuffle=True, random_state={seed}; then map groups to rows",
        folds=folds,
        source_row_folds=[[[data["train_indices"][i] for i in indices] for indices in pair] for pair in folds],
        group_folds=[[sorted({data["groups_train"][i] for i in indices}) for indices in pair] for pair in folds])
    check(restore_rows(data) == rows, "Split changed source rows, encoding, response or SeqID")
    return data, fold_payload


def prepare_metadata(previous_metadata, data, dataset_id, seed):
    metadata = copy.deepcopy(previous_metadata)
    by_row = {}
    for split in ("train", "test"):
        rows = metadata[split + "_source_rows"]
        for i, row in enumerate(rows):
            check(row not in by_row, "Repeated metadata source row")
            by_row[row] = {suffix: metadata[split + "_" + suffix][i]
                          for suffix in ("sequence_ids", "ptid_candidates", "source_types")}
    for split in ("train", "test"):
        rows = data[split + "_indices"]
        metadata[split + "_source_rows"] = rows
        for suffix in ("sequence_ids", "ptid_candidates", "source_types"):
            metadata[split + "_" + suffix] = [by_row[row][suffix] for row in rows]
        check(metadata[split + "_sequence_ids"] == data["groups_" + split], "Metadata SeqID mismatch")
    metadata.update(case_id=dataset_id, dataset_id=dataset_id,
        endpoint=metadata["drug_class"] + "/" + metadata["drug"],
        seed=MODEL_SEED, model_seed=MODEL_SEED, outer_split_seed=seed, cv_seed=seed,
        encoding_mapping_file="feature_mapping.json",
        seed_policy="seed/model_seed fix estimator randomness at 42; outer_split_seed/cv_seed vary together")
    return metadata


def overlap(left, right, seqmeta):
    def patients(groups, clinical=False):
        return {patient for group in groups for patient in seqmeta[group][0]
                if not clinical or "Clinical" in seqmeta[group][1]}
    return dict(seqids=len(set(left) & set(right)),
        all_ptid_keys=len(patients(left) & patients(right)),
        clinical_ptid_keys=len(patients(left, True) & patients(right, True)))


def prepare_audit(previous_audit, data, folds, metadata, parent_hashes):
    seqmeta = {}
    for split in ("train", "test"):
        for group, patients, types in zip(data["groups_" + split], metadata[split + "_ptid_candidates"],
                                           metadata[split + "_source_types"]):
            value = (patients, types)
            check(group not in seqmeta or seqmeta[group] == value, "Conflicting SeqID metadata")
            seqmeta[group] = value
    groups = data["groups_train"]
    audit = dict(case_id=metadata["case_id"], endpoint=metadata["endpoint"],
        outer_split_seed=metadata["outer_split_seed"], cv_seed=metadata["cv_seed"], model_seed=MODEL_SEED,
        n_rows=len(groups) + len(data["groups_test"]), n_train=len(groups), n_final=len(groups),
        n_test=len(data["groups_test"]), n_train_groups=len(set(groups)),
        n_test_groups=len(set(data["groups_test"])), unique_seqids=len(seqmeta),
        outer_overlap=overlap(groups, data["groups_test"], seqmeta),
        folds=[], parent_frozen_input_sha256=parent_hashes,
        reconstructed_row_content_sha256=value_sha(restore_rows(data)),
        all_source_rows_responses_transactions_groups_preserved=True,
        response=previous_audit["response"], historical=previous_audit.get("historical"),
        no_response_transform=True, all_published_repeated_rows_retained=True)
    for fit, valid in folds["folds"]:
        fitgroups, validgroups = [[groups[i] for i in indices] for indices in (fit, valid)]
        audit["folds"].append(dict(n_fit=len(fit), n_validation=len(valid),
            n_fit_groups=len(set(fitgroups)), n_validation_groups=len(set(validgroups)),
            overlap=overlap(fitgroups, validgroups, seqmeta)))
    encoded = {tuple(row) for row in data["tx_train"]}
    audit["test_rows_with_encoded_vector_seen_in_train"] = sum(tuple(row) in encoded for row in data["tx_test"])
    return audit


def endpoint_priority(entry):
    cls, drug = entry["endpoint"].split("/")
    return (0, NRTI_ORDER[drug]) if cls == "NRTI" else (1, entry["endpoint"])


def split_fingerprint(data, folds):
    """Portable cross-host evidence containing hashes/counts, never observations."""
    rows = restore_rows(data)
    return dict(n_rows=len(rows), n_train=len(data["y_train"]), n_test=len(data["y_test"]),
        n_groups=len({row[3] for row in rows}), n_train_groups=len(set(data["groups_train"])),
        n_test_groups=len(set(data["groups_test"])),
        source_records_sha256=value_sha(rows),
        data_core_sha256=value_sha({key: data[key] for key in CORE_DATA_KEYS}),
        folds_core_sha256=value_sha({key: folds[key] for key in CORE_FOLD_KEYS}))


def reconstruction_fingerprints(previous):
    """Reconstruct all splits without writing, returning only hashes and counts."""
    previous = Path(previous).resolve()
    study = read(previous / "study.json")
    entries = sorted(study["datasets"], key=endpoint_priority)
    check(len(entries) == 25, "Expected 25 previous endpoints")
    results = []
    for entry in entries:
        old_root = previous / entry["root"]
        original, old_folds = read(old_root / "data.json"), read(old_root / "folds.json")
        memberships = set()
        for seed in OUTER_SPLIT_SEEDS:
            data, folds = prepare_split(original, seed)
            membership = tuple(sorted(set(data["groups_test"])))
            check(membership not in memberships, "Repeated outer membership")
            memberships.add(membership)
            if seed == 42:
                check(data == original, "Seed42 data mismatch")
                check(all(folds[key] == old_folds[key] for key in CORE_FOLD_KEYS), "Seed42 fold mismatch")
            results.append(dict(dataset_id=entry["dataset_id"] + f"__seed_{seed}",
                original_dataset_id=entry["dataset_id"], endpoint=entry["endpoint"],
                outer_split_seed=seed, **split_fingerprint(data, folds)))
    return dict(version="hiv-repeated-split-reconstruction-fingerprints-v1",
        parent_study_sha256=sha(previous / "study.json"), datasets=results,
        aggregate_sha256=value_sha(results))


def freeze(root, previous, cpus):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    check(root != previous and not root.is_relative_to(previous) and not previous.is_relative_to(root),
          "New and previous study roots must be separate")
    check(cpus and len(cpus) == len(set(cpus)) and all(type(cpu) is int and cpu >= 0 for cpu in cpus),
          "CPU allocation must be a nonempty unique list of nonnegative integers")
    check(not (root / "study.json").exists() and not (root / "registered.json").exists(),
          "Study is already registered; refusing to overwrite")
    check(not (root / "repeats").exists() and not (root / "data_manifest.json").exists(),
          "Prepared output already exists; inspect the partial attempt instead of automatically retrying")
    old_study = read(previous / "study.json")
    check(sha(previous / "study.json") == read(previous / "registered.json")["study_sha256"],
          "Previous study registration changed")
    verified_files(previous, old_study["files"])
    old_entries = sorted(old_study["datasets"], key=endpoint_priority)
    check(len(old_entries) == 25 and len({e["endpoint"] for e in old_entries}) == 25,
          "Expected exactly 25 distinct previous endpoints")
    source_files = sorted((root / "source").rglob("*.py"))
    check(source_files and (root / "audit/model.py").is_file(), "Install the new source and audit first")
    required = {"hiv_shared.py", "hiv_baseline.py", "hiv_dips.py", "dispatch.py", "run_study.py",
                "analyze_endpoint.py", "aggregate_repeats.py", "hiv_nonlinear_baselines.py", "model.py"}
    check(required <= {p.name for p in source_files}, "Incomplete new source snapshot")
    grid_counts = baseline_grid_counts(previous / "source/hiv_nonlinear_baselines.py")
    check(baseline_grid_counts(root / "source/hiv_nonlinear_baselines.py") == grid_counts,
          "New baseline candidate grids differ")
    parent_sources = {name: digest for name, digest in old_study["files"].items()
                      if name.startswith("source/") or name.startswith("audit/")}
    source_hashes = {str(path.relative_to(root)): sha(path) for path in source_files + [root / "audit/model.py"]}
    date = datetime.now(timezone.utc).isoformat()
    entries, manifest_entries = [], []
    parent_inputs = {}
    seed42_checks = []
    # Endpoint-major order interleaves all five seeds, keeping all NRTIs first.
    for old_entry in old_entries:
        old_root = previous / old_entry["root"]
        old_protocol = read(old_root / "protocol.json")
        check(sha(old_root / "protocol.json") == old_entry["protocol_sha256"], "Parent protocol mismatch")
        old_data, old_folds = read(old_root / "data.json"), read(old_root / "folds.json")
        old_meta, old_audit = read(old_root / "metadata.json"), read(old_root / "data_audit.json")
        hashes = {name: sha(old_root / name) for name in
                  ("protocol.json", "data.json", "folds.json", "metadata.json", "data_audit.json", "feature_mapping.json")}
        parent_inputs[old_entry["dataset_id"]] = dict(root=old_entry["root"], files=hashes,
            reconstructed_row_content_sha256=value_sha(restore_rows(old_data)))
        test_memberships = set()
        for seed in OUTER_SPLIT_SEEDS:
            endpoint_id = old_entry["dataset_id"]
            dataset_id = endpoint_id + f"__seed_{seed}"
            eroot = root / "repeats" / f"seed_{seed}" / "endpoints" / endpoint_id
            data, folds = prepare_split(old_data, seed)
            if seed == 42:
                check(all(data[k] == old_data[k] for k in CORE_DATA_KEYS), "Seed42 data differs: " + endpoint_id)
                check(all(folds[k] == old_folds[k] for k in CORE_FOLD_KEYS), "Seed42 folds differ: " + endpoint_id)
                seed42_checks.append(endpoint_id)
            membership = tuple(sorted(set(data["groups_test"])))
            check(membership not in test_memberships, "Repeated outer test membership: " + endpoint_id)
            test_memberships.add(membership)
            meta = prepare_metadata(old_meta, data, dataset_id, seed)
            audit = prepare_audit(old_audit, data, folds, meta, hashes)
            eroot.mkdir(parents=True)
            for source_path in source_files + [root / "audit/model.py"]:
                destination = eroot / source_path.relative_to(root)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, destination)
            shutil.copyfile(old_root / "feature_mapping.json", eroot / "feature_mapping.json")
            for name, value in (("data.json", data), ("folds.json", folds), ("metadata.json", meta), ("data_audit.json", audit)):
                write_new(eroot / name, value)
            protocol = copy.deepcopy(old_protocol)
            protocol.update(version="hiv-seqid-grouped-repeated-split-v1", state="frozen",
                dataset_id=dataset_id, endpoint=old_entry["endpoint"], seed=MODEL_SEED,
                outer_split_seed=seed, cv_seed=seed, model_seed=MODEL_SEED,
                n_train=len(data["y_train"]), n_final=len(data["y_final"]), n_test=len(data["y_test"]),
                baselines=list(BASELINES), baseline_grid_counts=grid_counts,
                baseline_grid="Original seven HIV grids, 59 candidates; mean three-fold RMSE; first ParameterGrid index resolves ties. Finite fits remain eligible with recorded warnings. Invalid candidate folds are retained and excluded; no valid candidate or failed final fit fails that method.",
                split=f"GroupShuffleSplit(n_splits=1,test_size=0.1,random_state={seed}) on unique SeqID groups; no response values passed",
                cv=f"Sorted unique outer-training SeqIDs split by KFold(3,shuffle=True,random_state={seed}), then mapped back to every associated source row; minimize mean fold RMSE; exact ties choose largest tau",
                seed_policy="outer_split_seed=cv_seed; legacy seed=model_seed=42 fixes model randomness across repetitions",
                repetition_policy="All five split seeds were fixed before repeated-study results; no result-based seed selection or replacement; all methods refitted independently",
                previous_root=str(previous), original_dataset_id=endpoint_id,
                parent_study_sha256=sha(previous / "study.json"), parent_dataset_id=endpoint_id,
                parent_protocol_sha256=hashes["protocol.json"], parent_frozen_input_sha256=hashes,
                parent_source_sha256=parent_sources, resources=dict(memory_gib=8, threads=1),
                registered_at=date, no_automatic_retry=True)
            check(len(protocol["taus"]) == 101, "Expected the unchanged 101-point path")
            protocol["files"] = {str(path.relative_to(eroot)): sha(path) for path in sorted(eroot.rglob("*")) if path.is_file()}
            write_new(eroot / "protocol.json", protocol)
            digest = sha(eroot / "protocol.json")
            write_new(eroot / "registered.json", dict(protocol_sha256=digest, at=date))
            entry = dict(dataset_id=dataset_id, endpoint=old_entry["endpoint"], endpoint_id=endpoint_id,
                original_dataset_id=endpoint_id,
                root=str(eroot.relative_to(root)), outer_split_seed=seed, cv_seed=seed, model_seed=MODEL_SEED,
                protocol_sha256=digest, baselines=list(BASELINES))
            entries.append(entry)
            manifest_entries.append(dict(entry, n_train=protocol["n_train"], n_final=protocol["n_final"],
                n_test=protocol["n_test"], files=protocol["files"],
                reconstruction=split_fingerprint(data, folds),
                reconstructed_row_content_sha256=audit["reconstructed_row_content_sha256"]))
    check(len(entries) == 125 and len({e["dataset_id"] for e in entries}) == 125, "Expected 125 unique datasets")
    # Detect any concurrent source editing before publishing a study registration.
    verified_files(root, source_hashes)
    manifest = dict(version="hiv-repeated-split-data-v1", endpoint_count=25, dataset_count=125,
        outer_split_seeds=list(OUTER_SPLIT_SEEDS), model_seed=MODEL_SEED,
        cv_seed_policy="cv_seed equals outer_split_seed", parent_study=str(previous), previous_root=str(previous),
        parent_study_sha256=sha(previous / "study.json"), parent_source_sha256=parent_sources,
        parent_inputs=parent_inputs, seed42_equal_previous_endpoints=seed42_checks,
        seed_list_fixed_before_results=True, seed_selection_uses_results=False,
        environment=dict(numpy=np.__version__, sklearn=sklearn.__version__),
        preparation_sha256=sha(Path(__file__)), datasets=manifest_entries)
    write_new(root / "data_manifest.json", manifest)
    study = dict(version="hiv-repeated-split-study-v1", state="frozen", registered_at=date,
        scope="All 25 HIV endpoints freshly fitted under five predeclared SeqID splits; six NRTIs main table and other 19 endpoints appendix; seven baselines",
        datasets=entries, outer_split_seeds=list(OUTER_SPLIT_SEEDS), model_seed=MODEL_SEED,
        previous_root=str(previous),
        cv_seed_policy="cv_seed equals outer_split_seed", baseline_grid_counts=grid_counts,
        resources=dict(cpus=list(cpus), memory_gib=8, threads=1),
        expected=dict(endpoints=25, repeats=5, datasets=125, dips_paths=500, dips_path_points=50500,
            baseline_cv_fits=22125, baseline_final_fits=875, total_models=1000, worker_processes=1375),
        primary_selection="CV-min", secondary_selection="CV-1SE", paper_pr2_key="PseudoR2_train_null",
        parent_study_sha256=sha(previous / "study.json"), parent_source_sha256=parent_sources,
        data_manifest_sha256=sha(root / "data_manifest.json"), seed_list_fixed_before_results=True,
        seed_selection_uses_results=False, no_automatic_retry=True,
        new_results_do_not_replace_historical_artifacts=True, requested_main_class="NRTI",
        appendix_classes=["NNRTI", "PI", "INI", "CAI"])
    registered_paths = source_files + [root / "audit/model.py", root / "data_manifest.json"]
    registered_paths.extend(path for path in sorted((root / "repeats").rglob("*")) if path.is_file())
    registered_paths.extend(path for path in (root / "prepare_study.py", root / "test_prepare_study.py") if path.is_file())
    study["files"] = {str(path.relative_to(root)): sha(path) for path in registered_paths}
    write_new(root / "study.json", study)
    write_new(root / "registered.json", dict(study_sha256=sha(root / "study.json"), at=date))
    print(json.dumps(dict(status="frozen", expected=study["expected"], resources=study["resources"],
        study_sha256=sha(root / "study.json"), registered_files=len(study["files"])), indent=2))
    return study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--previous", required=True, type=Path)
    parser.add_argument("--cpus", required=True, nargs="+", type=int)
    args = parser.parse_args()
    freeze(args.root, args.previous, args.cpus)


if __name__ == "__main__":
    main()
