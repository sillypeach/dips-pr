#!/usr/bin/env python3
"""Count a frozen PATH dictionary exactly, without fitting or reading responses.

The integer-bitset traversal reproduces LCMEnumerator.iter_patterns(None).
M counts yielded eligible patterns; B counts attempted child intersections,
including support/canonical failures. Support propagation changes B, not M.
No u/v screening, predictions, lambda choices, or response values are used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from datetime import datetime, timezone

VERSION = "path-dictionary-counts-v1"
STAT_KEYS = ("nodes_visited", "pruned_support", "pruned_v", "pruned_canonical",
             "screened_u", "emitted_closed", "pruned_dual", "pruned_topk")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value, *, sort_keys=False):
    return hashlib.sha256(json.dumps(value, separators=(",", ":"),
        sort_keys=sort_keys, allow_nan=False).encode("utf-8")).hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Path escapes frozen root: " + str(relative))
    return path


def count_dictionary(transactions, min_support, max_len, pattern_space="closed",
                     propagate_support=True, timeout=None):
    """Return exact M/B/stats, or status=timeout with explicitly partial counts.

    Item identifiers may be any integers, including nonconsecutive/negative
    identifiers. Duplicate items within one transaction have no effect. Prefix
    and transaction sets are Python integer bitmasks; no numeric packages or
    materialized pattern dictionary are needed.
    """
    if (type(min_support) is not int or min_support < 1 or
            type(max_len) is not int or max_len < 1):
        raise ValueError("min_support and max_len must be positive integers")
    if pattern_space not in ("all", "closed") or type(propagate_support) is not bool:
        raise ValueError("Invalid pattern_space/propagate_support")
    if timeout is not None and (not math.isfinite(timeout) or timeout < 0):
        raise ValueError("timeout must be finite and nonnegative")
    n = len(transactions)
    if n < 1:
        raise ValueError("At least one transaction is required")
    started = time.perf_counter()
    deadline = None if timeout is None else started + timeout
    stats = dict.fromkeys(STAT_KEYS, 0)
    eligible = 0

    def check_time():
        if deadline is not None and time.perf_counter() >= deadline:
            raise TimeoutError

    try:
        check_time()
        item_masks = {}
        for row_index, transaction in enumerate(transactions):
            if row_index % 1024 == 0:
                check_time()
            for item in transaction:
                if type(item) is not int:
                    raise ValueError("Transaction items must be integers")
                item_masks[item] = item_masks.get(item, 0) | (1 << row_index)
        # Rank order is identical to the frozen enumerator's sorted item IDs.
        masks = tuple(item_masks[item] for item in sorted(item_masks))
        item_bits = tuple(1 << rank for rank in range(len(masks)))
        all_items = tuple(range(len(masks)))

        def frequent_children(prefix, mask, last, extensions):
            for item in extensions:
                if prefix & item_bits[item] or item <= last:
                    continue
                child_mask = mask & masks[item]
                stats["nodes_visited"] += 1
                if stats["nodes_visited"] % 1024 == 0:
                    check_time()
                if child_mask.bit_count() < min_support:
                    stats["pruned_support"] += 1
                    continue
                yield item, child_mask

        stack = [(0, (1 << n) - 1, -1, all_items)]
        while stack:
            check_time()
            prefix, mask, last, extensions = stack.pop()
            if prefix.bit_count() >= max_len:
                continue
            children = frequent_children(prefix, mask, last, extensions)
            if propagate_support:
                children = list(children)
                # Keep support-valid extensions even if canonicality fails.
                extensions = tuple(item for item, _ in children)
            for item, child_mask in children:
                closed = prefix | item_bits[item]
                if pattern_space == "closed":
                    canonical = True
                    for other in all_items:
                        if prefix & item_bits[other] or other == item:
                            continue
                        if child_mask & masks[other] == child_mask:
                            if other < item:
                                canonical = False
                                break
                            closed |= item_bits[other]
                    if not canonical or closed.bit_count() > max_len:
                        stats["pruned_canonical"] += 1
                        continue
                eligible += 1
                # Continue after the generator item, not max(closure).
                stack.append((closed, child_mask, item, extensions))
        status = "complete"
    except TimeoutError:
        status = "timeout"
    result = dict(status=status, eligible_patterns=eligible,
                  nodes_visited=stats["nodes_visited"], stats=stats,
                  seconds=time.perf_counter() - started)
    if status == "complete":
        validate_counts(result)
    else:
        result["partial_counts"] = True
    return result


def validate_counts(result):
    """Reject incomplete or internally inconsistent denominator records."""
    if result.get("status") != "complete":
        raise ValueError("Only complete counts can be used as denominators")
    stats = result.get("stats", {})
    if set(stats) != set(STAT_KEYS):
        raise ValueError("Invalid enumeration statistics schema")
    if any(type(value) is not int or value < 0 for value in stats.values()):
        raise ValueError("Enumeration statistics must be nonnegative integers")
    for key in ("eligible_patterns", "nodes_visited"):
        if type(result.get(key)) is not int or result[key] < 0:
            raise ValueError("Invalid count: " + key)
    if any(stats[key] for key in ("pruned_v", "screened_u", "emitted_closed",
                                  "pruned_dual", "pruned_topk")):
        raise ValueError("A bare enumeration cannot contain screening events")
    if (result["nodes_visited"] != stats["nodes_visited"] or
            result["eligible_patterns"] != stats["nodes_visited"] -
            stats["pruned_support"] - stats["pruned_canonical"]):
        raise ValueError("M = B - support - canonical count identity failed")
    if not isinstance(result.get("seconds"), (int, float)) or not (
            math.isfinite(result["seconds"]) and result["seconds"] >= 0):
        raise ValueError("Invalid count duration")


def load_protocol(root):
    """Verify registration and every frozen source/input hash once per CLI run."""
    root = Path(root).resolve()
    plan = load_json(root / "protocol.json")
    protocol_sha256 = sha256_file(root / "protocol.json")
    registered = load_json(root / "registered.json")
    if registered["protocol_sha256"] != protocol_sha256:
        raise ValueError("Changed registered protocol")
    if "source/model.py" not in plan["hashes"]:
        raise ValueError("Frozen model hash is missing")
    for relative, expected in plan["hashes"].items():
        if sha256_file(safe_path(root, relative)) != expected:
            raise ValueError("Changed registered source/input: " + relative)
    return plan, protocol_sha256


def dataset_inputs(root, plan, protocol_sha256, dataset_id, split):
    """Bind one training dictionary to saved rows, folds and frozen configuration."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", dataset_id) or dataset_id in (".", ".."):
        raise ValueError("Invalid dataset identifier")
    entries = [entry for entry in plan["datasets"] if entry["dataset_id"] == dataset_id]
    if len(entries) != 1:
        raise ValueError("Dataset must identify exactly one registered entry")
    entry = entries[0]
    for key, digest_key in (("data", "sha256"), ("folds", "folds_sha256")):
        expected = entry[digest_key]
        if (plan["hashes"].get(entry[key]) != expected or
                sha256_file(safe_path(root, entry[key])) != expected):
            raise ValueError("Changed or inconsistently registered " + key)
    data = load_json(safe_path(root, entry["data"]))
    folds = load_json(safe_path(root, entry["folds"]))
    if folds.get("data_sha256") != entry["sha256"] or folds.get("dataset_id") != dataset_id:
        raise ValueError("Saved folds do not belong to this dataset")
    transactions = data["transactions"]
    train, test = data["train_indices"], data["test_indices"]
    if (any(type(index) is not int for index in train + test) or
            sorted(train + test) != list(range(len(transactions)))):
        raise ValueError("Outer indices must partition all transactions")
    if split == "train":
        rows = train
    elif re.fullmatch(r"fold[0-9]+", split):
        index = int(split[4:])
        if index >= len(folds["folds"]):
            raise ValueError("Requested fold does not exist")
        inner, validation = folds["folds"][index]
        if (any(type(i) is not int for i in inner + validation) or
                sorted(inner + validation) != list(range(len(train)))):
            raise ValueError("Invalid saved training fold")
        rows = [train[i] for i in inner]
    else:
        raise ValueError("split must be train or fold0/fold1/...")
    cfg = plan["config"]
    config = dict(min_support=max(int(cfg["support_minimum"]),
        math.ceil(float(cfg["support_fraction"]) * len(rows))),
        max_len=int(entry["maximum_true_length"]),
        pattern_space=cfg["pattern_space"], propagate_support=cfg["propagate_support"])
    provenance = dict(version=VERSION, dataset_id=dataset_id, split=split,
        protocol_sha256=protocol_sha256, data_sha256=entry["sha256"],
        folds_sha256=entry["folds_sha256"],
        source_model_sha256=plan["hashes"]["source/model.py"],
        row_indices_sha256=json_sha256(rows), row_count=len(rows), config=config,
        config_sha256=json_sha256(config, sort_keys=True))
    return provenance, [transactions[i] for i in rows]


def validate_cache(record, provenance):
    for key, value in provenance.items():
        if record.get(key) != value:
            raise ValueError("Existing count cache provenance mismatch: " + key)
    validate_counts(record)


def save_atomic(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            tmp = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(tmp, path)
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def count_registered(root, plan, protocol_sha256, dataset_id, split="train",
                     timeout=None, retry_incomplete=False):
    provenance, transactions = dataset_inputs(root, plan, protocol_sha256, dataset_id, split)
    path = safe_path(root, f"analysis/dictionary_counts/{dataset_id}/{split}.json")
    if path.exists():
        old = load_json(path)
        # Changed provenance is never overwritten, including failed counts.
        for key, value in provenance.items():
            if old.get(key) != value:
                raise ValueError("Existing count cache provenance mismatch: " + key)
        if old.get("status") == "complete":
            validate_cache(old, provenance)
            return old, True
        if not retry_incomplete:
            raise ValueError("Existing incomplete cache is not reusable; explicitly pass --retry-incomplete")
    result = count_dictionary(transactions, **provenance["config"], timeout=timeout)
    record = dict(provenance, **result,
        created_at=datetime.now(timezone.utc).isoformat(),
        counter_sha256=sha256_file(__file__), timeout_seconds=timeout,
        scope="One exact enumeration with support/closure/length/propagated-support constraints and no u/v screening; no fitting")
    save_atomic(path, record)
    return record, False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--dataset", nargs="+", action="append", required=True,
                        help="Explicit registered ID(s); repeat or provide a list. No implicit all-dataset run.")
    parser.add_argument("--split", nargs="+", action="append", help="train (default), fold0, fold1, ...")
    parser.add_argument("--timeout", type=float, help="Seconds per dictionary; partial counts never have complete status")
    parser.add_argument("--retry-incomplete", action="store_true",
                        help="Explicitly recount an existing timeout/incomplete cache with matching provenance")
    args = parser.parse_args(argv)
    plan, protocol_sha256 = load_protocol(args.root)
    datasets = list(dict.fromkeys(item for group in args.dataset for item in group))
    splits = list(dict.fromkeys(item for group in (args.split or [["train"]]) for item in group))
    failed = False
    for dataset_id in datasets:
        for split in splits:
            try:
                result, reused = count_registered(args.root, plan, protocol_sha256, dataset_id,
                    split, args.timeout, args.retry_incomplete)
                print(json.dumps(dict(dataset_id=dataset_id, split=split,
                    status=result["status"], reused=reused,
                    eligible_patterns=result["eligible_patterns"],
                    nodes_visited=result["nodes_visited"], seconds=result["seconds"])), flush=True)
                failed |= result["status"] != "complete"
            except (ValueError, KeyError, OSError) as error:
                failed = True
                print(json.dumps(dict(dataset_id=dataset_id, split=split,
                    status="failed", error=str(error))), flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, KeyError, OSError) as error:
        print("Dictionary count validation failed: " + str(error), file=sys.stderr)
        sys.exit(1)
