"""Uno LCM 5.3 closed dictionaries and memory-mapped sparse feature matrices.

Mining accepts only training transactions, never responses or planted patterns.
LCM's complete closed dictionary is restricted to lengths 1..max_len.  The
returned lexicographic pattern list matches CSC columns exactly.  No top-K,
deduplication by transaction support, or post-mining feature filter is applied.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from scipy.sparse import csc_matrix

VERSION = "uno-lcm53-closed-csc-v1"
SOURCE_URL = "https://research.nii.ac.jp/~uno/code/lcm53.zip"
_PATTERN = re.compile(r"^([0-9 ]*)\s*\(([0-9]+)\)\s*$")


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _transactions(transactions):
    rows = []
    for row in transactions:
        items = list(row)
        if any(type(item) is not int or item < 0 for item in items):
            raise ValueError("LCM item IDs must be nonnegative Python integers")
        rows.append(tuple(sorted(set(items))))
    if not rows:
        raise ValueError("At least one transaction is required")
    return rows


def _masks(rows):
    masks = {}
    for index, row in enumerate(rows):
        for item in row:
            masks[item] = masks.get(item, 0) | (1 << index)
    return masks


def _pattern_mask(pattern, masks, n):
    mask = (1 << n) - 1
    for item in pattern:
        mask &= masks.get(item, 0)
    return mask


def _mask_tids(mask, n):
    bits = np.frombuffer(mask.to_bytes((n + 7) // 8, "little"), dtype=np.uint8)
    return np.flatnonzero(np.unpackbits(bits, bitorder="little", count=n))


def _records(path, n, minimum, maximum):
    """Read at most one TID line at once; never make per-nnz Python objects."""
    with Path(path).open() as stream:
        while True:
            line = stream.readline()
            if not line:
                return
            match = _PATTERN.fullmatch(line.rstrip("\n"))
            if match is None:
                raise ValueError("Malformed LCM itemset/support line")
            pattern = tuple(sorted(map(int, match.group(1).split())))
            support = int(match.group(2))
            if (not 1 <= len(pattern) <= maximum or len(set(pattern)) != len(pattern)
                    or not minimum <= support <= n):
                raise ValueError("LCM returned an invalid itemset or frequency")
            line = stream.readline()
            if not line:
                raise ValueError("Missing LCM TID line")
            if not re.fullmatch(r"[0-9 \t\r\n]*", line):
                raise ValueError("Malformed LCM TID line")
            tids = np.fromstring(line, sep=" ", dtype=np.int64)
            if len(tids) != support or np.any(tids < 0) or np.any(tids >= n):
                raise ValueError("LCM TID count/range disagrees with frequency")
            tids.sort()
            if np.any(tids[1:] == tids[:-1]):
                raise ValueError("Duplicate LCM TID")
            yield pattern, support, tids


def _allocate(workdir, n, supports):
    supports = np.asarray(supports, dtype=np.int64)
    nnz = int(supports.sum())
    size = len(supports)
    dtype = np.int32 if max(n, size, nnz) < 2**31 else np.int64
    pointers = np.lib.format.open_memmap(workdir / "indptr.npy", mode="w+",
                                        dtype=dtype, shape=(size + 1,))
    pointers[0] = 0
    np.cumsum(supports, out=pointers[1:])
    # Empty mmap-backed arrays are supported by NumPy's .npy header format.
    indices = np.lib.format.open_memmap(workdir / "indices.npy", mode="w+",
                                       dtype=dtype, shape=(nnz,))
    values = np.lib.format.open_memmap(workdir / "data.npy", mode="w+",
                                      dtype=np.float64, shape=(nnz,))
    values[:] = 1.0
    return pointers, indices, values


def _matrix(pointers, indices, values, n):
    for array in (pointers, indices, values):
        array.flush()
    matrix = csc_matrix((values, indices, pointers), shape=(n, len(pointers) - 1),
                        copy=False)
    matrix.has_sorted_indices = True
    matrix.has_canonical_format = True
    return matrix


def _write_patterns(path, patterns):
    with path.open("w") as stream:
        for pattern in patterns:
            stream.write(" ".join(map(str, pattern)) + "\n")


def _new_workdir(workdir):
    root = Path(workdir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(root.iterdir()):
        raise ValueError("Dictionary workdir must be empty; never overwrite a prior run")
    return root


def _seal(root, receipt):
    receipt["file_sha256"] = {name: _sha(root / name) for name in
                              ("patterns.txt", "data.npy", "indices.npy", "indptr.npy")}
    _save(root / "receipt.json", receipt)


def load_dictionary(workdir, *, mmap_mode="r"):
    """Verify hashes and reopen arrays read-only or private copy-on-write.

    The caller additionally binds this receipt to its dataset/split protocol.
    No unverified or incomplete cache is accepted.  Hashing streams array files
    and does not materialize their contents in memory. ``mmap_mode='c'`` permits
    estimators to modify a private mapping without changing the cached files.
    """
    if mmap_mode not in ("r", "c"):
        raise ValueError("Dictionary mmap_mode must be 'r' or private copy-on-write 'c'")
    root = Path(workdir).resolve()
    receipt = json.loads((root / "receipt.json").read_text())
    if receipt.get("version") != VERSION or receipt.get("status") != "complete":
        raise ValueError("Incomplete or unsupported LCM dictionary cache")
    expected = receipt.get("file_sha256", {})
    if set(expected) != {"patterns.txt", "data.npy", "indices.npy", "indptr.npy"}:
        raise ValueError("Incomplete LCM dictionary cache hashes")
    for name, digest in expected.items():
        if _sha(root / name) != digest:
            raise ValueError("Changed LCM dictionary cache: " + name)
    patterns = [tuple(map(int, line.split()))
                for line in (root / "patterns.txt").read_text().splitlines()]
    values, indices, pointers = [np.load(root / name, mmap_mode=mmap_mode, allow_pickle=False)
                                for name in ("data.npy", "indices.npy", "indptr.npy")]
    if (len(patterns) != receipt["n_patterns"] or len(pointers) != len(patterns) + 1
            or len(values) != receipt["nnz"] or len(indices) != len(values)
            or pointers[0] != 0 or pointers[-1] != len(values)
            or np.any(pointers[1:] < pointers[:-1])):
        raise ValueError("LCM dictionary cache dimensions disagree")
    matrix = csc_matrix((values, indices, pointers),
                       shape=(receipt["n_rows"], len(patterns)), copy=False)
    matrix.has_sorted_indices = True
    matrix.has_canonical_format = True
    return dict(patterns=patterns, matrix=matrix, receipt=receipt)


def mine_closed(transactions, min_support, max_len, binary, workdir, *, timeout=None):
    """Return ``{patterns, matrix, receipt}``; arrays remain backed by workdir.

    patterns: lexicographically ordered list of tuples of original item IDs.
    matrix: float64 CSC, shape=(training rows, number of eligible patterns).
    receipt: executable/input/output hashes, support/length rules and timings.

    ``workdir`` must be empty.  Failure outputs are preserved and are never
    reusable as a complete dictionary.  LCM source is unmodified and external.
    """
    if type(min_support) is not int or min_support < 1:
        raise ValueError("min_support must be a positive integer")
    if type(max_len) is not int or max_len < 1:
        raise ValueError("max_len must be a positive integer")
    binary = Path(binary).resolve()
    if not binary.is_file():
        raise FileNotFoundError(binary)
    rows = _transactions(transactions)
    root = _new_workdir(workdir)
    started = time.perf_counter()
    input_path, output_path = root / "transactions.txt", root / "lcm_output.txt"
    with input_path.open("w", newline="\n") as stream:
        for row in rows:
            stream.write(" ".join(map(str, row)) + "\n")
    command = [str(binary), "CfI", "-l", "1", "-u", str(max_len),
               str(input_path), str(min_support), str(output_path)]
    receipt = dict(version=VERSION, source_url=SOURCE_URL, binary_sha256=_sha(binary),
                   input_sha256=_sha(input_path), n_rows=len(rows),
                   min_support=min_support, max_len=max_len, pattern_space="closed",
                   command=command, item_ids="original, nonnegative, zero permitted",
                   transaction_ids="zero-based, empty transaction rows preserved")
    _save(root / "request.json", receipt)
    try:
        masks = _masks(rows)
        # LCM returns success without an output file when no singleton is
        # frequent; establish that exact empty case from training X alone.
        trivial = not any(mask.bit_count() >= min_support for mask in masks.values())
        with (root / "stdout.log").open("w") as out, (root / "stderr.log").open("w") as err:
            if trivial:
                output_path.write_text("")
                out.write("0\n")
            else:
                subprocess.run(command, check=True, stdout=out, stderr=err, timeout=timeout)
        if not output_path.is_file():
            raise RuntimeError("LCM did not produce an output file")
        native_patterns, native_supports = [], []
        for pattern, support, _ in _records(output_path, len(rows), min_support, max_len):
            native_patterns.append(pattern)
            native_supports.append(support)
        count_lines = (root / "stdout.log").read_text().splitlines()
        # Official LCM prints no count histogram for a zero-output search.
        declared_count = int(count_lines[0]) if count_lines else 0
        if declared_count != len(native_patterns):
            raise ValueError("LCM declared count differs from parsed complete dictionary")
        order = sorted(range(len(native_patterns)), key=native_patterns.__getitem__)
        patterns = [native_patterns[i] for i in order]
        if any(a == b for a, b in zip(patterns, patterns[1:])):
            raise ValueError("LCM emitted duplicate patterns")
        supports = np.asarray(native_supports, dtype=np.int64)[order]
        inverse = np.empty(len(order), dtype=np.int64)
        inverse[order] = np.arange(len(order))
        pointers, indices, values = _allocate(root, len(rows), supports)
        audit_columns = set(np.linspace(0, len(patterns) - 1,
                                       min(64, len(patterns)), dtype=int).tolist())
        for native_index, (pattern, _, tids) in enumerate(
                _records(output_path, len(rows), min_support, max_len)):
            column = int(inverse[native_index])
            if pattern != patterns[column]:
                raise ValueError("LCM file changed between parser passes")
            indices[pointers[column]:pointers[column + 1]] = tids
            if column in audit_columns:
                expected = _mask_tids(_pattern_mask(pattern, masks, len(rows)), len(rows))
                if not np.array_equal(tids, expected):
                    raise ValueError("LCM TIDs disagree with original transactions")
        matrix = _matrix(pointers, indices, values, len(rows))
        _write_patterns(root / "patterns.txt", patterns)
        receipt.update(status="complete", eligible_patterns=len(patterns), n_patterns=len(patterns),
                       nnz=int(matrix.nnz), csc_storage_bytes=int(matrix.data.nbytes +
                       matrix.indices.nbytes + matrix.indptr.nbytes),
                       output_sha256=_sha(output_path), patterns_sha256=_sha(root / "patterns.txt"),
                       membership_audit_columns=len(audit_columns),
                       all_tid_counts_ranges_uniqueness_checked=True,
                       trivial_empty_dictionary=trivial, seconds=time.perf_counter() - started)
        _seal(root, receipt)
        return dict(patterns=patterns, matrix=matrix, receipt=receipt)
    except Exception as error:
        _save(root / "failure.json", dict(receipt, status="failed", error=repr(error),
              seconds=time.perf_counter() - started))
        raise


def transform_patterns(transactions, patterns, workdir):
    """Encode an already fixed dictionary on held-out rows, without re-mining.

    Returns the same ``{patterns, matrix, receipt}`` shape as ``mine_closed``.
    Pattern order is preserved; zero-support held-out columns remain present.
    """
    rows = _transactions(transactions)
    patterns = [tuple(pattern) for pattern in patterns]
    if any(not pattern or tuple(sorted(set(pattern))) != pattern or
           any(type(item) is not int or item < 0 for item in pattern) for pattern in patterns):
        raise ValueError("Expected nonempty canonical patterns")
    if len(set(patterns)) != len(patterns):
        raise ValueError("Duplicate patterns")
    root = _new_workdir(workdir)
    started = time.perf_counter()
    masks = _masks(rows)
    supports = np.fromiter((_pattern_mask(p, masks, len(rows)).bit_count() for p in patterns),
                           dtype=np.int64, count=len(patterns))
    pointers, indices, values = _allocate(root, len(rows), supports)
    for column, pattern in enumerate(patterns):
        indices[pointers[column]:pointers[column + 1]] = _mask_tids(
            _pattern_mask(pattern, masks, len(rows)), len(rows))
    matrix = _matrix(pointers, indices, values, len(rows))
    _write_patterns(root / "patterns.txt", patterns)
    receipt = dict(version=VERSION, status="complete", operation="fixed-pattern transform",
                   n_rows=len(rows), n_patterns=len(patterns), nnz=int(matrix.nnz),
                   patterns_sha256=_sha(root / "patterns.txt"),
                   seconds=time.perf_counter() - started)
    _seal(root, receipt)
    return dict(patterns=patterns, matrix=matrix, receipt=receipt)
