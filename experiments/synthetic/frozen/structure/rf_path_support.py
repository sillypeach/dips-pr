"""Extract explicit positive itemsets from a fitted forest on binary items.

This is a declared rule-extraction assessment of an ordinary forest, not a
claim that RandomForest has a native sparse itemset support. Every non-root
root-to-node prefix is considered, including internal nodes. Its full signed
conditions are retained while traversing. The primary ``positive_rules`` set
contains only prefixes with no negative literals. ``positive_projection`` is
a separate diagnostic: it discards negative literals and is NOT an equivalent
tree rule. Both sets contain unique, nonempty itemsets, bounded by max_len and
their own support on X_fit. No response, test rows, or planted truth is read.

Tree API: https://scikit-learn.org/stable/auto_examples/tree/plot_unveil_tree_structure.html
Binary split semantics are evaluated on {0, 1}, rather than assuming every
threshold is 0.5. A rule's support is recomputed on all supplied fitting rows,
not taken from bootstrap-weighted node counts. No closure conversion is done:
it would change the literal rule that the forest actually expressed.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np

VERSION = "rf-signed-prefix-support-v1"


def _items(mask):
    result = []
    while mask:
        bit = mask & -mask
        result.append(bit.bit_length() - 1)
        mask ^= bit
    return tuple(result)


def _binary_matrix(X):
    # Forest input here has only the original items, not the large LCM matrix.
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim != 2 or min(X.shape) < 1:
        raise ValueError("X_fit must be a nonempty two-dimensional binary matrix")
    if X.dtype.kind not in "buif" or not np.isfinite(X).all():
        raise ValueError("X_fit must contain finite binary numeric values")
    if not np.logical_or(X == 0, X == 1).all():
        raise ValueError("X_fit must contain only 0 and 1")
    return X.astype(bool, copy=False)


def extract_rf_path_support(forest, X_fit, min_support, max_len):
    """Return JSON-safe rule lists and extraction diagnostics.

    ``positive_rules`` and ``positive_projection`` are sorted lists of sorted
    item-ID lists. Their parallel ``*_details`` lists have ``items``,
    ``support_count`` (unweighted fitting rows), ``tree_count`` (distinct trees),
    and ``path_count`` (root-to-node prefix occurrences, including duplicates).
    Projection details additionally distinguish pure-positive and mixed-sign
    occurrences. Neither a popularity threshold nor a truth-dependent filter
    is applied. The first 20 distinct signed prefixes are kept as an audit
    sample; full signed paths are not serialized.

    A prefix with many negative literals can contribute a short projection;
    max_len bounds the output positive itemset, not the signed path length.
    Empty positive projections never become output candidates. Inconsistent,
    binary-domain-empty, or fitting-support-empty branches contribute neither
    rules nor projections. The fitted forest is never modified.
    """
    for name, value in (("min_support", min_support), ("max_len", max_len)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    min_support, max_len = int(min_support), int(max_len)
    X = _binary_matrix(X_fit)
    n, p = X.shape
    trees = getattr(forest, "estimators_", None)
    if trees is None or len(trees) == 0 or not hasattr(forest, "n_features_in_"):
        raise ValueError("forest must be fitted and expose estimators_ and n_features_in_")
    if int(forest.n_features_in_) != p:
        raise ValueError("X_fit feature count does not match the fitted forest")
    all_rows = (1 << n) - 1
    one_masks = tuple(sum(1 << int(row) for row in np.flatnonzero(X[:, j]))
                      for j in range(p))
    zero_masks = tuple(all_rows ^ mask for mask in one_masks)
    support_cache = {0: n}
    primary, projected = {}, {}
    signed_seen = set()
    signed_sample = []
    stats = dict(trees=len(trees), nodes_visited=0, nonroot_prefix_occurrences=0,
                 signed_prefix_occurrences=0, pure_positive_prefix_occurrences=0,
                 mixed_sign_prefix_occurrences=0, negative_only_prefix_occurrences=0,
                 empty_condition_prefix_occurrences=0, domain_empty_branches=0,
                 contradictory_branches=0, fit_empty_branches=0,
                 unconstrained_branches=0, repeated_literal_branches=0,
                 overlength_positive_prefix_occurrences=0,
                 low_support_positive_prefix_occurrences=0)

    def positive_support(mask):
        if mask not in support_cache:
            rows = all_rows
            for item in _items(mask):
                rows &= one_masks[item]
            support_cache[mask] = rows.bit_count()
        return support_cache[mask]

    def record(store, mask, support, tree_index, pure):
        row = store.setdefault(mask, dict(support_count=support, trees=set(),
                                         path_count=0, pure_path_count=0,
                                         mixed_path_count=0))
        row["trees"].add(tree_index)
        row["path_count"] += 1
        row["pure_path_count" if pure else "mixed_path_count"] += 1

    for tree_index, estimator in enumerate(trees):
        tree = getattr(estimator, "tree_", None)
        if tree is None:
            raise ValueError("Every forest estimator must expose a fitted tree_")
        left = np.asarray(tree.children_left)
        right = np.asarray(tree.children_right)
        features = np.asarray(tree.feature)
        thresholds = np.asarray(tree.threshold)
        size = len(left)
        if not size or any(len(a) != size for a in (right, features, thresholds)):
            raise ValueError("Invalid fitted tree arrays")
        # Validation also protects custom/serialized tree adapters against cycles.
        visited = set()
        stack = [(0, 0, 0, all_rows)]
        while stack:
            node, positive, negative, fitting_rows = stack.pop()
            if node < 0 or node >= size or node in visited:
                raise ValueError("Invalid child index, repeated child, or tree cycle")
            visited.add(node)
            stats["nodes_visited"] += 1
            if node:
                stats["nonroot_prefix_occurrences"] += 1
                if positive or negative:
                    stats["signed_prefix_occurrences"] += 1
                    category = ("pure_positive" if not negative else
                                "mixed_sign" if positive else "negative_only")
                    stats[category + "_prefix_occurrences"] += 1
                    signed_key = (positive, negative)
                    if signed_key not in signed_seen:
                        signed_seen.add(signed_key)
                        if len(signed_sample) < 20:
                            signed_sample.append(dict(positive_items=list(_items(positive)),
                                negative_items=list(_items(negative)),
                                support_count=fitting_rows.bit_count(),
                                tree_index=tree_index, node_index=node))
                else:
                    stats["empty_condition_prefix_occurrences"] += 1
                if positive:
                    if positive.bit_count() > max_len:
                        stats["overlength_positive_prefix_occurrences"] += 1
                    else:
                        support = positive_support(positive)
                        if support < min_support:
                            stats["low_support_positive_prefix_occurrences"] += 1
                        else:
                            pure = negative == 0
                            record(projected, positive, support, tree_index, pure)
                            if pure:
                                record(primary, positive, support, tree_index, True)
            lchild, rchild = int(left[node]), int(right[node])
            if lchild == -1 and rchild == -1:
                continue
            if lchild < 0 or rchild < 0:
                raise ValueError("A nonleaf must have two valid children")
            feature, threshold = int(features[node]), float(thresholds[node])
            if not 0 <= feature < p or np.isnan(threshold):
                raise ValueError("Invalid feature index or NaN split threshold")
            bit = 1 << feature
            # Push right first for deterministic left-to-right depth-first output.
            for child, go_left in ((rchild, False), (lchild, True)):
                allowed = tuple(value for value in (0, 1)
                                if ((value <= threshold) if go_left else (value > threshold)))
                if not allowed:
                    stats["domain_empty_branches"] += 1
                    continue
                next_positive, next_negative = positive, negative
                next_rows = fitting_rows
                if len(allowed) == 2:
                    stats["unconstrained_branches"] += 1
                else:
                    value = allowed[0]
                    if (value == 1 and negative & bit) or (value == 0 and positive & bit):
                        stats["contradictory_branches"] += 1
                        continue
                    if (value == 1 and positive & bit) or (value == 0 and negative & bit):
                        stats["repeated_literal_branches"] += 1
                    if value:
                        next_positive |= bit
                        next_rows &= one_masks[feature]
                    else:
                        next_negative |= bit
                        next_rows &= zero_masks[feature]
                if not next_rows:
                    stats["fit_empty_branches"] += 1
                    continue
                stack.append((child, next_positive, next_negative, next_rows))

    def details(store, is_projection):
        result = []
        for mask in sorted(store, key=lambda m: (m.bit_count(), _items(m))):
            data = store[mask]
            row = dict(items=list(_items(mask)), support_count=data["support_count"],
                       tree_count=len(data["trees"]), path_count=data["path_count"])
            if is_projection:
                row.update(pure_positive_path_count=data["pure_path_count"],
                           mixed_sign_path_count=data["mixed_path_count"])
            result.append(row)
        return result

    primary_details = details(primary, False)
    projected_details = details(projected, True)
    stats.update(unique_signed_prefixes=len(signed_seen),
                 unique_positive_rules=len(primary),
                 unique_positive_projections=len(projected),
                 projection_only_itemsets=len(set(projected) - set(primary)))
    return dict(version=VERSION, n_fit=n, n_items=p, min_support=min_support,
                max_len=max_len, primary_definition="all-positive root-to-node prefixes",
                projection_definition="positive literals of signed prefixes; not equivalent rules",
                support_definition="unweighted positive-conjunction support on X_fit",
                positive_rules=[row["items"] for row in primary_details],
                positive_projection=[row["items"] for row in projected_details],
                positive_rule_details=primary_details,
                positive_projection_details=projected_details,
                signed_prefix_audit_sample=signed_sample, stats=stats)
