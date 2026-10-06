"""Annotate frozen position rules with observed joint mutation profiles.

This is a deterministic, training-only descriptive annotation. It does not fit
models, change rules or coefficients, or infer mutation-specific effects.
Reference residues come from the hash-matched source CompMutList field.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import re


AUDIT = Path(__file__).resolve().parent
PACKAGE = AUDIT.parent
ROOT = PACKAGE.parents[1]
FROZEN = ROOT / "outputs/hiv_full_retrain_20260928_v1/endpoints"
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
SIMPLE = re.compile(r"([ACDEFGHIKLMNPQRSTVWY])(\d+)([ACDEFGHIKLMNPQRSTVWY])")
GENERAL = re.compile(r"([A-Z])(\d+)(.*)")
TIEBREAK = "largest joint count, then lexicographic tuple of source mutation strings in ascending position order"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_contexts(evidence):
    datasets, contexts, provenance, references = {}, {}, {}, {}
    checked_rows = 0
    for endpoint in evidence["endpoints"]:
        name = endpoint["endpoint"]
        drug_class, drug = name.split("/")
        directory = FROZEN / f"hiv_{drug_class}_{drug}"
        protocol = read_json(directory / "protocol.json")
        inputs = {}
        for filename in ("metadata.json", "data.json", "feature_mapping.json"):
            path = directory / filename
            assert sha256(path) == protocol["files"][filename], (name, filename, "hash mismatch")
            inputs[filename] = read_json(path)
        metadata, data, mapping = (inputs[n] for n in ("metadata.json", "data.json", "feature_mapping.json"))
        source = ROOT / metadata["source_file"]
        assert sha256(source) == metadata["source_sha256"] == mapping["source_sha256"], name
        if drug_class not in datasets:
            with source.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            references[drug_class] = defaultdict(Counter)
            for row in rows:
                by_position = defaultdict(list)
                for token in row["CompMutList"].split(","):
                    token = token.strip()
                    match = GENERAL.fullmatch(token)
                    if match:
                        ref, position, _ = match.groups()
                        position = int(position)
                        by_position[position].append(token)
                        references[drug_class][position][ref] += 1
                row["_mutations"] = dict(by_position)
            datasets[drug_class] = rows
            provenance[drug_class] = {
                "path": str(source.relative_to(ROOT)), "sha256": sha256(source),
                "record_count": len(rows), "annotation_field": "CompMutList",
                "protein": mapping["protein"],
            }
        rows = datasets[drug_class]
        indices = metadata["train_source_rows"]
        ids = metadata["train_sequence_ids"]
        assert data["tx_train"] == data["tx_final"] and data["y_train"] == data["y_final"], name
        assert len(indices) == len(ids) == len(data["tx_final"]) == len(data["y_final"]), name
        for source_index, seqid, transaction, response in zip(indices, ids, data["tx_final"], data["y_final"]):
            row = rows[source_index]
            assert row["SeqID"] == seqid, (name, "source row/SeqID mismatch")
            actual = [j for j in range(metadata["n_items"]) if row[f"P{j+1}"].strip() not in ("", "-", ".")]
            assert actual == transaction, (name, "encoded transaction mismatch")
            assert float(row[drug]) == response, (name, "response mismatch")
            checked_rows += 1
        contexts[name] = {
            "rows": [rows[i] for i in indices],
            "hashes": {n: sha256(directory / n) for n in ("protocol.json", *inputs)},
        }
    conflicts = {
        cls: {str(position): dict(counts) for position, counts in positions.items() if len(counts) != 1}
        for cls, positions in references.items()
    }
    assert not any(conflicts.values()), ("Inconsistent source reference residues", conflicts)
    return contexts, provenance, references, checked_rows


def raw_token_issue(raw):
    if len(raw) == 1 and raw in AMINO_ACIDS:
        return None
    if raw == "X":
        return "unknown_X"
    if "#" in raw or "_" in raw:
        return "insertion_or_special_marker"
    if "*" in raw:
        return "asterisk_special_marker"
    if len(raw) > 1 and set(raw).issubset(AMINO_ACIDS):
        return "mixed_amino_acid_token"
    return "other_non_simple_token"


def annotate(term, contexts):
    positions = [int(column[1:]) for column in term["columns"]]
    assert positions == sorted(positions) == [item + 1 for item in term["item_ids"]]
    rows = contexts[term["endpoint"]]["rows"]
    supported = [row for row in rows if all(row[f"P{p}"].strip() not in ("", "-", ".") for p in positions)]
    assert len(supported) == term["training_support"], (term["endpoint"], positions, "support mismatch")
    counts, issues = Counter(), Counter()
    raw_counts = {p: Counter() for p in positions}
    for row in supported:
        joint, row_issues = [], set()
        for position in positions:
            raw = row[f"P{position}"].strip()
            raw_counts[position][raw] += 1
            issue = raw_token_issue(raw)
            if issue:
                row_issues.add(issue)
                continue
            annotations = row["_mutations"].get(position, [])
            candidates = []
            for token in annotations:
                match = SIMPLE.fullmatch(token)
                if match and int(match[2]) == position and match[3] == raw and match[1] != match[3]:
                    candidates.append(token)
            if len(candidates) != 1:
                row_issues.add("missing_or_nonunique_matching_simple_source_annotation")
            else:
                joint.append(candidates[0])
        if row_issues:
            issues.update(row_issues)
        else:
            assert len(joint) == len(positions)
            counts[tuple(joint)] += 1
    ranked = sorted(counts.items(), key=lambda value: (-value[1], value[0]))
    best_profile, best_count = ranked[0] if ranked else ((), 0)
    resolved = sum(counts.values())
    ties = [list(profile) for profile, count in ranked if count == best_count]
    # Original coefficient, support, and other fitted-model fields are copied.
    return {
        **term,
        "positions": positions,
        "annotation_status": "observed_training_profile" if ranked else "NA_no_fully_resolved_joint_profile",
        "modal_fully_resolved_joint_mutations": list(best_profile) if ranked else None,
        "modal_profile": " + ".join(best_profile) if ranked else "NA",
        "modal_profile_display": " + ".join(best_profile) if ranked else "NA",
        "modal_count": best_count,
        "support": len(supported),
        "representative_mutations": list(best_profile) if ranked else None,
        "representative_count": best_count,
        "joint_count": best_count,
        "joint_count_over_original_support": f"{best_count}/{len(supported)}" if ranked else "NA",
        "resolved_rows": resolved,
        "excluded_rows": len(supported) - resolved,
        "excluded_reason_row_counts_nonexclusive": dict(sorted(issues.items())),
        "distinct_resolved_joint_profiles": len(counts),
        "modal_ties": ties,
        "tiebreak": TIEBREAK,
        "raw_token_counts_in_rule_support": {str(p): dict(sorted(raw_counts[p].items())) for p in positions},
        "resolved_joint_profile_counts": [{"mutations": list(profile), "count": count} for profile, count in ranked],
    }


def build():
    input_path = AUDIT / "hiv_audit.json"
    evidence = read_json(input_path)
    contexts, sources, references, checked_rows = load_contexts(evidence)
    terms = [annotate(term, contexts) for term in evidence["records"]["all_CV_min_terms"]]
    index = {(term["endpoint"], tuple(term["columns"])): term for term in terms}
    assert len(index) == len(terms)
    top = []
    for original in evidence["records"]["top_three_multi_item_per_endpoint"]:
        term = index[original["endpoint"], tuple(original["columns"])]
        assert term["coefficient"] == original["coefficient"] and term["training_support"] == original["training_support"]
        top.append({**term, "rank_among_multi_item_by_absolute_coefficient": original["rank_among_multi_item_by_absolute_coefficient"]})
    report = {
        "version": "source-joint-mutation-annotations-v1",
        "scope": "Descriptive training-support annotations of unchanged position rules, not mutation-specific fits or effects. No training or held-out selection.",
        "builder_sha256": sha256(Path(__file__)),
        "input_audit_sha256": sha256(input_path),
        "source_files": sources,
        "frozen_endpoint_input_sha256": {name: context["hashes"] for name, context in contexts.items()},
        "reference_residue_provenance": "Reference amino acids are copied and cross-checked from the released source CompMutList strings. No external reference genome or mutation-specific fitted model is inferred.",
        "reference_residue_counts": {cls: {str(p): dict(counts) for p, counts in sorted(positions.items())} for cls, positions in references.items()},
        "fully_resolved_definition": "At every rule position, the raw token is one standard amino acid and exactly one simple RefPositionAlt CompMutList token agrees with it, with distinct reference and alternate residues. All tokens must co-occur in the same fitting row.",
        "excluded_token_note": "Mixtures, X, insertion markers, asterisk/deletion-like markers, other special tokens, and missing or nonunique source annotations remain in the original rule support but are excluded from the fully resolved annotation pool. Their biological meaning is not inferred from their marker.",
        "denominator_note": "joint_count/original training support, including records excluded from the resolved pool; not joint_count/resolved_rows. Exclusion reason counts can overlap for one row.",
        "interpretation_note": "The positional coefficient and CV recurrence apply to the original position conjunction, not specifically to its modal mutation annotation. Marginal allele sets are never combined into unobserved joint profiles.",
        "tiebreak": TIEBREAK,
        "checks": {
            "endpoints": len(contexts), "source_files_hash_matched": len(sources),
            "frozen_input_files_hash_matched": 3 * len(contexts),
            "training_rows_checked": checked_rows, "catalog_terms": len(terms),
            "top_multi_item_terms": len(top), "nrti_top_terms": sum(t["drug_class"] == "NRTI" for t in top),
            "all_supports_match": True, "reference_conflicts": 0,
            "catalog_terms_without_resolved_profile": sum(t["resolved_rows"] == 0 for t in terms),
            "top_terms_without_resolved_profile": sum(t["resolved_rows"] == 0 for t in top),
        },
        "top_multi_item_annotations": top,
        "catalog_annotations": [
            {key: term[key] for key in (
                "endpoint", "columns", "positions", "training_support",
                "representative_mutations", "representative_count", "modal_ties",
                "modal_profile_display", "modal_count", "support", "resolved_rows",
                "excluded_rows", "tiebreak", "annotation_status",
            )}
            for term in terms
        ],
        "terms": terms,
    }
    stream = io.StringIO(newline="")
    columns = ["endpoint", "model", "positions", "position_rule", "coefficient", "training_support", "n_fit", "modal_fully_resolved_joint_profile", "joint_count", "joint_count_over_original_support", "resolved_rows", "excluded_rows", "excluded_reason_row_counts_nonexclusive", "distinct_resolved_joint_profiles", "modal_tie_count", "tied_profiles", "tiebreak", "annotation_status", "interpretation"]
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for term in terms:
        writer.writerow({
            "endpoint": term["endpoint"], "model": term["model"],
            "positions": ";".join(map(str, term["positions"])), "position_rule": " + ".join(term["columns"]),
            "coefficient": format(term["coefficient"], ".17g"), "training_support": term["training_support"], "n_fit": term["n_fit"],
            "modal_fully_resolved_joint_profile": term["modal_profile"], "joint_count": term["joint_count"],
            "joint_count_over_original_support": term["joint_count_over_original_support"],
            "resolved_rows": term["resolved_rows"], "excluded_rows": term["excluded_rows"],
            "excluded_reason_row_counts_nonexclusive": json.dumps(term["excluded_reason_row_counts_nonexclusive"], sort_keys=True),
            "distinct_resolved_joint_profiles": term["distinct_resolved_joint_profiles"],
            "modal_tie_count": len(term["modal_ties"]), "tied_profiles": json.dumps(term["modal_ties"]),
            "tiebreak": TIEBREAK, "annotation_status": term["annotation_status"],
            "interpretation": "Observed training annotation; coefficient belongs to the original position rule, not this mutation profile.",
        })
    return report, stream.getvalue()


def main():
    report, csv_text = build()
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output_json = AUDIT / "hiv_mutation_annotations.json"
    output_csv = PACKAGE / "Supplementary_Data/hiv_term_mutation_annotations.csv"
    output_json.write_text(text, encoding="utf-8")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text(csv_text, encoding="utf-8")
    print(json.dumps({"checks": report["checks"], "outputs": {str(output_json): sha256(output_json), str(output_csv): sha256(output_csv)}}, indent=2))


if __name__ == "__main__":
    main()
