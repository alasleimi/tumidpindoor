"""Independent post-run audit for the corrected CNN full benchmark.

This file deliberately does not import the benchmark runner.  It checks the
serialized evidence and frozen source copies without relying on runner state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


EXPECTED_PROTOCOL_COUNTS = {
    "paper_condition_map": 11,
    "environment_blind_map": 16,
    "protocol_independent": 9,
}
EXPECTED_ROOMS = ("room_a", "room_b")
EXPECTED_QUERIES = 1600
EXPECTED_ROWS = 2 * EXPECTED_QUERIES * sum(EXPECTED_PROTOCOL_COUNTS.values())
EXPECTED_UNIQUE_METHODS = 25


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left: float, right: float, atol: float = 1e-10) -> bool:
    return math.isclose(float(left), float(right), rel_tol=1e-10, abs_tol=atol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    result_path = output_dir / "results.json"
    raw_path = output_dir / "raw_rows.jsonl"
    integrity_path = output_dir / "integrity_audit.json"
    frozen_runner = output_dir / "frozen_sources" / "run_cnn_fair.py"
    frozen_receiver = output_dir / "frozen_sources" / "multimodal_receiver.py"

    result = json.loads(result_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in raw_path.read_text(encoding="utf-8").splitlines() if line]
    runner_audit = json.loads(integrity_path.read_text(encoding="utf-8"))
    checks: dict[str, bool] = {}

    checks["full_complete_status"] = result.get("status") == "FULL_COMPLETE"
    checks["raw_sha_matches_result"] = sha256_file(raw_path) == result.get("raw_rows_sha256")
    checks["integrity_sha_matches_result"] = sha256_file(integrity_path) == result.get("integrity_audit_sha256")
    checks["frozen_runner_sha_matches_result"] = sha256_file(frozen_runner) == result.get("source_sha256")
    checks["frozen_receiver_sha_matches_result"] = sha256_file(frozen_receiver) == result.get("receiver_source_sha256")
    checks["completion_runner_hash_matches_frozen"] = result.get("live_source_hashes_at_completion", {}).get("runner") == sha256_file(frozen_runner)
    checks["completion_receiver_hash_matches_frozen"] = result.get("live_source_hashes_at_completion", {}).get("receiver") == sha256_file(frozen_receiver)
    checks["runner_self_audit_pass"] = runner_audit.get("status") == "PASS" and all(runner_audit.get("checks", {}).values())
    checks["exact_row_count_115200"] = len(rows) == EXPECTED_ROWS

    keys = [(r["protocol"], r["room"], r["method"], r["physical_query_id"]) for r in rows]
    checks["unique_protocol_room_method_query_keys"] = len(keys) == len(set(keys))
    checks["all_numeric_outputs_finite"] = all(
        np.isfinite(float(r["error_m"]))
        and np.asarray(r["truth_xy_m"]).shape == (2,)
        and np.asarray(r["prediction_xy_m"]).shape == (2,)
        and np.all(np.isfinite(r["truth_xy_m"]))
        and np.all(np.isfinite(r["prediction_xy_m"]))
        for r in rows
    )
    checks["stored_error_recomputes_from_coordinates"] = all(
        close(
            r["error_m"],
            np.linalg.norm(np.asarray(r["prediction_xy_m"], dtype=float) - np.asarray(r["truth_xy_m"], dtype=float)),
        )
        for r in rows
    )
    checks["privileged_flag_only_on_paper_condition_protocol"] = all(
        bool(r["uses_privileged_condition_identity"]) == (r["protocol"] == "paper_condition_map")
        for r in rows
    )
    checks["two_sided_alias_has_no_rows"] = all(r["method"] != "two_sided_vt_registration" for r in rows)

    method_sets: dict[str, set[str]] = defaultdict(set)
    counts: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        method_sets[row["protocol"]].add(row["method"])
        counts[(row["room"], row["protocol"], row["method"])] += 1
    checks["protocol_method_cardinalities_11_16_9"] = all(
        len(method_sets[protocol]) == expected for protocol, expected in EXPECTED_PROTOCOL_COUNTS.items()
    )
    checks["each_room_protocol_method_has_1600_queries"] = all(
        counts[(room, protocol, method)] == EXPECTED_QUERIES
        for room in EXPECTED_ROOMS
        for protocol, methods in method_sets.items()
        for method in methods
    )
    checks["exactly_25_unique_methods"] = len(set().union(*method_sets.values())) == EXPECTED_UNIQUE_METHODS

    # A method may occur in both map brackets only if the map choice actually
    # changes its complete prediction vector.
    vectors: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        vectors[(row["room"], row["method"], row["protocol"])][row["physical_query_id"]] = tuple(row["prediction_xy_m"])
    duplicated = []
    variants: dict[tuple[str, str], list[tuple[str, dict[str, tuple[float, float]]]]] = defaultdict(list)
    for (room, method, protocol), predictions in vectors.items():
        variants[(room, method)].append((protocol, predictions))
    for (room, method), protocol_variants in variants.items():
        for index, (left_name, left) in enumerate(protocol_variants):
            for right_name, right in protocol_variants[index + 1 :]:
                common = set(left) & set(right)
                if common and all(left[key] == right[key] for key in common):
                    duplicated.append([room, method, left_name, right_name])
    checks["no_complete_prediction_vector_duplicated_across_protocols"] = not duplicated

    summary_index = {(s["protocol"], s["room"], s["method"]): s for s in result["summaries"]}
    checks["summary_count_72"] = len(summary_index) == 72
    summary_ok = True
    for key, summary in summary_index.items():
        errors = np.asarray(
            [r["error_m"] for r in rows if (r["protocol"], r["room"], r["method"]) == key],
            dtype=float,
        )
        summary_ok &= len(errors) == EXPECTED_QUERIES
        summary_ok &= close(summary["mean_error_m"], errors.mean())
        summary_ok &= close(summary["rmse_m"], np.sqrt(np.mean(errors**2)))
        summary_ok &= close(summary["median_error_m"], np.median(errors))
        summary_ok &= close(summary["p90_error_m"], np.quantile(errors, 0.9))
    checks["all_summary_metrics_recompute"] = bool(summary_ok)

    row_index = {(r["protocol"], r["room"], r["method"], int(r["query"])): r for r in rows}
    interval_ok = True
    for interval in result["paired_intervals"]:
        protocol, room, method = interval["protocol"], interval["room"], interval["method"]
        baseline = interval["baseline"]
        gains = np.asarray(
            [
                row_index[("protocol_independent", room, baseline, query)]["error_m"]
                - row_index[(protocol, room, method, query)]["error_m"]
                for query in range(EXPECTED_QUERIES)
            ],
            dtype=float,
        )
        interval_ok &= interval["baseline_protocol"] == "protocol_independent"
        interval_ok &= interval["bootstrap_unit"] == "physical off-grid query paired across methods/protocols"
        interval_ok &= int(interval["bootstrap_clusters"]) == EXPECTED_QUERIES
        interval_ok &= int(interval["bootstrap_repetitions"]) == 2000
        interval_ok &= close(interval["mean_error_reduction_m"], gains.mean())
        interval_ok &= float(interval["ci95_low_m"]) <= float(interval["ci95_high_m"])
    checks["paired_intervals_are_physical_query_clustered_and_recompute"] = bool(interval_ok)

    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_rows": len(rows),
        "expected_rows": EXPECTED_ROWS,
        "unique_methods": len(set().union(*method_sets.values())),
        "protocol_method_counts": {key: len(value) for key, value in sorted(method_sets.items())},
        "duplicate_complete_vectors": duplicated,
        "results_sha256": sha256_file(result_path),
        "raw_rows_sha256": sha256_file(raw_path),
        "frozen_runner_sha256": sha256_file(frozen_runner),
        "frozen_receiver_sha256": sha256_file(frozen_receiver),
    }
    target = args.audit_output or (output_dir / "independent_integrity_audit.json")
    target.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))
    if audit["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
