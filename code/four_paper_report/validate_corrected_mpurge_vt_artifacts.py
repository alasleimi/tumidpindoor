"""Read-only integrity and benchmark-completeness audit for the corrected VT suite.

This intentionally runs independently of ``corrected_mpurge_vt_suite.py`` so
that it can catch incomplete outputs, bookkeeping errors, leakage metadata, or
hash drift instead of reusing the producer's assumptions.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable


NATIVE_METHODS = {
    f"corrected_{native}_{cross}_{grouping}"
    for native in ("mpurge", "pba")
    for cross in ("printed", "order")
    for grouping in ("star", "components")
}
REQUIRED_PRIMARY_METHODS = NATIVE_METHODS | {
    "cycle_consistent_delay",
    "aoa_power_inverse_consensus",
    "beta_marginal_survival_multimodal",
    "bernoulli_rfs_multimodal",
    "two_sided_vt_registration",
    "multipath_bundle_adjustment_noncheating",
    "diffassign_bic_delay",
    "diffassign_bic_multimodal",
    "selfsup_deepsets_multimodal",
    "selfsup_attention_multimodal",
}
ORACLE_METHOD = "diffassign_multimodal_oracle_cardinality_ablation"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def iter_jsonl(path: Path) -> Iterable[dict]:
    handle = gzip.open(path, mode="rt", encoding="utf-8") if path.suffix == ".gz" else path.open(mode="r", encoding="utf-8")
    with handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as error:
                    raise AssertionError(f"invalid JSONL at {path.name}:{line_number}: {error}") from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def block_key(row: dict) -> tuple[int, float]:
    return int(row["scene_seed"]), float(row["noise_std_m"])


def finite_number(value, label: str) -> float:
    require(isinstance(value, (int, float)) and math.isfinite(float(value)), f"{label} is not finite: {value!r}")
    return float(value)


def audit_manifest(output_dir: Path) -> int:
    manifest = output_dir / "SHA256SUMS.txt"
    require(manifest.is_file(), "missing SHA256SUMS.txt")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, relative = line.split("  ", 1)
        target = output_dir / Path(relative)
        require(target.is_file(), f"manifest target missing: {relative}")
        require(sha256(target) == expected, f"SHA-256 mismatch: {relative}")
        checked += 1
    require(checked > 0, "empty SHA256SUMS.txt")
    return checked


def audit(output_dir: Path, *, allow_quick: bool) -> dict:
    required_files = {
        "config.json",
        "results.json",
        "block_audit.json",
        "block_metrics.jsonl",
        "raw_candidates_and_beta_sets.jsonl.gz",
        "raw_associations.jsonl.gz",
        "query_predictions.jsonl.gz",
        "report.md",
        "SHA256SUMS.txt",
    }
    missing = sorted(name for name in required_files if not (output_dir / name).is_file())
    require(not missing, f"missing artifacts: {missing}")

    config = load_json(output_dir / "config.json")
    results = load_json(output_dir / "results.json")
    block_audits = load_json(output_dir / "block_audit.json")
    arguments = config["arguments"]
    expected_blocks = {
        (int(scene), float(noise))
        for scene in arguments["scene_seeds"]
        for noise in arguments["noise_stds"]
    }
    require(results["blocks"] == len(expected_blocks), "reported block count is wrong")
    if allow_quick:
        require(results["status"] in {"QUICK_SMOKE_COMPLETE", "FULL_COMPLETE"}, "run is not complete")
    else:
        require(results["status"] == "FULL_COMPLETE", "expected a completed full run")

    contract = config["paper_contract"]
    barrier = config["information_barrier"]
    require(contract["total_window_P"] == 5 and contract["half_window_p"] == 2, "P=5 -> p=2 contract missing")
    require(contract["conflict_priority"] == "composite_dissimilarity", "wrong conflict sort")
    require(set(contract["cross_modes"]) == {"printed", "order"}, "cross-check bracket incomplete")
    require(contract["dynamic_beta_until_empty"] and contract["max_cardinality_gated_evaluation"], "evaluation contract incomplete")
    require(barrier["same_acquisition_multimodal_features_allowed"], "multimodal acquisition rule was disabled")
    for forbidden in ("extra_dense_sampling", "extra_forward_simulator_training_examples", "VT_or_ray_ID_input", "query_truth_input", "true_cardinality_primary"):
        require(barrier[forbidden] is False, f"information barrier violated: {forbidden}")

    for source, expected_hash in config["source_hashes"].items():
        source_path = Path(source)
        require(source_path.is_file(), f"source file missing: {source}")
        require(sha256(source_path) == expected_hash, f"source changed since run: {source}")

    metrics = list(iter_jsonl(output_dir / "block_metrics.jsonl"))
    require(metrics, "empty block_metrics.jsonl")
    require({block_key(row) for row in metrics} == expected_blocks, "metric block coverage mismatch")
    primary_by_block: dict[tuple[int, float], set[str]] = defaultdict(set)
    oracle_rows = []
    for row in metrics:
        method = row["method"]
        if row["primary"]:
            key = block_key(row)
            require(method not in primary_by_block[key], f"duplicate primary row: {key} {method}")
            primary_by_block[key].add(method)
        if method == ORACLE_METHOD:
            oracle_rows.append(row)
            require(not row["primary"] and row["oracle_excluded"], "oracle ablation leaked into primary results")
        for field in ("runtime_s", "precision", "recall", "f1", "chamfer_m", "hausdorff_m", "localization_mean_m"):
            finite_number(row[field], f"metric {method}.{field}")
        tp, fp, fn = int(row["tp"]), int(row["fp"]), int(row["fn"])
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        require(abs(row["precision"] - precision) < 1e-10, f"precision arithmetic mismatch: {method}")
        require(abs(row["recall"] - recall) < 1e-10, f"recall arithmetic mismatch: {method}")
        require(abs(row["f1"] - f1) < 1e-10, f"F1 arithmetic mismatch: {method}")
        require(sum(tx["tp"] for tx in row["gated_matches"]) == tp, f"TP aggregation mismatch: {method}")
        require(sum(tx["fp"] for tx in row["gated_matches"]) == fp, f"FP aggregation mismatch: {method}")
        require(sum(tx["fn"] for tx in row["gated_matches"]) == fn, f"FN aggregation mismatch: {method}")
    require(len(oracle_rows) == len(expected_blocks), "oracle diagnostic must occur exactly once per block")
    for key in expected_blocks:
        require(primary_by_block[key] == REQUIRED_PRIMARY_METHODS,
                f"primary method coverage mismatch for {key}: {sorted(primary_by_block[key] ^ REQUIRED_PRIMARY_METHODS)}")

    summaries = results["primary_summaries"]
    require({row["method"] for row in summaries} == REQUIRED_PRIMARY_METHODS, "primary summary method coverage mismatch")
    for row in summaries:
        require(row["blocks"] == len(expected_blocks), f"summary block count mismatch: {row['method']}")
        require(row["independent_scenes"] == len(arguments["scene_seeds"]), f"scene count mismatch: {row['method']}")
        for field in ("f1", "precision", "recall", "chamfer_m", "hausdorff_m", "localization_mean_m"):
            mean = finite_number(row[f"macro_{field}"], f"summary {row['method']}.{field}")
            low, high = row[f"scene_bootstrap_95ci_{field}"]
            low = finite_number(low, f"CI low {row['method']}.{field}")
            high = finite_number(high, f"CI high {row['method']}.{field}")
            require(low <= mean + 1e-12 and mean <= high + 1e-12, f"mean outside CI: {row['method']}.{field}")

    require(len(block_audits) == len(expected_blocks), "block_audit count mismatch")
    require({block_key(row["block"]) for row in block_audits} == expected_blocks, "block_audit coverage mismatch")
    for row in block_audits:
        require(len(row["evaluation_truth_cardinality_by_tx"]) == 4, "unexpected Tx count in evaluator")
        for method in NATIVE_METHODS:
            detail = row["method_details"][method]
            require(detail["dynamic_beta_exhausted"], f"beta did not empty: {block_key(row['block'])} {method}")
        mutual = row["method_details"]["two_sided_vt_registration"]
        require(mutual["mutual_nearest_per_survey_pair"] and mutual["roundtrip_best_observation_and_candidate"],
                f"two-sided round-trip audit missing: {block_key(row['block'])}")
        for tx_detail in row["method_details"]["multipath_bundle_adjustment_noncheating"]["selection"]:
            require(tx_detail["fixed_survey_poses"] and tx_detail["joint_soft_associations"],
                    f"noncheating BA audit missing: {block_key(row['block'])}")
            require(set(tx_detail["forbidden_inputs"]) == {"query_pose", "odometry", "true_z", "ray_id", "VT_id", "dense_simulation"},
                    f"noncheating BA forbidden-input declaration changed: {block_key(row['block'])}")

    beta_curves: dict[tuple[int, float, str, int], list[dict]] = defaultdict(list)
    candidate_ids: dict[tuple[int, float, str, int], set[int]] = defaultdict(set)
    candidate_rows = beta_rows = 0
    for row in iter_jsonl(output_dir / "raw_candidates_and_beta_sets.jsonl.gz"):
        key = (*block_key(row), row["method"], int(row["tx_id"]))
        if row["record_type"] == "candidate":
            candidate_rows += 1
            candidate_id = int(row["candidate_id"])
            require(candidate_id not in candidate_ids[key], f"duplicate candidate ID: {key} {candidate_id}")
            candidate_ids[key].add(candidate_id)
            require(int(row["support_count"]) >= 0, f"negative support: {key} {candidate_id}")
            require(len(row["coord_m"]) == 2 and all(math.isfinite(float(v)) for v in row["coord_m"]), f"bad coordinate: {key}")
            forbidden_keys = {"truth", "true_cardinality", "source_id", "ray_id", "vt_id"}
            require(not forbidden_keys.intersection({name.lower() for name in row}), f"latent field in candidate record: {key}")
        elif row["record_type"] == "beta_set":
            beta_rows += 1
            require(row["method"] in NATIVE_METHODS, f"beta row for non-native method: {row['method']}")
            beta_curves[key].append(row)
        else:
            raise AssertionError(f"unknown raw candidate record type: {row['record_type']}")
    require(candidate_rows > 0 and beta_rows > 0, "raw candidate/beta evidence is empty")
    for block in expected_blocks:
        for method in NATIVE_METHODS:
            for tx in range(4):
                key = (*block, method, tx)
                curve = sorted(beta_curves[key], key=lambda row: row["beta"])
                require(curve, f"missing beta curve: {key}")
                require(any(abs(float(row["beta"]) - 1.0) < 1e-9 for row in curve), f"beta=1 missing: {key}")
                require(curve[-1]["kept_candidate_ids"] == [], f"final beta set not empty: {key}")
                require(all(set(row["kept_candidate_ids"]).issubset(candidate_ids[key]) for row in curve), f"unknown candidate in beta curve: {key}")

    association_rows = 0
    for row in iter_jsonl(output_dir / "raw_associations.jsonl.gz"):
        association_rows += 1
        require(row["record_type"] == "association", "bad association record type")
        key = (*block_key(row), row["method"], int(row["tx_id"]))
        require(int(row["candidate_id"]) in candidate_ids[key], f"association references unknown candidate: {key}")
        require(int(row["anchor"]) >= 0 and int(row["path"]) >= 0, f"bad association index: {key}")
        observable_fields = (
            "range_residual_m", "range_m", "dissimilarity", "power_db",
            "observed_power_db", "angle_residual_deg",
        )
        present = [field for field in observable_fields if field in row]
        require(present, f"association has no observable residual/value: {key}")
        for field in present:
            finite_number(row[field], f"association {field}")
    require(association_rows > 0, "raw association evidence is empty")

    query_rows = 0
    for row in iter_jsonl(output_dir / "query_predictions.jsonl.gz"):
        query_rows += 1
        require(block_key(row) in expected_blocks, "query block outside configuration")
        require(len(row["target_xy_m"]) == 2 and len(row["estimate_xy_m"]) == 2, "bad query coordinate")
        finite_number(row["error_m"], "query error")
    require(query_rows == len(metrics) * int(arguments["query_points"]), "query prediction count mismatch")

    checkpoint_files = list((output_dir / "checkpoints").glob("*.pt"))
    require(len(checkpoint_files) == 2 * len(expected_blocks), "neural checkpoint count mismatch")
    manifest_entries = audit_manifest(output_dir)
    return {
        "status": "PASS",
        "run_status": results["status"],
        "blocks": len(expected_blocks),
        "primary_methods": len(REQUIRED_PRIMARY_METHODS),
        "metric_rows": len(metrics),
        "candidate_rows": candidate_rows,
        "beta_rows": beta_rows,
        "association_rows": association_rows,
        "query_rows": query_rows,
        "checkpoint_files": len(checkpoint_files),
        "manifest_entries": manifest_entries,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--allow-quick", action="store_true")
    args = parser.parse_args()
    print(json.dumps(audit(args.output_dir.resolve(), allow_quick=args.allow_quick), indent=2))


if __name__ == "__main__":
    main()
