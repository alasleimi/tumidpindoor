"""Mechanical and statistical audit of the completed fair MPUrge-MAP suite."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "research" / "four_paper_report" / "corrected_mpurge_map_full"
BASELINE = "majdi_mpurge_map_corrected_prose"
LEARNED = {
    "pointnet_delay_direct",
    "pointnet_multimodal_direct",
    "caez_grid_probability_delay",
    "residual_to_analytic_delay",
    "pointnet_analytic_reranker_delay",
    "cross_attention_multimodal",
    "candidate_self_attention_multimodal",
}
DELAY_ONLY = {
    "majdi_mpurge_map_corrected_prose",
    "majdi_mca_eps1.6_k1",
    "majdi_original_mpurge_P5_corrected",
    "ablation_A_chamfer_inverse_top3",
    "ablation_B_temperature_softmax_only",
    "ablation_C_path_count_empty_only",
    "ablation_D_chamfer_temperature_pathcount",
    "subset_consensus_chamfer",
    "graph_diffusion_corrected",
    "frozen_evomdp_rank",
    "delay_feature_extratrees",
    "rrle_moe",
    "pointnet_delay_direct",
    "caez_grid_probability_delay",
    "residual_to_analytic_delay",
    "pointnet_analytic_reranker_delay",
}


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_query_block(rows: list[dict], candidate: str, baseline: str, draws: int = 5000) -> dict:
    lookup = {(row["method"], row["condition"], int(row["query"])): float(row["error_m"]) for row in rows}
    conditions = sorted({row["condition"] for row in rows})
    queries = sorted({int(row["query"]) for row in rows})
    gain = np.asarray([
        [lookup[(baseline, condition, query)] - lookup[(candidate, condition, query)] for condition in conditions]
        for query in queries
    ])
    rng = np.random.default_rng(20260807)
    indices = rng.integers(0, len(queries), size=(draws, len(queries)))
    sample = gain[indices].mean(axis=(1, 2))
    return {
        "candidate": candidate,
        "baseline": baseline,
        "positive_means_candidate_better": True,
        "mean_reduction_m": float(np.mean(gain)),
        "query_block_95ci_m": [float(np.quantile(sample, 0.025)), float(np.quantile(sample, 0.975))],
        "probability_positive": float(np.mean(sample > 0.0)),
    }


def main() -> None:
    checks: list[dict] = []

    def check(condition: bool, name: str, detail=None) -> None:
        checks.append({"name": name, "pass": bool(condition), "detail": detail})
        if not condition:
            raise AssertionError(f"{name}: {detail}")

    results = read_json(RUN / "results.json")
    aggregates = read_json(RUN / "aggregate_metrics.json")
    bootstraps = read_json(RUN / "paired_bootstrap_vs_corrected_majid.json")
    selection = read_json(RUN / "selection_freeze.json")
    invariance = read_json(RUN / "invariance_and_edge_case_audit.json")
    applicability = read_json(RUN / "applicability_and_exclusions.json")
    manifest = read_json(RUN / "run_manifest.json")
    raw = read_jsonl(RUN / "raw_predictions.jsonl")
    seed_rows = read_jsonl(RUN / "learned_seed_predictions.jsonl")

    methods = sorted({row["method"] for row in raw})
    conditions = sorted({row["condition"] for row in raw})
    queries = sorted({int(row["query"]) for row in raw})
    check(results["status"] == "FULL_RUN_COMPLETE", "full status")
    check(len(methods) == 25, "25 primary methods", methods)
    check(len(conditions) == 16, "16 conditions", conditions)
    check(len(queries) == 100, "100 locked queries")
    check(len(raw) == 25 * 16 * 100 == 40000, "primary raw row count", len(raw))
    keys = [(row["method"], row["condition"], int(row["query"])) for row in raw]
    check(len(set(keys)) == len(keys), "primary row keys unique")
    check(all(np.isfinite(row["error_m"]) for row in raw), "all primary errors finite")
    recomputed = [
        abs(float(row["error_m"]) - float(np.linalg.norm(np.asarray(row["prediction_xy_m"]) - np.asarray(row["truth_xy_m"]))))
        for row in raw
    ]
    check(max(recomputed) <= 1.0e-10, "errors reconstruct from raw coordinates", max(recomputed))
    truths: dict[tuple[str, int], tuple[float, float]] = {}
    truth_alignment_ok = True
    for row in raw:
        key = (row["condition"], int(row["query"]))
        value = tuple(row["truth_xy_m"])
        if key in truths and truths[key] != value:
            truth_alignment_ok = False
        truths[key] = value
    check(truth_alignment_ok and len(truths) == 1600, "truth aligned across methods", len(truths))

    check(len(seed_rows) == 7 * 3 * 16 * 100 == 33600, "learned seed row count", len(seed_rows))
    seed_keys = [(row["method"], int(row["seed"]), row["condition"], int(row["query"])) for row in seed_rows]
    check(len(set(seed_keys)) == len(seed_keys), "seed row keys unique")
    seed_coverage = {
        method: sorted({int(row["seed"]) for row in seed_rows if row["method"] == method})
        for method in sorted(LEARNED)
    }
    check(all(value == [211, 337, 461] for value in seed_coverage.values()), "all learned methods have three seeds", seed_coverage)
    check(all(sum(row["method"] == method and int(row["seed"]) == seed for row in seed_rows) == 1600
              for method in LEARNED for seed in (211, 337, 461)), "1600 rows per learned method/seed")

    check(selection["locked_query_arrays_accessed_by_selection"] is False, "locked queries untouched during selection")
    split = selection["whole_RP_split"]
    check(len(split["training_groups"]) == 64 and len(split["validation_groups"]) == 16, "64/16 whole-RP split")
    check(set(split["training_groups"]).isdisjoint(split["validation_groups"]), "RP groups disjoint")
    check(selection["training_augmentation"]["new_simulator_calls"] == 0, "no augmentation simulator calls")
    check(selection["training_augmentation"]["samples"] == 1280, "1280 stored-survey augmentations")
    check(invariance["all_pass"] is True, "all invariance and edge checks pass", invariance["checks"])
    check(all(row["pass"] for row in invariance["checks"]), "each invariance check passes")
    aod = next(row for row in applicability["excluded"] if row["method_family"] == "AoD-conditioned fingerprinting")
    check(aod["status"].startswith("NOT_INSTANTIATED"), "AoD wording is acquisition-fair", aod)
    check("eligible" in aod["reason"], "AoD note explicitly permits same-acquisition sensing")

    with np.load(RUN / "protocol_observations.npz", allow_pickle=False) as protocol:
        check(protocol["reference_xy_m"].shape == (80, 2), "protocol has 80 survey locations")
        check(protocol["query_xy_m"].shape == (100, 2), "protocol has 100 query locations")
        check(protocol["survey_delay"].shape == (16, 80, 9), "survey delay tensor shape")
        check(protocol["query_delay"].shape == (16, 100, 9), "query delay tensor shape")
        check(protocol["survey_cir"].shape == (16, 80, 1, 8, 256), "survey CIR tensor shape")
        check(protocol["query_cir"].shape == (16, 100, 1, 8, 256), "query CIR tensor shape")
        check("survey_tx_ids" not in protocol.files and "query_tx_ids" not in protocol.files, "no simulator/source IDs stored")
        check(not np.array_equal(protocol["survey_delay"][4], protocol["survey_delay"][0]), "condition-specific survey corruption materialized")

    source_hash_results = []
    for name, expected in manifest["source_sha256"].items():
        path = Path(name)
        actual = sha256(path)
        source_hash_results.append({"path": name, "expected": expected, "actual": actual, "pass": actual == expected})
    artifact_hash_results = []
    for name, expected in manifest["artifact_sha256"].items():
        path = Path(name)
        if not path.is_absolute():
            path = ROOT / path
        actual = sha256(path)
        artifact_hash_results.append({"path": name, "expected": expected, "actual": actual, "pass": actual == expected})
    check(all(row["pass"] for row in artifact_hash_results), "all result artifact hashes match", [row for row in artifact_hash_results if not row["pass"]])
    source_mismatch = [row for row in source_hash_results if not row["pass"]]
    check(
        len(source_mismatch) <= 1 and all(Path(row["path"]).name == "multimodal_receiver.py" for row in source_mismatch),
        "only disclosed concurrent receiver source mutation may mismatch",
        source_mismatch,
    )

    pooled = {row["method"]: row for row in aggregates["pooled"]}
    condition_rows = aggregates["per_condition"]
    pooled_ranking = sorted(aggregates["pooled"], key=lambda row: (row["mean_error_m"], row["method"]))
    macro_ranking = sorted(aggregates["condition_macro"], key=lambda row: (row["macro_mean_error_m"], row["method"]))
    check([row["method"] for row in pooled_ranking] == [row["method"] for row in macro_ranking], "pooled and condition-macro mean ranking identical (balanced blocks)")

    best_per_condition = {}
    baseline_rank = {}
    for condition in conditions:
        ranked = sorted((row for row in condition_rows if row["condition"] == condition), key=lambda row: (row["mean_error_m"], row["method"]))
        best_per_condition[condition] = ranked[:5]
        baseline_rank[condition] = 1 + next(index for index, row in enumerate(ranked) if row["method"] == BASELINE)

    bootstrap_by_method = {row["candidate"]: row for row in bootstraps}
    check(len(bootstrap_by_method) == 24, "bootstrap for every non-baseline method")
    ablations = [
        "ablation_A_chamfer_inverse_top3",
        "ablation_B_temperature_softmax_only",
        "ablation_C_path_count_empty_only",
        "ablation_D_chamfer_temperature_pathcount",
    ]
    ablation_rows = [{**pooled[name], "paired_vs_corrected_majid": bootstrap_by_method[name]} for name in ablations]
    corrected = pooled[BASELINE]

    seed_metrics = []
    for method in sorted(LEARNED):
        for seed in (211, 337, 461):
            values = np.asarray([row["error_m"] for row in seed_rows if row["method"] == method and int(row["seed"]) == seed])
            seed_metrics.append({"method": method, "seed": seed, "mean_error_m": float(np.mean(values)), "rmse_m": float(np.sqrt(np.mean(values**2)))})

    best_delay = min((pooled[name] for name in DELAY_ONLY), key=lambda row: row["mean_error_m"])
    multimodal = [row for row in pooled_ranking if row["method"] not in DELAY_ONLY]
    best_multimodal = multimodal[0]
    sensor_attribution = {
        "best_delay_only": best_delay,
        "best_multimodal": best_multimodal,
        "best_multimodal_vs_best_delay_only": paired_query_block(raw, best_multimodal["method"], best_delay["method"]),
        "pointnet_multimodal_vs_delay": paired_query_block(raw, "pointnet_multimodal_direct", "pointnet_delay_direct"),
        "extratrees_multimodal_vs_delay": paired_query_block(raw, "multimodal_feature_extratrees", "delay_feature_extratrees"),
        "cross_attention_vs_corrected_majid": paired_query_block(raw, "cross_attention_multimodal", BASELINE),
        "self_attention_vs_corrected_majid": paired_query_block(raw, "candidate_self_attention_multimodal", BASELINE),
        "interpretation": (
            "The largest gains are sensor-enabled, not isolated architecture effects: multimodal PointNet and ExtraTrees each greatly beat their delay-only counterparts. "
            "Cross/self-attention also use richer tokens and therefore establish viable same-acquisition systems, not a delay-only attention ablation."
        ),
    }

    audit = {
        "schema": "corrected-mpurge-map-focused-audit-v1",
        "status": "PASS_WITH_CONCURRENT_SOURCE_MUTATION_DISCLOSED" if source_mismatch else "PASS",
        "checks": checks,
        "artifact_hash_checks": artifact_hash_results,
        "source_hash_checks": source_hash_results,
        "source_provenance_note": (
            "All result artifacts hash-match. After completion another concurrent task changed multimodal_receiver.py; "
            "the run manifest preserves the executed/frozen expected hash and the audit records the current mismatch instead of rewriting history."
            if source_mismatch else "All source and result hashes match."
        ),
        "corrected_majid_baseline": corrected,
        "ablations": ablation_rows,
        "pooled_ranking": pooled_ranking,
        "condition_macro_ranking": macro_ranking,
        "best_five_per_condition": best_per_condition,
        "corrected_majid_rank_per_condition": baseline_rank,
        "paired_bootstrap_vs_corrected_majid": bootstraps,
        "seed_coverage": seed_coverage,
        "seed_metrics": seed_metrics,
        "sensor_attribution": sensor_attribution,
        "runtime_s": results["total_runtime_s"],
        "raw_counts": results["counts"],
    }
    output = RUN / "focused_result_audit.json"
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({
        "status": audit["status"],
        "checks": len(checks),
        "pooled_top_10": pooled_ranking[:10],
        "corrected_majid": corrected,
        "ablations": ablation_rows,
        "sensor_attribution": sensor_attribution,
        "output": str(output),
    }, indent=2))


if __name__ == "__main__":
    main()
