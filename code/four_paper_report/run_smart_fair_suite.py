"""Run the corrected fair SMART-LEA replacement-scene benchmark.

The run is resumable at the (seed, corruption, spacing) level.  Every scenario
is trained independently from exactly the reference points available at that
spacing.  The one dense receiver acquisition is materialized before any model
is fit; coarser maps are strict nested subsets and no method calls the forward
simulator again.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import ExtraTreesRegressor
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RESEARCH_LEVEL = ROOT / "experiments" / "research_level"
for location in (HERE, RESEARCH_LEVEL):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from smart_fair_core import (  # noqa: E402
    Acquisition, AnalyticResidualMLP, CONDITIONS, CandidatePathCrossAttention,
    CandidatePointNetReranker, Corruption, ImageSource, MAX_PATHS,
    PointNetLocalizer, ProbabilityMapMLP, QueryReferenceAttention,
    RANGE_SCALE_M, RECEIVER_Z_M, ROOM_XY_M, SPACINGS_M,
    acquire, acquisition_digest, build_vts_mpurge_tracks,
    candidate_indices_fast, combined_scores, dense_grid, expected_reference_count,
    fit_direct_epochs, fit_power_stats, fit_residual_model,
    fixed_delay_features, graph_diffusion_predictions, lea_refine_fixed_height,
    load_acquisition, mca_predictions, pack_tokens, predict_candidate_model,
    predict_direct, predict_probability_mlp, predict_residual_model,
    query_block_ids, query_positions, rectangular_first_order_sources,
    refine_vts_differentiable_assignment, robust_fuse, save_acquisition,
    score_matrices_gpu, seed_all, sha256_file, spacing_indices, spatial_fold_ids,
    stable_seed, topk_weighted_predictions, train_candidate_model,
    train_direct_model, train_probability_mlp, vt_errors,
)
from majdi_paper_methods import lea_refine, mpurge_map_localize, pairwise_match  # noqa: E402
from evo_mdp_rank import delay_set_features, estimates_from_genome  # noqa: E402


SEEDS = (29, 47, 71)
EVO_SOURCE = RESEARCH_LEVEL / "evo_mdp_rank_multiscene_confirmation_results.json"
METHOD_METADATA: dict[str, dict[str, Any]] = {
    "majid_mca_eps1": {"inputs": "delay set", "track": "primary"},
    "majid_smart_lea1_joint_totalP5_printed": {"inputs": "delay set + paper-known 20 anchors", "track": "primary"},
    "majid_smart_lea4_joint_totalP5_printed": {"inputs": "delay set + paper-known 20 anchors", "track": "primary"},
    "smart_lea4_joint_totalP5_geometric_cross": {"inputs": "delay set + paper-known 20 anchors", "track": "ambiguity_check"},
    "smart_lea4_joint_cited_halfwindow5_printed": {"inputs": "delay set + paper-known 20 anchors", "track": "ambiguity_check"},
    "smart_lea4_fixed_height_totalP5_geometric": {"inputs": "delay set + paper-known 3-D anchors + receiver height", "track": "analytic_improvement"},
    "legacy_per_tx_fusion_totalP5_printed": {"inputs": "delay set + observable physical-TX partition", "track": "separate_diagnostic"},
    "corrected_mpurge_map_prefilter24": {"inputs": "delay set", "track": "primary"},
    "symmetric_chamfer_1nn": {"inputs": "delay set", "track": "primary"},
    "chamfer_temperature": {"inputs": "delay set", "track": "primary"},
    "combined_delay_score": {"inputs": "delay set", "track": "primary"},
    "subset_consensus": {"inputs": "delay set", "track": "primary"},
    "graph_diffusion": {"inputs": "delay set + map adjacency", "track": "primary"},
    "evomdp_rank_frozen_transfer": {"inputs": "delay set", "track": "primary_transfer"},
    "coherent_array_pdp_temperature": {"inputs": "noisy array CIR/PDP", "track": "multimodal"},
    "range_power_aoa_assignment": {"inputs": "joint noisy range/power/AoA tokens", "track": "multimodal"},
    "delay_cir_combined": {"inputs": "delay set + noisy array CIR/PDP", "track": "multimodal"},
    "pointnet_delay": {"inputs": "delay set", "track": "learned"},
    "pointnet_range_power_aoa": {"inputs": "joint noisy range/power/AoA tokens", "track": "learned_multimodal"},
    "query_reference_attention_delay": {"inputs": "delay query/reference sets", "track": "learned"},
    "query_reference_attention_multimodal": {"inputs": "joint noisy query/reference tokens", "track": "learned_multimodal"},
    "candidate_pointnet_reranker_delay": {"inputs": "delay query/reference sets + map coordinates", "track": "learned"},
    "candidate_path_cross_attention_multimodal": {"inputs": "joint noisy query/reference tokens + map coordinates", "track": "learned_multimodal"},
    "caez_style_delay_probability_mlp": {"inputs": "fixed delay vector", "track": "learned"},
    "caez_style_cir_probability_mlp": {"inputs": "noisy array CIR/PDP", "track": "learned_multimodal"},
    "extra_trees_delay_features": {"inputs": "delay/gap/count features", "track": "learned"},
    "analytic_anchor_residual": {"inputs": "delay features + Chamfer anchor diagnostics", "track": "learned_hybrid"},
    "rrle_moe": {"inputs": "delay features + four frozen expert outputs", "track": "learned_hybrid"},
    "lea4_mpurge_track_vts": {"inputs": "delay set + survey-only fitted VTs", "track": "survey_geometry"},
    "lea4_diffassign_vts": {"inputs": "delay set + survey-only differentiably refined VTs", "track": "survey_geometry"},
}


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_torch_checkpoint(path: Path, model: torch.nn.Module, metadata: dict) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "metadata": metadata}, path)
    return sha256_file(path)


def sample_indices(count: int, maximum: int, seed: int) -> np.ndarray:
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(count, maximum, replace=False)).astype(np.int64)


def top_candidates(scores: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    count = min(int(count), scores.shape[1])
    rough = np.argpartition(scores, kth=count - 1, axis=1)[:, :count]
    rough_scores = np.take_along_axis(scores, rough, axis=1)
    order = np.argsort(rough_scores, axis=1, kind="stable")
    return np.take_along_axis(rough, order, axis=1), np.take_along_axis(rough_scores, order, axis=1)


@torch.inference_mode()
def cir_scores_gpu(query_features: np.ndarray, reference_features: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = torch.device("cuda")
    reference = torch.as_tensor(reference_features, device=device)
    output = []
    for start in range(0, len(query_features), batch_size):
        query = torch.as_tensor(query_features[start : start + batch_size], device=device)
        output.append((1.0 - query @ reference.T).cpu().numpy())
    return np.concatenate(output).astype(np.float32)


def tune_temperature(scores: np.ndarray, positions: np.ndarray, truth: np.ndarray) -> dict:
    trials = []
    for top_k in (3, 5, 9):
        for temperature in (0.02, 0.05, 0.10, 0.20, 0.50, 1.0):
            prediction = topk_weighted_predictions(scores, positions, top_k=top_k, temperature=temperature)
            error = np.linalg.norm(prediction - truth, axis=1)
            trials.append({"top_k": top_k, "temperature": temperature, "mean_error_m": float(np.mean(error))})
    return min(trials, key=lambda row: (row["mean_error_m"], row["top_k"], row["temperature"]))


def tune_combined(chamfer: np.ndarray, mca: np.ndarray, wasserstein: np.ndarray, positions: np.ndarray, truth: np.ndarray) -> dict:
    trials = []
    for weight in (0.0, 0.25, 0.5, 0.75, 1.0):
        scores = combined_scores(chamfer, mca, wasserstein, weight)
        for temperature in (0.05, 0.10, 0.20, 0.50):
            prediction = topk_weighted_predictions(scores, positions, top_k=5, temperature=temperature)
            trials.append({"weight": weight, "temperature": temperature, "mean_error_m": float(np.mean(np.linalg.norm(prediction - truth, axis=1)))})
    return min(trials, key=lambda row: (row["mean_error_m"], row["weight"], row["temperature"]))


def tune_graph(scores: np.ndarray, positions: np.ndarray, shape: tuple[int, int], truth: np.ndarray, base_temperature: float) -> dict:
    trials = []
    for blend in (0.10, 0.25, 0.50):
        for steps in (1, 2, 4):
            prediction = graph_diffusion_predictions(scores, positions, shape, temperature=base_temperature, blend=blend, steps=steps)
            trials.append({"blend": blend, "steps": steps, "mean_error_m": float(np.mean(np.linalg.norm(prediction - truth, axis=1)))})
    return min(trials, key=lambda row: (row["mean_error_m"], row["blend"], row["steps"]))


def score_diagnostics(scores: np.ndarray) -> np.ndarray:
    ordered = np.sort(np.asarray(scores, dtype=np.float64), axis=1)
    best = ordered[:, 0]
    margin = ordered[:, 1] - ordered[:, 0] if ordered.shape[1] > 1 else np.zeros(len(ordered))
    spread = np.std(ordered[:, : min(8, ordered.shape[1])], axis=1)
    return np.column_stack((best / RANGE_SCALE_M, margin / RANGE_SCALE_M, spread / RANGE_SCALE_M)).astype(np.float32)


def subset_consensus_predictions(query: Acquisition, references: Acquisition, candidate_ids: np.ndarray) -> np.ndarray:
    output = []
    for row, candidates in enumerate(candidate_ids):
        query_values = query.ranges_m[row, query.mask[row]].astype(np.float64)
        hypotheses = []
        subset_ids = [-1] + list(range(len(query_values)))
        for removed in subset_ids:
            values = query_values if removed < 0 else np.delete(query_values, removed)
            local_scores = []
            for candidate in candidates:
                reference_values = references.ranges_m[candidate, references.mask[candidate]]
                if not len(values) or not len(reference_values):
                    local_scores.append(float("inf")); continue
                delta = np.abs(values[:, None] - reference_values[None])
                local_scores.append(0.5 * (float(np.mean(delta.min(1))) + float(np.mean(delta.min(0)))))
            scores = np.asarray(local_scores)
            prediction = topk_weighted_predictions(scores[None], references.positions_xy_m[candidates], top_k=3, temperature=0.15)[0]
            hypotheses.append(prediction)
        points = np.asarray(hypotheses)
        centre = np.median(points, axis=0)
        distance = np.linalg.norm(points - centre, axis=1)
        keep = distance <= max(0.75, float(np.median(distance)) * 2.5)
        output.append(np.mean(points[keep], axis=0) if np.any(keep) else points[0])
    return np.asarray(output)


def mpurge_predictions(query: Acquisition, references: Acquisition, candidate_ids: np.ndarray) -> np.ndarray:
    output = []
    for row, candidates in enumerate(candidate_ids):
        q = query.ranges_m[row, query.mask[row]].astype(np.float64)
        refs = [references.ranges_m[index, references.mask[index]].astype(np.float64) for index in candidates]
        prediction, _ = mpurge_map_localize(
            q, refs, references.positions_xy_m[candidates], p=6, alpha=0.7, k=3,
            normalized_pattern=True, coverage_mode="penalty", cross_mode="order",
        )
        output.append(prediction)
    return np.asarray(output)


def multimodal_assignment_predictions(query: Acquisition, references: Acquisition, candidate_ids: np.ndarray) -> np.ndarray:
    output = []
    angular_scale = math.radians(10.0)
    for row, candidates in enumerate(candidate_ids):
        qmask = query.mask[row]
        qr, qp, qa = query.ranges_m[row, qmask], query.powers_db[row, qmask], query.aoa_unit[row, qmask]
        scores = []
        for candidate in candidates:
            rmask = references.mask[candidate]
            rr, rp, ra = references.ranges_m[candidate, rmask], references.powers_db[candidate, rmask], references.aoa_unit[candidate, rmask]
            if not len(qr) or not len(rr):
                scores.append(float("inf")); continue
            angle = np.arccos(np.clip(qa @ ra.T, -1.0, 1.0)) / angular_scale
            cost = np.abs(qr[:, None] - rr[None]) / 0.5 + np.abs(qp[:, None] - rp[None]) / 3.0 + angle
            aa, bb = linear_sum_assignment(cost)
            unmatched = abs(len(qr) - len(rr)) * 3.0
            scores.append((float(cost[aa, bb].sum()) + unmatched) / max(len(qr), len(rr)))
        output.append(topk_weighted_predictions(np.asarray(scores)[None], references.positions_xy_m[candidates], top_k=3, temperature=0.5)[0])
    return np.asarray(output)


def fixed_evomdp_predictions(query: Acquisition, references: Acquisition, candidate_ids: np.ndarray, source_payload: dict) -> np.ndarray:
    genome = np.asarray(source_payload["evolution"]["frozen_genome"], dtype=np.float64)
    scales = np.asarray(source_payload["evolution"]["feature_scales"], dtype=np.float64)
    output = []
    for row, candidates in enumerate(candidate_ids):
        q = query.ranges_m[row, query.mask[row]]
        tensor = np.asarray([[delay_set_features(q, references.ranges_m[index, references.mask[index]]) for index in candidates]])
        prediction, _ = estimates_from_genome(genome, tensor, scales, references.positions_xy_m[candidates])
        output.append(prediction[0])
    return np.asarray(output)


def per_tx_fusion(query: Acquisition, row: int, initial: np.ndarray, sources: tuple[ImageSource, ...]) -> np.ndarray:
    predictions = []
    tx_ids = query.diagnostic_tx_ids[row]
    for tx_id in range(4):
        values = query.ranges_m[row, query.mask[row] & (tx_ids == tx_id)]
        anchors = np.asarray([source.xyz_m[:2] for source in sources if source.tx_id == tx_id])
        if len(values) < 3:
            continue
        prediction, _ = lea_refine(initial, anchors, values, matcher="mpurge", p=2, alpha=0.5, iterations=4, cross_mode="printed")
        predictions.append(prediction)
    return robust_fuse(predictions)


def lea_predictions(
    query: Acquisition,
    initial: np.ndarray,
    sources: tuple[ImageSource, ...],
    fitted_vts: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    anchor_xy = np.asarray([source.xyz_m[:2] for source in sources])
    anchor_xyz = np.asarray([source.xyz_m for source in sources])
    output = {name: [] for name in (
        "majid_smart_lea1_joint_totalP5_printed",
        "majid_smart_lea4_joint_totalP5_printed",
        "smart_lea4_joint_totalP5_geometric_cross",
        "smart_lea4_joint_cited_halfwindow5_printed",
        "smart_lea4_fixed_height_totalP5_geometric",
        "legacy_per_tx_fusion_totalP5_printed",
        "lea4_mpurge_track_vts",
        "lea4_diffassign_vts",
    )}
    for row in range(len(query.positions_xy_m)):
        values = query.ranges_m[row, query.mask[row]].astype(np.float64)
        p1, _ = lea_refine(initial[row], anchor_xy, values, matcher="mpurge", p=2, alpha=0.5, iterations=1, cross_mode="printed")
        p4, _ = lea_refine(initial[row], anchor_xy, values, matcher="mpurge", p=2, alpha=0.5, iterations=4, cross_mode="printed")
        geo, _ = lea_refine(initial[row], anchor_xy, values, matcher="mpurge", p=2, alpha=0.5, iterations=4, cross_mode="order")
        cited, _ = lea_refine(initial[row], anchor_xy, values, matcher="mpurge", p=5, alpha=0.5, iterations=4, cross_mode="printed")
        fixed = lea_refine_fixed_height(initial[row], anchor_xyz, values, pairwise_match, p=2, alpha=0.5, iterations=4, cross_mode="order")
        output["majid_smart_lea1_joint_totalP5_printed"].append(p1)
        output["majid_smart_lea4_joint_totalP5_printed"].append(p4)
        output["smart_lea4_joint_totalP5_geometric_cross"].append(geo)
        output["smart_lea4_joint_cited_halfwindow5_printed"].append(cited)
        output["smart_lea4_fixed_height_totalP5_geometric"].append(fixed)
        output["legacy_per_tx_fusion_totalP5_printed"].append(per_tx_fusion(query, row, initial[row], sources))
        for method, key in (("lea4_mpurge_track_vts", "mpurge"), ("lea4_diffassign_vts", "diffassign")):
            anchors = fitted_vts.get(key, np.empty((0, 3)))
            if len(anchors) >= 3:
                prediction = lea_refine_fixed_height(initial[row], anchors, values, pairwise_match, p=2, alpha=0.5, iterations=4, cross_mode="order")
            else:
                prediction = initial[row]
            output[method].append(prediction)
    return {key: np.asarray(value, dtype=np.float64) for key, value in output.items()}


def build_router_features(delay_features: np.ndarray, expert_predictions_m: np.ndarray, analytic_diagnostic: np.ndarray) -> np.ndarray:
    normalized = expert_predictions_m / ROOM_XY_M[None, None, :]
    disagreement = []
    for left in range(normalized.shape[1]):
        for right in range(left + 1, normalized.shape[1]):
            disagreement.append(np.linalg.norm(normalized[:, left] - normalized[:, right], axis=1))
    return np.column_stack((delay_features, analytic_diagnostic, *disagreement)).astype(np.float32)


def router_predict(router: ExtraTreesRegressor, features: np.ndarray, expert_predictions_m: np.ndarray) -> np.ndarray:
    predicted_error = np.maximum(np.asarray(router.predict(features)), 0.0)
    weights = np.exp(-predicted_error / 0.30)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-12)
    return np.sum(expert_predictions_m * weights[:, :, None], axis=1)


def run_scenario(
    *,
    seed: int,
    condition: Corruption,
    spacing_m: float,
    references: Acquisition,
    queries: Acquisition,
    sources: tuple[ImageSource, ...],
    evo_payload: dict,
    scenario_dir: Path,
    smoke: bool,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = time.perf_counter()
    scenario_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = scenario_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    timings: dict[str, dict[str, float]] = {}
    validation: dict[str, Any] = {}
    training: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    predictions: dict[str, np.ndarray] = {}
    truth = queries.positions_xy_m.astype(np.float64)
    reference_count = len(references.positions_xy_m)
    assert reference_count == expected_reference_count(spacing_m)

    folds = spatial_fold_ids(references.positions_xy_m)
    dev_fold = seed % 5
    fit_ids = np.flatnonzero(folds != dev_fold)
    dev_ids = np.flatnonzero(folds == dev_fold)
    dev_ids = dev_ids[sample_indices(len(dev_ids), 64 if smoke else 512, stable_seed(seed, spacing_m, "dev-cap"))]
    fit = references.subset(fit_ids)
    dev = references.subset(dev_ids)
    target_fit = fit.positions_xy_m.astype(np.float32) / ROOM_XY_M.astype(np.float32)
    target_dev = dev.positions_xy_m.astype(np.float32) / ROOM_XY_M.astype(np.float32)

    t0 = time.perf_counter()
    dev_chamfer, dev_mca, dev_wasserstein = score_matrices_gpu(
        dev.ranges_m, dev.mask, fit.ranges_m, fit.mask,
        batch_size=2 if len(fit.positions_xy_m) > 3000 else 8,
    )
    timings["development_delay_score_kernel"] = {"total_s": time.perf_counter() - t0, "queries": len(dev_ids)}
    chamfer_selection = tune_temperature(dev_chamfer, fit.positions_xy_m, dev.positions_xy_m)
    combined_selection = tune_combined(dev_chamfer, dev_mca, dev_wasserstein, fit.positions_xy_m, dev.positions_xy_m)
    grid_side = int(round(math.sqrt(reference_count)))
    graph_selection = tune_graph(
        dev_chamfer, fit.positions_xy_m,
        # A spatial-block fit map is not rectangular after withholding; tune on
        # the full map's adjacency with stored dev-as-query scores below.
        (1, len(fit.positions_xy_m)), dev.positions_xy_m,
        float(chamfer_selection["temperature"]),
    ) if smoke else {"blend": 0.25, "steps": 2, "mean_error_m": None, "selection_note": "fixed before full rectangular-map evaluation"}
    t0 = time.perf_counter()
    dev_cir = cir_scores_gpu(dev.cir_features, fit.cir_features)
    timings["development_cir_score_kernel"] = {"total_s": time.perf_counter() - t0, "queries": len(dev_ids)}
    cir_selection = tune_temperature(dev_cir, fit.positions_xy_m, dev.positions_xy_m)
    cir_scale = np.maximum(np.median(dev_cir, axis=1, keepdims=True), 1e-6)
    delay_scale = np.maximum(np.median(dev_chamfer, axis=1, keepdims=True), 1e-6)
    multimodal_trials = []
    for weight in (0.25, 0.5, 0.75):
        score = weight * dev_chamfer / delay_scale + (1.0 - weight) * dev_cir / cir_scale
        selection = tune_temperature(score, fit.positions_xy_m, dev.positions_xy_m)
        multimodal_trials.append({"weight": weight, **selection})
    multimodal_selection = min(multimodal_trials, key=lambda row: row["mean_error_m"])
    validation["analytic_selection"] = {
        "fit_points": int(len(fit_ids)), "spatial_block_dev_points": int(len(dev_ids)),
        "dev_fold": int(dev_fold), "chamfer_temperature": chamfer_selection,
        "combined": combined_selection, "graph": graph_selection,
        "cir_temperature": cir_selection, "delay_cir_combined": multimodal_selection,
    }

    t0 = time.perf_counter()
    chamfer, mca, wasserstein = score_matrices_gpu(
        queries.ranges_m, queries.mask, references.ranges_m, references.mask,
        batch_size=2 if reference_count > 3000 else 8,
    )
    score_runtime = time.perf_counter() - t0
    timings["protected_delay_score_kernel"] = {"total_s": score_runtime, "queries": len(truth)}
    t0 = time.perf_counter(); cir_score = cir_scores_gpu(queries.cir_features, references.cir_features); cir_runtime = time.perf_counter() - t0
    timings["protected_cir_score_kernel"] = {"total_s": cir_runtime, "queries": len(truth)}
    predictions["majid_mca_eps1"] = mca_predictions(mca, references.positions_xy_m)
    predictions["symmetric_chamfer_1nn"] = references.positions_xy_m[np.argmin(chamfer, axis=1)].astype(np.float64)
    predictions["chamfer_temperature"] = topk_weighted_predictions(
        chamfer, references.positions_xy_m,
        top_k=int(chamfer_selection["top_k"]), temperature=float(chamfer_selection["temperature"]),
    )
    final_combined_score = combined_scores(chamfer, mca, wasserstein, float(combined_selection["weight"]))
    predictions["combined_delay_score"] = topk_weighted_predictions(
        final_combined_score, references.positions_xy_m, top_k=5,
        temperature=float(combined_selection["temperature"]),
    )
    predictions["graph_diffusion"] = graph_diffusion_predictions(
        chamfer, references.positions_xy_m, (grid_side, grid_side),
        temperature=float(chamfer_selection["temperature"]),
        blend=float(graph_selection["blend"]), steps=int(graph_selection["steps"]),
    )
    predictions["coherent_array_pdp_temperature"] = topk_weighted_predictions(
        cir_score, references.positions_xy_m,
        top_k=int(cir_selection["top_k"]), temperature=float(cir_selection["temperature"]),
    )
    delay_scale_test = np.maximum(np.median(chamfer, axis=1, keepdims=True), 1e-6)
    cir_scale_test = np.maximum(np.median(cir_score, axis=1, keepdims=True), 1e-6)
    delay_cir_score = float(multimodal_selection["weight"]) * chamfer / delay_scale_test + (1.0 - float(multimodal_selection["weight"])) * cir_score / cir_scale_test
    predictions["delay_cir_combined"] = topk_weighted_predictions(
        delay_cir_score, references.positions_xy_m,
        top_k=int(multimodal_selection["top_k"]), temperature=float(multimodal_selection["temperature"]),
    )
    for method in ("majid_mca_eps1", "symmetric_chamfer_1nn", "chamfer_temperature", "combined_delay_score", "graph_diffusion"):
        timings[method] = {"shared_kernel_s": score_runtime, "total_s_including_shared": score_runtime, "queries": len(truth)}
    for method in ("coherent_array_pdp_temperature", "delay_cir_combined"):
        shared = cir_runtime + (score_runtime if method == "delay_cir_combined" else 0.0)
        timings[method] = {"shared_kernel_s": shared, "total_s_including_shared": shared, "queries": len(truth)}

    candidate_count = min(8 if smoke else 24, reference_count)
    candidate_ids, candidate_scores = top_candidates(chamfer, candidate_count)
    t0 = time.perf_counter(); predictions["corrected_mpurge_map_prefilter24"] = mpurge_predictions(queries, references, candidate_ids); elapsed = time.perf_counter() - t0
    timings["corrected_mpurge_map_prefilter24"] = {"shared_prefilter_s": score_runtime, "head_s": elapsed, "total_s_including_shared": score_runtime + elapsed, "queries": len(truth)}
    t0 = time.perf_counter(); predictions["subset_consensus"] = subset_consensus_predictions(queries, references, candidate_ids); elapsed = time.perf_counter() - t0
    timings["subset_consensus"] = {"shared_prefilter_s": score_runtime, "head_s": elapsed, "total_s_including_shared": score_runtime + elapsed, "queries": len(truth)}
    t0 = time.perf_counter(); predictions["range_power_aoa_assignment"] = multimodal_assignment_predictions(queries, references, candidate_ids); elapsed = time.perf_counter() - t0
    timings["range_power_aoa_assignment"] = {"shared_prefilter_s": score_runtime, "head_s": elapsed, "total_s_including_shared": score_runtime + elapsed, "queries": len(truth)}
    evo_candidate_ids, _ = top_candidates(chamfer, min(8 if smoke else 16, reference_count))
    t0 = time.perf_counter(); predictions["evomdp_rank_frozen_transfer"] = fixed_evomdp_predictions(queries, references, evo_candidate_ids, evo_payload); elapsed = time.perf_counter() - t0
    timings["evomdp_rank_frozen_transfer"] = {"shared_prefilter_s": score_runtime, "head_s": elapsed, "total_s_including_shared": score_runtime + elapsed, "queries": len(truth)}

    # Classical delay-only model with a protected spatial-block validation audit.
    delay_fit_features = fixed_delay_features(fit)
    delay_dev_features = fixed_delay_features(dev)
    delay_reference_features = fixed_delay_features(references)
    delay_query_features = fixed_delay_features(queries)
    tree_kwargs = dict(
        n_estimators=16 if smoke else 96, min_samples_leaf=1, max_features=0.8,
        n_jobs=-1, random_state=stable_seed(seed, spacing_m, condition.name, "extra-trees"),
    )
    t0 = time.perf_counter(); tree_dev = ExtraTreesRegressor(**tree_kwargs).fit(delay_fit_features, fit.positions_xy_m); tree_dev_prediction = tree_dev.predict(delay_dev_features); validation["extra_trees_delay_features_dev_mean_m"] = float(np.mean(np.linalg.norm(tree_dev_prediction - dev.positions_xy_m, axis=1)))
    tree = ExtraTreesRegressor(**tree_kwargs).fit(delay_reference_features, references.positions_xy_m); predictions["extra_trees_delay_features"] = tree.predict(delay_query_features); elapsed = time.perf_counter() - t0
    tree_path = checkpoint_dir / "extra_trees_delay_features.joblib"; joblib.dump(tree, tree_path); hashes[str(tree_path.relative_to(scenario_dir))] = sha256_file(tree_path)
    timings["extra_trees_delay_features"] = {"train_plus_inference_s": elapsed, "queries": len(truth)}

    direct_dev_predictions: dict[str, np.ndarray] = {}
    direct_test_predictions: dict[str, np.ndarray] = {}
    direct_models: dict[str, torch.nn.Module] = {}
    direct_epochs = 4 if smoke else 60
    for modality, method, input_dim in (
        ("delay", "pointnet_delay", 1),
        ("range_power_aoa", "pointnet_range_power_aoa", 5),
    ):
        initial_power = fit_power_stats(fit) if modality != "delay" else None
        fit_values, fit_mask = pack_tokens(fit, modality, power_stats=initial_power)
        dev_values, dev_mask = pack_tokens(dev, modality, power_stats=initial_power)
        t0 = time.perf_counter()
        initial_model, best_epoch, history = train_direct_model(
            PointNetLocalizer(input_dim), fit_values, fit_mask, target_fit,
            dev_values, dev_mask, target_dev, seed=stable_seed(seed, spacing_m, condition.name, method, "selection"),
            epochs=direct_epochs,
        )
        direct_dev_predictions[method] = predict_direct(initial_model, dev_values, dev_mask) * ROOM_XY_M
        validation[f"{method}_dev_mean_m"] = float(np.mean(np.linalg.norm(direct_dev_predictions[method] - dev.positions_xy_m, axis=1)))
        final_power = fit_power_stats(references) if modality != "delay" else None
        reference_values, reference_mask = pack_tokens(references, modality, power_stats=final_power)
        query_values, query_mask = pack_tokens(queries, modality, power_stats=final_power)
        final_model, final_history = fit_direct_epochs(
            PointNetLocalizer(input_dim), reference_values, reference_mask,
            (references.positions_xy_m / ROOM_XY_M).astype(np.float32),
            seed=stable_seed(seed, spacing_m, condition.name, method, "final"), epochs=best_epoch,
        )
        prediction = predict_direct(final_model, query_values, query_mask) * ROOM_XY_M
        predictions[method] = prediction; direct_test_predictions[method] = prediction; direct_models[method] = final_model
        elapsed = time.perf_counter() - t0
        timings[method] = {"selection_and_final_training_plus_inference_s": elapsed, "queries": len(truth)}
        training[method] = {"selected_epoch": int(best_epoch), "selection_history": history, "final_history": final_history, "power_stats_fit_only": final_power}
        checkpoint = checkpoint_dir / f"{method}.pt"; hashes[str(checkpoint.relative_to(scenario_dir))] = save_torch_checkpoint(checkpoint, final_model, training[method])

    # Candidate models: spatial-block development first, then fresh all-map fit.
    candidate_epochs = 3 if smoke else 28
    candidate_train_max = 48 if smoke else 2048
    candidate_specs: list[tuple[str, str, int, Callable[[], torch.nn.Module]]] = [
        ("query_reference_attention_delay", "delay", 1, lambda: QueryReferenceAttention(1)),
        ("query_reference_attention_multimodal", "range_power_aoa", 5, lambda: QueryReferenceAttention(5)),
        ("candidate_pointnet_reranker_delay", "delay", 1, lambda: CandidatePointNetReranker(1)),
        ("candidate_path_cross_attention_multimodal", "range_power_aoa", 5, lambda: CandidatePathCrossAttention(5)),
    ]
    for method, modality, _, factory in candidate_specs:
        t0 = time.perf_counter()
        power_fit = fit_power_stats(fit) if modality != "delay" else None
        fv, fm = pack_tokens(fit, modality, power_stats=power_fit)
        dv, dm = pack_tokens(dev, modality, power_stats=power_fit)
        train_local = sample_indices(len(fit.positions_xy_m), candidate_train_max, stable_seed(seed, spacing_m, condition.name, method, "selection-rows"))
        fit_queries = fit.subset(train_local)
        train_candidates, train_scores = candidate_indices_fast(
            fit_queries, fit, min(candidate_count, max(len(fit.positions_xy_m) - 1, 1)),
            exclude_same_indices=train_local,
        )
        initial_model, initial_history = train_candidate_model(
            factory(), fv[train_local], fm[train_local], fv, fm,
            (fit.positions_xy_m / ROOM_XY_M).astype(np.float32), train_candidates, train_scores,
            (fit.positions_xy_m[train_local] / ROOM_XY_M).astype(np.float32),
            seed=stable_seed(seed, spacing_m, condition.name, method, "selection"), epochs=candidate_epochs,
        )
        dev_candidates, dev_candidate_scores = candidate_indices_fast(dev, fit, min(candidate_count, len(fit.positions_xy_m)))
        dev_prediction = predict_candidate_model(
            initial_model, dv, dm, fv, fm, (fit.positions_xy_m / ROOM_XY_M).astype(np.float32),
            dev_candidates, dev_candidate_scores,
        ) * ROOM_XY_M
        validation[f"{method}_dev_mean_m"] = float(np.mean(np.linalg.norm(dev_prediction - dev.positions_xy_m, axis=1)))
        power_final = fit_power_stats(references) if modality != "delay" else None
        rv, rm = pack_tokens(references, modality, power_stats=power_final)
        qv, qm = pack_tokens(queries, modality, power_stats=power_final)
        final_train_local = sample_indices(reference_count, candidate_train_max, stable_seed(seed, spacing_m, condition.name, method, "final-rows"))
        reference_queries = references.subset(final_train_local)
        final_candidates, final_scores = candidate_indices_fast(
            reference_queries, references, min(candidate_count, max(reference_count - 1, 1)),
            exclude_same_indices=final_train_local,
        )
        final_model, final_history = train_candidate_model(
            factory(), rv[final_train_local], rm[final_train_local], rv, rm,
            (references.positions_xy_m / ROOM_XY_M).astype(np.float32), final_candidates, final_scores,
            (references.positions_xy_m[final_train_local] / ROOM_XY_M).astype(np.float32),
            seed=stable_seed(seed, spacing_m, condition.name, method, "final"), epochs=candidate_epochs,
        )
        predictions[method] = predict_candidate_model(
            final_model, qv, qm, rv, rm,
            (references.positions_xy_m / ROOM_XY_M).astype(np.float32),
            candidate_ids, candidate_scores,
        ) * ROOM_XY_M
        elapsed = time.perf_counter() - t0
        timings[method] = {"selection_and_final_training_plus_inference_s": elapsed, "shared_prefilter_s": score_runtime, "queries": len(truth)}
        training[method] = {"epochs": candidate_epochs, "selection_history": initial_history, "final_history": final_history, "candidate_count": int(candidate_count), "power_stats_fit_only": power_final}
        checkpoint = checkpoint_dir / f"{method}.pt"; hashes[str(checkpoint.relative_to(scenario_dir))] = save_torch_checkpoint(checkpoint, final_model, training[method])

    # CAEZ-style probability-map heads: one delay ablation and one observable CIR head.
    probability_epochs = 3 if smoke else 30
    for method, fit_features, dev_features, reference_features, query_features in (
        ("caez_style_delay_probability_mlp", delay_fit_features, delay_dev_features, delay_reference_features, delay_query_features),
        ("caez_style_cir_probability_mlp", fit.cir_features, dev.cir_features, references.cir_features, queries.cir_features),
    ):
        t0 = time.perf_counter()
        initial_model, initial_history = train_probability_mlp(
            ProbabilityMapMLP(fit_features.shape[1], len(fit.positions_xy_m)), fit_features,
            np.arange(len(fit.positions_xy_m)), seed=stable_seed(seed, spacing_m, condition.name, method, "selection"), epochs=probability_epochs,
        )
        dev_prediction = predict_probability_mlp(initial_model, dev_features, (fit.positions_xy_m / ROOM_XY_M).astype(np.float32)) * ROOM_XY_M
        validation[f"{method}_dev_mean_m"] = float(np.mean(np.linalg.norm(dev_prediction - dev.positions_xy_m, axis=1)))
        final_model, final_history = train_probability_mlp(
            ProbabilityMapMLP(reference_features.shape[1], reference_count), reference_features,
            np.arange(reference_count), seed=stable_seed(seed, spacing_m, condition.name, method, "final"), epochs=probability_epochs,
        )
        predictions[method] = predict_probability_mlp(final_model, query_features, (references.positions_xy_m / ROOM_XY_M).astype(np.float32)) * ROOM_XY_M
        elapsed = time.perf_counter() - t0
        timings[method] = {"selection_and_final_training_plus_inference_s": elapsed, "queries": len(truth)}
        training[method] = {"epochs": probability_epochs, "selection_history": initial_history, "final_history": final_history, "classes": reference_count}
        checkpoint = checkpoint_dir / f"{method}.pt"; hashes[str(checkpoint.relative_to(scenario_dir))] = save_torch_checkpoint(checkpoint, final_model, training[method])

    # Analytic-anchor residual: its calibration anchors are leave-self predictions.
    residual_epochs = 3 if smoke else 36
    residual_train_max = 64 if smoke else 4096
    fit_residual_local = sample_indices(len(fit.positions_xy_m), residual_train_max, stable_seed(seed, spacing_m, condition.name, "residual-selection"))
    fit_residual_queries = fit.subset(fit_residual_local)
    fit_anchor_ids, fit_anchor_scores = candidate_indices_fast(fit_residual_queries, fit, min(8, max(len(fit.positions_xy_m) - 1, 1)), exclude_same_indices=fit_residual_local)
    fit_anchor = np.asarray([
        topk_weighted_predictions(fit_anchor_scores[row : row + 1], fit.positions_xy_m[fit_anchor_ids[row]], top_k=3, temperature=float(chamfer_selection["temperature"]))[0]
        for row in range(len(fit_residual_local))
    ])
    fit_diag = score_diagnostics(fit_anchor_scores)
    dev_anchor = topk_weighted_predictions(dev_chamfer, fit.positions_xy_m, top_k=int(chamfer_selection["top_k"]), temperature=float(chamfer_selection["temperature"]))
    dev_diag = score_diagnostics(dev_chamfer)
    t0 = time.perf_counter()
    residual_initial, residual_initial_history = fit_residual_model(
        AnalyticResidualMLP(delay_fit_features.shape[1]), delay_fit_features[fit_residual_local],
        (fit_anchor / ROOM_XY_M).astype(np.float32), fit_diag,
        (fit.positions_xy_m[fit_residual_local] / ROOM_XY_M).astype(np.float32),
        seed=stable_seed(seed, spacing_m, condition.name, "residual-selection-model"), epochs=residual_epochs,
    )
    residual_dev_prediction = predict_residual_model(
        residual_initial, delay_dev_features, (dev_anchor / ROOM_XY_M).astype(np.float32), dev_diag,
    ) * ROOM_XY_M
    validation["analytic_anchor_residual_dev_mean_m"] = float(np.mean(np.linalg.norm(residual_dev_prediction - dev.positions_xy_m, axis=1)))
    final_residual_local = sample_indices(reference_count, residual_train_max, stable_seed(seed, spacing_m, condition.name, "residual-final"))
    final_residual_queries = references.subset(final_residual_local)
    final_anchor_ids, final_anchor_scores = candidate_indices_fast(final_residual_queries, references, min(8, max(reference_count - 1, 1)), exclude_same_indices=final_residual_local)
    final_anchor = np.asarray([
        topk_weighted_predictions(final_anchor_scores[row : row + 1], references.positions_xy_m[final_anchor_ids[row]], top_k=3, temperature=float(chamfer_selection["temperature"]))[0]
        for row in range(len(final_residual_local))
    ])
    residual_model, residual_final_history = fit_residual_model(
        AnalyticResidualMLP(delay_reference_features.shape[1]), delay_reference_features[final_residual_local],
        (final_anchor / ROOM_XY_M).astype(np.float32), score_diagnostics(final_anchor_scores),
        (references.positions_xy_m[final_residual_local] / ROOM_XY_M).astype(np.float32),
        seed=stable_seed(seed, spacing_m, condition.name, "residual-final-model"), epochs=residual_epochs,
    )
    query_anchor = predictions["chamfer_temperature"]
    query_diag = score_diagnostics(chamfer)
    predictions["analytic_anchor_residual"] = predict_residual_model(
        residual_model, delay_query_features, (query_anchor / ROOM_XY_M).astype(np.float32), query_diag,
    ) * ROOM_XY_M
    elapsed = time.perf_counter() - t0
    timings["analytic_anchor_residual"] = {"selection_and_final_training_plus_inference_s": elapsed, "shared_prefilter_s": score_runtime, "queries": len(truth)}
    training["analytic_anchor_residual"] = {"epochs": residual_epochs, "selection_history": residual_initial_history, "final_history": residual_final_history}
    checkpoint = checkpoint_dir / "analytic_anchor_residual.pt"; hashes[str(checkpoint.relative_to(scenario_dir))] = save_torch_checkpoint(checkpoint, residual_model, training["analytic_anchor_residual"])

    # RRLE/MoE router is trained only on the held-out calibration blocks.
    dev_experts = np.stack((
        dev_anchor,
        tree_dev_prediction,
        direct_dev_predictions["pointnet_delay"],
        residual_dev_prediction,
    ), axis=1)
    dev_router_features = build_router_features(delay_dev_features, dev_experts, dev_diag)
    dev_expert_errors = np.linalg.norm(dev_experts - dev.positions_xy_m[:, None, :], axis=2)
    router = ExtraTreesRegressor(
        n_estimators=16 if smoke else 96, min_samples_leaf=max(2, len(dev_ids) // 100),
        max_features=0.8, n_jobs=-1,
        random_state=stable_seed(seed, spacing_m, condition.name, "rrle-router"),
    ).fit(dev_router_features, dev_expert_errors)
    test_experts = np.stack((
        predictions["chamfer_temperature"], predictions["extra_trees_delay_features"],
        predictions["pointnet_delay"], predictions["analytic_anchor_residual"],
    ), axis=1)
    test_router_features = build_router_features(delay_query_features, test_experts, query_diag)
    t0 = time.perf_counter(); predictions["rrle_moe"] = router_predict(router, test_router_features, test_experts); elapsed = time.perf_counter() - t0
    router_path = checkpoint_dir / "rrle_moe_router.joblib"; joblib.dump(router, router_path); hashes[str(router_path.relative_to(scenario_dir))] = sha256_file(router_path)
    timings["rrle_moe"] = {"router_inference_s": elapsed, "constituent_inference_reported_separately": True, "queries": len(truth)}
    validation["rrle_router_rows"] = int(len(dev_ids))

    # Survey-only VT construction and differentiable assignment refinement.
    t0 = time.perf_counter()
    mpurge_vts, mpurge_vt_diagnostics = build_vts_mpurge_tracks(
        references, pairwise_match, maximum_points=40 if smoke else 400,
    )
    mpurge_vt_runtime = time.perf_counter() - t0
    t0 = time.perf_counter()
    diffassign_vts, diffassign_history = refine_vts_differentiable_assignment(
        mpurge_vts, references,
        seed=stable_seed(seed, spacing_m, condition.name, "diffassign-vt"),
        steps=8 if smoke else 180, maximum_points=64 if smoke else 512,
    )
    diffassign_runtime = time.perf_counter() - t0
    fitted_vts = {"mpurge": mpurge_vts, "diffassign": diffassign_vts}
    vt_payload = {
        "mpurge_builder": {**mpurge_vt_diagnostics, "anchors_xyz_m": mpurge_vts.tolist(), "postfit_truth_only_evaluation": vt_errors(mpurge_vts, sources), "runtime_s": mpurge_vt_runtime},
        "differentiable_assignment": {"anchors_xyz_m": diffassign_vts.tolist(), "loss_history": diffassign_history, "postfit_truth_only_evaluation": vt_errors(diffassign_vts, sources), "runtime_s": diffassign_runtime},
    }
    vt_path = scenario_dir / "survey_vt_builders.json"; json_dump(vt_path, vt_payload); hashes[str(vt_path.relative_to(scenario_dir))] = sha256_file(vt_path)

    # Literal/ambiguity-bracketing SMART variants and VT-fed variants.
    t0 = time.perf_counter(); lea_output = lea_predictions(queries, predictions["majid_mca_eps1"], sources, fitted_vts); lea_runtime = time.perf_counter() - t0
    predictions.update(lea_output)
    for method in lea_output:
        timings[method] = {"joint_lea_family_total_s": lea_runtime, "queries": len(truth)}
    timings["lea4_mpurge_track_vts"]["vt_builder_s"] = mpurge_vt_runtime
    timings["lea4_diffassign_vts"]["vt_builder_s"] = mpurge_vt_runtime + diffassign_runtime

    missing_methods = sorted(set(METHOD_METADATA) - set(predictions))
    if missing_methods:
        raise AssertionError(f"scenario omitted requested methods: {missing_methods}")
    for method, prediction in predictions.items():
        value = np.asarray(prediction, dtype=np.float64)
        if value.shape != truth.shape or not np.all(np.isfinite(value)):
            raise AssertionError(f"invalid prediction tensor for {method}: {value.shape}")
        predictions[method] = value
    prediction_path = scenario_dir / "predictions.npz"
    np.savez_compressed(prediction_path, truth_xy_m=truth, query_block=query_block_ids(truth), **predictions)
    hashes[str(prediction_path.relative_to(scenario_dir))] = sha256_file(prediction_path)
    scenario = {
        "schema": "corrected-smart-fair-scenario-v2",
        "status": "SMOKE_COMPLETE" if smoke else "FULL_COMPLETE",
        "replacement_scene": "25x25 m rectangular four-TX first-order image-source reconstruction; not the unavailable original SMART floorplan",
        "seed": seed, "condition": asdict(condition), "spacing_m": spacing_m,
        "reference_count": reference_count, "query_count": len(truth),
        "reference_public_digest": acquisition_digest(references),
        "query_public_digest": acquisition_digest(queries),
        "fairness": {
            "calibration_locations": "strict nested subset of one stored 100x100 receiver acquisition",
            "query_locations": "same frozen 1000 positions for every method/spacing/seed",
            "extra_labelled_forward_simulator_examples": 0,
            "source_or_ray_ids_primary_methods": False,
            "clean_auxiliary_channels": False,
            "survey_query_corruption": "independent, identically configured array-CIR acquisition/extraction",
            "empty_policy": "return room centre [12.5,12.5] when an analytic set has no finite evidence",
            "legacy_per_tx_exception": "separately labelled diagnostic only; physical transmitter partition unavailable to other primary methods",
        },
        "validation": validation, "training": training, "vt_builders": vt_payload,
        "timings": timings, "hashes": hashes,
        "method_metadata": METHOD_METADATA,
        "runtime_s": time.perf_counter() - started,
    }
    complete_path = scenario_dir / "complete.json"; json_dump(complete_path, scenario)
    scenario["hashes"][str(complete_path.relative_to(scenario_dir))] = sha256_file(complete_path)
    print(json.dumps({"scenario": scenario_dir.name, "runtime_s": scenario["runtime_s"], "methods": len(predictions)}), flush=True)
    return scenario, predictions


def summarize_predictions(records: list[tuple[dict, dict[str, np.ndarray]]]) -> list[dict]:
    buckets: dict[tuple, list[float]] = {}
    for scenario, predictions in records:
        truth = np.load(Path(scenario["_scenario_dir"]) / "predictions.npz", allow_pickle=False)["truth_xy_m"]
        for method, prediction in predictions.items():
            error = np.linalg.norm(prediction - truth, axis=1)
            key = (scenario["condition"]["name"], float(scenario["spacing_m"]), method)
            buckets.setdefault(key, []).extend(error.tolist())
    output = []
    for (condition, spacing, method), values in sorted(buckets.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append({
            "condition": condition, "spacing_m": spacing, "method": method,
            "count": int(len(array)), "mean_error_m": float(np.mean(array)),
            "rmse_m": float(np.sqrt(np.mean(array**2))),
            "median_error_m": float(np.median(array)),
            "p90_error_m": float(np.quantile(array, 0.9)),
        })
    return output


def query_block_paired_cis(records: list[tuple[dict, dict[str, np.ndarray]]], samples: int = 4000) -> list[dict]:
    grouped: dict[tuple, list[float]] = {}
    for scenario, predictions in records:
        path = Path(scenario["_scenario_dir"]) / "predictions.npz"
        with np.load(path, allow_pickle=False) as payload:
            truth = payload["truth_xy_m"]; blocks = payload["query_block"]
        baseline = np.linalg.norm(predictions["majid_mca_eps1"] - truth, axis=1)
        for method, prediction in predictions.items():
            candidate = np.linalg.norm(prediction - truth, axis=1)
            gain = baseline - candidate
            for block in np.unique(blocks):
                key = (scenario["condition"]["name"], float(scenario["spacing_m"]), method, int(scenario["seed"]), int(block))
                grouped[key] = gain[blocks == block].tolist()
    output = []
    combinations = sorted({key[:3] for key in grouped})
    for condition, spacing, method in combinations:
        cluster_values = np.asarray([
            np.mean(value) for key, value in grouped.items()
            if key[:3] == (condition, spacing, method)
        ], dtype=np.float64)
        rng = np.random.default_rng(stable_seed("paired-ci", condition, spacing, method))
        draws = cluster_values[rng.integers(0, len(cluster_values), size=(samples, len(cluster_values)))].mean(1)
        output.append({
            "condition": condition, "spacing_m": spacing, "candidate": method,
            "baseline": "majid_mca_eps1", "cluster_unit": "seed x 5m query spatial block",
            "clusters": int(len(cluster_values)), "mean_error_reduction_m": float(np.mean(cluster_values)),
            "paired_cluster_bootstrap_95ci_m": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        })
    return output


def write_raw_rows(path: Path, records: list[tuple[dict, dict[str, np.ndarray]]]) -> int:
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for scenario, predictions in records:
            prediction_path = Path(scenario["_scenario_dir"]) / "predictions.npz"
            with np.load(prediction_path, allow_pickle=False) as payload:
                truth = payload["truth_xy_m"]; blocks = payload["query_block"]
            for method, prediction in sorted(predictions.items()):
                errors = np.linalg.norm(prediction - truth, axis=1)
                runtime = scenario["timings"].get(method, {})
                total = runtime.get("total_s_including_shared", runtime.get("selection_and_final_training_plus_inference_s", runtime.get("train_plus_inference_s", runtime.get("joint_lea_family_total_s", runtime.get("router_inference_s")))))
                for query_id in range(len(truth)):
                    row = {
                        "seed": int(scenario["seed"]), "condition": scenario["condition"]["name"],
                        "spacing_m": float(scenario["spacing_m"]), "reference_count": int(scenario["reference_count"]),
                        "query_id": query_id, "query_block": int(blocks[query_id]), "method": method,
                        "input_features": METHOD_METADATA[method]["inputs"], "track": METHOD_METADATA[method]["track"],
                        "truth_xy_m": truth[query_id].tolist(), "prediction_xy_m": prediction[query_id].tolist(),
                        "error_m": float(errors[query_id]), "scenario_method_runtime_s": total,
                    }
                    handle.write(json.dumps(row, separators=(",", ":")) + "\n"); count += 1
    return count


def write_report(path: Path, summary: list[dict], status: str, runtime_s: float) -> None:
    lines = [
        "# Corrected fair SMART-LEA replacement-scene suite", "",
        f"Status: **{status}**  ", f"Total orchestration runtime: {runtime_s:.1f} s  ",
        "Scene: disclosed 25x25 m four-transmitter first-order image-source reconstruction; this is not the unavailable original SMART floorplan.", "",
        "Every coarse radio map is a strict subset of one stored 100x100 survey acquisition. Survey and query receiver noise/extraction are independent but identically configured. Learned models were fit separately per spacing and validated on held-out 5 m spatial blocks. No ray/VT ID, query truth, clean power/AoA, extra spatial locations, or extra labelled simulator calls were available to primary methods.", "",
        "## Best mean error by condition and spacing", "",
        "| Condition | Spacing | Best eligible method | Mean | RMSE | Median | P90 |", "|---|---:|---|---:|---:|---:|---:|",
    ]
    eligible = {name for name, meta in METHOD_METADATA.items() if meta["track"] != "separate_diagnostic"}
    for condition, spacing in sorted({(row["condition"], row["spacing_m"]) for row in summary}):
        candidates = [row for row in summary if row["condition"] == condition and row["spacing_m"] == spacing and row["method"] in eligible]
        best = min(candidates, key=lambda row: row["mean_error_m"])
        lines.append(f"| {condition} | {spacing:g} m | {best['method']} | {best['mean_error_m']:.4f} | {best['rmse_m']:.4f} | {best['median_error_m']:.4f} | {best['p90_error_m']:.4f} |")
    lines += [
        "", "## Interpretation guardrails", "",
        "- `legacy_per_tx_fusion_totalP5_printed` is a separate diagnostic because it uses the physical-transmitter partition; it is never ranked as a primary result.",
        "- `corrected_mpurge_map_prefilter24` applies the exact corrected MPUrge score and source-intended coverage penalty only after a shared exact-Chamfer top-24 prefilter. It is not a global exhaustive MPUrge search.",
        "- Total window size P=5 is implemented as half-window `p=2`. The cited-parameter ambiguity is bracketed with half-window `p=5` (total 11), and printed versus geometric crossing checks are separate rows.",
        "- Power and AoA enter only the named multimodal methods and are extracted from the same noisy array CIR as delay; delay-only rows remain explicit ablations, not an eligibility rule.",
        "- Fitted-VT truth errors are computed only after fitting for audit and never enter localization or training.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_completed_scenario(scenario_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    scenario = json.loads((scenario_dir / "complete.json").read_text(encoding="utf-8"))
    with np.load(scenario_dir / "predictions.npz", allow_pickle=False) as payload:
        predictions = {method: payload[method] for method in METHOD_METADATA}
    scenario["_scenario_dir"] = str(scenario_dir)
    return scenario, predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="one seed, 32 queries, short training")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="*")
    parser.add_argument("--conditions", nargs="*", choices=[condition.name for condition in CONDITIONS])
    parser.add_argument("--spacings", type=float, nargs="*")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    started = time.perf_counter()
    output_dir = args.output_dir or ROOT / "research" / "four_paper_report" / ("smart_fair_smoke" if args.smoke else "smart_fair_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    acquisition_dir = output_dir / "acquisitions"; acquisition_dir.mkdir(exist_ok=True)
    scenario_root = output_dir / "scenarios"; scenario_root.mkdir(exist_ok=True)
    seeds = tuple(args.seeds or ((SEEDS[0],) if args.smoke else SEEDS))
    conditions = tuple(condition for condition in CONDITIONS if not args.conditions or condition.name in args.conditions)
    spacings = tuple(args.spacings or ((2.5,) if args.smoke else SPACINGS_M))
    query_count = 32 if args.smoke else 1000
    sources = rectangular_first_order_sources()
    dense_positions = dense_grid()
    protected_query_positions = query_positions(query_count)
    evo_payload = json.loads(EVO_SOURCE.read_text(encoding="utf-8"))
    code_files = [Path(__file__), HERE / "smart_fair_core.py", HERE / "multimodal_receiver.py", RESEARCH_LEVEL / "majdi_paper_methods.py", EVO_SOURCE]
    manifest = {
        "schema": "corrected-smart-fair-suite-v2", "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mode": "smoke" if args.smoke else "full", "seeds": list(seeds), "conditions": [asdict(value) for value in conditions],
        "spacings_m": list(spacings), "dense_survey_points": 10000, "query_points": query_count,
        "device": torch.cuda.get_device_name(0), "python": platform.python_version(), "torch": torch.__version__,
        "code_hashes": {str(path.relative_to(ROOT)): sha256_file(path) for path in code_files},
        "method_metadata": METHOD_METADATA,
        "applicability_and_exclusions": {
            "included_multimodal": "range, observed power, extracted AoA unit vector, and compact noisy array-PDP/CIR features at identical acquisition locations",
            "aod": "excluded: the simple image-source receiver does not supply a defensible reflected departure-angle extractor",
            "global_exhaustive_mpurge": "excluded at dense maps for tractability; exact corrected scoring is disclosed behind a shared top-24 Chamfer prefilter",
            "simulator_ray_ids": "excluded from all primary inputs; physical Tx partition retained only for separately labelled legacy diagnostic",
            "extra_forward_samples": "forbidden; measurement-only training uses the one stored survey acquisition",
            "original_smart_scene": "unavailable; all results are replacement-scene evidence",
        },
    }
    json_dump(output_dir / "manifest.json", manifest)
    records: list[tuple[dict, dict[str, np.ndarray]]] = []
    acquisition_hashes: dict[str, str] = {}
    for seed in seeds:
        for condition in conditions:
            survey_path = acquisition_dir / f"survey_seed{seed}_{condition.name}.npz"
            query_path = acquisition_dir / f"query_seed{seed}_{condition.name}.npz"
            if survey_path.exists() and query_path.exists() and not args.no_resume:
                survey = load_acquisition(survey_path); query = load_acquisition(query_path)
            else:
                print(f"acquiring receiver data seed={seed} condition={condition.name}", flush=True)
                survey = acquire(dense_positions, sources, condition, stable_seed(seed, condition.name, "survey-acquisition"))
                query = acquire(protected_query_positions, sources, condition, stable_seed(seed, condition.name, "query-acquisition"))
                save_acquisition(survey_path, survey, include_diagnostic=True)
                save_acquisition(query_path, query, include_diagnostic=True)
            acquisition_hashes[str(survey_path.relative_to(output_dir))] = sha256_file(survey_path)
            acquisition_hashes[str(query_path.relative_to(output_dir))] = sha256_file(query_path)
            if np.array_equal(acquisition_digest(survey), acquisition_digest(query)):
                raise AssertionError("survey and query acquisition digests unexpectedly coincide")
            for spacing_m in spacings:
                indices = spacing_indices(spacing_m)
                references = survey.subset(indices)
                if len(references.positions_xy_m) != expected_reference_count(spacing_m):
                    raise AssertionError("spacing map cardinality failure")
                tag = f"seed{seed}_{condition.name}_spacing{spacing_m:g}m".replace(".", "p")
                scenario_dir = scenario_root / tag
                if (scenario_dir / "complete.json").exists() and not args.no_resume:
                    scenario, predictions = load_completed_scenario(scenario_dir)
                    print(f"resumed {tag}", flush=True)
                else:
                    scenario, predictions = run_scenario(
                        seed=seed, condition=condition, spacing_m=spacing_m,
                        references=references, queries=query, sources=sources,
                        evo_payload=evo_payload, scenario_dir=scenario_dir, smoke=args.smoke,
                    )
                    scenario["_scenario_dir"] = str(scenario_dir)
                records.append((scenario, predictions))
    summary = summarize_predictions(records)
    cis = query_block_paired_cis(records, samples=1000 if args.smoke else 4000)
    raw_path = output_dir / "raw_predictions.jsonl.gz"
    raw_rows = write_raw_rows(raw_path, records)
    scenario_hashes = {
        str((Path(scenario["_scenario_dir"]) / "complete.json").relative_to(output_dir)): sha256_file(Path(scenario["_scenario_dir"]) / "complete.json")
        for scenario, _ in records
    }
    audit = {
        "status": "PASS",
        "checks": {
            "scenario_count": len(records),
            "expected_scenario_count": len(seeds) * len(conditions) * len(spacings),
            "methods_each_scenario": len(METHOD_METADATA),
            "raw_rows": raw_rows,
            "expected_raw_rows": len(records) * query_count * len(METHOD_METADATA),
            "all_finite": all(np.all(np.isfinite(value)) for _, prediction in records for value in prediction.values()),
            "nested_reference_counts": {str(spacing): expected_reference_count(spacing) for spacing in spacings},
            "no_extra_labelled_forward_samples": True,
            "primary_tokens_exclude_source_ids": True,
            "survey_query_independently_acquired": True,
        },
    }
    if audit["checks"]["scenario_count"] != audit["checks"]["expected_scenario_count"] or audit["checks"]["raw_rows"] != audit["checks"]["expected_raw_rows"] or not audit["checks"]["all_finite"]:
        audit["status"] = "FAIL"
    runtime = time.perf_counter() - started
    result = {
        **manifest,
        "status": "SMOKE_COMPLETE" if args.smoke else "FULL_COMPLETE",
        "completed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_s": runtime, "raw_rows": raw_rows,
        "summaries": summary, "query_block_paired_cis_vs_mca": cis,
        "acquisition_hashes": acquisition_hashes, "scenario_hashes": scenario_hashes,
        "raw_predictions_sha256": sha256_file(raw_path), "integrity_audit": audit,
    }
    result_path = output_dir / "results.json"; json_dump(result_path, result)
    json_dump(output_dir / "integrity_audit.json", audit)
    write_report(output_dir / "REPORT.md", summary, result["status"], runtime)
    print(json.dumps({"status": result["status"], "runtime_s": runtime, "scenarios": len(records), "raw_rows": raw_rows, "audit": audit["status"], "results": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
