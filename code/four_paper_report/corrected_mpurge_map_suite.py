"""Corrected, acquisition-budget-fair MPUrge-MAP replacement-scene suite.

This is deliberately a new implementation.  It does not modify the earlier
paper-reconstruction scripts or their artifacts.  The suite materializes one
replacement scene at the paper's 80 survey locations and 100 locked off-grid
queries.  Every condition has one survey acquisition tensor and one
independently corrupted query tensor, shared verbatim by every applicable
method.  Learned augmentation only corrupts those 80 stored survey records;
it never calls the simulator at a new coordinate.

The primary fairness constraint is acquisition location/map budget.  Delay,
received power, arrival direction and complex MPC coefficient are physically
measurable channels emitted by the replacement simulator and may be used when
the method declares them.  No ray/source identities, reflection order, VT
labels, query coordinates, clean auxiliary channels, or dense spatial samples
are exposed to a localizer.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from typing import Callable, Iterable, Sequence

import joblib
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesRegressor

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER_DIR = ROOT / "experiments" / "paper_protocol_replications"
RESEARCH_LEVEL = ROOT / "experiments" / "research_level"
for _path in (HERE, PAPER_DIR, RESEARCH_LEVEL):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from common import rectangular_sources, stable_seed as legacy_seed  # noqa: E402
from majdi_paper_methods import Match, dissimilarity_matrix, size_unify  # noqa: E402
from multimodal_receiver import (  # noqa: E402
    MultimodalFingerprint,
    coherent_adp_distance,
    corrupt_stored_fingerprint,
    simulate_multimodal_fingerprint,
)


SCHEMA = "corrected-fair-mpurge-map-replacement-v2-multimodal"
ROOM = np.asarray([4.0, 5.0, 3.0], dtype=np.float64)
ROOM_DIAGONAL = float(np.linalg.norm(ROOM[:2]))
AP = np.asarray([[1.1, 1.7, 2.5]], dtype=np.float64)
RECEIVER_Z = 1.2
MAX_PATHS = 9
CARRIER_HZ = 60.0e9
C_MPS = 299_792_458.0
NOISE_STDS_M = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
LEARNED_SEEDS = (211, 337, 461)
EVO_FEATURE_NAMES = (
    "symmetric_chamfer",
    "quantile_wasserstein",
    "gap_spectrum_wasserstein",
    "log_cardinality_ratio",
    "partial_assignment_with_unmatched",
    "normalized_all_pairs_pattern",
    "mca_similarity_complement",
    "mean_and_scale_difference",
)
EVO_SCALES = np.asarray(
    [3.704226858503369, 5.782882857737321, 2.2543017669874876,
     0.3794896217049037, 3.40041515104709, 1.9615522684722464,
     0.9157972926152402, 8.085794019213322], dtype=np.float64,
)
EVO_WEIGHTS = np.asarray(
    [0.2601700412839441, 0.004335138750150508, 0.08742117274951308,
     0.07275433360025367, 0.0013673586862682236, 0.11085703866267362,
     0.4579854287797303, 0.005109487487466361], dtype=np.float64,
)
EVO_TEMPERATURE = 0.05256257099306191
EVO_TOP_K = 3

OBJECTS = (
    np.asarray([[2, 2, 1], [2, 2, 2], [2, 3, 2], [2, 3, 1]], dtype=np.float64),
    np.asarray([[1, 3, 0], [1, 3, 1], [3, 3, 1], [3, 3, 0]], dtype=np.float64),
    np.asarray([[0, 0, 1], [0, 1, 1], [4, 1, 1], [4, 0, 1]], dtype=np.float64),
    np.asarray([[0, 4, 3], [0, 5, 2], [2, 5, 2], [2, 4, 3]], dtype=np.float64),
    np.asarray([[3, 5, 0], [4, 3, 0], [4, 5, 3], [4, 5, 2]], dtype=np.float64),
)


def stable_seed(*parts: object) -> int:
    payload = ":".join(map(str, (SCHEMA, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(value), indent=2), encoding="utf-8")


def configure_device(require_cuda: bool = True) -> torch.device:
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full corrected suite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reference_positions() -> np.ndarray:
    xs = np.arange(0.25, 4.0, 0.5)
    ys = np.arange(0.25, 5.0, 0.5)
    return np.asarray([[x, y] for y in ys for x in xs], dtype=np.float64)


def locked_query_positions(count: int) -> np.ndarray:
    # Preserve the earlier replacement-scene query realization exactly.
    rng = np.random.default_rng(legacy_seed("mpurge-map-test"))
    references = reference_positions()
    output: list[np.ndarray] = []
    while len(output) < count:
        point = rng.uniform([0.05, 0.05], [3.95, 4.95])
        if float(np.min(np.linalg.norm(references - point, axis=1))) > 0.035:
            output.append(point)
    return np.asarray(output, dtype=np.float64)


@dataclass(frozen=True)
class Condition:
    name: str
    delay_sigma_m: float
    objects: tuple[np.ndarray, ...]


def conditions(quick: bool) -> list[Condition]:
    rows = [Condition("ideal", 0.0, tuple())]
    rows.extend(Condition(f"awgn_sigma_{sigma:g}m", sigma, tuple()) for sigma in NOISE_STDS_M)
    rows.extend(Condition(f"object_{index + 1}", 0.0, (quad,)) for index, quad in enumerate(OBJECTS))
    rows.append(Condition("all_five_objects", 0.0, OBJECTS))
    if quick:
        return [rows[0], rows[4], rows[-1]]
    return rows


def _pack_observations(rows: Sequence[MultimodalFingerprint]) -> dict[str, np.ndarray]:
    count = len(rows)
    delay = np.zeros((count, MAX_PATHS), dtype=np.float32)
    power = np.zeros((count, MAX_PATHS), dtype=np.float32)
    aoa = np.zeros((count, MAX_PATHS, 3), dtype=np.float32)
    cir = np.zeros((count, 1, 8, 256), dtype=np.complex64)
    mask = np.zeros((count, MAX_PATHS), dtype=bool)
    for row, observation in enumerate(rows):
        n = min(len(observation), MAX_PATHS)
        if n:
            delay[row, :n] = observation.ranges_m[:n]
            power[row, :n] = observation.powers_db[:n]
            aoa[row, :n] = observation.aoa_unit[:n]
            mask[row, :n] = True
        if observation.cir.size:
            block = observation.cir[:1, :8, :256]
            cir[row, :block.shape[0], :block.shape[1], :block.shape[2]] = block
    return {"delay": delay, "power": power, "aoa": aoa, "cir": cir, "mask": mask}


def materialize_protocol(path: Path, *, quick: bool) -> dict:
    """The only routine allowed to invoke the replacement simulator."""

    sources = rectangular_sources(ROOM, AP, maximum_order=2, include_floor_ceiling=True)
    references = reference_positions()
    queries = locked_query_positions(12 if quick else 100)
    condition_rows = conditions(quick)
    survey_packed, query_packed = [], []
    for condition in condition_rows:
        survey = []
        query = []
        for index, xy in enumerate(references):
            # Obstacles follow the printed evaluation convention: calibration
            # is performed before the query obstruction is introduced.
            rng = np.random.default_rng(stable_seed("survey", condition.name, index))
            survey.append(simulate_multimodal_fingerprint(
                [xy[0], xy[1], RECEIVER_Z], sources, rng=rng,
                maximum_paths=MAX_PATHS,
                snr_db=80.0 if condition.delay_sigma_m == 0.0 else 25.0,
                range_noise_std_m=condition.delay_sigma_m,
                obstructions=tuple(), separate_transmitters=False,
            ))
        for index, xy in enumerate(queries):
            rng = np.random.default_rng(stable_seed("query", condition.name, index))
            query.append(simulate_multimodal_fingerprint(
                [xy[0], xy[1], RECEIVER_Z], sources, rng=rng,
                maximum_paths=MAX_PATHS,
                snr_db=80.0 if condition.delay_sigma_m == 0.0 else 25.0,
                range_noise_std_m=condition.delay_sigma_m,
                obstructions=condition.objects, separate_transmitters=False,
            ))
        survey_packed.append(_pack_observations(survey))
        query_packed.append(_pack_observations(query))

    payload: dict[str, np.ndarray] = {
        "condition_names": np.asarray([row.name for row in condition_rows]),
        "condition_delay_sigma_m": np.asarray([row.delay_sigma_m for row in condition_rows]),
        "reference_xy_m": references,
        "query_xy_m": queries,
    }
    for prefix, blocks in (("survey", survey_packed), ("query", query_packed)):
        for field in ("delay", "power", "aoa", "cir", "mask"):
            payload[f"{prefix}_{field}"] = np.stack([block[field] for block in blocks])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "path": str(path), "sha256": sha256(path), "conditions": len(condition_rows),
        "survey_locations": len(references), "locked_queries": len(queries),
        "simulator_calls": int(len(condition_rows) * (len(references) + len(queries))),
        "new_spatial_training_locations": 0,
        "observable_channels": ["delay_m", "received_power_db", "arrival_unit_vector", "noisy_8_element_array_CIR"],
        "forbidden_not_stored": ["ray_id", "source_id", "reflection_order", "VT_id", "query_truth_as_input"],
    }


def unpack_set(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64)[np.asarray(mask, dtype=bool)]


def centroid(reference_xy: np.ndarray) -> np.ndarray:
    return np.mean(np.asarray(reference_xy, dtype=np.float64), axis=0)


def corrected_pairwise_match(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    half_window_p: int,
    alpha: float,
    normalized_pattern: bool,
) -> list[Match]:
    """Prose-consistent matching with composite-dissimilarity conflict order."""

    a, b, ia, ib = size_unify(values_a, values_b, norm="l1")
    accepted: list[Match] = []
    iteration = 0

    def crosses(left: Match, right: Match) -> bool:
        return (left.value_a - right.value_a) * (left.value_b - right.value_b) < 0.0

    while len(a) and len(b):
        delta = dissimilarity_matrix(
            a, b, p=half_window_p, alpha=alpha,
            normalized_pattern=normalized_pattern,
        )
        row_best = np.argmin(delta, axis=1)
        col_best = np.argmin(delta, axis=0)
        potential_indices = [(i, int(j)) for i, j in enumerate(row_best) if int(col_best[int(j)]) == i]
        potential = [
            Match(int(ia[i]), int(ib[j]), float(a[i]), float(b[j]),
                  float(delta[i, j]), iteration)
            for i, j in potential_indices
        ]
        remove_a = {i for i, _ in potential_indices}
        remove_b = {j for _, j in potential_indices}
        keep_a = np.asarray([i not in remove_a for i in range(len(a))])
        keep_b = np.asarray([j not in remove_b for j in range(len(b))])
        a, ia = a[keep_a], ia[keep_a]
        b, ib = b[keep_b], ib[keep_b]
        prior_ok = [candidate for candidate in potential if not any(crosses(candidate, prior) for prior in accepted)]
        current: list[Match] = []
        for candidate in sorted(prior_ok, key=lambda item: (item.dissimilarity, item.index_a, item.index_b)):
            if not any(crosses(candidate, other) for other in current):
                current.append(candidate)
        accepted.extend(current)
        iteration += 1
    return accepted


def corrected_mpurge_scores(
    query: np.ndarray,
    references: Sequence[np.ndarray],
    *,
    half_window_p: int = 6,
    alpha: float = 0.7,
    normalized_pattern: bool = True,
    coverage_penalty: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.sort(np.asarray(query, dtype=np.float64))
    scores = np.full(len(references), np.inf, dtype=np.float64)
    coverage = np.zeros(len(references), dtype=np.float64)
    for index, reference in enumerate(references):
        reference = np.sort(np.asarray(reference, dtype=np.float64))
        matches = corrected_pairwise_match(
            query, reference, half_window_p=half_window_p, alpha=alpha,
            normalized_pattern=normalized_pattern,
        )
        if not matches:
            continue
        mean = float(np.mean([item.dissimilarity for item in matches]))
        coverage[index] = len(matches) / max(len(query), len(reference), 1)
        scores[index] = mean / max(coverage[index], np.finfo(np.float64).eps) if coverage_penalty else mean
    return scores, coverage


def inverse_topk(scores: np.ndarray, reference_xy: np.ndarray, k: int = 3) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return centroid(reference_xy)
    selected = finite[np.argsort(scores[finite], kind="stable")[:min(k, len(finite))]]
    values = np.maximum(scores[selected], np.finfo(np.float64).eps)
    weights = 1.0 / values
    weights /= weights.sum()
    return np.sum(reference_xy[selected] * weights[:, None], axis=0)


def softmax_topk(scores: np.ndarray, reference_xy: np.ndarray, temperature: float, k: int = 3) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return centroid(reference_xy)
    selected = finite[np.argsort(scores[finite], kind="stable")[:min(k, len(finite))]]
    relative = scores[selected] - np.min(scores[selected])
    weights = np.exp(-np.clip(relative / max(temperature, 1.0e-9), 0.0, 700.0))
    weights /= weights.sum()
    return np.sum(reference_xy[selected] * weights[:, None], axis=0)


def symmetric_chamfer(query: np.ndarray, reference: np.ndarray, *, empty_cost: float = np.inf) -> float:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not len(query) and not len(reference):
        return 0.0
    if not len(query) or not len(reference):
        return float(empty_cost)
    residual = np.abs(query[:, None] - reference[None, :])
    return 0.5 * (float(np.mean(np.min(residual, axis=1))) + float(np.mean(np.min(residual, axis=0))))


def chamfer_scores(query: np.ndarray, references: Sequence[np.ndarray], *, path_count_rule: float | None = None) -> np.ndarray:
    output = []
    for reference in references:
        if path_count_rule is None:
            value = symmetric_chamfer(query, reference)
        else:
            empty_cost = ROOM_DIAGONAL + path_count_rule * abs(len(query) - len(reference))
            value = symmetric_chamfer(query, reference, empty_cost=empty_cost)
            value += path_count_rule * abs(len(query) - len(reference)) / max(len(query), len(reference), 1)
        output.append(value)
    return np.asarray(output, dtype=np.float64)


def path_count_adjusted(scores: np.ndarray, query: np.ndarray, references: Sequence[np.ndarray], strength: float) -> np.ndarray:
    adjusted = np.asarray(scores, dtype=np.float64).copy()
    for index, reference in enumerate(references):
        mismatch = abs(len(query) - len(reference))
        if np.isfinite(adjusted[index]):
            adjusted[index] *= 1.0 + strength * mismatch / max(len(query), len(reference), 1)
        elif len(query) and not len(reference):
            adjusted[index] = ROOM_DIAGONAL + strength * mismatch
    return adjusted


def mca_predict(query: np.ndarray, references: Sequence[np.ndarray], reference_xy: np.ndarray, epsilon_m: float = 1.6) -> np.ndarray:
    if not len(query):
        return centroid(reference_xy)
    similarity = np.zeros(len(references), dtype=np.float64)
    for index, reference in enumerate(references):
        if not len(reference):
            continue
        nearest = np.min(np.abs(query[:, None] - reference[None, :]), axis=1)
        accepted = nearest < epsilon_m
        similarity[index] = np.sum((epsilon_m - nearest[accepted]) ** 2)
    if not np.any(similarity > 0.0):
        return centroid(reference_xy)
    return reference_xy[int(np.argmax(similarity))].copy()


def subset_consensus(query: np.ndarray, references: Sequence[np.ndarray], reference_xy: np.ndarray) -> np.ndarray:
    if not len(query):
        return centroid(reference_xy)
    hypotheses = [inverse_topk(chamfer_scores(query, references), reference_xy)]
    if len(query) <= 6:
        width = min(3, len(query))
        subsets = [query[np.asarray(items)] for items in itertools.combinations(range(len(query)), width)]
    else:
        subsets = [np.delete(query, index) for index in range(len(query))]
    for subset in subsets:
        hypotheses.append(inverse_topk(chamfer_scores(subset, references), reference_xy))
    points = np.asarray(hypotheses)
    middle = np.median(points, axis=0)
    distance = np.linalg.norm(points - middle, axis=1)
    keep = distance <= max(0.75, 2.5 * float(np.median(distance)))
    return np.mean(points[keep], axis=0) if np.any(keep) else hypotheses[0]


def _quantile_distance(a: np.ndarray, b: np.ndarray, count: int) -> float:
    if not len(a) or not len(b):
        return 30.0
    q = np.linspace(0.0, 1.0, count)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def evo_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if not len(a) or not len(b):
        return np.full(len(EVO_FEATURE_NAMES), 30.0 + abs(len(a) - len(b)), dtype=np.float64)
    residual = np.abs(a[:, None] - b[None, :])
    chamfer = 0.5 * (float(np.mean(np.min(residual, axis=1))) + float(np.mean(np.min(residual, axis=0))))
    wasserstein = _quantile_distance(a, b, 17)
    ga, gb = np.diff(a), np.diff(b)
    gap = _quantile_distance(ga, gb, 11) if len(ga) and len(gb) else wasserstein
    cardinality = abs(float(np.log((len(a) + 0.5) / (len(b) + 0.5))))
    rows, columns = linear_sum_assignment(residual)
    assignment = (float(np.sum(residual[rows, columns])) + 1.5 * abs(len(a) - len(b))) / max(len(a), len(b))
    pattern_matrix = dissimilarity_matrix(a, b, p=3, alpha=0.0, normalized_pattern=True)
    pattern = 0.5 * (float(np.mean(np.min(pattern_matrix, axis=1))) + float(np.mean(np.min(pattern_matrix, axis=0))))
    nearest = np.min(residual, axis=1)
    accepted = nearest < 1.6
    similarity = float(np.sum((1.6 - nearest[accepted]) ** 2))
    mca_complement = max(0.0, 1.0 - similarity / (len(a) * 1.6**2))
    moments = abs(float(np.mean(a) - np.mean(b))) + abs(float(np.std(a) - np.std(b)))
    return np.asarray([chamfer, wasserstein, gap, cardinality, assignment, pattern, mca_complement, moments])


def evo_scores(query: np.ndarray, references: Sequence[np.ndarray]) -> np.ndarray:
    return np.asarray([np.sum((evo_features(query, reference) / EVO_SCALES) * EVO_WEIGHTS) for reference in references])


def graph_transition(reference_xy: np.ndarray, neighbours: int = 6) -> np.ndarray:
    distance = np.linalg.norm(reference_xy[:, None, :] - reference_xy[None, :, :], axis=2)
    matrix = np.zeros_like(distance)
    for row in range(len(distance)):
        chosen = np.argsort(distance[row], kind="stable")[1:neighbours + 1]
        weights = np.exp(-np.square(distance[row, chosen] / 0.75))
        matrix[row, chosen] = weights
        matrix[row, row] = 1.0
    matrix /= np.maximum(matrix.sum(1, keepdims=True), 1.0e-12)
    return matrix


def graph_predict(scores: np.ndarray, reference_xy: np.ndarray, *, temperature: float, strength: float, steps: int) -> np.ndarray:
    finite = np.isfinite(scores)
    if not np.any(finite):
        return centroid(reference_xy)
    safe = np.asarray(scores, dtype=np.float64).copy()
    safe[~finite] = np.max(safe[finite]) + ROOM_DIAGONAL
    probability = np.exp(-np.clip((safe - np.min(safe)) / max(temperature, 1.0e-9), 0.0, 700.0))
    probability /= probability.sum()
    transition = graph_transition(reference_xy)
    for _ in range(steps):
        probability = (1.0 - strength) * probability + strength * (transition.T @ probability)
        probability /= probability.sum()
    return np.sum(reference_xy * probability[:, None], axis=0)


def _angular_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    cosine = np.clip(a @ b.T, -1.0, 1.0)
    return np.arccos(cosine)


def beta_survival_scores(query: dict[str, np.ndarray], references: Sequence[dict[str, np.ndarray]], *, angle_weight: float, power_weight: float) -> np.ndarray:
    output = []
    for reference in references:
        if not len(query["delay"]) or not len(reference["delay"]):
            output.append(np.inf)
            continue
        delay = np.abs(query["delay"][:, None] - reference["delay"][None, :])
        angle = _angular_distance(query["aoa"], reference["aoa"])
        power = np.abs(query["power"][:, None] - reference["power"][None, :]) / 10.0
        cost = delay + angle_weight * angle + power_weight * power
        # Beta(2,5)-shaped survival emphasizes mutually supported low-cost paths.
        normalized = np.clip(cost / (cost + 1.0), 0.0, 1.0)
        survival = np.square(1.0 - normalized) * (1.0 + 2.0 * normalized)
        q_cost = np.sum(np.min(cost, axis=1) * np.max(survival, axis=1)) / max(np.sum(np.max(survival, axis=1)), 1.0e-9)
        r_cost = np.sum(np.min(cost, axis=0) * np.max(survival, axis=0)) / max(np.sum(np.max(survival, axis=0)), 1.0e-9)
        count = abs(len(query["delay"]) - len(reference["delay"])) / max(len(query["delay"]), len(reference["delay"]))
        output.append(0.5 * (q_cost + r_cost) + 0.25 * count)
    return np.asarray(output, dtype=np.float64)


def coherent_scores(query: dict[str, np.ndarray], references: Sequence[dict[str, np.ndarray]], delay_weight: float) -> np.ndarray:
    if not len(query["delay"]):
        return np.full(len(references), np.inf)
    chamfer = chamfer_scores(query["delay"], [item["delay"] for item in references])
    output = []
    for index, reference in enumerate(references):
        if not len(reference["delay"]):
            output.append(np.inf)
            continue
        distance = coherent_adp_distance(query["cir"], reference["cir"])
        output.append(distance + delay_weight * chamfer[index])
    return np.asarray(output)


def assigned_vt_centres(survey: Sequence[dict[str, np.ndarray]], reference_xy: np.ndarray, eps_m: float) -> np.ndarray:
    endpoints = []
    for observation, xy in zip(survey, reference_xy, strict=True):
        if len(observation["delay"]):
            endpoints.append(xy[None, :] + observation["delay"][:, None] * observation["aoa"][:, :2])
    if not endpoints:
        return np.empty((0, 2), dtype=np.float64)
    endpoints_array = np.vstack(endpoints)
    labels = DBSCAN(eps=eps_m, min_samples=4).fit_predict(endpoints_array)
    return np.vstack([np.median(endpoints_array[labels == label], axis=0) for label in sorted(set(labels)) if label >= 0]) if np.any(labels >= 0) else np.empty((0, 2))


def assigned_vt_predict(query: dict[str, np.ndarray], centres: np.ndarray, reference_xy: np.ndarray, consensus_eps_m: float) -> np.ndarray:
    if not len(query["delay"]) or not len(centres):
        return centroid(reference_xy)
    candidates, path_ids = [], []
    for path in range(len(query["delay"])):
        values = centres - query["delay"][path] * query["aoa"][path, :2][None, :]
        keep = ((values[:, 0] >= -0.25) & (values[:, 0] <= ROOM[0] + 0.25) &
                (values[:, 1] >= -0.25) & (values[:, 1] <= ROOM[1] + 0.25))
        candidates.extend(values[keep])
        path_ids.extend([path] * int(np.sum(keep)))
    if not candidates:
        return centroid(reference_xy)
    candidates_array = np.asarray(candidates)
    labels = DBSCAN(eps=consensus_eps_m, min_samples=2).fit_predict(candidates_array)
    best = None
    best_key = None
    path_ids_array = np.asarray(path_ids)
    for label in sorted(set(labels)):
        if label < 0:
            continue
        member = labels == label
        centre_value = np.median(candidates_array[member], axis=0)
        spread = float(np.median(np.linalg.norm(candidates_array[member] - centre_value, axis=1)))
        key = (len(np.unique(path_ids_array[member])), int(np.sum(member)), -spread)
        if best_key is None or key > best_key:
            best_key, best = key, centre_value
    return centroid(reference_xy) if best is None else np.clip(best, [0.0, 0.0], ROOM[:2])


def observation_from_arrays(bundle, prefix: str, condition: int, index: int) -> dict[str, np.ndarray]:
    mask = np.asarray(bundle[f"{prefix}_mask"][condition, index], dtype=bool)
    return {
        "delay": np.asarray(bundle[f"{prefix}_delay"][condition, index, mask], dtype=np.float64),
        "power": np.asarray(bundle[f"{prefix}_power"][condition, index, mask], dtype=np.float64),
        "aoa": np.asarray(bundle[f"{prefix}_aoa"][condition, index, mask], dtype=np.float64),
        "cir": np.asarray(bundle[f"{prefix}_cir"][condition, index], dtype=np.complex64),
    }


def corrupt_stored_observation(
    observation: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    sigma_m: float,
    dropout: float,
    clutter: int,
) -> dict[str, np.ndarray]:
    """Calibration-only augmentation; contains no simulator or geometry call."""

    delay = np.asarray(observation["delay"], dtype=np.float64).copy()
    power = np.asarray(observation["power"], dtype=np.float64).copy()
    aoa = np.asarray(observation["aoa"], dtype=np.float64).copy()
    fingerprint = MultimodalFingerprint(
        ranges_m=delay,
        powers_db=power,
        aoa_unit=aoa,
        tx_ids=np.zeros(len(delay), dtype=np.int64),
        cir=np.asarray(observation["cir"], dtype=np.complex64),
        noise_variance=0.0,
        range_bin_m=C_MPS / 2.0e9,
    )
    augmented = corrupt_stored_fingerprint(
        fingerprint,
        rng=rng,
        extra_range_std_m=sigma_m,
        extra_power_std_db=min(3.0, 0.2 + 0.4 * sigma_m),
        extra_angle_std_deg=min(9.0, 0.5 + 1.5 * sigma_m),
        dropout_probability=dropout,
    )
    # A new synthetic MPC would lack a coupled observed CIR, so clutter is
    # deliberately not synthesized in this multimodal calibration augmenter.
    _ = clutter
    order = np.argsort(augmented.ranges_m, kind="stable")
    return {
        "delay": augmented.ranges_m[order],
        "power": augmented.powers_db[order],
        "aoa": augmented.aoa_unit[order],
        "cir": augmented.cir.copy(),
    }


@dataclass
class AugmentedCalibration:
    queries: list[dict[str, np.ndarray]]
    survey_variants: list[list[dict[str, np.ndarray]]]
    group: np.ndarray
    copy: np.ndarray
    target_xy: np.ndarray


def build_calibration_augmentations(base: Sequence[dict[str, np.ndarray]], reference_xy: np.ndarray, copies: int) -> AugmentedCalibration:
    sigmas = (0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    dropouts = (0.0, 0.0, 0.08, 0.15, 0.25, 0.4)
    survey_variants: list[list[dict[str, np.ndarray]]] = []
    queries: list[dict[str, np.ndarray]] = []
    groups, copy_ids, targets = [], [], []
    for copy in range(copies):
        sigma = sigmas[copy % len(sigmas)]
        dropout = dropouts[(copy * 3) % len(dropouts)]
        clutter = 1 if copy % 11 == 10 else 0
        survey_variant = []
        for group, observation in enumerate(base):
            survey_variant.append(corrupt_stored_observation(
                observation,
                np.random.default_rng(stable_seed("augment-survey", copy, group)),
                sigma_m=0.75 * sigma, dropout=0.7 * dropout, clutter=clutter if copy % 17 == 16 else 0,
            ))
        survey_variants.append(survey_variant)
        for group, observation in enumerate(base):
            queries.append(corrupt_stored_observation(
                observation,
                np.random.default_rng(stable_seed("augment-query", copy, group)),
                sigma_m=sigma, dropout=dropout, clutter=clutter,
            ))
            groups.append(group)
            copy_ids.append(copy)
            targets.append(reference_xy[group])
    return AugmentedCalibration(
        queries=queries, survey_variants=survey_variants,
        group=np.asarray(groups, dtype=np.int64), copy=np.asarray(copy_ids, dtype=np.int64),
        target_xy=np.asarray(targets, dtype=np.float32),
    )


def grouped_split(reference_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Deterministic spatial blocks: one of five interleaved 2-D cells is held out.
    ix = np.rint((reference_xy[:, 0] - 0.25) / 0.5).astype(int)
    iy = np.rint((reference_xy[:, 1] - 0.25) / 0.5).astype(int)
    validation = np.flatnonzero((2 * ix + 3 * iy) % 5 == 0)
    training = np.asarray([index for index in range(len(reference_xy)) if index not in set(validation)], dtype=np.int64)
    if len(validation) != 16 or len(training) != 64:
        raise AssertionError("expected a 64/16 whole-RP split")
    return training, validation


def delay_feature_vector(delay: np.ndarray) -> np.ndarray:
    delay = np.sort(np.asarray(delay, dtype=np.float64))
    q = np.linspace(0.0, 1.0, 17)
    quantiles = np.quantile(delay, q) if len(delay) else np.zeros(len(q))
    gaps = np.diff(delay)
    gap_q = np.quantile(gaps, np.linspace(0.0, 1.0, 7)) if len(gaps) else np.zeros(7)
    statistics = np.asarray([
        len(delay) / MAX_PATHS,
        float(np.mean(delay)) if len(delay) else 0.0,
        float(np.std(delay)) if len(delay) else 0.0,
        float(np.min(delay)) if len(delay) else 0.0,
        float(np.max(delay)) if len(delay) else 0.0,
    ])
    return np.concatenate((quantiles / 15.0, gap_q / 5.0, statistics / np.asarray([1.0, 15.0, 5.0, 15.0, 15.0])))


def multimodal_feature_vector(observation: dict[str, np.ndarray]) -> np.ndarray:
    delay = delay_feature_vector(observation["delay"])
    if not len(observation["delay"]):
        return np.concatenate((delay, np.zeros(20)))
    power_q = np.quantile(observation["power"], np.linspace(0.0, 1.0, 7)) / 80.0
    weighted = np.exp((observation["power"] - np.max(observation["power"])) / 10.0)
    weighted /= max(float(np.sum(weighted)), 1.0e-12)
    direction = np.sum(observation["aoa"] * weighted[:, None], axis=0)
    direction_stats = np.concatenate((direction, np.std(observation["aoa"], axis=0)))
    cir = np.abs(np.asarray(observation["cir"], dtype=np.complex128)).reshape(-1)
    if len(cir):
        summary = np.asarray([float(np.mean(block)) for block in np.array_split(cir, 7)])
        summary /= max(float(np.linalg.norm(summary)), 1.0e-12)
    else:
        summary = np.zeros(7)
    return np.concatenate((delay, power_q, direction_stats, summary))


def token_tensor(observations: Sequence[dict[str, np.ndarray]], *, multimodal: bool) -> tuple[np.ndarray, np.ndarray]:
    features = 5 if multimodal else 1
    values = np.zeros((len(observations), MAX_PATHS, features), dtype=np.float32)
    mask = np.zeros((len(observations), MAX_PATHS), dtype=bool)
    for row, observation in enumerate(observations):
        count = min(len(observation["delay"]), MAX_PATHS)
        if not count:
            continue
        values[row, :count, 0] = observation["delay"][:count] / 15.0
        if multimodal:
            values[row, :count, 1] = np.clip((observation["power"][:count] + 80.0) / 80.0, -1.0, 1.0)
            values[row, :count, 2:5] = observation["aoa"][:count]
        mask[row, :count] = True
    return values, mask


class MaskedSetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 48, self_attention: bool = False):
        super().__init__()
        self.hidden = hidden
        self.token = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.self_attention = self_attention
        if self_attention:
            self.q = nn.Linear(hidden, hidden, bias=False)
            self.k = nn.Linear(hidden, hidden, bias=False)
            self.v = nn.Linear(hidden, hidden, bias=False)
            self.norm = nn.LayerNorm(hidden)
        self.output_dim = 2 * hidden + 1

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token(values)
        numeric = mask[..., None].to(hidden.dtype)
        if self.self_attention:
            logits = self.q(hidden) @ self.k(hidden).transpose(-1, -2) / math.sqrt(self.hidden)
            logits = logits.masked_fill(~mask[:, None, :], -1.0e4)
            attention = torch.softmax(logits, dim=-1) * numeric.transpose(1, 2)
            attention = attention / attention.sum(-1, keepdim=True).clamp_min(1.0e-8)
            hidden = self.norm(hidden + attention @ self.v(hidden))
        mean = (hidden * numeric).sum(1) / numeric.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        count = torch.log1p(mask.sum(1, keepdim=True).to(hidden.dtype))
        return torch.cat((mean, maximum, count), dim=-1)


class DirectPointNet(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = MaskedSetEncoder(input_dim)
        self.head = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.head(self.encoder(values, mask)))


class GridProbabilityMLP(nn.Module):
    def __init__(self, candidates: int):
        super().__init__()
        self.encoder = MaskedSetEncoder(1)
        self.head = nn.Sequential(nn.Linear(self.encoder.output_dim, 128), nn.GELU(), nn.Linear(128, candidates))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, candidate_xy_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(self.encoder(values, mask))
        weights = torch.softmax(logits, dim=-1)
        return weights @ candidate_xy_norm, logits


class ResidualAnalyticHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = MaskedSetEncoder(1)
        self.head = nn.Sequential(nn.Linear(self.encoder.output_dim + 2, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2))

    def forward(self, values: torch.Tensor, mask: torch.Tensor, analytic_xy_norm: torch.Tensor) -> torch.Tensor:
        delta = 0.22 * torch.tanh(self.head(torch.cat((self.encoder(values, mask), analytic_xy_norm), dim=-1)))
        return torch.clamp(analytic_xy_norm + delta, 0.0, 1.0)


class PointNetCandidateReranker(nn.Module):
    def __init__(self, input_dim: int = 1):
        super().__init__()
        self.encoder = MaskedSetEncoder(input_dim)
        width = self.encoder.output_dim
        self.score = nn.Sequential(nn.Linear(width * 3 + 5, 160), nn.GELU(), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(width + 2, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, query, query_mask, references, reference_mask, candidate_xy, diagnostics, analytic_xy):
        batch, candidates, paths, features = references.shape
        query_encoded = self.encoder(query, query_mask)
        reference_encoded = self.encoder(references.reshape(batch * candidates, paths, features), reference_mask.reshape(batch * candidates, paths)).reshape(batch, candidates, -1)
        expanded = query_encoded[:, None].expand(-1, candidates, -1)
        logits = self.score(torch.cat((expanded, reference_encoded, torch.abs(expanded - reference_encoded), diagnostics, candidate_xy), dim=-1)).squeeze(-1)
        learned = torch.sum(torch.softmax(logits, dim=-1)[..., None] * candidate_xy, dim=1)
        gate = torch.sigmoid(self.gate(torch.cat((query_encoded, analytic_xy), dim=-1)))
        return (1.0 - gate) * analytic_xy + gate * learned, logits, gate


class PathCrossAttentionReranker(nn.Module):
    """Actual query-path to reference-path attention, without positional encodings."""

    def __init__(self, input_dim: int = 5, hidden: int = 48):
        super().__init__()
        self.hidden = hidden
        self.embed = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Sequential(nn.Linear(4 * hidden + 5, 160), nn.GELU(), nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1))
        self.gate = nn.Sequential(nn.Linear(2 * hidden + 2, 64), nn.GELU(), nn.Linear(64, 1))

    @staticmethod
    def pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        numeric = mask[..., None].to(hidden.dtype)
        mean = (hidden * numeric).sum(-2) / numeric.sum(-2).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask[..., None], -torch.inf).amax(-2)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        return torch.cat((mean, maximum), dim=-1)

    def forward(self, query, query_mask, references, reference_mask, candidate_xy, diagnostics, analytic_xy):
        batch, candidates, paths, features = references.shape
        qh = self.embed(query)[:, None].expand(-1, candidates, -1, -1)
        rh = self.embed(references)
        logits_qr = torch.einsum("bkph,bkrh->bkpr", self.q(qh), self.k(rh)) / math.sqrt(self.hidden)
        logits_qr = logits_qr.masked_fill(~reference_mask[:, :, None, :], -1.0e4)
        weights_qr = torch.softmax(logits_qr, dim=-1) * reference_mask[:, :, None, :].to(logits_qr.dtype)
        weights_qr = weights_qr / weights_qr.sum(-1, keepdim=True).clamp_min(1.0e-8)
        attended_q = torch.einsum("bkpr,bkrh->bkph", weights_qr, self.v(rh))
        qmask = query_mask[:, None, :].expand(-1, candidates, -1)
        q_pool = self.pool(qh + attended_q, qmask)
        logits_rq = logits_qr.transpose(-1, -2).masked_fill(~qmask[:, :, None, :], -1.0e4)
        weights_rq = torch.softmax(logits_rq, dim=-1) * qmask[:, :, None, :].to(logits_rq.dtype)
        weights_rq = weights_rq / weights_rq.sum(-1, keepdim=True).clamp_min(1.0e-8)
        attended_r = torch.einsum("bkrp,bkph->bkrh", weights_rq, self.v(qh))
        r_pool = self.pool(rh + attended_r, reference_mask)
        logits = self.score(torch.cat((q_pool, r_pool, diagnostics, candidate_xy), dim=-1)).squeeze(-1)
        learned = torch.sum(torch.softmax(logits, dim=-1)[..., None] * candidate_xy, dim=1)
        global_query = self.pool(self.embed(query), query_mask)
        gate = torch.sigmoid(self.gate(torch.cat((global_query, analytic_xy), dim=-1)))
        return (1.0 - gate) * analytic_xy + gate * learned, logits, gate


class CandidateSelfAttentionReranker(nn.Module):
    """Path-self-attention pooling plus candidate-set self-attention."""

    def __init__(self, input_dim: int = 5, hidden: int = 48):
        super().__init__()
        self.path_encoder = MaskedSetEncoder(input_dim, hidden=hidden, self_attention=True)
        width = self.path_encoder.output_dim
        self.candidate = nn.Sequential(nn.Linear(width * 3 + 5, 128), nn.GELU(), nn.Linear(128, hidden))
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.score = nn.Sequential(nn.Linear(hidden, 48), nn.GELU(), nn.Linear(48, 1))

    def forward(self, query, query_mask, references, reference_mask, candidate_xy, diagnostics, analytic_xy):
        batch, candidates, paths, features = references.shape
        qe = self.path_encoder(query, query_mask)
        re = self.path_encoder(references.reshape(batch * candidates, paths, features), reference_mask.reshape(batch * candidates, paths)).reshape(batch, candidates, -1)
        expanded = qe[:, None].expand(-1, candidates, -1)
        token = self.candidate(torch.cat((expanded, re, torch.abs(expanded - re), diagnostics, candidate_xy), dim=-1))
        attention = torch.softmax(self.q(token) @ self.k(token).transpose(-1, -2) / math.sqrt(token.shape[-1]), dim=-1)
        token = token + attention @ self.v(token)
        logits = self.score(token).squeeze(-1)
        prediction = torch.sum(torch.softmax(logits, dim=-1)[..., None] * candidate_xy, dim=1)
        return prediction, logits, torch.ones((batch, 1), device=query.device)


@dataclass
class CalibrationScoreCache:
    corrected: np.ndarray
    coverage: np.ndarray
    chamfer: np.ndarray
    evo: np.ndarray


def compute_calibration_score_cache(augment: AugmentedCalibration, output: Path) -> CalibrationScoreCache:
    rows, references = len(augment.queries), len(augment.survey_variants[0])
    corrected = np.full((rows, references), np.inf, dtype=np.float32)
    coverage = np.zeros((rows, references), dtype=np.float32)
    chamfer = np.full((rows, references), np.inf, dtype=np.float32)
    evo = np.full((rows, references), np.inf, dtype=np.float32)
    started = time.perf_counter()
    for row, query in enumerate(augment.queries):
        survey = augment.survey_variants[int(augment.copy[row])]
        ref_delay = [item["delay"] for item in survey]
        score, cover = corrected_mpurge_scores(query["delay"], ref_delay)
        corrected[row] = score
        coverage[row] = cover
        chamfer[row] = chamfer_scores(query["delay"], ref_delay)
        evo[row] = evo_scores(query["delay"], ref_delay)
        if (row + 1) % 160 == 0:
            print(f"calibration score cache {row + 1}/{rows}", flush=True)
    np.savez_compressed(
        output,
        corrected=corrected,
        coverage=coverage,
        chamfer=chamfer,
        evo=evo,
        group=augment.group,
        copy=augment.copy,
        runtime_s=np.asarray([time.perf_counter() - started]),
    )
    return CalibrationScoreCache(corrected, coverage, chamfer, evo)


def _allowed_scores(values: np.ndarray, allowed: np.ndarray, source: int | None) -> tuple[np.ndarray, np.ndarray]:
    allowed = np.asarray(allowed, dtype=np.int64)
    score = np.asarray(values[allowed], dtype=np.float64).copy()
    if source is not None:
        score[allowed == int(source)] = np.inf
    return score, allowed


def _score_prediction(values: np.ndarray, allowed: np.ndarray, reference_xy: np.ndarray, *, source: int | None = None,
                      head: str = "inverse", temperature: float = 0.1) -> np.ndarray:
    score, identifiers = _allowed_scores(values, allowed, source)
    if head == "inverse":
        return inverse_topk(score, reference_xy[identifiers])
    if head == "softmax":
        return softmax_topk(score, reference_xy[identifiers], temperature)
    raise ValueError(head)


def select_analytic_configs(
    augment: AugmentedCalibration,
    cache: CalibrationScoreCache,
    reference_xy: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
) -> dict:
    validation_rows = np.flatnonzero(np.isin(augment.group, validation_groups))
    truth = augment.target_xy[validation_rows]

    def mean_for(builder: Callable[[int], np.ndarray]) -> float:
        predictions = np.asarray([builder(int(row)) for row in validation_rows])
        return float(np.mean(np.linalg.norm(predictions - truth, axis=1)))

    temperature_grid = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0)
    path_grid = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    b_rows = []
    for temperature in temperature_grid:
        value = mean_for(lambda row, t=temperature: _score_prediction(
            cache.corrected[row], train_groups, reference_xy, head="softmax", temperature=t,
        ))
        b_rows.append({"temperature": temperature, "mean_error_m": value})
    best_b = min(b_rows, key=lambda row: (row["mean_error_m"], row["temperature"]))

    c_rows = []
    for strength in path_grid:
        def c_prediction(row: int, strength_value: float = strength) -> np.ndarray:
            query = augment.queries[row]["delay"]
            survey = augment.survey_variants[int(augment.copy[row])]
            refs = [survey[int(index)]["delay"] for index in train_groups]
            adjusted = path_count_adjusted(cache.corrected[row, train_groups], query, refs, strength_value)
            return inverse_topk(adjusted, reference_xy[train_groups])
        c_rows.append({"path_count_strength": strength, "mean_error_m": mean_for(c_prediction)})
    best_c = min(c_rows, key=lambda row: (row["mean_error_m"], row["path_count_strength"]))

    d_rows = []
    for strength, temperature in itertools.product(path_grid, temperature_grid):
        def d_prediction(row: int, s=strength, t=temperature) -> np.ndarray:
            query = augment.queries[row]["delay"]
            survey = augment.survey_variants[int(augment.copy[row])]
            refs = [survey[int(index)]["delay"] for index in train_groups]
            score = np.asarray(cache.chamfer[row, train_groups], dtype=np.float64)
            for ref_index, reference in enumerate(refs):
                if not np.isfinite(score[ref_index]) and len(query) and not len(reference):
                    score[ref_index] = ROOM_DIAGONAL + s * abs(len(query) - len(reference))
                if np.isfinite(score[ref_index]):
                    score[ref_index] += s * abs(len(query) - len(reference)) / max(len(query), len(reference), 1)
            return softmax_topk(score, reference_xy[train_groups], t)
        d_rows.append({"path_count_strength": strength, "temperature": temperature, "mean_error_m": mean_for(d_prediction)})
    best_d = min(d_rows, key=lambda row: (row["mean_error_m"], row["path_count_strength"], row["temperature"]))

    graph_rows = []
    for temperature, strength, steps in itertools.product((0.02, 0.05, 0.1, 0.2, 0.5, 1.0), (0.1, 0.25, 0.5, 0.75), (1, 2, 3)):
        value = mean_for(lambda row, t=temperature, s=strength, n=steps: graph_predict(
            cache.corrected[row, train_groups], reference_xy[train_groups], temperature=t, strength=s, steps=n,
        ))
        graph_rows.append({"temperature": temperature, "strength": strength, "steps": steps, "mean_error_m": value})
    best_graph = min(graph_rows, key=lambda row: (row["mean_error_m"], row["steps"], row["strength"], row["temperature"]))

    return {
        "ablation_B_temperature_softmax_only": {"selected": best_b, "grid": b_rows},
        "ablation_C_path_count_empty_only": {"selected": best_c, "grid": c_rows},
        "ablation_D_all_three": {"selected": best_d, "grid": d_rows},
        "graph_diffusion": {"selected": best_graph, "grid": graph_rows},
    }


def select_multimodal_configs(
    augment: AugmentedCalibration,
    reference_xy: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    *,
    quick: bool,
) -> dict:
    rows = np.flatnonzero(np.isin(augment.group, validation_groups))
    if quick:
        rows = rows[:min(len(rows), 24)]
    truth = augment.target_xy[rows]
    configs = [(0.5, 0.1), (1.0, 0.1), (2.0, 0.1), (1.0, 0.25), (2.0, 0.25)]
    beta_predictions: dict[tuple[float, float], list[np.ndarray]] = {config: [] for config in configs}
    coherent_weights = (0.0, 0.05, 0.1, 0.25, 0.5)
    coherent_predictions: dict[float, list[np.ndarray]] = {weight: [] for weight in coherent_weights}
    survival_predictions: dict[tuple[float, float, float], list[np.ndarray]] = {}
    score_cache: list[tuple[np.ndarray, dict[tuple[float, float], np.ndarray], dict[float, np.ndarray]]] = []
    for row in rows:
        query = augment.queries[int(row)]
        survey_all = augment.survey_variants[int(augment.copy[int(row)])]
        survey = [survey_all[int(index)] for index in train_groups]
        beta_by = {config: beta_survival_scores(query, survey, angle_weight=config[0], power_weight=config[1]) for config in configs}
        coherent_by = {weight: coherent_scores(query, survey, weight) for weight in coherent_weights}
        for config, score in beta_by.items():
            beta_predictions[config].append(inverse_topk(score, reference_xy[train_groups]))
        for weight, score in coherent_by.items():
            coherent_predictions[weight].append(inverse_topk(score, reference_xy[train_groups]))
        score_cache.append((np.asarray([symmetric_chamfer(query["delay"], item["delay"]) for item in survey]), beta_by, coherent_by))
    beta_rows = [{"angle_weight": config[0], "power_weight": config[1],
                  "mean_error_m": float(np.mean(np.linalg.norm(np.asarray(beta_predictions[config]) - truth, axis=1)))} for config in configs]
    coherent_rows = [{"delay_weight": weight,
                      "mean_error_m": float(np.mean(np.linalg.norm(np.asarray(coherent_predictions[weight]) - truth, axis=1)))} for weight in coherent_weights]
    best_beta = min(beta_rows, key=lambda row: (row["mean_error_m"], row["angle_weight"], row["power_weight"]))
    best_coherent = min(coherent_rows, key=lambda row: (row["mean_error_m"], row["delay_weight"]))
    beta_key = (best_beta["angle_weight"], best_beta["power_weight"])
    coherent_key = best_coherent["delay_weight"]
    survival_rows = []
    for beta_weight, cir_weight, delay_weight in itertools.product((0.5, 1.0, 2.0), (0.25, 0.5, 1.0, 2.0), (0.0, 0.1, 0.25)):
        predictions = []
        for chamfer_value, beta_by, coherent_by in score_cache:
            components = beta_weight * beta_by[beta_key] + cir_weight * coherent_by[coherent_key] + delay_weight * chamfer_value
            predictions.append(inverse_topk(components, reference_xy[train_groups]))
        error = float(np.mean(np.linalg.norm(np.asarray(predictions) - truth, axis=1)))
        survival_rows.append({"beta_weight": beta_weight, "cir_weight": cir_weight, "delay_weight": delay_weight, "mean_error_m": error})
    best_survival = min(survival_rows, key=lambda row: (row["mean_error_m"], row["beta_weight"], row["cir_weight"], row["delay_weight"]))

    vt_rows = []
    vt_grid = ((0.2, 0.2), (0.3, 0.25), (0.4, 0.3), (0.5, 0.4), (0.75, 0.5))
    for vt_eps, consensus_eps in vt_grid:
        predictions = []
        for row in rows:
            survey_all = augment.survey_variants[int(augment.copy[int(row)])]
            survey = [survey_all[int(index)] for index in train_groups]
            centres = assigned_vt_centres(survey, reference_xy[train_groups], vt_eps)
            predictions.append(assigned_vt_predict(augment.queries[int(row)], centres, reference_xy[train_groups], consensus_eps))
        vt_rows.append({"vt_cluster_eps_m": vt_eps, "query_consensus_eps_m": consensus_eps,
                        "mean_error_m": float(np.mean(np.linalg.norm(np.asarray(predictions) - truth, axis=1)))})
    best_vt = min(vt_rows, key=lambda row: (row["mean_error_m"], row["vt_cluster_eps_m"], row["query_consensus_eps_m"]))
    return {
        "beta_survival": {"selected": best_beta, "grid": beta_rows},
        "coherent_adp_toa": {"selected": best_coherent, "grid": coherent_rows},
        "survival_cir_toa": {"selected": best_survival, "grid": survival_rows},
        "assigned_vt_consensus": {"selected": best_vt, "grid": vt_rows},
    }


def fit_extra_trees_models(
    augment: AugmentedCalibration,
    reference_xy: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    output_dir: Path,
    *,
    quick: bool,
) -> tuple[dict[str, ExtraTreesRegressor], dict]:
    train_rows = np.flatnonzero(np.isin(augment.group, train_groups))
    validation_rows = np.flatnonzero(np.isin(augment.group, validation_groups))
    feature_sets = {
        "delay_feature_extratrees": np.asarray([delay_feature_vector(item["delay"]) for item in augment.queries]),
        "multimodal_feature_extratrees": np.asarray([multimodal_feature_vector(item) for item in augment.queries]),
    }
    models, selection = {}, {}
    leaves = (1, 2) if quick else (1, 2, 4, 8, 12)
    for name, features in feature_sets.items():
        rows = []
        for leaf in leaves:
            model = ExtraTreesRegressor(
                n_estimators=32 if quick else 320,
                min_samples_leaf=leaf,
                max_features=1.0,
                n_jobs=-1,
                random_state=stable_seed("extra-select", name, leaf),
            )
            model.fit(features[train_rows], augment.target_xy[train_rows])
            prediction = model.predict(features[validation_rows])
            rows.append({"min_samples_leaf": leaf, "mean_error_m": float(np.mean(np.linalg.norm(prediction - augment.target_xy[validation_rows], axis=1)))})
        best = min(rows, key=lambda row: (row["mean_error_m"], row["min_samples_leaf"]))
        final = ExtraTreesRegressor(
            n_estimators=48 if quick else 480,
            min_samples_leaf=int(best["min_samples_leaf"]),
            max_features=1.0,
            n_jobs=-1,
            random_state=stable_seed("extra-final", name),
        )
        final.fit(features, augment.target_xy)
        checkpoint = output_dir / "checkpoints" / f"{name}.joblib"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, checkpoint)
        models[name] = final
        selection[name] = {"selected": best, "grid": rows, "checkpoint": str(checkpoint)}
    return models, selection


def expert_predictions_for_rows(
    augment: AugmentedCalibration,
    cache: CalibrationScoreCache,
    rows: np.ndarray,
    allowed: np.ndarray,
    reference_xy: np.ndarray,
    graph_config: dict,
    *,
    exclude_source: bool,
) -> tuple[np.ndarray, np.ndarray]:
    predictions, meta = [], []
    for row in rows:
        source = int(augment.group[row]) if exclude_source and int(augment.group[row]) in set(allowed.tolist()) else None
        analytic = _score_prediction(cache.corrected[row], allowed, reference_xy, source=source)
        chamfer = _score_prediction(cache.chamfer[row], allowed, reference_xy, source=source)
        evo = _score_prediction(cache.evo[row], allowed, reference_xy, source=source, head="softmax", temperature=EVO_TEMPERATURE)
        graph_score, graph_ids = _allowed_scores(cache.corrected[row], allowed, source)
        graph = graph_predict(graph_score, reference_xy[graph_ids], temperature=graph_config["temperature"], strength=graph_config["strength"], steps=int(graph_config["steps"]))
        experts = np.vstack((analytic, chamfer, evo, graph))
        predictions.append(experts)
        spread = np.mean(np.linalg.norm(experts[:, None, :] - experts[None, :, :], axis=2))
        meta.append(np.concatenate((delay_feature_vector(augment.queries[row]["delay"]), experts.reshape(-1) / np.tile(ROOM[:2], 4), np.asarray([spread / ROOM_DIAGONAL]))))
    return np.asarray(predictions), np.asarray(meta)


def fit_rrle_router(
    augment: AugmentedCalibration,
    cache: CalibrationScoreCache,
    reference_xy: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    graph_config: dict,
    output_dir: Path,
    *,
    quick: bool,
) -> tuple[ExtraTreesRegressor, dict]:
    train_rows = np.flatnonzero(np.isin(augment.group, train_groups))
    validation_rows = np.flatnonzero(np.isin(augment.group, validation_groups))
    train_predictions, train_meta = expert_predictions_for_rows(
        augment, cache, train_rows, train_groups, reference_xy, graph_config, exclude_source=True,
    )
    validation_predictions, validation_meta = expert_predictions_for_rows(
        augment, cache, validation_rows, train_groups, reference_xy, graph_config, exclude_source=False,
    )
    train_errors = np.linalg.norm(train_predictions - augment.target_xy[train_rows, None, :], axis=2)
    leaves = (2,) if quick else (1, 2, 4, 8, 12)
    temperatures = (0.05, 0.1, 0.2, 0.5, 1.0)
    grid = []
    for leaf in leaves:
        model = ExtraTreesRegressor(
            n_estimators=32 if quick else 240, min_samples_leaf=leaf, max_features=0.8,
            n_jobs=-1, random_state=stable_seed("rrle-select", leaf),
        )
        model.fit(train_meta, train_errors)
        predicted_error = np.maximum(model.predict(validation_meta), 0.0)
        for temperature in temperatures:
            weights = np.exp(-predicted_error / temperature)
            weights /= np.maximum(weights.sum(1, keepdims=True), 1.0e-12)
            prediction = np.sum(validation_predictions * weights[..., None], axis=1)
            grid.append({"min_samples_leaf": leaf, "temperature": temperature,
                         "mean_error_m": float(np.mean(np.linalg.norm(prediction - augment.target_xy[validation_rows], axis=1)))})
    best = min(grid, key=lambda row: (row["mean_error_m"], row["min_samples_leaf"], row["temperature"]))
    all_rows = np.arange(len(augment.queries))
    all_predictions, all_meta = expert_predictions_for_rows(
        augment, cache, all_rows, np.arange(len(reference_xy)), reference_xy, graph_config, exclude_source=True,
    )
    all_errors = np.linalg.norm(all_predictions - augment.target_xy[:, None, :], axis=2)
    final = ExtraTreesRegressor(
        n_estimators=48 if quick else 400, min_samples_leaf=int(best["min_samples_leaf"]), max_features=0.8,
        n_jobs=-1, random_state=stable_seed("rrle-final"),
    )
    final.fit(all_meta, all_errors)
    final._rrle_temperature = float(best["temperature"])  # type: ignore[attr-defined]
    checkpoint = output_dir / "checkpoints" / "rrle_moe_router.joblib"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, checkpoint)
    return final, {"selected": best, "grid": grid, "experts": ["corrected_mpurge", "chamfer", "frozen_evomdp", "graph_diffusion"], "checkpoint": str(checkpoint)}


@dataclass
class CandidateTensorData:
    query_delay: np.ndarray
    query_delay_mask: np.ndarray
    query_multi: np.ndarray
    query_multi_mask: np.ndarray
    reference_delay: np.ndarray
    reference_delay_mask: np.ndarray
    reference_multi: np.ndarray
    reference_multi_mask: np.ndarray
    candidate_xy_norm: np.ndarray
    diagnostics: np.ndarray
    analytic_xy_norm: np.ndarray
    target_xy_norm: np.ndarray
    candidate_ids: np.ndarray


def build_augmented_candidate_data(
    augment: AugmentedCalibration,
    cache: CalibrationScoreCache,
    reference_xy: np.ndarray,
    allowed: np.ndarray,
    *,
    candidate_budget: int = 12,
) -> CandidateTensorData:
    n = len(augment.queries)
    query_delay, query_delay_mask = token_tensor(augment.queries, multimodal=False)
    query_multi, query_multi_mask = token_tensor(augment.queries, multimodal=True)
    reference_delay = np.zeros((n, candidate_budget, MAX_PATHS, 1), dtype=np.float32)
    reference_multi = np.zeros((n, candidate_budget, MAX_PATHS, 5), dtype=np.float32)
    reference_delay_mask = np.zeros((n, candidate_budget, MAX_PATHS), dtype=bool)
    reference_multi_mask = np.zeros((n, candidate_budget, MAX_PATHS), dtype=bool)
    candidate_xy = np.zeros((n, candidate_budget, 2), dtype=np.float32)
    diagnostics = np.zeros((n, candidate_budget, 3), dtype=np.float32)
    analytic_xy = np.zeros((n, 2), dtype=np.float32)
    candidate_ids = np.zeros((n, candidate_budget), dtype=np.int64)
    allowed_set = set(map(int, allowed))
    for row, query in enumerate(augment.queries):
        source = int(augment.group[row]) if int(augment.group[row]) in allowed_set else None
        score, identifiers = _allowed_scores(cache.corrected[row], allowed, source)
        analytic_xy[row] = inverse_topk(score, reference_xy[identifiers]) / ROOM[:2]
        finite = np.flatnonzero(np.isfinite(score))
        ordered = finite[np.argsort(score[finite], kind="stable")] if len(finite) else np.arange(len(score))
        if len(ordered) < candidate_budget:
            remainder = [index for index in range(len(score)) if index not in set(map(int, ordered))]
            ordered = np.concatenate((ordered, np.asarray(remainder, dtype=np.int64)))
        chosen_rows = ordered[:candidate_budget]
        chosen = identifiers[chosen_rows]
        candidate_ids[row] = chosen
        candidate_xy[row] = reference_xy[chosen] / ROOM[:2]
        survey = augment.survey_variants[int(augment.copy[row])]
        selected_observations = [survey[int(index)] for index in chosen]
        rd, rdm = token_tensor(selected_observations, multimodal=False)
        rm, rmm = token_tensor(selected_observations, multimodal=True)
        reference_delay[row], reference_delay_mask[row] = rd, rdm
        reference_multi[row], reference_multi_mask[row] = rm, rmm
        selected_scores = score[chosen_rows]
        finite_scores = selected_scores[np.isfinite(selected_scores)]
        sentinel = float(np.max(finite_scores) * 10.0) if len(finite_scores) else 1.0e6
        selected_scores = np.where(np.isfinite(selected_scores), selected_scores, sentinel)
        selected_coverage = cache.coverage[row, chosen]
        selected_count = np.asarray([len(survey[int(index)]["delay"]) for index in chosen])
        diagnostics[row, :, 0] = np.log1p(np.maximum(selected_scores, 0.0)) / 5.0
        diagnostics[row, :, 1] = selected_coverage
        diagnostics[row, :, 2] = np.abs(len(query["delay"]) - selected_count) / MAX_PATHS
    return CandidateTensorData(
        query_delay, query_delay_mask, query_multi, query_multi_mask,
        reference_delay, reference_delay_mask, reference_multi, reference_multi_mask,
        candidate_xy, diagnostics, analytic_xy,
        (augment.target_xy / ROOM[:2]).astype(np.float32), candidate_ids,
    )


def build_evaluation_candidate_data(
    queries: Sequence[dict[str, np.ndarray]],
    survey: Sequence[dict[str, np.ndarray]],
    scores: np.ndarray,
    coverage: np.ndarray,
    reference_xy: np.ndarray,
    *,
    candidate_budget: int = 12,
) -> CandidateTensorData:
    n = len(queries)
    qd, qdm = token_tensor(queries, multimodal=False)
    qm, qmm = token_tensor(queries, multimodal=True)
    rd = np.zeros((n, candidate_budget, MAX_PATHS, 1), dtype=np.float32)
    rdm = np.zeros((n, candidate_budget, MAX_PATHS), dtype=bool)
    rm = np.zeros((n, candidate_budget, MAX_PATHS, 5), dtype=np.float32)
    rmm = np.zeros((n, candidate_budget, MAX_PATHS), dtype=bool)
    cxy = np.zeros((n, candidate_budget, 2), dtype=np.float32)
    diag = np.zeros((n, candidate_budget, 3), dtype=np.float32)
    analytic = np.zeros((n, 2), dtype=np.float32)
    ids = np.zeros((n, candidate_budget), dtype=np.int64)
    for row, query in enumerate(queries):
        analytic[row] = inverse_topk(scores[row], reference_xy) / ROOM[:2]
        finite = np.flatnonzero(np.isfinite(scores[row]))
        order = finite[np.argsort(scores[row, finite], kind="stable")] if len(finite) else np.arange(len(reference_xy))
        if len(order) < candidate_budget:
            remainder = [index for index in range(len(reference_xy)) if index not in set(map(int, order))]
            order = np.concatenate((order, np.asarray(remainder, dtype=np.int64)))
        chosen = order[:candidate_budget]
        ids[row] = chosen
        cxy[row] = reference_xy[chosen] / ROOM[:2]
        rd[row], rdm[row] = token_tensor([survey[int(index)] for index in chosen], multimodal=False)
        rm[row], rmm[row] = token_tensor([survey[int(index)] for index in chosen], multimodal=True)
        selected_scores = scores[row, chosen]
        finite_scores = selected_scores[np.isfinite(selected_scores)]
        sentinel = float(np.max(finite_scores) * 10.0) if len(finite_scores) else 1.0e6
        selected_scores = np.where(np.isfinite(selected_scores), selected_scores, sentinel)
        diag[row, :, 0] = np.log1p(np.maximum(selected_scores, 0.0)) / 5.0
        diag[row, :, 1] = coverage[row, chosen]
        diag[row, :, 2] = np.abs(len(query["delay"]) - np.asarray([len(survey[int(index)]["delay"]) for index in chosen])) / MAX_PATHS
    return CandidateTensorData(qd, qdm, qm, qmm, rd, rdm, rm, rmm, cxy, diag, analytic,
                               np.zeros((n, 2), dtype=np.float32), ids)


def analytic_xy_for_augment(augment: AugmentedCalibration, cache: CalibrationScoreCache,
                            reference_xy: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    allowed_set = set(map(int, allowed))
    output = []
    for row in range(len(augment.queries)):
        source = int(augment.group[row]) if int(augment.group[row]) in allowed_set else None
        output.append(_score_prediction(cache.corrected[row], allowed, reference_xy, source=source) / ROOM[:2])
    return np.asarray(output, dtype=np.float32)


def _torch_batches(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> Iterable[np.ndarray]:
    order = indices[rng.permutation(len(indices))]
    for start in range(0, len(order), batch_size):
        yield order[start:start + batch_size]


def _tensor(array: np.ndarray, index: np.ndarray, device: torch.device) -> torch.Tensor:
    value = torch.as_tensor(array[index], device=device)
    return value.float() if value.dtype == torch.float64 else value


def _candidate_forward(model: nn.Module, data: CandidateTensorData, index: np.ndarray, device: torch.device, multimodal: bool):
    if multimodal:
        query, query_mask = data.query_multi, data.query_multi_mask
        references, reference_mask = data.reference_multi, data.reference_multi_mask
    else:
        query, query_mask = data.query_delay, data.query_delay_mask
        references, reference_mask = data.reference_delay, data.reference_delay_mask
    return model(
        _tensor(query, index, device), _tensor(query_mask, index, device),
        _tensor(references, index, device), _tensor(reference_mask, index, device),
        _tensor(data.candidate_xy_norm, index, device), _tensor(data.diagnostics, index, device),
        _tensor(data.analytic_xy_norm, index, device),
    )


@torch.no_grad()
def predict_torch_model(name: str, model: nn.Module, *,
                        delay_values: np.ndarray | None, delay_mask: np.ndarray | None,
                        multi_values: np.ndarray | None, multi_mask: np.ndarray | None,
                        analytic_xy_norm: np.ndarray | None,
                        candidate_data: CandidateTensorData | None,
                        class_xy_norm: np.ndarray | None,
                        device: torch.device, batch_size: int = 256) -> np.ndarray:
    model.eval()
    count = len(candidate_data.query_delay) if candidate_data is not None else len(delay_values if delay_values is not None else multi_values)
    output = []
    for start in range(0, count, batch_size):
        index = np.arange(start, min(start + batch_size, count))
        if name == "pointnet_delay_direct":
            prediction = model(_tensor(delay_values, index, device), _tensor(delay_mask, index, device))
        elif name == "pointnet_multimodal_direct":
            prediction = model(_tensor(multi_values, index, device), _tensor(multi_mask, index, device))
        elif name == "caez_grid_probability_delay":
            classes = torch.as_tensor(class_xy_norm, dtype=torch.float32, device=device)
            prediction, _ = model(_tensor(delay_values, index, device), _tensor(delay_mask, index, device), classes)
        elif name == "residual_to_analytic_delay":
            prediction = model(_tensor(delay_values, index, device), _tensor(delay_mask, index, device), _tensor(analytic_xy_norm, index, device))
        elif name == "pointnet_analytic_reranker_delay":
            prediction, _, _ = _candidate_forward(model, candidate_data, index, device, False)
        elif name in ("cross_attention_multimodal", "candidate_self_attention_multimodal"):
            prediction, _, _ = _candidate_forward(model, candidate_data, index, device, True)
        else:
            raise ValueError(name)
        output.append(prediction.detach().cpu().numpy())
    return np.vstack(output) * ROOM[:2]


def train_one_model(
    name: str,
    seed: int,
    augment: AugmentedCalibration,
    train_rows: np.ndarray,
    validation_rows: np.ndarray,
    train_groups: np.ndarray,
    selection_candidate: CandidateTensorData,
    full_candidate: CandidateTensorData,
    selection_analytic: np.ndarray,
    full_analytic: np.ndarray,
    device: torch.device,
    output_dir: Path,
    *,
    max_epochs: int,
    quick: bool,
) -> tuple[nn.Module, dict]:
    delay_values, delay_mask = token_tensor(augment.queries, multimodal=False)
    multi_values, multi_mask = token_tensor(augment.queries, multimodal=True)
    target = (augment.target_xy / ROOM[:2]).astype(np.float32)

    def factory(final: bool) -> nn.Module:
        if name == "pointnet_delay_direct":
            return DirectPointNet(1)
        if name == "pointnet_multimodal_direct":
            return DirectPointNet(5)
        if name == "caez_grid_probability_delay":
            return GridProbabilityMLP(len(augment.survey_variants[0]) if final else len(train_groups))
        if name == "residual_to_analytic_delay":
            return ResidualAnalyticHead()
        if name == "pointnet_analytic_reranker_delay":
            return PointNetCandidateReranker(1)
        if name == "cross_attention_multimodal":
            return PathCrossAttentionReranker(5)
        if name == "candidate_self_attention_multimodal":
            return CandidateSelfAttentionReranker(5)
        raise ValueError(name)

    def loss_for(model: nn.Module, rows: np.ndarray, *, final: bool) -> torch.Tensor:
        y = _tensor(target, rows, device)
        if name == "pointnet_delay_direct":
            prediction = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device))
            return F.smooth_l1_loss(prediction, y)
        if name == "pointnet_multimodal_direct":
            prediction = model(_tensor(multi_values, rows, device), _tensor(multi_mask, rows, device))
            return F.smooth_l1_loss(prediction, y)
        if name == "caez_grid_probability_delay":
            ids = np.arange(len(augment.survey_variants[0])) if final else train_groups
            classes = torch.as_tensor(reference_positions()[ids] / ROOM[:2], dtype=torch.float32, device=device)
            prediction, logits = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device), classes)
            nearest = torch.argmin(torch.cdist(y, classes), dim=1)
            return F.smooth_l1_loss(prediction, y) + 0.15 * F.cross_entropy(logits, nearest)
        if name == "residual_to_analytic_delay":
            analytic = full_analytic if final else selection_analytic
            prediction = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device), _tensor(analytic, rows, device))
            return F.smooth_l1_loss(prediction, y)
        data = full_candidate if final else selection_candidate
        prediction, logits, gate = _candidate_forward(model, data, rows, device, name != "pointnet_analytic_reranker_delay")
        nearest = torch.argmin(torch.linalg.norm(_tensor(data.candidate_xy_norm, rows, device) - y[:, None, :], dim=2), dim=1)
        return F.smooth_l1_loss(prediction, y) + 0.12 * F.cross_entropy(logits, nearest) + 0.001 * torch.mean(gate)

    def validation_error(model: nn.Module) -> float:
        model.eval()
        values = []
        with torch.no_grad():
            for start in range(0, len(validation_rows), 256):
                rows = validation_rows[start:start + 256]
                if name == "pointnet_delay_direct":
                    pred = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device))
                elif name == "pointnet_multimodal_direct":
                    pred = model(_tensor(multi_values, rows, device), _tensor(multi_mask, rows, device))
                elif name == "caez_grid_probability_delay":
                    classes = torch.as_tensor(reference_positions()[train_groups] / ROOM[:2], dtype=torch.float32, device=device)
                    pred, _ = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device), classes)
                elif name == "residual_to_analytic_delay":
                    pred = model(_tensor(delay_values, rows, device), _tensor(delay_mask, rows, device), _tensor(selection_analytic, rows, device))
                else:
                    pred, _, _ = _candidate_forward(model, selection_candidate, rows, device, name != "pointnet_analytic_reranker_delay")
                values.append(pred.cpu().numpy())
        prediction = np.vstack(values) * ROOM[:2]
        return float(np.mean(np.linalg.norm(prediction - augment.target_xy[validation_rows], axis=1)))

    seed_everything(seed)
    selection_model = factory(False).to(device)
    optimizer = torch.optim.AdamW(selection_model.parameters(), lr=1.2e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(stable_seed("train-order", name, seed))
    history, best_error, best_epoch, best_state = [], np.inf, 1, None
    eval_every = 1 if quick else 4
    for epoch in range(1, max_epochs + 1):
        selection_model.train()
        losses = []
        for rows in _torch_batches(train_rows, 128, rng):
            loss = loss_for(selection_model, rows, final=False)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(selection_model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch % eval_every == 0 or epoch == max_epochs:
            error = validation_error(selection_model)
            history.append({"epoch": epoch, "training_loss": float(np.mean(losses)), "validation_mean_error_m": error})
            if error < best_error - 1.0e-9:
                best_error, best_epoch = error, epoch
                best_state = deepcopy({key: value.detach().cpu() for key, value in selection_model.state_dict().items()})

    seed_everything(stable_seed("refit", name, seed))
    final_model = factory(True).to(device)
    optimizer = torch.optim.AdamW(final_model.parameters(), lr=1.2e-3, weight_decay=1.0e-4)
    all_rows = np.arange(len(augment.queries))
    refit_rng = np.random.default_rng(stable_seed("refit-order", name, seed))
    for _ in range(best_epoch):
        final_model.train()
        for rows in _torch_batches(all_rows, 128, refit_rng):
            loss = loss_for(final_model, rows, final=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 5.0)
            optimizer.step()
    checkpoint = output_dir / "checkpoints" / f"{name}_seed{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "schema": SCHEMA, "method": name, "seed": seed, "selected_epoch": best_epoch,
        "validation_mean_error_m": best_error, "state_dict": final_model.state_dict(),
        "input_contract": "stored calibration augmentations only",
    }, checkpoint)
    return final_model, {
        "seed": seed, "selected_epoch": best_epoch, "validation_mean_error_m": best_error,
        "history": history, "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
        "selection_state_retained_only_for_audit": best_state is not None,
    }


def train_neural_models(
    augment: AugmentedCalibration,
    cache: CalibrationScoreCache,
    reference_xy: np.ndarray,
    train_groups: np.ndarray,
    validation_groups: np.ndarray,
    device: torch.device,
    output_dir: Path,
    *,
    max_epochs: int,
    quick: bool,
) -> tuple[dict[str, list[nn.Module]], dict]:
    train_rows = np.flatnonzero(np.isin(augment.group, train_groups))
    validation_rows = np.flatnonzero(np.isin(augment.group, validation_groups))
    selection_candidate = build_augmented_candidate_data(augment, cache, reference_xy, train_groups)
    full_candidate = build_augmented_candidate_data(augment, cache, reference_xy, np.arange(len(reference_xy)))
    selection_analytic = analytic_xy_for_augment(augment, cache, reference_xy, train_groups)
    full_analytic = analytic_xy_for_augment(augment, cache, reference_xy, np.arange(len(reference_xy)))
    names = (
        "pointnet_delay_direct",
        "pointnet_multimodal_direct",
        "caez_grid_probability_delay",
        "residual_to_analytic_delay",
        "pointnet_analytic_reranker_delay",
        "cross_attention_multimodal",
        "candidate_self_attention_multimodal",
    )
    seeds = LEARNED_SEEDS[:1] if quick else LEARNED_SEEDS
    models: dict[str, list[nn.Module]] = {name: [] for name in names}
    records: dict[str, list[dict]] = {name: [] for name in names}
    for name in names:
        for seed in seeds:
            started = time.perf_counter()
            model, record = train_one_model(
                name, seed, augment, train_rows, validation_rows, train_groups,
                selection_candidate, full_candidate, selection_analytic, full_analytic,
                device, output_dir, max_epochs=max_epochs, quick=quick,
            )
            record["training_and_refit_runtime_s"] = time.perf_counter() - started
            models[name].append(model)
            records[name].append(record)
            print(f"trained {name} seed={seed} epoch={record['selected_epoch']} val={record['validation_mean_error_m']:.4f}", flush=True)
    return models, records


METHOD_FEATURES = {
    "majdi_mpurge_map_corrected_prose": "delay + survey RP coordinates",
    "majdi_mca_eps1.6_k1": "delay + survey RP coordinates",
    "majdi_original_mpurge_P5_corrected": "delay + survey RP coordinates",
    "ablation_A_chamfer_inverse_top3": "delay + survey RP coordinates",
    "ablation_B_temperature_softmax_only": "delay + survey RP coordinates",
    "ablation_C_path_count_empty_only": "delay/path count + survey RP coordinates",
    "ablation_D_chamfer_temperature_pathcount": "delay/path count + survey RP coordinates",
    "subset_consensus_chamfer": "delay + survey RP coordinates",
    "graph_diffusion_corrected": "delay + survey RP coordinates/adjacency",
    "frozen_evomdp_rank": "delay-derived set distances + survey RP coordinates",
    "beta_survival_multimodal": "delay + received power + AoA + survey RP coordinates",
    "coherent_adp_toa": "noisy 8-element CIR + delay + survey RP coordinates",
    "survival_cir_toa": "delay + received power + AoA + noisy CIR + survey RP coordinates",
    "assigned_vt_consensus_aoa": "delay + AoA + survey RP coordinates",
    "residual_graph_multimodal": "delay + received power + AoA + noisy CIR + survey adjacency",
    "delay_feature_extratrees": "delay only; supervised stored calibration",
    "multimodal_feature_extratrees": "delay + power + AoA + noisy CIR summaries; supervised stored calibration",
    "rrle_moe": "delay-derived expert predictions/disagreement + survey RP coordinates",
    "pointnet_delay_direct": "unordered delay tokens; supervised stored calibration",
    "pointnet_multimodal_direct": "unordered delay/power/AoA tokens; supervised stored calibration",
    "caez_grid_probability_delay": "unordered delay encoding -> survey-grid probability",
    "residual_to_analytic_delay": "unordered delay tokens + corrected analytic coordinate",
    "pointnet_analytic_reranker_delay": "query/reference delay sets + corrected analytic diagnostics/RP coordinates",
    "cross_attention_multimodal": "genuine query-path/reference-path cross-attention over delay/power/AoA",
    "candidate_self_attention_multimodal": "path self-attention pooling + permutation-equivariant candidate attention",
}


def _no_evidence_override(prediction: np.ndarray, queries: Sequence[dict[str, np.ndarray]], reference_xy: np.ndarray) -> np.ndarray:
    output = np.asarray(prediction, dtype=np.float64).copy()
    fallback = centroid(reference_xy)
    for index, query in enumerate(queries):
        if not len(query["delay"]) or not np.all(np.isfinite(output[index])):
            output[index] = fallback
    return output


def evaluate_suite(
    protocol_path: Path,
    analytic_configs: dict,
    multimodal_configs: dict,
    tree_models: dict[str, ExtraTreesRegressor],
    rrle_router: ExtraTreesRegressor,
    neural_models: dict[str, list[nn.Module]],
    device: torch.device,
) -> tuple[list[dict], list[dict], dict]:
    bundle = np.load(protocol_path, allow_pickle=False)
    condition_names = [str(value) for value in bundle["condition_names"]]
    reference_xy = np.asarray(bundle["reference_xy_m"], dtype=np.float64)
    truth = np.asarray(bundle["query_xy_m"], dtype=np.float64)
    primary_rows: list[dict] = []
    seed_rows: list[dict] = []
    runtimes = {method: 0.0 for method in METHOD_FEATURES}
    graph_config = analytic_configs["graph_diffusion"]["selected"]
    b_config = analytic_configs["ablation_B_temperature_softmax_only"]["selected"]
    c_config = analytic_configs["ablation_C_path_count_empty_only"]["selected"]
    d_config = analytic_configs["ablation_D_all_three"]["selected"]
    beta_config = multimodal_configs["beta_survival"]["selected"]
    coherent_config = multimodal_configs["coherent_adp_toa"]["selected"]
    survival_config = multimodal_configs["survival_cir_toa"]["selected"]
    vt_config = multimodal_configs["assigned_vt_consensus"]["selected"]
    rrle_temperature = float(rrle_router._rrle_temperature)  # type: ignore[attr-defined]

    for condition_index, condition_name in enumerate(condition_names):
        survey = [observation_from_arrays(bundle, "survey", condition_index, index) for index in range(len(reference_xy))]
        queries = [observation_from_arrays(bundle, "query", condition_index, index) for index in range(len(truth))]
        reference_delay = [item["delay"] for item in survey]
        corrected_matrix = np.full((len(queries), len(reference_xy)), np.inf, dtype=np.float64)
        coverage_matrix = np.zeros_like(corrected_matrix)
        chamfer_matrix = np.full_like(corrected_matrix, np.inf)
        evo_matrix = np.full_like(corrected_matrix, np.inf)
        original_matrix = np.full_like(corrected_matrix, np.inf)
        beta_matrix = np.full_like(corrected_matrix, np.inf)
        coherent_matrix = np.full_like(corrected_matrix, np.inf)
        survival_matrix = np.full_like(corrected_matrix, np.inf)
        predictions: dict[str, list[np.ndarray]] = {
            name: [] for name in METHOD_FEATURES
            if name not in neural_models and name not in tree_models and name != "rrle_moe"
        }
        vt_started = time.perf_counter()
        vt_centres = assigned_vt_centres(survey, reference_xy, float(vt_config["vt_cluster_eps_m"]))
        runtimes["assigned_vt_consensus_aoa"] += time.perf_counter() - vt_started

        for query_index, query in enumerate(queries):
            delay = query["delay"]
            if not len(delay):
                fallback = centroid(reference_xy)
                for method in predictions:
                    predictions[method].append(fallback.copy())
                continue

            started = time.perf_counter()
            corrected, coverage = corrected_mpurge_scores(delay, reference_delay)
            corrected_matrix[query_index], coverage_matrix[query_index] = corrected, coverage
            predictions["majdi_mpurge_map_corrected_prose"].append(inverse_topk(corrected, reference_xy))
            runtimes["majdi_mpurge_map_corrected_prose"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["majdi_mca_eps1.6_k1"].append(mca_predict(delay, reference_delay, reference_xy))
            runtimes["majdi_mca_eps1.6_k1"] += time.perf_counter() - started

            started = time.perf_counter()
            original, _ = corrected_mpurge_scores(
                delay, reference_delay, half_window_p=2, alpha=0.5,
                normalized_pattern=False, coverage_penalty=False,
            )
            original_matrix[query_index] = original
            predictions["majdi_original_mpurge_P5_corrected"].append(inverse_topk(original, reference_xy))
            runtimes["majdi_original_mpurge_P5_corrected"] += time.perf_counter() - started

            started = time.perf_counter()
            chamfer = chamfer_scores(delay, reference_delay)
            chamfer_matrix[query_index] = chamfer
            predictions["ablation_A_chamfer_inverse_top3"].append(inverse_topk(chamfer, reference_xy))
            runtimes["ablation_A_chamfer_inverse_top3"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["ablation_B_temperature_softmax_only"].append(softmax_topk(corrected, reference_xy, float(b_config["temperature"])))
            runtimes["ablation_B_temperature_softmax_only"] += time.perf_counter() - started

            started = time.perf_counter()
            count_adjusted = path_count_adjusted(corrected, delay, reference_delay, float(c_config["path_count_strength"]))
            predictions["ablation_C_path_count_empty_only"].append(inverse_topk(count_adjusted, reference_xy))
            runtimes["ablation_C_path_count_empty_only"] += time.perf_counter() - started

            started = time.perf_counter()
            all_three = chamfer_scores(delay, reference_delay, path_count_rule=float(d_config["path_count_strength"]))
            predictions["ablation_D_chamfer_temperature_pathcount"].append(softmax_topk(all_three, reference_xy, float(d_config["temperature"])))
            runtimes["ablation_D_chamfer_temperature_pathcount"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["subset_consensus_chamfer"].append(subset_consensus(delay, reference_delay, reference_xy))
            runtimes["subset_consensus_chamfer"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["graph_diffusion_corrected"].append(graph_predict(
                corrected, reference_xy, temperature=float(graph_config["temperature"]),
                strength=float(graph_config["strength"]), steps=int(graph_config["steps"]),
            ))
            runtimes["graph_diffusion_corrected"] += time.perf_counter() - started

            started = time.perf_counter()
            evo = evo_scores(delay, reference_delay)
            evo_matrix[query_index] = evo
            predictions["frozen_evomdp_rank"].append(softmax_topk(evo, reference_xy, EVO_TEMPERATURE, EVO_TOP_K))
            runtimes["frozen_evomdp_rank"] += time.perf_counter() - started

            started = time.perf_counter()
            beta = beta_survival_scores(
                query, survey, angle_weight=float(beta_config["angle_weight"]),
                power_weight=float(beta_config["power_weight"]),
            )
            beta_matrix[query_index] = beta
            predictions["beta_survival_multimodal"].append(inverse_topk(beta, reference_xy))
            runtimes["beta_survival_multimodal"] += time.perf_counter() - started

            started = time.perf_counter()
            coherent = coherent_scores(query, survey, float(coherent_config["delay_weight"]))
            coherent_matrix[query_index] = coherent
            predictions["coherent_adp_toa"].append(inverse_topk(coherent, reference_xy))
            runtimes["coherent_adp_toa"] += time.perf_counter() - started

            started = time.perf_counter()
            survival = (float(survival_config["beta_weight"]) * beta +
                        float(survival_config["cir_weight"]) * coherent +
                        float(survival_config["delay_weight"]) * chamfer)
            survival_matrix[query_index] = survival
            predictions["survival_cir_toa"].append(inverse_topk(survival, reference_xy))
            runtimes["survival_cir_toa"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["assigned_vt_consensus_aoa"].append(assigned_vt_predict(
                query, vt_centres, reference_xy, float(vt_config["query_consensus_eps_m"]),
            ))
            runtimes["assigned_vt_consensus_aoa"] += time.perf_counter() - started

            started = time.perf_counter()
            predictions["residual_graph_multimodal"].append(graph_predict(
                survival, reference_xy, temperature=float(graph_config["temperature"]),
                strength=float(graph_config["strength"]), steps=int(graph_config["steps"]),
            ))
            runtimes["residual_graph_multimodal"] += time.perf_counter() - started

        # Empty queries took the branch above; matrices remain invalid and are
        # intentionally handled by the shared centroid override below.
        for method in list(predictions):
            predictions[method] = list(_no_evidence_override(np.asarray(predictions[method]), queries, reference_xy))

        delay_values, delay_mask = token_tensor(queries, multimodal=False)
        multi_values, multi_mask = token_tensor(queries, multimodal=True)
        analytic_xy_norm = np.asarray(predictions["majdi_mpurge_map_corrected_prose"]) / ROOM[:2]
        candidate_data = build_evaluation_candidate_data(
            queries, survey, corrected_matrix, coverage_matrix, reference_xy,
        )
        class_xy_norm = reference_xy / ROOM[:2]
        for name, model_list in neural_models.items():
            started = time.perf_counter()
            per_seed = []
            for seed, model in zip(LEARNED_SEEDS[:len(model_list)], model_list, strict=True):
                seed_prediction = predict_torch_model(
                    name, model,
                    delay_values=delay_values, delay_mask=delay_mask,
                    multi_values=multi_values, multi_mask=multi_mask,
                    analytic_xy_norm=analytic_xy_norm,
                    candidate_data=candidate_data if "reranker" in name or "attention" in name else None,
                    class_xy_norm=class_xy_norm,
                    device=device,
                )
                seed_prediction = _no_evidence_override(seed_prediction, queries, reference_xy)
                per_seed.append(seed_prediction)
                for query_index, prediction in enumerate(seed_prediction):
                    seed_rows.append({
                        "condition": condition_name, "query": query_index, "method": name,
                        "seed": int(seed), "truth_xy_m": truth[query_index].tolist(),
                        "prediction_xy_m": prediction.tolist(),
                        "error_m": float(np.linalg.norm(prediction - truth[query_index])),
                    })
            predictions[name] = list(np.mean(np.stack(per_seed), axis=0))
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            runtimes[name] += time.perf_counter() - started

        started = time.perf_counter()
        predictions["delay_feature_extratrees"] = list(_no_evidence_override(
            tree_models["delay_feature_extratrees"].predict(np.asarray([delay_feature_vector(item["delay"]) for item in queries])),
            queries, reference_xy,
        ))
        runtimes["delay_feature_extratrees"] += time.perf_counter() - started
        started = time.perf_counter()
        predictions["multimodal_feature_extratrees"] = list(_no_evidence_override(
            tree_models["multimodal_feature_extratrees"].predict(np.asarray([multimodal_feature_vector(item) for item in queries])),
            queries, reference_xy,
        ))
        runtimes["multimodal_feature_extratrees"] += time.perf_counter() - started

        started = time.perf_counter()
        rrle_experts, rrle_meta = [], []
        for query_index, query in enumerate(queries):
            experts = np.vstack((
                predictions["majdi_mpurge_map_corrected_prose"][query_index],
                predictions["ablation_A_chamfer_inverse_top3"][query_index],
                predictions["frozen_evomdp_rank"][query_index],
                predictions["graph_diffusion_corrected"][query_index],
            ))
            rrle_experts.append(experts)
            spread = np.mean(np.linalg.norm(experts[:, None, :] - experts[None, :, :], axis=2))
            rrle_meta.append(np.concatenate((delay_feature_vector(query["delay"]), experts.reshape(-1) / np.tile(ROOM[:2], 4), [spread / ROOM_DIAGONAL])))
        rrle_experts = np.asarray(rrle_experts)
        predicted_error = np.maximum(rrle_router.predict(np.asarray(rrle_meta)), 0.0)
        weights = np.exp(-predicted_error / rrle_temperature)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1.0e-12)
        predictions["rrle_moe"] = list(_no_evidence_override(np.sum(rrle_experts * weights[..., None], axis=1), queries, reference_xy))
        runtimes["rrle_moe"] += time.perf_counter() - started

        if set(predictions) != set(METHOD_FEATURES):
            missing = sorted(set(METHOD_FEATURES) - set(predictions))
            extra = sorted(set(predictions) - set(METHOD_FEATURES))
            raise AssertionError(f"method registry mismatch missing={missing} extra={extra}")
        for method, values in predictions.items():
            for query_index, prediction in enumerate(values):
                primary_rows.append({
                    "protocol": "MPUrge-MAP executable replacement scene",
                    "condition": condition_name,
                    "query": query_index,
                    "method": method,
                    "features": METHOD_FEATURES[method],
                    "truth_xy_m": truth[query_index].tolist(),
                    "prediction_xy_m": np.asarray(prediction).tolist(),
                    "error_m": float(np.linalg.norm(np.asarray(prediction) - truth[query_index])),
                    "query_path_count": int(len(queries[query_index]["delay"])),
                    "survey_path_counts": [int(len(item["delay"])) for item in survey],
                })
        print(f"evaluated {condition_name} ({len(queries)} locked queries, {len(predictions)} methods)", flush=True)
    bundle.close()
    runtime_rows = {
        method: {
            "evaluation_runtime_s": float(value),
            "mean_runtime_ms_per_query": 1000.0 * float(value) / max(len(condition_names) * len(truth), 1),
        }
        for method, value in runtimes.items()
    }
    return primary_rows, seed_rows, runtime_rows


def summarize_rows(rows: Sequence[dict]) -> dict:
    methods = sorted({row["method"] for row in rows})
    conditions = sorted({row["condition"] for row in rows})

    def metrics(values: np.ndarray) -> dict:
        return {
            "count": int(len(values)),
            "mean_error_m": float(np.mean(values)),
            "rmse_m": float(np.sqrt(np.mean(np.square(values)))),
            "median_error_m": float(np.median(values)),
            "p90_error_m": float(np.quantile(values, 0.9)),
        }

    per_condition = []
    for method in methods:
        for condition in conditions:
            values = np.asarray([row["error_m"] for row in rows if row["method"] == method and row["condition"] == condition])
            per_condition.append({"method": method, "condition": condition, **metrics(values)})
    pooled = []
    macro = []
    for method in methods:
        values = np.asarray([row["error_m"] for row in rows if row["method"] == method])
        pooled.append({"method": method, **metrics(values)})
        selected = [row for row in per_condition if row["method"] == method]
        macro.append({
            "method": method,
            "conditions": len(selected),
            "macro_mean_error_m": float(np.mean([row["mean_error_m"] for row in selected])),
            "macro_rmse_m": float(np.mean([row["rmse_m"] for row in selected])),
            "macro_median_error_m": float(np.mean([row["median_error_m"] for row in selected])),
            "macro_p90_error_m": float(np.mean([row["p90_error_m"] for row in selected])),
        })
    return {"per_condition": per_condition, "pooled": pooled, "condition_macro": macro}


def paired_query_block_bootstrap(rows: Sequence[dict], baseline: str, *, draws: int = 5000) -> list[dict]:
    methods = sorted({row["method"] for row in rows})
    lookup = {(row["method"], row["condition"], row["query"]): row["error_m"] for row in rows}
    conditions = sorted({row["condition"] for row in rows})
    queries = sorted({int(row["query"]) for row in rows})
    baseline_matrix = np.asarray([[lookup[(baseline, condition, query)] for condition in conditions] for query in queries])
    rng = np.random.default_rng(stable_seed("paired-bootstrap"))
    indices = rng.integers(0, len(queries), size=(draws, len(queries)))
    output = []
    for method in methods:
        if method == baseline:
            continue
        candidate = np.asarray([[lookup[(method, condition, query)] for condition in conditions] for query in queries])
        gain = baseline_matrix - candidate
        samples = gain[indices].mean(axis=(1, 2))
        condition_ci = {}
        for column, condition in enumerate(conditions):
            condition_samples = gain[:, column][indices].mean(axis=1)
            condition_ci[condition] = {
                "mean_reduction_m": float(np.mean(gain[:, column])),
                "95ci_m": [float(np.quantile(condition_samples, 0.025)), float(np.quantile(condition_samples, 0.975))],
            }
        output.append({
            "candidate": method, "baseline": baseline,
            "positive_means_candidate_better": True,
            "mean_reduction_m": float(np.mean(gain)),
            "query_block_clustered_95ci_m": [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))],
            "bootstrap_draws": draws,
            "blocks": len(queries),
            "conditions_per_block": len(conditions),
            "probability_reduction_positive": float(np.mean(samples > 0.0)),
            "per_condition_query_bootstrap": condition_ci,
        })
    return output


@torch.no_grad()
def invariance_audit(neural_models: dict[str, list[nn.Module]], tree_models: dict[str, ExtraTreesRegressor], device: torch.device) -> dict:
    rng = np.random.default_rng(stable_seed("invariance-audit"))
    batch, candidates, paths = 3, 5, 6
    delay = rng.normal(size=(batch, paths, 1)).astype(np.float32)
    multi = rng.normal(size=(batch, paths, 5)).astype(np.float32)
    mask = np.asarray([[True, True, True, True, False, False], [True] * 6, [True, True, True, False, False, False]])

    def permute_paths(values: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        output, output_mask = values.copy(), masks.copy()
        for row in range(len(values)):
            order = rng.permutation(values.shape[-2])
            output[row] = values[row, order]
            output_mask[row] = masks[row, order]
        return output, output_mask

    def padded(values: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return (np.concatenate((values, np.zeros((*values.shape[:-2], 3, values.shape[-1]), dtype=values.dtype)), axis=-2),
                np.concatenate((masks, np.zeros((*masks.shape[:-1], 3), dtype=bool)), axis=-1))

    class_xy = torch.as_tensor(reference_positions() / ROOM[:2], dtype=torch.float32, device=device)
    analytic = rng.uniform(size=(batch, 2)).astype(np.float32)
    checks = []
    for name in ("pointnet_delay_direct", "pointnet_multimodal_direct", "caez_grid_probability_delay", "residual_to_analytic_delay"):
        model = neural_models[name][0].eval()
        source, source_mask = (delay, mask) if name != "pointnet_multimodal_direct" else (multi, mask)
        shuffled, shuffled_mask = permute_paths(source, source_mask)
        extended, extended_mask = padded(source, source_mask)
        if name.startswith("pointnet"):
            base = model(torch.as_tensor(source, device=device), torch.as_tensor(source_mask, device=device))
            shuffled_value = model(torch.as_tensor(shuffled, device=device), torch.as_tensor(shuffled_mask, device=device))
            padded_value = model(torch.as_tensor(extended, device=device), torch.as_tensor(extended_mask, device=device))
        elif name.startswith("caez"):
            base, _ = model(torch.as_tensor(source, device=device), torch.as_tensor(source_mask, device=device), class_xy)
            shuffled_value, _ = model(torch.as_tensor(shuffled, device=device), torch.as_tensor(shuffled_mask, device=device), class_xy)
            padded_value, _ = model(torch.as_tensor(extended, device=device), torch.as_tensor(extended_mask, device=device), class_xy)
        else:
            analytic_tensor = torch.as_tensor(analytic, device=device)
            base = model(torch.as_tensor(source, device=device), torch.as_tensor(source_mask, device=device), analytic_tensor)
            shuffled_value = model(torch.as_tensor(shuffled, device=device), torch.as_tensor(shuffled_mask, device=device), analytic_tensor)
            padded_value = model(torch.as_tensor(extended, device=device), torch.as_tensor(extended_mask, device=device), analytic_tensor)
        shuffle_delta = float(torch.max(torch.abs(base - shuffled_value)).cpu())
        padding_delta = float(torch.max(torch.abs(base - padded_value)).cpu())
        checks.append({"method": name, "path_shuffle_max_abs_delta": shuffle_delta, "padding_max_abs_delta": padding_delta,
                       "pass": shuffle_delta <= 2.0e-5 and padding_delta <= 2.0e-5})

    candidate_xy = rng.uniform(size=(batch, candidates, 2)).astype(np.float32)
    diagnostics = rng.normal(size=(batch, candidates, 3)).astype(np.float32)
    analytic_xy = rng.uniform(size=(batch, 2)).astype(np.float32)
    reference_delay = rng.normal(size=(batch, candidates, paths, 1)).astype(np.float32)
    reference_multi = rng.normal(size=(batch, candidates, paths, 5)).astype(np.float32)
    reference_mask = np.ones((batch, candidates, paths), dtype=bool)
    reference_mask[:, :, -1] = False

    for name in ("pointnet_analytic_reranker_delay", "cross_attention_multimodal", "candidate_self_attention_multimodal"):
        model = neural_models[name][0].eval()
        q, qmask = (delay, mask) if name == "pointnet_analytic_reranker_delay" else (multi, mask)
        refs = reference_delay if name == "pointnet_analytic_reranker_delay" else reference_multi
        args = [torch.as_tensor(q, device=device), torch.as_tensor(qmask, device=device),
                torch.as_tensor(refs, device=device), torch.as_tensor(reference_mask, device=device),
                torch.as_tensor(candidate_xy, device=device), torch.as_tensor(diagnostics, device=device),
                torch.as_tensor(analytic_xy, device=device)]
        base = model(*args)[0]
        q_shuffled, qm_shuffled = permute_paths(q, qmask)
        refs_shuffled, refs_mask_shuffled = refs.copy(), reference_mask.copy()
        for b in range(batch):
            for k in range(candidates):
                order = rng.permutation(paths)
                refs_shuffled[b, k] = refs[b, k, order]
                refs_mask_shuffled[b, k] = reference_mask[b, k, order]
        shuffled_value = model(
            torch.as_tensor(q_shuffled, device=device), torch.as_tensor(qm_shuffled, device=device),
            torch.as_tensor(refs_shuffled, device=device), torch.as_tensor(refs_mask_shuffled, device=device),
            args[4], args[5], args[6],
        )[0]
        permutation = rng.permutation(candidates)
        candidate_value = model(
            args[0], args[1], args[2][:, permutation], args[3][:, permutation],
            args[4][:, permutation], args[5][:, permutation], args[6],
        )[0]
        q_padded, qm_padded = padded(q, qmask)
        refs_padded = np.concatenate((refs, np.zeros((batch, candidates, 3, refs.shape[-1]), dtype=np.float32)), axis=2)
        refs_mask_padded = np.concatenate((reference_mask, np.zeros((batch, candidates, 3), dtype=bool)), axis=2)
        padded_value = model(
            torch.as_tensor(q_padded, device=device), torch.as_tensor(qm_padded, device=device),
            torch.as_tensor(refs_padded, device=device), torch.as_tensor(refs_mask_padded, device=device),
            args[4], args[5], args[6],
        )[0]
        deltas = {
            "path_shuffle_max_abs_delta": float(torch.max(torch.abs(base - shuffled_value)).cpu()),
            "candidate_permutation_max_abs_delta": float(torch.max(torch.abs(base - candidate_value)).cpu()),
            "padding_max_abs_delta": float(torch.max(torch.abs(base - padded_value)).cpu()),
        }
        checks.append({"method": name, **deltas, "pass": max(deltas.values()) <= 2.0e-5})

    observation = {
        "delay": np.asarray([3.0, 1.0, 2.0]), "power": np.asarray([-30.0, -40.0, -35.0]),
        "aoa": np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        "cir": np.ones((1, 8, 256), dtype=np.complex64),
    }
    order = np.asarray([2, 0, 1])
    shuffled_observation = {"delay": observation["delay"][order], "power": observation["power"][order],
                            "aoa": observation["aoa"][order], "cir": observation["cir"]}
    delay_delta = float(np.max(np.abs(delay_feature_vector(observation["delay"]) - delay_feature_vector(shuffled_observation["delay"]))))
    multi_delta = float(np.max(np.abs(multimodal_feature_vector(observation) - multimodal_feature_vector(shuffled_observation))))
    checks.append({"method": "tree_feature_extractors", "path_shuffle_max_abs_delta": max(delay_delta, multi_delta),
                   "pass": max(delay_delta, multi_delta) <= 1.0e-12})
    fallback = _no_evidence_override(np.asarray([[99.0, 99.0]]), [{"delay": np.empty(0)}], reference_positions())[0]
    empty_pass = bool(np.allclose(fallback, centroid(reference_positions())))
    checks.append({"method": "shared_empty_no_evidence_policy", "prediction_xy_m": fallback.tolist(), "pass": empty_pass})
    return {
        "tolerance": 2.0e-5,
        "checks": checks,
        "all_pass": all(row["pass"] for row in checks),
        "claimed_set_models_checked": 7,
        "tree_feature_extractors_checked": True,
        "empty_policy_checked": True,
    }


def applicability_record() -> dict:
    included = [{"method": method, "status": "RUNNABLE_AND_RUN", "features": features,
                 "same_acquisition_locations": True} for method, features in METHOD_FEATURES.items()]
    excluded = [
        {
            "method_family": "AoD-conditioned fingerprinting",
            "status": "NOT_INSTANTIATED_NO_REQUESTED_METHOD_REQUIRES_AOD",
            "reason": "The requested MAP methods run here use receive AoA and/or noisy CIR and do not require AoD. A same-acquisition noisy transmit-array/channel-sounding AoD is eligible under this benchmark; the present helper simply does not instantiate that sensor. Latent ray departure vectors would remain forbidden as direct inputs.",
            "not_penalized_for_feature_type": True,
        },
        {
            "method_family": "native multi-transmitter Beta VT survival",
            "status": "TASK_MISMATCH_NATIVE_FORM",
            "reason": "The MAP scene has one AP and the native method outputs/filters VTs per elementary transmitter. A measurable delay/power/AoA Beta-survival localization adapter is run instead.",
            "not_penalized_for_feature_type": True,
        },
        {
            "method_family": "oracle assigned-VT consensus",
            "status": "ORACLE_EXCLUDED",
            "reason": "True VT/ray associations are simulator identities. The run reconstructs unlabeled VT clusters from survey range/AoA endpoints and performs association-free query consensus.",
            "not_penalized_for_feature_type": True,
        },
        {
            "method_family": "SDF/NeRF floor-plan-conditioned localizer",
            "status": "MISSING_ACQUISITION_CHANNEL",
            "reason": "No measured depth/images/floor-plan mesh is part of these 80/100 acquisitions. Supplying the simulator geometry would expose a clean auxiliary channel.",
            "not_penalized_for_feature_type": True,
        },
        {
            "method_family": "temporal HMM/Kalman/GRU",
            "status": "TASK_MISMATCH",
            "reason": "The 100 locked queries are independently drawn static locations, not a causal trajectory.",
            "not_penalized_for_feature_type": True,
        },
    ]
    return {"fairness_axis": "same acquisition locations and calibration-map budget", "included": included, "excluded": excluded}


def _write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(json_ready(row), separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Plumbing smoke; not a result")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reuse-protocol", action="store_true")
    parser.add_argument("--copies", type=int)
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    output_dir = args.output_dir or ROOT / "research" / "four_paper_report" / ("corrected_mpurge_map_smoke" if args.quick else "corrected_mpurge_map_full")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    device = configure_device(require_cuda=not args.quick)
    protocol_path = output_dir / "protocol_observations.npz"
    if not (args.reuse_protocol and protocol_path.exists()):
        protocol_manifest = materialize_protocol(protocol_path, quick=args.quick)
    else:
        with np.load(protocol_path, allow_pickle=False) as existing:
            protocol_manifest = {
                "path": str(protocol_path), "sha256": sha256(protocol_path),
                "conditions": len(existing["condition_names"]),
                "survey_locations": len(existing["reference_xy_m"]),
                "locked_queries": len(existing["query_xy_m"]),
                "simulator_calls": "reused materialized artifact; zero this invocation",
                "new_spatial_training_locations": 0,
            }

    # Selection receives only the ideal survey acquisitions and RP coordinates.
    # Locked query arrays remain unopened until selection_freeze.json is written.
    with np.load(protocol_path, allow_pickle=False) as selection_bundle:
        reference_xy = np.asarray(selection_bundle["reference_xy_m"], dtype=np.float64)
        base_survey = [observation_from_arrays(selection_bundle, "survey", 0, index) for index in range(len(reference_xy))]
        condition_name_hash = hashlib.sha256(np.asarray(selection_bundle["condition_names"]).tobytes()).hexdigest()
    train_groups, validation_groups = grouped_split(reference_xy)
    copies = args.copies if args.copies is not None else (2 if args.quick else 16)
    max_epochs = args.epochs if args.epochs is not None else (4 if args.quick else 80)
    augment = build_calibration_augmentations(base_survey, reference_xy, copies)
    score_cache_path = output_dir / "calibration_score_cache.npz"
    cache = compute_calibration_score_cache(augment, score_cache_path)
    analytic_configs = select_analytic_configs(augment, cache, reference_xy, train_groups, validation_groups)
    multimodal_configs = select_multimodal_configs(augment, reference_xy, train_groups, validation_groups, quick=args.quick)
    tree_models, tree_selection = fit_extra_trees_models(
        augment, reference_xy, train_groups, validation_groups, output_dir, quick=args.quick,
    )
    graph_config = analytic_configs["graph_diffusion"]["selected"]
    rrle_router, rrle_selection = fit_rrle_router(
        augment, cache, reference_xy, train_groups, validation_groups, graph_config, output_dir, quick=args.quick,
    )
    neural_models, neural_training = train_neural_models(
        augment, cache, reference_xy, train_groups, validation_groups, device, output_dir,
        max_epochs=max_epochs, quick=args.quick,
    )
    selection_freeze = {
        "schema": SCHEMA,
        "status": "SMOKE_SELECTION" if args.quick else "FULL_SELECTION_FROZEN_BEFORE_LOCKED_EVALUATION",
        "claim_boundary": "executable replacement-scene reconstruction; not exact NIST Q-D paper reproduction",
        "protocol_observation_sha256": sha256(protocol_path),
        "condition_name_hash": condition_name_hash,
        "selection_inputs": "80 stored survey acquisitions and their RP coordinates only",
        "locked_query_arrays_accessed_by_selection": False,
        "training_augmentation": {
            "source": "numeric corruption/dropout of stored survey observations only",
            "physical_RP_groups": len(reference_xy), "copies_per_group": copies,
            "samples": len(augment.queries), "new_simulator_calls": 0,
        },
        "whole_RP_split": {"training_groups": train_groups.tolist(), "validation_groups": validation_groups.tolist()},
        "analytic_configs": analytic_configs,
        "multimodal_configs": multimodal_configs,
        "tree_selection": tree_selection,
        "rrle_selection": rrle_selection,
        "neural_training": neural_training,
        "learned_seeds": list(LEARNED_SEEDS[:1] if args.quick else LEARNED_SEEDS),
        "maximum_epochs": max_epochs,
        "calibration_score_cache_sha256": sha256(score_cache_path),
    }
    selection_path = output_dir / "selection_freeze.json"
    write_json(selection_path, selection_freeze)

    primary_rows, seed_rows, runtimes = evaluate_suite(
        protocol_path, analytic_configs, multimodal_configs, tree_models,
        rrle_router, neural_models, device,
    )
    raw_path = output_dir / "raw_predictions.jsonl"
    seed_raw_path = output_dir / "learned_seed_predictions.jsonl"
    _write_jsonl(raw_path, primary_rows)
    _write_jsonl(seed_raw_path, seed_rows)
    aggregates = summarize_rows(primary_rows)
    bootstrap = paired_query_block_bootstrap(
        primary_rows, "majdi_mpurge_map_corrected_prose", draws=800 if args.quick else 5000,
    )
    invariance = invariance_audit(neural_models, tree_models, device)
    applicability = applicability_record()
    aggregate_path = output_dir / "aggregate_metrics.json"
    bootstrap_path = output_dir / "paired_bootstrap_vs_corrected_majid.json"
    invariance_path = output_dir / "invariance_and_edge_case_audit.json"
    applicability_path = output_dir / "applicability_and_exclusions.json"
    runtime_path = output_dir / "runtimes.json"
    write_json(aggregate_path, aggregates)
    write_json(bootstrap_path, bootstrap)
    write_json(invariance_path, invariance)
    write_json(applicability_path, applicability)
    write_json(runtime_path, runtimes)
    result = {
        "schema": SCHEMA,
        "status": "QUICK_SMOKE_NOT_EVIDENCE" if args.quick else "FULL_RUN_COMPLETE",
        "claim": "corrected, acquisition-budget-fair executable MPUrge-MAP replacement-scene benchmark",
        "protocol": protocol_manifest,
        "fairness": {
            "same_80_survey_and_locked_query_acquisitions": True,
            "condition_specific_survey_shared_by_all_reference_methods": True,
            "independent_survey_query_corruption": True,
            "extra_spatial_training_samples": 0,
            "simulator_ray_or_VT_ID_exposed": False,
            "physically_measurable_non_delay_features_allowed": True,
            "shared_empty_query_fallback": "centroid of available survey RP coordinates",
        },
        "counts": {
            "conditions": len({row["condition"] for row in primary_rows}),
            "locked_queries": len({row["query"] for row in primary_rows}),
            "methods": len({row["method"] for row in primary_rows}),
            "primary_raw_rows": len(primary_rows),
            "learned_seed_raw_rows": len(seed_rows),
        },
        "artifacts": {
            "protocol_observations": str(protocol_path),
            "selection_freeze": str(selection_path),
            "raw_predictions": str(raw_path),
            "learned_seed_predictions": str(seed_raw_path),
            "aggregate_metrics": str(aggregate_path),
            "paired_bootstrap": str(bootstrap_path),
            "invariance_audit": str(invariance_path),
            "applicability": str(applicability_path),
            "runtimes": str(runtime_path),
        },
        "pooled_metrics": aggregates["pooled"],
        "condition_macro_metrics": aggregates["condition_macro"],
        "invariance_all_pass": invariance["all_pass"],
        "total_runtime_s": time.perf_counter() - started,
        "device": str(device),
    }
    result_path = output_dir / "results.json"
    write_json(result_path, result)
    source_paths = [Path(__file__), HERE / "multimodal_receiver.py", PAPER_DIR / "common.py", RESEARCH_LEVEL / "majdi_paper_methods.py"]
    artifact_paths = [protocol_path, score_cache_path, selection_path, raw_path, seed_raw_path, aggregate_path,
                      bootstrap_path, invariance_path, applicability_path, runtime_path, result_path]
    artifact_paths.extend(Path(row["checkpoint"]) for records in neural_training.values() for row in records)
    artifact_paths.extend([Path(tree_selection[name]["checkpoint"]) for name in tree_selection])
    artifact_paths.append(Path(rrle_selection["checkpoint"]))
    manifest = {
        "schema": SCHEMA,
        "created_after_full_evaluation": True,
        "source_sha256": {str(path): sha256(path) for path in source_paths},
        "artifact_sha256": {str(path): sha256(path) for path in artifact_paths if path.exists()},
        "environment": {
            "python": sys.version,
            "numpy": np.__version__, "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    manifest_path = output_dir / "run_manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({
        "status": result["status"], "output_dir": str(output_dir),
        "counts": result["counts"], "invariance_all_pass": result["invariance_all_pass"],
        "total_runtime_s": result["total_runtime_s"],
        "top_pooled": sorted(aggregates["pooled"], key=lambda row: row["mean_error_m"])[:8],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
