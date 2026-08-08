"""Corrected, leakage-controlled original-MPUrge VT-construction benchmark.

All estimators consume only the frozen calibration acquisitions.  Geometric
image sources are used by the disclosed replacement generator and by the
evaluator, but their coordinates/identity/cardinality never enter a primary
estimator.  The only labelled-cardinality method is explicitly suffixed
``oracle_cardinality_ablation`` and excluded from primary summaries.

The native MPUrge/PBA branches make the paper ambiguity explicit:

* total window P=5 is converted once to half-window p=2;
* current-iteration conflicts are ordered by composite dissimilarity;
* printed-algebra and geometric-order cross checks are both run;
* star and conflict-resolved connected-component groupings are both stored;
* transmitter identity is retained through matching, grouping and support;
* beta is swept dynamically until every candidate is removed.

Multimodal methods use delay, observed power, AoA or CIR extracted from the
same noisy array-CIR acquisitions as the delay-only native methods.  They do
not receive extra spatial samples, ray IDs, VT labels or query coordinates.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import gzip
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import time
from typing import Iterable, Literal, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.stats import beta as beta_distribution
import torch
from torch import nn


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from mpurge_vt_receiver_snapshot_20260807 import (  # noqa: E402
    MultimodalFingerprint,
    simulate_multimodal_fingerprint,
)


SCHEMA = "corrected-original-mpurge-vt-suite-v2-frozen-receiver"
TOTAL_WINDOW_P = 5
HALF_WINDOW_P = (TOTAL_WINDOW_P - 1) // 2
ALPHA = 0.5
MIN_GROUP_RPS = 3
PRIMARY_GATE_M = 0.30
DEFAULT_NOISE_STDS_M = (0.0, 0.25, 1.0, 3.0)
DEFAULT_SCENE_SEEDS = (26080701, 26080702, 26080703, 26080704, 26080705, 26080706)
DEFAULT_ALGORITHM_SEED = 20260807


@dataclass(frozen=True)
class Source:
    xyz_m: np.ndarray
    tx_id: int
    order: int


@dataclass(frozen=True)
class Scene:
    seed: int
    room_xyz_m: np.ndarray
    transmitters_xyz_m: np.ndarray
    sources_by_tx: tuple[tuple[Source, ...], ...]
    truth_vts_by_tx: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class Match:
    index_a: int
    index_b: int
    value_a: float
    value_b: float
    dissimilarity: float
    iteration: int


@dataclass(frozen=True)
class Acquisition:
    fingerprint: MultimodalFingerprint
    indirect: MultimodalFingerprint
    direct_removed: bool
    direct_residual_m: float | None


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def stable_seed(*parts: object) -> int:
    payload = (SCHEMA + ":" + ":".join(map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32 - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mirror_wall(point: np.ndarray, room: np.ndarray) -> list[np.ndarray]:
    x, y, z = map(float, point)
    width, depth, _ = map(float, room)
    return [
        np.asarray([-x, y, z]),
        np.asarray([2.0 * width - x, y, z]),
        np.asarray([x, -y, z]),
        np.asarray([x, 2.0 * depth - y, z]),
    ]


def make_scene(seed: int) -> Scene:
    """Random disclosed rectangular image-source scene with variable VT count."""

    rng = np.random.default_rng(seed)
    room = np.asarray([rng.uniform(16.0, 21.0), rng.uniform(15.0, 20.0), 3.0])
    transmitters = np.column_stack(
        (
            rng.uniform(2.0, room[0] - 2.0, 4),
            rng.uniform(2.0, room[1] - 2.0, 4),
            np.full(4, 1.35),
        )
    )
    sources_by_tx: list[tuple[Source, ...]] = []
    truth: list[np.ndarray] = []
    for tx_id, transmitter in enumerate(transmitters):
        walls = mirror_wall(transmitter, room)
        wall_count = int(rng.integers(3, 5))
        selected = list(rng.choice(4, size=wall_count, replace=False))
        vts = [walls[index] for index in selected]
        # Zero, one or two additional image sources make cardinality unknown
        # and scene-dependent without handing that count to an estimator.
        extra = int(rng.integers(0, 3))
        for _ in range(extra):
            axis = int(rng.integers(0, 2))
            coordinate = rng.uniform(0.25, 0.75) * room[axis]
            virtual = transmitter.copy()
            virtual[axis] = 2.0 * coordinate - virtual[axis]
            # Move the image outside the occupied rectangle when the random
            # internal reflection would otherwise nearly coincide with LoS.
            if 0.0 < virtual[axis] < room[axis]:
                virtual[axis] += (-1.0 if virtual[axis] < room[axis] / 2 else 1.0) * room[axis]
            vts.append(virtual)
        vt_array = np.asarray(vts, dtype=np.float64)
        truth.append(vt_array)
        sources = [Source(transmitter.copy(), tx_id, 0)]
        sources.extend(Source(point.copy(), tx_id, 1 + (index >= wall_count)) for index, point in enumerate(vt_array))
        sources_by_tx.append(tuple(sources))
    return Scene(seed, room, transmitters, tuple(sources_by_tx), tuple(truth))


def grid_positions(scene: Scene, count: int, *, seed: int) -> np.ndarray:
    side = int(math.ceil(math.sqrt(count)))
    xs = np.linspace(0.8, scene.room_xyz_m[0] - 0.8, side)
    ys = np.linspace(0.8, scene.room_xyz_m[1] - 0.8, side)
    points = np.asarray([[x, y, 1.2] for row, y in enumerate(ys) for x in (xs if row % 2 == 0 else xs[::-1])])
    points = points[:count].copy()
    rng = np.random.default_rng(seed)
    points[:, :2] += rng.normal(0.0, 0.06, (len(points), 2))
    points[:, 0] = np.clip(points[:, 0], 0.4, scene.room_xyz_m[0] - 0.4)
    points[:, 1] = np.clip(points[:, 1], 0.4, scene.room_xyz_m[1] - 0.4)
    return points


def query_positions(scene: Scene, count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.column_stack(
        (
            rng.uniform(0.5, scene.room_xyz_m[0] - 0.5, count),
            rng.uniform(0.5, scene.room_xyz_m[1] - 0.5, count),
            np.full(count, 1.2),
        )
    )


def subset_fingerprint(fingerprint: MultimodalFingerprint, indices: np.ndarray) -> MultimodalFingerprint:
    indices = np.asarray(indices, dtype=np.int64)
    return MultimodalFingerprint(
        ranges_m=fingerprint.ranges_m[indices].copy(),
        powers_db=fingerprint.powers_db[indices].copy(),
        aoa_unit=fingerprint.aoa_unit[indices].copy(),
        tx_ids=fingerprint.tx_ids[indices].copy(),
        cir=fingerprint.cir.copy(),
        noise_variance=float(fingerprint.noise_variance),
        range_bin_m=float(fingerprint.range_bin_m),
    )


def remove_direct_observable(
    fingerprint: MultimodalFingerprint,
    position: np.ndarray,
    transmitter: np.ndarray,
    noise_std_m: float,
) -> Acquisition:
    """Remove LoS only through known-TX expected range and an observable gate."""

    if not len(fingerprint):
        return Acquisition(fingerprint, fingerprint, False, None)
    expected = float(np.linalg.norm(position - transmitter))
    residuals = np.abs(fingerprint.ranges_m - expected)
    index = int(np.argmin(residuals))
    gate = max(2.0 * fingerprint.range_bin_m, 3.0 * float(noise_std_m) + fingerprint.range_bin_m)
    removed = bool(residuals[index] <= gate)
    keep = np.delete(np.arange(len(fingerprint)), index) if removed else np.arange(len(fingerprint))
    return Acquisition(fingerprint, subset_fingerprint(fingerprint, keep), removed, float(residuals[index]))


def acquire(
    scene: Scene,
    positions: np.ndarray,
    noise_std_m: float,
    *,
    seed: int,
    maximum_paths: int,
) -> tuple[tuple[Acquisition, ...], ...]:
    """Materialize identical noisy physical acquisitions for every method."""

    by_tx: list[tuple[Acquisition, ...]] = []
    snr_db = max(8.0, 24.0 - 4.0 * float(noise_std_m))
    dropout = min(0.35, 0.04 + 0.055 * float(noise_std_m))
    for tx_id, sources in enumerate(scene.sources_by_tx):
        rows = []
        for frame, position in enumerate(positions):
            fingerprint = simulate_multimodal_fingerprint(
                position,
                sources,
                rng=np.random.default_rng(stable_seed(seed, tx_id, frame)),
                maximum_paths=maximum_paths,
                snr_db=snr_db,
                range_noise_std_m=float(noise_std_m),
                n_bins=384,
                extra_dropout_probability=dropout,
                separate_transmitters=True,
            )
            rows.append(remove_direct_observable(fingerprint, position, scene.transmitters_xyz_m[tx_id], noise_std_m))
        by_tx.append(tuple(rows))
    return tuple(by_tx)


def _ordered(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(array, kind="stable")
    return array[order], order.astype(np.int64)


def _best_ordered_subset(longer: np.ndarray, target: np.ndarray, norm: str) -> np.ndarray:
    n, m = len(longer), len(target)
    if m == 0:
        return np.empty(0, dtype=np.int64)
    power = 1 if norm == "l1" else 2
    cost = np.full((m + 1, n + 1), np.inf)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    cost[0] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choose = cost[i - 1, j - 1] + abs(longer[j - 1] - target[i - 1]) ** power
            if choose <= cost[i, j - 1]:
                cost[i, j], take[i, j] = choose, True
            else:
                cost[i, j] = cost[i, j - 1]
    selected: list[int] = []
    i, j = m, n
    while i:
        if take[i, j]:
            selected.append(j - 1)
            i -= 1
        j -= 1
    return np.asarray(selected[::-1], dtype=np.int64)


def size_unify(values_a: Sequence[float], values_b: Sequence[float], norm: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, ia = _ordered(values_a)
    b, ib = _ordered(values_b)
    if len(a) > len(b):
        keep = _best_ordered_subset(a, b, norm)
        a, ia = a[keep], ia[keep]
    elif len(b) > len(a):
        keep = _best_ordered_subset(b, a, norm)
        b, ib = b[keep], ib[keep]
    return a, b, ia, ib


def padded_windows(values: np.ndarray, half_window_p: int) -> np.ndarray:
    if not len(values):
        return np.empty((0, 2 * half_window_p + 1))
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0)
    tiny = np.finfo(np.float64).eps * scale * 16.0
    lower = values[0] - tiny * np.arange(half_window_p, 0, -1)
    upper = values[-1] + tiny * np.arange(1, half_window_p + 1)
    padded = np.concatenate((lower, values, upper))
    return np.lib.stride_tricks.sliding_window_view(padded, 2 * half_window_p + 1).copy() - values[:, None]


def dissimilarity_matrix(a: np.ndarray, b: np.ndarray, *, half_window_p: int, alpha: float) -> np.ndarray:
    if not len(a) or not len(b):
        return np.empty((len(a), len(b)))
    wa, wb = padded_windows(a, half_window_p), padded_windows(b, half_window_p)
    upper = np.triu_indices(2 * half_window_p + 1, k=1)
    pa = np.abs(wa[:, :, None] - wa[:, None, :])[:, upper[0], upper[1]]
    pb = np.abs(wb[:, :, None] - wb[:, None, :])[:, upper[0], upper[1]]
    pattern = np.sum(np.abs(pa[:, None] - pb[None]), axis=2)
    distance = np.abs(a[:, None] - b[None])
    return alpha * distance + (1.0 - alpha) * pattern


def crosses(a: tuple[float, float], b: tuple[float, float], mode: Literal["printed", "order"]) -> bool:
    if mode == "printed":
        return (a[0] - a[1]) * (b[0] - b[1]) < 0.0
    if mode == "order":
        return (a[0] - b[0]) * (a[1] - b[1]) < 0.0
    raise ValueError(mode)


def filter_current_by_composite(potential: Sequence[Match], mode: Literal["printed", "order"]) -> list[Match]:
    """Source-faithful conflict priority: lower composite dissimilarity first."""

    retained: list[Match] = []
    for candidate in sorted(potential, key=lambda m: (m.dissimilarity, m.index_a, m.index_b)):
        if not any(crosses((candidate.value_a, candidate.value_b), (prior.value_a, prior.value_b), mode) for prior in retained):
            retained.append(candidate)
    return retained


def corrected_pairwise_match(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    total_window_P: int = TOTAL_WINDOW_P,
    alpha: float = ALPHA,
    size_norm: Literal["l1", "l2"] = "l1",
    cross_mode: Literal["printed", "order"] = "order",
) -> list[Match]:
    if total_window_P <= 0 or total_window_P % 2 != 1:
        raise ValueError("total_window_P must be a positive odd integer")
    half_window_p = (total_window_P - 1) // 2
    a, b, ia, ib = size_unify(values_a, values_b, size_norm)
    accepted: list[Match] = []
    iteration = 0
    while len(a) and len(b):
        delta = dissimilarity_matrix(a, b, half_window_p=half_window_p, alpha=alpha)
        row_best, col_best = np.argmin(delta, axis=1), np.argmin(delta, axis=0)
        pairs = [(i, int(j)) for i, j in enumerate(row_best) if int(col_best[int(j)]) == i]
        if not pairs:
            break
        potential = [Match(int(ia[i]), int(ib[j]), float(a[i]), float(b[j]), float(delta[i, j]), iteration) for i, j in pairs]
        # Algorithm 2 removes every potential before either filtering stage.
        keep_a = np.asarray([i not in {x for x, _ in pairs} for i in range(len(a))])
        keep_b = np.asarray([j not in {y for _, y in pairs} for j in range(len(b))])
        a, ia, b, ib = a[keep_a], ia[keep_a], b[keep_b], ib[keep_b]
        against_previous = [
            candidate for candidate in potential
            if not any(crosses((candidate.value_a, candidate.value_b), (prior.value_a, prior.value_b), cross_mode) for prior in accepted)
        ]
        accepted.extend(filter_current_by_composite(against_previous, cross_mode))
        iteration += 1
    return accepted


def trilaterate_2d(anchors: np.ndarray, ranges: np.ndarray) -> np.ndarray | None:
    anchors, ranges = np.asarray(anchors, float), np.asarray(ranges, float)
    if len(anchors) < 3 or len(np.unique(np.round(anchors, 8), axis=0)) < 3:
        return None
    a0, r0 = anchors[0], ranges[0]
    matrix = 2.0 * (anchors[1:] - a0)
    rhs = r0**2 - ranges[1:] ** 2 + np.sum(anchors[1:] ** 2, axis=1) - np.sum(a0**2)
    initial, *_ = np.linalg.lstsq(matrix, rhs, rcond=None)
    fit = least_squares(lambda x: np.linalg.norm(anchors - x[None], axis=1) - ranges, initial, loss="soft_l1", max_nfev=80)
    return fit.x if fit.success and np.all(np.isfinite(fit.x)) else None


def neighbourhoods(positions: np.ndarray, size: int) -> list[np.ndarray]:
    unique: dict[tuple[int, ...], np.ndarray] = {}
    for centre in positions:
        ids = np.argsort(np.linalg.norm(positions - centre[None], axis=1), kind="stable")[: min(size, len(positions))]
        key = tuple(sorted(map(int, ids)))
        unique[key] = np.asarray(key, dtype=np.int64)
    return [unique[key] for key in sorted(unique)]


def graph_groups(nodes: set[tuple[int, int]], edges: dict[tuple[tuple[int, int], tuple[int, int]], float], interpretation: str) -> list[frozenset[tuple[int, int]]]:
    adjacency = {node: set() for node in nodes}
    for (left, right), _ in edges.items():
        adjacency[left].add(right)
        adjacency[right].add(left)
    if interpretation == "star":
        groups = set()
        for node, neighbours in adjacency.items():
            group = frozenset({node, *neighbours})
            rp = [entry[0] for entry in group]
            if len(group) >= MIN_GROUP_RPS and len(set(rp)) == len(rp):
                groups.add(group)
        return sorted(groups, key=lambda g: sorted(g))
    if interpretation != "components":
        raise ValueError(interpretation)
    output, seen = [], set()
    for root in sorted(nodes):
        if root in seen:
            continue
        stack, component = [root], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        chosen = []
        for rp in sorted({node[0] for node in component}):
            options = [node for node in component if node[0] == rp]
            def score(node):
                local = [value for edge, value in edges.items() if node in edge and edge[0] in component and edge[1] in component]
                return (-len(adjacency[node] & component), float(np.mean(local)) if local else np.inf, node[1])
            chosen.append(min(options, key=score))
        if len(chosen) >= MIN_GROUP_RPS:
            output.append(frozenset(chosen))
    return sorted(set(output), key=lambda g: sorted(g))


def complete_link_cluster_records(records: list[dict], radius_m: float) -> list[dict]:
    clusters: list[list[dict]] = []
    for record in sorted(records, key=lambda row: tuple(row["coord_m"])):
        point = np.asarray(record["coord_m"], float)
        for cluster in clusters:
            if max(np.linalg.norm(point - np.asarray(member["coord_m"])) for member in cluster) <= radius_m:
                cluster.append(record)
                break
        else:
            clusters.append([record])
    output = []
    for candidate_id, cluster in enumerate(clusters):
        coords = np.asarray([row["coord_m"] for row in cluster])
        associations: dict[tuple[int, int], dict] = {}
        for row in cluster:
            for association in row.get("associations", []):
                key = (int(association["anchor"]), int(association["path"]))
                prior = associations.get(key)
                if prior is None or float(association.get("dissimilarity", np.inf)) < float(prior.get("dissimilarity", np.inf)):
                    associations[key] = association
        output.append({
            "candidate_id": candidate_id,
            "coord_m": np.median(coords, axis=0).tolist(),
            "raw_cluster_size": len(cluster),
            "support_count": len({key[0] for key in associations}),
            "associations": list(associations.values()),
            "mean_composite_dissimilarity": float(np.mean([row.get("mean_composite_dissimilarity", 0.0) for row in cluster])),
            "existence_probability": None,
        })
    return output


def construct_native(
    anchors_xy: np.ndarray,
    acquisitions: Sequence[Acquisition],
    transmitter_xy: np.ndarray,
    room_xy: np.ndarray,
    *,
    method: Literal["mpurge", "pba"],
    cross_mode: Literal["printed", "order"],
    grouping: Literal["star", "components"],
    neighbourhood_size: int,
) -> list[dict]:
    ranges = [row.indirect.ranges_m for row in acquisitions]
    raw: list[dict] = []
    for neighbourhood in neighbourhoods(anchors_xy, neighbourhood_size):
        nodes: set[tuple[int, int]] = set()
        edges: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for left_local in range(len(neighbourhood)):
            for right_local in range(left_local + 1, len(neighbourhood)):
                left, right = int(neighbourhood[left_local]), int(neighbourhood[right_local])
                matches = corrected_pairwise_match(
                    ranges[left], ranges[right], total_window_P=TOTAL_WINDOW_P, alpha=ALPHA,
                    size_norm="l2" if method == "pba" else "l1", cross_mode=cross_mode,
                )
                for match in matches:
                    a, b = (left, match.index_a), (right, match.index_b)
                    edge = (a, b) if a < b else (b, a)
                    nodes.update((a, b))
                    edges[edge] = min(edges.get(edge, np.inf), match.dissimilarity)
        for group in graph_groups(nodes, edges, grouping):
            ordered = sorted(group)
            fit = trilaterate_2d(
                anchors_xy[np.asarray([node[0] for node in ordered])],
                np.asarray([ranges[node[0]][node[1]] for node in ordered]),
            )
            if fit is None:
                continue
            if np.linalg.norm(fit - transmitter_xy) <= PRIMARY_GATE_M:
                continue
            if not (-room_xy[0] <= fit[0] <= 2.0 * room_xy[0] and -room_xy[1] <= fit[1] <= 2.0 * room_xy[1]):
                continue
            local_dissim = [value for edge, value in edges.items() if edge[0] in group and edge[1] in group]
            associations = [
                {"anchor": int(anchor), "path": int(path), "range_m": float(ranges[anchor][path]),
                 "dissimilarity": float(np.mean(local_dissim)) if local_dissim else 0.0, "source": "pairwise_group"}
                for anchor, path in ordered
            ]
            raw.append({"coord_m": fit.tolist(), "associations": associations,
                        "mean_composite_dissimilarity": float(np.mean(local_dissim)) if local_dissim else 0.0})
    return complete_link_cluster_records(raw, radius_m=0.35)


def support_associations(candidate: np.ndarray, anchors: np.ndarray, acquisitions: Sequence[Acquisition], range_tolerance_m: float, angle_tolerance_deg: float | None = None) -> list[dict]:
    output = []
    for anchor_id, (anchor, acquisition) in enumerate(zip(anchors, acquisitions, strict=True)):
        fp = acquisition.indirect
        if not len(fp):
            continue
        expected_range = float(np.linalg.norm(candidate - anchor))
        range_residual = np.abs(fp.ranges_m - expected_range)
        if angle_tolerance_deg is None:
            index = int(np.argmin(range_residual))
            if range_residual[index] <= range_tolerance_m:
                output.append({"anchor": anchor_id, "path": index, "range_residual_m": float(range_residual[index]), "source": "range_support"})
        else:
            direction = candidate - anchor
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            observed = fp.aoa_unit[:, :2]
            observed /= np.maximum(np.linalg.norm(observed, axis=1, keepdims=True), 1e-9)
            angular = np.rad2deg(np.arccos(np.clip(observed @ direction, -1.0, 1.0)))
            cost = range_residual / max(range_tolerance_m, 1e-6) + angular / max(angle_tolerance_deg, 1e-6)
            index = int(np.argmin(cost))
            if range_residual[index] <= range_tolerance_m and angular[index] <= angle_tolerance_deg:
                output.append({"anchor": anchor_id, "path": index, "range_residual_m": float(range_residual[index]),
                               "angle_residual_deg": float(angular[index]), "observed_power_db": float(fp.powers_db[index]),
                               "source": "multimodal_support"})
    return output


def dynamic_beta_sets(candidates: list[dict], step: float) -> list[dict]:
    if not candidates:
        return [{"beta": 0.0, "threshold": 0.0, "kept_candidate_ids": []}]
    support = np.asarray([row["support_count"] for row in candidates], float)
    mean = float(np.mean(support))
    if mean <= 0.0:
        return [{"beta": 0.0, "threshold": 0.0, "kept_candidate_ids": [row["candidate_id"] for row in candidates]},
                {"beta": step, "threshold": 0.0, "kept_candidate_ids": []}]
    maximum_beta = float(np.max(support) / mean)
    count = int(math.floor(maximum_beta / step + 1e-12)) + 2
    rows = []
    for index in range(count + 1):
        beta = round(index * step, 10)
        threshold = beta * mean
        kept = [row["candidate_id"] for row in candidates if row["support_count"] >= threshold - 1e-12]
        rows.append({"beta": beta, "threshold": threshold, "kept_candidate_ids": kept})
        if index > 0 and not kept:
            break
    if rows[-1]["kept_candidate_ids"]:
        beta = round(rows[-1]["beta"] + step, 10)
        rows.append({"beta": beta, "threshold": beta * mean, "kept_candidate_ids": []})
    return rows


def cluster_points(points: np.ndarray, radius_m: float) -> list[np.ndarray]:
    if not len(points):
        return []
    clusters: list[list[int]] = []
    for index in np.lexsort((points[:, 1], points[:, 0])):
        for cluster in clusters:
            if max(np.linalg.norm(points[index] - points[member]) for member in cluster) <= radius_m:
                cluster.append(int(index))
                break
        else:
            clusters.append([int(index)])
    return [np.asarray(cluster, dtype=np.int64) for cluster in clusters]


def inverse_aoa_proposals(anchors: np.ndarray, acquisitions: Sequence[Acquisition]) -> tuple[np.ndarray, list[dict]]:
    points, audit = [], []
    for anchor_id, (anchor, acquisition) in enumerate(zip(anchors, acquisitions, strict=True)):
        fp = acquisition.indirect
        for path in range(len(fp)):
            unit = fp.aoa_unit[path, :2].astype(float)
            unit /= max(float(np.linalg.norm(unit)), 1e-9)
            points.append(anchor + fp.ranges_m[path] * unit)
            audit.append({"anchor": anchor_id, "path": path, "range_m": float(fp.ranges_m[path]),
                          "power_db": float(fp.powers_db[path]), "aoa_xy": unit.tolist(), "source": "range_aoa_inverse"})
    return np.asarray(points, dtype=float).reshape(-1, 2), audit


def aoa_inverse_consensus(anchors: np.ndarray, acquisitions: Sequence[Acquisition], noise_std_m: float, *, survival: bool) -> list[dict]:
    points, audit = inverse_aoa_proposals(anchors, acquisitions)
    radius = max(0.40, 1.20 * float(noise_std_m) + 0.20)
    output = []
    for indices in cluster_points(points, radius):
        associations = [audit[int(index)] for index in indices]
        unique_anchors = len({row["anchor"] for row in associations})
        if unique_anchors < 3:
            continue
        probability = float(beta_distribution.sf(0.15, 1.0 + unique_anchors, 1.0 + len(anchors) - unique_anchors))
        if survival and probability < 0.90:
            continue
        power = np.asarray([row["power_db"] for row in associations])
        weight = np.exp(np.clip((power - np.max(power)) / 10.0, -20.0, 0.0))
        weight /= np.sum(weight)
        coordinate = np.sum(points[indices] * weight[:, None], axis=0)
        output.append({"candidate_id": len(output), "coord_m": coordinate.tolist(), "raw_cluster_size": len(indices),
                       "support_count": unique_anchors, "associations": associations,
                       "mean_composite_dissimilarity": 0.0, "existence_probability": probability})
    return output


def two_sided_vt_registration(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    room_xy: np.ndarray,
    noise_std_m: float,
) -> list[dict]:
    """Mutual survey-to-VT / VT-to-survey registration without latent IDs.

    The initial graph contains only pairwise mutual-nearest matches for each
    pair of survey locations.  Refinement then retains an observation iff the
    VT is its best candidate *and* it is the VT's best token at that survey
    location.  This is a genuine round-trip constraint, not an alias of the
    one-way inverse-consensus clusterer.
    """

    points, records = inverse_aoa_proposals(anchors, acquisitions)
    if len(points) < 3:
        return []
    anchor_id = np.asarray([row["anchor"] for row in records], dtype=int)
    range_gate = max(0.25, 0.80 * float(noise_std_m) + 0.15)
    endpoint_gate = max(0.50, 1.30 * float(noise_std_m) + 0.25)
    angle_gate_deg = 18.0
    angle_gate = math.radians(angle_gate_deg)

    def token_candidate_cost(token: int, candidate: np.ndarray) -> float:
        anchor = anchors[anchor_id[token]]
        delta = candidate - anchor
        predicted_range = max(float(np.linalg.norm(delta)), 1.0e-9)
        direction = delta / predicted_range
        observed = np.asarray(records[token]["aoa_xy"], dtype=float)
        angular = math.acos(float(np.clip(np.dot(direction, observed), -1.0, 1.0)))
        return (abs(predicted_range - float(records[token]["range_m"])) / range_gate
                + angular / angle_gate)

    adjacency = [set() for _ in records]
    unique_anchors = sorted(set(anchor_id.tolist()))
    for left_pos, left_anchor in enumerate(unique_anchors):
        left = np.flatnonzero(anchor_id == left_anchor)
        for right_anchor in unique_anchors[left_pos + 1:]:
            right = np.flatnonzero(anchor_id == right_anchor)
            if not len(left) or not len(right):
                continue
            cost = np.empty((len(left), len(right)), dtype=float)
            for i, token_i in enumerate(left):
                for j, token_j in enumerate(right):
                    midpoint = 0.5 * (points[token_i] + points[token_j])
                    endpoint = float(np.linalg.norm(points[token_i] - points[token_j])) / endpoint_gate
                    cost[i, j] = endpoint + token_candidate_cost(int(token_i), midpoint) + token_candidate_cost(int(token_j), midpoint)
            row_best = np.argmin(cost, axis=1)
            column_best = np.argmin(cost, axis=0)
            for i, j in enumerate(row_best):
                if column_best[j] == i and cost[i, j] <= 3.0:
                    a, b = int(left[i]), int(right[j])
                    adjacency[a].add(b)
                    adjacency[b].add(a)

    components, seen = [], set()
    for root in range(len(records)):
        if root in seen or not adjacency[root]:
            continue
        stack, component = [root], set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(adjacency[node] - component)
        seen |= component
        if len({anchor_id[index] for index in component}) >= 3:
            components.append(np.asarray(sorted(component), dtype=int))
    if not components:
        return []

    candidates = [np.median(points[component], axis=0) for component in components]
    # Merge graph fragments only when all initial representatives agree.
    merged = cluster_points(np.asarray(candidates), max(0.30, 0.55 * float(noise_std_m) + 0.20))
    candidates = [np.median(np.asarray(candidates)[indices], axis=0) for indices in merged]

    final_members: list[list[int]] = []
    for _ in range(4):
        if not candidates:
            break
        cost = np.asarray([[token_candidate_cost(token, candidate) for candidate in candidates]
                           for token in range(len(records))], dtype=float)
        observation_best = np.argmin(cost, axis=1)
        members: list[list[int]] = [[] for _ in candidates]
        for candidate_id in range(len(candidates)):
            for survey in unique_anchors:
                local = np.flatnonzero(anchor_id == survey)
                if not len(local):
                    continue
                token = int(local[np.argmin(cost[local, candidate_id])])
                if observation_best[token] == candidate_id and cost[token, candidate_id] <= 2.0:
                    members[candidate_id].append(token)
        refined, retained = [], []
        for candidate, token_ids in zip(candidates, members, strict=True):
            if len({anchor_id[token] for token in token_ids}) < 3:
                continue

            def residual(value: np.ndarray) -> np.ndarray:
                rows = []
                for token in token_ids:
                    delta = value - anchors[anchor_id[token]]
                    predicted_range = max(float(np.linalg.norm(delta)), 1.0e-9)
                    predicted_direction = delta / predicted_range
                    observed_direction = np.asarray(records[token]["aoa_xy"], dtype=float)
                    rows.extend([
                        (predicted_range - float(records[token]["range_m"])) / range_gate,
                        (predicted_direction[0] * observed_direction[1]
                         - predicted_direction[1] * observed_direction[0]) / math.sin(angle_gate),
                    ])
                return np.asarray(rows, dtype=float)

            lower = np.asarray([-room_xy[0], -room_xy[1]], dtype=float)
            upper = np.asarray([2.0 * room_xy[0], 2.0 * room_xy[1]], dtype=float)
            feasible_start = np.clip(candidate, lower + 1.0e-9, upper - 1.0e-9)
            fit = least_squares(
                residual, feasible_start,
                bounds=(lower, upper),
                loss="soft_l1", f_scale=1.0, max_nfev=80,
            )
            refined.append(fit.x)
            retained.append(token_ids)
        candidates, final_members = refined, retained

    output = []
    for candidate, token_ids in zip(candidates, final_members, strict=True):
        associations = []
        costs = []
        for token in token_ids:
            value = token_candidate_cost(token, candidate)
            costs.append(value)
            associations.append({**records[token], "roundtrip_cost": value,
                                 "source": "mutual_survey_vt_roundtrip"})
        output.append({"candidate_id": len(output), "coord_m": np.asarray(candidate).tolist(),
                       "raw_cluster_size": len(token_ids),
                       "support_count": len({anchor_id[token] for token in token_ids}),
                       "associations": associations,
                       "mean_composite_dissimilarity": float(np.mean(costs)),
                       "existence_probability": None})
    return output


def cycle_consistent_candidates(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    room_xy: np.ndarray,
    noise_std_m: float,
    *, seed: int,
    draws: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    available = np.asarray([index for index, row in enumerate(acquisitions) if len(row.indirect) > 0], dtype=int)
    raw = []
    if len(available) < 3:
        return []
    for _ in range(draws):
        ids = rng.choice(available, 3, replace=False)
        paths = [int(rng.integers(len(acquisitions[int(index)].indirect))) for index in ids]
        fit = trilaterate_2d(anchors[ids], np.asarray([acquisitions[int(index)].indirect.ranges_m[path] for index, path in zip(ids, paths)]))
        if fit is None or not (-room_xy[0] <= fit[0] <= 2 * room_xy[0] and -room_xy[1] <= fit[1] <= 2 * room_xy[1]):
            continue
        associations = support_associations(fit, anchors, acquisitions, max(0.18, 0.8 * noise_std_m + 0.12))
        if len(associations) >= max(3, int(math.ceil(0.12 * len(anchors)))):
            raw.append({"coord_m": fit.tolist(), "associations": associations, "mean_composite_dissimilarity": 0.0})
    clustered = complete_link_cluster_records(raw, radius_m=max(0.40, 0.75 * noise_std_m + 0.25))
    return [row for row in clustered if row["support_count"] >= max(3, int(math.ceil(0.15 * len(anchors))))]


def rfs_lifecycle(anchors: np.ndarray, acquisitions: Sequence[Acquisition], noise_std_m: float) -> list[dict]:
    tracks: list[dict] = []
    gate = max(0.50, 1.25 * noise_std_m + 0.25)
    for anchor_id, (anchor, acquisition) in enumerate(zip(anchors, acquisitions, strict=True)):
        fp = acquisition.indirect
        detections = []
        for path in range(len(fp)):
            direction = fp.aoa_unit[path, :2].astype(float)
            direction /= max(float(np.linalg.norm(direction)), 1e-9)
            detections.append((anchor + fp.ranges_m[path] * direction, path, float(fp.powers_db[path])))
        touched = set()
        for point, path, power in sorted(detections, key=lambda row: -row[2]):
            if tracks:
                distance = np.asarray([np.linalg.norm(point - np.asarray(track["coord_m"])) for track in tracks])
                index = int(np.argmin(distance))
            else:
                index, distance = -1, np.asarray([])
            association = {"anchor": anchor_id, "path": path, "power_db": power, "source": "bernoulli_rfs_detection"}
            if index >= 0 and distance[index] <= gate and index not in touched:
                track = tracks[index]
                weight = min(0.35, 1.0 / (track["updates"] + 1.0))
                track["coord_m"] = ((1.0 - weight) * np.asarray(track["coord_m"]) + weight * point).tolist()
                track["existence_probability"] += (1.0 - track["existence_probability"]) * 0.35
                track["updates"] += 1
                track["associations"].append(association)
                touched.add(index)
            else:
                tracks.append({"coord_m": point.tolist(), "existence_probability": 0.35, "updates": 1, "associations": [association]})
                touched.add(len(tracks) - 1)
        for index, track in enumerate(tracks):
            if index not in touched:
                track["existence_probability"] *= 0.94
    output = []
    for track in tracks:
        support = len({row["anchor"] for row in track["associations"]})
        if track["existence_probability"] >= 0.55 and support >= 3:
            output.append({"candidate_id": len(output), "coord_m": track["coord_m"], "raw_cluster_size": track["updates"],
                           "support_count": support, "associations": track["associations"],
                           "mean_composite_dissimilarity": 0.0,
                           "existence_probability": float(track["existence_probability"])})
    return output


def _ba_observable_tokens(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    selected_anchors: np.ndarray,
) -> dict[str, np.ndarray]:
    selected = set(np.asarray(selected_anchors, dtype=int).tolist())
    rows = []
    for anchor_id, (anchor, acquisition) in enumerate(zip(anchors, acquisitions, strict=True)):
        if anchor_id not in selected:
            continue
        fp = acquisition.indirect
        for path in range(len(fp)):
            direction = fp.aoa_unit[path, :2].astype(float)
            direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
            rows.append((anchor_id, path, anchor.copy(), float(fp.ranges_m[path]),
                         direction, float(fp.powers_db[path])))
    if not rows:
        return {"anchor_id": np.empty(0, int), "path": np.empty(0, int),
                "anchors": np.empty((0, 2)), "ranges": np.empty(0),
                "directions": np.empty((0, 2)), "powers": np.empty(0)}
    return {"anchor_id": np.asarray([row[0] for row in rows], int),
            "path": np.asarray([row[1] for row in rows], int),
            "anchors": np.asarray([row[2] for row in rows], float),
            "ranges": np.asarray([row[3] for row in rows], float),
            "directions": np.asarray([row[4] for row in rows], float),
            "powers": np.asarray([row[5] for row in rows], float)}


def _ba_initial_points(tokens: dict[str, np.ndarray], k: int, bounds: np.ndarray, seed: int) -> np.ndarray:
    endpoints = tokens["anchors"] + tokens["ranges"][:, None] * tokens["directions"]
    rng = np.random.default_rng(seed)
    if not len(endpoints):
        return rng.uniform(bounds[0], bounds[1], size=(k, 2))
    power = tokens["powers"]
    chosen = [int(np.argmax(power + rng.normal(0.0, 1.0e-8, len(power))))]
    while len(chosen) < k:
        distance = np.min(np.linalg.norm(endpoints[:, None] - endpoints[np.asarray(chosen)][None], axis=2), axis=1)
        chosen.append(int(np.argmax(distance + rng.normal(0.0, 1.0e-8, len(distance)))))
    return np.clip(endpoints[np.asarray(chosen)], bounds[0] + 1.0e-4, bounds[1] - 1.0e-4)


def _fit_bundle_k(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    selected_anchors: np.ndarray,
    room_xy: np.ndarray,
    *, k: int, steps: int, seed: int, device: torch.device,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fixed-survey-pose multipath BA jointly fitting VTs and soft associations."""

    tokens = _ba_observable_tokens(anchors, acquisitions, selected_anchors)
    if not len(tokens["ranges"]):
        return np.empty((0, 2)), np.empty(0), float("inf")
    bounds_np = np.asarray([[-room_xy[0], -room_xy[1]], [2.0 * room_xy[0], 2.0 * room_xy[1]]], float)
    initial = _ba_initial_points(tokens, k, bounds_np, stable_seed(seed, "ba-init", k))
    fraction = np.clip((initial - bounds_np[0]) / (bounds_np[1] - bounds_np[0]), 1.0e-5, 1.0 - 1.0e-5)
    raw = nn.Parameter(torch.as_tensor(np.log(fraction / (1.0 - fraction)), dtype=torch.float32, device=device))
    mixture_logits = nn.Parameter(torch.zeros(k, dtype=torch.float32, device=device))
    anchor_t = torch.as_tensor(tokens["anchors"], dtype=torch.float32, device=device)
    range_t = torch.as_tensor(tokens["ranges"], dtype=torch.float32, device=device)
    direction_t = torch.as_tensor(tokens["directions"], dtype=torch.float32, device=device)
    power_t = torch.as_tensor(tokens["powers"], dtype=torch.float32, device=device)
    bounds = torch.as_tensor(bounds_np, dtype=torch.float32, device=device)
    confidence = torch.sigmoid((power_t - torch.median(power_t)) / 6.0) + 0.25
    optimizer = torch.optim.Adam([raw, mixture_logits], lr=0.045)
    for _ in range(steps):
        points = bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)
        delta = points[None] - anchor_t[:, None]
        predicted_range = torch.linalg.vector_norm(delta, dim=2)
        predicted_direction = delta / torch.clamp(predicted_range[..., None], min=1.0e-6)
        range_residual = torch.nn.functional.smooth_l1_loss(
            predicted_range, range_t[:, None].expand_as(predicted_range), reduction="none", beta=0.35
        ) / 0.35
        cosine = torch.einsum("nc,nkc->nk", direction_t, predicted_direction)
        angle = torch.acos(torch.clamp(cosine, -1.0 + 1.0e-6, 1.0 - 1.0e-6))
        cost = range_residual + angle / math.radians(10.0)
        log_mix = torch.log_softmax(mixture_logits, dim=0)
        observation_nll = -torch.logsumexp(log_mix[None] - cost, dim=1)
        data_loss = torch.sum(confidence * observation_nll) / torch.sum(confidence)
        coverage = torch.mean(torch.min(cost, dim=0).values)
        if k > 1:
            pairwise = torch.cdist(points, points)
            repulsion = torch.sum(torch.exp(-pairwise.square() / (2.0 * 0.45**2)) - torch.eye(k, device=device)) / (k * (k - 1))
        else:
            repulsion = torch.zeros((), device=device)
        loss = data_loss + 0.12 * coverage + 0.01 * repulsion
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    points_np = (bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)).detach().cpu().numpy()
    mixture_np = torch.softmax(mixture_logits, dim=0).detach().cpu().numpy()
    return points_np, mixture_np, float(loss.detach().cpu())


def _bundle_validation_score(points: np.ndarray, mixture: np.ndarray, tokens: dict[str, np.ndarray]) -> float:
    if not len(points) or not len(tokens["ranges"]):
        return float("inf")
    delta = points[None] - tokens["anchors"][:, None]
    predicted_range = np.linalg.norm(delta, axis=2)
    predicted_direction = delta / np.maximum(predicted_range[..., None], 1.0e-9)
    range_cost = np.abs(predicted_range - tokens["ranges"][:, None]) / 0.35
    angle = np.arccos(np.clip(np.einsum("nc,nkc->nk", tokens["directions"], predicted_direction), -1.0, 1.0))
    cost = range_cost + angle / math.radians(10.0)
    log_mix = np.log(np.maximum(mixture, 1.0e-12))[None]
    values = -(np.logaddexp.reduce(log_mix - cost, axis=1))
    confidence = 1.0 / (1.0 + np.exp(-(tokens["powers"] - np.median(tokens["powers"])) / 6.0)) + 0.25
    return float(np.sum(confidence * values) / np.sum(confidence))


def multipath_bundle_adjustment_noncheating(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    room_xy: np.ndarray,
    noise_std_m: float,
    *, kmax: int, steps: int, seed: int, device: torch.device,
) -> tuple[list[dict], dict]:
    """Model-selected BA using only known survey poses and receiver observables."""

    fit = np.asarray([index for index in range(len(anchors)) if index % 4 != 0], dtype=int)
    validation = np.asarray([index for index in range(len(anchors)) if index % 4 == 0], dtype=int)
    validation_tokens = _ba_observable_tokens(anchors, acquisitions, validation)
    selection = []
    for k in range(1, kmax + 1):
        points, mixture, fit_loss = _fit_bundle_k(
            anchors, acquisitions, fit, room_xy, k=k, steps=steps,
            seed=stable_seed(seed, "ba-select", k), device=device,
        )
        heldout = _bundle_validation_score(points, mixture, validation_tokens)
        observations = max(len(validation_tokens["ranges"]), 2)
        parameters = 3 * k - 1
        bic = heldout + 0.5 * parameters * math.log(observations) / observations
        selection.append({"k": k, "fit_loss": fit_loss,
                          "validation_negative_loglikelihood": heldout, "bic_per_observation": bic})
    chosen = min(selection, key=lambda row: (row["bic_per_observation"], row["k"]))["k"]
    points, mixture, final_loss = _fit_bundle_k(
        anchors, acquisitions, np.arange(len(anchors)), room_xy, k=chosen,
        steps=max(steps, int(1.25 * steps)), seed=stable_seed(seed, "ba-refit"), device=device,
    )
    output = []
    for slot, (point, existence) in enumerate(zip(points, mixture, strict=True)):
        associations = support_associations(
            point, anchors, acquisitions,
            max(0.30, 0.90 * float(noise_std_m) + 0.18), 20.0,
        )
        support = len({row["anchor"] for row in associations})
        if support < 3:
            continue
        output.append({"candidate_id": len(output), "bundle_slot": slot,
                       "coord_m": point.tolist(), "raw_cluster_size": 1,
                       "support_count": support, "associations": associations,
                       "mean_composite_dissimilarity": final_loss,
                       "existence_probability": float(existence)})
    return output, {"selected_k": chosen, "selection": selection, "final_loss": final_loss,
                    "fixed_survey_poses": True, "joint_soft_associations": True,
                    "validation_rule": "anchor_index_mod_4_equals_0",
                    "forbidden_inputs": ["query_pose", "odometry", "true_z", "ray_id", "VT_id", "dense_simulation"]}


def padded_observations(acquisitions: Sequence[Acquisition], multimodal: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    maximum = max((len(row.indirect) for row in acquisitions), default=1)
    ranges = np.zeros((len(acquisitions), maximum), dtype=np.float32)
    directions = np.zeros((len(acquisitions), maximum, 2), dtype=np.float32)
    mask = np.zeros((len(acquisitions), maximum), dtype=bool)
    for index, row in enumerate(acquisitions):
        count = len(row.indirect)
        if not count:
            continue
        ranges[index, :count] = row.indirect.ranges_m
        direction = row.indirect.aoa_unit[:, :2]
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-9)
        directions[index, :count] = direction
        mask[index, :count] = True
    return ranges, directions if multimodal else np.zeros_like(directions), mask


def soft_set_loss(
    raw: torch.Tensor,
    anchors: torch.Tensor,
    ranges: torch.Tensor,
    directions: torch.Tensor,
    mask: torch.Tensor,
    bounds: torch.Tensor,
    *, multimodal: bool,
) -> torch.Tensor:
    low, high = bounds[0], bounds[1]
    points = low + (high - low) * torch.sigmoid(raw)
    delta = points[None] - anchors[:, None]
    predicted_range = torch.linalg.vector_norm(delta, dim=2)
    observed = ranges[:, :, None]
    cost = torch.abs(observed - predicted_range[:, None]) / 0.35
    if multimodal:
        predicted_direction = delta / torch.clamp(predicted_range[:, :, None], min=1e-6)
        cosine = torch.einsum("amc,akc->amk", directions, predicted_direction)
        angle = torch.acos(torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6))
        cost = cost + angle / math.radians(8.0)
    tau = 0.15
    observed_term = -tau * torch.logsumexp(-cost / tau, dim=2)
    observed_loss = torch.sum(observed_term * mask) / torch.clamp(mask.sum(), min=1)
    masked = cost.masked_fill(~mask[:, :, None], torch.inf)
    candidate_min = torch.min(masked, dim=1).values
    finite = torch.isfinite(candidate_min)
    candidate_loss = torch.sum(torch.where(finite, candidate_min, torch.zeros_like(candidate_min))) / torch.clamp(finite.sum(), min=1)
    return observed_loss + 0.30 * candidate_loss


def optimize_diffassign_k(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    room_xy: np.ndarray,
    *, k: int,
    multimodal: bool,
    steps: int,
    restarts: int,
    seed: int,
    device: torch.device,
    fit_indices: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    ranges_np, directions_np, mask_np = padded_observations(acquisitions, multimodal)
    indices = np.arange(len(anchors)) if fit_indices is None else np.asarray(fit_indices)
    anchor = torch.as_tensor(anchors[indices], dtype=torch.float32, device=device)
    ranges = torch.as_tensor(ranges_np[indices], device=device)
    directions = torch.as_tensor(directions_np[indices], device=device)
    mask = torch.as_tensor(mask_np[indices], device=device)
    bounds = torch.as_tensor([[-room_xy[0], -room_xy[1]], [2 * room_xy[0], 2 * room_xy[1]]], dtype=torch.float32, device=device)
    best = None
    for restart in range(restarts):
        generator = torch.Generator(device=device).manual_seed(stable_seed(seed, k, restart, multimodal))
        raw = nn.Parameter(torch.randn((k, 2), generator=generator, device=device))
        optimizer = torch.optim.Adam([raw], lr=0.06)
        for _ in range(steps):
            loss = soft_set_loss(raw, anchor, ranges, directions, mask, bounds, multimodal=multimodal)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        value = float(loss.detach().cpu())
        points = (bounds[0] + (bounds[1] - bounds[0]) * torch.sigmoid(raw)).detach().cpu().numpy()
        if best is None or value < best[0]:
            best = (value, points)
    assert best is not None
    return best[1], best[0]


def observable_set_score(points: np.ndarray, anchors: np.ndarray, acquisitions: Sequence[Acquisition], multimodal: bool) -> float:
    if not len(points):
        return float("inf")
    values = []
    for anchor, acquisition in zip(anchors, acquisitions, strict=True):
        fp = acquisition.indirect
        if not len(fp):
            continue
        delta = points - anchor[None]
        predicted = np.linalg.norm(delta, axis=1)
        cost = np.abs(fp.ranges_m[:, None] - predicted[None]) / 0.35
        if multimodal:
            direction = delta / np.maximum(predicted[:, None], 1e-9)
            observed = fp.aoa_unit[:, :2]
            observed /= np.maximum(np.linalg.norm(observed, axis=1, keepdims=True), 1e-9)
            angle = np.arccos(np.clip(observed @ direction.T, -1.0, 1.0))
            cost += angle / math.radians(8.0)
        values.append(float(np.mean(np.min(cost, axis=1)) + 0.30 * np.mean(np.min(cost, axis=0))))
    return float(np.mean(values)) if values else float("inf")


def diffassign_model_selected(
    anchors: np.ndarray,
    acquisitions: Sequence[Acquisition],
    room_xy: np.ndarray,
    *, multimodal: bool,
    kmax: int,
    steps: int,
    restarts: int,
    seed: int,
    device: torch.device,
    oracle_k: int | None = None,
) -> tuple[list[dict], dict]:
    fit = np.asarray([index for index in range(len(anchors)) if index % 4 != 0], dtype=int)
    validation = np.asarray([index for index in range(len(anchors)) if index % 4 == 0], dtype=int)
    candidates = [int(oracle_k)] if oracle_k is not None else list(range(1, kmax + 1))
    selection = []
    for k in candidates:
        points, fit_loss = optimize_diffassign_k(anchors, acquisitions, room_xy, k=k, multimodal=multimodal,
                                                 steps=steps, restarts=restarts, seed=stable_seed(seed, "select"),
                                                 device=device, fit_indices=fit)
        held = [acquisitions[int(index)] for index in validation]
        validation_score = observable_set_score(points, anchors[validation], held, multimodal)
        criterion = validation_score + 0.055 * k * math.log(max(len(fit), 2)) / max(len(fit), 1)
        selection.append({"k": k, "fit_loss": fit_loss, "validation_reconstruction": validation_score,
                          "mdl_criterion": criterion})
    chosen = min(selection, key=lambda row: (row["mdl_criterion"], row["k"]))["k"]
    points, final_loss = optimize_diffassign_k(anchors, acquisitions, room_xy, k=chosen, multimodal=multimodal,
                                               steps=max(steps, int(steps * 1.25)), restarts=restarts,
                                               seed=stable_seed(seed, "refit"), device=device)
    output = []
    for point in points:
        associations = support_associations(point, anchors, acquisitions, 0.65, 14.0 if multimodal else None)
        output.append({"candidate_id": len(output), "coord_m": point.tolist(), "raw_cluster_size": 1,
                       "support_count": len({row["anchor"] for row in associations}), "associations": associations,
                       "mean_composite_dissimilarity": final_loss, "existence_probability": 1.0})
    return output, {"selected_k": chosen, "oracle_k": oracle_k, "selection": selection, "final_loss": final_loss}


class SelfSupervisedVTSet(nn.Module):
    def __init__(self, mode: Literal["deepsets", "attention"], kmax: int, hidden: int = 64):
        super().__init__()
        self.mode, self.kmax = mode, kmax
        self.token = nn.Sequential(nn.Linear(8, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        if mode == "attention":
            layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 2, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.context = nn.TransformerEncoder(layer, 2)
        else:
            self.context = None
        self.head = nn.Sequential(nn.Linear(2 * hidden + 1, 128), nn.GELU(), nn.Linear(128, kmax * 3))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.token(tokens)
        if self.context is not None:
            hidden = self.context(hidden, src_key_padding_mask=~mask)
        numeric = mask[..., None].to(hidden.dtype)
        mean = torch.sum(hidden * numeric, dim=1) / torch.clamp(numeric.sum(dim=1), min=1.0)
        maximum = hidden.masked_fill(~mask[..., None], -torch.inf).amax(dim=1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        count = torch.log1p(mask.sum(dim=1, keepdim=True).to(hidden.dtype))
        raw = self.head(torch.cat((mean, maximum, count), dim=1)).reshape(len(tokens), self.kmax, 3)
        return raw[..., :2], raw[..., 2]


def pack_scene_tokens(anchors: np.ndarray, by_tx: Sequence[Sequence[Acquisition]], room_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    diagonal = float(np.linalg.norm(room_xy))
    for acquisitions in by_tx:
        tokens = []
        for anchor_id, (anchor, acquisition) in enumerate(zip(anchors, acquisitions, strict=True)):
            fp = acquisition.indirect
            for path in range(len(fp)):
                direction = fp.aoa_unit[path, :2]
                direction /= max(float(np.linalg.norm(direction)), 1e-9)
                tokens.append([anchor[0] / room_xy[0], anchor[1] / room_xy[1], fp.ranges_m[path] / diagonal,
                               np.clip((fp.powers_db[path] + 45.0) / 20.0, -4.0, 4.0), direction[0], direction[1],
                               anchor_id / max(len(anchors) - 1, 1), 1.0])
        rows.append(np.asarray(tokens, dtype=np.float32).reshape(-1, 8))
    maximum = max(max(map(len, rows)), 1)
    values = np.zeros((len(rows), maximum, 8), dtype=np.float32)
    mask = np.zeros((len(rows), maximum), dtype=bool)
    for index, tokens in enumerate(rows):
        values[index, :len(tokens)] = tokens
        mask[index, :len(tokens)] = True
    return values, mask


def neural_reconstruction_loss(
    raw_xy: torch.Tensor,
    existence_logits: torch.Tensor,
    anchors: torch.Tensor,
    range_tensor: torch.Tensor,
    direction_tensor: torch.Tensor,
    observation_mask: torch.Tensor,
    room_xy: torch.Tensor,
    pseudo_count: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    low, high = -room_xy, 2.0 * room_xy
    points = low[None, None] + (high - low)[None, None] * torch.sigmoid(raw_xy)
    probability = torch.sigmoid(existence_logits)
    delta = points[:, None] - anchors[None, :, None]
    predicted_range = torch.linalg.vector_norm(delta, dim=3)
    predicted_direction = delta / torch.clamp(predicted_range[..., None], min=1e-6)
    range_cost = torch.abs(range_tensor[:, :, :, None] - predicted_range[:, :, None]) / 0.35
    cosine = torch.einsum("tamc,takc->tamk", direction_tensor, predicted_direction)
    angle = torch.acos(torch.clamp(cosine, -1.0 + 1e-6, 1.0 - 1e-6))
    cost = range_cost + angle / math.radians(8.0)
    log_weight = torch.log_softmax(existence_logits, dim=1)[:, None, None]
    observed_term = -0.15 * torch.logsumexp(log_weight - cost / 0.15, dim=3)
    observed_loss = torch.sum(observed_term * observation_mask) / torch.clamp(observation_mask.sum(), min=1)
    masked = cost.masked_fill(~observation_mask[..., None], torch.inf)
    candidate_min = torch.min(masked, dim=2).values
    finite = torch.isfinite(candidate_min)
    weighted_candidate = torch.where(finite, candidate_min, torch.zeros_like(candidate_min)) * probability[:, None]
    candidate_loss = weighted_candidate.sum() / torch.clamp((finite * probability[:, None]).sum(), min=1.0)
    count_loss = torch.mean((probability.sum(dim=1) - pseudo_count) ** 2)
    sharpness = torch.mean(probability * (1.0 - probability))
    loss = observed_loss + 0.25 * candidate_loss + 0.08 * count_loss + 0.01 * sharpness
    return loss, points, probability


def selfsupervised_set_estimator(
    anchors: np.ndarray,
    by_tx: Sequence[Sequence[Acquisition]],
    room_xy: np.ndarray,
    *, mode: Literal["deepsets", "attention"],
    kmax: int,
    epochs: int,
    seed: int,
    device: torch.device,
    checkpoint: Path,
) -> tuple[dict[int, list[dict]], dict]:
    seed_everything(seed)
    token_np, token_mask_np = pack_scene_tokens(anchors, by_tx, room_xy)
    max_paths = max(max((len(row.indirect) for rows in by_tx for row in rows), default=1), 1)
    ranges = np.zeros((len(by_tx), len(anchors), max_paths), dtype=np.float32)
    directions = np.zeros((len(by_tx), len(anchors), max_paths, 2), dtype=np.float32)
    observed_mask = np.zeros((len(by_tx), len(anchors), max_paths), dtype=bool)
    cardinality = np.zeros((len(by_tx), len(anchors)), dtype=float)
    for tx, rows in enumerate(by_tx):
        for anchor_id, row in enumerate(rows):
            count = len(row.indirect)
            cardinality[tx, anchor_id] = count
            if not count:
                continue
            ranges[tx, anchor_id, :count] = row.indirect.ranges_m
            unit = row.indirect.aoa_unit[:, :2]
            unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-9)
            directions[tx, anchor_id, :count] = unit
            observed_mask[tx, anchor_id, :count] = True
    pseudo_count = np.clip(np.quantile(cardinality, 0.90, axis=1), 1, kmax).astype(np.float32)
    model = SelfSupervisedVTSet(mode, kmax).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    tokens = torch.as_tensor(token_np, device=device)
    token_mask = torch.as_tensor(token_mask_np, device=device)
    anchor_t = torch.as_tensor(anchors, dtype=torch.float32, device=device)
    range_t = torch.as_tensor(ranges, device=device)
    direction_t = torch.as_tensor(directions, device=device)
    observation_t = torch.as_tensor(observed_mask, device=device)
    room_t = torch.as_tensor(room_xy, dtype=torch.float32, device=device)
    pseudo_t = torch.as_tensor(pseudo_count, device=device)
    history = []
    model.train()
    for epoch in range(epochs):
        raw, logits = model(tokens, token_mask)
        loss, _, _ = neural_reconstruction_loss(raw, logits, anchor_t, range_t, direction_t, observation_t, room_t, pseudo_t)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if epoch % max(1, epochs // 10) == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    model.eval()
    with torch.no_grad():
        raw, logits = model(tokens, token_mask)
        _, points, probability = neural_reconstruction_loss(raw, logits, anchor_t, range_t, direction_t, observation_t, room_t, pseudo_t)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema": SCHEMA, "mode": mode, "kmax": kmax, "state_dict": model.state_dict(),
                "pseudo_count_from_observed_cardinality": pseudo_count.tolist()}, checkpoint)
    points_np, probability_np = points.cpu().numpy(), probability.cpu().numpy()
    output: dict[int, list[dict]] = {}
    for tx in range(len(by_tx)):
        selected = np.flatnonzero(probability_np[tx] >= 0.5)
        if not len(selected):
            selected = np.asarray([int(np.argmax(probability_np[tx]))])
        rows = []
        for slot in selected:
            point = points_np[tx, slot]
            associations = support_associations(point, anchors, by_tx[tx], 0.65, 14.0)
            rows.append({"candidate_id": len(rows), "slot": int(slot), "coord_m": point.tolist(), "raw_cluster_size": 1,
                         "support_count": len({row["anchor"] for row in associations}), "associations": associations,
                         "mean_composite_dissimilarity": float(history[-1]["loss"]),
                         "existence_probability": float(probability_np[tx, slot])})
        output[tx] = rows
    return output, {"mode": mode, "epochs": epochs, "history": history,
                    "pseudo_count_from_observed_path_cardinality": pseudo_count.tolist(),
                    "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint)}


def maximum_cardinality_gated_match(estimates: np.ndarray, truth: np.ndarray, gate_m: float) -> dict:
    estimates = np.asarray(estimates, float).reshape(-1, 2)
    truth = np.asarray(truth, float).reshape(-1, 2)
    ne, nt = len(estimates), len(truth)
    if ne == 0 or nt == 0:
        return {"matches": [], "tp": 0, "fp": ne, "fn": nt}
    distance = np.linalg.norm(estimates[:, None] - truth[None], axis=2)
    n = ne + nt
    reward = (min(ne, nt) + 1.0) * (gate_m + 1.0)
    invalid = reward * (n + 2.0)
    cost = np.zeros((n, n), dtype=float)
    cost[:ne, :nt] = np.where(distance <= gate_m, distance - reward, invalid)
    rows, columns = linear_sum_assignment(cost)
    matches = []
    for row, column in zip(rows, columns):
        if row < ne and column < nt and distance[row, column] <= gate_m:
            matches.append({"estimate": int(row), "truth": int(column), "distance_m": float(distance[row, column])})
    tp = len(matches)
    return {"matches": matches, "tp": tp, "fp": ne - tp, "fn": nt - tp}


def set_distances(estimates: np.ndarray, truth: np.ndarray, empty_penalty_m: float) -> tuple[float, float]:
    estimates, truth = np.asarray(estimates, float).reshape(-1, 2), np.asarray(truth, float).reshape(-1, 2)
    if not len(estimates) or not len(truth):
        return float(empty_penalty_m), float(empty_penalty_m)
    distance = np.linalg.norm(estimates[:, None] - truth[None], axis=2)
    nearest_e, nearest_t = np.min(distance, axis=1), np.min(distance, axis=0)
    return float(0.5 * (nearest_e.mean() + nearest_t.mean())), float(max(nearest_e.max(), nearest_t.max()))


def evaluate_candidate_set(by_tx: dict[int, list[dict]], truth_by_tx: Sequence[np.ndarray], room_xy: np.ndarray) -> dict:
    tp = fp = fn = 0
    chamfer, hausdorff, matching = [], [], []
    for tx, truth in enumerate(truth_by_tx):
        estimate = np.asarray([row["coord_m"] for row in by_tx.get(tx, [])], dtype=float).reshape(-1, 2)
        gated = maximum_cardinality_gated_match(estimate, np.asarray(truth)[:, :2], PRIMARY_GATE_M)
        tp += gated["tp"]
        fp += gated["fp"]
        fn += gated["fn"]
        c, h = set_distances(estimate, np.asarray(truth)[:, :2], float(np.linalg.norm(room_xy)))
        chamfer.append(c)
        hausdorff.append(h)
        matching.append({"tx_id": tx, **gated})
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-15)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1,
            "chamfer_m": float(np.mean(chamfer)), "hausdorff_m": float(np.mean(hausdorff)), "gated_matches": matching}


def candidate_score_grid(grid: np.ndarray, by_tx: dict[int, list[dict]], query_by_tx: Sequence[Acquisition]) -> np.ndarray:
    score = np.zeros(len(grid), dtype=float)
    used = 0
    for tx, query in enumerate(query_by_tx):
        observed = query.indirect.ranges_m
        points = np.asarray([row["coord_m"] for row in by_tx.get(tx, [])], dtype=float).reshape(-1, 2)
        if not len(observed) or not len(points):
            score += 4.0
            continue
        predicted = np.linalg.norm(grid[:, None] - points[None], axis=2)
        difference = np.abs(predicted[:, :, None] - observed[None, None])
        p_to_o = np.mean(np.min(difference, axis=2), axis=1)
        o_to_p = np.mean(np.min(difference, axis=1), axis=1)
        score += 0.5 * (p_to_o + o_to_p) + 0.12 * abs(len(points) - len(observed))
        used += 1
    return score / max(used, 1)


def localize_queries(by_tx: dict[int, list[dict]], query_positions_xyz: np.ndarray,
                     query_acquisitions: Sequence[Sequence[Acquisition]], room_xy: np.ndarray,
                     grid_side: int) -> tuple[list[dict], float]:
    xs, ys = np.linspace(0.0, room_xy[0], grid_side), np.linspace(0.0, room_xy[1], grid_side)
    grid = np.asarray([[x, y] for y in ys for x in xs])
    rows = []
    for frame, truth in enumerate(query_positions_xyz):
        query = [query_acquisitions[tx][frame] for tx in range(len(query_acquisitions))]
        score = candidate_score_grid(grid, by_tx, query)
        chosen = np.argsort(score, kind="stable")[: min(4, len(score))]
        relative = score[chosen] - score[chosen[0]]
        weight = np.exp(-relative / 0.12)
        weight /= np.sum(weight)
        estimate = np.sum(grid[chosen] * weight[:, None], axis=0)
        rows.append({"query": frame, "target_xy_m": truth[:2].tolist(), "estimate_xy_m": estimate.tolist(),
                     "error_m": float(np.linalg.norm(estimate - truth[:2])), "minimum_set_score": float(score[chosen[0]])})
    return rows, float(np.mean([row["error_m"] for row in rows]))


def bootstrap_ci(values: np.ndarray, *, seed: int, repetitions: int = 5000) -> list[float]:
    values = np.asarray(values, float)
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(values), (repetitions, len(values)))
    return np.quantile(values[draw].mean(axis=1), (0.025, 0.975)).tolist()


def summarize_blocks(rows: list[dict], *, primary_only: bool) -> list[dict]:
    selected = [row for row in rows if (not primary_only or row.get("primary", False))]
    output = []
    keys = sorted({(row["method"], row.get("beta")) for row in selected}, key=lambda x: (x[0], -1 if x[1] is None else x[1]))
    for method, beta in keys:
        local = [row for row in selected if row["method"] == method and row.get("beta") == beta]
        # Scene-clustered macro: average noise conditions inside each independent scene.
        scene_ids = sorted({row["scene_seed"] for row in local})
        scene_metrics = {metric: np.asarray([np.mean([row[metric] for row in local if row["scene_seed"] == scene]) for scene in scene_ids])
                         for metric in ("f1", "precision", "recall", "chamfer_m", "hausdorff_m", "localization_mean_m")}
        summary = {"method": method, "beta": beta, "blocks": len(local), "independent_scenes": len(scene_ids),
                   "primary": bool(local[0].get("primary", False)), "oracle_excluded": bool(local[0].get("oracle_excluded", False))}
        for metric, values in scene_metrics.items():
            summary[f"macro_{metric}"] = float(np.mean(values))
            summary[f"scene_bootstrap_95ci_{metric}"] = bootstrap_ci(values, seed=stable_seed("ci", method, beta, metric))
        output.append(summary)
    return output


METHOD_REGISTRY = {
    "corrected_mpurge": {"features": "delay/range and observable Tx channel identity", "training": "none", "truth_cardinality": False,
                          "applicability": "native original-MPUrge VT construction"},
    "corrected_pba": {"features": "delay/range and observable Tx channel identity", "training": "none", "truth_cardinality": False,
                       "applicability": "native PBA comparator under corrected window/conflict/cross bracket"},
    "cycle_consistent_delay": {"features": "delay/range, survey coordinates, Tx identity", "training": "none", "truth_cardinality": False,
                               "applicability": "variable-cardinality delay-only VT construction"},
    "bernoulli_rfs_multimodal": {"features": "same-acquisition delay, corrupted power, corrupted AoA, Tx identity", "training": "none", "truth_cardinality": False,
                                  "applicability": "sequential survey VT-track construction"},
    "two_sided_vt_registration": {"features": "same-acquisition delay, corrupted AoA/power, survey coordinates, Tx identity", "training": "none; fixed mutual gates", "truth_cardinality": False,
                                    "applicability": "mutual survey-to-VT and VT-to-survey round-trip registration"},
    "multipath_bundle_adjustment_noncheating": {"features": "same-acquisition delay, corrupted AoA/power, known calibration survey poses, Tx identity", "training": "per-survey robust BA with held-out-anchor BIC", "truth_cardinality": False,
                                                 "applicability": "joint VT and soft-association refinement; survey poses fixed"},
    "aoa_power_inverse_consensus": {"features": "same-acquisition delay, corrupted power, corrupted AoA, Tx identity", "training": "none", "truth_cardinality": False,
                                    "applicability": "multimodal survey VT construction"},
    "beta_marginal_survival_multimodal": {"features": "same-acquisition delay, corrupted power, corrupted AoA, missing detections", "training": "none", "truth_cardinality": False,
                                           "applicability": "multimodal existence-filtered VT construction"},
    "diffassign_bic_delay": {"features": "delay/range, survey coordinates, Tx identity", "training": "per-survey self-supervised reconstruction", "truth_cardinality": False,
                              "applicability": "unknown-cardinality differentiable VT construction"},
    "diffassign_bic_multimodal": {"features": "same-acquisition delay and corrupted AoA", "training": "per-survey self-supervised reconstruction", "truth_cardinality": False,
                                   "applicability": "unknown-cardinality multimodal differentiable VT construction"},
    "diffassign_multimodal_oracle_cardinality_ablation": {"features": "same-acquisition delay and corrupted AoA", "training": "per-survey reconstruction", "truth_cardinality": True,
                                                           "applicability": "labelled diagnostic only; excluded from primary"},
    "selfsup_deepsets_multimodal": {"features": "same-acquisition delay, corrupted power/AoA, survey coordinates", "training": "only range+AoA set reconstruction on this calibration survey", "truth_cardinality": False,
                                     "applicability": "self-supervised variable-existence VT output"},
    "selfsup_attention_multimodal": {"features": "same-acquisition delay, corrupted power/AoA, survey coordinates", "training": "only range+AoA set reconstruction on this calibration survey", "truth_cardinality": False,
                                     "applicability": "permutation-invariant attention VT output"},
    "old_supervised_vt_pointnet": {"features": "not run", "training": "5,000 simulator-labelled wall-VT sets in legacy code", "truth_cardinality": True,
                                    "applicability": "excluded: prohibited extra labels/examples and fixed true cardinality"},
    "CAEZ_probability_MLP": {"features": "CIR magnitude", "training": "supervised receiver-coordinate labels", "truth_cardinality": False,
                              "applicability": "excluded here: receiver coordinate output, no VT-set output adapter without labels"},
    "DICHASUS_ADP_8NN": {"features": "coherent CIR", "training": "nonparametric receiver map", "truth_cardinality": False,
                          "applicability": "excluded here: receiver-coordinate retrieval, not VT construction"},
}


def write_jsonl_gz(path: Path, rows: Iterable[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as target:
        for row in rows:
            target.write(json.dumps(json_ready(row), allow_nan=False, separators=(",", ":")) + "\n")


def write_json_gz_atomic(path: Path, payload: dict) -> None:
    """Commit one completed block atomically for safe interruption/resume."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="\n") as target:
        json.dump(json_ready(payload), target, allow_nan=False, separators=(",", ":"))
    os.replace(temporary, path)


def read_json_gz(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        return json.load(source)


def json_fingerprint(payload: dict) -> str:
    canonical = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_leaderboards(result: dict, csv_path: Path, tex_path: Path) -> None:
    """Write one row per executed method; oracle diagnostics stay labelled."""

    rows = list(result["primary_summaries"])
    present = {row["method"] for row in rows}
    for row in result["all_beta_summaries"]:
        if row["method"] not in present and row.get("beta") is None:
            rows.append(row)
            present.add(row["method"])
    rows.sort(key=lambda row: (-row["macro_f1"], row["method"]))
    fields = ["method", "beta", "primary", "oracle_excluded", "blocks", "independent_scenes",
              "macro_precision", "macro_recall", "macro_f1", "scene_bootstrap_95ci_f1",
              "macro_chamfer_m", "macro_hausdorff_m", "macro_localization_mean_m"]
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    def escape(value: object) -> str:
        return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%")

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\caption{Corrected MPUrge VT-construction benchmark. Scene-clustered macro means; the oracle row is diagnostic and excluded from primary claims.}",
        r"\label{tab:corrected-mpurge-vt-full}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Method & $\beta$ & P & R & F1 & Chamfer (m) & Hausdorff (m) & Loc. (m) \\",
        r"\midrule",
    ]
    for row in rows:
        method = escape(row["method"] + (" [oracle]" if row.get("oracle_excluded") else ""))
        beta = "--" if row.get("beta") is None else f"{row['beta']:.2f}"
        tex.append(
            f"{method} & {beta} & {row['macro_precision']:.4f} & {row['macro_recall']:.4f} & "
            f"{row['macro_f1']:.4f} & {row['macro_chamfer_m']:.4f} & "
            f"{row['macro_hausdorff_m']:.4f} & {row['macro_localization_mean_m']:.4f} \\\\" 
        )
    tex.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    tex_path.write_text("\n".join(tex), encoding="utf-8")


def register_candidates(raw_candidates: list[dict], raw_associations: list[dict], *, block: dict, method: str,
                        tx: int, candidates: list[dict], beta: float | None = None) -> None:
    for candidate in candidates:
        row = {"record_type": "candidate", **block, "method": method, "tx_id": tx, "beta": beta,
               **{key: value for key, value in candidate.items() if key != "associations"}}
        raw_candidates.append(row)
        for association in candidate.get("associations", []):
            raw_associations.append({"record_type": "association", **block, "method": method, "tx_id": tx, "beta": beta,
                                     "candidate_id": candidate["candidate_id"], **association})


def run_block(
    scene_seed: int,
    noise_std_m: float,
    args: argparse.Namespace,
    device: torch.device,
    checkpoint_dir: Path,
) -> tuple[list[dict], list[dict], list[dict], list[dict], dict]:
    scene = make_scene(scene_seed)
    block_seed = stable_seed(args.algorithm_seed, scene_seed, noise_std_m)
    anchors_xyz = grid_positions(scene, args.survey_points, seed=stable_seed(block_seed, "survey"))
    queries_xyz = query_positions(scene, args.query_points, seed=stable_seed(block_seed, "query"))
    survey = acquire(scene, anchors_xyz, noise_std_m, seed=stable_seed(block_seed, "survey-acq"), maximum_paths=args.maximum_paths)
    query = acquire(scene, queries_xyz, noise_std_m, seed=stable_seed(block_seed, "query-acq"), maximum_paths=args.maximum_paths)
    anchors = anchors_xyz[:, :2]
    room_xy = scene.room_xyz_m[:2]
    block = {"scene_seed": scene_seed, "noise_std_m": float(noise_std_m), "noise_seed": stable_seed(block_seed, "survey-acq")}
    raw_candidates: list[dict] = []
    raw_associations: list[dict] = []
    query_rows: list[dict] = []
    metrics: list[dict] = []
    audit: dict = {"block": block, "room_xyz_m": scene.room_xyz_m.tolist(), "transmitters_xyz_m": scene.transmitters_xyz_m.tolist(),
                   "evaluation_truth_cardinality_by_tx": [len(value) for value in scene.truth_vts_by_tx],
                   "direct_removal": [{"tx_id": tx, "removed": sum(row.direct_removed for row in rows), "frames": len(rows)} for tx, rows in enumerate(survey)],
                   "method_details": {}}

    def evaluate(method: str, by_tx: dict[int, list[dict]], runtime_s: float, *, beta: float | None = None,
                 primary: bool = True, oracle_excluded: bool = False) -> None:
        set_metrics = evaluate_candidate_set(by_tx, scene.truth_vts_by_tx, room_xy)
        predictions, localization = localize_queries(by_tx, queries_xyz, query, room_xy, args.grid_side)
        row = {**block, "method": method, "beta": beta, "runtime_s": runtime_s, "primary": primary,
               "oracle_excluded": oracle_excluded, **{key: value for key, value in set_metrics.items() if key != "gated_matches"},
               "localization_mean_m": localization, "candidate_count": sum(map(len, by_tx.values())),
               "gated_matches": set_metrics["gated_matches"]}
        metrics.append(row)
        for prediction in predictions:
            query_rows.append({**block, "method": method, "beta": beta, **prediction})

    # Corrected native branches and exhaustive beta curves.
    for native in ("mpurge", "pba"):
        for cross_mode in ("printed", "order"):
            for grouping in ("star", "components"):
                method = f"corrected_{native}_{cross_mode}_{grouping}"
                stamp = time.perf_counter()
                candidates_by_tx = {}
                beta_by_tx = {}
                for tx in range(len(survey)):
                    candidates = construct_native(anchors, survey[tx], scene.transmitters_xyz_m[tx, :2], room_xy,
                                                  method=native, cross_mode=cross_mode, grouping=grouping,
                                                  neighbourhood_size=args.neighbourhood_size)
                    candidates_by_tx[tx] = candidates
                    beta_by_tx[tx] = dynamic_beta_sets(candidates, args.beta_step)
                    register_candidates(raw_candidates, raw_associations, block=block, method=method, tx=tx, candidates=candidates)
                runtime = time.perf_counter() - stamp
                union_beta = sorted({row["beta"] for rows in beta_by_tx.values() for row in rows})
                for beta in union_beta:
                    kept_by_tx = {}
                    for tx, candidates in candidates_by_tx.items():
                        available = [row for row in beta_by_tx[tx] if row["beta"] == beta]
                        kept_ids = set(available[0]["kept_candidate_ids"] if available else [])
                        kept = [row for row in candidates if row["candidate_id"] in kept_ids]
                        kept_by_tx[tx] = kept
                        raw_candidates.append({"record_type": "beta_set", **block, "method": method, "tx_id": tx, "beta": beta,
                                               "kept_candidate_ids": sorted(kept_ids),
                                               "candidates": [{"candidate_id": row["candidate_id"], "coord_m": row["coord_m"],
                                                               "support_count": row["support_count"]} for row in kept]})
                    evaluate(method, kept_by_tx, runtime, beta=beta, primary=abs(beta - 1.0) < 1e-9)
                audit["method_details"][method] = {"runtime_s": runtime, "dynamic_beta_exhausted": all(not rows[-1]["kept_candidate_ids"] for rows in beta_by_tx.values()),
                                                    "last_beta_by_tx": [rows[-1]["beta"] for rows in beta_by_tx.values()]}

    # Delay-only cycle-consistent clustering.
    stamp = time.perf_counter()
    cycle = {tx: cycle_consistent_candidates(anchors, survey[tx], room_xy, noise_std_m,
                                              seed=stable_seed(block_seed, "cycle", tx), draws=args.cycle_draws)
             for tx in range(len(survey))}
    runtime = time.perf_counter() - stamp
    for tx, rows in cycle.items():
        register_candidates(raw_candidates, raw_associations, block=block, method="cycle_consistent_delay", tx=tx, candidates=rows)
    evaluate("cycle_consistent_delay", cycle, runtime)

    # Same-acquisition range/AoA/power inverse consensus and survival filter.
    for survival, name in ((False, "aoa_power_inverse_consensus"), (True, "beta_marginal_survival_multimodal")):
        stamp = time.perf_counter()
        by_tx = {tx: aoa_inverse_consensus(anchors, survey[tx], noise_std_m, survival=survival) for tx in range(len(survey))}
        runtime = time.perf_counter() - stamp
        for tx, rows in by_tx.items():
            register_candidates(raw_candidates, raw_associations, block=block, method=name, tx=tx, candidates=rows)
        evaluate(name, by_tx, runtime)

    # Bernoulli/RFS track lifecycle on the survey acquisition order.
    stamp = time.perf_counter()
    rfs = {tx: rfs_lifecycle(anchors, survey[tx], noise_std_m) for tx in range(len(survey))}
    runtime = time.perf_counter() - stamp
    for tx, rows in rfs.items():
        register_candidates(raw_candidates, raw_associations, block=block, method="bernoulli_rfs_multimodal", tx=tx, candidates=rows)
    evaluate("bernoulli_rfs_multimodal", rfs, runtime)

    # Independent mutual/round-trip registration (not inverse-consensus alias).
    stamp = time.perf_counter()
    two_sided = {tx: two_sided_vt_registration(anchors, survey[tx], room_xy, noise_std_m)
                 for tx in range(len(survey))}
    runtime = time.perf_counter() - stamp
    for tx, rows in two_sided.items():
        register_candidates(raw_candidates, raw_associations, block=block,
                            method="two_sided_vt_registration", tx=tx, candidates=rows)
    audit["method_details"]["two_sided_vt_registration"] = {
        "runtime_s": runtime, "mutual_nearest_per_survey_pair": True,
        "roundtrip_best_observation_and_candidate": True,
    }
    evaluate("two_sided_vt_registration", two_sided, runtime)

    # Noncheating fixed-survey-pose multipath BA with held-out-anchor BIC.
    stamp = time.perf_counter()
    bundle, bundle_details = {}, []
    for tx in range(len(survey)):
        rows, detail = multipath_bundle_adjustment_noncheating(
            anchors, survey[tx], room_xy, noise_std_m, kmax=args.kmax,
            steps=args.ba_steps, seed=stable_seed(block_seed, "noncheating-ba", tx), device=device,
        )
        bundle[tx] = rows
        bundle_details.append({"tx_id": tx, **detail})
    runtime = time.perf_counter() - stamp
    for tx, rows in bundle.items():
        register_candidates(raw_candidates, raw_associations, block=block,
                            method="multipath_bundle_adjustment_noncheating", tx=tx, candidates=rows)
    audit["method_details"]["multipath_bundle_adjustment_noncheating"] = {
        "runtime_s": runtime, "selection": bundle_details,
    }
    evaluate("multipath_bundle_adjustment_noncheating", bundle, runtime)

    # Model-selected-cardinality differentiable assignment, delay and multimodal.
    for multimodal, name in ((False, "diffassign_bic_delay"), (True, "diffassign_bic_multimodal")):
        stamp = time.perf_counter()
        by_tx, details = {}, []
        for tx in range(len(survey)):
            rows, detail = diffassign_model_selected(anchors, survey[tx], room_xy, multimodal=multimodal,
                                                     kmax=args.kmax, steps=args.diffassign_steps, restarts=args.diffassign_restarts,
                                                     seed=stable_seed(block_seed, name, tx), device=device)
            by_tx[tx] = rows
            details.append({"tx_id": tx, **detail})
        runtime = time.perf_counter() - stamp
        for tx, rows in by_tx.items():
            register_candidates(raw_candidates, raw_associations, block=block, method=name, tx=tx, candidates=rows)
        audit["method_details"][name] = {"runtime_s": runtime, "selection": details}
        evaluate(name, by_tx, runtime)

    # Explicitly labelled oracle-cardinality diagnostic; never primary.
    stamp = time.perf_counter()
    oracle, oracle_details = {}, []
    for tx in range(len(survey)):
        rows, detail = diffassign_model_selected(anchors, survey[tx], room_xy, multimodal=True, kmax=args.kmax,
                                                 steps=args.diffassign_steps, restarts=args.diffassign_restarts,
                                                 seed=stable_seed(block_seed, "oracle", tx), device=device,
                                                 oracle_k=len(scene.truth_vts_by_tx[tx]))
        oracle[tx] = rows
        oracle_details.append({"tx_id": tx, **detail})
    runtime = time.perf_counter() - stamp
    for tx, rows in oracle.items():
        register_candidates(raw_candidates, raw_associations, block=block,
                            method="diffassign_multimodal_oracle_cardinality_ablation", tx=tx, candidates=rows)
    audit["method_details"]["diffassign_multimodal_oracle_cardinality_ablation"] = {"runtime_s": runtime, "selection": oracle_details}
    evaluate("diffassign_multimodal_oracle_cardinality_ablation", oracle, runtime, primary=False, oracle_excluded=True)

    # Self-supervised DeepSets and permutation-invariant self-attention.
    for mode, name in (("deepsets", "selfsup_deepsets_multimodal"), ("attention", "selfsup_attention_multimodal")):
        stamp = time.perf_counter()
        checkpoint = checkpoint_dir / f"scene{scene_seed}_sigma{noise_std_m:g}_{name}.pt"
        by_tx, detail = selfsupervised_set_estimator(anchors, survey, room_xy, mode=mode, kmax=args.kmax,
                                                     epochs=args.neural_epochs, seed=stable_seed(block_seed, name),
                                                     device=device, checkpoint=checkpoint)
        runtime = time.perf_counter() - stamp
        for tx, rows in by_tx.items():
            register_candidates(raw_candidates, raw_associations, block=block, method=name, tx=tx, candidates=rows)
        audit["method_details"][name] = {"runtime_s": runtime, **detail}
        evaluate(name, by_tx, runtime)

    return metrics, raw_candidates, raw_associations, query_rows, audit


def self_tests() -> dict:
    assert TOTAL_WINDOW_P == 2 * HALF_WINDOW_P + 1 == 5
    windows = padded_windows(np.asarray([1.0, 2.0, 4.0]), HALF_WINDOW_P)
    assert windows.shape == (3, 5)
    # Composite ordering must win even when raw difference suggests the reverse.
    good = Match(0, 0, 1.0, 3.0, 0.1, 0)
    bad = Match(1, 1, 2.0, 2.1, 1.0, 0)
    ordered = filter_current_by_composite([bad, good], "order")
    assert ordered[0] == good
    # The contradictory printed/geometric checks are genuinely bracketed.
    assert crosses((4.0, 2.0), (3.0, 2.5), "order")
    assert not crosses((4.0, 2.0), (3.0, 2.5), "printed")
    beta = dynamic_beta_sets([{"candidate_id": 0, "support_count": 2}, {"candidate_id": 1, "support_count": 5}], 0.1)
    assert beta[-1]["kept_candidate_ids"] == [] and beta[-1]["beta"] > 0
    # Gated assignment maximizes cardinality before distance.
    matched = maximum_cardinality_gated_match(np.asarray([[0.0, 0.0], [0.28, 0.0]]), np.asarray([[0.01, 0.0], [0.29, 0.0]]), 0.30)
    assert matched["tp"] == 2
    assert "truth" not in inspect.signature(selfsupervised_set_estimator).parameters
    assert "truth" not in inspect.signature(two_sided_vt_registration).parameters
    assert "truth" not in inspect.signature(multipath_bundle_adjustment_noncheating).parameters
    assert all(not row.get("truth_cardinality", False) for name, row in METHOD_REGISTRY.items() if "oracle" not in name and name != "old_supervised_vt_pointnet")
    return {"total_window_P": TOTAL_WINDOW_P, "half_window_p": HALF_WINDOW_P, "tests": 8, "status": "PASS"}


def make_report(result: dict) -> str:
    lines = [
        "# Corrected original-MPUrge VT-construction suite",
        "",
        f"Status: **{result['status']}**. This is a disclosed replacement-scene benchmark, not the unavailable original simulator reproduction.",
        "",
        "## Correctness and information barrier",
        "",
        "- Total window `P=5` is explicitly converted to half-window `p=2`.",
        "- Current-iteration conflicts are ordered by the full composite dissimilarity.",
        "- Printed-algebra and geometric-order cross-checks, plus star/component grouping interpretations, are all retained.",
        "- Candidate support and beta filtering never mix transmitter channels; every native beta curve ends with an empty set.",
        "- Primary learned/optimized methods see only the same calibration acquisitions. No VT/ray IDs, true VT cardinality, query truth, dense sampling, or additional simulator-labelled examples are used.",
        "- The oracle-cardinality differentiable-assignment row is explicitly excluded from primary ranking.",
        "",
        "## Primary scene-clustered macro results",
        "",
        "| Method | Beta | F1 | Chamfer (m) | Localization (m) |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(result["primary_summaries"], key=lambda value: (-value["macro_f1"], value["method"])):
        beta = "-" if row["beta"] is None else f"{row['beta']:.2f}"
        lines.append(f"| {row['method']} | {beta} | {row['macro_f1']:.4f} | {row['macro_chamfer_m']:.4f} | {row['macro_localization_mean_m']:.4f} |")
    lines.extend(["", "## Feature inputs and exclusions", "", "| Method/family | Inputs | Fit | Applicability |", "|---|---|---|---|"])
    for name, row in METHOD_REGISTRY.items():
        lines.append(f"| {name} | {row['features']} | {row['training']} | {row['applicability']} |")
    lines.extend(["", "Raw candidate memberships, supports and associations are in the gzip JSONL artifacts named in `artifacts`; query estimates are stored separately.", ""])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research" / "four_paper_report" / "corrected_mpurge_vt_full_release")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--scene-seeds", nargs="+", type=int, default=list(DEFAULT_SCENE_SEEDS))
    parser.add_argument("--noise-stds", nargs="+", type=float, default=list(DEFAULT_NOISE_STDS_M))
    parser.add_argument("--algorithm-seed", type=int, default=DEFAULT_ALGORITHM_SEED)
    parser.add_argument("--survey-points", type=int, default=36)
    parser.add_argument("--query-points", type=int, default=20)
    parser.add_argument("--maximum-paths", type=int, default=9)
    parser.add_argument("--neighbourhood-size", type=int, default=7)
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--cycle-draws", type=int, default=700)
    parser.add_argument("--kmax", type=int, default=7)
    parser.add_argument("--diffassign-steps", type=int, default=90)
    parser.add_argument("--diffassign-restarts", type=int, default=2)
    parser.add_argument("--ba-steps", type=int, default=60)
    parser.add_argument("--neural-epochs", type=int, default=120)
    parser.add_argument("--grid-side", type=int, default=31)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--self-test-only", action="store_true")
    parser.add_argument("--resume", action="store_true", help="reuse only hash-matched atomic block shards")
    args = parser.parse_args(argv)
    if args.quick:
        args.scene_seeds = args.scene_seeds[:1]
        args.noise_stds = args.noise_stds[:2]
        args.survey_points = min(args.survey_points, 20)
        args.query_points = min(args.query_points, 6)
        args.neighbourhood_size = min(args.neighbourhood_size, 6)
        args.beta_step = max(args.beta_step, 0.5)
        args.cycle_draws = min(args.cycle_draws, 100)
        args.kmax = min(args.kmax, 5)
        args.diffassign_steps = min(args.diffassign_steps, 12)
        args.diffassign_restarts = 1
        args.ba_steps = min(args.ba_steps, 8)
        args.neural_epochs = min(args.neural_epochs, 8)
        args.grid_side = min(args.grid_side, 15)
    return args


def main(argv: Sequence[str] | None = None) -> dict:
    args = parse_args(argv)
    tests = self_tests()
    if args.self_test_only:
        print(json.dumps(tests, indent=2))
        return tests
    if args.device == "cuda" and not torch.cuda.is_available():
        if args.require_cuda:
            raise RuntimeError("CUDA required but unavailable")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = args.output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    receiver_snapshot = HERE / "mpurge_vt_receiver_snapshot_20260807.py"
    source_hashes = {str(Path(__file__).resolve()): sha256(Path(__file__).resolve()),
                     str(receiver_snapshot.resolve()): sha256(receiver_snapshot)}
    fingerprint_payload = {"schema": SCHEMA, "arguments": json_ready(vars(args)), "source_hashes": source_hashes}
    run_fingerprint = json_fingerprint(fingerprint_payload)
    executed_sources = args.output_dir / "executed_sources"
    executed_sources.mkdir(exist_ok=True)
    executed_source_paths = {}
    for source_name, expected_hash in source_hashes.items():
        source_path = Path(source_name)
        destination = executed_sources / source_path.name
        if destination.exists() and sha256(destination) != expected_hash:
            raise RuntimeError(f"executed-source collision: {destination}")
        if not destination.exists():
            shutil.copy2(source_path, destination)
        if sha256(destination) != expected_hash:
            raise RuntimeError(f"executed-source hash mismatch: {destination}")
        executed_source_paths[source_name] = str(destination.resolve())
    config = {"schema": SCHEMA, "phase": "quick_smoke" if args.quick else "full", "arguments": json_ready(vars(args)),
              "run_fingerprint": run_fingerprint,
              "paper_contract": {"total_window_P": TOTAL_WINDOW_P, "half_window_p": HALF_WINDOW_P, "alpha": ALPHA,
                                  "conflict_priority": "composite_dissimilarity", "cross_modes": ["printed", "order"],
                                  "grouping_interpretations": ["star", "components"], "dynamic_beta_until_empty": True,
                                  "max_cardinality_gated_evaluation": True},
              "information_barrier": {"same_acquisition_multimodal_features_allowed": True, "extra_dense_sampling": False,
                                      "extra_forward_simulator_training_examples": False, "VT_or_ray_ID_input": False,
                                      "query_truth_input": False, "true_cardinality_primary": False},
              "receiver_provenance": {"kind": "immutable_vendored_snapshot",
                                       "path": str(receiver_snapshot.resolve()),
                                       "sha256": source_hashes[str(receiver_snapshot.resolve())],
                                       "supersedes_mutated_shared_receiver": True},
              "executed_source_copies": executed_source_paths,
              "source_hashes": source_hashes, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    config_path = args.output_dir / "config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("refusing to mix incompatible blocks in an existing output directory")
        config = previous
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    started = time.perf_counter()
    all_metrics, all_candidates, all_associations, all_queries, audits = [], [], [], [], []
    shards = args.output_dir / "block_shards"
    shards.mkdir(exist_ok=True)
    for scene_seed in args.scene_seeds:
        for noise in args.noise_stds:
            noise_token = format(float(noise), ".17g").replace("-", "m").replace(".", "p")
            shard_path = shards / f"scene{scene_seed}_sigma{noise_token}.json.gz"
            if shard_path.exists():
                if not args.resume:
                    raise RuntimeError(f"block shard exists; pass --resume: {shard_path}")
                shard = read_json_gz(shard_path)
                if shard.get("run_fingerprint") != run_fingerprint:
                    raise RuntimeError(f"incompatible block shard: {shard_path}")
                metrics = shard["metrics"]
                candidates = shard["candidates"]
                associations = shard["associations"]
                query_rows = shard["query_rows"]
                audit = shard["audit"]
                resumed = True
            else:
                stamp = time.perf_counter()
                metrics, candidates, associations, query_rows, audit = run_block(
                    scene_seed, float(noise), args, device, checkpoints
                )
                audit["wall_s"] = time.perf_counter() - stamp
                shard = {"schema": SCHEMA, "run_fingerprint": run_fingerprint,
                         "metrics": metrics, "candidates": candidates,
                         "associations": associations, "query_rows": query_rows, "audit": audit}
                write_json_gz_atomic(shard_path, shard)
                resumed = False
            all_metrics.extend(metrics)
            all_candidates.extend(candidates)
            all_associations.extend(associations)
            all_queries.extend(query_rows)
            audits.append(audit)
            print(json.dumps({"finished_scene": scene_seed, "noise_std_m": noise, "block_wall_s": audit["wall_s"],
                              "metric_rows": len(metrics), "candidates": len(candidates),
                              "resumed": resumed, "completed_blocks": len(audits)}), flush=True)
    paths = {
        "block_metrics": args.output_dir / "block_metrics.jsonl",
        "raw_candidates": args.output_dir / "raw_candidates_and_beta_sets.jsonl.gz",
        "raw_associations": args.output_dir / "raw_associations.jsonl.gz",
        "query_predictions": args.output_dir / "query_predictions.jsonl.gz",
        "audit": args.output_dir / "block_audit.json",
        "leaderboard_csv": args.output_dir / "all_method_leaderboard.csv",
        "leaderboard_tex": args.output_dir / "all_method_leaderboard.tex",
    }
    with paths["block_metrics"].open("w", encoding="utf-8", newline="\n") as target:
        for row in all_metrics:
            target.write(json.dumps(json_ready(row), allow_nan=False) + "\n")
    write_jsonl_gz(paths["raw_candidates"], all_candidates)
    write_jsonl_gz(paths["raw_associations"], all_associations)
    write_jsonl_gz(paths["query_predictions"], all_queries)
    paths["audit"].write_text(json.dumps(json_ready(audits), indent=2, allow_nan=False), encoding="utf-8")
    result = {
        "schema": SCHEMA,
        "status": "QUICK_SMOKE_COMPLETE" if args.quick else "FULL_COMPLETE",
        "claim": "corrected source-bracketed original-MPUrge VT construction on disclosed replacement scenes; not exact paper reproduction",
        "self_tests": tests,
        "config": config,
        "method_registry": METHOD_REGISTRY,
        "primary_summaries": summarize_blocks(all_metrics, primary_only=True),
        "all_beta_summaries": summarize_blocks(all_metrics, primary_only=False),
        "blocks": len(args.scene_seeds) * len(args.noise_stds),
        "completed_block_keys": [audit["block"] for audit in audits],
        "run_fingerprint": run_fingerprint,
        "runtime_s": time.perf_counter() - started,
        "device": {"requested": args.device, "used": str(device), "cuda_available": torch.cuda.is_available(),
                   "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "torch": torch.__version__, "python": platform.python_version(), "platform": platform.platform()},
        "artifacts": {name: str(path.resolve()) for name, path in paths.items()},
    }
    result_path = args.output_dir / "results.json"
    report_path = args.output_dir / "report.md"
    result["artifacts"]["results"] = str(result_path.resolve())
    result["artifacts"]["report"] = str(report_path.resolve())
    result["artifacts"]["block_shards"] = str(shards.resolve())
    result_path.write_text(json.dumps(json_ready(result), indent=2, allow_nan=False), encoding="utf-8")
    report_path.write_text(make_report(result), encoding="utf-8")
    write_leaderboards(result, paths["leaderboard_csv"], paths["leaderboard_tex"])
    manifest_lines = []
    for path in sorted([p for p in args.output_dir.rglob("*") if p.is_file()]):
        if path.name == "SHA256SUMS.txt":
            continue
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(args.output_dir).as_posix()}")
    (args.output_dir / "SHA256SUMS.txt").write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "runtime_s": result["runtime_s"], "output": str(result_path),
                      "primary": result["primary_summaries"]}, indent=2), flush=True)
    return result


if __name__ == "__main__":
    main()
