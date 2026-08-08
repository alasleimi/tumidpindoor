"""Shared deterministic simulators, set models, and metrics for paper protocols."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def stable_seed(*parts: object) -> int:
    payload = "paper-protocol-v1:" + ":".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(payload.encode()).digest()[:4], "little")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class Source:
    xyz_m: np.ndarray
    tx_id: int
    order: int


@dataclass(frozen=True)
class Fingerprint:
    ranges_m: np.ndarray
    powers_db: np.ndarray
    tx_ids: np.ndarray

    def __len__(self) -> int:
        return len(self.ranges_m)


def _mirror_axis(point: np.ndarray, axis: int, coordinate: float) -> np.ndarray:
    result = np.asarray(point, dtype=np.float64).copy()
    result[axis] = 2.0 * coordinate - result[axis]
    return result


def rectangular_sources(
    room_xyz_m: Sequence[float],
    transmitters_xyz_m: np.ndarray,
    *,
    maximum_order: int = 2,
    include_floor_ceiling: bool = True,
) -> tuple[Source, ...]:
    """LoS plus rectangular-room image sources through the requested order."""

    width, depth, height = map(float, room_xyz_m)
    planes = [(0, 0.0), (0, width), (1, 0.0), (1, depth)]
    if include_floor_ceiling:
        planes += [(2, 0.0), (2, height)]
    output: list[Source] = []
    for tx_id, transmitter in enumerate(np.asarray(transmitters_xyz_m, dtype=np.float64)):
        seen: dict[tuple[float, float, float], int] = {
            tuple(np.round(transmitter, 10)): 0
        }
        frontier = [(transmitter.copy(), tuple())]
        for order in range(maximum_order + 1):
            following = []
            for point, history in frontier:
                key = tuple(np.round(point, 10))
                if seen.get(key, order) == order:
                    output.append(Source(point.copy(), tx_id, order))
                if order == maximum_order:
                    continue
                for plane_index, (axis, coordinate) in enumerate(planes):
                    # Immediate reflection at the same plane creates the prior image.
                    if history and plane_index == history[-1]:
                        continue
                    image = _mirror_axis(point, axis, coordinate)
                    image_key = tuple(np.round(image, 10))
                    new_order = order + 1
                    if new_order <= seen.get(image_key, 10**9):
                        seen[image_key] = new_order
                        following.append((image, history + (plane_index,)))
            frontier = following
    unique: dict[tuple[int, float, float, float], Source] = {}
    for source in output:
        key = (source.tx_id, *np.round(source.xyz_m, 9))
        prior = unique.get(key)
        if prior is None or source.order < prior.order:
            unique[key] = source
    return tuple(sorted(unique.values(), key=lambda s: (s.tx_id, s.order, *s.xyz_m.tolist())))


def _point_in_triangle(point: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    v0, v1, v2 = c - a, b - a, point - a
    dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
    dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
    denominator = dot00 * dot11 - dot01 * dot01
    if abs(float(denominator)) < 1.0e-12:
        return False
    u = (dot11 * dot02 - dot01 * dot12) / denominator
    v = (dot00 * dot12 - dot01 * dot02) / denominator
    return bool(u >= -1.0e-9 and v >= -1.0e-9 and u + v <= 1.0 + 1.0e-9)


def segment_intersects_quad(start: np.ndarray, end: np.ndarray, quad: np.ndarray) -> bool:
    quad = np.asarray(quad, dtype=np.float64)
    normal = np.cross(quad[1] - quad[0], quad[2] - quad[0])
    denominator = float(np.dot(normal, end - start))
    if abs(denominator) < 1.0e-12:
        return False
    t = float(np.dot(normal, quad[0] - start) / denominator)
    if not (1.0e-7 < t < 1.0 - 1.0e-7):
        return False
    point = start + t * (end - start)
    return _point_in_triangle(point, quad[0], quad[1], quad[2]) or _point_in_triangle(
        point, quad[0], quad[2], quad[3]
    )


def simulate_fingerprint(
    position_xyz_m: Sequence[float],
    sources: Sequence[Source],
    *,
    maximum_paths: int,
    rng: np.random.Generator | None = None,
    range_noise_std_m: float = 0.0,
    snr_db: float | None = None,
    obstructions: Sequence[np.ndarray] = (),
    missing_probability: float = 0.0,
) -> Fingerprint:
    position = np.asarray(position_xyz_m, dtype=np.float64)
    rows = []
    for source_index, source in enumerate(sources):
        distance = float(np.linalg.norm(position - source.xyz_m))
        blocked = any(segment_intersects_quad(position, source.xyz_m, quad) for quad in obstructions)
        if blocked:
            continue
        # Reflection loss makes path ranking deterministic but nontrivial.
        ripple = 1.5 * math.sin(0.73 * source_index + 0.31 * position[0] - 0.17 * position[1])
        power = -20.0 * math.log10(max(distance, 0.05)) - 7.0 * source.order + ripple
        rows.append((distance, power, source.tx_id))
    if not rows:
        return Fingerprint(np.empty(0), np.empty(0), np.empty(0, dtype=np.int64))
    rows.sort(key=lambda row: -row[1])
    rows = rows[:maximum_paths]
    ranges = np.asarray([row[0] for row in rows], dtype=np.float64)
    powers = np.asarray([row[1] for row in rows], dtype=np.float64)
    tx_ids = np.asarray([row[2] for row in rows], dtype=np.int64)
    if rng is not None:
        if range_noise_std_m > 0.0:
            ranges += rng.normal(0.0, range_noise_std_m, len(ranges))
        if snr_db is not None:
            # Delay uncertainty equivalent for a 2 GHz system, scaled by SNR.
            sigma = 0.15 * 10.0 ** (-float(snr_db) / 20.0)
            ranges += rng.normal(0.0, sigma, len(ranges))
        if missing_probability > 0.0:
            keep = rng.random(len(ranges)) >= missing_probability
            ranges, powers, tx_ids = ranges[keep], powers[keep], tx_ids[keep]
    positive = np.isfinite(ranges) & (ranges > 0.0)
    ranges, powers, tx_ids = ranges[positive], powers[positive], tx_ids[positive]
    order = np.argsort(ranges, kind="stable")
    return Fingerprint(ranges[order], powers[order], tx_ids[order])


def pack_fingerprints(
    fingerprints: Sequence[Fingerprint],
    *,
    maximum_paths: int,
    range_scale_m: float,
    maximum_tx: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((len(fingerprints), maximum_paths, 4), dtype=np.float32)
    mask = np.zeros((len(fingerprints), maximum_paths), dtype=bool)
    for row, fingerprint in enumerate(fingerprints):
        count = min(maximum_paths, len(fingerprint))
        if count == 0:
            continue
        ranges = fingerprint.ranges_m[:count]
        powers = fingerprint.powers_db[:count]
        tx = fingerprint.tx_ids[:count]
        values[row, :count, 0] = ranges / max(range_scale_m, 1.0e-6)
        values[row, :count, 1] = np.clip((powers + 80.0) / 80.0, -1.0, 1.0)
        values[row, :count, 2] = tx / max(maximum_tx - 1, 1)
        values[row, :count, 3] = 1.0
        mask[row, :count] = True
    return values, mask


def padded_ranges(fingerprints: Sequence[Fingerprint], maximum_paths: int) -> np.ndarray:
    output = np.full((len(fingerprints), maximum_paths), np.nan, dtype=np.float64)
    for index, fingerprint in enumerate(fingerprints):
        values = np.sort(fingerprint.ranges_m)[:maximum_paths]
        output[index, : len(values)] = values
    return output


def symmetric_chamfer_scores(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    query = query[np.isfinite(query)]
    refs = np.asarray(references, dtype=np.float64)
    if not len(query):
        return np.full(len(refs), np.inf)
    valid = np.isfinite(refs)
    difference = np.abs(refs[:, :, None] - query[None, None, :])
    difference[~valid, :] = np.inf
    ref_to_query = np.min(difference, axis=2)
    ref_term = np.sum(np.where(valid, ref_to_query, 0.0), axis=1) / np.maximum(valid.sum(1), 1)
    query_to_ref = np.min(difference, axis=1)
    query_term = np.mean(query_to_ref, axis=1)
    return 0.5 * (ref_term + query_term)


def weighted_knn(scores: np.ndarray, positions_xy_m: np.ndarray, k: int = 3) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return np.mean(positions_xy_m, axis=0)
    selected = finite[np.argsort(scores[finite], kind="stable")[: min(k, len(finite))]]
    safe = np.maximum(scores[selected], 1.0e-6)
    weights = 1.0 / safe
    weights /= weights.sum()
    return np.sum(positions_xy_m[selected] * weights[:, None], axis=0)


def robust_subset_prediction(
    fingerprint: Fingerprint,
    reference_ranges: np.ndarray,
    reference_xy_m: np.ndarray,
    *,
    k: int = 3,
) -> np.ndarray:
    """Triplet/jackknife consensus adapter of the frozen subset-consensus idea."""

    values = np.sort(fingerprint.ranges_m)
    hypotheses = [weighted_knn(symmetric_chamfer_scores(values, reference_ranges), reference_xy_m, k)]
    if len(values) <= 6:
        subsets = [values[np.asarray(indices)] for indices in __import__("itertools").combinations(range(len(values)), min(3, len(values)))]
    else:
        subsets = [np.delete(values, index) for index in range(len(values))]
    for subset in subsets:
        hypotheses.append(weighted_knn(symmetric_chamfer_scores(subset, reference_ranges), reference_xy_m, k))
    points = np.asarray(hypotheses)
    centre = np.median(points, axis=0)
    distance = np.linalg.norm(points - centre, axis=1)
    inliers = distance <= max(0.75, float(np.median(distance)) * 2.5)
    return np.mean(points[inliers], axis=0) if np.any(inliers) else hypotheses[0]


class SetEncoder(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.output_dim = 2 * hidden + 1

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token(values)
        numeric = mask[..., None].to(hidden.dtype)
        mean = (hidden * numeric).sum(1) / numeric.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        count = torch.log1p(mask.sum(1, keepdim=True).to(hidden.dtype))
        return torch.cat((mean, maximum, count), dim=1)


class PointNetRegressor(nn.Module):
    def __init__(self, context_dim: int = 0, hidden: int = 64):
        super().__init__()
        self.encoder = SetEncoder(hidden)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + context_dim, 128),
            nn.GELU(),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 2),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        encoded = self.encoder(values, mask)
        if context is not None:
            encoded = torch.cat((encoded, context), dim=1)
        return self.head(encoded)


class CandidateAttention(nn.Module):
    """Permutation-invariant fingerprint encoder with candidate-set attention."""

    def __init__(self, hidden: int = 64, context_dim: int = 0):
        super().__init__()
        self.encoder = SetEncoder(hidden)
        width = self.encoder.output_dim
        self.score = nn.Sequential(
            nn.Linear(width * 3 + 2 + context_dim, 160), nn.GELU(),
            nn.Linear(160, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(
        self,
        query_values: torch.Tensor,
        query_mask: torch.Tensor,
        reference_values: torch.Tensor,
        reference_mask: torch.Tensor,
        reference_xy: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.encoder(query_values, query_mask)
        batch, candidates, paths, features = reference_values.shape
        reference = self.encoder(
            reference_values.reshape(batch * candidates, paths, features),
            reference_mask.reshape(batch * candidates, paths),
        ).reshape(batch, candidates, -1)
        expanded = query[:, None].expand(-1, candidates, -1)
        pieces = [expanded, reference, torch.abs(expanded - reference), reference_xy]
        if context is not None:
            pieces.append(context[:, None].expand(-1, candidates, -1))
        logits = self.score(torch.cat(pieces, dim=-1)).squeeze(-1)
        weight = torch.softmax(logits, dim=1)
        return torch.sum(reference_xy * weight[..., None], dim=1), logits


class CompletionNet(nn.Module):
    """Small best-of-K conditional set completion network."""

    def __init__(self, maximum_output: int, latent_dim: int = 12):
        super().__init__()
        self.maximum_output = maximum_output
        self.latent_dim = latent_dim
        self.encoder = SetEncoder(48)
        self.body = nn.Sequential(
            nn.Linear(self.encoder.output_dim + latent_dim, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.values = nn.Linear(128, maximum_output)
        self.count = nn.Linear(128, maximum_output + 1)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.encoder(inputs, mask)
        if latent.ndim == 2:
            latent = latent[:, None, :]
        repeated = context[:, None].expand(-1, latent.shape[1], -1)
        hidden = self.body(torch.cat((repeated, latent), dim=-1))
        return torch.sort(torch.sigmoid(self.values(hidden)), dim=-1).values, self.count(hidden)


def configure_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the full paper-protocol run")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    return device


def train_pointnet(
    model: PointNetRegressor,
    values: np.ndarray,
    masks: np.ndarray,
    targets: np.ndarray,
    *,
    context: np.ndarray | None,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    seed_all(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5)
    history = []
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(values))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            x = torch.as_tensor(values[index], device=device)
            m = torch.as_tensor(masks[index], device=device)
            y = torch.as_tensor(targets[index], device=device)
            c = torch.as_tensor(context[index], device=device) if context is not None else None
            prediction = model(x, m, c)
            loss = F.smooth_l1_loss(prediction, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


def train_candidate_attention(
    model: CandidateAttention,
    query_values: np.ndarray,
    query_masks: np.ndarray,
    targets: np.ndarray,
    reference_values: np.ndarray,
    reference_masks: np.ndarray,
    reference_xy: np.ndarray,
    *,
    context: np.ndarray | None,
    device: torch.device,
    epochs: int,
    batch_size: int,
    seed: int,
) -> list[float]:
    seed_all(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.2e-3, weight_decay=1.0e-5)
    reference_values_t = torch.as_tensor(reference_values, device=device)
    reference_masks_t = torch.as_tensor(reference_masks, device=device)
    reference_xy_t = torch.as_tensor(reference_xy, device=device)
    nearest = np.argmin(np.linalg.norm(targets[:, None] - reference_xy[None], axis=2), axis=1)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(query_values))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            batch = len(index)
            x = torch.as_tensor(query_values[index], device=device)
            m = torch.as_tensor(query_masks[index], device=device)
            y = torch.as_tensor(targets[index], device=device)
            labels = torch.as_tensor(nearest[index], device=device)
            c = torch.as_tensor(context[index], device=device) if context is not None else None
            prediction, logits = model(
                x, m,
                reference_values_t[None].expand(batch, -1, -1, -1),
                reference_masks_t[None].expand(batch, -1, -1),
                reference_xy_t[None].expand(batch, -1, -1), c,
            )
            loss = F.smooth_l1_loss(prediction, y) + 0.15 * F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return history


@torch.no_grad()
def pointnet_predict(
    model: PointNetRegressor,
    values: np.ndarray,
    masks: np.ndarray,
    *,
    context: np.ndarray | None,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    model.eval()
    output = []
    for start in range(0, len(values), batch_size):
        x = torch.as_tensor(values[start : start + batch_size], device=device)
        m = torch.as_tensor(masks[start : start + batch_size], device=device)
        c = torch.as_tensor(context[start : start + batch_size], device=device) if context is not None else None
        output.append(model(x, m, c).cpu().numpy())
    return np.concatenate(output)


@torch.no_grad()
def attention_predict(
    model: CandidateAttention,
    values: np.ndarray,
    masks: np.ndarray,
    reference_values: np.ndarray,
    reference_masks: np.ndarray,
    reference_xy: np.ndarray,
    *,
    context: np.ndarray | None,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    model.eval()
    output = []
    rv = torch.as_tensor(reference_values, device=device)
    rm = torch.as_tensor(reference_masks, device=device)
    rp = torch.as_tensor(reference_xy, device=device)
    for start in range(0, len(values), batch_size):
        stop = min(start + batch_size, len(values))
        batch = stop - start
        x = torch.as_tensor(values[start:stop], device=device)
        m = torch.as_tensor(masks[start:stop], device=device)
        c = torch.as_tensor(context[start:stop], device=device) if context is not None else None
        prediction, _ = model(
            x, m, rv[None].expand(batch, -1, -1, -1),
            rm[None].expand(batch, -1, -1), rp[None].expand(batch, -1, -1), c,
        )
        output.append(prediction.cpu().numpy())
    return np.concatenate(output)


def summarize_rows(rows: Sequence[dict], *, metric_key: str = "error_m") -> list[dict]:
    keys = sorted({(row["protocol"], row["condition"], row["method"]) for row in rows})
    output = []
    for protocol, condition, method in keys:
        selected = [row for row in rows if row["protocol"] == protocol and row["condition"] == condition and row["method"] == method]
        values = np.asarray([row[metric_key] for row in selected], dtype=np.float64)
        output.append({
            "protocol": protocol,
            "condition": condition,
            "method": method,
            "count": len(values),
            "mean_error_m": float(np.mean(values)),
            "rmse_m": float(np.sqrt(np.mean(values**2))),
            "median_error_m": float(np.median(values)),
            "p80_error_m": float(np.quantile(values, 0.8)),
            "p90_error_m": float(np.quantile(values, 0.9)),
            "max_error_m": float(np.max(values)),
        })
    return output
