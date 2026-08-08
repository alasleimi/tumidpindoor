"""Leakage-controlled components for the corrected SMART-LEA replacement suite.

This module intentionally lives outside the legacy reconstruction.  It exposes
only measured acquisition tensors to learned/map methods.  Ray/source identity
is retained solely in a diagnostic array for the explicitly labelled legacy
per-transmitter-fusion check.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import least_squares, linear_sum_assignment
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import NearestNeighbors
import torch
from torch import nn
from torch.nn import functional as F

from multimodal_receiver import corrupt_stored_fingerprint, simulate_multimodal_fingerprint


ROOM_XY_M = np.asarray([25.0, 25.0], dtype=np.float64)
ROOM_XYZ_M = np.asarray([25.0, 25.0, 3.0], dtype=np.float64)
RECEIVER_Z_M = 1.2
TRANSMITTERS_XYZ_M = np.asarray(
    [[3.1, 4.2, 2.5], [20.8, 4.7, 2.5], [5.3, 20.2, 2.5], [19.1, 19.7, 2.5]],
    dtype=np.float64,
)
SPACINGS_M = (0.25, 0.5, 1.5, 2.5)
DENSE_STEP_M = 0.25
MAX_PATHS = 20
RANGE_SCALE_M = 75.0


def stable_seed(*parts: object) -> int:
    payload = "smart-fair-v2:" + ":".join(map(str, parts))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "little")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ImageSource:
    xyz_m: np.ndarray
    tx_id: int
    order: int


@dataclass(frozen=True)
class Corruption:
    name: str
    range_sigma_m: float
    power_sigma_db: float
    angle_sigma_deg: float
    path_dropout_probability: float = 0.0


CONDITIONS = (
    Corruption("clean", 0.0, 0.0, 0.0, 0.0),
    Corruption("matched_noisy", 0.25, 2.0, 5.0, 0.0),
)


@dataclass
class Acquisition:
    positions_xy_m: np.ndarray
    ranges_m: np.ndarray
    powers_db: np.ndarray
    aoa_unit: np.ndarray
    cir_features: np.ndarray
    mask: np.ndarray
    # Diagnostic-only.  It must never be returned by pack_tokens/features.
    diagnostic_tx_ids: np.ndarray

    def subset(self, indices: np.ndarray) -> "Acquisition":
        indices = np.asarray(indices, dtype=np.int64)
        return Acquisition(
            self.positions_xy_m[indices],
            self.ranges_m[indices],
            self.powers_db[indices],
            self.aoa_unit[indices],
            self.cir_features[indices],
            self.mask[indices],
            self.diagnostic_tx_ids[indices],
        )

    def public_arrays(self) -> dict[str, np.ndarray]:
        """Arrays available to eligible models; excludes simulator identities."""

        return {
            "positions_xy_m": self.positions_xy_m,
            "ranges_m": self.ranges_m,
            "powers_db": self.powers_db,
            "aoa_unit": self.aoa_unit,
            "cir_features": self.cir_features,
            "mask": self.mask,
        }


def rectangular_first_order_sources() -> tuple[ImageSource, ...]:
    """Four physical transmitters plus four wall images per transmitter."""

    output: list[ImageSource] = []
    for tx_id, transmitter in enumerate(TRANSMITTERS_XYZ_M):
        output.append(ImageSource(transmitter.copy(), tx_id, 0))
        for axis, coordinate in ((0, 0.0), (0, 25.0), (1, 0.0), (1, 25.0)):
            image = transmitter.copy()
            image[axis] = 2.0 * coordinate - image[axis]
            output.append(ImageSource(image, tx_id, 1))
    return tuple(output)


def dense_grid() -> np.ndarray:
    values = np.arange(0.0, 25.0, DENSE_STEP_M, dtype=np.float64)
    return np.asarray([[x, y] for y in values for x in values], dtype=np.float64)


def spacing_indices(spacing_m: float) -> np.ndarray:
    ratio = spacing_m / DENSE_STEP_M
    factor = int(round(ratio))
    if not np.isclose(ratio, factor):
        raise ValueError(f"spacing {spacing_m} is not nested in the 0.25 m survey")
    axis = np.arange(100, dtype=np.int64)
    keep = axis[axis % factor == 0]
    return np.asarray([y * 100 + x for y in keep for x in keep], dtype=np.int64)


def expected_reference_count(spacing_m: float) -> int:
    return {0.25: 10000, 0.5: 2500, 1.5: 289, 2.5: 100}[float(spacing_m)]


def query_positions(count: int, seed: int = 20260807) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform([0.05, 0.05], [24.95, 24.95], size=(count, 2)).astype(np.float64)


def _cir_features(cir: np.ndarray, bins: int = 64) -> np.ndarray:
    """Compact observable array-PDP: mean antenna magnitude, delay pooled."""

    value = np.abs(np.asarray(cir, dtype=np.complex64)).mean(axis=1)
    if value.shape[-1] % bins:
        raise ValueError("CIR bins must be divisible by pooled feature bins")
    value = value.reshape(value.shape[0], bins, value.shape[-1] // bins).mean(axis=2)
    value = np.log1p(value)
    norm = max(float(np.linalg.norm(value)), 1e-8)
    return (value / norm).reshape(-1).astype(np.float32)


def acquire(
    positions_xy_m: np.ndarray,
    sources: Sequence[ImageSource],
    condition: Corruption,
    seed: int,
) -> Acquisition:
    """Generate exactly one independently corrupted array-CIR acquisition per location.

    Range, power and AoA belong to the same extracted MPC token.  Corruption is
    applied before sorting; a dropped path removes the complete token.  No clean
    channel is copied into the public tensors.
    """

    positions = np.asarray(positions_xy_m, dtype=np.float64)
    count = len(positions)
    output_ranges = np.zeros((count, MAX_PATHS), dtype=np.float32)
    output_powers = np.zeros((count, MAX_PATHS), dtype=np.float32)
    output_directions = np.zeros((count, MAX_PATHS, 3), dtype=np.float32)
    output_cir = np.zeros((count, 4 * 64), dtype=np.float32)
    output_mask = np.zeros((count, MAX_PATHS), dtype=bool)
    output_tx = np.full((count, MAX_PATHS), -1, dtype=np.int16)
    for row in range(count):
        rng = np.random.default_rng(stable_seed(seed, "array-cir", row))
        fingerprint = simulate_multimodal_fingerprint(
            [positions[row, 0], positions[row, 1], RECEIVER_Z_M], sources,
            rng=rng, maximum_paths=MAX_PATHS,
            snr_db=80.0 if condition.name == "clean" else 20.0,
            range_noise_std_m=0.0,
            bandwidth_hz=2.0e9, carrier_hz=60.0e9, n_bins=256,
            detection_threshold_db=6.0, extra_dropout_probability=0.0,
            separate_transmitters=True,
        )
        if condition.name != "clean":
            fingerprint = corrupt_stored_fingerprint(
                fingerprint,
                rng=np.random.default_rng(stable_seed(seed, "extractor-corruption", row)),
                extra_range_std_m=condition.range_sigma_m,
                extra_power_std_db=condition.power_sigma_db,
                extra_angle_std_deg=condition.angle_sigma_deg,
                dropout_probability=condition.path_dropout_probability,
            )
        n = min(len(fingerprint), MAX_PATHS)
        output_ranges[row, :n] = fingerprint.ranges_m[:n]
        output_powers[row, :n] = fingerprint.powers_db[:n]
        output_directions[row, :n] = fingerprint.aoa_unit[:n]
        output_mask[row, :n] = True
        output_tx[row, :n] = fingerprint.tx_ids[:n]
        output_cir[row] = _cir_features(fingerprint.cir)
    return Acquisition(
        positions.astype(np.float32), output_ranges, output_powers,
        output_directions, output_cir, output_mask, output_tx,
    )


def save_acquisition(path: Path, acquisition: Acquisition, *, include_diagnostic: bool = True) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = acquisition.public_arrays()
    if include_diagnostic:
        arrays["diagnostic_tx_ids"] = acquisition.diagnostic_tx_ids
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def load_acquisition(path: Path) -> Acquisition:
    with np.load(path, allow_pickle=False) as payload:
        return Acquisition(
            payload["positions_xy_m"], payload["ranges_m"], payload["powers_db"],
            payload["aoa_unit"], payload["cir_features"], payload["mask"],
            payload["diagnostic_tx_ids"],
        )


def acquisition_digest(acquisition: Acquisition, *, public_only: bool = True) -> str:
    digest = hashlib.sha256()
    arrays = acquisition.public_arrays()
    if not public_only:
        arrays = {**arrays, "diagnostic_tx_ids": acquisition.diagnostic_tx_ids}
    for key in sorted(arrays):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(arrays[key]).tobytes())
    return digest.hexdigest()


def fit_power_stats(acquisition: Acquisition) -> tuple[float, float]:
    values = acquisition.powers_db[acquisition.mask]
    if not len(values):
        return -60.0, 10.0
    return float(np.mean(values)), max(float(np.std(values)), 1.0)


def pack_tokens(
    acquisition: Acquisition,
    modality: str,
    *,
    power_stats: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Pack public measurements without any source/ray/configuration identity."""

    ranges = acquisition.ranges_m.astype(np.float32) / RANGE_SCALE_M
    mask = acquisition.mask.copy()
    if modality == "delay":
        values = ranges[..., None]
    elif modality == "range_power_aoa":
        if power_stats is None:
            raise ValueError("multimodal tokens require fit-partition power statistics")
        mean, scale = power_stats
        standardized = np.clip((acquisition.powers_db - mean) / scale, -5.0, 5.0)
        values = np.concatenate(
            (ranges[..., None], standardized[..., None], acquisition.aoa_unit), axis=2
        ).astype(np.float32)
    else:
        raise ValueError(f"unknown modality {modality!r}")
    values[~mask] = 0.0
    return values, mask


def fixed_delay_features(acquisition: Acquisition) -> np.ndarray:
    """Delay-only fixed vector for trees/MLPs; invalid entries cannot expose power."""

    ranges = acquisition.ranges_m.astype(np.float64).copy()
    ranges[~acquisition.mask] = np.nan
    filled = np.nan_to_num(ranges, nan=RANGE_SCALE_M)
    gaps = np.diff(filled, axis=1)
    count = acquisition.mask.sum(1, keepdims=True).astype(np.float64)
    safe_count = np.maximum(count, 1.0)
    mean = np.sum(np.where(acquisition.mask, ranges, 0.0), axis=1, keepdims=True) / safe_count
    centred = np.where(acquisition.mask, ranges - mean, 0.0)
    std = np.sqrt(np.sum(centred**2, axis=1, keepdims=True) / safe_count)
    output = np.concatenate((filled / RANGE_SCALE_M, gaps / RANGE_SCALE_M, count / MAX_PATHS, mean / RANGE_SCALE_M, std / RANGE_SCALE_M), axis=1)
    return output.astype(np.float32)


def spatial_fold_ids(positions_xy_m: np.ndarray, folds: int = 5) -> np.ndarray:
    positions = np.asarray(positions_xy_m)
    bx = np.minimum((positions[:, 0] // 5.0).astype(int), 4)
    by = np.minimum((positions[:, 1] // 5.0).astype(int), 4)
    return (bx + 2 * by) % folds


def query_block_ids(positions_xy_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_xy_m)
    bx = np.minimum((positions[:, 0] // 5.0).astype(int), 4)
    by = np.minimum((positions[:, 1] // 5.0).astype(int), 4)
    return by * 5 + bx


def _torch_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full suite")
    return torch.device("cuda")


@torch.inference_mode()
def score_matrices_gpu(
    query_ranges_m: np.ndarray,
    query_mask: np.ndarray,
    reference_ranges_m: np.ndarray,
    reference_mask: np.ndarray,
    *,
    epsilon_m: float = 1.0,
    batch_size: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact masked 1-D symmetric Chamfer, MCA, and ordered-Wasserstein proxy."""

    device = _torch_device()
    refs = torch.as_tensor(reference_ranges_m, device=device, dtype=torch.float32)
    rmask = torch.as_tensor(reference_mask, device=device, dtype=torch.bool)
    q_all = torch.as_tensor(query_ranges_m, device=device, dtype=torch.float32)
    qm_all = torch.as_tensor(query_mask, device=device, dtype=torch.bool)
    chamfer_parts: list[np.ndarray] = []
    mca_parts: list[np.ndarray] = []
    wasserstein_parts: list[np.ndarray] = []
    for start in range(0, len(query_ranges_m), batch_size):
        q = q_all[start : start + batch_size]
        qm = qm_all[start : start + batch_size]
        # [B, R, Qpaths, Rpaths]
        difference = torch.abs(q[:, None, :, None] - refs[None, :, None, :])
        valid = qm[:, None, :, None] & rmask[None, :, None, :]
        difference = difference.masked_fill(~valid, torch.inf)
        q_to_r = difference.amin(3)
        r_to_q = difference.amin(2)
        q_count = qm.sum(1).clamp_min(1).to(torch.float32)
        r_count = rmask.sum(1).clamp_min(1).to(torch.float32)
        q_term = torch.where(qm[:, None, :], q_to_r, 0.0).sum(2) / q_count[:, None]
        r_term = torch.where(rmask[None, :, :], r_to_q, 0.0).sum(2) / r_count[None, :]
        chamfer = 0.5 * (q_term + r_term)
        accepted = (q_to_r < epsilon_m) & qm[:, None, :]
        mca = torch.where(accepted, (epsilon_m - q_to_r) ** 2, 0.0).sum(2)
        # Equal-cardinality sorted fingerprints make aligned L1 the exact 1-D W1.
        aligned = torch.abs(q[:, None, :] - refs[None, :, :])
        aligned_valid = qm[:, None, :] & rmask[None, :, :]
        wasserstein = torch.where(aligned_valid, aligned, 0.0).sum(2) / aligned_valid.sum(2).clamp_min(1)
        empty = (qm.sum(1) == 0)[:, None] | (rmask.sum(1) == 0)[None, :]
        chamfer = chamfer.masked_fill(empty, torch.inf)
        wasserstein = wasserstein.masked_fill(empty, torch.inf)
        mca = mca.masked_fill(empty, 0.0)
        chamfer_parts.append(chamfer.cpu().numpy())
        mca_parts.append(mca.cpu().numpy())
        wasserstein_parts.append(wasserstein.cpu().numpy())
        del difference, valid, q_to_r, r_to_q, chamfer, mca, wasserstein
    return np.concatenate(chamfer_parts), np.concatenate(mca_parts), np.concatenate(wasserstein_parts)


def topk_weighted_predictions(
    scores: np.ndarray,
    positions_xy_m: np.ndarray,
    *,
    top_k: int,
    temperature: float,
) -> np.ndarray:
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    output = np.full((len(scores), 2), ROOM_XY_M / 2.0, dtype=np.float64)
    for row, values in enumerate(np.asarray(scores, dtype=np.float64)):
        finite = np.flatnonzero(np.isfinite(values))
        if not len(finite):
            continue
        chosen = finite[np.argsort(values[finite], kind="stable")[: min(top_k, len(finite))]]
        relative = values[chosen] - values[chosen[0]]
        weights = np.exp(-np.clip(relative / max(temperature, 1e-9), 0.0, 700.0))
        weights /= weights.sum()
        output[row] = np.sum(positions[chosen] * weights[:, None], axis=0)
    return output


def mca_predictions(similarity: np.ndarray, positions_xy_m: np.ndarray) -> np.ndarray:
    if not len(similarity):
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(positions_xy_m)[np.argmax(similarity, axis=1)].astype(np.float64)


def combined_scores(chamfer: np.ndarray, mca: np.ndarray, wasserstein: np.ndarray, weight: float) -> np.ndarray:
    """Per-query scale-normalized mixture; smaller is better."""

    c = np.asarray(chamfer, dtype=np.float64)
    w = np.asarray(wasserstein, dtype=np.float64)
    m = np.asarray(mca, dtype=np.float64)
    c_scale = np.nanmedian(np.where(np.isfinite(c), c, np.nan), axis=1, keepdims=True)
    w_scale = np.nanmedian(np.where(np.isfinite(w), w, np.nan), axis=1, keepdims=True)
    m_scale = np.maximum(np.max(m, axis=1, keepdims=True), 1e-8)
    c_norm = c / np.maximum(c_scale, 1e-8)
    w_norm = w / np.maximum(w_scale, 1e-8)
    m_complement = 1.0 - m / m_scale
    secondary = 0.5 * w_norm + 0.5 * m_complement
    return weight * c_norm + (1.0 - weight) * secondary


def graph_diffusion_predictions(
    scores: np.ndarray,
    positions_xy_m: np.ndarray,
    shape: tuple[int, int],
    *,
    temperature: float,
    blend: float,
    steps: int,
) -> np.ndarray:
    device = _torch_device()
    value = torch.as_tensor(scores, device=device, dtype=torch.float32)
    relative = value - value.amin(1, keepdim=True)
    posterior = torch.softmax(-relative / max(temperature, 1e-8), dim=1)
    posterior = posterior.reshape(len(scores), 1, shape[0], shape[1])
    kernel = torch.tensor(
        [[0.05, 0.15, 0.05], [0.15, 0.20, 0.15], [0.05, 0.15, 0.05]],
        device=device,
        dtype=torch.float32,
    ).reshape(1, 1, 3, 3)
    for _ in range(steps):
        spread = F.conv2d(posterior, kernel, padding=1)
        spread /= spread.sum((2, 3), keepdim=True).clamp_min(1e-12)
        posterior = (1.0 - blend) * posterior + blend * spread
    flat = posterior.reshape(len(scores), -1)
    coordinates = torch.as_tensor(positions_xy_m, device=device, dtype=torch.float32)
    return (flat @ coordinates).cpu().numpy().astype(np.float64)


def exact_local_chamfer(query: np.ndarray, reference: np.ndarray) -> float:
    query = np.asarray(query, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if not len(query) or not len(reference):
        return float("inf")
    delta = np.abs(query[:, None] - reference[None, :])
    return 0.5 * (float(np.mean(delta.min(1))) + float(np.mean(delta.min(0))))


def candidate_indices_fast(
    query: Acquisition,
    reference: Acquisition,
    candidate_count: int,
    *,
    exclude_same_indices: np.ndarray | None = None,
    prefilter_count: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """GPU aligned-delay prefilter followed by exact local Chamfer ranking."""

    device = _torch_device()
    q = torch.as_tensor(query.ranges_m, device=device)
    r = torch.as_tensor(reference.ranges_m, device=device)
    qm = torch.as_tensor(query.mask, device=device)
    rm = torch.as_tensor(reference.mask, device=device)
    all_indices: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    for start in range(0, len(query.positions_xy_m), 128):
        stop = min(start + 128, len(query.positions_xy_m))
        difference = torch.abs(q[start:stop, None, :] - r[None, :, :])
        valid = qm[start:stop, None, :] & rm[None, :, :]
        distance = torch.where(valid, difference, 0.0).sum(2) / valid.sum(2).clamp_min(1)
        if exclude_same_indices is not None:
            rows = torch.arange(stop - start, device=device)
            columns = torch.as_tensor(exclude_same_indices[start:stop], device=device)
            distance[rows, columns] = torch.inf
        rough = torch.topk(distance, k=min(prefilter_count, len(reference.positions_xy_m)), largest=False).indices.cpu().numpy()
        for local, candidates in enumerate(rough):
            row = start + local
            qvalues = query.ranges_m[row, query.mask[row]]
            scores = np.asarray([
                exact_local_chamfer(qvalues, reference.ranges_m[index, reference.mask[index]])
                for index in candidates
            ])
            order = np.argsort(scores, kind="stable")[: min(candidate_count, len(scores))]
            all_indices.append(candidates[order].astype(np.int64))
            all_scores.append(scores[order].astype(np.float32))
    return np.asarray(all_indices, dtype=np.int64), np.asarray(all_scores, dtype=np.float32)


class DelaySetEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.token = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()
        )
        self.output_dim = hidden * 2 + 1

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.token(values)
        numeric = mask[..., None].to(hidden.dtype)
        mean = (hidden * numeric).sum(1) / numeric.sum(1).clamp_min(1.0)
        maximum = hidden.masked_fill(~mask[..., None], -torch.inf).amax(1)
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        count = torch.log1p(mask.sum(1, keepdim=True).to(hidden.dtype))
        return torch.cat((mean, maximum, count), dim=1)


class PointNetLocalizer(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = DelaySetEncoder(input_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 128), nn.GELU(), nn.Dropout(0.05),
            nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 2), nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(values, mask))


class QueryReferenceAttention(nn.Module):
    """True candidate-level cross-attention: query is Q, references are K/V."""

    def __init__(self, input_dim: int, hidden: int = 64, heads: int = 4):
        super().__init__()
        self.encoder = DelaySetEncoder(input_dim, hidden // 2)
        width = self.encoder.output_dim
        self.project = nn.Linear(width, hidden)
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.logit = nn.Sequential(nn.Linear(hidden * 2 + 3, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(
        self,
        query_values: torch.Tensor,
        query_mask: torch.Tensor,
        reference_values: torch.Tensor,
        reference_mask: torch.Tensor,
        reference_xy: torch.Tensor,
        analytic_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.project(self.encoder(query_values, query_mask))
        batch, candidates, paths, features = reference_values.shape
        r = self.project(
            self.encoder(
                reference_values.reshape(batch * candidates, paths, features),
                reference_mask.reshape(batch * candidates, paths),
            )
        ).reshape(batch, candidates, -1)
        attended, _ = self.cross(q[:, None], r, r, need_weights=False)
        context = attended.expand(-1, candidates, -1)
        query_expand = q[:, None].expand(-1, candidates, -1)
        pieces = torch.cat((context, torch.abs(query_expand - r), reference_xy, analytic_scores[..., None]), dim=2)
        logits = self.logit(pieces).squeeze(2)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights[..., None] * reference_xy, dim=1), logits


class CandidatePointNetReranker(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 64):
        super().__init__()
        self.encoder = DelaySetEncoder(input_dim, hidden // 2)
        width = self.encoder.output_dim
        self.score = nn.Sequential(
            nn.Linear(width * 3 + 3, 128), nn.GELU(), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, qv, qm, rv, rm, rxy, analytic):
        q = self.encoder(qv, qm)
        batch, candidates, paths, features = rv.shape
        r = self.encoder(rv.reshape(batch * candidates, paths, features), rm.reshape(batch * candidates, paths)).reshape(batch, candidates, -1)
        qe = q[:, None].expand(-1, candidates, -1)
        logits = self.score(torch.cat((qe, r, torch.abs(qe - r), rxy, analytic[..., None]), dim=2)).squeeze(2)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights[..., None] * rxy, dim=1), logits


class CandidatePathCrossAttention(nn.Module):
    """Path-level cross-attention reranker with no positional/source encoding."""

    def __init__(self, input_dim: int, hidden: int = 48, heads: int = 4):
        super().__init__()
        self.token = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.score = nn.Sequential(nn.Linear(hidden + 3, 96), nn.GELU(), nn.Linear(96, 1))

    def forward(self, qv, qm, rv, rm, rxy, analytic):
        batch, candidates, paths, _ = rv.shape
        q = self.token(qv)[:, None].expand(-1, candidates, -1, -1).reshape(batch * candidates, paths, -1)
        qmask = qm[:, None].expand(-1, candidates, -1).reshape(batch * candidates, paths)
        r = self.token(rv).reshape(batch * candidates, paths, -1)
        rmask = rm.reshape(batch * candidates, paths)
        attended, _ = self.cross(q, r, r, key_padding_mask=~rmask, need_weights=False)
        numeric = qmask[..., None].to(attended.dtype)
        pooled = (attended * numeric).sum(1) / numeric.sum(1).clamp_min(1.0)
        pooled = pooled.reshape(batch, candidates, -1)
        logits = self.score(torch.cat((pooled, rxy, analytic[..., None]), dim=2)).squeeze(2)
        weights = torch.softmax(logits, dim=1)
        return torch.sum(weights[..., None] * rxy, dim=1), logits


class ProbabilityMapMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(input_dim, 192), nn.GELU(), nn.Dropout(0.08),
            nn.Linear(192, 192), nn.GELU(), nn.Linear(192, classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.body(features)


class AnalyticResidualMLP(nn.Module):
    def __init__(self, feature_dim: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(feature_dim + 5, 160), nn.GELU(), nn.Linear(160, 80), nn.GELU(), nn.Linear(80, 2), nn.Tanh()
        )

    def forward(self, features, anchor_xy, diagnostic):
        residual = self.body(torch.cat((features, anchor_xy, diagnostic), dim=1)) * 0.20
        return torch.clamp(anchor_xy + residual, 0.0, 1.0)


def fit_residual_model(
    model: AnalyticResidualMLP,
    features: np.ndarray,
    anchors_normalized: np.ndarray,
    diagnostic: np.ndarray,
    targets_normalized: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 256,
) -> tuple[AnalyticResidualMLP, list[float]]:
    seed_all(seed)
    device = _torch_device(); model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=2e-5)
    rng = np.random.default_rng(seed); history = []
    for _ in range(max(int(epochs), 1)):
        order = rng.permutation(len(features)); losses = []; model.train()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            prediction = model(
                torch.as_tensor(features[ids], device=device),
                torch.as_tensor(anchors_normalized[ids], device=device),
                torch.as_tensor(diagnostic[ids], device=device),
            )
            target = torch.as_tensor(targets_normalized[ids], device=device)
            loss = F.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def predict_residual_model(
    model: AnalyticResidualMLP,
    features: np.ndarray,
    anchors_normalized: np.ndarray,
    diagnostic: np.ndarray,
    batch_size: int = 512,
) -> np.ndarray:
    device = _torch_device(); model.eval(); output = []
    for start in range(0, len(features), batch_size):
        output.append(model(
            torch.as_tensor(features[start : start + batch_size], device=device),
            torch.as_tensor(anchors_normalized[start : start + batch_size], device=device),
            torch.as_tensor(diagnostic[start : start + batch_size], device=device),
        ).cpu().numpy())
    return np.concatenate(output)


def train_direct_model(
    model: nn.Module,
    train_values: np.ndarray,
    train_mask: np.ndarray,
    train_targets: np.ndarray,
    dev_values: np.ndarray,
    dev_mask: np.ndarray,
    dev_targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 256,
) -> tuple[nn.Module, int, list[dict]]:
    seed_all(seed)
    device = _torch_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    best_state = None
    best_epoch = 1
    best_dev = float("inf")
    history: list[dict] = []
    patience = max(8, epochs // 5)
    stale = 0
    for epoch in range(1, epochs + 1):
        model.train()
        order = rng.permutation(len(train_values))
        losses = []
        for start in range(0, len(order), batch_size):
            index = order[start : start + batch_size]
            prediction = model(
                torch.as_tensor(train_values[index], device=device),
                torch.as_tensor(train_mask[index], device=device),
            )
            target = torch.as_tensor(train_targets[index], device=device)
            loss = F.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        dev_prediction = predict_direct(model, dev_values, dev_mask)
        dev_error = float(np.mean(np.linalg.norm(dev_prediction - dev_targets, axis=1)))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "dev_mean_normalized_error": dev_error})
        if dev_error < best_dev - 1e-5:
            best_dev = dev_error
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, history


def fit_direct_epochs(
    model: nn.Module,
    values: np.ndarray,
    mask: np.ndarray,
    targets: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 256,
) -> tuple[nn.Module, list[float]]:
    """Retrain a selected direct architecture on every allowed calibration RP."""

    seed_all(seed)
    device = _torch_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(max(int(epochs), 1)):
        order = rng.permutation(len(values)); losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            prediction = model(
                torch.as_tensor(values[ids], device=device),
                torch.as_tensor(mask[ids], device=device),
            )
            target = torch.as_tensor(targets[ids], device=device)
            loss = F.smooth_l1_loss(prediction, target)
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def predict_direct(model: nn.Module, values: np.ndarray, mask: np.ndarray, batch_size: int = 512) -> np.ndarray:
    device = _torch_device()
    model.eval()
    output = []
    for start in range(0, len(values), batch_size):
        output.append(
            model(
                torch.as_tensor(values[start : start + batch_size], device=device),
                torch.as_tensor(mask[start : start + batch_size], device=device),
            ).cpu().numpy()
        )
    return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)


def train_candidate_model(
    model: nn.Module,
    query_values: np.ndarray,
    query_mask: np.ndarray,
    reference_values: np.ndarray,
    reference_mask: np.ndarray,
    reference_xy_normalized: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    targets_normalized: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 96,
) -> tuple[nn.Module, list[float]]:
    seed_all(seed)
    device = _torch_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1e-5)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(query_values))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            candidates = candidate_indices[ids]
            qv = torch.as_tensor(query_values[ids], device=device)
            qm = torch.as_tensor(query_mask[ids], device=device)
            rv = torch.as_tensor(reference_values[candidates], device=device)
            rm = torch.as_tensor(reference_mask[candidates], device=device)
            rxy = torch.as_tensor(reference_xy_normalized[candidates], device=device)
            analytic = torch.as_tensor(candidate_scores[ids], device=device)
            analytic = (analytic - analytic.mean(1, keepdim=True)) / analytic.std(1, keepdim=True).clamp_min(1e-5)
            target = torch.as_tensor(targets_normalized[ids], device=device)
            prediction, logits = model(qv, qm, rv, rm, rxy, analytic)
            labels = torch.argmin(torch.linalg.norm(rxy - target[:, None], dim=2), dim=1)
            loss = F.smooth_l1_loss(prediction, target) + 0.12 * F.cross_entropy(logits, labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def predict_candidate_model(
    model: nn.Module,
    query_values: np.ndarray,
    query_mask: np.ndarray,
    reference_values: np.ndarray,
    reference_mask: np.ndarray,
    reference_xy_normalized: np.ndarray,
    candidate_indices: np.ndarray,
    candidate_scores: np.ndarray,
    *,
    batch_size: int = 96,
) -> np.ndarray:
    device = _torch_device()
    model.eval()
    output = []
    for start in range(0, len(query_values), batch_size):
        stop = min(start + batch_size, len(query_values))
        candidates = candidate_indices[start:stop]
        analytic = torch.as_tensor(candidate_scores[start:stop], device=device)
        analytic = (analytic - analytic.mean(1, keepdim=True)) / analytic.std(1, keepdim=True).clamp_min(1e-5)
        prediction, _ = model(
            torch.as_tensor(query_values[start:stop], device=device),
            torch.as_tensor(query_mask[start:stop], device=device),
            torch.as_tensor(reference_values[candidates], device=device),
            torch.as_tensor(reference_mask[candidates], device=device),
            torch.as_tensor(reference_xy_normalized[candidates], device=device),
            analytic,
        )
        output.append(prediction.cpu().numpy())
    return np.concatenate(output) if output else np.empty((0, 2), dtype=np.float32)


def train_probability_mlp(
    model: ProbabilityMapMLP,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    epochs: int,
    batch_size: int = 256,
) -> tuple[ProbabilityMapMLP, list[float]]:
    seed_all(seed)
    device = _torch_device()
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=2e-5)
    rng = np.random.default_rng(seed)
    history = []
    for _ in range(epochs):
        order = rng.permutation(len(features))
        losses = []
        model.train()
        for start in range(0, len(order), batch_size):
            ids = order[start : start + batch_size]
            logits = model(torch.as_tensor(features[ids], device=device))
            loss = F.cross_entropy(logits, torch.as_tensor(labels[ids], device=device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(float(np.mean(losses)))
    return model, history


@torch.inference_mode()
def predict_probability_mlp(model: ProbabilityMapMLP, features: np.ndarray, class_positions_normalized: np.ndarray) -> np.ndarray:
    device = _torch_device()
    positions = torch.as_tensor(class_positions_normalized, device=device)
    model.eval()
    output = []
    for start in range(0, len(features), 256):
        probability = torch.softmax(model(torch.as_tensor(features[start : start + 256], device=device)), dim=1)
        output.append((probability @ positions).cpu().numpy())
    return np.concatenate(output)


def initial_vt_from_track(positions_xy_m: np.ndarray, ranges_m: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions_xy_m, dtype=np.float64)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    if len(positions) < 3:
        return np.asarray([12.5, 12.5, 1.3])
    base = 0
    a = 2.0 * (positions[1:] - positions[base])
    b = (
        np.sum(positions[1:] ** 2, axis=1) - np.sum(positions[base] ** 2)
        - (ranges[1:] ** 2 - ranges[base] ** 2)
    )
    xy, *_ = np.linalg.lstsq(a, b, rcond=None)
    h2 = np.median(ranges**2 - np.sum((positions - xy[None]) ** 2, axis=1))
    return np.asarray([xy[0], xy[1], math.sqrt(max(float(h2), 0.01))])


def fit_vt_track(positions_xy_m: np.ndarray, ranges_m: np.ndarray) -> np.ndarray:
    initial = initial_vt_from_track(positions_xy_m, ranges_m)
    initial = np.clip(initial, [-49.999, -49.999, 1e-4], [74.999, 74.999, 24.999])
    result = least_squares(
        lambda parameter: np.sqrt(
            np.sum((positions_xy_m - parameter[None, :2]) ** 2, axis=1) + parameter[2] ** 2
        ) - ranges_m,
        initial,
        bounds=([-50.0, -50.0, 0.0], [75.0, 75.0, 25.0]),
        loss="soft_l1",
        f_scale=0.25,
        max_nfev=300,
    )
    return result.x


def build_vts_mpurge_tracks(
    acquisition: Acquisition,
    pairwise_match_function,
    *,
    maximum_points: int = 400,
) -> tuple[np.ndarray, dict]:
    """Survey-only neighbor propagation with corrected MPUrge associations."""

    valid_rows = np.flatnonzero(acquisition.mask.sum(1) >= 3)
    if len(valid_rows) > maximum_points:
        # Spatial maximin-like uniform selection through a regular index stride.
        choose = np.linspace(0, len(valid_rows) - 1, maximum_points).round().astype(int)
        valid_rows = valid_rows[choose]
    positions = acquisition.positions_xy_m[valid_rows].astype(np.float64)
    neighbor_count = min(7, len(positions))
    graph = NearestNeighbors(n_neighbors=neighbor_count).fit(positions)
    distances, indices = graph.kneighbors(positions)
    row = np.repeat(np.arange(len(positions)), neighbor_count - 1)
    col = indices[:, 1:].reshape(-1)
    weights = distances[:, 1:].reshape(-1)
    adjacency = csr_matrix((weights, (row, col)), shape=(len(positions), len(positions)))
    adjacency = adjacency.minimum(adjacency.T) + adjacency.maximum(adjacency.T)
    tree = minimum_spanning_tree(adjacency).tocoo()
    neighbours: list[list[int]] = [[] for _ in range(len(positions))]
    for a, b in zip(tree.row, tree.col, strict=True):
        neighbours[int(a)].append(int(b)); neighbours[int(b)].append(int(a))
    root = int(np.argmin(np.linalg.norm(positions - ROOM_XY_M[None] / 2.0, axis=1)))
    track_for_measurement: list[np.ndarray | None] = [None] * len(positions)
    root_ranges = acquisition.ranges_m[valid_rows[root], acquisition.mask[valid_rows[root]]]
    track_for_measurement[root] = np.arange(len(root_ranges), dtype=np.int64)
    queue = [root]
    visited = {root}
    accepted_pairs = 0
    total_pairs = 0
    while queue:
        parent = queue.pop(0)
        parent_row = valid_rows[parent]
        parent_ranges = acquisition.ranges_m[parent_row, acquisition.mask[parent_row]].astype(np.float64)
        parent_tracks = track_for_measurement[parent]
        assert parent_tracks is not None
        for child in neighbours[parent]:
            if child in visited:
                continue
            visited.add(child); queue.append(child)
            child_row = valid_rows[child]
            child_ranges = acquisition.ranges_m[child_row, acquisition.mask[child_row]].astype(np.float64)
            matches = pairwise_match_function(
                parent_ranges, child_ranges, p=2, alpha=0.5,
                normalized_pattern=False, cross_mode="order",
            )
            association = np.full(len(child_ranges), -1, dtype=np.int64)
            used_parent: set[int] = set(); used_child: set[int] = set()
            for match in matches:
                if match.index_a < len(parent_tracks) and match.index_b < len(child_ranges):
                    association[match.index_b] = parent_tracks[match.index_a]
                    used_parent.add(match.index_a); used_child.add(match.index_b)
            accepted_pairs += len(used_child); total_pairs += min(len(parent_ranges), len(child_ranges))
            remaining_parent = [i for i in range(len(parent_ranges)) if i not in used_parent]
            remaining_child = [i for i in range(len(child_ranges)) if i not in used_child]
            if remaining_parent and remaining_child:
                cost = np.abs(parent_ranges[remaining_parent, None] - child_ranges[None, remaining_child])
                aa, bb = linear_sum_assignment(cost)
                for ia, ib in zip(aa, bb, strict=True):
                    association[remaining_child[int(ib)]] = parent_tracks[remaining_parent[int(ia)]]
            # New/unmatched detections do not invent simulator identities.
            next_track = int(max((int(x.max()) for x in track_for_measurement if x is not None and len(x)), default=-1) + 1)
            for index in np.flatnonzero(association < 0):
                association[index] = next_track; next_track += 1
            track_for_measurement[child] = association
    tracks: dict[int, list[tuple[np.ndarray, float]]] = {}
    for local, global_row in enumerate(valid_rows):
        association = track_for_measurement[local]
        if association is None:
            continue
        values = acquisition.ranges_m[global_row, acquisition.mask[global_row]]
        for measurement, track in zip(values, association, strict=True):
            tracks.setdefault(int(track), []).append((positions[local], float(measurement)))
    ranked = sorted(tracks.items(), key=lambda item: (-len(item[1]), item[0]))[:MAX_PATHS]
    anchors = []
    support = []
    for _, observations in ranked:
        xy = np.asarray([item[0] for item in observations], dtype=np.float64)
        ranges = np.asarray([item[1] for item in observations], dtype=np.float64)
        if len(ranges) < 6:
            continue
        estimate = fit_vt_track(xy, ranges)
        anchors.append([estimate[0], estimate[1], RECEIVER_Z_M + estimate[2]])
        support.append(len(ranges))
    return np.asarray(anchors, dtype=np.float64), {
        "survey_points_used": int(len(valid_rows)),
        "tracks_found": int(len(tracks)),
        "anchors_retained": int(len(anchors)),
        "median_anchor_support": float(np.median(support)) if support else 0.0,
        "mpurge_pair_acceptance": accepted_pairs / max(total_pairs, 1),
    }


def sinkhorn(matrix: torch.Tensor, iterations: int = 20) -> torch.Tensor:
    log_value = matrix
    for _ in range(iterations):
        log_value = log_value - torch.logsumexp(log_value, dim=2, keepdim=True)
        log_value = log_value - torch.logsumexp(log_value, dim=1, keepdim=True)
    return torch.exp(log_value)


def refine_vts_differentiable_assignment(
    initial_anchors_xyz_m: np.ndarray,
    acquisition: Acquisition,
    *,
    seed: int,
    steps: int = 180,
    maximum_points: int = 512,
) -> tuple[np.ndarray, list[float]]:
    """Survey-only soft one-to-one assignment optimization on CUDA."""

    if len(initial_anchors_xyz_m) < 3:
        return initial_anchors_xyz_m.copy(), []
    seed_all(seed)
    device = _torch_device()
    valid = np.flatnonzero(acquisition.mask.sum(1) >= len(initial_anchors_xyz_m))
    if not len(valid):
        return initial_anchors_xyz_m.copy(), []
    rng = np.random.default_rng(seed)
    if len(valid) > maximum_points:
        valid = np.sort(rng.choice(valid, maximum_points, replace=False))
    positions = torch.as_tensor(acquisition.positions_xy_m[valid], device=device)
    observed = torch.as_tensor(acquisition.ranges_m[valid, : len(initial_anchors_xyz_m)], device=device)
    initial = torch.as_tensor(initial_anchors_xyz_m, device=device, dtype=torch.float32)
    xy = nn.Parameter(initial[:, :2].clone())
    h0 = torch.clamp(torch.abs(initial[:, 2] - RECEIVER_Z_M), min=0.05)
    raw_h = nn.Parameter(torch.log(torch.expm1(h0)))
    optimizer = torch.optim.Adam((xy, raw_h), lr=0.025)
    history = []
    for step in range(steps):
        ids = torch.randint(0, len(positions), (min(128, len(positions)),), device=device)
        point = positions[ids]
        target = observed[ids]
        height = F.softplus(raw_h)
        predicted = torch.sqrt(torch.sum((point[:, None, :] - xy[None, :, :]) ** 2, dim=2) + height[None, :] ** 2 + 1e-8)
        residual = torch.abs(target[:, :, None] - predicted[:, None, :])
        assignment = sinkhorn(-residual / 0.12, iterations=16)
        data_loss = torch.sum(assignment * F.smooth_l1_loss(
            target[:, :, None].expand_as(residual), predicted[:, None, :].expand_as(residual), reduction="none", beta=0.25
        ), dim=(1, 2)).mean() / predicted.shape[1]
        tether = 2e-4 * torch.mean((xy - initial[:, :2]) ** 2)
        loss = data_loss + tether
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        with torch.no_grad():
            xy.clamp_(-50.0, 75.0); raw_h.clamp_(-5.0, 4.0)
        if step % 5 == 0 or step == steps - 1:
            history.append(float(loss.detach()))
    height = F.softplus(raw_h).detach().cpu().numpy()
    result = np.column_stack((xy.detach().cpu().numpy(), RECEIVER_Z_M + height)).astype(np.float64)
    return result, history


def vt_errors(anchors_xyz_m: np.ndarray, truth_sources: Sequence[ImageSource]) -> dict:
    if not len(anchors_xyz_m):
        return {"matched": 0, "mean_m": None, "median_m": None, "max_m": None}
    truth = np.asarray([source.xyz_m for source in truth_sources], dtype=np.float64)
    distance = np.linalg.norm(anchors_xyz_m[:, None, :] - truth[None, :, :], axis=2)
    row, col = linear_sum_assignment(distance)
    values = distance[row, col]
    return {
        "matched": int(len(values)), "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)), "max_m": float(np.max(values)),
    }


def lea_refine_fixed_height(
    initial_xy_m: np.ndarray,
    anchors_xyz_m: np.ndarray,
    measured_ranges_m: np.ndarray,
    pairwise_match_function,
    *,
    p: int,
    alpha: float,
    iterations: int,
    cross_mode: str,
) -> np.ndarray:
    position = np.asarray(initial_xy_m, dtype=np.float64).copy()
    anchors = np.asarray(anchors_xyz_m, dtype=np.float64)
    measured = np.asarray(measured_ranges_m, dtype=np.float64)
    for _ in range(iterations):
        receiver = np.asarray([position[0], position[1], RECEIVER_Z_M])
        recalculated = np.linalg.norm(anchors - receiver[None, :], axis=1)
        matches = pairwise_match_function(
            measured, recalculated, p=p, alpha=alpha,
            normalized_pattern=False, size_norm="l1", cross_mode=cross_mode,
        )
        if len(matches) < 3:
            continue
        selected = anchors[np.asarray([match.index_b for match in matches])]
        observed = np.asarray([match.value_a for match in matches])
        solution = least_squares(
            lambda xy: np.sqrt(
                np.sum((selected[:, :2] - xy[None, :]) ** 2, axis=1)
                + (selected[:, 2] - RECEIVER_Z_M) ** 2
            ) - observed,
            position,
            bounds=([0.0, 0.0], ROOM_XY_M),
            max_nfev=80,
        )
        position = solution.x
    return position


def robust_fuse(predictions: Sequence[np.ndarray]) -> np.ndarray:
    values = np.asarray(predictions, dtype=np.float64)
    if not len(values):
        return ROOM_XY_M / 2.0
    centre = np.median(values, axis=0)
    distance = np.linalg.norm(values - centre, axis=1)
    keep = distance <= max(2.5, float(np.median(distance)) * 2.5)
    return np.mean(values[keep], axis=0) if np.any(keep) else centre
