"""Fair, multimodal reconstruction of Majid's two-room CNN protocol.

This runner never expands the printed calibration locations.  Rich challengers
may use noisy power, AoA, or array CIR from the same materialized acquisitions;
latent ray IDs, obstacle labels, and extra simulator-labelled positions are not
model inputs.  The original NIST Q-D project is unavailable, so all results are
explicitly replacement-scene evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Iterable, Sequence

import numpy as np
from scipy.linalg import expm
from scipy.optimize import linear_sum_assignment
from scipy.special import betaln
from sklearn.cluster import DBSCAN
from sklearn.ensemble import ExtraTreesRegressor
import torch
from torch import nn
from torch.nn import functional as F


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LEGACY = ROOT / "experiments" / "paper_protocol_replications"
RESEARCH_LEVEL = ROOT / "experiments" / "research_level"
RECEIVER_SOURCE_PATH = HERE / "multimodal_receiver.py"
# Capture dependencies at import time.  ``main`` vendors these exact bytes into
# the artifact before any simulation/training, so a later shared-file edit
# cannot silently change the provenance of an already running process.
FROZEN_RECEIVER_SOURCE = RECEIVER_SOURCE_PATH.read_bytes()
FROZEN_RUNNER_SOURCE = Path(__file__).resolve().read_bytes()
FROZEN_RECEIVER_SHA256 = hashlib.sha256(FROZEN_RECEIVER_SOURCE).hexdigest()
FROZEN_RUNNER_SHA256 = hashlib.sha256(FROZEN_RUNNER_SOURCE).hexdigest()
for location in (HERE, LEGACY, RESEARCH_LEVEL):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from common import rectangular_sources, stable_seed  # noqa: E402
from majdi_paper_methods import (  # noqa: E402
    mpurge_map_localize,
    published_mca_localize,
)
from evo_mdp_rank import delay_set_features  # noqa: E402
from multimodal_receiver import (  # noqa: E402
    MultimodalFingerprint,
    coherent_adp_distance,
    corrupt_stored_fingerprint,
    normalized_cir_magnitude,
    pack_paths,
    simulate_multimodal_fingerprint,
)
from run_cnn_protocol import (  # noqa: E402
    PUBLISHED_RMSE_M,
    PaperCNN,
    ROOMS,
    ap_positions,
    obstruction_configs,
    reference_grid,
)


SCHEMA = "majid-cnn-fair-multimodal-reconstruction-v2"
MAX_PATHS = 9
SNR_DB = 20.0
METHOD_FEATURES = {
    "majid_cnn_classifier": "nine observed-power-ranked delays + RP/AP coordinates",
    "majid_cnn_regressor": "nine observed-power-ranked delays + RP/AP coordinates",
    "majid_mca_eps1": "unordered delays + RP coordinates",
    "majid_mca_eps2": "unordered delays + RP coordinates",
    "corrected_mpurge_map": "unordered delays + RP coordinates",
    "symmetric_chamfer_inverse3": "unordered delays + RP coordinates",
    "symmetric_chamfer_softmax3": "unordered delays + RP coordinates",
    "subset_consensus": "unordered delays + RP coordinates",
    "graph_diffusion": "unordered delays + RP coordinates and RP adjacency",
    "evomdp_frozen_transfer": "eight delay-set discrepancies + RP coordinates",
    "range_power_chamfer": "noisy delay and noisy observed power + RP coordinates",
    "range_aoa_chamfer": "noisy delay and noisy AoA + RP coordinates",
    "dichasus_coherent_adp_8nn": "noisy 8-element complex array CIR + RP coordinates",
    "assigned_vt_inverse_consensus": "noisy delay/AoA + survey-built VT modes + coarse delay pose",
    "beta_marginal_vt_survival": "noisy delay/AoA + survey-built VT/visibility modes + RP coordinates",
    "pointnet_delay": "unordered noisy delays + AP coordinate",
    "pointnet_multimodal": "unordered noisy delay/power/AoA tokens + AP coordinate",
    "set_attention_multimodal": "unordered noisy delay/power/AoA tokens + AP coordinate",
    "candidate_pointnet_reranker": "query/reference noisy delay/power/AoA sets + candidate RP/AP coordinates + analytic score",
    "candidate_set_attention": "self-attended query/reference noisy multipath sets + candidate RP/AP coordinates + analytic score",
    "genuine_candidate_cross_attention": "query paths attending to reference paths + candidate RP/AP coordinates + analytic score",
    "caez_cir_probability_mlp": "downsampled noisy array-CIR magnitude + AP coordinate",
    "analytic_anchor_residual": "noisy multipath set + AP coordinate + corrected-MPUrge coordinate/ambiguity",
    "survival_cir_toa_extratrees": "noisy CIR survival tail, ToA, detected-path count and AP coordinate",
    "rrle_map_aware_moe": "analytic/learned expert predictions, uncertainties and disagreements",
}

# Kept as documentation only.  The former numerical row was an exact copy of
# ``assigned_vt_inverse_consensus`` and therefore was not an independent method.
METHOD_ALIASES = {
    "two_sided_vt_registration": {
        "canonical_method": "assigned_vt_inverse_consensus",
        "status": "documentation_only_not_an_independent_numerical_row",
        "reason": "the implemented survey-side VT clustering plus query-side inverse step is exactly the canonical pipeline",
    }
}

PAPER_CONDITION_PROTOCOL = "paper_condition_map"
ENVIRONMENT_BLIND_PROTOCOL = "environment_blind_map"
PROTOCOL_INDEPENDENT = "protocol_independent"
MAP_INFORMATION = {
    PAPER_CONDITION_PROTOCOL: "privileged_true_obstacle_condition_map",
    ENVIRONMENT_BLIND_PROTOCOL: "same_AP_all_conditions_selected_only_by_observed_fingerprint",
    PROTOCOL_INDEPENDENT: "independent_of_query_obstacle_condition_protocol_see_method_provenance",
}

VT_MAP_INFORMATION = "pooled_survey_built_VT_modes_no_query_condition_identity"


def map_information_for(protocol: str, method: str) -> str:
    if protocol == PROTOCOL_INDEPENDENT and method in {
        "assigned_vt_inverse_consensus",
        "beta_marginal_vt_survival",
    }:
        return VT_MAP_INFORMATION
    if protocol == PROTOCOL_INDEPENDENT:
        return "no_reference_map_and_no_obstacle_condition_identity"
    return MAP_INFORMATION[protocol]

# These sets are both executable protocol definitions and integrity-audit
# expectations.  A method is emitted in more than one bracket only when its
# actual numerical pipeline changes with the available map information.
PAPER_CONDITION_METHODS = frozenset(
    {
        "majid_mca_eps1",
        "majid_mca_eps2",
        "corrected_mpurge_map",
        "symmetric_chamfer_inverse3",
        "symmetric_chamfer_softmax3",
        "subset_consensus",
        "graph_diffusion",
        "evomdp_frozen_transfer",
        "range_power_chamfer",
        "range_aoa_chamfer",
        "dichasus_coherent_adp_8nn",
    }
)
ENVIRONMENT_BLIND_METHODS = frozenset(
    {
        *PAPER_CONDITION_METHODS,
        "candidate_pointnet_reranker",
        "candidate_set_attention",
        "genuine_candidate_cross_attention",
        "analytic_anchor_residual",
        "rrle_map_aware_moe",
    }
)
PROTOCOL_INDEPENDENT_METHODS = frozenset(
    {
        "majid_cnn_classifier",
        "majid_cnn_regressor",
        "pointnet_delay",
        "pointnet_multimodal",
        "set_attention_multimodal",
        "caez_cir_probability_mlp",
        "survival_cir_toa_extratrees",
        "assigned_vt_inverse_consensus",
        "beta_marginal_vt_survival",
    }
)
PROTOCOL_METHODS = {
    PAPER_CONDITION_PROTOCOL: PAPER_CONDITION_METHODS,
    ENVIRONMENT_BLIND_PROTOCOL: ENVIRONMENT_BLIND_METHODS,
    PROTOCOL_INDEPENDENT: PROTOCOL_INDEPENDENT_METHODS,
}
ROOM_CHECKPOINT_SCHEMA = f"{SCHEMA}-room-checkpoint-v1"

# Candidate tensors are expensive but identical for the three candidate
# architectures.  This in-process cache contains only stored calibration
# observations; it does not materialize any new simulator sample.
_CANDIDATE_TRAIN_CACHE: dict[int, tuple] = {}


@dataclass
class RoomData:
    name: str
    room: np.ndarray
    references: np.ndarray
    aps: np.ndarray
    configurations: tuple
    train_fingerprints: list[MultimodalFingerprint]
    labels: np.ndarray
    ap_ids: np.ndarray
    config_ids: np.ndarray
    test_rows: list[dict]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def materialize_fingerprint(
    room: np.ndarray,
    ap: np.ndarray,
    xy: np.ndarray,
    objects: Sequence[np.ndarray],
    namespace: str,
    *,
    n_bins: int,
) -> MultimodalFingerprint:
    sources = rectangular_sources(room, ap[None], maximum_order=2, include_floor_ceiling=True)
    return simulate_multimodal_fingerprint(
        [xy[0], xy[1], 1.2],
        sources,
        rng=np.random.default_rng(stable_seed("fair-mm", namespace)),
        maximum_paths=MAX_PATHS,
        snr_db=SNR_DB,
        bandwidth_hz=2.0e9,
        carrier_hz=60.0e9,
        n_bins=n_bins,
        detection_threshold_db=6.0,
        obstructions=objects,
        separate_transmitters=True,
    )


def build_room_data(room_name: str, quick: bool) -> RoomData:
    room = np.asarray(ROOMS[room_name], dtype=np.float64)
    references = reference_grid(room_name)
    all_aps = ap_positions(room_name)
    ap_count = 6 if quick else 60
    aps = all_aps[:ap_count]
    configurations = obstruction_configs(room)
    n_bins = 128 if quick else 512
    train_fingerprints: list[MultimodalFingerprint] = []
    labels, ap_ids, config_ids = [], [], []
    for ap_id, ap in enumerate(aps):
        for config_id, objects in enumerate(configurations):
            for rp, xy in enumerate(references):
                train_fingerprints.append(
                    materialize_fingerprint(
                        room,
                        ap,
                        xy,
                        objects,
                        f"cnn-train-{room_name}-{ap_id}-{config_id}-{rp}",
                        n_bins=n_bins,
                    )
                )
                labels.append(rp)
                ap_ids.append(ap_id)
                config_ids.append(config_id)
    rng = np.random.default_rng(stable_seed("cnn-fair-test", room_name))
    test_rows: list[dict] = []
    test_configs = (0, 1, 4, 6)
    per_condition = 24 if quick else 400
    for condition_number, config_id in enumerate(test_configs):
        chosen_aps = (
            condition_number % ap_count,
            (condition_number + max(1, ap_count // 2)) % ap_count,
        ) if quick else (
            (7 + 11 * condition_number) % ap_count,
            (41 + 7 * condition_number) % ap_count,
        )
        for query in range(per_condition):
            ap_id = int(chosen_aps[query % 2])
            xy = rng.uniform([0.05, 0.05], [room[0] - 0.05, room[1] - 0.05])
            fingerprint = materialize_fingerprint(
                room,
                aps[ap_id],
                xy,
                configurations[config_id],
                f"cnn-test-{room_name}-{config_id}-{query}",
                n_bins=n_bins,
            )
            test_rows.append(
                {
                    "condition": f"obstacles_{condition_number}",
                    "config_id": int(config_id),
                    "ap_id": ap_id,
                    "query": int(query),
                    "xy": xy,
                    "fingerprint": fingerprint,
                }
            )
    return RoomData(
        name=room_name,
        room=room,
        references=references,
        aps=aps,
        configurations=configurations,
        train_fingerprints=train_fingerprints,
        labels=np.asarray(labels, dtype=np.int64),
        ap_ids=np.asarray(ap_ids, dtype=np.int64),
        config_ids=np.asarray(config_ids, dtype=np.int64),
        test_rows=test_rows,
    )


def paper_split(data: RoomData) -> tuple[np.ndarray, np.ndarray]:
    """Paper-like 80/20 split: four of twenty RPs per AP/config are held out."""

    rng = np.random.default_rng(stable_seed("cnn-fair-split", data.name))
    fit, validation = [], []
    groups = data.ap_ids * 7 + data.config_ids
    for group in np.unique(groups):
        ids = np.flatnonzero(groups == group)
        ids = ids[rng.permutation(len(ids))]
        validation.extend(ids[:4])
        fit.extend(ids[4:])
    return np.asarray(fit, dtype=np.int64), np.asarray(validation, dtype=np.int64)


def encode_paper_cnn(
    fingerprints: Sequence[MultimodalFingerprint],
    aps: np.ndarray,
    references: np.ndarray,
    room: np.ndarray,
    delay_min: float,
    delay_max: float,
) -> np.ndarray:
    output = []
    ref_xyz = np.column_stack(
        (references[:, 0] / room[0], references[:, 1] / room[1], np.full(len(references), 1.2 / room[2]))
    ).reshape(-1)
    scale = max(float(delay_max - delay_min), 1.0e-6)
    for fingerprint, ap in zip(fingerprints, aps, strict=True):
        delays = np.zeros(MAX_PATHS, dtype=np.float32)
        count = min(MAX_PATHS, len(fingerprint))
        # Fingerprints are stored in observed-power rank, as Eq. (2) requires.
        delays[:count] = (fingerprint.ranges_m[:count] - delay_min) / scale
        ap_norm = np.asarray([ap[0] / room[0], ap[1] / room[1], ap[2] / room[2]])
        vector = np.concatenate((delays, ref_xyz, ap_norm))
        output.append(vector.reshape(1, 6, 12, order="F"))
    return np.asarray(output, dtype=np.float32)


def train_paper_cnn(
    model: PaperCNN,
    x: np.ndarray,
    labels: np.ndarray,
    targets_norm: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    device: torch.device,
    epochs: int,
    seed: int,
) -> tuple[list[dict], dict[str, torch.Tensor]]:
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    rng = np.random.default_rng(seed)
    history, best_loss, best_state = [], float("inf"), None
    for epoch in range(epochs):
        order = fit_ids[rng.permutation(len(fit_ids))]
        model.train()
        losses = []
        for start in range(0, len(order), 10):
            ids = order[start : start + 10]
            output = model(torch.as_tensor(x[ids], device=device))
            if model.head_kind == "classifier":
                loss = F.cross_entropy(output, torch.as_tensor(labels[ids], device=device))
            else:
                delta = output - torch.as_tensor(targets_norm[ids], device=device)
                # Literal Eq. (4): square root of the batch mean squared 2-D error.
                loss = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1)) + 1.0e-12)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            value = model(torch.as_tensor(x[validation_ids], device=device))
            if model.head_kind == "classifier":
                validation_loss = F.cross_entropy(value, torch.as_tensor(labels[validation_ids], device=device))
            else:
                delta = value - torch.as_tensor(targets_norm[validation_ids], device=device)
                validation_loss = torch.sqrt(torch.mean(torch.sum(delta * delta, dim=1)) + 1.0e-12)
        numeric = float(validation_loss.cpu())
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": numeric})
        if numeric < best_loss:
            best_loss = numeric
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    return history, best_state


def path_arrays(
    fingerprints: Sequence[MultimodalFingerprint], room: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    multimodal, mask = pack_paths(
        fingerprints, maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(room))
    )
    delay = multimodal[:, :, :1].copy()
    padded = np.full((len(fingerprints), MAX_PATHS), np.nan, dtype=np.float64)
    for index, fingerprint in enumerate(fingerprints):
        count = min(MAX_PATHS, len(fingerprint))
        padded[index, :count] = fingerprint.ranges_m[:count]
    return delay, multimodal, mask, padded


class SetEncoder(nn.Module):
    def __init__(self, input_dim: int, attention: bool = False, width: int = 96):
        super().__init__()
        self.attention = attention
        self.input = nn.Sequential(nn.Linear(input_dim, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        if attention:
            self.blocks = nn.ModuleList(
                [nn.MultiheadAttention(width, 4, batch_first=True) for _ in range(2)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(2)])
        self.output_dim = width * 2 + 1

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.input(values)
        if self.attention:
            # MultiheadAttention produces NaNs when every token is padding.
            # Supply one zero-valued sentinel internally, while keeping the
            # original mask for the permutation-invariant pooled output.
            safe_mask = mask.clone()
            empty = ~safe_mask.any(1)
            if torch.any(empty):
                safe_mask[empty, 0] = True
                hidden = hidden.clone()
                hidden[empty, 0] = 0.0
            for block, norm in zip(self.blocks, self.norms, strict=True):
                update, _ = block(hidden, hidden, hidden, key_padding_mask=~safe_mask, need_weights=False)
                hidden = norm(hidden + update)
        valid = mask.unsqueeze(-1)
        mean = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1)
        maximum = hidden.masked_fill(~valid, -1.0e4).max(1).values
        maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
        count = torch.log1p(mask.sum(1, keepdim=True).float())
        return torch.cat((mean, maximum, count), dim=1)


class DirectSetRegressor(nn.Module):
    def __init__(self, input_dim: int, attention: bool):
        super().__init__()
        self.encoder = SetEncoder(input_dim, attention=attention)
        self.head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 3, 192),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 2),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor, ap: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.encoder(values, mask), ap), dim=1))


class ProbabilityMLP(nn.Module):
    def __init__(self, input_dim: int, classes: int = 20):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(), nn.BatchNorm1d(256),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, classes),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.network(value)


class CandidateModel(nn.Module):
    """PointNet, self-attention, or genuine query-to-reference cross-attention."""

    def __init__(self, input_dim: int, mode: str):
        super().__init__()
        if mode not in {"pointnet", "self_attention", "cross_attention"}:
            raise ValueError(mode)
        self.mode = mode
        self.query_encoder = SetEncoder(input_dim, attention=mode == "self_attention", width=64)
        self.reference_encoder = SetEncoder(input_dim, attention=mode == "self_attention", width=64)
        if mode == "cross_attention":
            self.path = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Linear(64, 64))
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            pair_dim = 64 * 3 + 6
        else:
            pair_dim = self.query_encoder.output_dim * 3 + 6
        self.score = nn.Sequential(nn.Linear(pair_dim, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        references: torch.Tensor,
        reference_mask: torch.Tensor,
        candidate_xy: torch.Tensor,
        analytic_score: torch.Tensor,
        ap: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, candidates, paths, dimensions = references.shape
        if self.mode == "cross_attention":
            q = self.path(query)
            r = self.path(references.reshape(batch * candidates, paths, dimensions)).reshape(batch, candidates, paths, 64)
            logits = torch.einsum("bqd,bkpd->bkqp", self.q_proj(q), self.k_proj(r)) / 8.0
            logits = logits.masked_fill(~reference_mask[:, :, None, :], -1.0e4)
            attention = torch.softmax(logits, dim=-1)
            context = torch.einsum("bkqp,bkpd->bkqd", attention, self.v_proj(r))
            q_expand = q[:, None].expand(-1, candidates, -1, -1)
            pair_paths = torch.cat((q_expand, context, q_expand - context), dim=-1)
            valid = query_mask[:, None, :, None]
            pair = (pair_paths * valid).sum(2) / valid.sum(2).clamp_min(1)
        else:
            q = self.query_encoder(query, query_mask)
            r = self.reference_encoder(
                references.reshape(batch * candidates, paths, dimensions),
                reference_mask.reshape(batch * candidates, paths),
            ).reshape(batch, candidates, -1)
            q_expand = q[:, None].expand(-1, candidates, -1)
            pair = torch.cat((q_expand, r, torch.abs(q_expand - r)), dim=-1)
        safe_score = torch.nan_to_num(analytic_score, nan=20.0, posinf=20.0, neginf=0.0)
        score_scale = safe_score.unsqueeze(-1)
        extra = torch.cat((candidate_xy, score_scale, ap[:, None].expand(-1, candidates, -1)), dim=-1)
        residual = self.score(torch.cat((pair, extra), dim=-1)).squeeze(-1)
        centred = safe_score - torch.amin(safe_score, dim=1, keepdim=True)
        scale = torch.quantile(torch.clamp(centred, 0.0, 20.0), 0.5, dim=1, keepdim=True).clamp_min(0.05)
        analytic_logit = -centred / scale
        logits = analytic_logit + 0.35 * residual
        weights = torch.softmax(logits, dim=1)
        prediction = torch.sum(weights.unsqueeze(-1) * candidate_xy, dim=1)
        return prediction, logits


class ResidualModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.encoder = SetEncoder(input_dim, attention=False)
        self.network = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 3 + 2 + 2, 192), nn.ReLU(),
            nn.Linear(192, 96), nn.ReLU(), nn.Linear(96, 2), nn.Tanh(),
        )

    def forward(self, values, mask, ap, anchor, diagnostics):
        residual = self.network(torch.cat((self.encoder(values, mask), ap, anchor, diagnostics), dim=1))
        return anchor + 0.25 * residual


class Router(nn.Module):
    def __init__(self, experts: int):
        super().__init__()
        feature_dim = experts * 2 + experts + experts * (experts - 1) // 2
        self.network = nn.Sequential(nn.Linear(feature_dim, 96), nn.ReLU(), nn.Linear(96, 48), nn.ReLU(), nn.Linear(48, experts))

    def forward(self, predictions: torch.Tensor, uncertainties: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        disagreements = []
        for left in range(predictions.shape[1]):
            for right in range(left + 1, predictions.shape[1]):
                disagreements.append(torch.linalg.norm(predictions[:, left] - predictions[:, right], dim=1, keepdim=True))
        feature = torch.cat((predictions.flatten(1), uncertainties, *disagreements), dim=1)
        weights = torch.softmax(self.network(feature), dim=1)
        return torch.sum(weights.unsqueeze(-1) * predictions, dim=1), weights


def train_direct_models(
    model_factory,
    values: np.ndarray,
    mask: np.ndarray,
    ap_context: np.ndarray,
    targets_norm: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    *,
    device: torch.device,
    seeds: Sequence[int],
    epochs: int,
    checkpoint_dir: Path,
    name: str,
) -> tuple[list[nn.Module], list[dict]]:
    models, histories = [], []
    for seed in seeds:
        seed_everything(seed)
        model = model_factory().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=1.0e-5)
        rng = np.random.default_rng(seed)
        best, state, history = float("inf"), None, []
        for epoch in range(epochs):
            order = fit_ids[rng.permutation(len(fit_ids))]
            model.train()
            losses = []
            for start in range(0, len(order), 256):
                ids = order[start : start + 256]
                prediction = model(
                    torch.as_tensor(values[ids], device=device),
                    torch.as_tensor(mask[ids], device=device),
                    torch.as_tensor(ap_context[ids], device=device),
                )
                loss = F.smooth_l1_loss(prediction, torch.as_tensor(targets_norm[ids], device=device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                validation = model(
                    torch.as_tensor(values[validation_ids], device=device),
                    torch.as_tensor(mask[validation_ids], device=device),
                    torch.as_tensor(ap_context[validation_ids], device=device),
                )
                validation_loss = F.smooth_l1_loss(
                    validation, torch.as_tensor(targets_norm[validation_ids], device=device)
                )
            numeric = float(validation_loss.cpu())
            history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": numeric})
            if numeric < best:
                best = numeric
                state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert state is not None
        model.load_state_dict(state)
        checkpoint = checkpoint_dir / f"{name}_seed{seed}.pt"
        torch.save({"state_dict": state, "seed": seed, "history": history}, checkpoint)
        models.append(model)
        histories.append({"seed": seed, "best_validation_loss": best, "epochs": history})
    return models, histories


@torch.no_grad()
def ensemble_direct(models, values, mask, context, device, room) -> np.ndarray:
    predictions = []
    for model in models:
        model.eval()
        blocks = []
        for start in range(0, len(values), 512):
            stop = min(start + 512, len(values))
            blocks.append(
                model(
                    torch.as_tensor(values[start:stop], device=device),
                    torch.as_tensor(mask[start:stop], device=device),
                    torch.as_tensor(context[start:stop], device=device),
                ).cpu().numpy()
            )
        predictions.append(np.concatenate(blocks) * room[:2])
    return np.mean(predictions, axis=0)


def inverse_decode(scores: np.ndarray, positions: np.ndarray, k: int = 3) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return np.mean(positions, axis=0)
    chosen = finite[np.argsort(scores[finite], kind="stable")[: min(k, len(finite))]]
    weights = 1.0 / np.maximum(scores[chosen], 1.0e-9)
    weights /= weights.sum()
    return np.sum(positions[chosen] * weights[:, None], axis=0)


def softmax_decode(scores: np.ndarray, positions: np.ndarray, k: int = 3, temperature: float = 0.03) -> np.ndarray:
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return np.mean(positions, axis=0)
    chosen = finite[np.argsort(scores[finite], kind="stable")[: min(k, len(finite))]]
    relative = scores[chosen] - scores[chosen[0]]
    weights = np.exp(-np.clip(relative / max(temperature, 1.0e-9), 0.0, 700.0))
    weights /= weights.sum()
    return np.sum(positions[chosen] * weights[:, None], axis=0)


def chamfer_score(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if not len(a) or not len(b):
        return float("inf")
    residual = np.abs(a[:, None] - b[None, :])
    return 0.5 * (float(np.mean(np.min(residual, axis=1))) + float(np.mean(np.min(residual, axis=0))))


def multimodal_chamfer(query: MultimodalFingerprint, reference: MultimodalFingerprint, mode: str) -> float:
    if not len(query) or not len(reference):
        return float("inf")
    range_difference = np.abs(query.ranges_m[:, None] - reference.ranges_m[None, :])
    if mode == "power":
        power_difference = np.abs(query.powers_db[:, None] - reference.powers_db[None, :]) / 8.0
        cost = range_difference + power_difference
    elif mode == "aoa":
        cosine = np.clip(query.aoa_unit @ reference.aoa_unit.T, -1.0, 1.0)
        angle = np.arccos(cosine)
        cost = range_difference + 1.5 * angle
    else:
        raise ValueError(mode)
    return 0.5 * (float(np.mean(np.min(cost, axis=1))) + float(np.mean(np.min(cost, axis=0))))


def mca_similarity(query: np.ndarray, reference: np.ndarray, epsilon: float) -> float:
    if not len(query) or not len(reference):
        return 0.0
    nearest = np.min(np.abs(query[:, None] - reference[None, :]), axis=1)
    accepted = nearest < epsilon
    return float(np.sum((epsilon - nearest[accepted]) ** 2))


def graph_diffusion_prediction(scores: np.ndarray, positions: np.ndarray) -> np.ndarray:
    finite_scores = np.nan_to_num(scores, nan=1.0e6, posinf=1.0e6)
    ranks = np.argsort(np.argsort(finite_scores, kind="stable"), kind="stable") / max(len(scores) - 1, 1)
    likelihood = np.exp(-2.0 * ranks)
    distance = np.linalg.norm(positions[:, None] - positions[None, :], axis=2)
    positive = distance[distance > 0]
    spacing = float(np.min(positive)) if len(positive) else 1.0
    adjacency = np.zeros_like(distance)
    for row in range(len(positions)):
        neighbours = np.argsort(distance[row], kind="stable")[1:3]
        adjacency[row, neighbours] = np.exp(-(distance[row, neighbours] ** 2) / (2.0 * (0.75 * spacing) ** 2))
    adjacency = np.maximum(adjacency, adjacency.T)
    degree = np.sum(adjacency, axis=1)
    normalized = np.diag(1.0 / np.sqrt(np.maximum(degree, 1.0e-9)))
    laplacian = np.eye(len(positions)) - normalized @ adjacency @ normalized
    posterior = expm(-0.05 * laplacian) @ likelihood
    graph = positions[int(np.argmax(posterior))]
    baseline = inverse_decode(scores, positions, 3)
    return 0.75 * baseline + 0.25 * graph


def subset_consensus_prediction(query: np.ndarray, reference_sets: Sequence[np.ndarray], positions: np.ndarray) -> np.ndarray:
    if not len(query):
        return np.mean(positions, axis=0)
    subsets = [query]
    if len(query) <= 5:
        from itertools import combinations
        subsets.extend(query[np.asarray(item)] for item in combinations(range(len(query)), min(3, len(query))))
    else:
        subsets.extend(np.delete(query, index) for index in range(len(query)))
    hypotheses = []
    for subset in subsets:
        scores = np.asarray([chamfer_score(subset, reference) for reference in reference_sets])
        hypotheses.append(inverse_decode(scores, positions, 3))
    hypotheses = np.asarray(hypotheses)
    centre = np.median(hypotheses, axis=0)
    radius = np.linalg.norm(hypotheses - centre, axis=1)
    gate = max(0.75, 2.5 * float(np.median(radius)))
    accepted = radius <= gate
    return np.mean(hypotheses[accepted], axis=0) if np.any(accepted) else hypotheses[0]


EVO_WEIGHTS = np.asarray(
    [0.2601700413, 0.0043351388, 0.0874211727, 0.0727543336, 0.0013673587, 0.1108570387, 0.4579854288, 0.0051094875]
)
EVO_TEMPERATURE = 0.052562571


def evo_scales(data: RoomData, fit_ids: np.ndarray) -> np.ndarray:
    # Calibration-only cross-location pairs; no locked query enters scaling.
    features = []
    selected = fit_ids[:: max(1, len(fit_ids) // 600)]
    for index in selected:
        other = selected[(np.flatnonzero(selected == index)[0] + 7) % len(selected)]
        features.append(delay_set_features(data.train_fingerprints[int(index)].ranges_m, data.train_fingerprints[int(other)].ranges_m))
    values = np.asarray(features)
    scale = np.median(values, axis=0)
    return np.maximum(scale, 1.0e-3)


def evo_score(query: np.ndarray, reference: np.ndarray, scales: np.ndarray) -> float:
    return float(np.sum(EVO_WEIGHTS * delay_set_features(query, reference) / scales))


def reference_lookup(data: RoomData) -> np.ndarray:
    lookup = np.empty((len(data.aps), 7, 20), dtype=np.int64)
    for index, (ap, config, label) in enumerate(zip(data.ap_ids, data.config_ids, data.labels, strict=True)):
        lookup[int(ap), int(config), int(label)] = int(index)
    return lookup


def reference_candidate_indices(
    lookup: np.ndarray,
    ap_id: int,
    config_id: int,
    protocol: str,
    rp: int,
    *,
    allowed_reference_mask: np.ndarray | None = None,
    excluded_reference_index: int | None = None,
) -> np.ndarray:
    """Return eligible stored acquisitions for one RP without self leakage.

    The deployable bracket may search the seven stored environmental
    conditions, but it may not use their identities as a query feature.  The
    privileged paper bracket is deliberately restricted to the true condition.
    ``allowed_reference_mask`` builds a fit-only map for model selection, and
    ``excluded_reference_index`` prevents a calibration observation from being
    compared with itself when it is reused as a pseudo-query.
    """

    if protocol == PAPER_CONDITION_PROTOCOL:
        configs = (int(config_id),)
    elif protocol == ENVIRONMENT_BLIND_PROTOCOL:
        configs = tuple(range(lookup.shape[1]))
    else:
        raise ValueError(f"unknown reference-map protocol {protocol!r}")
    indices = np.asarray([lookup[int(ap_id), config, int(rp)] for config in configs], dtype=np.int64)
    if allowed_reference_mask is not None:
        mask = np.asarray(allowed_reference_mask, dtype=bool)
        if mask.shape != (int(np.max(lookup)) + 1,):
            raise ValueError("allowed_reference_mask has the wrong length")
        indices = indices[mask[indices]]
    if excluded_reference_index is not None:
        indices = indices[indices != int(excluded_reference_index)]
        assert int(excluded_reference_index) not in indices
    return indices


def choose_reference_per_rp(
    query: MultimodalFingerprint,
    data: RoomData,
    lookup: np.ndarray,
    ap_id: int,
    config_id: int,
    protocol: str,
    score_kind: str,
    scales: np.ndarray | None = None,
    *,
    allowed_reference_mask: np.ndarray | None = None,
    excluded_reference_index: int | None = None,
) -> tuple[list[MultimodalFingerprint], np.ndarray, np.ndarray]:
    references, scores, selected_indices = [], [], []
    for rp in range(20):
        candidate_indices = reference_candidate_indices(
            lookup,
            ap_id,
            config_id,
            protocol,
            rp,
            allowed_reference_mask=allowed_reference_mask,
            excluded_reference_index=excluded_reference_index,
        )
        if not len(candidate_indices):
            raise ValueError(
                f"no eligible reference for AP={ap_id}, RP={rp}, protocol={protocol}; "
                "the fit-only map must retain at least one environmental acquisition per RP"
            )
        if excluded_reference_index is not None:
            assert int(excluded_reference_index) not in candidate_indices
        if allowed_reference_mask is not None:
            assert np.all(np.asarray(allowed_reference_mask, dtype=bool)[candidate_indices])
        candidates = [data.train_fingerprints[int(index)] for index in candidate_indices]
        if score_kind == "chamfer":
            candidate_scores = [chamfer_score(query.ranges_m, item.ranges_m) for item in candidates]
        elif score_kind == "power":
            candidate_scores = [multimodal_chamfer(query, item, "power") for item in candidates]
        elif score_kind == "aoa":
            candidate_scores = [multimodal_chamfer(query, item, "aoa") for item in candidates]
        elif score_kind == "evo":
            assert scales is not None
            candidate_scores = [evo_score(query.ranges_m, item.ranges_m, scales) for item in candidates]
        elif score_kind == "adp":
            candidate_scores = [coherent_adp_distance(query.cir, item.cir) for item in candidates]
        elif score_kind == "mpurge":
            # In the environment-blind protocol, select one of the seven
            # stored condition fingerprints by an observable Chamfer screen,
            # then apply corrected MPUrge to that selected RP fingerprint.
            # This avoids both the forbidden configuration label and seven
            # redundant O(P^2) MPUrge evaluations per RP.
            screen = [chamfer_score(query.ranges_m, item.ranges_m) for item in candidates]
            screened = int(np.argmin(screen))
            candidate_scores = [float("inf")] * len(candidates)
            _, score = mpurge_map_localize(
                query.ranges_m,
                [candidates[screened].ranges_m],
                data.references[[rp]],
                p=6,
                alpha=0.7,
                k=1,
                normalized_pattern=True,
                coverage_mode="penalty",
            )
            candidate_scores[screened] = float(score[0])
        else:
            raise ValueError(score_kind)
        chosen = int(np.argmin(candidate_scores))
        references.append(candidates[chosen])
        scores.append(candidate_scores[chosen])
        selected_indices.append(int(candidate_indices[chosen]))
    selected = np.asarray(selected_indices, dtype=np.int64)
    if excluded_reference_index is not None:
        assert int(excluded_reference_index) not in selected
    if allowed_reference_mask is not None:
        assert np.all(np.asarray(allowed_reference_mask, dtype=bool)[selected])
    return references, np.asarray(scores, dtype=np.float64), selected


def mca_prediction(
    query,
    data,
    lookup,
    ap_id,
    config_id,
    protocol,
    epsilon,
    *,
    allowed_reference_mask=None,
    excluded_reference_index=None,
):
    scores = []
    for rp in range(20):
        candidates = reference_candidate_indices(
            lookup,
            ap_id,
            config_id,
            protocol,
            rp,
            allowed_reference_mask=allowed_reference_mask,
            excluded_reference_index=excluded_reference_index,
        )
        if not len(candidates):
            raise ValueError(f"no eligible MCA reference for AP={ap_id}, RP={rp}")
        scores.append(
            max(
                mca_similarity(query.ranges_m, data.train_fingerprints[int(index)].ranges_m, epsilon)
                for index in candidates
            )
        )
    return data.references[int(np.argmax(scores))]


def adp_prediction(scores: np.ndarray, positions: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    chosen = []
    for index in order:
        if all(np.linalg.norm(positions[index] - positions[prior]) >= 0.01 for prior in chosen):
            chosen.append(int(index))
        if len(chosen) == 8:
            break
    chosen = np.asarray(chosen, dtype=np.int64)
    if not len(chosen):
        return np.mean(positions, axis=0)
    local_scale = max(float(np.median(scores[chosen] - scores[chosen[0]])), 1.0e-5)
    weights = np.exp(-np.clip((scores[chosen] - scores[chosen[0]]) / (0.25 * local_scale), 0.0, 700.0))
    weights /= weights.sum()
    return np.sum(positions[chosen] * weights[:, None], axis=0)


def geometric_median(points: np.ndarray, iterations: int = 50) -> np.ndarray:
    if not len(points):
        raise ValueError("empty point set")
    estimate = np.median(points, axis=0)
    for _ in range(iterations):
        distance = np.linalg.norm(points - estimate, axis=1)
        if np.any(distance < 1.0e-8):
            return points[int(np.argmin(distance))].copy()
        weights = 1.0 / np.maximum(distance, 1.0e-8)
        following = np.sum(points * weights[:, None], axis=0) / weights.sum()
        if np.linalg.norm(following - estimate) < 1.0e-8:
            break
        estimate = following
    return estimate


def build_vt_modes(data: RoomData, lookup: np.ndarray) -> list[np.ndarray]:
    """Survey-only VT modes per AP; latent image coordinates are never used."""

    modes = []
    for ap_id in range(len(data.aps)):
        proposals = []
        for config in range(7):
            for rp, xy in enumerate(data.references):
                fp = data.train_fingerprints[int(lookup[ap_id, config, rp])]
                receiver = np.asarray([xy[0], xy[1], 1.2])
                for measured_range, direction in zip(fp.ranges_m, fp.aoa_unit, strict=True):
                    proposals.append(receiver + measured_range * direction)
        proposals = np.asarray(proposals, dtype=np.float64)
        if len(proposals) < 3:
            modes.append(np.empty((0, 4), dtype=np.float64))
            continue
        labels = DBSCAN(eps=1.0, min_samples=3).fit_predict(proposals)
        rows = []
        for label in sorted(set(labels) - {-1}):
            cluster = proposals[labels == label]
            centre = geometric_median(cluster)
            rows.append(np.r_[centre, len(cluster) / (7.0 * 20.0)])
        modes.append(np.asarray(rows, dtype=np.float64) if rows else np.empty((0, 4), dtype=np.float64))
    return modes


def inverse_vt_consensus(query: MultimodalFingerprint, modes: np.ndarray, coarse_xy: np.ndarray) -> np.ndarray:
    if not len(query) or not len(modes):
        return coarse_xy.copy()
    proposals, distances = [], []
    coarse = np.asarray([coarse_xy[0], coarse_xy[1], 1.2])
    for measured_range, direction in zip(query.ranges_m, query.aoa_unit, strict=True):
        candidate = modes[:, :3] - measured_range * direction[None, :]
        distance = np.linalg.norm(candidate - coarse[None, :], axis=1)
        selected = int(np.argmin(distance))
        if distance[selected] <= 2.0:
            proposals.append(candidate[selected])
            distances.append(distance[selected])
    if len(proposals) < 2:
        return coarse_xy.copy()
    estimate = geometric_median(np.asarray(proposals))[:2]
    # Exact no-evidence fallback and a conservative trust blend.
    activation = min(1.0, len(proposals) / 4.0)
    return (1.0 - activation) * coarse_xy + activation * estimate


def beta_vt_prediction(query: MultimodalFingerprint, modes: np.ndarray, positions: np.ndarray) -> np.ndarray:
    if not len(query) or not len(modes):
        return np.mean(positions, axis=0)
    scores = []
    expected = min(len(modes), max(3, int(np.sum(modes[:, 3] >= 0.10))))
    centres = modes[:, :3]
    for xy in positions:
        receiver = np.asarray([xy[0], xy[1], 1.2])
        proposals = receiver[None, :] + query.ranges_m[:, None] * query.aoa_unit
        cost = np.linalg.norm(proposals[:, None, :] - centres[None, :, :], axis=2)
        rows, columns = linear_sum_assignment(cost)
        accepted = cost[rows, columns] <= 1.25
        matches = int(np.sum(accepted))
        residual = float(np.sum(np.minimum(cost[rows, columns], 2.5)))
        # Integrate detection probability p under Beta(2,1).
        marginal = betaln(2.0 + matches, 1.0 + max(expected - matches, 0)) - betaln(2.0, 1.0)
        scores.append(residual - marginal + 0.5 * max(len(query) - matches, 0))
    return softmax_decode(np.asarray(scores), positions, 3, 0.25)


def survival_features(fingerprints: Sequence[MultimodalFingerprint], aps: np.ndarray, room: np.ndarray) -> np.ndarray:
    rows = []
    for fingerprint, ap in zip(fingerprints, aps, strict=True):
        energy = np.sum(np.abs(fingerprint.cir) ** 2, axis=(0, 1)).astype(np.float64)
        energy /= max(float(np.sum(energy)), 1.0e-12)
        survival = np.cumsum(energy[::-1])[::-1]
        index = np.linspace(0, len(survival) - 1, 32).astype(np.int64)
        toa = float(fingerprint.ranges_m[0] / np.linalg.norm(room)) if len(fingerprint) else 0.0
        count = len(fingerprint) / MAX_PATHS
        rows.append(np.r_[survival[index], toa, count, ap / room])
    return np.asarray(rows, dtype=np.float32)


def caez_features(fingerprints: Sequence[MultimodalFingerprint], aps: np.ndarray, room: np.ndarray) -> np.ndarray:
    magnitude = normalized_cir_magnitude(fingerprints)
    # Average adjacent delay bins to retain the array profile with bounded memory.
    antennas = fingerprints[0].cir.shape[0] * fingerprints[0].cir.shape[1]
    bins = fingerprints[0].cir.shape[-1]
    reshaped = magnitude.reshape(len(fingerprints), antennas, bins)
    target_bins = 64 if bins >= 64 else bins
    width = bins // target_bins
    pooled = reshaped[:, :, : target_bins * width].reshape(len(fingerprints), antennas, target_bins, width).mean(-1)
    return np.column_stack((pooled.reshape(len(fingerprints), -1), aps / room)).astype(np.float32)


def summary_features(fingerprints: Sequence[MultimodalFingerprint], aps: np.ndarray, room: np.ndarray) -> np.ndarray:
    values, mask = pack_paths(fingerprints, maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(room)))
    return np.column_stack((values.reshape(len(values), -1), mask.astype(np.float32), aps / room)).astype(np.float32)


def train_probability_model(model, x, labels, fit_ids, validation_ids, device, epochs, seed):
    seed_everything(seed)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8.0e-4, weight_decay=1.0e-5)
    rng = np.random.default_rng(seed)
    best, state, history = float("inf"), None, []
    for epoch in range(epochs):
        order = fit_ids[rng.permutation(len(fit_ids))]
        model.train(); losses = []
        for start in range(0, len(order), 256):
            ids = order[start : start + 256]
            output = model(torch.as_tensor(x[ids], device=device))
            loss = F.cross_entropy(output, torch.as_tensor(labels[ids], device=device))
            optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad():
            validation = F.cross_entropy(
                model(torch.as_tensor(x[validation_ids], device=device)),
                torch.as_tensor(labels[validation_ids], device=device),
            )
        numeric = float(validation.cpu()); history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": numeric})
        if numeric < best:
            best = numeric; state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    assert state is not None
    model.load_state_dict(state)
    return model, history, state


def predict_probability(model, x, references, device):
    model.eval(); output = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            probability = torch.softmax(model(torch.as_tensor(x[start:start + 512], device=device)), dim=1)
            output.append(probability.cpu().numpy() @ references)
    return np.concatenate(output)


def candidate_tensors(
    fingerprints: Sequence[MultimodalFingerprint],
    meta: Sequence[tuple[int, int]],
    data: RoomData,
    lookup: np.ndarray,
    protocol: str,
    *,
    allowed_reference_mask: np.ndarray | None = None,
    excluded_reference_indices: Sequence[int | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    queries, query_masks = pack_paths(fingerprints, maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(data.room)))
    if excluded_reference_indices is None:
        excluded_reference_indices = [None] * len(fingerprints)
    if len(excluded_reference_indices) != len(fingerprints):
        raise ValueError("one excluded reference index is required per fingerprint")
    ref_values, ref_masks, analytic, selected_ids = [], [], [], []
    for fingerprint, (ap_id, config_id), excluded in zip(
        fingerprints, meta, excluded_reference_indices, strict=True
    ):
        selected, scores, indices = choose_reference_per_rp(
            fingerprint,
            data,
            lookup,
            ap_id,
            config_id,
            protocol,
            "mpurge",
            allowed_reference_mask=allowed_reference_mask,
            excluded_reference_index=excluded,
        )
        if excluded is not None:
            assert int(excluded) not in indices
        values, masks = pack_paths(selected, maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(data.room)))
        ref_values.append(values); ref_masks.append(masks); analytic.append(scores); selected_ids.append(indices)
    return (
        queries,
        query_masks,
        np.asarray(ref_values),
        np.asarray(ref_masks),
        np.asarray(analytic, dtype=np.float32),
        np.asarray(selected_ids, dtype=np.int64),
    )


def train_candidate_models(
    mode: str,
    data: RoomData,
    lookup: np.ndarray,
    fit_ids: np.ndarray,
    validation_ids: np.ndarray,
    ap_context: np.ndarray,
    targets_norm: np.ndarray,
    device: torch.device,
    seeds: Sequence[int],
    epochs: int,
    checkpoint_dir: Path,
) -> tuple[list[CandidateModel], list[dict]]:
    cache_key = id(data)
    if cache_key not in _CANDIDATE_TRAIN_CACHE:
        fit_reference_mask = np.zeros(len(data.train_fingerprints), dtype=bool)
        fit_reference_mask[fit_ids] = True
        # Measurement-only augmentation; no new propagation calls or locations.
        augmented, augmented_labels, augmented_meta, augmented_source_indices = [], [], [], []
        # An empty detected-path set cannot be changed by path-token noise and
        # would therefore be bit-identical to its stored map acquisition.  It
        # carries no candidate-ranking information, so omit it from reranker
        # training while retaining empty queries in validation/test fallbacks.
        informative_fit = np.asarray(
            [int(index) for index in fit_ids if len(data.train_fingerprints[int(index)]) > 0],
            dtype=np.int64,
        )
        chosen_fit = informative_fit if len(informative_fit) <= 9000 else informative_fit[:9000]
        for index in chosen_fit:
            original = data.train_fingerprints[int(index)]
            for copy in range(2):
                augmented.append(
                    corrupt_stored_fingerprint(
                        original,
                        rng=np.random.default_rng(stable_seed("cnn-candidate-augment", data.name, int(index), copy)),
                        extra_range_std_m=0.02,
                        extra_power_std_db=1.5,
                        extra_angle_std_deg=2.0,
                        dropout_probability=0.08,
                    )
                )
                augmented_labels.append(int(data.labels[index]))
                augmented_meta.append((int(data.ap_ids[index]), int(data.config_ids[index])))
                augmented_source_indices.append(int(index))
        q, qm, rv, rm, analytic, train_selected_ids = candidate_tensors(
            augmented,
            augmented_meta,
            data,
            lookup,
            ENVIRONMENT_BLIND_PROTOCOL,
            allowed_reference_mask=fit_reference_mask,
        )
        # These queries are independent measurement-noise repeats of a stored
        # fit acquisition, not the stored fingerprint itself.  Keeping that
        # measured calibration row in the candidate map is therefore both
        # physically faithful and necessary when the paper's per-AP/config
        # 80/20 split leaves only one fit configuration for an AP/RP cell.
        # This buys no location and makes no additional propagation call.
        assert all(
            not (
                np.array_equal(repeat.ranges_m, data.train_fingerprints[source].ranges_m)
                and np.array_equal(repeat.powers_db, data.train_fingerprints[source].powers_db)
                and np.array_equal(repeat.aoa_unit, data.train_fingerprints[source].aoa_unit)
                and np.array_equal(repeat.cir, data.train_fingerprints[source].cir)
            )
            for repeat, source in zip(augmented, augmented_source_indices, strict=True)
        )
        assert np.all(fit_reference_mask[train_selected_ids])
        labels = np.asarray(augmented_labels, dtype=np.int64)
        targets = data.references[labels] / data.room[:2]
        context = np.asarray([data.aps[ap] / data.room for ap, _ in augmented_meta], dtype=np.float32)
        # Independent repeat-measurement validation derived only from the
        # stored calibration acquisition.  Using the identical fingerprint as
        # both query and candidate would make a reference method validate on a
        # trivial self-match that never occurs for the off-grid test queries.
        val_fps = [
            corrupt_stored_fingerprint(
                data.train_fingerprints[int(index)],
                rng=np.random.default_rng(
                    stable_seed("cnn-candidate-validation", data.name, int(index))
                ),
                extra_range_std_m=0.02,
                extra_power_std_db=1.5,
                extra_angle_std_deg=2.0,
                dropout_probability=0.08,
            )
            for index in validation_ids
        ]
        val_meta = [(int(data.ap_ids[index]), int(data.config_ids[index])) for index in validation_ids]
        vq, vqm, vrv, vrm, vanalytic, val_selected_ids = candidate_tensors(
            val_fps,
            val_meta,
            data,
            lookup,
            ENVIRONMENT_BLIND_PROTOCOL,
            allowed_reference_mask=fit_reference_mask,
            excluded_reference_indices=[int(index) for index in validation_ids],
        )
        assert not np.any(val_selected_ids == validation_ids[:, None])
        assert np.all(fit_reference_mask[val_selected_ids])
        vtarget = data.references[data.labels[validation_ids]] / data.room[:2]
        vcontext = np.asarray([data.aps[ap] / data.room for ap, _ in val_meta], dtype=np.float32)
        _CANDIDATE_TRAIN_CACHE[cache_key] = (
            q, qm, rv, rm, analytic, labels, targets, context,
            vq, vqm, vrv, vrm, vanalytic, vtarget, vcontext,
        )
    (
        q, qm, rv, rm, analytic, labels, targets, context,
        vq, vqm, vrv, vrm, vanalytic, vtarget, vcontext,
    ) = _CANDIDATE_TRAIN_CACHE[cache_key]
    candidate_xy = (data.references / data.room[:2]).astype(np.float32)
    models, histories = [], []
    for seed in seeds:
        seed_everything(seed); model = CandidateModel(5, mode).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=9.0e-4, weight_decay=1.0e-5)
        rng = np.random.default_rng(seed); best, state, history = float("inf"), None, []
        for epoch in range(epochs):
            order = rng.permutation(len(q)); model.train(); losses = []
            for start in range(0, len(order), 128):
                ids = order[start:start + 128]
                batch_xy = np.broadcast_to(candidate_xy, (len(ids), 20, 2)).copy()
                prediction, logits = model(
                    torch.as_tensor(q[ids], device=device), torch.as_tensor(qm[ids], device=device),
                    torch.as_tensor(rv[ids], device=device), torch.as_tensor(rm[ids], device=device),
                    torch.as_tensor(batch_xy, device=device), torch.as_tensor(analytic[ids], device=device),
                    torch.as_tensor(context[ids], device=device),
                )
                loss = F.smooth_l1_loss(prediction, torch.as_tensor(targets[ids], device=device)) + 0.12 * F.cross_entropy(logits, torch.as_tensor(labels[ids], device=device))
                optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                blocks = []
                for start in range(0, len(vq), 256):
                    stop = min(start + 256, len(vq)); batch_xy = np.broadcast_to(candidate_xy, (stop - start, 20, 2)).copy()
                    pred, _ = model(
                        torch.as_tensor(vq[start:stop], device=device), torch.as_tensor(vqm[start:stop], device=device),
                        torch.as_tensor(vrv[start:stop], device=device), torch.as_tensor(vrm[start:stop], device=device),
                        torch.as_tensor(batch_xy, device=device), torch.as_tensor(vanalytic[start:stop], device=device),
                        torch.as_tensor(vcontext[start:stop], device=device),
                    ); blocks.append(pred.cpu())
                val_prediction = torch.cat(blocks)
                val_loss = F.smooth_l1_loss(val_prediction, torch.as_tensor(vtarget))
            numeric = float(val_loss); history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": numeric})
            if numeric < best:
                best = numeric; state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert state is not None; model.load_state_dict(state)
        torch.save({"state_dict": state, "seed": seed, "history": history}, checkpoint_dir / f"candidate_{mode}_seed{seed}.pt")
        models.append(model); histories.append({"seed": seed, "best_validation_loss": best, "epochs": history})
    return models, histories


@torch.no_grad()
def predict_candidates(models, fingerprints, meta, data, lookup, device, protocol):
    q, qm, rv, rm, analytic, _ = candidate_tensors(fingerprints, meta, data, lookup, protocol)
    context = np.asarray([data.aps[ap] / data.room for ap, _ in meta], dtype=np.float32)
    candidate_xy = (data.references / data.room[:2]).astype(np.float32)
    ensembles = []
    for model in models:
        model.eval(); blocks = []
        for start in range(0, len(q), 256):
            stop = min(start + 256, len(q)); batch_xy = np.broadcast_to(candidate_xy, (stop - start, 20, 2)).copy()
            prediction, _ = model(
                torch.as_tensor(q[start:stop], device=device), torch.as_tensor(qm[start:stop], device=device),
                torch.as_tensor(rv[start:stop], device=device), torch.as_tensor(rm[start:stop], device=device),
                torch.as_tensor(batch_xy, device=device), torch.as_tensor(analytic[start:stop], device=device),
                torch.as_tensor(context[start:stop], device=device),
            ); blocks.append(prediction.cpu().numpy() * data.room[:2])
        ensembles.append(np.concatenate(blocks))
    return np.mean(ensembles, axis=0)


def train_residual(
    data, lookup, values, mask, ap_context, fit_ids, validation_ids, device, seeds, epochs, checkpoint_dir
):
    # Reuse the measurement-only augmentations and analytic score tensors that
    # were constructed for the candidate models.  This prevents the residual
    # head from training on a trivial fingerprint-versus-itself analytic
    # anchor, while adding no simulator calls or labelled positions.
    cache = _CANDIDATE_TRAIN_CACHE.get(id(data))
    if cache is None:
        raise RuntimeError("candidate calibration cache must be built before residual training")
    (
        train_values, train_mask, _train_refs, _train_ref_mask, train_scores,
        _train_labels, train_targets, train_context,
        validation_values, validation_mask, _validation_refs, _validation_ref_mask,
        validation_scores, validation_targets, validation_context,
    ) = cache

    def anchors_and_diagnostics(score_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        anchor_rows, diagnostic_rows = [], []
        for scores in np.asarray(score_rows, dtype=np.float64):
            anchor_rows.append(inverse_decode(scores, data.references, 3) / data.room[:2])
            ordered = np.sort(scores[np.isfinite(scores)])
            diagnostic_rows.append(
                [
                    ordered[0] if len(ordered) else 20.0,
                    ordered[1] - ordered[0] if len(ordered) > 1 else 20.0,
                ]
            )
        return (
            np.asarray(anchor_rows, dtype=np.float32),
            np.clip(np.asarray(diagnostic_rows, dtype=np.float32), 0.0, 20.0) / 20.0,
        )

    train_anchors, train_diagnostics = anchors_and_diagnostics(train_scores)
    validation_anchors, validation_diagnostics = anchors_and_diagnostics(validation_scores)
    models, histories = [], []
    for seed in seeds:
        seed_everything(seed); model = ResidualModel(train_values.shape[-1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-5); rng = np.random.default_rng(seed); best, state, history = float("inf"), None, []
        for epoch in range(epochs):
            order = rng.permutation(len(train_values)); model.train(); losses = []
            for start in range(0, len(order), 256):
                ids = order[start:start + 256]
                prediction = model(torch.as_tensor(train_values[ids], device=device), torch.as_tensor(train_mask[ids], device=device), torch.as_tensor(train_context[ids], device=device), torch.as_tensor(train_anchors[ids], device=device), torch.as_tensor(train_diagnostics[ids], device=device))
                loss = F.smooth_l1_loss(prediction, torch.as_tensor(train_targets[ids], device=device)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                prediction = model(torch.as_tensor(validation_values, device=device), torch.as_tensor(validation_mask, device=device), torch.as_tensor(validation_context, device=device), torch.as_tensor(validation_anchors, device=device), torch.as_tensor(validation_diagnostics, device=device)); loss = F.smooth_l1_loss(prediction, torch.as_tensor(validation_targets, device=device))
            numeric = float(loss.cpu()); history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_loss": numeric})
            if numeric < best: best = numeric; state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        assert state is not None; model.load_state_dict(state); torch.save({"state_dict": state, "seed": seed, "history": history}, checkpoint_dir / f"residual_seed{seed}.pt"); models.append(model); histories.append({"seed": seed, "best_validation_loss": best, "epochs": history})
    return models, histories


@torch.no_grad()
def predict_residual(models, fingerprints, meta, data, lookup, device):
    values, mask = pack_paths(fingerprints, maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(data.room)))
    context = np.asarray([data.aps[ap] / data.room for ap, _ in meta], dtype=np.float32)
    anchors, diagnostics = [], []
    for fp, (ap, config) in zip(fingerprints, meta, strict=True):
        _, scores, _ = choose_reference_per_rp(
            fp, data, lookup, ap, config, ENVIRONMENT_BLIND_PROTOCOL, "mpurge"
        )
        anchors.append(inverse_decode(scores, data.references, 3) / data.room[:2]); ordered = np.sort(scores[np.isfinite(scores)]); diagnostics.append([ordered[0] if len(ordered) else 20.0, ordered[1] - ordered[0] if len(ordered) > 1 else 20.0])
    anchors = np.asarray(anchors, dtype=np.float32); diagnostics = np.clip(np.asarray(diagnostics, dtype=np.float32), 0.0, 20.0) / 20.0
    predictions = []
    for model in models:
        model.eval(); blocks = []
        for start in range(0, len(values), 512):
            stop = min(start + 512, len(values)); blocks.append(model(torch.as_tensor(values[start:stop], device=device), torch.as_tensor(mask[start:stop], device=device), torch.as_tensor(context[start:stop], device=device), torch.as_tensor(anchors[start:stop], device=device), torch.as_tensor(diagnostics[start:stop], device=device)).cpu().numpy() * data.room[:2])
        predictions.append(np.concatenate(blocks))
    return np.mean(predictions, axis=0)


def train_router(validation_predictions, validation_uncertainties, targets, device, seed, epochs):
    seed_everything(seed); model = Router(validation_predictions.shape[1]).to(device); optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3, weight_decay=1.0e-4); p = torch.as_tensor(np.asarray(validation_predictions, dtype=np.float32), device=device); u = torch.as_tensor(np.asarray(validation_uncertainties, dtype=np.float32), device=device); y = torch.as_tensor(np.asarray(targets, dtype=np.float32), device=device); history = []
    for epoch in range(epochs):
        prediction, weights = model(p, u); loss = F.smooth_l1_loss(prediction, y) + 1.0e-3 * torch.mean(torch.sum(weights * torch.log(weights.clamp_min(1.0e-8)), dim=1)); optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if epoch % 25 == 0 or epoch == epochs - 1: history.append({"epoch": epoch + 1, "loss": float(loss.detach().cpu())})
    return model, history


def analytic_bundle(
    fp,
    data,
    lookup,
    ap,
    config,
    protocol,
    scales,
    vt_modes,
    *,
    allowed_reference_mask=None,
    excluded_reference_index=None,
):
    reference_kwargs = {
        "allowed_reference_mask": allowed_reference_mask,
        "excluded_reference_index": excluded_reference_index,
    }
    selected, chamfer, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "chamfer", **reference_kwargs
    )
    _, mpurge, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "mpurge", **reference_kwargs
    )
    _, evo, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "evo", scales, **reference_kwargs
    )
    _, power, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "power", **reference_kwargs
    )
    _, aoa, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "aoa", **reference_kwargs
    )
    _, adp, _ = choose_reference_per_rp(
        fp, data, lookup, ap, config, protocol, "adp", **reference_kwargs
    )
    coarse = inverse_decode(mpurge, data.references, 3)
    result = {
        "majid_mca_eps1": mca_prediction(
            fp, data, lookup, ap, config, protocol, 1.0, **reference_kwargs
        ),
        "majid_mca_eps2": mca_prediction(
            fp, data, lookup, ap, config, protocol, 2.0, **reference_kwargs
        ),
        "corrected_mpurge_map": coarse,
        "symmetric_chamfer_inverse3": inverse_decode(chamfer, data.references, 3),
        "symmetric_chamfer_softmax3": softmax_decode(chamfer, data.references, 3, 0.03),
        "subset_consensus": subset_consensus_prediction(fp.ranges_m, [item.ranges_m for item in selected], data.references),
        "graph_diffusion": graph_diffusion_prediction(mpurge, data.references),
        "evomdp_frozen_transfer": softmax_decode(evo, data.references, 3, EVO_TEMPERATURE),
        "range_power_chamfer": softmax_decode(power, data.references, 3, 0.15),
        "range_aoa_chamfer": softmax_decode(aoa, data.references, 3, 0.15),
        "dichasus_coherent_adp_8nn": adp_prediction(adp, data.references),
    }
    uncertainty = {
        "mpurge": float(np.partition(mpurge, 1)[1] - np.min(mpurge)) if np.sum(np.isfinite(mpurge)) > 1 else 20.0,
        "chamfer": float(np.partition(chamfer, 1)[1] - np.min(chamfer)) if np.sum(np.isfinite(chamfer)) > 1 else 20.0,
        "adp": float(np.partition(adp, 1)[1] - np.min(adp)) if np.sum(np.isfinite(adp)) > 1 else 20.0,
    }
    return result, uncertainty


def train_room(data: RoomData, device: torch.device, quick: bool, output_dir: Path) -> tuple[dict, dict]:
    started = time.perf_counter(); output_dir.mkdir(parents=True, exist_ok=True); checkpoint_dir = output_dir / "checkpoints"; checkpoint_dir.mkdir(exist_ok=True)
    fit_ids, validation_ids = paper_split(data); lookup = reference_lookup(data); scales = evo_scales(data, fit_ids); sample_aps = data.aps[data.ap_ids]; targets = data.references[data.labels].astype(np.float32); targets_norm = (targets / data.room[:2]).astype(np.float32); ap_context = (sample_aps / data.room).astype(np.float32)
    fit_reference_mask = np.zeros(len(data.train_fingerprints), dtype=bool); fit_reference_mask[fit_ids] = True
    fit_map_coverage = np.asarray([[np.sum(fit_reference_mask[lookup[ap, :, rp]]) for rp in range(20)] for ap in range(len(data.aps))], dtype=np.int64)
    assert int(np.min(fit_map_coverage)) >= 1, "fit-only environment-blind map lost every condition for an AP/RP"
    delays = np.concatenate([data.train_fingerprints[int(index)].ranges_m for index in fit_ids if len(data.train_fingerprints[int(index)])])
    delay_min, delay_max = float(np.min(delays)), float(np.max(delays)); paper_x = encode_paper_cnn(data.train_fingerprints, sample_aps, data.references, data.room, delay_min, delay_max)
    paper_epochs = 3 if quick else 100; learned_epochs = 3 if quick else 45; candidate_epochs = 2 if quick else 30; seeds = [stable_seed(data.name, "learned", value) for value in ([0] if quick else [0, 1, 2])]
    classifier, regressor = PaperCNN("classifier"), PaperCNN("regressor")
    classifier_history, classifier_state = train_paper_cnn(classifier, paper_x, data.labels, targets_norm, fit_ids, validation_ids, device=device, epochs=paper_epochs, seed=stable_seed(data.name, "paper-classifier"))
    regressor_history, regressor_state = train_paper_cnn(regressor, paper_x, data.labels, targets_norm, fit_ids, validation_ids, device=device, epochs=paper_epochs, seed=stable_seed(data.name, "paper-regressor"))
    torch.save({"state_dict": classifier_state, "history": classifier_history}, checkpoint_dir / "majid_cnn_classifier.pt"); torch.save({"state_dict": regressor_state, "history": regressor_history}, checkpoint_dir / "majid_cnn_regressor.pt")
    delay_values, multimodal_values, masks, _ = path_arrays(data.train_fingerprints, data.room)
    point_delay, hist_point_delay = train_direct_models(lambda: DirectSetRegressor(1, False), delay_values, masks, ap_context, targets_norm, fit_ids, validation_ids, device=device, seeds=seeds, epochs=learned_epochs, checkpoint_dir=checkpoint_dir, name="pointnet_delay")
    point_multi, hist_point_multi = train_direct_models(lambda: DirectSetRegressor(5, False), multimodal_values, masks, ap_context, targets_norm, fit_ids, validation_ids, device=device, seeds=seeds, epochs=learned_epochs, checkpoint_dir=checkpoint_dir, name="pointnet_multimodal")
    attention_multi, hist_attention = train_direct_models(lambda: DirectSetRegressor(5, True), multimodal_values, masks, ap_context, targets_norm, fit_ids, validation_ids, device=device, seeds=seeds, epochs=learned_epochs, checkpoint_dir=checkpoint_dir, name="set_attention_multimodal")
    survival_x = survival_features(data.train_fingerprints, sample_aps, data.room); tree = ExtraTreesRegressor(n_estimators=80 if quick else 500, min_samples_leaf=2, max_features=0.8, n_jobs=-1, random_state=stable_seed(data.name, "survival-tree")).fit(survival_x[fit_ids], targets[fit_ids])
    cir_x = caez_features(data.train_fingerprints, sample_aps, data.room); caez_models, caez_histories = [], []
    for seed in seeds:
        model, history, state = train_probability_model(ProbabilityMLP(cir_x.shape[1]), cir_x, data.labels, fit_ids, validation_ids, device, learned_epochs, seed); torch.save({"state_dict": state, "history": history}, checkpoint_dir / f"caez_seed{seed}.pt"); caez_models.append(model); caez_histories.append({"seed": seed, "epochs": history})
    candidate_models, candidate_histories = {}, {}
    for mode in ("pointnet", "self_attention", "cross_attention"):
        candidate_models[mode], candidate_histories[mode] = train_candidate_models(mode, data, lookup, fit_ids, validation_ids, ap_context, targets_norm, device, seeds, candidate_epochs, checkpoint_dir)
    residual_models, residual_histories = train_residual(data, lookup, multimodal_values, masks, ap_context, fit_ids, validation_ids, device, seeds, learned_epochs, checkpoint_dir)
    vt_modes = build_vt_modes(data, lookup)
    model_bundle = {"classifier": classifier, "regressor": regressor, "point_delay": point_delay, "point_multi": point_multi, "attention_multi": attention_multi, "tree": tree, "caez": caez_models, "candidate": candidate_models, "residual": residual_models, "vt_modes": vt_modes, "lookup": lookup, "evo_scales": scales, "delay_min": delay_min, "delay_max": delay_max, "fit_ids": fit_ids, "validation_ids": validation_ids, "fit_reference_mask": fit_reference_mask}
    audit = {"fit_samples": len(fit_ids), "validation_samples": len(validation_ids), "training_samples_total": len(data.train_fingerprints), "test_samples": len(data.test_rows), "validation_reference_policy": "fit-only map plus explicit leave-self exclusion", "candidate_training_repeat_policy": "nonempty independently corrupted repeats may retain their original stored fit-map acquisition; zero-path repeats are omitted", "candidate_training_zero_path_rows_excluded": int(sum(len(data.train_fingerprints[int(index)]) == 0 for index in fit_ids)), "fit_map_min_conditions_per_ap_rp": int(np.min(fit_map_coverage)), "fit_map_max_conditions_per_ap_rp": int(np.max(fit_map_coverage)), "paper_epochs": paper_epochs, "learned_epochs": learned_epochs, "candidate_epochs": candidate_epochs, "learned_seeds": seeds, "classifier_parameters": sum(parameter.numel() for parameter in classifier.parameters()), "regressor_parameters": sum(parameter.numel() for parameter in regressor.parameters()), "evo_scales": scales.tolist(), "vt_mode_counts_per_ap": [len(item) for item in vt_modes], "histories": {"classifier": classifier_history, "regressor": regressor_history, "pointnet_delay": hist_point_delay, "pointnet_multimodal": hist_point_multi, "set_attention": hist_attention, "candidate": candidate_histories, "caez": caez_histories, "residual": residual_histories}, "training_runtime_s": time.perf_counter() - started}
    return model_bundle, audit


def evaluate_room(data: RoomData, models: dict, device: torch.device) -> tuple[list[dict], dict]:
    tests = data.test_rows; fingerprints = [row["fingerprint"] for row in tests]; meta = [(row["ap_id"], row["config_id"]) for row in tests]; aps = np.asarray([data.aps[ap] for ap, _ in meta]); context = (aps / data.room).astype(np.float32); truths = np.asarray([row["xy"] for row in tests]); paper_x = encode_paper_cnn(fingerprints, aps, data.references, data.room, models["delay_min"], models["delay_max"]); delay, multimodal, mask, _ = path_arrays(fingerprints, data.room)
    classifier = models["classifier"].to(device).eval(); regressor = models["regressor"].to(device).eval()
    with torch.no_grad():
        class_prediction = torch.softmax(classifier(torch.as_tensor(paper_x, device=device)), dim=1).cpu().numpy() @ data.references
        regression_prediction = regressor(torch.as_tensor(paper_x, device=device)).cpu().numpy() * data.room[:2]
    protocol_independent = {
        "majid_cnn_classifier": class_prediction,
        "majid_cnn_regressor": regression_prediction,
        "pointnet_delay": ensemble_direct(models["point_delay"], delay, mask, context, device, data.room),
        "pointnet_multimodal": ensemble_direct(models["point_multi"], multimodal, mask, context, device, data.room),
        "set_attention_multimodal": ensemble_direct(models["attention_multi"], multimodal, mask, context, device, data.room),
        "survival_cir_toa_extratrees": models["tree"].predict(survival_features(fingerprints, aps, data.room)),
    }
    environment_blind_shared = {
        "candidate_pointnet_reranker": predict_candidates(models["candidate"]["pointnet"], fingerprints, meta, data, models["lookup"], device, "environment_blind_map"),
        "candidate_set_attention": predict_candidates(models["candidate"]["self_attention"], fingerprints, meta, data, models["lookup"], device, "environment_blind_map"),
        "genuine_candidate_cross_attention": predict_candidates(models["candidate"]["cross_attention"], fingerprints, meta, data, models["lookup"], device, "environment_blind_map"),
        "analytic_anchor_residual": predict_residual(models["residual"], fingerprints, meta, data, models["lookup"], device),
    }
    cx = caez_features(fingerprints, aps, data.room); protocol_independent["caez_cir_probability_mlp"] = np.mean([predict_probability(model, cx, data.references, device) for model in models["caez"]], axis=0)
    rows = []; analytic_predictions = defaultdict(list); analytic_uncertainties = defaultdict(list)
    for protocol in ("paper_condition_map", "environment_blind_map"):
        for index, (fingerprint, row) in enumerate(zip(fingerprints, tests, strict=True)):
            prediction, uncertainty = analytic_bundle(fingerprint, data, models["lookup"], row["ap_id"], row["config_id"], protocol, models["evo_scales"], models["vt_modes"])
            for method, value in prediction.items(): analytic_predictions[(protocol, method)].append(value)
            for name, value in uncertainty.items(): analytic_uncertainties[(protocol, name)].append(value)
        for method in sorted({key[1] for key in analytic_predictions if key[0] == protocol}):
            prediction = np.asarray(analytic_predictions[(protocol, method)])
            for index, (truth, source) in enumerate(zip(truths, tests, strict=True)):
                rows.append({"protocol": protocol, "map_information": map_information_for(protocol, method), "uses_privileged_condition_identity": protocol == PAPER_CONDITION_PROTOCOL, "feature_provenance": METHOD_FEATURES[method], "room": data.name, "condition": source["condition"], "query": index, "physical_query_id": f"{data.name}:{source['condition']}:{source['query']}", "ap_id": int(source["ap_id"]), "config_id": int(source["config_id"]), "method": method, "features": METHOD_FEATURES[method], "truth_xy_m": truth.tolist(), "prediction_xy_m": prediction[index].tolist(), "error_m": float(np.linalg.norm(prediction[index] - truth))})

    # Both VT estimators use one pooled survey-built map and never receive the
    # query's hidden obstruction-condition identity.  Emit each exactly once,
    # outside the oracle/deployable condition-map comparison.  The assigned-VT
    # method uses the deployable, fingerprint-selected MPUrge estimate only as
    # a coarse pose anchor.
    blind_coarse = np.asarray(
        analytic_predictions[(ENVIRONMENT_BLIND_PROTOCOL, "corrected_mpurge_map")]
    )
    protocol_independent["assigned_vt_inverse_consensus"] = np.asarray(
        [
            inverse_vt_consensus(fp, models["vt_modes"][ap], blind_coarse[index])
            for index, (fp, (ap, _config)) in enumerate(zip(fingerprints, meta, strict=True))
        ]
    )
    protocol_independent["beta_marginal_vt_survival"] = np.asarray(
        [
            beta_vt_prediction(fp, models["vt_modes"][ap], data.references)
            for fp, (ap, _config) in zip(fingerprints, meta, strict=True)
        ]
    )
    for protocol, predictions in (
        (PROTOCOL_INDEPENDENT, protocol_independent),
        (ENVIRONMENT_BLIND_PROTOCOL, environment_blind_shared),
    ):
        for method, prediction in predictions.items():
            for index, (truth, source) in enumerate(zip(truths, tests, strict=True)):
                rows.append({"protocol": protocol, "map_information": map_information_for(protocol, method), "uses_privileged_condition_identity": False, "feature_provenance": METHOD_FEATURES[method], "room": data.name, "condition": source["condition"], "query": index, "physical_query_id": f"{data.name}:{source['condition']}:{source['query']}", "ap_id": int(source["ap_id"]), "config_id": int(source["config_id"]), "method": method, "features": METHOD_FEATURES[method], "truth_xy_m": truth.tolist(), "prediction_xy_m": prediction[index].tolist(), "error_m": float(np.linalg.norm(prediction[index] - truth))})
    # RRLE is trained on the paper's 20% validation partition, using three frozen experts.
    validation_ids = models["validation_ids"]
    val_fps = [
        corrupt_stored_fingerprint(
            data.train_fingerprints[int(index)],
            rng=np.random.default_rng(stable_seed("cnn-rrle-validation", data.name, int(index))),
            extra_range_std_m=0.02,
            extra_power_std_db=1.5,
            extra_angle_std_deg=2.0,
            dropout_probability=0.08,
        )
        for index in validation_ids
    ]
    val_meta = [(int(data.ap_ids[index]), int(data.config_ids[index])) for index in validation_ids]
    val_truth = data.references[data.labels[validation_ids]]
    val_analytic, val_uncertainty = [], []
    for fp, (ap, config), validation_index in zip(val_fps, val_meta, validation_ids, strict=True):
        result, uncertainty = analytic_bundle(
            fp,
            data,
            models["lookup"],
            ap,
            config,
            ENVIRONMENT_BLIND_PROTOCOL,
            models["evo_scales"],
            models["vt_modes"],
            allowed_reference_mask=models["fit_reference_mask"],
            excluded_reference_index=int(validation_index),
        ); val_analytic.append([result["corrected_mpurge_map"], result["symmetric_chamfer_inverse3"], result["dichasus_coherent_adp_8nn"]]); val_uncertainty.append([uncertainty["mpurge"], uncertainty["chamfer"], uncertainty["adp"]])
    val_predictions = np.asarray(val_analytic, dtype=np.float32) / data.room[:2]; val_uncertainty = np.clip(np.asarray(val_uncertainty, dtype=np.float32), 0.0, 20.0) / 20.0
    router, router_history = train_router(val_predictions, val_uncertainty, val_truth.astype(np.float32) / data.room[:2], device, stable_seed(data.name, "rrle"), 30 if len(tests) < 200 else 300)
    test_experts, test_uncertainty = [], []
    for index in range(len(tests)):
        values = [np.asarray(analytic_predictions[("environment_blind_map", method)])[index] for method in ("corrected_mpurge_map", "symmetric_chamfer_inverse3", "dichasus_coherent_adp_8nn")]; test_experts.append(values); test_uncertainty.append([analytic_uncertainties[("environment_blind_map", name)][index] for name in ("mpurge", "chamfer", "adp")])
    router.eval()
    with torch.no_grad(): rrle, weights = router(torch.as_tensor(np.asarray(test_experts, dtype=np.float32) / data.room[:2].astype(np.float32), device=device), torch.as_tensor((np.clip(np.asarray(test_uncertainty, dtype=np.float32), 0.0, 20.0) / 20.0).astype(np.float32), device=device)); rrle = rrle.cpu().numpy() * data.room[:2]; weights = weights.cpu().numpy()
    for index, (truth, source) in enumerate(zip(truths, tests, strict=True)):
        rows.append({"protocol": ENVIRONMENT_BLIND_PROTOCOL, "map_information": map_information_for(ENVIRONMENT_BLIND_PROTOCOL, "rrle_map_aware_moe"), "uses_privileged_condition_identity": False, "feature_provenance": METHOD_FEATURES["rrle_map_aware_moe"], "room": data.name, "condition": source["condition"], "query": index, "physical_query_id": f"{data.name}:{source['condition']}:{source['query']}", "ap_id": int(source["ap_id"]), "config_id": int(source["config_id"]), "method": "rrle_map_aware_moe", "features": METHOD_FEATURES["rrle_map_aware_moe"], "truth_xy_m": truth.tolist(), "prediction_xy_m": rrle[index].tolist(), "error_m": float(np.linalg.norm(rrle[index] - truth))})
    return rows, {"rrle_history": router_history, "rrle_mean_weights": np.mean(weights, axis=0).tolist()}


def summarize(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows: groups[(row["protocol"], row["room"], row["method"])].append(float(row["error_m"]))
    output = []
    for (protocol, room, method), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append({"protocol": protocol, "map_information": map_information_for(protocol, method), "room": room, "method": method, "feature_provenance": METHOD_FEATURES[method], "features": METHOD_FEATURES[method], "queries": len(array), "mean_error_m": float(np.mean(array)), "rmse_m": float(np.sqrt(np.mean(array ** 2))), "median_error_m": float(np.median(array)), "p90_error_m": float(np.quantile(array, 0.9))})
    return output


def summarize_conditions(rows: Sequence[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["protocol"], row["room"], row["condition"], row["method"])].append(float(row["error_m"]))
    output = []
    for (protocol, room, condition, method), values in sorted(groups.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append({
            "protocol": protocol,
            "map_information": map_information_for(protocol, method),
            "feature_provenance": METHOD_FEATURES[method],
            "room": room,
            "condition": condition,
            "method": method,
            "queries": len(array),
            "mean_error_m": float(np.mean(array)),
            "rmse_m": float(np.sqrt(np.mean(array ** 2))),
            "median_error_m": float(np.median(array)),
            "p90_error_m": float(np.quantile(array, 0.9)),
        })
    return output


def paired_intervals(rows: Sequence[dict], bootstrap_reps: int = 2000) -> list[dict]:
    by_key = {(row["protocol"], row["room"], row["method"], row["query"]): row for row in rows}
    groups = sorted({(row["protocol"], row["room"], row["method"]) for row in rows}); output = []
    for protocol, room, method in groups:
            baseline = "majid_cnn_classifier" if room == "room_a" else "majid_cnn_regressor"
            if method == baseline: continue
            query_ids = sorted({row["query"] for row in rows if row["protocol"] == protocol and row["room"] == room and row["method"] == method})
            if not query_ids or not all((PROTOCOL_INDEPENDENT, room, baseline, query) in by_key for query in query_ids): continue
            gain = np.asarray([by_key[(PROTOCOL_INDEPENDENT, room, baseline, query)]["error_m"] - by_key[(protocol, room, method, query)]["error_m"] for query in query_ids])
            cluster_names = np.asarray([
                by_key[(PROTOCOL_INDEPENDENT, room, baseline, query)]["physical_query_id"]
                for query in query_ids
            ])
            clusters = sorted(set(cluster_names.tolist()))
            cluster_indices = [np.flatnonzero(cluster_names == cluster) for cluster in clusters]
            rng = np.random.default_rng(stable_seed("cnn-bootstrap", protocol, room, method)); samples = np.empty(bootstrap_reps)
            for repeat in range(bootstrap_reps):
                chosen = rng.integers(0, len(cluster_indices), len(cluster_indices))
                sampled = np.concatenate([cluster_indices[int(index)] for index in chosen])
                samples[repeat] = np.mean(gain[sampled])
            output.append({"protocol": protocol, "map_information": map_information_for(protocol, method), "room": room, "method": method, "baseline": baseline, "baseline_protocol": PROTOCOL_INDEPENDENT, "mean_error_reduction_m": float(np.mean(gain)), "ci95_low_m": float(np.quantile(samples, 0.025)), "ci95_high_m": float(np.quantile(samples, 0.975)), "bootstrap_unit": "physical off-grid query paired across methods/protocols", "bootstrap_clusters": len(clusters), "bootstrap_repetitions": bootstrap_reps})
    return output


def assert_no_complete_prediction_duplication(rows: Sequence[dict]) -> None:
    """Reject a numerical method copied wholesale under two protocol labels."""

    grouped: dict[tuple[str, str, str], dict[str, tuple[float, float]]] = defaultdict(dict)
    for row in rows:
        grouped[(row["room"], row["method"], row["protocol"])][row["physical_query_id"]] = tuple(
            float(value) for value in row["prediction_xy_m"]
        )
    by_method: dict[tuple[str, str], list[tuple[str, dict[str, tuple[float, float]]]]] = defaultdict(list)
    for (room, method, protocol), predictions in grouped.items():
        by_method[(room, method)].append((protocol, predictions))
    for (room, method), variants in by_method.items():
        for left_index, (left_protocol, left) in enumerate(variants):
            for right_protocol, right in variants[left_index + 1 :]:
                common = sorted(set(left) & set(right))
                if common and all(left[key] == right[key] for key in common):
                    raise AssertionError(
                        f"{room}/{method} is a complete numerical duplicate under "
                        f"{left_protocol} and {right_protocol}"
                    )


def audit_rows(rows: Sequence[dict], room_audits: dict, quick: bool) -> dict:
    expected_queries = 96 if quick else 1600
    expected_combinations = sum(len(methods) for methods in PROTOCOL_METHODS.values())
    expected_rows = len(ROOMS) * expected_queries * expected_combinations
    keys = [
        (row["protocol"], row["room"], row["method"], row["physical_query_id"])
        for row in rows
    ]
    checks = {
        "expected_row_count": len(rows) == expected_rows,
        "unique_protocol_room_method_query_keys": len(keys) == len(set(keys)),
        "all_numeric_outputs_finite": all(
            np.isfinite(float(row["error_m"]))
            and np.all(np.isfinite(row["truth_xy_m"]))
            and np.all(np.isfinite(row["prediction_xy_m"]))
            for row in rows
        ),
        "two_sided_alias_has_no_numeric_rows": not any(
            row["method"] == "two_sided_vt_registration" for row in rows
        ),
        "row_map_information_matches_method_provenance": all(
            row["map_information"] == map_information_for(row["protocol"], row["method"])
            for row in rows
        ),
        "privileged_flag_scoped_to_oracle_bracket": all(
            bool(row["uses_privileged_condition_identity"])
            == (row["protocol"] == PAPER_CONDITION_PROTOCOL)
            for row in rows
        ),
        "fit_only_maps_retain_candidates": all(
            int(audit["training"]["fit_map_min_conditions_per_ap_rp"]) >= 1
            for audit in room_audits.values()
        ),
        "permutation_invariance": all(
            audit["invariance"]["status"] == "PASS" for audit in room_audits.values()
        ),
    }
    for room in ROOMS:
        room_rows = [row for row in rows if row["room"] == room]
        for protocol, expected_methods in PROTOCOL_METHODS.items():
            observed_methods = {row["method"] for row in room_rows if row["protocol"] == protocol}
            checks[f"{room}_{protocol}_method_set"] = observed_methods == expected_methods
            counts = defaultdict(int)
            for row in room_rows:
                if row["protocol"] == protocol:
                    counts[row["method"]] += 1
            checks[f"{room}_{protocol}_query_counts"] = (
                set(counts) == expected_methods
                and all(count == expected_queries for count in counts.values())
            )
    duplication_ok = True
    try:
        assert_no_complete_prediction_duplication(rows)
    except AssertionError:
        duplication_ok = False
    checks["no_complete_prediction_vector_duplicated_across_protocols"] = duplication_ok
    audit = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "observed_rows": len(rows),
        "expected_rows": expected_rows,
        "protocol_method_counts": {
            protocol: len(methods) for protocol, methods in PROTOCOL_METHODS.items()
        },
        "unique_methods": len(set().union(*PROTOCOL_METHODS.values())),
        "queries_per_room": expected_queries,
    }
    return audit


def save_room_checkpoint(
    output_dir: Path,
    room: str,
    rows: Sequence[dict],
    audit: dict,
    quick: bool,
) -> None:
    room_dir = output_dir / room
    row_path = room_dir / "completed_rows.jsonl.gz"
    row_temp = room_dir / "completed_rows.jsonl.gz.tmp"
    with gzip.open(row_temp, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    row_temp.replace(row_path)
    metadata = {
        "schema": ROOM_CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "room": room,
        "quick": bool(quick),
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "receiver_sha256": FROZEN_RECEIVER_SHA256,
        "rows": len(rows),
        "rows_sha256": sha256_file(row_path),
        "audit": audit,
    }
    metadata_path = room_dir / "complete.json"
    metadata_temp = room_dir / "complete.json.tmp"
    metadata_temp.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    metadata_temp.replace(metadata_path)


def load_room_checkpoint(output_dir: Path, room: str, quick: bool) -> tuple[list[dict], dict] | None:
    room_dir = output_dir / room
    row_path = room_dir / "completed_rows.jsonl.gz"
    metadata_path = room_dir / "complete.json"
    if not row_path.is_file() or not metadata_path.is_file():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected = {
        "schema": ROOM_CHECKPOINT_SCHEMA,
        "status": "COMPLETE",
        "room": room,
        "quick": bool(quick),
        "runner_sha256": FROZEN_RUNNER_SHA256,
        "receiver_sha256": FROZEN_RECEIVER_SHA256,
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        return None
    if metadata.get("rows_sha256") != sha256_file(row_path):
        return None
    with gzip.open(row_path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle]
    if len(rows) != int(metadata.get("rows", -1)):
        return None
    return rows, metadata["audit"]


def invariance_audit(models: dict, data: RoomData, device: torch.device) -> dict:
    sample = data.test_rows[0]["fingerprint"]
    if len(sample) < 2: return {"status": "SKIPPED_TOO_FEW_PATHS"}
    order = np.arange(len(sample))[::-1]
    shuffled = MultimodalFingerprint(sample.ranges_m[order], sample.powers_db[order], sample.aoa_unit[order], sample.tx_ids[order], sample.cir, sample.noise_variance, sample.range_bin_m)
    aps = np.asarray([data.aps[data.test_rows[0]["ap_id"]]]); context = (aps / data.room).astype(np.float32); original_values, original_mask = pack_paths([sample], maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(data.room))); shuffled_values, shuffled_mask = pack_paths([shuffled], maximum_paths=MAX_PATHS, range_scale_m=float(np.linalg.norm(data.room)))
    result = {}
    for name, ensemble in (("pointnet_multimodal", models["point_multi"]), ("set_attention_multimodal", models["attention_multi"])):
        first = ensemble_direct(ensemble, original_values, original_mask, context, device, data.room)[0]; second = ensemble_direct(ensemble, shuffled_values, shuffled_mask, context, device, data.room)[0]; result[name] = {"permutation_delta_m": float(np.linalg.norm(first - second)), "pass": bool(np.linalg.norm(first - second) < 1.0e-5)}
    result["status"] = "PASS" if all(item["pass"] for item in result.values()) else "FAIL"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "research" / "four_paper_report" / "cnn_fair_full")
    args = parser.parse_args(); started = time.perf_counter(); args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen_dir = args.output_dir / "frozen_sources"; frozen_dir.mkdir(exist_ok=True)
    frozen_receiver_path = frozen_dir / "multimodal_receiver.py"; frozen_receiver_path.write_bytes(FROZEN_RECEIVER_SOURCE)
    frozen_runner_path = frozen_dir / "run_cnn_fair.py"; frozen_runner_path.write_bytes(FROZEN_RUNNER_SOURCE)
    assert sha256_file(frozen_receiver_path) == FROZEN_RECEIVER_SHA256
    assert sha256_file(frozen_runner_path) == FROZEN_RUNNER_SHA256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); all_rows, room_audits = [], {}
    for room_name in ROOMS:
        resumed = None if args.no_resume else load_room_checkpoint(args.output_dir, room_name, args.quick)
        if resumed is not None:
            rows, room_audit = resumed
            all_rows.extend(rows); room_audits[room_name] = room_audit
            print(f"[{room_name}] resumed {len(rows)} audited rows from exact-source checkpoint", flush=True)
            continue
        room_started = time.perf_counter(); print(f"[{room_name}] materializing noisy sparse acquisitions", flush=True); data = build_room_data(room_name, args.quick); print(f"[{room_name}] training models on {len(data.train_fingerprints)} printed calibration samples", flush=True); models, training_audit = train_room(data, device, args.quick, args.output_dir / room_name); print(f"[{room_name}] evaluating {len(data.test_rows)} locked off-grid queries", flush=True); rows, evaluation_audit = evaluate_room(data, models, device); invariance = invariance_audit(models, data, device); all_rows.extend(rows); room_audits[room_name] = {"training": training_audit, "evaluation": evaluation_audit, "invariance": invariance, "runtime_s": time.perf_counter() - room_started}; print(f"[{room_name}] complete in {room_audits[room_name]['runtime_s']:.1f}s", flush=True)
        save_room_checkpoint(args.output_dir, room_name, rows, room_audits[room_name], args.quick)
        print(f"[{room_name}] exact-source resume checkpoint committed", flush=True)
    integrity = audit_rows(all_rows, room_audits, args.quick)
    integrity_path = args.output_dir / "integrity_audit.json"
    integrity_path.write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    if integrity["status"] != "PASS":
        raise AssertionError(f"CNN integrity audit failed; see {integrity_path}")
    raw_path = args.output_dir / "raw_rows.jsonl"
    with raw_path.open("w", encoding="utf-8") as handle:
        for row in all_rows: handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    result = {"schema": SCHEMA, "status": "QUICK_SMOKE" if args.quick else "FULL_COMPLETE", "claim": "faithful where printed and otherwise frozen replacement-scene reconstruction; not an exact NIST Q-D reproduction", "fairness": {"feature_rule": "any noisy physical modality extracted jointly from the same sparse noisy array-CIR acquisition", "acquisition_unit": "one simulated noisy complex array CIR per AP-condition-position; delay, observed power, AoA and CIR are jointly derived", "always_forbidden": ["extra calibration positions", "dense simulator map", "extra labelled forward simulations", "latent ray/image-source/VT identity", "query truth"], "observable_physical_transmitter_identity": "allowed when a protocol supplies separable waveform/channel labels; not used by the current CNN feature tensors", "deployable_bracket_forbidden": ["hidden obstacle-configuration identity as a model input (retained only as an evaluation stratum)"], "environment_map_protocols": {PAPER_CONDITION_PROTOCOL: "privileged/oracle-condition paper-matched bracket: a reference-based method receives the true matching obstruction-condition map", ENVIRONMENT_BLIND_PROTOCOL: "deployable bracket: seven stored conditions are pooled and selected only through observed fingerprint similarity", PROTOCOL_INDEPENDENT: "one condition-protocol-independent numerical row; row-level map_information distinguishes no-map direct models from pooled survey-built VT maps"}}, "paper_specified": {"rooms_xyz_m": {key: value.tolist() for key, value in ROOMS.items()}, "rps": 20, "ap_positions": 60, "obstruction_configurations": 7, "fingerprints_per_room": 8400, "carrier_ghz": 60, "bandwidth_ghz": 2, "snr_db": 20, "maximum_paths": 9, "cnn_epochs": 100, "cnn_batch": 10, "optimizer": "Adam 1e-3", "locked_queries_per_room": 1600}, "published_context_rmse_m": PUBLISHED_RMSE_M, "method_features": METHOD_FEATURES, "method_aliases": METHOD_ALIASES, "protocol_methods": {key: sorted(value) for key, value in PROTOCOL_METHODS.items()}, "room_audits": room_audits, "integrity_audit": str(integrity_path.resolve()), "integrity_audit_sha256": sha256_file(integrity_path), "summaries": summarize(all_rows), "condition_summaries": summarize_conditions(all_rows), "paired_intervals": paired_intervals(all_rows, 200 if args.quick else 2000), "raw_rows": str(raw_path.resolve()), "raw_rows_sha256": sha256_file(raw_path), "source": str(frozen_runner_path.resolve()), "source_sha256": FROZEN_RUNNER_SHA256, "receiver_source": str(frozen_receiver_path.resolve()), "receiver_source_sha256": FROZEN_RECEIVER_SHA256, "live_source_hashes_at_completion": {"runner": sha256_file(Path(__file__)), "receiver": sha256_file(RECEIVER_SOURCE_PATH)}, "device": str(device), "runtime_s": time.perf_counter() - started}
    result_path = args.output_dir / "results.json"; result_path.write_text(json.dumps(result, indent=2), encoding="utf-8"); print(json.dumps({"status": result["status"], "rows": len(all_rows), "runtime_s": result["runtime_s"], "results": str(result_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
