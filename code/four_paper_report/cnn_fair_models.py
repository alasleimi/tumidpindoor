"""Neural components for the fair multimodal CNN-paper reconstruction."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def masked_mean_max(encoded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.unsqueeze(-1)
    mean = (encoded * valid).sum(1) / valid.sum(1).clamp_min(1)
    maximum = encoded.masked_fill(~valid, -1.0e4).max(1).values
    maximum = torch.where(mask.any(1, keepdim=True), maximum, torch.zeros_like(maximum))
    count = torch.log1p(mask.sum(1, keepdim=True).float())
    return torch.cat((mean, maximum, count), dim=1)


class PathEncoder(nn.Module):
    def __init__(self, input_dim: int, width: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class DirectSetRegressor(nn.Module):
    """PointNet or permutation-invariant self-attention direct regressor."""

    def __init__(self, input_dim: int, *, attention: bool, context_dim: int = 3):
        super().__init__()
        self.attention = bool(attention)
        self.encoder = PathEncoder(input_dim, 64)
        if self.attention:
            self.blocks = nn.ModuleList(
                [nn.MultiheadAttention(64, 4, batch_first=True) for _ in range(2)]
            )
            self.norms = nn.ModuleList([nn.LayerNorm(64) for _ in range(2)])
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 1 + context_dim, 160),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 2),
            nn.Sigmoid(),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        if self.attention:
            for attention, norm in zip(self.blocks, self.norms, strict=True):
                update, _ = attention(
                    encoded,
                    encoded,
                    encoded,
                    key_padding_mask=~mask,
                    need_weights=False,
                )
                encoded = norm(encoded + update)
        pooled = masked_mean_max(encoded, mask)
        return self.head(torch.cat((pooled, context), dim=1))


class ProbabilityMapMLP(nn.Module):
    """CAEZ-style probability-map transfer with a protocol-specific frontend."""

    def __init__(self, input_dim: int, cells: int = 20):
        super().__init__()
        layers: list[nn.Module] = []
        current = input_dim
        for _ in range(4):
            layers.extend((nn.Linear(current, 256), nn.ReLU(), nn.Dropout(0.05)))
            current = 256
        layers.append(nn.Linear(current, cells))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class CandidateSetRanker(nn.Module):
    """PointNet-pair or genuine path-to-path cross-attention RP ranker."""

    def __init__(self, input_dim: int, *, cross_attention: bool):
        super().__init__()
        self.cross_attention = bool(cross_attention)
        self.encoder = PathEncoder(input_dim, 64)
        if self.cross_attention:
            self.query = nn.Linear(64, 64, bias=False)
            self.key = nn.Linear(64, 64, bias=False)
            self.value = nn.Linear(64, 64, bias=False)
            pair_width = (64 * 2 + 1) * 2 + 64 + 6
        else:
            pair_width = (64 * 2 + 1) * 3 + 6
        self.score = nn.Sequential(
            nn.Linear(pair_width, 192),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(192, 96),
            nn.ReLU(),
            nn.Linear(96, 1),
        )

    def forward(
        self,
        query_values: torch.Tensor,
        query_mask: torch.Tensor,
        reference_values: torch.Tensor,
        reference_mask: torch.Tensor,
        reference_xy_norm: torch.Tensor,
        context: torch.Tensor,
        analytic_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # reference tensors: [batch,candidate,path,feature]
        batch, candidates, paths, _ = reference_values.shape
        query_encoded = self.encoder(query_values)
        reference_encoded = self.encoder(reference_values.reshape(batch * candidates, paths, -1))
        reference_encoded = reference_encoded.reshape(batch, candidates, paths, 64)
        query_pool = masked_mean_max(query_encoded, query_mask)
        reference_pool = masked_mean_max(
            reference_encoded.reshape(batch * candidates, paths, 64),
            reference_mask.reshape(batch * candidates, paths),
        ).reshape(batch, candidates, -1)
        query_repeated = query_pool[:, None, :].expand(-1, candidates, -1)
        analytic = analytic_scores.unsqueeze(-1)
        coordinates = reference_xy_norm[None].expand(batch, -1, -1)
        context_repeated = context[:, None, :].expand(-1, candidates, -1)
        side = torch.cat((coordinates, context_repeated, analytic), dim=-1)

        if self.cross_attention:
            q = self.query(query_encoded)[:, None].expand(-1, candidates, -1, -1)
            k = self.key(reference_encoded)
            v = self.value(reference_encoded)
            logits = torch.einsum("brqd,brkd->brqk", q, k) / math.sqrt(q.shape[-1])
            logits = logits.masked_fill(~reference_mask[:, :, None, :], -1.0e4)
            weights = torch.softmax(logits, dim=-1)
            attended = torch.einsum("brqk,brkd->brqd", weights, v)
            qmask = query_mask[:, None, :, None]
            cross_mean = (attended * qmask).sum(2) / qmask.sum(2).clamp_min(1)
            pair = torch.cat(
                (
                    query_repeated,
                    reference_pool,
                    cross_mean,
                    torch.abs(query_repeated - reference_pool),
                    side,
                ),
                dim=-1,
            )
        else:
            pair = torch.cat(
                (
                    query_repeated,
                    reference_pool,
                    torch.abs(query_repeated - reference_pool),
                    side,
                ),
                dim=-1,
            )
        candidate_logits = self.score(pair).squeeze(-1)
        probability = torch.softmax(candidate_logits, dim=1)
        prediction = torch.sum(probability.unsqueeze(-1) * coordinates, dim=1)
        return prediction, candidate_logits


class AnalyticResidual(nn.Module):
    """Bounded residual on top of a calibration-only analytic position."""

    def __init__(self, input_dim: int, context_dim: int = 3):
        super().__init__()
        self.encoder = PathEncoder(input_dim, 64)
        self.head = nn.Sequential(
            nn.Linear(64 * 2 + 1 + context_dim + 4, 160),
            nn.ReLU(),
            nn.Dropout(0.08),
            nn.Linear(160, 96),
            nn.ReLU(),
            nn.Linear(96, 2),
            nn.Tanh(),
        )

    def forward(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
        analytic_xy_norm: torch.Tensor,
        analytic_diagnostics: torch.Tensor,
    ) -> torch.Tensor:
        pooled = masked_mean_max(self.encoder(values), mask)
        residual = self.head(
            torch.cat((pooled, context, analytic_xy_norm, analytic_diagnostics), dim=1)
        )
        # At most 20% of a room dimension in normalized coordinates.
        return torch.clamp(analytic_xy_norm + 0.20 * residual, 0.0, 1.0)


class MixtureRouter(nn.Module):
    def __init__(self, feature_dim: int, experts: int):
        super().__init__()
        self.experts = int(experts)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, 96),
            nn.ReLU(),
            nn.Dropout(0.05),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Linear(64, experts),
        )

    def forward(self, features: torch.Tensor, predictions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = torch.softmax(self.network(features), dim=1)
        output = torch.sum(weights.unsqueeze(-1) * predictions, dim=1)
        return output, weights


def coordinate_classification_loss(
    prediction: torch.Tensor,
    logits: torch.Tensor,
    truth_norm: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    return F.smooth_l1_loss(prediction, truth_norm) + 0.20 * F.cross_entropy(logits, labels)
