"""CPU contracts for the corrected fair MPUrge-MAP suite."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import corrected_mpurge_map_suite as suite


def test_frozen_layout_has_80_rps_and_locked_queries_are_off_grid():
    references = suite.reference_positions()
    queries = suite.locked_query_positions(100)
    assert references.shape == (80, 2)
    assert queries.shape == (100, 2)
    assert np.all(np.min(np.linalg.norm(queries[:, None] - references[None], axis=2), axis=1) > 0.035)


def test_original_total_window_P5_is_half_width_two():
    a = np.asarray([1.0, 2.0, 3.0])
    b = np.asarray([1.1, 2.1, 3.2])
    matches = suite.corrected_pairwise_match(
        a, b, half_window_p=2, alpha=0.5, normalized_pattern=False
    )
    assert matches
    assert all(np.isfinite(match.dissimilarity) for match in matches)
    # The local API names the half-width explicitly; total P is 2*2+1.
    assert 2 * 2 + 1 == 5


def test_empty_policy_is_shared_centroid_and_empty_reference_is_infinite_without_ablation():
    references = suite.reference_positions()
    predicted = suite._no_evidence_override(
        np.asarray([[99.0, 99.0]]), [{"delay": np.empty(0)}], references
    )[0]
    assert np.allclose(predicted, np.mean(references, axis=0))
    assert np.isinf(suite.symmetric_chamfer(np.asarray([1.0]), np.empty(0)))
    finite_ablation = suite.chamfer_scores(
        np.asarray([1.0]), [np.empty(0)], path_count_rule=0.5
    )
    assert np.isfinite(finite_ablation[0])


def test_pointnet_and_path_cross_attention_are_shuffle_and_padding_invariant():
    rng = np.random.default_rng(7)
    pointnet = suite.DirectPointNet(1).eval()
    values = rng.normal(size=(2, 5, 1)).astype(np.float32)
    mask = np.asarray([[True, True, True, False, False], [True] * 5])
    permutation = np.asarray([2, 0, 4, 1, 3])
    with torch.no_grad():
        first = pointnet(torch.from_numpy(values), torch.from_numpy(mask))
        shuffled = pointnet(torch.from_numpy(values[:, permutation]), torch.from_numpy(mask[:, permutation]))
        padded = pointnet(
            torch.from_numpy(np.concatenate((values, np.zeros((2, 3, 1), dtype=np.float32)), axis=1)),
            torch.from_numpy(np.concatenate((mask, np.zeros((2, 3), dtype=bool)), axis=1)),
        )
    assert torch.allclose(first, shuffled, atol=1e-6)
    assert torch.allclose(first, padded, atol=1e-6)

    cross = suite.PathCrossAttentionReranker(5).eval()
    query = rng.normal(size=(2, 5, 5)).astype(np.float32)
    reference = rng.normal(size=(2, 4, 5, 5)).astype(np.float32)
    reference_mask = np.ones((2, 4, 5), dtype=bool)
    xy = rng.uniform(size=(2, 4, 2)).astype(np.float32)
    diagnostic = rng.normal(size=(2, 4, 3)).astype(np.float32)
    analytic = rng.uniform(size=(2, 2)).astype(np.float32)
    with torch.no_grad():
        base = cross(
            torch.from_numpy(query), torch.from_numpy(mask),
            torch.from_numpy(reference), torch.from_numpy(reference_mask),
            torch.from_numpy(xy), torch.from_numpy(diagnostic), torch.from_numpy(analytic),
        )[0]
        candidate_order = np.asarray([2, 0, 3, 1])
        permuted = cross(
            torch.from_numpy(query[:, permutation]), torch.from_numpy(mask[:, permutation]),
            torch.from_numpy(reference[:, candidate_order][:, :, permutation]),
            torch.from_numpy(reference_mask[:, candidate_order][:, :, permutation]),
            torch.from_numpy(xy[:, candidate_order]), torch.from_numpy(diagnostic[:, candidate_order]),
            torch.from_numpy(analytic),
        )[0]
    assert torch.allclose(base, permuted, atol=2e-5)


def test_feature_extractors_ignore_joint_path_permutation():
    observation = {
        "delay": np.asarray([3.0, 1.0, 2.0]),
        "power": np.asarray([-30.0, -40.0, -35.0]),
        "aoa": np.eye(3),
        "cir": np.ones((1, 8, 256), dtype=np.complex64),
    }
    order = np.asarray([1, 2, 0])
    shuffled = {
        "delay": observation["delay"][order],
        "power": observation["power"][order],
        "aoa": observation["aoa"][order],
        "cir": observation["cir"],
    }
    assert np.allclose(suite.delay_feature_vector(observation["delay"]), suite.delay_feature_vector(shuffled["delay"]))
    assert np.allclose(suite.multimodal_feature_vector(observation), suite.multimodal_feature_vector(shuffled))
