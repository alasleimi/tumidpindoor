from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from smart_fair_core import (  # noqa: E402
    Acquisition, PointNetLocalizer, SPACINGS_M, dense_grid,
    expected_reference_count, pack_tokens, score_matrices_gpu, spacing_indices,
)


def fixture_acquisition() -> Acquisition:
    positions = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    ranges = np.zeros((2, 20), dtype=np.float32); ranges[:, :3] = [[1, 3, 7], [2, 5, 9]]
    powers = np.zeros_like(ranges); powers[:, :3] = [[-20, -30, -40], [-22, -31, -42]]
    aoa = np.zeros((2, 20, 3), dtype=np.float32); aoa[:, :3, 0] = 1.0
    cir = np.zeros((2, 256), dtype=np.float32); cir[:, 0] = 1.0
    mask = np.zeros((2, 20), dtype=bool); mask[:, :3] = True
    tx = np.full((2, 20), -1, dtype=np.int16); tx[:, :3] = [[0, 1, 2], [2, 1, 0]]
    return Acquisition(positions, ranges, powers, aoa, cir, mask, tx)


def test_spacing_maps_are_exact_nested_subsets_with_required_counts():
    grid = dense_grid()
    assert len(grid) == 10000
    for spacing in SPACINGS_M:
        indices = spacing_indices(spacing)
        assert len(indices) == expected_reference_count(spacing)
        assert np.all(indices >= 0) and np.all(indices < len(grid))
        assert len(np.unique(indices)) == len(indices)


def test_delay_tokens_cannot_change_when_auxiliary_or_identity_channels_change():
    acquisition = fixture_acquisition()
    first, mask = pack_tokens(acquisition, "delay")
    changed = fixture_acquisition()
    changed.powers_db[:] = 999
    changed.aoa_unit[:] = -0.5
    changed.cir_features[:] = 0.25
    changed.diagnostic_tx_ids[:] = 3
    second, second_mask = pack_tokens(changed, "delay")
    assert np.array_equal(first, second)
    assert np.array_equal(mask, second_mask)
    assert "diagnostic_tx_ids" not in acquisition.public_arrays()


def test_pointnet_is_permutation_invariant_in_eval_mode():
    torch.manual_seed(7)
    acquisition = fixture_acquisition()
    values, mask = pack_tokens(acquisition, "delay")
    model = PointNetLocalizer(1).eval()
    permutation = np.asarray([2, 0, 1] + list(range(3, 20)))
    with torch.inference_mode():
        expected = model(torch.as_tensor(values), torch.as_tensor(mask))
        actual = model(torch.as_tensor(values[:, permutation]), torch.as_tensor(mask[:, permutation]))
    assert torch.allclose(expected, actual, atol=1e-7, rtol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA score kernel")
def test_chamfer_and_mca_scores_are_path_permutation_invariant():
    acquisition = fixture_acquisition()
    q = acquisition.subset(np.asarray([0]))
    baseline = score_matrices_gpu(q.ranges_m, q.mask, acquisition.ranges_m, acquisition.mask, batch_size=1)
    permutation = np.asarray([2, 0, 1] + list(range(3, 20)))
    permuted = score_matrices_gpu(
        q.ranges_m[:, permutation], q.mask[:, permutation],
        acquisition.ranges_m[:, permutation], acquisition.mask[:, permutation], batch_size=1,
    )
    for left, right in zip(baseline[:2], permuted[:2], strict=True):
        assert np.allclose(left, right, atol=1e-6)

