from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multimodal_receiver import (
    coherent_adp_distance,
    pack_paths,
    simulate_multimodal_fingerprint,
)


@dataclass(frozen=True)
class Source:
    xyz_m: np.ndarray
    tx_id: int
    order: int


def test_receiver_is_finite_and_feature_aligned():
    sources = [
        Source(np.asarray([1.0, 1.0, 2.5]), 0, 0),
        Source(np.asarray([-1.0, 1.0, 2.5]), 0, 1),
        Source(np.asarray([1.0, 5.0, 2.5]), 0, 1),
    ]
    fp = simulate_multimodal_fingerprint(
        [2.0, 2.0, 1.2], sources, rng=np.random.default_rng(7), maximum_paths=5
    )
    assert fp.cir.shape == (1, 8, 256)
    assert fp.aoa_unit.shape == (len(fp), 3)
    assert len(fp.ranges_m) == len(fp.powers_db) == len(fp.tx_ids)
    assert np.all(np.isfinite(fp.ranges_m))
    assert np.allclose(np.linalg.norm(fp.aoa_unit, axis=1), 1.0, atol=1.0e-6)


def test_materialization_is_seed_deterministic_but_noise_changes_seed():
    sources = [Source(np.asarray([1.0, 1.0, 2.5]), 0, 0)]
    first = simulate_multimodal_fingerprint([2.0, 2.0, 1.2], sources, rng=np.random.default_rng(9))
    repeat = simulate_multimodal_fingerprint([2.0, 2.0, 1.2], sources, rng=np.random.default_rng(9))
    other = simulate_multimodal_fingerprint([2.0, 2.0, 1.2], sources, rng=np.random.default_rng(10))
    assert np.array_equal(first.cir, repeat.cir)
    assert not np.array_equal(first.cir, other.cir)


def test_pack_uses_observed_modalities_and_adp_self_distance_is_small():
    sources = [Source(np.asarray([1.0, 1.0, 2.5]), 0, 0)]
    fp = simulate_multimodal_fingerprint([2.0, 2.0, 1.2], sources, rng=np.random.default_rng(11))
    values, mask = pack_paths([fp], maximum_paths=9, range_scale_m=10.0)
    assert values.shape == (1, 9, 5)
    assert int(mask.sum()) == len(fp)
    assert coherent_adp_distance(fp.cir, fp.cir) <= 1.0e-8
