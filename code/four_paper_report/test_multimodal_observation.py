from __future__ import annotations

import numpy as np

from experiments.four_paper_report.multimodal_observation import (
    normalized_cir_tensor,
    path_feature_tensor,
    simulate_multimodal_observation,
)
from experiments.paper_protocol_replications.common import rectangular_sources


def _observation(seed: int = 7):
    sources = rectangular_sources([4.0, 5.0, 3.0], np.asarray([[1.0, 1.5, 2.5]]), maximum_order=2)
    return simulate_multimodal_observation(
        [2.2, 3.1, 1.2], sources, rng=np.random.default_rng(seed), false_alarm_rate=0.0
    )


def test_deterministic_for_seed_and_physically_valid():
    first, second = _observation(), _observation()
    np.testing.assert_allclose(first.ranges_m, second.ranges_m)
    np.testing.assert_allclose(first.powers_db, second.powers_db)
    np.testing.assert_allclose(first.aoa_unit, second.aoa_unit)
    np.testing.assert_allclose(first.cir, second.cir)
    first.validate()
    assert np.any(np.abs(first.cir) > 0.0)


def test_noise_changes_every_continuous_modality():
    first, second = _observation(7), _observation(8)
    assert not np.allclose(first.cir, second.cir)
    # Cardinality can differ after noisy detection; compare a robust summary.
    assert not np.isclose(np.sum(first.powers_db), np.sum(second.powers_db))
    assert not np.isclose(np.sum(first.ranges_m), np.sum(second.ranges_m))
    assert not np.isclose(np.sum(first.aoa_unit), np.sum(second.aoa_unit))


def test_packers_only_export_requested_observables():
    observations = [_observation(1), _observation(2)]
    values, mask = path_feature_tensor(
        observations,
        maximum_paths=9,
        range_scale_m=10.0,
        maximum_tx=1,
        fields=("range", "power", "aoa", "complex", "snr", "tx"),
    )
    assert values.shape == (2, 9, 9)
    assert mask.shape == (2, 9)
    assert np.all(values[~mask] == 0.0)
    cir = normalized_cir_tensor(observations)
    assert cir.shape == (2, 2, 8, 512)
    norms = np.sqrt(np.sum(np.square(cir), axis=(1, 2, 3)))
    np.testing.assert_allclose(norms, 1.0, atol=2.0e-6)
