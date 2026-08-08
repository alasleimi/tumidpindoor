from __future__ import annotations

import inspect
import numpy as np
import pytest

from multimodal_receiver import MultimodalFingerprint
from run_cnn_fair import (
    ENVIRONMENT_BLIND_PROTOCOL,
    METHOD_ALIASES,
    METHOD_FEATURES,
    PAPER_CONDITION_PROTOCOL,
    PROTOCOL_METHODS,
    PROTOCOL_INDEPENDENT,
    RoomData,
    analytic_bundle,
    candidate_tensors,
    paired_intervals,
    reference_candidate_indices,
    reference_lookup,
    train_candidate_models,
)


def _fingerprint(index: int) -> MultimodalFingerprint:
    ranges = np.asarray([1.0 + index / 100.0, 1.2 + index / 100.0])
    return MultimodalFingerprint(
        ranges_m=ranges,
        powers_db=np.asarray([-30.0, -34.0]),
        aoa_unit=np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        tx_ids=np.zeros(2, dtype=np.int64),
        cir=np.ones((1, 8, 16), dtype=np.complex64),
        noise_variance=0.01,
        range_bin_m=0.15,
    )


def _room_data() -> RoomData:
    fingerprints, labels, aps, configs = [], [], [], []
    for config in range(7):
        for rp in range(20):
            index = config * 20 + rp
            fingerprints.append(_fingerprint(index))
            labels.append(rp)
            aps.append(0)
            configs.append(config)
    return RoomData(
        name="test_room",
        room=np.asarray([20.0, 8.0, 3.0]),
        references=np.column_stack((np.arange(20), np.zeros(20))),
        aps=np.asarray([[2.0, 3.0, 2.5]]),
        configurations=tuple(() for _ in range(7)),
        train_fingerprints=fingerprints,
        labels=np.asarray(labels),
        ap_ids=np.asarray(aps),
        config_ids=np.asarray(configs),
        test_rows=[],
    )


def test_fit_map_candidate_selection_is_strictly_leave_self() -> None:
    data = _room_data()
    lookup = reference_lookup(data)
    held_out = int(lookup[0, 3, 7])
    fit_mask = np.ones(len(data.train_fingerprints), dtype=bool)
    fit_mask[held_out] = False

    for rp in range(20):
        candidates = reference_candidate_indices(
            lookup,
            0,
            3,
            ENVIRONMENT_BLIND_PROTOCOL,
            rp,
            allowed_reference_mask=fit_mask,
            excluded_reference_index=held_out,
        )
        assert held_out not in candidates
        assert np.all(fit_mask[candidates])

    query = data.train_fingerprints[held_out]
    *_, selected = candidate_tensors(
        [query],
        [(0, 3)],
        data,
        lookup,
        ENVIRONMENT_BLIND_PROTOCOL,
        allowed_reference_mask=fit_mask,
        excluded_reference_indices=[held_out],
    )
    assert held_out not in selected[0]
    assert np.all(fit_mask[selected[0]])


def test_independent_fit_repeat_can_use_singleton_measured_map_row() -> None:
    """A noisy repeat is not the identical stored acquisition.

    The paper split can leave exactly one fit configuration for an AP/RP.
    Training may compare an independently corrupted repeat with that already
    measured map row; excluding it would make the candidate set empty.  This
    does not relax the fit-only rule used by validation or test.
    """
    data = _room_data()
    lookup = reference_lookup(data)
    fit_mask = np.zeros(len(data.train_fingerprints), dtype=bool)
    for rp in range(20):
        fit_mask[int(lookup[0, 0, rp])] = True
    source = int(lookup[0, 0, 2])
    repeat = _fingerprint(999)
    assert not np.array_equal(repeat.ranges_m, data.train_fingerprints[source].ranges_m)

    *_, selected = candidate_tensors(
        [repeat],
        [(0, 0)],
        data,
        lookup,
        ENVIRONMENT_BLIND_PROTOCOL,
        allowed_reference_mask=fit_mask,
    )
    assert int(selected[0, 2]) == source
    assert np.all(fit_mask[selected[0]])

    with pytest.raises(ValueError, match="no eligible reference"):
        candidate_tensors(
            [repeat],
            [(0, 0)],
            data,
            lookup,
            ENVIRONMENT_BLIND_PROTOCOL,
            allowed_reference_mask=fit_mask,
            excluded_reference_indices=[source],
        )


def test_oracle_condition_bracket_is_explicit_and_alias_is_not_a_result() -> None:
    data = _room_data()
    lookup = reference_lookup(data)
    candidates = reference_candidate_indices(
        lookup, 0, 4, PAPER_CONDITION_PROTOCOL, 8
    )
    assert candidates.tolist() == [int(lookup[0, 4, 8])]
    assert "two_sided_vt_registration" not in METHOD_FEATURES
    assert METHOD_ALIASES["two_sided_vt_registration"]["canonical_method"] == "assigned_vt_inverse_consensus"
    assert "AP coordinates" in METHOD_FEATURES["candidate_pointnet_reranker"]
    assert "detected-path count" in METHOD_FEATURES["survival_cir_toa_extratrees"]
    assert "assigned_vt_inverse_consensus" not in PROTOCOL_METHODS[PAPER_CONDITION_PROTOCOL]
    assert "assigned_vt_inverse_consensus" in PROTOCOL_METHODS[PROTOCOL_INDEPENDENT]
    assert "beta_marginal_vt_survival" in PROTOCOL_METHODS[PROTOCOL_INDEPENDENT]


def test_full_split_ap28_rp2_singleton_fit_repeat_remains_eligible() -> None:
    """Regression for the full-run AP28/RP2 singleton fit-map cell."""

    lookup = np.arange(60 * 7 * 20, dtype=np.int64).reshape(60, 7, 20)
    fit_mask = np.zeros(lookup.size, dtype=bool)
    source = int(lookup[28, 4, 2])
    fit_mask[source] = True
    candidates = reference_candidate_indices(
        lookup,
        28,
        4,
        ENVIRONMENT_BLIND_PROTOCOL,
        2,
        allowed_reference_mask=fit_mask,
    )
    assert candidates.tolist() == [source]

    # Candidate-training pseudo-queries are noisy repeats and may retain this
    # sole stored fit acquisition.  Held-out validation still excludes its
    # source explicitly.
    implementation = inspect.getsource(train_candidate_models)
    assert "excluded_reference_indices=augmented_source_indices" not in implementation
    assert "excluded_reference_indices=[int(index) for index in validation_ids]" in implementation


def test_paired_interval_uses_protocol_independent_baseline_without_row_duplication() -> None:
    rows = []
    for query in range(8):
        common = {
            "room": "room_a",
            "query": query,
            "condition": f"obstacles_{query // 2}",
            "ap_id": query % 2,
            "physical_query_id": f"room_a:obstacles_{query // 2}:{query}",
        }
        rows.append({**common, "protocol": PROTOCOL_INDEPENDENT, "method": "majid_cnn_classifier", "error_m": 2.0})
        rows.append({**common, "protocol": PAPER_CONDITION_PROTOCOL, "method": "majid_mca_eps1", "error_m": 1.0})
    intervals = paired_intervals(rows, bootstrap_reps=20)
    assert len(intervals) == 1
    assert intervals[0]["baseline_protocol"] == PROTOCOL_INDEPENDENT
    assert intervals[0]["protocol"] == PAPER_CONDITION_PROTOCOL
    assert intervals[0]["mean_error_reduction_m"] == 1.0
    assert intervals[0]["bootstrap_unit"].startswith("physical off-grid query")
    assert intervals[0]["bootstrap_clusters"] == 8
