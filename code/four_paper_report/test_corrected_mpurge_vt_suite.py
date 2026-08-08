from __future__ import annotations

import inspect

import numpy as np

import corrected_mpurge_vt_suite as suite


def test_total_window_and_composite_conflict_priority():
    assert suite.TOTAL_WINDOW_P == 5
    assert suite.HALF_WINDOW_P == 2
    assert suite.padded_windows(np.asarray([1.0, 2.0, 4.0]), suite.HALF_WINDOW_P).shape[1] == 5
    good = suite.Match(0, 0, 1.0, 3.0, 0.1, 0)
    bad = suite.Match(1, 1, 2.0, 2.1, 1.0, 0)
    assert suite.filter_current_by_composite([bad, good], "order")[0] == good


def test_cross_bracket_and_dynamic_beta_exhaustion():
    pair_a, pair_b = (4.0, 2.0), (3.0, 2.5)
    assert suite.crosses(pair_a, pair_b, "order")
    assert not suite.crosses(pair_a, pair_b, "printed")
    rows = suite.dynamic_beta_sets(
        [{"candidate_id": 0, "support_count": 2}, {"candidate_id": 1, "support_count": 5}],
        0.1,
    )
    assert rows[-1]["kept_candidate_ids"] == []


def test_gated_assignment_maximizes_cardinality_before_distance():
    result = suite.maximum_cardinality_gated_match(
        np.asarray([[0.0, 0.0], [0.28, 0.0]]),
        np.asarray([[0.01, 0.0], [0.29, 0.0]]),
        0.30,
    )
    assert result["tp"] == 2


def test_primary_estimators_do_not_accept_truth_or_truth_cardinality():
    assert "truth" not in inspect.signature(suite.selfsupervised_set_estimator).parameters
    assert "truth" not in inspect.signature(suite.diffassign_model_selected).parameters
    primary = {
        name: row
        for name, row in suite.METHOD_REGISTRY.items()
        if "oracle" not in name and name != "old_supervised_vt_pointnet"
    }
    assert all(not row["truth_cardinality"] for row in primary.values())
