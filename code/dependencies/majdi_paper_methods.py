"""Literal delay-only methods from Majdi Abdmoulah's papers and theses.

This module deliberately contains the paper algorithms, not proxies for them.
The implementations follow the pseudocode and equations in:

* Abdmoulah (2019), bachelor thesis, Chapter 4 (PBA);
* Abdmoulah (2022), master thesis, Chapter 4 (LEA);
* Abdmoulah & Steinbach (2025), MPUrge, Section V;
* Abdmoulah & Steinbach (2026), MPUrge-MAP, Section III; and
* Abdmoulah & Steinbach (2026), SMART-LEA, Section IV.

The source papers contain two genuine internal inconsistencies. Equation (5)
multiplies mean dissimilarity by match coverage, which rewards low coverage
when candidates are ranked in ascending order.  The prose says low coverage
is penalized.  ``coverage_mode='literal'`` implements the printed equation;
``coverage_mode='penalty'`` implements the stated intent by division.  They
are separate named estimators so an experiment cannot silently repair the
paper.  Bachelor-thesis Equation (4.8) also does not mathematically test the
line intersection described in its prose. ``cross_mode='printed'`` preserves
that equation and ``cross_mode='order'`` implements the stated intersection
test.  Again, neither interpretation is silently substituted for the other.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class Match:
    """One bijective MPC match, retaining original vector indices."""

    index_a: int
    index_b: int
    value_a: float
    value_b: float
    dissimilarity: float
    iteration: int


def _ordered(values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError("MPC vectors must contain only finite values")
    order = np.argsort(array, kind="stable")
    return array[order], order.astype(np.int64)


def _best_ordered_subset(
    longer: np.ndarray,
    target: np.ndarray,
    *,
    norm: Literal["l1", "l2"],
) -> np.ndarray:
    """Exact exhaustive-subset optimum using monotone dynamic programming.

    Both papers define size unification by enumerating all subsets of the
    longer sorted vector and selecting the one closest to the shorter vector.
    Dynamic programming returns exactly that optimum without an exponential
    materialization of all subsets.
    """

    n, m = len(longer), len(target)
    if m > n:
        raise ValueError("target cannot be longer than the candidate vector")
    if m == 0:
        return np.empty(0, dtype=np.int64)
    power = 1 if norm == "l1" else 2
    cost = np.full((m + 1, n + 1), np.inf, dtype=np.float64)
    take = np.zeros((m + 1, n + 1), dtype=bool)
    cost[0, :] = 0.0
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            skip = cost[i, j - 1]
            choose = cost[i - 1, j - 1] + abs(longer[j - 1] - target[i - 1]) ** power
            # Exhaustive enumeration keeps the first minimum. Prefer a take
            # on equality so reconstruction selects the lexicographically
            # earliest ordered subset.
            if choose <= skip:
                cost[i, j] = choose
                take[i, j] = True
            else:
                cost[i, j] = skip
    selected: list[int] = []
    i, j = m, n
    while i:
        if j == 0:
            raise RuntimeError("Failed to reconstruct size-unification subset")
        if take[i, j]:
            selected.append(j - 1)
            i -= 1
        j -= 1
    return np.asarray(selected[::-1], dtype=np.int64)


def size_unify(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    norm: Literal["l1", "l2"] = "l1",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Paper-exact size unification with original-index bookkeeping."""

    a, ia = _ordered(values_a)
    b, ib = _ordered(values_b)
    if len(a) > len(b):
        keep = _best_ordered_subset(a, b, norm=norm)
        a, ia = a[keep], ia[keep]
    elif len(b) > len(a):
        keep = _best_ordered_subset(b, a, norm=norm)
        b, ib = b[keep], ib[keep]
    return a, b, ia, ib


def _padded_windows(values: np.ndarray, p: int) -> np.ndarray:
    """Zero-centred P=2p+1 windows with negligible border increments."""

    if p < 0:
        raise ValueError("p must be non-negative")
    if not len(values):
        return np.empty((0, 2 * p + 1), dtype=np.float64)
    # The thesis specifies equally spaced border points at an "extremely
    # small" distance.  A scale-relative value makes that disclosed choice
    # explicit and stable across metres/seconds representations.
    scale = max(float(np.ptp(values)), float(np.max(np.abs(values))), 1.0)
    tiny = np.finfo(np.float64).eps * scale * 16.0
    lower = values[0] - tiny * np.arange(p, 0, -1, dtype=np.float64)
    upper = values[-1] + tiny * np.arange(1, p + 1, dtype=np.float64)
    padded = np.concatenate((lower, values, upper))
    windows = np.lib.stride_tricks.sliding_window_view(padded, 2 * p + 1).copy()
    return windows - values[:, None]


def dissimilarity_matrix(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    p: int,
    alpha: float,
    normalized_pattern: bool,
) -> np.ndarray:
    """Equations (8)--(11), with optional MPUrge-MAP normalization."""

    a = np.asarray(values_a, dtype=np.float64).reshape(-1)
    b = np.asarray(values_b, dtype=np.float64).reshape(-1)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if not len(a) or not len(b):
        return np.empty((len(a), len(b)), dtype=np.float64)
    wa = _padded_windows(a, p)
    wb = _padded_windows(b, p)
    # This is the "extensive" pattern metric described in Bachelor Thesis
    # Sec. 4.1.2.a, not the much simpler aligned-entry L1 distance.  For every
    # window it compares every within-window pairwise separation with the
    # equivalent separation in the other window.  There are
    # C(2p+1, 2) = sum_{m=1}^{2p} m contributions, exactly the accumulation
    # factor printed in MPUrge-MAP Eq. (4).
    upper = np.triu_indices(2 * p + 1, k=1)
    pairwise_a = np.abs(wa[:, :, None] - wa[:, None, :])[:, upper[0], upper[1]]
    pairwise_b = np.abs(wb[:, :, None] - wb[:, None, :])[:, upper[0], upper[1]]
    pattern = np.sum(
        np.abs(pairwise_a[:, None, :] - pairwise_b[None, :, :]), axis=2
    )
    if normalized_pattern:
        accumulation = sum(range(1, 2 * p + 1))
        if accumulation:
            pattern = pattern / float(accumulation)
    distance = np.abs(a[:, None] - b[None, :])
    return alpha * distance + (1.0 - alpha) * pattern


def _crosses(
    pair_a: tuple[float, float],
    pair_b: tuple[float, float],
    *,
    mode: Literal["printed", "order"],
) -> bool:
    """Evaluate either reading of the paper's contradictory cross check."""

    if mode == "printed":
        # Bachelor thesis Eq. (4.8). This tests whether the two segments have
        # opposite signed endpoint differences, not whether they intersect.
        return (pair_a[0] - pair_a[1]) * (pair_b[0] - pair_b[1]) < 0.0
    if mode == "order":
        # Two segments joining parallel ordered axes cross iff their endpoint
        # order reverses. This implements the prose and Figures 4.4/4.
        return (pair_a[0] - pair_b[0]) * (pair_a[1] - pair_b[1]) < 0.0
    raise ValueError(f"Unknown cross mode {mode!r}")


def pairwise_match(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    p: int = 2,
    alpha: float = 0.5,
    normalized_pattern: bool = False,
    size_norm: Literal["l1", "l2"] = "l1",
    cross_mode: Literal["printed", "order"] = "printed",
) -> list[Match]:
    """PBA/MPUrge Merge--Purge--Urge matching, including removal order."""

    a, b, ia, ib = size_unify(values_a, values_b, norm=size_norm)
    accepted: list[Match] = []
    iteration = 0
    while len(a) and len(b):
        delta = dissimilarity_matrix(
            a, b, p=p, alpha=alpha, normalized_pattern=normalized_pattern
        )
        row_best = np.argmin(delta, axis=1)
        col_best = np.argmin(delta, axis=0)
        potential_indices = [
            (i, int(j)) for i, j in enumerate(row_best) if int(col_best[int(j)]) == i
        ]
        if not potential_indices:
            raise RuntimeError("Mutual-nearest extraction unexpectedly produced no pair")

        potential = [
            Match(
                int(ia[i]),
                int(ib[j]),
                float(a[i]),
                float(b[j]),
                float(delta[i, j]),
                iteration,
            )
            for i, j in potential_indices
        ]

        # The pseudocode removes every potential pair before either filtering
        # stage. Rejected pairs are therefore not reconsidered later.
        remove_a = {i for i, _ in potential_indices}
        remove_b = {j for _, j in potential_indices}
        keep_a = np.asarray([i not in remove_a for i in range(len(a))])
        keep_b = np.asarray([j not in remove_b for j in range(len(b))])
        a, ia = a[keep_a], ia[keep_a]
        b, ib = b[keep_b], ib[keep_b]

        previous_filtered = [
            candidate
            for candidate in potential
            if not any(
                _crosses(
                    (candidate.value_a, candidate.value_b),
                    (prior.value_a, prior.value_b),
                    mode=cross_mode,
                )
                for prior in accepted
            )
        ]
        current_filtered: list[Match] = []
        for candidate in sorted(
            previous_filtered,
            key=lambda match: (abs(match.value_a - match.value_b), match.index_a, match.index_b),
        ):
            if not any(
                _crosses(
                    (candidate.value_a, candidate.value_b),
                    (prior.value_a, prior.value_b),
                    mode=cross_mode,
                )
                for prior in current_filtered
            ):
                current_filtered.append(candidate)
        accepted.extend(current_filtered)
        iteration += 1
    return accepted


def mpurge_overall_score(
    matches: Sequence[Match],
    length_a: int,
    length_b: int,
    *,
    coverage_mode: Literal["plain", "literal", "penalty"],
) -> float:
    """Original mean score or either reading of MPUrge-MAP Equation (5)."""

    if not matches:
        return float("inf")
    mean = float(np.mean([match.dissimilarity for match in matches]))
    coverage = len(matches) / max(length_a, length_b, 1)
    if coverage_mode == "plain":
        return mean
    if coverage_mode == "literal":
        return mean * coverage
    if coverage_mode == "penalty":
        return mean / max(coverage, np.finfo(np.float64).eps)
    raise ValueError(f"Unknown coverage mode {coverage_mode!r}")


def mpurge_map_localize(
    query: Sequence[float],
    reference_fingerprints: Sequence[Sequence[float]],
    reference_positions_m: np.ndarray,
    *,
    p: int,
    alpha: float,
    k: int,
    normalized_pattern: bool,
    coverage_mode: Literal["plain", "literal", "penalty"],
    cross_mode: Literal["printed", "order"] = "printed",
) -> tuple[np.ndarray, np.ndarray]:
    """Algorithm 1 of MPUrge-MAP, returning position and all RP scores."""

    positions = np.asarray(reference_positions_m, dtype=np.float64)
    if len(reference_fingerprints) != len(positions):
        raise ValueError("Reference fingerprints and positions differ in length")
    scores = np.full(len(positions), np.inf, dtype=np.float64)
    for index, reference in enumerate(reference_fingerprints):
        matches = pairwise_match(
            query,
            reference,
            p=p,
            alpha=alpha,
            normalized_pattern=normalized_pattern,
            size_norm="l1",
            cross_mode=cross_mode,
        )
        scores[index] = mpurge_overall_score(
            matches, len(query), len(reference), coverage_mode=coverage_mode
        )
    finite = np.flatnonzero(np.isfinite(scores))
    if not len(finite):
        return np.mean(positions, axis=0), scores
    chosen = finite[np.argsort(scores[finite], kind="stable")[: min(k, len(finite))]]
    safe = np.maximum(scores[chosen], np.finfo(np.float64).eps)
    weights = 1.0 / safe
    weights /= weights.sum()
    return np.sum(positions[chosen] * weights[:, None], axis=0), scores


def published_mca_localize(
    query: Sequence[float],
    reference_fingerprints: Sequence[Sequence[float]],
    reference_positions_m: np.ndarray,
    *,
    epsilon_m: float,
    k: int = 1,
) -> np.ndarray:
    """Published MCA score followed by the paper's k-best weighted head."""

    query_array = np.asarray(query, dtype=np.float64)
    scores = []
    for reference in reference_fingerprints:
        reference_array = np.asarray(reference, dtype=np.float64)
        if not len(query_array) or not len(reference_array):
            scores.append(0.0)
            continue
        nearest = np.min(np.abs(query_array[:, None] - reference_array[None, :]), axis=1)
        accepted = nearest < epsilon_m
        scores.append(float(np.sum((epsilon_m - nearest[accepted]) ** 2)))
    scores_array = np.asarray(scores)
    chosen = np.argsort(-scores_array, kind="stable")[: min(k, len(scores_array))]
    if k == 1 or not np.any(scores_array[chosen] > 0):
        return np.asarray(reference_positions_m, dtype=np.float64)[chosen[0]].copy()
    weights = np.maximum(scores_array[chosen], 0.0)
    weights /= weights.sum()
    return np.sum(np.asarray(reference_positions_m)[chosen] * weights[:, None], axis=0)


def trilaterate(
    anchors_m: np.ndarray,
    ranges_m: Sequence[float],
    initial_position_m: Sequence[float],
) -> np.ndarray:
    """Nonlinear least-squares trilateration used by LEA/SMART-LEA."""

    anchors = np.asarray(anchors_m, dtype=np.float64)
    ranges = np.asarray(ranges_m, dtype=np.float64)
    initial = np.asarray(initial_position_m, dtype=np.float64)
    if anchors.ndim != 2 or anchors.shape[0] != len(ranges):
        raise ValueError("Invalid anchor/range arrays")
    if len(anchors) < anchors.shape[1] + 1:
        return initial.copy()
    solution = least_squares(
        lambda point: np.linalg.norm(anchors - point[None, :], axis=1) - ranges,
        initial,
        method="trf",
    )
    return solution.x.astype(np.float64)


def lea_iteration(
    initial_position_m: Sequence[float],
    virtual_transmitters_m: np.ndarray,
    measured_fingerprint_m: Sequence[float],
    *,
    matcher: Literal["pba", "mpurge"],
    p: int,
    alpha: float,
    cross_mode: Literal["printed", "order"] = "printed",
) -> tuple[np.ndarray, list[Match]]:
    """One master-thesis LEA or SMART-LEA refinement iteration."""

    initial = np.asarray(initial_position_m, dtype=np.float64)
    vts = np.asarray(virtual_transmitters_m, dtype=np.float64)
    recalculated = np.linalg.norm(vts - initial[None, :], axis=1)
    # pairwise_match's A indices refer to measured paths and B indices to the
    # recalculated vector, hence B preserves the VT association.
    matches = pairwise_match(
        measured_fingerprint_m,
        recalculated,
        p=p,
        alpha=alpha,
        normalized_pattern=False,
        size_norm="l2" if matcher == "pba" else "l1",
        cross_mode=cross_mode,
    )
    dimension = initial.size
    if len(matches) < dimension + 1:
        return initial.copy(), matches
    anchors = np.asarray([vts[match.index_b] for match in matches])
    ranges = np.asarray([match.value_a for match in matches])
    return trilaterate(anchors, ranges, initial), matches


def lea_refine(
    initial_position_m: Sequence[float],
    virtual_transmitters_m: np.ndarray,
    measured_fingerprint_m: Sequence[float],
    *,
    matcher: Literal["pba", "mpurge"],
    p: int,
    alpha: float,
    iterations: int,
    cross_mode: Literal["printed", "order"] = "printed",
) -> tuple[np.ndarray, list[list[Match]]]:
    """Algorithm 2 iterative LEA/SMART-LEA."""

    position = np.asarray(initial_position_m, dtype=np.float64).copy()
    history: list[list[Match]] = []
    for _ in range(iterations):
        position, matches = lea_iteration(
            position,
            virtual_transmitters_m,
            measured_fingerprint_m,
            matcher=matcher,
            p=p,
            alpha=alpha,
            cross_mode=cross_mode,
        )
        history.append(matches)
    return position, history
