"""EvoMDP-Rank: a deployable evolutionary delay-set localization metric.

The evolutionary search learns one small, scene-independent rule for combining
observable discrepancies between a query MDP and each stored reference MDP.
It never selects reference locations and never sees protected-test positions or
errors.  The learned genome is frozen before the test tensors are constructed.

This is a direct localization method, not a proxy for a generative model.  Its
features are mathematically defined set distances (Chamfer, Wasserstein,
gap-spectrum Wasserstein, partial assignment, normalized all-pairs pattern,
MCA complement, cardinality, and moments). Differential evolution learns their
convex mixture, temperature, and top-k interpolation head on development data.
"""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import differential_evolution, linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parent))
from majdi_nist_benchmark import load_lroom, reference_indices  # noqa: E402
from majdi_paper_methods import (  # noqa: E402
    dissimilarity_matrix,
    mpurge_map_localize,
    published_mca_localize,
)
from obstruction_aware_hough_gate import (  # noqa: E402
    TEST_CONDITIONS,
    TRAIN_CONDITIONS,
    obstruct,
)
from qd_dataset import DEFAULT_QD_ROOT  # noqa: E402
from vt_splatloc_benchmark import query_ids, stable_rng  # noqa: E402


SEED = 20260719
SPACING_M = 1.0
MAX_PATHS = 9
FEATURE_NAMES = (
    "symmetric_chamfer",
    "quantile_wasserstein",
    "gap_spectrum_wasserstein",
    "log_cardinality_ratio",
    "partial_assignment_with_unmatched",
    "normalized_all_pairs_pattern",
    "mca_similarity_complement",
    "mean_and_scale_difference",
)


def strongest_delays(fp, maximum: int = MAX_PATHS) -> np.ndarray:
    if len(fp) <= maximum:
        chosen = np.arange(len(fp))
    else:
        chosen = np.argsort(-fp.gain_db, kind="stable")[:maximum]
    return np.sort(fp.ranges_m[chosen].astype(np.float64, copy=True))


def _quantile_distance(a: np.ndarray, b: np.ndarray, count: int) -> float:
    if not len(a) or not len(b):
        return 30.0
    q = np.linspace(0.0, 1.0, count)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def delay_set_features(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Eight nonnegative observable fingerprint discrepancies.

    Seven are symmetric. The MCA complement intentionally preserves Majdi's
    query-to-reference direction, where each measured query MPC votes for its
    nearest stored MPC.
    """

    a = np.sort(np.asarray(a, dtype=np.float64).reshape(-1))
    b = np.sort(np.asarray(b, dtype=np.float64).reshape(-1))
    if not len(a) or not len(b):
        empty_cost = 30.0 + abs(len(a) - len(b))
        return np.asarray([empty_cost] * len(FEATURE_NAMES), dtype=np.float64)
    residual = np.abs(a[:, None] - b[None, :])
    chamfer = 0.5 * (float(np.mean(np.min(residual, axis=1))) +
                     float(np.mean(np.min(residual, axis=0))))
    wasserstein = _quantile_distance(a, b, 17)
    gap_a, gap_b = np.diff(a), np.diff(b)
    gap_wasserstein = _quantile_distance(gap_a, gap_b, 11) if len(gap_a) and len(gap_b) else wasserstein
    cardinality = abs(float(np.log((len(a) + 0.5) / (len(b) + 0.5))))
    rows, cols = linear_sum_assignment(residual)
    unmatched = abs(len(a) - len(b))
    assignment = (float(np.sum(residual[rows, cols])) + 1.5 * unmatched) / max(len(a), len(b))
    # A radius of three is the P=7 legacy context. The metric itself is the
    # corrected all-pairs Eq. (4) construction, normalized by C(7,2).
    pattern_matrix = dissimilarity_matrix(
        a, b, p=3, alpha=0.0, normalized_pattern=True
    )
    pattern = 0.5 * (float(np.mean(np.min(pattern_matrix, axis=1))) +
                     float(np.mean(np.min(pattern_matrix, axis=0))))
    epsilon = 1.6
    nearest = np.min(residual, axis=1)
    accepted = nearest < epsilon
    similarity = float(np.sum((epsilon - nearest[accepted]) ** 2))
    mca_complement = max(0.0, 1.0 - similarity / (len(a) * epsilon**2))
    moments = abs(float(np.mean(a) - np.mean(b))) + abs(float(np.std(a) - np.std(b)))
    return np.asarray([
        chamfer, wasserstein, gap_wasserstein, cardinality, assignment,
        pattern, mca_complement, moments,
    ])


def make_cases(positions, fingerprints, references, conditions, split: str):
    cases = []
    for condition in conditions:
        for frame in query_ids(len(positions), references, split):
            query = obstruct(
                fingerprints[frame], condition,
                stable_rng("L-Room", SPACING_M, split, frame, condition.name),
            )
            cases.append((condition.name, int(frame), strongest_delays(query)))
    return cases


def feature_tensor(cases, reference_delays) -> np.ndarray:
    return np.asarray([
        [delay_set_features(query, reference) for reference in reference_delays]
        for _, _, query in cases
    ])


def decode(genome: np.ndarray) -> tuple[np.ndarray, float, int]:
    # copy=True is essential: optimizers own and reuse the candidate array.
    # Mutating a slice here silently corrupts the evolutionary population.
    logits = np.array(genome[:len(FEATURE_NAMES)], dtype=np.float64, copy=True)
    logits -= np.max(logits)
    weights = np.exp(logits)
    weights /= np.sum(weights)
    temperature = float(np.exp(genome[len(FEATURE_NAMES)]))
    top_k = int(np.clip(np.rint(genome[len(FEATURE_NAMES) + 1]), 1, 6))
    return weights, temperature, top_k


def estimates_from_genome(genome, tensor, scales, reference_positions):
    weights, temperature, top_k = decode(genome)
    scores = np.sum((tensor / scales[None, None, :]) * weights[None, None, :], axis=2)
    estimates = []
    for row in scores:
        chosen = np.argsort(row, kind="stable")[:top_k]
        relative = (row[chosen] - row[chosen[0]]) / max(temperature, 1e-9)
        interpolation = np.exp(-np.clip(relative, 0.0, 700.0))
        interpolation /= np.sum(interpolation)
        estimates.append(np.sum(reference_positions[chosen] * interpolation[:, None], axis=0))
    return np.asarray(estimates), scores


def macro_objective(genome, tensor, scales, reference_positions, truths, names):
    estimates, _ = estimates_from_genome(genome, tensor, scales, reference_positions)
    errors = np.linalg.norm(estimates - truths, axis=1)
    condition_scores = []
    for condition in sorted(set(names)):
        values = errors[np.asarray([name == condition for name in names])]
        condition_scores.append(float(np.mean(values)) + 0.15 * float(np.quantile(values, 0.9)))
    # Very weak shrinkage prevents arbitrary saturated logits without dictating
    # which metric should win.
    return float(np.mean(condition_scores) + 2e-4 * np.mean(np.square(genome[:8])))


def summarize(rows):
    output = []
    for condition, method in sorted({(r["condition"], r["method"]) for r in rows}):
        values = np.asarray([r["error_m"] for r in rows if r["condition"] == condition and r["method"] == method])
        output.append(dict(
            condition=condition, method=method, queries=len(values),
            mean_error_m=float(np.mean(values)), rmse_m=float(np.sqrt(np.mean(values**2))),
            median_error_m=float(np.median(values)), p90_error_m=float(np.quantile(values, 0.9)),
        ))
    return output


def paired_bootstrap(rows, candidate, baseline):
    left = {(r["condition"], r["frame"]): r["error_m"] for r in rows if r["method"] == candidate}
    right = {(r["condition"], r["frame"]): r["error_m"] for r in rows if r["method"] == baseline}
    if set(left) != set(right):
        raise ValueError("paired rows differ")
    frames = sorted({frame for _, frame in left})
    conditions = sorted({condition for condition, _ in left})
    gains = np.asarray([[right[(c, f)] - left[(c, f)] for c in conditions] for f in frames])
    rng = np.random.default_rng(SEED + len(baseline))
    sample = gains[rng.integers(0, len(frames), size=(5000, len(frames)))].mean(axis=(1, 2))
    return dict(candidate=candidate, baseline=baseline, mean_reduction_m=float(np.mean(gains)),
                frame_clustered_95ci_m=[float(np.quantile(sample, .025)), float(np.quantile(sample, .975))])


def evaluate(maxiter: int = 70, popsize: int = 12) -> dict:
    positions, fingerprints, link = load_lroom(DEFAULT_QD_ROOT)
    references = reference_indices(positions, SPACING_M)
    reference_positions = positions[references]
    reference_delays = [strongest_delays(fingerprints[int(frame)]) for frame in references]

    # Construct and optimize on development only. Protected test cases are not
    # even materialized until differential evolution has returned.
    development = make_cases(positions, fingerprints, references, TRAIN_CONDITIONS, "development")
    development_tensor = feature_tensor(development, reference_delays)
    scales = np.maximum(np.median(development_tensor.reshape(-1, len(FEATURE_NAMES)), axis=0), 1e-6)
    development_truth = positions[np.asarray([frame for _, frame, _ in development])]
    development_names = [name for name, _, _ in development]
    bounds = [(-4.0, 4.0)] * len(FEATURE_NAMES) + [(-4.0, 2.0), (1.0, 6.0)]
    result = differential_evolution(
        macro_objective, bounds,
        args=(development_tensor, scales, reference_positions, development_truth, development_names),
        seed=SEED, maxiter=maxiter, popsize=popsize, tol=1e-5,
        polish=True, workers=1, updating="immediate",
    )
    frozen_genome = result.x.copy()
    weights, temperature, top_k = decode(frozen_genome)
    genome_sha = hashlib.sha256(frozen_genome.tobytes()).hexdigest()

    # Protected test begins here, after the genome is frozen.
    test = make_cases(positions, fingerprints, references, TEST_CONDITIONS, "test")
    test_tensor = feature_tensor(test, reference_delays)
    evo_estimates, _ = estimates_from_genome(
        frozen_genome, test_tensor, scales, reference_positions
    )
    rows = []
    for index, (condition, frame, query) in enumerate(test):
        truth = positions[frame]
        estimates = {
            "evo_mdp_rank": evo_estimates[index],
            "majdi_mca_eps1.6": published_mca_localize(
                query, reference_delays, reference_positions, epsilon_m=1.6, k=1
            ),
            "majdi_mpurge_map_source_intended": mpurge_map_localize(
                query, reference_delays, reference_positions, p=6, alpha=0.7, k=3,
                normalized_pattern=True, coverage_mode="penalty", cross_mode="order",
            )[0],
        }
        for method, estimate in estimates.items():
            rows.append(dict(
                condition=condition, frame=frame, method=method,
                error_m=float(np.linalg.norm(estimate - truth)), estimate_m=estimate.tolist(),
            ))
    summaries = summarize(rows)
    macro = {
        method: float(np.mean([r["mean_error_m"] for r in summaries if r["method"] == method]))
        for method in sorted({r["method"] for r in summaries})
    }
    return dict(
        evidence_tier=2,
        method="EvoMDP-Rank",
        claim="delay-only evolutionary metric learned on development corruptions and frozen before protected NIST L-Room test",
        causal_deployment_constraint="evolves one transferable ranking rule; never chooses target-specific measurement locations or a post-hoc dense subset",
        protocol=dict(
            scenario="official NIST Q-D L-Room", link=asdict(link), spacing_m=SPACING_M,
            references=references.tolist(), maximum_paths=MAX_PATHS,
            development_conditions=[asdict(c) for c in TRAIN_CONDITIONS],
            protected_test_conditions=[asdict(c) for c in TEST_CONDITIONS],
            development_cases=len(development), protected_test_cases=len(test),
        ),
        evolution=dict(
            algorithm="scipy differential_evolution", seed=SEED, maxiter=maxiter,
            popsize=popsize, evaluations=int(result.nfev), success=bool(result.success),
            message=str(result.message), development_objective=float(result.fun),
            feature_names=list(FEATURE_NAMES), feature_scales=scales.tolist(),
            feature_weights=dict(zip(FEATURE_NAMES, weights.tolist(), strict=True)),
            temperature=temperature, top_k=top_k, frozen_genome=frozen_genome.tolist(),
            frozen_genome_sha256=genome_sha,
        ),
        macro_mean_error_m=macro,
        paired_vs_majdi=[
            paired_bootstrap(rows, "evo_mdp_rank", "majdi_mca_eps1.6"),
            paired_bootstrap(rows, "evo_mdp_rank", "majdi_mpurge_map_source_intended"),
        ],
        summaries=summaries,
        rows=rows,
    )


if __name__ == "__main__":
    output = Path(__file__).with_name("evo_mdp_rank_results.json")
    payload = evaluate()
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "macro_mean_error_m": payload["macro_mean_error_m"],
        "paired_vs_majdi": payload["paired_vs_majdi"],
        "evolution": payload["evolution"],
    }, indent=2))
