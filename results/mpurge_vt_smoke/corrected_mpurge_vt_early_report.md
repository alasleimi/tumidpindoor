# Corrected MPUrge VT-construction suite — early report status

Generated 2026-08-07 for the early-report deadline. This note separates completed evidence from an interrupted full run; it does not promote smoke-test numbers to final claims.

## What is complete

- Implementation: `experiments/four_paper_report/corrected_mpurge_vt_suite.py`
- Protocol/unit tests: `experiments/four_paper_report/test_corrected_mpurge_vt_suite.py`
- Method and paper inventory: `research/four_paper_report/requested_method_inventory.md`
- Completed CUDA smoke artifacts: `research/four_paper_report/corrected_mpurge_vt_smoke/`
- Smoke artifact integrity: all 12 entries in `SHA256SUMS.txt` recomputed successfully (0 mismatches).
- Test result on 2026-08-07: 4/4 tests passed. These cover total `P=5 -> p=2`, composite-dissimilarity conflict priority, both cross-check interpretations/dynamic beta exhaustion, and maximum-cardinality gated assignment.

## Completed smoke results (diagnostic only)

The smoke run used one disclosed replacement scene at two range-noise levels (0 and 0.25 m), 20 sparse survey positions, 6 held-out query positions, only 8 neural epochs, only 12 differentiable-assignment steps, and a 15-by-15 localization grid. The VT gate was 0.30 m. Consequently its one-scene bootstrap intervals are degenerate and the neural rankings are not final evidence.

| Method | VT F1 | Precision | Recall | Chamfer (m) | Hausdorff (m) | Localization mean (m) |
|---|---:|---:|---:|---:|---:|---:|
| Bernoulli/RFS multimodal | 0.9554 | 0.9150 | 1.0000 | 0.0559 | 0.1986 | 0.2864 |
| Beta-marginal survival multimodal | 0.8515 | 0.7521 | 1.0000 | 0.0807 | 0.1882 | 0.2853 |
| AoA/power inverse consensus | 0.7729 | 0.6390 | 1.0000 | 0.0948 | 0.2874 | 0.3044 |
| Corrected PBA, order/components, beta=1 | 0.2572 | 0.2910 | 0.3125 | 8.7439 | 32.8345 | 5.1719 |
| Corrected MPUrge, order/components, beta=1 | 0.2556 | 0.2897 | 0.3125 | 8.6965 | 32.8345 | 5.0178 |
| Cycle-consistent delay | 0.1584 | 0.0905 | 0.6875 | 3.2785 | 21.4026 | 0.9159 |
| Differentiable assignment, BIC multimodal | 0.0556 | 0.0500 | 0.0625 | 8.8376 | 21.4899 | 2.1195 |
| Self-supervised DeepSets multimodal | 0.0000 | 0.0000 | 0.0000 | 13.9164 | 22.7315 | 8.1569 |
| Self-supervised attention multimodal | 0.0000 | 0.0000 | 0.0000 | 14.8455 | 30.7247 | 10.9501 |

The smoke result supports only a triage conclusion: the analytic multimodal RFS, beta-survival, and inverse-consensus branches merit the full comparison. It does **not** justify rejecting permutation-invariant attention or DeepSets, because their smoke schedule was deliberately truncated and there is only one scene. Conversely, it also does not establish the high RFS score as a general result.

## Exact protocol implemented

- The paper's printed total window is fixed at `P=5`, hence the implemented half-window is `p=(P-1)/2=2`.
- Conflicts are resolved in increasing composite dissimilarity order, not index order.
- Both the literal printed cross bracket and the geometric/order-preserving interpretation are evaluated, with both star and conflict-resolved connected-component grouping recorded.
- Transmitter identity remains an observable channel throughout pairing, grouping, support counting, and evaluation.
- Beta is swept dynamically until no candidate survives; raw pre/post-beta candidate sets and memberships are saved.
- VT evaluation first maximizes the number of matches inside the 0.30 m gate, then minimizes distance among maximum-cardinality assignments. Precision, recall, F1, Chamfer, Hausdorff, and downstream set-grid localization error are all recorded.
- Each scene/noise block receives independent survey, query, and receiver-noise realizations. Every method receives the exact same sparse acquisitions and locations.
- Primary methods never receive true VT coordinates/cardinality, ray/source identity, or query truth, and receive no extra dense simulator examples. Same-acquisition noisy delay, power, AoA, Tx channel identity, and array CIR are allowed.
- The legacy supervised VT PointNet is excluded from primary comparison because its 5,000 labelled VT sets and true-cardinality setup exceed this paper-specific acquisition budget. The oracle-cardinality differentiable assignment is stored only as an explicitly excluded diagnostic.

## Full-run status and caveats

The intended full CUDA run is 24 independent scene/noise blocks: 6 scene seeds times noise standard deviations `{0, 0.25, 1, 3}` m, with 36 survey positions, 20 held-out queries, 90 differentiable-assignment steps with two restarts, 120 epochs for each self-supervised set model, and a 31-by-31 localization grid.

The process started at 12:57 local time and terminated without a final result at approximately 13:42 after writing 14 neural checkpoints, corresponding to checkpoint stages for 7/24 blocks (all four noise levels in scene 26080701 and 0, 0.25, and 1 m in scene 26080702). It wrote no `block_metrics.jsonl`, `results.json`, report, or manifest because the current v1 writer aggregates in memory and commits these only after all blocks. Therefore the partial checkpoints are **not** a valid partial benchmark and no numerical claim is drawn from them. The exact exit cause was not captured by the launching shell.

There is also a provenance caveat: the suite source still matches its launch SHA-256 (`e55421dd...`), but the shared `multimodal_receiver.py` was modified concurrently after the run imported it. The launch config records the executed receiver source hash (`7e990931...`), whereas the current file no longer has that hash. The completed smoke artifact manifest itself remains internally intact. A restart should first freeze an executed-source copy and add per-block atomic shards/resume support.

## Report-safe wording

“A leakage-controlled corrected-MPUrge suite was implemented and unit-tested. A one-scene/two-noise CUDA smoke test identified analytic multimodal RFS, survival, and inverse-consensus branches as the strongest candidates, while truncated DeepSets/attention training was inconclusive. The 24-block full run terminated after seven checkpoint stages and produced no aggregate metrics, so only the smoke figures are reported as diagnostic, not as final benchmark results.”
