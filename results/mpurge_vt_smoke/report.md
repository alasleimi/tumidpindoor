# Corrected original-MPUrge VT-construction suite

Status: **QUICK_SMOKE_COMPLETE**. This is a disclosed replacement-scene benchmark, not the unavailable original simulator reproduction.

## Correctness and information barrier

- Total window `P=5` is explicitly converted to half-window `p=2`.
- Current-iteration conflicts are ordered by the full composite dissimilarity.
- Printed-algebra and geometric-order cross-checks, plus star/component grouping interpretations, are all retained.
- Candidate support and beta filtering never mix transmitter channels; every native beta curve ends with an empty set.
- Primary learned/optimized methods see only the same calibration acquisitions. No VT/ray IDs, true VT cardinality, query truth, dense sampling, or additional simulator-labelled examples are used.
- The oracle-cardinality differentiable-assignment row is explicitly excluded from primary ranking.

## Primary scene-clustered macro results

| Method | Beta | F1 | Chamfer (m) | Localization (m) |
|---|---:|---:|---:|---:|
| bernoulli_rfs_multimodal | - | 0.9554 | 0.0559 | 0.2864 |
| beta_marginal_survival_multimodal | - | 0.8515 | 0.0807 | 0.2853 |
| aoa_power_inverse_consensus | - | 0.7729 | 0.0948 | 0.3044 |
| corrected_pba_order_components | 1.00 | 0.2572 | 8.7439 | 5.1719 |
| corrected_mpurge_order_components | 1.00 | 0.2556 | 8.6965 | 5.0178 |
| cycle_consistent_delay | - | 0.1584 | 3.2785 | 0.9159 |
| corrected_pba_printed_components | 1.00 | 0.1442 | 4.3695 | 0.6892 |
| corrected_mpurge_printed_components | 1.00 | 0.1412 | 4.2370 | 0.6799 |
| corrected_mpurge_order_star | 1.00 | 0.0867 | 2.3526 | 0.3959 |
| corrected_pba_order_star | 1.00 | 0.0860 | 2.3336 | 0.3724 |
| corrected_pba_printed_star | 1.00 | 0.0758 | 2.6123 | 0.7048 |
| corrected_mpurge_printed_star | 1.00 | 0.0756 | 2.6459 | 0.7124 |
| diffassign_bic_multimodal | - | 0.0556 | 8.8376 | 2.1195 |
| diffassign_bic_delay | - | 0.0000 | 10.7496 | 3.5806 |
| selfsup_attention_multimodal | - | 0.0000 | 14.8455 | 10.9501 |
| selfsup_deepsets_multimodal | - | 0.0000 | 13.9164 | 8.1569 |

## Feature inputs and exclusions

| Method/family | Inputs | Fit | Applicability |
|---|---|---|---|
| corrected_mpurge | delay/range and observable Tx channel identity | none | native original-MPUrge VT construction |
| corrected_pba | delay/range and observable Tx channel identity | none | native PBA comparator under corrected window/conflict/cross bracket |
| cycle_consistent_delay | delay/range, survey coordinates, Tx identity | none | variable-cardinality delay-only VT construction |
| bernoulli_rfs_multimodal | same-acquisition delay, corrupted power, corrupted AoA, Tx identity | none | sequential survey VT-track construction |
| aoa_power_inverse_consensus | same-acquisition delay, corrupted power, corrupted AoA, Tx identity | none | multimodal survey VT construction |
| beta_marginal_survival_multimodal | same-acquisition delay, corrupted power, corrupted AoA, missing detections | none | multimodal existence-filtered VT construction |
| diffassign_bic_delay | delay/range, survey coordinates, Tx identity | per-survey self-supervised reconstruction | unknown-cardinality differentiable VT construction |
| diffassign_bic_multimodal | same-acquisition delay and corrupted AoA | per-survey self-supervised reconstruction | unknown-cardinality multimodal differentiable VT construction |
| diffassign_multimodal_oracle_cardinality_ablation | same-acquisition delay and corrupted AoA | per-survey reconstruction | labelled diagnostic only; excluded from primary |
| selfsup_deepsets_multimodal | same-acquisition delay, corrupted power/AoA, survey coordinates | only range+AoA set reconstruction on this calibration survey | self-supervised variable-existence VT output |
| selfsup_attention_multimodal | same-acquisition delay, corrupted power/AoA, survey coordinates | only range+AoA set reconstruction on this calibration survey | permutation-invariant attention VT output |
| old_supervised_vt_pointnet | not run | 5,000 simulator-labelled wall-VT sets in legacy code | excluded: prohibited extra labels/examples and fixed true cardinality |
| CAEZ_probability_MLP | CIR magnitude | supervised receiver-coordinate labels | excluded here: receiver coordinate output, no VT-set output adapter without labels |
| DICHASUS_ADP_8NN | coherent CIR | nonparametric receiver map | excluded here: receiver-coordinate retrieval, not VT construction |

Raw candidate memberships, supports and associations are in the gzip JSONL artifacts named in `artifacts`; query estimates are stored separately.
