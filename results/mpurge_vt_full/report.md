# Corrected original-MPUrge VT-construction suite

Status: **FULL_COMPLETE**. This is a disclosed replacement-scene benchmark, not the unavailable original simulator reproduction.

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
| two_sided_vt_registration | - | 0.5717 | 1.0275 | 1.6455 |
| bernoulli_rfs_multimodal | - | 0.5562 | 0.6528 | 1.4383 |
| beta_marginal_survival_multimodal | - | 0.5395 | 0.7766 | 1.4106 |
| diffassign_bic_multimodal | - | 0.5156 | 1.0577 | 1.5711 |
| multipath_bundle_adjustment_noncheating | - | 0.5003 | 2.6358 | 2.0033 |
| aoa_power_inverse_consensus | - | 0.4217 | 0.8241 | 1.6250 |
| diffassign_bic_delay | - | 0.3485 | 3.5305 | 1.7616 |
| corrected_mpurge_order_components | 1.00 | 0.1313 | 6.1565 | 3.2410 |
| corrected_pba_order_components | 1.00 | 0.1303 | 6.3960 | 3.3268 |
| corrected_pba_printed_components | 1.00 | 0.1158 | 5.9749 | 3.3874 |
| selfsup_deepsets_multimodal | - | 0.1140 | 3.6951 | 2.0763 |
| corrected_mpurge_printed_components | 1.00 | 0.1103 | 6.1957 | 3.4667 |
| selfsup_attention_multimodal | - | 0.0773 | 2.7805 | 1.9563 |
| corrected_mpurge_order_star | 1.00 | 0.0475 | 2.8050 | 1.8173 |
| corrected_pba_order_star | 1.00 | 0.0471 | 2.8160 | 1.8044 |
| cycle_consistent_delay | - | 0.0329 | 3.9508 | 2.3876 |
| corrected_mpurge_printed_star | 1.00 | 0.0321 | 3.1931 | 1.9752 |
| corrected_pba_printed_star | 1.00 | 0.0310 | 3.2148 | 1.9251 |

## Feature inputs and exclusions

| Method/family | Inputs | Fit | Applicability |
|---|---|---|---|
| corrected_mpurge | delay/range and observable Tx channel identity | none | native original-MPUrge VT construction |
| corrected_pba | delay/range and observable Tx channel identity | none | native PBA comparator under corrected window/conflict/cross bracket |
| cycle_consistent_delay | delay/range, survey coordinates, Tx identity | none | variable-cardinality delay-only VT construction |
| bernoulli_rfs_multimodal | same-acquisition delay, corrupted power, corrupted AoA, Tx identity | none | sequential survey VT-track construction |
| two_sided_vt_registration | same-acquisition delay, corrupted AoA/power, survey coordinates, Tx identity | none; fixed mutual gates | mutual survey-to-VT and VT-to-survey round-trip registration |
| multipath_bundle_adjustment_noncheating | same-acquisition delay, corrupted AoA/power, known calibration survey poses, Tx identity | per-survey robust BA with held-out-anchor BIC | joint VT and soft-association refinement; survey poses fixed |
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
