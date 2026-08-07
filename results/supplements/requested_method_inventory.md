# Requested-method implementation and dependency inventory

Date audited: 2026-08-07

This inventory resolves the method names used in the discussion to the code and frozen artifacts that actually exist in the workspace. It records what each implementation consumes at inference, how it was fitted, what it returns, and where simulator or ground-truth information enters. A result being executable does **not** automatically make it comparable to every Majid protocol.

## Protocol key used below

- **VT construction:** original MPUrge/bachelor-thesis task: estimate a set of virtual-transmitter (VT) coordinates and score it by one-to-one precision/recall/F1.
- **Static delay map:** MPUrge-MAP task: localize a single query from its unordered delay/range set against surveyed reference fingerprints.
- **SMART-LEA:** known- or survey-estimated-VT, multi-transmitter iterative localization.
- **CNN fingerprinting:** supervised direct localization from a packed fingerprint.
- **Multi-anchor VT:** range plus global arrival direction from several anchors, with a surveyed/fitted VT map.
- **Sequential extension:** an ordered trajectory/session; past observations are allowed, future observations are not.

## Alias and integrity summary

| Requested name | Exact implemented object | Frozen full artifact | Integrity / portability verdict |
|---|---|---|---|
| LEA-4 | Four calls to `lea_iteration`, normally SMART matching (`mpurge`, `p=2`, `alpha=0.5`) | `research/paper_protocol_replications/smart_lea_results_calibrated_vt.json` | Analytic and deployable once a VT map exists; replacement-scene reconstruction is not the unavailable original project. |
| Assigned-VT inverse consensus | Coarse joint VT MAP followed by path/VT inversion and robust proposal consensus | `experiments/research_level/spatial_joint_vt_inverse_consensus_results.json` | No query truth or ray ID in the estimator; needs range, global AoA, multi-anchor VT fields, and a coarse pose. |
| CAEZ-5G probability-map MLP | Published 4368-to-484 probability network, memory-safe batch adaptation | `research/deep_campaign/benchmarks/caez_5g_official_protected.json` | Clean public measured-data supervised run; not a pretrained drop-in for Majid delays. |
| DICHASUS coherent ADP 8-NN | Exact global ADP top-64, spatial NMS, soft barycentric top-8 | `experiments/research_level/dichasus_adp_barycentric_results.json` | Clean public measured coherent-CIR run; requires complex calibrated CIR, not delay sets. |
| Beta-marginal VT survival | Survey-built VT/visibility map plus Beta-integrated detection likelihood | `experiments/research_level/canonical_beta_survival_portfolio_results.json` | Observable at test time, but uses delay **and global AoA** plus survey positions. It is not a delay-only MPUrge comparison. |
| Analytic-anchor residual graph | Fraunhofer timing-wKNN coordinate plus learned multi-anchor CIR residual | `research/deep_campaign/benchmarks/fraunhofer_{5g,uwb}_residual_test.json` | Clean public measured-data supervised run; must be retrained for another sensor/schema. |
| RRLE / map-aware MoE | Frozen kNN/Ridge/ExtraTrees experts plus OOF-trained router | `research/deep_campaign/experiments/novel_view/rrle/{5g,uwb}/results.json` | Clean measured-data MoE experiment. Architecture transfers; weights and experts do not. |
| Survival-CIR+ToA ExtraTrees | Eight-anchor survival-tail/ToA feature vector and five-seed ExtraTrees ensemble | `research/deep_campaign/experiments/open_dataset_bundle/ipin2023_track7/results.json` | Clean public measured-data supervised run; requires raw CIR and ToA. |
| Causal analytic HMM | HMM over the full corrected-MPUrge RP score vector | `research/campaign_8plus4/g2_replication_results.json` | Strongest clean sequential result; causal, frozen, and no query coordinate enters prediction. |
| Two-sided VT registration | **Composite alias**, mapping by differentiable assignment and localization by assigned-VT inverse consensus | The two component artifacts above | Both halves ran, but not as one end-to-end benchmark on one dataset. Do not describe it as a single jointly tested model. |
| “Non-cheating multipath BA” | No such deployable artifact exists. Closest object is controlled G3 joint BA. | `research/campaign_8plus4/map_repair_bundle_suite_results.json` | G3 uses simulator-side associations, truth-derived noisy odometry, true height, and the first true position. It is an oracle-associated controlled trial, not non-cheating BA. |
| Symmetric Chamfer | Near-pure fixed Chamfer control with top-3 interpolation | `experiments/research_level/evo_mdp_rank_ablation_results.json` | Clean delay-only static control; no learned network and no test truth input. |
| Differentiable assignment | Directly optimize four VT coordinates per transmitter with symmetric soft-min set loss | `research/paper_protocol_replications/mpurge_vt_results.json` | Unsupervised with respect to VT truth; full replacement-scene run. Requires surveyed anchor coordinates and per-transmitter indirect ranges. |
| Genuine cross-attention | Query paths attend to each candidate RP's reference paths, guarded by corrected MPUrge | `research/vt_gpu_followups/mpurge_cross_attention_full/mpurge_candidate_cross_attention_results.json` | Genuine delay-only attention, no path IDs/power/AoA; modest confirmed static gain. |
| “Old weird attention / PointNet” | Several different legacy objects; see section 15 | Several artifacts | Must not be treated as one model. One pooled `CandidateAttention` was mislabeled cross-attention and used unrealistically clean power/context channels; the delay-sensor ablations and the later analytic reranker are separate. The corrected multimodal rows retain power after adding receiver noise. |

## 1. LEA-4 / SMART-LEA-4

**Aliases.** `LEA-4`, `SMART-LEA-4`, and `smart_lea_mpurge_4iter` refer to the same four-iteration refinement when the SMART/MPUrge matcher is used. `smart_lea_source_intended_4iter` is a separate cross-term-reading variant in the canonical portfolio.

**Code and artifacts.** The reusable implementation is `lea_iteration` / `lea_refine` in `experiments/research_level/majdi_paper_methods.py`. The paper-density runner is `experiments/paper_protocol_replications/run_smart_lea_protocol.py`; its calibration-fair full artifact is `research/paper_protocol_replications/smart_lea_results_calibrated_vt.json`. The exact common-row NIST invocation is in `experiments/research_level/majdi_paper_structured_portfolio.py`, with `experiments/research_level/majdi_paper_structured_portfolio_results.json` and the aggregate `experiments/research_level/majdi_full_portfolio_benchmark_results.json`.

**Input and output.** Each iteration consumes an initial receiver coordinate, a VT coordinate set, and the query's measured range set. It recalculates ranges from the current receiver to every VT, matches the unordered measured and recalculated sets, then runs nonlinear least-squares trilateration using the matched VTs. The output is a refined receiver coordinate plus the four match histories. With fewer than dimension-plus-one matches it returns the prior coordinate unchanged.

**Fit/training.** There is no learned model. Matcher, `p`, `alpha`, cross-mode, iteration count, survey spacing, and VT-building method are protocol choices. The full reconstruction uses a disclosed 25 m by 25 m replacement room, four transmitters, 10,000 dense survey points, nested 0.25/0.5/1.5/2.5 m maps, 1,000 locked queries, and clean / 0.25 m range-noise conditions.

**Simulator/truth dependencies.** Inference does not read a simulator or query truth. It does require a VT map. The full paper runner generated replacement image-source fingerprints; the calibration-fair artifact estimates VTs from calibration data instead of handing the evaluator's true wall-image coordinates to the localizer. It is therefore an analogous reconstruction, not an exact reproduction of Majid's missing simulator project.

**Applicability and adaptation.** Directly applicable to SMART-LEA and to any multi-transmitter experiment with a usable VT map. It is eligible in MPUrge-MAP when the VT map is built only from the same allowed calibration acquisitions; handing it a simulator-truth VT map would instead violate the information budget. New-scene adaptation means rebuilding/calibrating VTs and optionally selecting matching constants on development data; there is no neural retraining.

## 2. Assigned-VT inverse receiver consensus

**Aliases.** `assigned-VT inverse consensus`, `assigned_vt_receiver_consensus`, `frozen_assigned_vt_receiver_consensus`, and “receiver-hypothesis consensus” are the same family. It is the localization half of “two-sided VT registration.”

**Code and artifacts.** Development/transfer code is `experiments/research_level/sionna_2d_vt_inverse_consensus.py`. The frozen NIST evaluation is `experiments/research_level/spatial_joint_vt_inverse_consensus.py` with `experiments/research_level/spatial_joint_vt_inverse_consensus_results.json`. Transfer artifacts include `experiments/research_level/sionna_2d_vt_inverse_consensus_results.json`, `experiments/research_level/sionna_2d_vt_inverse_consensus_knife_confirmation_results.json`, and `experiments/research_level/lroom_frozen_vt_inverse_consensus_results.json`.

**Input and output.** For every non-clutter Hungarian query-path/VT match it forms

`receiver proposal = VT centre - measured range * global-AoA unit vector`.

Proposals across anchors are robustly combined (the frozen NIST configuration uses geometric median, 0.25 m trust radius, and 0.125 m activation radius) around a coarse joint trajectory-MAP estimate. Output is a continuous receiver/route coordinate and an audit of accepted matches.

**Fit/training.** No neural training. Estimator type and radii were selected on a one-screen Sionna development split, survived two Sionna confirmations, and were frozen before NIST SpatialSharing. The NIST test uses three links/anchors, 2 m survey spacing, and second-half non-reference frames.

**Simulator/truth dependencies.** Test inference uses measured range, measured global direction, fitted/surveyed Gaussian VT fields, candidate assignment, and a coarse pose. It does not consume query coordinates, localization errors, ray IDs, or interaction labels. NIST/Sionna generate the evaluation observations and supply truth only for scoring. Survey positions are required to build the VT fields. The result is unusually low (NIST macro about 0.01062 m), so the saturated route/grid and idealized global-AoA assumptions must travel with the number.

**Applicability and adaptation.** Direct for multi-anchor range+AoA VT localization; conditionally useful in SMART-LEA, MPUrge-MAP, or CNN protocols when noisy global AoA is synthesized or measured at the same allowed acquisitions and the VT field is survey-built. Adaptation requires a new surveyed VT field and possibly development-selected radii, not gradient retraining.

## 3. CAEZ-5G probability-map MLP

**Code and artifacts.** `experiments/deep_campaign/caez_5g_benchmark.py`; protocol/training artifacts under `research/deep_campaign/benchmarks/caez_5g_*`, especially `caez_5g_protocol_freeze.json`, `caez_5g_official_training.json`, `caez_5g_official_reimplementation_seed0.pt`, `caez_5g_official_protected.json`, and `caez_5g_official_protected_predictions.npz`. The pinned upstream implementation is in `research/deep_campaign/external/neural-positioning`.

**Input and output.** Input is the public CAEZ complex channel tensor (4 by 4 by 273), converted to a unit-Frobenius CSI-magnitude vector of length 4,368. The network is 4368-512-512-512-512-484, with BatchNorm after the first layer, ReLU, and a softmax over a 22 by 22 spatial grid. Four-nearest feasible barycentric target probabilities are used. Output is a 484-cell probability map, posterior-mean x/y coordinate, and posterior variance.

**Fit/training.** Supervised on 269,784 of 337,230 labelled public samples in the official seed-0 80:20 split; Adam at 1e-4, 50 epochs. Batch 2,048 replaced the published batch 10 for memory/time and was preregistered. The last 500 samples remained protected until the checkpoint existed. Protected mean error is 0.028415 m versus the paper's 0.007 m context.

**Simulator/truth dependencies.** None: public measured CSI and surveyed labels only. Position labels are required for training but never passed at inference; no ray identity exists.

**Applicability and adaptation.** It is evidence that probability-map supervision is practical, not a pretrained Majid localizer. Direct use requires CAEZ's antenna/subcarrier shape and grid. Transfer to Majid would require a new delay/CIR frontend, a new output grid, and full supervised retraining on labelled fingerprints.

## 4. DICHASUS coherent ADP barycentric 8-NN

**Code and artifacts.** `experiments/research_level/dichasus_adp_barycentric_localizer.py`, artifact `experiments/research_level/dichasus_adp_barycentric_results.json`, and exact top-64 caches in `research/DICHASUS-005x/dichasus-0057-vs-0053-0054-0055-adp-top64.npz` (plus the analogous 0055 selection cache). Minimal coherent features are produced by `experiments/research_level/dichasus_complete_runs_frozen_evaluation.py`.

**Input and output.** Input is calibrated complex 13-tap impulse-response data across antennas. ADP dissimilarity coherently correlates antenna vectors per tap, normalizes by their power, converts similarity to distance, and sums over taps. Exact global top-64 retrieval is followed by 0.01 m spatial non-maximum suppression and a locally scale-normalized softmax over eight neighbours at temperature 0.25. Output is the barycentric x/y coordinate.

**Fit/training.** Nonparametric. Complete runs 0053+0054 form the selection map, complete 0055 chooses neighbour count/radius/temperature, then 0053+0054+0055 form the final map and untouched complete 0057 (5,001 queries) is tested. Mean is 0.039128 m versus 0.043457 m for coherent ADP 1-NN.

**Simulator/truth dependencies.** None. This is official measured DICHASUS CSI/CIR with survey coordinates. Labels choose the development hyperparameters and score test, not inference.

**Applicability and adaptation.** Direct only where coherent complex multi-antenna CIR is available. It does not operate on Majid's scalar delay set. It can be rerun without neural training on a new coherent fingerprint map; k/radius/temperature should be reselected on a development run.

## 5. Beta-marginal VT survival

**Aliases.** `beta_marginal_detection_survival`, Beta survival, and detection-marginal survival. Do not confuse it with the later rejected sector-marginal follow-up.

**Code and artifacts.** Canonical wrapper `experiments/research_level/canonical_beta_survival_portfolio.py`; implementation in `experiments/research_level/detection_marginal_survival_lroom.py` and `experiments/research_level/soft_vt_survival_likelihood.py`. Frozen result `experiments/research_level/canonical_beta_survival_portfolio_results.json`; selection source `experiments/research_level/detection_marginal_survival_lroom_results.json`.

**Input and output.** Survey fingerprints with position, range, and global AoA are converted into VT proposals `x_ref + r*u`, clustered into persistent modes, and used to estimate route visibility. For each candidate receiver, one-to-one observed-path/VT assignment, missed expected VTs, and clutter contribute an NLL. A query-level detection probability is integrated under a Beta prior using 12-node Gauss-Legendre quadrature. Output is the minimum-NLL candidate coordinate, NLL margin, and posterior mean detection rate.

**Fit/training.** No network. Survival/map parameters and the selected Beta(2,1) prior were chosen on first-half L-Room development and frozen. The canonical artifact has 89 queries in each of eight conditions (712 rows) and a 0.151826 m macro mean.

**Simulator/truth dependencies.** Test inference forbids query coordinate/error, condition identity, geometry, Mpc/ray identity, and simulator calls. NIST Q-D supplies observations and scoring truth. Survey positions and global AoA are material inputs. Current synthetic corruptions perturb ranges and path presence but do not model coupled AoA/range error, which limits real-world claims.

**Applicability and adaptation.** Direct for static survey-based delay+global-AoA VT localization and eligible for MPUrge-MAP when both observables are noisily extracted at the same allowed survey/query acquisitions. Adaptation requires rebuilding VT/visibility maps and selecting survival priors on development data; no neural retraining.

## 6. Analytic-anchor residual graph

**Aliases.** Fraunhofer residual graph, deterministic residual graph, heteroscedastic residual graph. “Analytic anchor” here means timing-wKNN, not a VT ground-truth anchor.

**Code and artifacts.** `experiments/deep_campaign/fraunhofer_residual_graph.py` with feature preparation in `experiments/deep_campaign/fraunhofer_common.py`. Selection/test artifacts are `research/deep_campaign/benchmarks/fraunhofer_{5g,uwb}_residual_selection.json`, `fraunhofer_{5g,uwb}_residual_test.json`, prediction NPZs, and five seed checkpoints named `fraunhofer_*_residual_test__*_seed*.pt`.

**Input and output.** Per available anchor: 64 relative-dB CIR bins, log energy, peak fraction, RMS delay spread, TD distance, TD_OFFSET distance, and availability. A shared anchor encoder with learned anchor embedding is symmetrically aggregated by attention-weighted mean plus max. The timing-wKNN coordinate is an explicit base; the model predicts a bounded two-dimensional correction (and the heteroscedastic version also predicts uncertainty). Output is corrected x/y.

**Fit/training.** Supervised on labelled Fraunhofer train bursts. The analytic base is generated out-of-fold with blocked/purged folds; architecture/checkpoint epoch is selected on development, refit, then five seeds are ensembled on protected test. Deterministic residual graph: 5G 0.448217 m versus timing anchor 0.461008 m; UWB 0.196883 m versus 0.402995 m.

**Simulator/truth dependencies.** None: measured data plus training positions. No ray/interactions. Query labels are used only for loss/evaluation.

**Applicability and adaptation.** A strong template for a Majid hybrid if each anchor has CIR/timing features and labelled calibration bursts. Existing weights are schema-specific. A new frontend and complete supervised retraining are required for scalar delay sets or different anchor counts/features.

## 7. RRLE / map-aware mixture of experts

**Aliases.** RRLE and map-aware MoE refer to `nvs_rrle.py`; this is not the analytic agreement gate used in the NIST campaign.

**Code and artifacts.** `experiments/deep_campaign/nvs_rrle.py` with common feature projection in `experiments/deep_campaign/nvs_common.py`. Frozen artifacts are `research/deep_campaign/experiments/novel_view/rrle/5g/{selection.json,results.json,locked_predictions.npz,checkpoints/*}` and the equivalent `uwb` directory.

**Input and output.** PCA-64 projection of all per-anchor Fraunhofer CIR/timing features, missing-anchor fraction, three expert x/y predictions (weighted kNN, Ridge, ExtraTrees), three uncertainty estimates, and three pairwise expert disagreements. A small router predicts expert weights, optionally top-k truncated, and returns their convex x/y mixture plus gate weights.

**Fit/training.** Experts are independently selected and fitted. Five blocked OOF folds with a 20-row purge create leakage-safe training predictions for the router; experts stay frozen while the router trains. It is then refit with the frozen configuration and five seeds on locked test. RRLE mean: 1.137864 m on 5G versus selected kNN 1.355058 m; 0.777229 m on UWB versus kNN 0.925826 m.

**Simulator/truth dependencies.** None. Measured labelled Fraunhofer data only. The router's labels are training coordinates; test truth never enters routing.

**Applicability and adaptation.** The architecture is directly relevant to combining corrected MPUrge, Chamfer, PointNet, VT, and other heads, but these frozen experts/router are not. Adaptation requires leakage-safe OOF predictions and labelled calibration data, then refitting every expert and the router.

## 8. Survival-CIR + ToA ExtraTrees

**Aliases.** `survival_toa_tree`, survival-tail CIR+ToA tree, or Survival-CIR+ToA ExtraTrees.

**Code and artifacts.** `experiments/deep_campaign/open_dataset_bundle/ipin2023_t7_benchmark.py`; frozen result `research/deep_campaign/experiments/open_dataset_bundle/ipin2023_track7/results.json`; predictions and parsed caches in the same directory.

**Input and output.** One complete burst from eight anchors: each anchor has 128 complex CIR values, ToA/window data, and visibility. Magnitude energy is normalized and reverse-cumulatively summed into a survival tail, binned to 16 values per anchor, then concatenated with power, receiver-centred relative ToA in metres, and visibility. A five-seed ExtraTrees ensemble outputs x/y.

**Fit/training.** From the public IPIN 2023 Track 7 data, 12,000 whole training bursts were retained; the first 9,600 fit and last 2,400 chronological bursts selected leaf size, then all 12,000 refit. The protected official test contains 4,000 non-time-overlapping bursts. Mean is 1.702086 m versus 1.884070 m for the amplitude+ToA tree; paired gain 0.181984 m with block CI [0.120354, 0.246209].

**Simulator/truth dependencies.** None. Public measured data and labels. Dataset files are publicly downloadable, but the audit found no explicit reuse licence statement.

**Applicability and adaptation.** Direct for multi-anchor raw CIR+ToA. Not a drop-in for an unordered scalar-delay MPUrge fingerprint. It is readily retrainable on a similarly structured labelled dataset; a delay-only version would be a new representation/experiment.

## 9. Causal analytic HMM / particle bank

**Aliases.** G2, analytic HMM, HMM particle bank, and causal likelihood filter.

**Code and artifacts.** Core functions `rank_likelihood`, `transition_matrix`, and `hmm_predict` are in `experiments/campaign_8plus4/sequence_session_suite.py`. The fresh prospective confirmation is `experiments/campaign_8plus4/g2_replication_confirmation.py`, frozen by `research/campaign_8plus4/g2_replication_freeze.json`, with result `research/campaign_8plus4/g2_replication_results.json` and raw rows `g2_replication_rows.jsonl`. The earlier internal development/holdout artifact is `sequence_session_suite_results.json` and must not replace the fresh confirmation.

**Input and output.** At each time step it consumes corrected MPUrge's score vector over the same 20 surveyed RPs. Score ranks become an emission `exp(-beta*rank_fraction)`. A Gaussian transition over RP-coordinate distance propagates the previous belief; only the current emission then updates it. Output is the posterior-weighted coordinate of the top-k RP states.

**Fit/training.** No gradient training. Early physical frames selected beta=8, transition sigma=0.6 m, stay=0, top-k=5. Those values and source/checkpoint hashes were frozen before the three shifted 200-frame confirmation segments were opened. On 1,350 rows / 270 physical frames it reached 0.380196 m versus PointNet 0.812053 m and corrected MPUrge 0.965374 m.

**Simulator/truth dependencies.** Predictor inputs are only past belief, current analytic scores, and surveyed RP coordinates. No query coordinate or future observation enters prediction. NIST Q-D generated the fixed confirmation data and truth is used only for scoring. Temporal order is a real extra information budget, so it cannot be placed in a static single-shot table without qualification.

**Applicability and adaptation.** Directly wraps any MPUrge-style method that exposes a score over fixed RPs on a sequential route. No retraining; transition/emission constants can be selected on an earlier trajectory. For a new RP geometry, recompute the transition matrix.

## 10. Two-sided VT registration

This name is a **research-level composite**, not a class/function or one frozen run.

- **Mapping side:** differentiable assignment (section 13) aligns the unordered ranges predicted by candidate VTs to the unordered indirect ranges observed jointly at surveyed anchors.
- **Localization side:** assigned-VT inverse consensus (section 2) assigns query paths to mapped VT modes, inverts each match into a receiver proposal, and robustly registers those proposals across anchors.

Both halves were implemented and executed, but on different task/data protocols. No artifact constructs VTs with the differentiable mapper and then feeds exactly those estimates into assigned-VT consensus for one locked end-to-end test. Therefore it is valid to say “both sides worked separately,” but not “the end-to-end two-sided system achieved 0.01062 m.”

The end-to-end version is adaptable in principle without supervised labels, but needs per-transmitter survey ranges for mapping and query range+global-AoA observations for localization. It also needs a development-only policy for matching/clutter/trust radii.

## 11. Multipath bundle adjustment: requested non-cheating version versus implemented G3

**Exact finding.** There is no deployable “non-cheating multipath BA” implementation/artifact in the workspace. The only matching code is `experiments/campaign_8plus4/map_repair_bundle_suite.py`; result `research/campaign_8plus4/map_repair_bundle_suite_results.json`, raw rows `map_repair_bundle_suite_rows.jsonl`.

**What G3 optimizes.** All trajectory x/y states, all VT x/y corrections, and a clock nuisance per frame under range, odometry, first-anchor, VT-map, and clock priors. It outputs the whole trajectory, corrected VT map, and clock sequence. On a disclosed affine+translation map mismatch it reduced fixed-map bundle mean 0.799814 m to 0.053342 m.

**Why it is not the requested non-cheating BA.** Four dependencies are material:

1. `associate()` reads each query observation's simulator-provided `virtual_transmitters_m` and assigns it to the nearest survey truth-map VT. Those are fixed simulator-side path associations.
2. Odometry is constructed as the difference of true query x/y plus synthetic Gaussian noise.
3. True query height `z` is supplied for every frame.
4. The first true query x/y is supplied as a 0.01 m anchor.

Later x/y targets do not directly enter the least-squares residual, which is why the artifact honestly calls this a controlled trial, but truth-derived odometry and oracle path/VT identities prevent a deployable leakage-free claim.

**Applicable protocols and adaptation.** It is a useful positive control for T2/self-healing maps and a demonstration that joint states can repair a warped VT map. It does not currently apply to a real Majid experiment unless real odometry/height and an observable association method replace those oracles. There is no training; after replacing those inputs, map/clock/odometry prior scales would need development selection and a new locked evaluation.

## 12. Symmetric Chamfer delay-set localizer

**Code and artifacts.** Feature equation in `experiments/research_level/evo_mdp_rank.py`; protected interpretation control in `experiments/research_level/evo_mdp_rank_ablation.py`; result `experiments/research_level/evo_mdp_rank_ablation_results.json`, included as `symmetric_chamfer_control` in `experiments/research_level/majdi_full_portfolio_benchmark_results.json`.

**Input and output.** The query and each RP are strongest-nine sorted range sets. Their symmetric set discrepancy is one half the mean query-to-reference nearest delay plus one half the mean reference-to-query nearest delay. RPs are ranked and the top three coordinates are exponentially interpolated at temperature 0.03.

**Important exactness note.** The artifact's `chamfer_only` genome uses logit 4 for Chamfer and -4 for the seven other features, so it is a **near-pure** fixed Chamfer control rather than mathematically zero weight on every other feature. Development median feature scales are also applied. This is the exact object behind the reported 0.499199 m macro.

**Fit/training and dependencies.** No neural training. Development rows provide feature scales; the control was run post-freeze on the protected rows and explicitly was not selected from their results. Inference needs only query delays, surveyed RP delay sets, and RP coordinates. NIST produces data/truth; no AoA, power, ray ID, or query truth enters the estimator.

**Applicability and adaptation.** Direct clean comparator for static delay-only MPUrge-MAP. It can be run on any delay map without retraining. For a strict mathematically pure Chamfer deployment, set the seven other weights exactly to zero and re-freeze interpolation choices on development data rather than silently treating the near-pure artifact as exact.

## 13. Differentiable VT assignment

**Aliases.** `challenger_diffassign`, differentiable assignment, and soft set alignment.

**Code and artifacts.** `diffassign` in `experiments/paper_protocol_replications/run_mpurge_vt_protocol.py`; full result `research/paper_protocol_replications/mpurge_vt_results.json`.

**Input and output.** For one known physical transmitter, it consumes surveyed anchor x/y coordinates and four indirect range observations at every anchor. Four candidate VT x/y coordinates are parameterized inside a bounded box. For each anchor, predicted candidate-VT ranges are compared with observed ranges in both directions; log-sum-exp at temperature 0.08 supplies differentiable soft minima. Adam, 350 steps, and six random restarts minimize only this symmetric reconstruction loss. Output is four VT coordinates; the four transmitters are concatenated to 16.

**Fit/training.** Per-instance unsupervised optimization, not dataset training. The full replacement-scene protocol has 64 survey positions, four physical transmitters, three realizations, and sigma {0, 0.25, 1, 3} m. One-to-one 0.2 m detection F1 is 0.917 clean and 0.896 at sigma 0.25 m, but collapses at larger noise.

**Simulator/truth dependencies.** True VT positions are used only for evaluation. The loss sees anchor coordinates and indirect ranges. The runner knows physical-transmitter positions to remove the direct path and evaluates each transmitter separately; a real deployment needs an observable way to separate transmitters and identify/remove LoS. The observations themselves come from a disclosed rectangular replacement generator.

**Applicability and adaptation.** Direct for original MPUrge VT construction, not a receiver localizer by itself. It can be optimized on a new survey without neural retraining; bounds, candidate count, temperature, robust loss/noise handling, and LoS policy need adaptation.

## 14. Genuine candidate query-to-reference cross-attention

**Aliases.** `mpurge_candidate_cross_attention`, genuine candidate cross-attention, path-to-path cross-attention. It is **not** the old `CandidateAttention` class despite the similar result label.

**Code and artifacts.** `experiments/vt_gpu_followups/mpurge_candidate_cross_attention_gpu_benchmark.py`; frozen selection `research/vt_gpu_followups/mpurge_cross_attention_full/frozen_selection.json`; result `research/vt_gpu_followups/mpurge_cross_attention_full/mpurge_candidate_cross_attention_results.json`; raw rows and three checkpoints in the same directory.

**Input and output.** Query delays are Q; every shortlisted RP's reference delays are K/V. Shared path encoders have no positional encoding. Four relative pair features (signed/absolute raw and centred delay differences) bias the attention logits. Corrected MPUrge scores and match coverage enter analytic features. The learned centred residual modifies the analytic candidate score through a validation-selected strength/gate; strength zero is exactly corrected MPUrge. Top-k adjusted candidate coordinates yield the x/y output.

**Fit/training.** 20 NIST L-Room survey RPs, 60 realizations times seven patterns = 8,400 labelled survey augmentations. Whole-RP grouped validation selects shortlist/top-k/residual strength/ambiguity threshold; source RPs are not mixed across folds. Three fresh seeds (113,127,151), CUDA full run, then 450 matched locked rows. Mean 0.976577 m versus corrected MPUrge 1.033754 m; physical-frame blocked gain CI [0.002815, 0.116400]. P90 is worse, so the gain is modest rather than universal.

**Simulator/truth dependencies.** Delay ranges, masks, RP coordinates, analytic scores, and survey labels only. No power, AoA, path order/index, ray identity, or test coordinate. NIST data generation and target scoring remain external to the predictor. The shared second-half screen is matched but no longer pristine, as the artifact states.

**Applicability and adaptation.** Directly relevant to static delay-only MPUrge-MAP. It must be retrained/refrozen for a new survey/reference layout; the analytic skip path makes safe adaptation possible. It is not directly a VT-construction or SMART-LEA method.

## 15. PointNet and the legacy “weird attention” aliases

There are four materially different objects. Reports should name which one is meant.

### 15.1 Delay-sensor-ablation direct PointNet / self-attention regressors

**Code.** `experiments/vt_gpu_followups/majdi_delay_set_gpu_benchmark.py`, class `DelaySetRegressor`.

**Models/input/output.** `range_all_pointnet` embeds each scalar delay, max-pools, adds log-cardinality, and directly regresses x/y. `range_all_attention` applies two masked four-head self-attention layers to the query delay set without positional encoding, then mean+max pools and directly regresses x/y. These models use delays only—no power/context leak.

**Training/artifacts.** Same 8,400 survey augmentations and whole-reference grouped validation as the NIST CNN-matched screen. Canonical PointNet control is `research/vt_gpu_followups/majdi_delay_set/majdi_delay_set_gpu_benchmark_3seed_exact.json` (1.072569 m); deterministic attention rerun is `majdi_delay_set_attention_deterministic_3seed.json` (1.247294 m). They were weak because each had to infer an absolute coordinate from only 20 physical survey labels and did not compare the query to each candidate fingerprint. They are retrainable but are not the architecture to call “genuine cross-attention.”

### 15.2 Old pooled `CandidateAttention` and paper-protocol PointNet

**Code.** `SetEncoder`, `PointNetRegressor`, and `CandidateAttention` in `experiments/paper_protocol_replications/common.py`, used by `run_mpurge_map_protocol.py`, `run_smart_lea_protocol.py`, and `run_cnn_protocol.py`.

**Exact architecture.** `CandidateAttention` separately mean/max-pools the entire query and each reference set, concatenates query encoding, reference encoding, their absolute difference, candidate coordinate, and optional context, and passes that through an MLP scorer. There is no query-path-to-reference-path attention operation. The historical output name `challenger_candidate_cross_attention` is therefore misleading. PointNet directly regresses coordinates from the same pooled encoding.

**Integrity issue.** Packed path tokens contain normalized range, deterministic power, transmitter ID, and validity. CNN variants also receive AP/configuration context. The robustness conditions corrupt ranges/path presence without a matched physical corruption of deterministic power and all configuration channels. Consequently the full artifacts `mpurge_map_results.json`, `smart_lea_results_calibrated_vt.json`, and `cnn_protocol_results_column_major.json` are implementation records, but those old pooled-attention/PointNet rows should not enter a fair noisy-observation leaderboard in that form. The architectures remain eligible and are being retrained after power, direction, and CIR are generated and corrupted as physical observations at the same acquisition locations.

### 15.3 Analytic-anchored MPUrge PointNet reranker

**Code/artifact.** `experiments/vt_gpu_followups/mpurge_pointnet_hybrid_gpu_benchmark.py`; frozen result `research/vt_gpu_followups/mpurge_pointnet_hybrid/mpurge_pointnet_hybrid_evidence_guard_gpu_benchmark_results.json` and three-seed checkpoints.

**Input/output.** It independently PointNet-embeds unordered query and shortlisted RP delay sets, adds 16 analytic MPUrge match diagnostics, and reweights the top eight candidate RP coordinates. Output is a convex RP combination blended with the exact corrected-MPUrge coordinate; no-evidence cases return the exact analytic fallback. It does not use power or AoA.

**Training/result.** 8,400 survey augmentations, source-RP leave-one-out candidate maps, whole-RP validation, three seeds, and the same 450 locked rows. The adaptively designed evidence-guard proposal is 0.832129 m; the stricter validation-bootstrap guard is 0.915549 m versus corrected MPUrge 1.033754 m. It is the relevant PointNet-style MPUrge improvement, but the 0.8321 design was refined after an initial screen and should be labelled accordingly. New maps require supervised refit/refreeze.

### 15.4 Original-MPUrge VT-set PointNet

**Code/artifact.** `VTPointNet`, `pointnet_examples`, and `train_pointnet_set` in `experiments/paper_protocol_replications/run_mpurge_vt_protocol.py`; rows in `research/paper_protocol_replications/mpurge_vt_results.json` under `challenger_pointnet_set`.

**Input/output/training.** Input tokens are survey-anchor x/y, a measured range, and validity; output is four VT x/y coordinates per transmitter with a set-Chamfer loss. It is trained on 5,000 simulator-generated labelled VT sets for 100 epochs, then evaluated on the related rectangular replacement family. F1 is 0.438 clean and 0.375 at sigma 0.25 m.

**Comparability.** This model has a labelled synthetic truth source that Majid's unsupervised VT construction and differentiable assignment do not receive. It is a transfer experiment, not a fair primary VT-construction competitor. Retraining is possible if a credible simulation/domain-randomization source is accepted; otherwise it cannot be adapted from unlabelled real survey ranges alone.

## Practical adaptation matrix

| Method | Can run unchanged on a new Majid map? | What must be rebuilt or retrained? |
|---|---|---|
| LEA-4 | Algorithm yes; result no | VT map; optionally matching constants. |
| Assigned-VT inverse consensus | Only with matching range+global-AoA schema | VT Gaussian fields, coarse localizer, radii audit. |
| CAEZ probability MLP | No | Input frontend/grid and full supervised network. |
| DICHASUS ADP 8-NN | Only with coherent calibrated CIR | Fingerprint map and dev k/radius/temperature. |
| Beta survival | Algorithm yes with AoA | VT/visibility map and survival-prior development selection. |
| Residual graph | No | Feature frontend, analytic anchor, supervised network. |
| RRLE | No pretrained transfer | Experts, OOF predictions, router. |
| Survival-CIR+ToA trees | Only with identical eight-anchor features | Full supervised tree refit. |
| Causal HMM | Usually yes | RP transition matrix; optionally earlier-trajectory parameter selection. |
| Symmetric Chamfer | Yes for any delay map | Nothing beyond RP map; re-freeze interpolation if changed. |
| Differentiable assignment | Yes as per-instance optimizer | LoS/Tx separation policy, candidate count/bounds/noise loss. |
| Genuine cross-attention | No pretrained transfer | Survey augmentation training and whole-RP refreeze. |
| Analytic PointNet reranker | No pretrained transfer | Survey training/refreeze; analytic MPUrge frontend remains reusable. |
| G3 BA | No deployable version exists | Replace oracle association/truth-derived odometry/height/anchor, then rerun. |
