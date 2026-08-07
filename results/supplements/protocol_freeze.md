# Four-paper fair benchmark protocol freeze

Revision 2 frozen: 2026-08-07, before the corrected full evaluation rows were opened. Revision 2 records the user's explicit correction that fairness constrains the number and locations of acquisitions, **not** the feature modality.

## Claim boundary

The original simulator projects, channel files, and random seeds used in the four papers are not present.  Every numerical table produced here is therefore an **executable replacement-scene reconstruction** of the printed protocol, not an exact reproduction of a published number.  Published results are context columns and are never pooled with reconstruction results.

The four reconstructed tasks are:

1. MPUrge-MAP: static query localization from a surveyed multipath reference map.
2. MPUrge: construction of an unordered VT coordinate set from known survey positions and elementary per-transmitter multipath fingerprints.
3. SMART-LEA: static multi-transmitter localization with a calibration-derived VT map and iterative multipath/VT matching.
4. CNN: supervised two-room localization from the nine paths selected by observed received power, with the printed CNN heads and reconstructed MCA controls.

## Information and calibration contract

- **There is no delay-only eligibility rule.** A method may consume any physically measurable modality that the replacement simulator can generate at the same calibration and query acquisitions: delay/range, noisy received power or complex gain, AoA/AoD, a noisy complex array CIR, and a transmitter channel identifier when transmitters are separable by their waveform.  The leaderboard records each method's exact feature bundle instead of penalizing a method for using a richer sensor.
- The simulator's latent ray/source identity, reflection order, exact VT coordinate, obstacle-configuration label, and query coordinate are not measurements.  They may generate an observation or score a prediction, but they may never be passed to an estimator.
- Every exposed physical modality is corrupted.  A method cannot receive noisy delay together with deterministic simulator power, exact direction, or a clean CIR.  Path detection, dropout, false alarms, power ranking, delay, and angle are coupled through one materialized noisy acquisition wherever the replacement receiver supports this.
- The simulator-labelled data allowance is **paper-specific**.  Every challenger receives at most the locations, counts, and labels granted to Majid's method in that paper's protocol.  Thus the CNN protocol permits its printed 8,400 labelled simulated fingerprints per room, while a sparse-survey protocol does not permit a challenger to add a dense map or thousands of new labelled positions.  Within the allowed corpus, all physical modalities may be generated at those same acquisitions.  Calibration augmentation beyond it may only corrupt/drop/permute stored observations; it may not introduce new labelled coordinates, VT truth, or clean targets.
- For a given condition, reference-map and query acquisitions receive independent noise draws.  All methods consume the identical materialized observations.
- At sparse SMART spacing, a method may use and train on only that spacing's calibration RPs.  Dense-map training followed by sparse-map evaluation is forbidden.
- Hyperparameters and early stopping are chosen using whole calibration-location, spatial-block, AP, or scene groups as appropriate.  Locked off-grid queries do not choose a model, threshold, interpolation rule, empty-set policy, seed, or ablation.
- Primary learned results average three independently initialized models where computationally feasible.  Any one-seed exception is stated beside the result.

## Observation corruption

The papers that specify range AWGN retain that printed delay-error axis,

\[
  \widetilde r_i=r_i+\epsilon_i,\qquad \epsilon_i\sim\mathcal N(0,\sigma_r^2),
\]

followed by the disclosed censoring rule for non-positive/non-finite paths.  Reference and query noise is independent.  A feature-rich challenger may additionally consume the noisy signal observables from the same acquisition.  Because those papers do not print a power/angle receiver model, the reconstruction uses disclosed sensor settings and reports a sensitivity bracket; it does not silently leave either channel exact.

The CNN paper specifies signal SNR rather than range AWGN.  The replacement receiver therefore perturbs each clean complex path/array response before detection and strongest-path selection.  With clean amplitude
\(a_i=10^{P_i/20}e^{j\phi_i}\), it uses

\[
  \widetilde a_i=a_i+n_i,\qquad
  n_i\sim\mathcal{CN}(0,\sigma_n^2),\qquad
  \sigma_n^2=\frac{\sum_i |a_i|^2}{N\,10^{\mathrm{SNR}/10}},
\]

and ranks paths by \(10\log_{10}|\widetilde a_i|^2\).  A disclosed bandwidth/SNR delay-jitter surrogate is then applied to retained delays, and direction is estimated from the noisy array response (or a disclosed wrapped directional-noise surrogate when the array extractor is unavailable).  The noisy complex CIR itself remains available to methods that use it.  This is a physically coupled and leakage-safe replacement model, but it is not claimed to reproduce NIST Q-D's unpublished MPC extractor.

The same rule applies to the other replacement scenes: latent paths first create a complex impulse/array response, receiver noise is added, and all exported delay, power, angle, and CIR features are derived from that one acquisition.  If a simplified extracted-feature corruption is required, it is labelled and uses jointly materialized range, log-power, wrapped-angle, dropout, and false-alarm noise.  Results include a clean-feature ablation only as a leakage diagnostic, never as the noisy headline.

## Shared decisions and edge cases

- Empty query delay set: return the centroid of the available calibration coordinates for every static localizer unless an ablation explicitly tests another frozen rule.
- Empty reference fingerprint: its ordinary set distance is infinite.  The path-count empty rule is isolated as its own MPUrge-MAP ablation.
- Candidate/path order is randomly permuted in invariance tests; padding length and batch composition must not change predictions beyond numerical tolerance for a claimed set model.  AoA is encoded by unit vectors or sine/cosine pairs, never by an unwrapped scalar angle.
- MPUrge total window \(P=5\) means half-width 2 in the legacy API.  Original-VT construction retains transmitter identity throughout grouping and support computation, orders conflicts by composite dissimilarity, and sweeps beta until no candidate remains.
- MPUrge-MAP reports only the prose-consistent coverage penalty in the main table.  The literal printed multiplication is an equation-diagnostic appendix row, not another "Majid method."
- SMART's paper-described joint 20-reflector matching is the primary implementation.  The previous four independent truth-channel solves are a clearly named reconstruction ablation.
- Known VT cardinality and exact VT associations are not available to primary VT-construction methods.  Any such result is labelled oracle and excluded from the fair leaderboard.
- A causal HMM requires an ordered sequence and is evaluated only on a disclosed trajectory protocol, not on independently shuffled queries.  Raw-CIR, coherent-CSI, power, and AoA methods must be adapted to the noisy observables generated at the existing sparse acquisitions; they are not rejected merely because Majid's original estimator ignored those channels.

## Metrics and uncertainty

Static position protocols report mean Euclidean error, RMSE, median, and 90th percentile.  Pairwise changes are computed on identical physical queries and use a deterministic clustered bootstrap over the physical query (or AP/scene group), not duplicated augmentation rows.

VT construction reports gated maximum-cardinality one-to-one precision, recall, and F1, followed by matched-coordinate RMSE and symmetric set distance.  Confidence intervals resample independent scene/noise realizations.  The descriptive beta curve is separate from a development-frozen operating point.

Every artifact must retain raw predictions or VT sets, truth used only for scoring, condition/group keys, seed, runtime, configuration, and source hash.  An aggregate row without reconstructable raw evidence does not enter the report.

## Applicability rule

A requested method appears numerically whenever its output can be adapted to the paper task and its inputs can be produced as noisy physical observations at the same sparse acquisitions.  Richer feature shape is explicitly allowed.  `Not applicable` is reserved for a genuine output/task mismatch that cannot be adapted (for example, a causal decoder without an ordered sequence), never merely for using power, AoA, or CIR.  The feature column and same-feature ablations separate architectural gain from sensor gain.  This rule prevents substituting clean simulator truth or extra simulator coverage for a sensor measurement; it does not force every method to use the same sensor.
