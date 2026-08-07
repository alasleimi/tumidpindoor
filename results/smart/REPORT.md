# Corrected fair SMART-LEA replacement-scene suite

Status: **FULL_COMPLETE**  
Total orchestration runtime: 8513.4 s  
Scene: disclosed 25x25 m four-transmitter first-order image-source reconstruction; this is not the unavailable original SMART floorplan.

Every coarse radio map is a strict subset of one stored 100x100 survey acquisition. Survey and query receiver noise/extraction are independent but identically configured. Learned models were fit separately per spacing and validated on held-out 5 m spatial blocks. No ray/VT ID, query truth, clean power/AoA, extra spatial locations, or extra labelled simulator calls were available to primary methods.

## Best mean error by condition and spacing

| Condition | Spacing | Best eligible method | Mean | RMSE | Median | P90 |
|---|---:|---|---:|---:|---:|---:|
| clean | 0.25 m | smart_lea4_fixed_height_totalP5_geometric | 0.0198 | 0.0224 | 0.0187 | 0.0341 |
| clean | 0.5 m | smart_lea4_fixed_height_totalP5_geometric | 0.0562 | 0.8452 | 0.0187 | 0.0342 |
| clean | 1.5 m | range_power_aoa_assignment | 0.3197 | 0.3715 | 0.2813 | 0.5385 |
| clean | 2.5 m | range_power_aoa_assignment | 0.8576 | 1.2751 | 0.6847 | 1.3894 |
| matched_noisy | 0.25 m | delay_cir_combined | 0.1338 | 0.1524 | 0.1240 | 0.2276 |
| matched_noisy | 0.5 m | query_reference_attention_multimodal | 0.2266 | 0.3131 | 0.2038 | 0.3850 |
| matched_noisy | 1.5 m | range_power_aoa_assignment | 0.4931 | 0.5753 | 0.4482 | 0.8863 |
| matched_noisy | 2.5 m | range_power_aoa_assignment | 1.0460 | 1.9119 | 0.7351 | 1.5894 |

## Interpretation guardrails

- `legacy_per_tx_fusion_totalP5_printed` is a separate diagnostic because it uses the physical-transmitter partition; it is never ranked as a primary result.
- `corrected_mpurge_map_prefilter24` applies the exact corrected MPUrge score and source-intended coverage penalty only after a shared exact-Chamfer top-24 prefilter. It is not a global exhaustive MPUrge search.
- Total window size P=5 is implemented as half-window `p=2`. The cited-parameter ambiguity is bracketed with half-window `p=5` (total 11), and printed versus geometric crossing checks are separate rows.
- Power and AoA enter only the named multimodal methods and are extracted from the same noisy array CIR as delay; delay-only rows remain explicit ablations, not an eligibility rule.
- Fitted-VT truth errors are computed only after fitting for audit and never enter localization or training.
