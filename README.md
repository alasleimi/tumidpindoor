# Faithful indoor-localisation benchmarks

This repository contains the audited LaTeX report and machine-readable result snapshots for reconstructed benchmarks based on Majid Abdmoulah's four indoor-localisation papers.

The fairness rule is an acquisition/information-budget rule, **not** a delay-only rule. Same-acquisition noisy power, AoA/AoD and CIR/CSI are eligible; extra spatial calibration samples, ungranted forward simulations, query truth, and latent simulator ray/VT identities are not.

## Releases

- `report/early_report_2026-08-07.pdf`: evidence snapshot separating completed results, paper-reported context, and provisional diagnostics.
- `report/full_report.pdf`: final 19-page audited release covering all four reconstructed protocols.
- `report/full_report.tex`: LaTeX source; its two generated leaderboard hooks are stored beside it.

## Final audited evidence

- `results/mpurge_map/`: corrected MPUrge--MAP aggregate tables, paired intervals, and audit/provenance records.
- `results/smart/`: SMART full-run summaries and integrity manifest.
- `results/mpurge_vt_full/`: complete 24-block original-MPUrge VT leaderboard and independent audit.
- `results/cnn_full/`: complete two-room CNN summaries plus producer and independent integrity audits.
- `results/supplements/`: causal HMM and honest multilevel/irregular-map evidence.

The full report is the authoritative guide to information brackets and claim boundaries. In particular, paper-reported values are context only: replacement-scene values are not exact reproductions of the unavailable original simulator scenes.
