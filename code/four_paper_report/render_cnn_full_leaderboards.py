"""Render the independently audited corrected CNN result as TeX and Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DISPLAY = {
    "majid_cnn_classifier": "Majid CNN classifier",
    "majid_cnn_regressor": "Majid CNN regressor",
    "majid_mca_eps1": r"Majid MCA ($\epsilon=1$)",
    "majid_mca_eps2": r"Majid MCA ($\epsilon=2$)",
    "corrected_mpurge_map": "Corrected MPUrge--MAP",
    "symmetric_chamfer_inverse3": "Chamfer + inverse top-3",
    "symmetric_chamfer_softmax3": "Chamfer + softmax top-3",
    "subset_consensus": "Subset consensus",
    "graph_diffusion": "Graph diffusion",
    "evomdp_frozen_transfer": "EvoMDP frozen transfer",
    "range_power_chamfer": "Range--power Chamfer",
    "range_aoa_chamfer": "Range--AoA Chamfer",
    "dichasus_coherent_adp_8nn": "DICHASUS coherent ADP 8-NN",
    "candidate_pointnet_reranker": "Candidate PointNet reranker",
    "candidate_set_attention": "Candidate self-attention",
    "genuine_candidate_cross_attention": "Genuine candidate cross-attention",
    "analytic_anchor_residual": "Analytic-anchor residual",
    "rrle_map_aware_moe": "RRLE map-aware MoE",
    "pointnet_delay": "PointNet (delay)",
    "pointnet_multimodal": "PointNet (multimodal)",
    "set_attention_multimodal": "Set attention (multimodal)",
    "caez_cir_probability_mlp": "CAEZ CIR probability MLP",
    "survival_cir_toa_extratrees": "Survival-CIR + ToA ExtraTrees",
    "assigned_vt_inverse_consensus": "Assigned-VT inverse consensus",
    "beta_marginal_vt_survival": "Beta-marginal VT survival",
}

PROTOCOL_LABEL = {
    "paper_condition_map": "paper-condition (privileged, paper-matched) map",
    "environment_blind_map": "environment-blind deployable map",
    "protocol_independent": "protocol-independent direct/pooled-VT",
}


def tex_number(value: float) -> str:
    return f"{float(value):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tex-output", type=Path, required=True)
    parser.add_argument("--notes-output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.output_dir / "results.json").read_text(encoding="utf-8"))
    audit = json.loads((args.output_dir / "independent_integrity_audit.json").read_text(encoding="utf-8"))
    if result["status"] != "FULL_COMPLETE" or audit["status"] != "PASS":
        raise RuntimeError("refusing to render an incomplete or unaudited result")

    summaries = {(s["protocol"], s["room"], s["method"]): s for s in result["summaries"]}
    intervals = {(i["protocol"], i["room"], i["method"]): i for i in result["paired_intervals"]}
    tex = [
        "% Generated from the independently audited corrected CNN full run.",
        r"\section{Corrected full two-room CNN result}",
        r"\label{sec:cnn-corrected-full}",
        (
            r"This is a replacement-scene reconstruction, not a numerical reproduction of the "
            r"paper's unavailable NIST Q--D realization. It retains the printed two-room topology: "
            r"20 reference positions, 60 AP positions, seven obstruction conditions, 8,400 stored "
            r"calibration fingerprints per room, and 1,600 locked off-grid queries per room. Every "
            r"modality is extracted from the same noisy array-CIR acquisition; no extra calibration "
            r"position, dense forward simulation, latent ray/VT identity, or query truth is available."
        ),
        r"\paragraph{How to read the brackets.} The paper-condition bracket supplies a reference method with the true matching obstruction-condition map and is therefore privileged but closest to the paper prose. The environment-blind bracket pools all seven stored condition maps and chooses through observed fingerprint similarity only. Protocol-independent rows either use no reference map or a VT map built from the allowed pooled survey. They are emitted once rather than copied under multiple labels.",
        r"\paragraph{Uncertainty.} For Room A, error reductions are paired against the reconstructed Majid CNN classifier; for Room B they are paired against the reconstructed Majid CNN regressor. The 95\% intervals resample the 1,600 physical off-grid queries (2,000 bootstrap replicates), preserving the pairing across methods and protocol brackets. A positive reduction means lower localization error than that room's baseline.",
    ]
    notes = [
        "# Corrected CNN full result — report notes",
        "",
        "Status: FULL_COMPLETE and independent audit PASS.",
        "",
        "Claim boundary: replacement-scene evidence with the paper's printed counts; not an exact reproduction of the unavailable NIST Q-D realization.",
        "",
    ]

    for room in ("room_a", "room_b"):
        room_title = "Room A" if room == "room_a" else "Room B"
        tex.append(rf"\subsection{{{room_title}}}")
        notes.append(f"## {room_title}")
        notes.append("")
        for protocol in ("paper_condition_map", "environment_blind_map", "protocol_independent"):
            records = sorted(
                (s for (p, r, _), s in summaries.items() if p == protocol and r == room),
                key=lambda item: item["mean_error_m"],
            )
            tex.extend(
                [
                    rf"\subsubsection{{{PROTOCOL_LABEL[protocol].capitalize()}}}",
                    r"\begingroup\scriptsize\setlength{\tabcolsep}{3.2pt}",
                    r"\begin{longtable}{@{}r p{.28\linewidth} r r r r p{.23\linewidth}@{}}",
                    r"\toprule Rank & Method & Mean & RMSE & Median & P90 & Reduction vs. room baseline [95\% CI] \\",
                    r"\midrule\endfirsthead",
                    r"\toprule Rank & Method & Mean & RMSE & Median & P90 & Reduction vs. room baseline [95\% CI] \\",
                    r"\midrule\endhead",
                ]
            )
            notes.append(f"### {PROTOCOL_LABEL[protocol]}")
            notes.append("")
            notes.append("| Rank | Method | Mean m | RMSE m | Median m | P90 m | Reduction [95% CI] m |")
            notes.append("|---:|---|---:|---:|---:|---:|---:|")
            for rank, record in enumerate(records, 1):
                method = record["method"]
                interval = intervals.get((protocol, room, method))
                if interval:
                    reduction = (
                        f"{tex_number(interval['mean_error_reduction_m'])} "
                        f"[{tex_number(interval['ci95_low_m'])}, {tex_number(interval['ci95_high_m'])}]"
                    )
                else:
                    reduction = "baseline" if (
                        (room == "room_a" and method == "majid_cnn_classifier")
                        or (room == "room_b" and method == "majid_cnn_regressor")
                    ) else "---"
                label = DISPLAY.get(method, method.replace("_", r"\_"))
                tex.append(
                    f"{rank} & {label} & {tex_number(record['mean_error_m'])} & "
                    f"{tex_number(record['rmse_m'])} & {tex_number(record['median_error_m'])} & "
                    f"{tex_number(record['p90_error_m'])} & {reduction} \\\\" 
                )
                notes.append(
                    f"| {rank} | {DISPLAY.get(method, method)} | {tex_number(record['mean_error_m'])} | "
                    f"{tex_number(record['rmse_m'])} | {tex_number(record['median_error_m'])} | "
                    f"{tex_number(record['p90_error_m'])} | {reduction} |"
                )
            tex.extend([r"\bottomrule", r"\end{longtable}\endgroup"])
            notes.append("")

    confirmed = sorted(
        (interval for interval in result["paired_intervals"] if interval["ci95_low_m"] > 0.0),
        key=lambda item: item["mean_error_reduction_m"],
        reverse=True,
    )
    tex.extend(
        [
            r"\subsection{Audit and interpretation}",
            rf"The serialized evidence contains exactly {audit['observed_rows']:,} finite raw prediction rows, 25 unique numerical methods, and 72 room--protocol--method summaries. The independent audit recomputed every error and aggregate, verified the 11/16/9 protocol method topology, verified 1,600 queries for every room--protocol--method cell, and found no complete prediction vector copied across protocol labels. The frozen runner SHA--256 is \texttt{{{audit['frozen_runner_sha256']}}}; the frozen receiver SHA--256 is \texttt{{{audit['frozen_receiver_sha256']}}}.",
            r"The tables compare algorithms within the reconstructed receiver. Published paper numbers must remain context columns rather than being pooled with these rows: matching the paper's counts does not recreate its unavailable geometry, ray realization, or exact NIST Q--D outputs.",
        ]
    )
    notes.extend(["## Statistically confirmed positive reductions", ""])
    for item in confirmed:
        notes.append(
            f"- {item['room']} / {item['protocol']} / {DISPLAY.get(item['method'], item['method'])}: "
            f"{tex_number(item['mean_error_reduction_m'])} m "
            f"[{tex_number(item['ci95_low_m'])}, {tex_number(item['ci95_high_m'])}]."
        )
    notes.extend(
        [
            "",
            "## Integrity",
            "",
            f"- Rows: {audit['observed_rows']:,} / expected {audit['expected_rows']:,}.",
            f"- Unique methods: {audit['unique_methods']}.",
            f"- Frozen runner SHA-256: `{audit['frozen_runner_sha256']}`.",
            f"- Frozen receiver SHA-256: `{audit['frozen_receiver_sha256']}`.",
            "- All independent audit checks passed, including protocol separation and physical-query clustered intervals.",
        ]
    )
    args.tex_output.write_text("\n".join(tex) + "\n", encoding="utf-8")
    args.notes_output.write_text("\n".join(notes) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
