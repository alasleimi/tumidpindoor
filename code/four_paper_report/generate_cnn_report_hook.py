"""Generate the audited CNN LaTeX result hook from the frozen JSON artifact.

This is deliberately a presentation-only transform: it does not recompute or
alter benchmark outputs.  The independent auditor is the authority for row and
hash integrity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PROTOCOLS = (
    (
        "protocol_independent",
        "Protocol-independent methods",
        "These rows use no obstacle-condition label. Direct models use no reference map; "
        "the two VT rows use only pooled survey-built VT modes.",
    ),
    (
        "environment_blind_map",
        "Deployable environment-blind reference map",
        "All seven stored conditions are available, but the method must select or combine "
        "them using observed fingerprints only.",
    ),
    (
        "paper_condition_map",
        "Privileged paper-condition reference map",
        "This sensitivity bracket is not deployable: reference methods receive the true "
        "obstacle-condition map. It is never mixed into the primary ranking.",
    ),
)


def fmt(value: float) -> str:
    return f"{value:.4f}".lstrip("0")


def method_code(value: str) -> str:
    return rf"\code{{{value}}}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("audit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    result = json.loads(args.results.read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    if result.get("status") != "FULL_COMPLETE" or audit.get("status") != "PASS":
        raise SystemExit("Refusing to generate a full-result hook from unaudited input")

    summaries = {
        (row["protocol"], row["room"], row["method"]): row
        for row in result["summaries"]
    }
    intervals = {
        (row["protocol"], row["room"], row["method"]): row
        for row in result["paired_intervals"]
    }

    out: list[str] = []
    out += [
        r"\subsection*{Audited corrected CNN full run}",
        "",
        rf"\statusbox{{\good{{FULL\_COMPLETE; producer and independent audits PASS.}} "
        rf"The frozen CUDA run contains {audit['observed_rows']:,} prediction rows, "
        rf"{audit['unique_methods']} unique method identities, two rooms, and 1,600 locked "
        rf"off-grid physical queries per room. All {len(audit['checks'])} independent checks "
        rf"pass, including exact error and summary reconstruction, source/hash checks, "
        rf"protocol topology, finite outputs, and absence of complete prediction-vector "
        rf"duplicates across information brackets.}}",
        "",
        "Every value below is from the replacement scene, in metres.  A row is sorted by "
        "the average of its Room A and Room B mean errors.  Mean, RMSE, median, and P90 "
        "describe the same 1,600 protected queries; none is a paper-reported value.",
        "",
    ]

    for protocol, title, explanation in PROTOCOLS:
        methods = sorted(
            {m for (p, _r, m) in summaries if p == protocol},
            key=lambda m: (
                summaries[(protocol, "room_a", m)]["mean_error_m"]
                + summaries[(protocol, "room_b", m)]["mean_error_m"]
            )
            / 2,
        )
        out += [
            r"\begin{landscape}",
            rf"\subsubsection*{{{title}}}",
            explanation,
            r"\begingroup\scriptsize",
            r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
            r"\begin{longtable}{r p{0.30\linewidth} rrrr rrrr}",
            rf"\caption{{CNN replacement-scene results: {title.lower()}.}}\\",
            r"\toprule",
            r"& & \multicolumn{4}{c}{Room A} & \multicolumn{4}{c}{Room B}\\",
            r"Rank & Method & Mean & RMSE & Med. & P90 & Mean & RMSE & Med. & P90\\",
            r"\midrule\endfirsthead",
            r"\toprule",
            r"Rank & Method & A mean & A RMSE & A med. & A P90 & B mean & B RMSE & B med. & B P90\\",
            r"\midrule\endhead",
        ]
        for rank, method in enumerate(methods, 1):
            a = summaries[(protocol, "room_a", method)]
            b = summaries[(protocol, "room_b", method)]
            vals = [
                a["mean_error_m"],
                a["rmse_m"],
                a["median_error_m"],
                a["p90_error_m"],
                b["mean_error_m"],
                b["rmse_m"],
                b["median_error_m"],
                b["p90_error_m"],
            ]
            out.append(
                f"{rank} & {method_code(method)} & "
                + " & ".join(fmt(v) for v in vals)
                + r"\\"
            )
        out += [
            r"\bottomrule",
            r"\end{longtable}",
            r"\endgroup",
            r"\end{landscape}",
            "",
        ]

    selected = (
        ("protocol_independent", "pointnet_multimodal"),
        ("protocol_independent", "set_attention_multimodal"),
        ("protocol_independent", "beta_marginal_vt_survival"),
        ("environment_blind_map", "range_aoa_chamfer"),
    )
    out += [
        r"\begin{table}[ht]",
        r"\centering\scriptsize",
        r"\begin{tabularx}{\linewidth}{@{}p{0.21\linewidth}p{0.27\linewidth}lrrr@{}}",
        r"\toprule Bracket & Method & Room/baseline & Reduction & \multicolumn{2}{c}{95\% CI}\\\midrule",
    ]
    for protocol, method in selected:
        for room in ("room_a", "room_b"):
            row = intervals[(protocol, room, method)]
            bracket = {
                "protocol_independent": "independent",
                "environment_blind_map": "environment-blind",
            }[protocol]
            out.append(
                f"{bracket} & {method_code(method)} & {room.replace('_', ' ').title()} / "
                f"{method_code(row['baseline'])} & {fmt(row['mean_error_reduction_m'])} & "
                f"{fmt(row['ci95_low_m'])} & {fmt(row['ci95_high_m'])}\\\\"
            )
    out += [
        r"\bottomrule\end{tabularx}",
        r"\caption{Paired mean-error reductions against the stronger reconstructed Majid CNN row in each room. Bootstrap clusters are the 1,600 physical queries (2,000 repetitions); every displayed interval excludes zero.}",
        r"\end{table}",
        "",
        "The strongest overall row is multimodal PointNet: "
        f"{fmt(summaries[('protocol_independent','room_a','pointnet_multimodal')]['mean_error_m'])} m "
        "in Room A and "
        f"{fmt(summaries[('protocol_independent','room_b','pointnet_multimodal')]['mean_error_m'])} m "
        "in Room B.  It is genuinely permutation invariant: a shared path-token MLP followed "
        "by masked mean/max/count pooling sees jointly noisy delay, observed power, and receive "
        "AoA, plus the observable AP coordinate.  Multimodal set attention is second in Room A; "
        "beta-marginal VT survival is second in Room B.  In the deployable reference-map bracket, "
        "range--AoA Chamfer wins both rooms.  The privileged bracket is slightly worse here, a "
        "useful warning that being granted the true condition does not guarantee a better metric "
        "when reference noise and map selection interact.",
        "",
        "The original paper's reported RMSE values remain context only: classifier/regressor "
        "1.82/1.84 m in Room A and 2.14/1.47 m in Room B.  They cannot be directly compared with "
        "replacement-scene values because the original NIST Q-D scene and simulator project are "
        "not available.  The reconstructed Majid classifier/regressor rows are included above so "
        "within-scene component comparisons remain fair.",
        "",
        f"Run provenance: runtime {result['runtime_s']:.1f} s on {method_code(result['device'])}; "
        f"raw-row SHA-256 {method_code(result['raw_rows_sha256'])}; frozen runner "
        f"{method_code(result['source_sha256'])}; frozen receiver "
        f"{method_code(result['receiver_source_sha256'])}; result artifact "
        f"{method_code(audit['results_sha256'])}.  Learned set models use three frozen seeds per room. "
        rf"The map split is 6,720 fit and 1,680 validation acquisitions from the printed 8,400; "
        rf"all 1,600 query points per room stay locked until scoring.",
        "",
        "Candidate-model training uses independently measurement-corrupted, nonempty repeats of "
        "fit acquisitions.  A repeat may retain its original stored fit acquisition as a candidate "
        "so singleton AP/RP cells remain defined; it is not an extra site or propagation sample. "
        "Validation is stricter: fit-map only with explicit source exclusion. Empty locked queries "
        "remain in evaluation and use a frozen finite fallback.",
        "",
    ]

    args.output.write_text("\n".join(out), encoding="utf-8")


if __name__ == "__main__":
    main()
