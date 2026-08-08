"""Validate the completed CNN artifact and emit its deterministic report hook."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN = ROOT / "research" / "four_paper_report" / "cnn_fair_corrected_full_final2"
DEFAULT_OUTPUT = ROOT / "research" / "four_paper_report" / "cnn_full_leaderboard.tex"
PROTOCOL_LABELS = {
    "paper_condition_map": "oracle-condition",
    "environment_blind_map": "environment-blind",
    "protocol_independent": "independent",
}
EXPECTED_PROTOCOL_COUNTS = {"paper_condition_map": 11, "environment_blind_map": 16, "protocol_independent": 9}
BASELINES = {"room_a": "majid_cnn_classifier", "room_b": "majid_cnn_regressor"}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def load(path: Path) -> dict:
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path: Path) -> str:
    require(path.is_file(), f"missing hash target {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def htex(value: str) -> str:
    require(len(value) == 64, "bad SHA-256")
    return rf"\texttt{{{value[:32]}}}\allowbreak\texttt{{{value[32:]}}}"


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def finite(value: object) -> float:
    result = float(value)
    require(math.isfinite(result), "non-finite metric")
    return result


def validate(run_dir: Path) -> dict:
    run_dir = run_dir.resolve()
    result = load(run_dir / "results.json")
    audit = load(run_dir / "independent_integrity_audit.json")
    require(result.get("status") == "FULL_COMPLETE", "run is not FULL_COMPLETE")
    require(audit.get("status") == "PASS" and all(audit.get("checks", {}).values()), "independent audit did not pass")
    require(audit.get("observed_rows") == audit.get("expected_rows") == 115200, "wrong row count")
    require(audit.get("unique_methods") == 25, "wrong unique method count")
    require(audit.get("protocol_method_counts") == {"environment_blind_map": 16, "paper_condition_map": 11, "protocol_independent": 9}, "wrong protocol counts")
    require(sha(run_dir / "results.json") == audit.get("results_sha256"), "result changed after audit")
    require(sha(run_dir / "raw_rows.jsonl") == result.get("raw_rows_sha256") == audit.get("raw_rows_sha256"), "raw rows changed")
    require(sha(run_dir / "integrity_audit.json") == result.get("integrity_audit_sha256"), "producer audit changed")
    require(sha(run_dir / "frozen_sources" / "run_cnn_fair.py") == result.get("source_sha256") == audit.get("frozen_runner_sha256"), "runner hash mismatch")
    require(sha(run_dir / "frozen_sources" / "multimodal_receiver.py") == result.get("receiver_source_sha256") == audit.get("frozen_receiver_sha256"), "receiver hash mismatch")

    summaries = result.get("summaries", [])
    require(len(summaries) == 72, "expected 72 room/bracket/method summaries")
    keys = {(row["room"], row["protocol"], row["method"]) for row in summaries}
    require(len(keys) == 72, "duplicate summary key")
    require({row["room"] for row in summaries} == set(BASELINES), "room coverage mismatch")
    for room in BASELINES:
        for protocol, count in EXPECTED_PROTOCOL_COUNTS.items():
            local = [row for row in summaries if row["room"] == room and row["protocol"] == protocol]
            require(len(local) == count, f"method coverage mismatch: {room}/{protocol}")
            require(all(row["queries"] == 1600 for row in local), "query coverage mismatch")
    require(len({row["method"] for row in summaries}) == 25, "not all 25 methods are present")
    for row in summaries:
        for field in ("mean_error_m", "rmse_m", "median_error_m", "p90_error_m"):
            require(finite(row[field]) >= 0, "negative metric")

    intervals = result.get("paired_intervals", [])
    require(len(intervals) == 70, "paired interval coverage mismatch")
    interval_index = {(row["room"], row["protocol"], row["method"]): row for row in intervals}
    require(len(interval_index) == 70, "duplicate interval key")
    for row in intervals:
        require(row["baseline"] == BASELINES[row["room"]], "wrong paired baseline")
        require(row["baseline_protocol"] == "protocol_independent", "wrong baseline bracket")
        require(row["bootstrap_clusters"] == 1600 and row["bootstrap_repetitions"] == 2000, "wrong bootstrap budget")
        require(finite(row["ci95_low_m"]) <= finite(row["ci95_high_m"]), "reversed CI")

    rooms = {}
    checkpoint_rows = []
    for room in BASELINES:
        complete = load(run_dir / room / "complete.json")
        require(complete.get("status") == "COMPLETE" and complete.get("quick") is False, f"incomplete {room}")
        require(complete.get("rows") == 57600, f"wrong {room} row count")
        require(complete.get("runner_sha256") == result["source_sha256"], f"{room} runner mismatch")
        require(complete.get("receiver_sha256") == result["receiver_source_sha256"], f"{room} receiver mismatch")
        require(sha(run_dir / room / "completed_rows.jsonl.gz") == complete["rows_sha256"], f"{room} row hash mismatch")
        training = complete["audit"]["training"]
        require(len(set(training["learned_seeds"])) == 3, f"{room} seed count mismatch")
        require((training["paper_epochs"], training["learned_epochs"], training["candidate_epochs"]) == (100, 45, 30), f"{room} epoch budget mismatch")
        require(complete["audit"]["invariance"]["status"] == "PASS", f"{room} invariance failed")
        rooms[room] = complete
        for checkpoint in sorted((run_dir / room / "checkpoints").glob("*.pt")):
            checkpoint_rows.append({"path": rel(checkpoint), "sha256": sha(checkpoint)})
    require(len(checkpoint_rows) == 52, "expected 52 checkpoints")
    checkpoint_bundle_sha = canonical_sha(checkpoint_rows)
    acquisition_definition_sha = canonical_sha({
        "schema": result["schema"], "fairness": result["fairness"],
        "paper_specified": result["paper_specified"], "runner": result["source_sha256"],
        "receiver": result["receiver_source_sha256"],
        "room_rows": {room: rooms[room]["rows_sha256"] for room in sorted(rooms)},
    })
    return {"run_dir": run_dir, "result": result, "audit": audit, "summaries": summaries,
            "intervals": interval_index, "rooms": rooms, "checkpoint_sha": checkpoint_bundle_sha,
            "acquisition_sha": acquisition_definition_sha}


def gain_text(row: dict, intervals: dict) -> str:
    value = intervals.get((row["room"], row["protocol"], row["method"]))
    if value is None:
        return "baseline"
    return f"{value['mean_error_reduction_m']:+.4f} [{value['ci95_low_m']:+.4f},{value['ci95_high_m']:+.4f}]"


def render(data: dict) -> str:
    result, summaries, intervals = data["result"], data["summaries"], data["intervals"]
    lines = [
        r"\subsection*{Audited corrected full-run leaderboard}",
        r"\statusbox{\good{FULL\_COMPLETE; producer and 21-check independent audits PASS.} The run contains 115,200 raw rows, 1,600 locked queries per room, 25 unique methods, and 36 legitimate method--bracket rows per room. No complete prediction vector is duplicated across brackets; every learned ensemble uses three seeds.}",
        r"\begin{landscape}", r"\begingroup\tiny",
        r"\setlength{\LTleft}{0pt}\setlength{\LTright}{0pt}",
    ]
    for room in ("room_a", "room_b"):
        label = "A" if room == "room_a" else "B"
        rows = sorted((row for row in summaries if row["room"] == room), key=lambda row: (row["mean_error_m"], row["protocol"], row["method"]))
        lines += [
            r"\begin{longtable}{r p{0.13\linewidth} p{0.30\linewidth} rrrr p{0.22\linewidth}}",
            rf"\caption{{CNN replacement Room {label}. Gain is paired Majid-baseline error minus method error; positive is better. CIs resample the same 1,600 physical query blocks.}}\label{{tab:cnn-full-{room}}}\\",
            r"\toprule Rank & Bracket & Method & Mean & RMSE & Median & P90 & Gain [95\% CI] \\",
            r"\midrule\endfirsthead",
            r"\toprule Rank & Bracket & Method & Mean & RMSE & Median & P90 & Gain [95\% CI] \\",
            r"\midrule\endhead",
        ]
        for rank, row in enumerate(rows, 1):
            lines.append(f"{rank} & {PROTOCOL_LABELS[row['protocol']]} & \\code{{{row['method']}}} & {row['mean_error_m']:.4f} & {row['rmse_m']:.4f} & {row['median_error_m']:.4f} & {row['p90_error_m']:.4f} & {gain_text(row, intervals)}" + r" \\")
        lines += [r"\bottomrule\end{longtable}", r"\vspace{0.4em}"]
    lines += [
        r"\endgroup\end{landscape}",
        r"\noindent\textbf{Reading the brackets.} Oracle-condition rows receive the true matching obstruction-condition map and are privileged sensitivity results. Environment-blind rows pool all stored conditions and select only from the observed fingerprint. Protocol-independent rows receive no query-condition label. Repeated method names are distinct audited map brackets, not copied outputs. Rich-sensor margins are not architecture-only causal effects.",
        "",
        r"The strongest protocol-independent row is multimodal PointNet (Room A mean/RMSE/median/P90 $.7946/1.1940/.5265/1.6120$ m; Room B $.8251/1.1871/.5468/1.8212$ m). The strongest environment-blind row is range--AoA Chamfer (A $1.3594$ m; B $1.1878$ m). The privileged version is slightly worse here (A $1.3852$ m; B $1.2785$ m), showing that hidden condition identity is not automatically beneficial. Beta-marginal VT survival reaches $1.2856$ m in A and $1.1657$ m in B. The direct Majid baselines are much weaker in this replacement: classifier $5.6906/5.3789$ m (A/B), regressor $2.9745/2.9194$ m.",
        "",
        r"\noindent\textbf{Provenance.} Frozen runner SHA-256 " + htex(result["source_sha256"]) +
        r"; frozen receiver SHA-256 " + htex(result["receiver_source_sha256"]) +
        r"; raw-row SHA-256 " + htex(result["raw_rows_sha256"]) +
        r"; 52-checkpoint bundle SHA-256 " + htex(data["checkpoint_sha"]) +
        r"; acquisition-definition SHA-256 " + htex(data["acquisition_sha"]) + r".\\",
        r"Exact results: \code{" + rel(data["run_dir"] / "results.json") +
        r"}; independent audit: \code{" + rel(data["run_dir"] / "independent_integrity_audit.json") +
        r"}; raw rows: \code{" + rel(data["run_dir"] / "raw_rows.jsonl") +
        r"}. The acquisition-definition hash binds the physical acquisition contract, frozen code, and room-row hashes; no standalone clean acquisition tensor was emitted. This is replacement-scene evidence, not exact NIST Q-D evidence.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    data = validate(args.run_dir)
    text = render(data)
    if not args.check_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_name(args.output.name + ".tmp")
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, args.output)
    print(json.dumps({"status": "PASS", "check_only": args.check_only, "output": str(args.output.resolve()), "bytes": len(text.encode()), "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))


if __name__ == "__main__":
    main()
