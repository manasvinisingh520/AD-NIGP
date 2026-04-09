"""Phase 2: autosearch expression then embedding per region with unique study names.

Writes best_params to ranking_model_runs/autosearch_v2/{study_name}_best_params.json
where study_name matches scripts/run_phase3.py --prefix (e.g. v2b_EC_Astrocytes_expression).

Usage:
    python scripts/run_phase2.py --cell_type Astrocytes --cell_type Microglia [--prefix v2b] [--n_trials 50] [--dry-run]
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PYTHON = r"C:\Users\Manish\.conda\envs\sf2026\python.exe"
REPO = Path(__file__).resolve().parent.parent

REGION_MAP = {
    "Astrocytes": ["EC", "ITG", "PFC", "V2", "V1"],
    "Microglia": ["CrossRegion", "ITG", "PFC"],
    "MTG": ["MTG"],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell_type", action="append", required=True, choices=list(REGION_MAP.keys()))
    parser.add_argument("--prefix", type=str, default="v2b")
    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument(
        "--new_study",
        action="store_true",
        help="Pass through to autosearch: delete this study name in the DB then run (trial numbers start low).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--log",
        type=Path,
        default=None,
        help="Append stdout/stderr to this file (default: runs/phase2_{prefix}_UTC.log)",
    )
    args = parser.parse_args()

    log_path = args.log
    if log_path is None:
        (REPO / "runs").mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_path = REPO / "runs" / f"phase2_{args.prefix}_{ts}.log"

    def log(msg: str) -> None:
        line = msg.rstrip() + "\n"
        sys.stdout.write(line)
        sys.stdout.flush()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line)

    log(f"Phase 2 log: {log_path}")
    log(f"prefix={args.prefix} n_trials={args.n_trials}")

    total = 0
    for ct in args.cell_type:
        for region in REGION_MAP[ct]:
            for input_type in ("expression", "embedding"):
                total += 1
    n = 0

    for ct in args.cell_type:
        for region in REGION_MAP[ct]:
            for input_type in ("expression", "embedding"):
                n += 1
                study = f"{args.prefix}_{region}_{ct}_{input_type}"
                cmd = [
                    PYTHON,
                    "src/autosearch.py",
                    "--region",
                    region,
                    "--cell_type",
                    ct,
                    "--input_type",
                    input_type,
                    "--sampler",
                    "optuna",
                    "--n_trials",
                    str(args.n_trials),
                    "--study_name",
                    study,
                ]
                if args.new_study:
                    cmd.append("--new_study")
                label = f"[{n}/{total}] {region} {ct} {input_type} study={study}"
                log(f"\n{'=' * 60}\n{label}\n{'=' * 60}")
                log(" " + " ".join(cmd))
                if args.dry_run:
                    continue
                r = subprocess.run(cmd, cwd=str(REPO))
                if r.returncode != 0:
                    log(f"WARNING: exit code {r.returncode} for {label}")

    if args.dry_run:
        log("\nDry run -- no autosearch executed.")
    else:
        log("\nPhase 2 batch finished (check warnings above).")


if __name__ == "__main__":
    main()
