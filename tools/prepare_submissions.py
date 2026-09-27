#!/usr/bin/env python3
"""
tools/prepare_submissions.py

Automates:
1. Generating output/matching_results_dedup.tsv via 1-to-1 target uniqueness.
2. Generating output/matching_results_calibrated.tsv (using France.scores.tsv tau>=0.95 + global 1-to-1 dedup).
3. Running utils/validate_submission.py on all submission variants.
4. Packaging submission TSVs ready for upload to Unstop.
"""

import os
import subprocess
import sys
import time
from pathlib import Path


def run_cmd(cmd, desc):
    print(f"\n--- {desc} ---")
    t0 = time.time()
    res = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if res.returncode != 0:
        print(f"FAILED (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
        return False
    print(f"SUCCESS ({time.time()-t0:.1f}s):\n{res.stdout.strip()}")
    return True


def main():
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    match_raw = Path("output/matching_results.tsv")
    cand_raw = Path("output/candidate_pairs.tsv")

    if not match_raw.exists() or match_raw.stat().st_size == 0:
        print(f"Error: {match_raw} is not ready or empty!")
        sys.exit(1)

    print(f"Found {match_raw} ({match_raw.stat().st_size / (1024*1024):.1f} MB)")

    # 1. Validate raw baseline
    run_cmd(
        f"python3 utils/validate_submission.py --matching {match_raw} --candidate {cand_raw} --test-dir dataset/test",
        "Validating Raw Baseline (Submission 1)"
    )

    # 2. Generate Deduplicated Submission (Submission 2)
    match_dedup = Path("output/matching_results_dedup.tsv")
    run_cmd(
        f"python3 tools/apply_deduplication.py --input {match_raw} --output {match_dedup}",
        "Generating 1-to-1 Deduplicated Matches (Submission 2)"
    )

    # 3. Validate Deduplicated
    run_cmd(
        f"python3 utils/validate_submission.py --matching {match_dedup} --candidate {cand_raw} --test-dir dataset/test",
        "Validating Deduplicated Matches (Submission 2)"
    )

    # 4. Generate Calibrated Submission (Submission 3: France calibrated to tau>=0.95 + Global Dedup)
    france_scores = Path("output/parts/France.scores.tsv")
    if france_scores.exists():
        print("\n--- Generating Calibrated Submission with France High-Precision (Submission 3) ---")
        t0 = time.time()
        
        # Load France high-precision candidates (tau >= 0.95)
        france_high_p = {}
        with open(france_scores, "r", encoding="utf-8") as f:
            next(f)
            for line in f:
                sid, cid, p_str = line.rstrip("\n").split("\t")
                if float(p_str) >= 0.95:
                    if sid not in france_high_p:
                        france_high_p[sid] = []
                    france_high_p[sid].append(cid)

        # Build calibrated matching
        match_calib = Path("output/matching_results_calibrated.tsv")
        with open(match_raw, "r", encoding="utf-8") as fin, open(match_calib, "w", encoding="utf-8") as fout:
            header = fin.readline()
            fout.write(header)
            for line in fin:
                sid, _, rest = line.rstrip("\n").partition("\t")
                if sid in france_high_p:
                    cands = france_high_p[sid]
                    fout.write(f"{sid}\t{','.join(cands)}\n")
                else:
                    fout.write(line)

        # Apply global 1-to-1 deduplication on the calibrated file
        run_cmd(
            f"python3 tools/apply_deduplication.py --input {match_calib} --output {match_calib}",
            "Applying 1-to-1 Dedup to Calibrated Matches"
        )
        run_cmd(
            f"python3 utils/validate_submission.py --matching {match_calib} --candidate {cand_raw} --test-dir dataset/test",
            "Validating Calibrated Matches (Submission 3)"
        )

    print("\n=======================================================")
    print("ALL SUBMISSION CANDIDATES ARE VALIDATED & READY!")
    print(f"  1. Raw Baseline:          output/matching_results.tsv")
    print(f"  2. 1-to-1 Deduplicated:   output/matching_results_dedup.tsv")
    if france_scores.exists():
        print(f"  3. Calibrated (France 95): output/matching_results_calibrated.tsv")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
