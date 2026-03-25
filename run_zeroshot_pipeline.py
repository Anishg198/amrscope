#!/usr/bin/env python3
"""
Master orchestration script for the Zero-Shot AMR Prediction Pipeline.

Runs all steps in sequence:
  1. Build enhanced graph with drug class features
  2. Create zero-shot splits
  3. Train all models (5 seeds each)
  4. Comprehensive evaluation
  5. Ablation studies
  6. Publication figures

Usage:
    python run_zeroshot_pipeline.py                    # full pipeline
    python run_zeroshot_pipeline.py --steps 1,2,3      # specific steps
    python run_zeroshot_pipeline.py --fast             # single seed, fewer epochs
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STEPS = {
    1: ("pipeline/01_build_enhanced_graph.py", "Build enhanced graph"),
    2: ("pipeline/02_zeroshot_split.py",        "Create zero-shot splits"),
    3: ("pipeline/03_train.py",                 "Train all models"),
    4: ("pipeline/04_evaluate.py",              "Comprehensive evaluation"),
    5: ("pipeline/05_ablations.py",             "Ablation studies"),
    6: ("pipeline/06_visualize.py",             "Generate figures"),
}


def run_step(step_num: int, extra_args: list = None):
    script, desc = STEPS[step_num]
    script_path  = ROOT / script
    cmd          = [sys.executable, str(script_path)] + (extra_args or [])
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {desc}")
    print(f"{'='*60}")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n[ERROR] Step {step_num} failed (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n[OK] Step {step_num} completed in {elapsed:.1f}s")
    return elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=str, default=None,
                        help="Comma-separated step numbers, e.g. '1,2,3'")
    parser.add_argument("--fast", action="store_true",
                        help="Single seed, faster training for debugging")
    args = parser.parse_args()

    if args.steps:
        steps = [int(s) for s in args.steps.split(",")]
    else:
        steps = list(STEPS.keys())

    step3_args = []
    if args.fast:
        step3_args = ["--model", "zs_hetgat", "--seed", "42"]
        print("[FAST MODE] Training only ZS-HetGAT with seed=42")
    else:
        step3_args = ["--all_models"]

    print(f"\nZero-Shot AMR Prediction Pipeline")
    print(f"Running steps: {steps}")
    print(f"Working directory: {ROOT}")

    total_time = 0.0
    for step in steps:
        extra = step3_args if step == 3 else []
        elapsed = run_step(step, extra)
        total_time += elapsed

    print(f"\n{'='*60}")
    print(f"Pipeline complete!  Total time: {total_time:.1f}s")
    print(f"Results in: {ROOT / 'results/zeroshot/'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
