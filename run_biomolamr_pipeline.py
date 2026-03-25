"""
run_biomolamr_pipeline.py — Master orchestration script for BioMolAMR

Runs all pipeline steps in order, with optional skip flags.

Usage:
    python run_biomolamr_pipeline.py                  # run everything
    python run_biomolamr_pipeline.py --skip_esm2      # skip ESM-2 extraction
    python run_biomolamr_pipeline.py --from_step 3    # start from step 3
    python run_biomolamr_pipeline.py --only_eval      # just run evaluation
    python run_biomolamr_pipeline.py --app            # launch Streamlit app
    python run_biomolamr_pipeline.py --dry_run        # print steps, don't execute

Pipeline steps:
    0a  Extract ESM-2 protein embeddings      (~15-25 min on MPS/GPU, ~2h CPU)
    0b  Compute molecular drug fingerprints   (~2 min)
    0c  Build BioMolAMR heterogeneous graph   (~1 min)
    2b  Create extended 12-class ZS splits    (~1 min)
    3b  Train all models (5 seeds each)       (~2-4h on MPS/GPU)
    8   Extended evaluation + figures         (~5 min)
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Step definitions ──────────────────────────────────────────────────────────

STEPS = [
    {
        "id":     "00a",
        "name":   "Extract ESM-2 embeddings",
        "script": "pipeline/00a_extract_esm2_embeddings.py",
        "output": "data/processed/esm2_embeddings.pt",
        "note":   "~15-25 min on MPS/GPU. Uses esm2_t12_35M_UR50D (480-dim).",
    },
    {
        "id":     "00b",
        "name":   "Compute drug fingerprints",
        "script": "pipeline/00b_compute_drug_fingerprints.py",
        "output": "data/processed/drug_fingerprints.pt",
        "note":   "~2 min. Morgan(2048)+MACCS(167)+TopoTorsion(1024) = 3245-dim.",
    },
    {
        "id":     "00c",
        "name":   "Build BioMolAMR graph",
        "script": "pipeline/00c_build_biomolamr_graph.py",
        "output": "data/processed/biomolamr_graph.pkl",
        "note":   "~1 min. Requires 00a and 00b outputs.",
        "requires": ["data/processed/esm2_embeddings.pt",
                     "data/processed/drug_fingerprints.pt"],
    },
    {
        "id":     "02b",
        "name":   "Create extended 12-class ZS splits",
        "script": "pipeline/02b_extended_zeroshot_split.py",
        "output": "data/processed/extended_splits.pkl",
        "note":   "~1 min. Requires BioMolAMR graph.",
        "requires": ["data/processed/biomolamr_graph.pkl"],
    },
    {
        "id":     "03b",
        "name":   "Train all models (5 × 5 seeds)",
        "script": "pipeline/03b_train_biomolamr.py",
        "output": "results/biomolamr/models/",
        "note":   "~2-4h on MPS/GPU. Trains BioMolAMR + 4 baselines, 5 seeds each.",
        "requires": ["data/processed/biomolamr_graph.pkl",
                     "data/processed/extended_splits.pkl"],
    },
    {
        "id":     "08",
        "name":   "Extended evaluation + figures",
        "script": "pipeline/08_extended_evaluation.py",
        "output": "results/biomolamr/eval_summary.json",
        "note":   "~5 min. Generates LaTeX table + publication figures.",
        "requires": ["results/biomolamr/models/"],
    },
]

STEP_IDS = [s["id"] for s in STEPS]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _color(text, code):
    return f"\033[{code}m{text}\033[0m"

def green(t):  return _color(t, "32")
def yellow(t): return _color(t, "33")
def red(t):    return _color(t, "31")
def bold(t):   return _color(t, "1")


def check_output_exists(step):
    out = ROOT / step["output"]
    if out.suffix:        # file
        return out.exists()
    return out.is_dir() and any(out.iterdir())  # non-empty dir


def check_requires(step):
    missing = []
    for req in step.get("requires", []):
        p = ROOT / req
        if not (p.exists() and (p.is_file() or any(p.iterdir()) if p.is_dir() else True)):
            missing.append(req)
    return missing


def run_step(step, dry_run=False, force=False):
    script = ROOT / step["script"]

    if not script.exists():
        print(red(f"  ERROR: Script not found: {script}"))
        return False

    missing_reqs = check_requires(step)
    if missing_reqs:
        print(red(f"  BLOCKED: Missing required inputs:"))
        for r in missing_reqs:
            print(red(f"    - {r}"))
        return False

    if not force and check_output_exists(step):
        print(yellow(f"  SKIP (output already exists): {step['output']}"))
        return True

    if dry_run:
        print(yellow(f"  DRY RUN: python {step['script']}"))
        return True

    print(bold(f"  Running: python {step['script']}"))
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=ROOT,
    )
    elapsed = time.time() - t0
    if result.returncode == 0:
        print(green(f"  Done in {elapsed:.1f}s"))
        return True
    else:
        print(red(f"  FAILED (exit code {result.returncode}) after {elapsed:.1f}s"))
        return False


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BioMolAMR pipeline orchestrator")
    parser.add_argument("--skip_esm2",  action="store_true",
                        help="Skip ESM-2 extraction (use if embeddings already exist)")
    parser.add_argument("--skip_train", action="store_true",
                        help="Skip training (use if checkpoints already exist)")
    parser.add_argument("--from_step",  default="00a",
                        help="Start from this step ID (e.g. 00c, 02b, 03b, 08)")
    parser.add_argument("--only_eval",  action="store_true",
                        help="Run only step 08 (evaluation)")
    parser.add_argument("--force",      action="store_true",
                        help="Re-run steps even if output already exists")
    parser.add_argument("--dry_run",    action="store_true",
                        help="Print steps without executing")
    parser.add_argument("--app",        action="store_true",
                        help="Launch Streamlit app after pipeline")
    parser.add_argument("--step",       default=None,
                        help="Run only this single step ID")
    args = parser.parse_args()

    print()
    print(bold("=" * 64))
    print(bold("  BioMolAMR Pipeline"))
    print(bold("=" * 64))
    print()

    # Filter steps
    steps_to_run = STEPS[:]

    if args.only_eval:
        steps_to_run = [s for s in steps_to_run if s["id"] == "08"]

    elif args.step:
        steps_to_run = [s for s in steps_to_run if s["id"] == args.step]
        if not steps_to_run:
            print(red(f"Unknown step id: {args.step}. Choose from: {STEP_IDS}"))
            sys.exit(1)

    else:
        if args.from_step != "00a":
            if args.from_step not in STEP_IDS:
                print(red(f"Unknown --from_step: {args.from_step}. Choose from: {STEP_IDS}"))
                sys.exit(1)
            start = STEP_IDS.index(args.from_step)
            steps_to_run = steps_to_run[start:]

        if args.skip_esm2:
            steps_to_run = [s for s in steps_to_run if s["id"] != "00a"]

        if args.skip_train:
            steps_to_run = [s for s in steps_to_run if s["id"] != "03b"]

    # Status overview
    print(f"{'Step':<6} {'Name':<42} {'Status'}")
    print("-" * 70)
    for step in STEPS:
        exists = check_output_exists(step)
        skip   = step not in steps_to_run
        status = (green("done")  if exists else
                  yellow("skip") if skip  else
                  bold("will run"))
        print(f"  {step['id']:<5} {step['name']:<42} {status}")
    print()

    if args.dry_run:
        print(yellow("DRY RUN — no commands will be executed."))
        print()

    # Execute
    success_count = 0
    for step in steps_to_run:
        print(bold(f"[{step['id']}] {step['name']}"))
        print(f"       {step['note']}")
        ok = run_step(step, dry_run=args.dry_run, force=args.force)
        if ok:
            success_count += 1
        else:
            print(red("  Pipeline halted."))
            sys.exit(1)
        print()

    print(bold("=" * 64))
    print(green(f"  Pipeline complete: {success_count}/{len(steps_to_run)} steps succeeded."))
    print(bold("=" * 64))
    print()

    # Summary of outputs
    print("Key outputs:")
    for step in STEPS:
        out = ROOT / step["output"]
        if out.exists():
            size = ""
            if out.is_file():
                size = f"  ({out.stat().st_size / 1e6:.1f} MB)"
            print(f"  {green('✓')} {step['output']}{size}")
        else:
            print(f"  {yellow('○')} {step['output']}  (not yet generated)")
    print()

    if args.app:
        print(bold("Launching Streamlit app…"))
        app_path = ROOT / "app/streamlit_app.py"
        subprocess.run(["streamlit", "run", str(app_path)], cwd=ROOT)


if __name__ == "__main__":
    main()
