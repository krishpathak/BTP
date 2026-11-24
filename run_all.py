import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    ("download", [sys.executable, "-u", "src/data_download.py"],
     "Downloading INCART and Fantasia recordings"),
    ("synthetic", [sys.executable, "-u", "src/synthetic_demo.py"],
     "Synthetic signal experiment"),
    ("build", [sys.executable, "-u", "src/build_dataset.py",
               "--n-jobs", "-1", "--nvm-stride", "5"],
     "Extracting features for every record"),
    ("validate", [sys.executable, "-u", "src/validate_detector.py"],
     "Checking R-peak detection against annotations"),
    ("classify", [sys.executable, "-u", "src/classify.py"],
     "Hyper-parameter search and cross-validation"),
    ("figures", [sys.executable, "-u", "src/figures.py"],
     "Generating figures"),
]

NAMES = [s[0] for s in STAGES]


def run(stage, cmd, blurb):
    print("\n" + "=" * 78)
    print(f"[{stage}] {blurb}")
    print("=" * 78, flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[{stage}] FAILED after {dt / 60:.1f} min (exit {proc.returncode})")
        sys.exit(proc.returncode)
    print(f"\n[{stage}] done in {dt / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Run the whole pipeline")
    ap.add_argument("--from", dest="start", choices=NAMES, default=NAMES[0])
    ap.add_argument("--to", dest="end", choices=NAMES, default=NAMES[-1])
    ap.add_argument("--only", choices=NAMES, default=None)
    args = ap.parse_args()

    if args.only:
        lo = hi = NAMES.index(args.only)
    else:
        lo, hi = NAMES.index(args.start), NAMES.index(args.end)

    t0 = time.time()
    for stage, cmd, blurb in STAGES[lo:hi + 1]:
        run(stage, cmd, blurb)
    print(f"\nFinished in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
