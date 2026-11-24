import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from joblib import Parallel, delayed, parallel_backend

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from preprocessing import (clip_artifacts, pan_tompkins, pick_lead_ii,
                           resample_to, wavelet_denoise)

BEAT_SYMBOLS = set("NLRBAaJSVrFejnEf/Qq?")


def reference_beats(record, db, annotator, fs_in):
    ann = wfdb.rdann(str(C.DATA_RAW / db / record), annotator)
    sym = np.asarray(ann.symbol)
    keep = np.array([s in BEAT_SYMBOLS for s in sym], dtype=bool)
    samples = np.asarray(ann.sample)[keep].astype(float)
    return np.round(samples * C.TARGET_FS / fs_in).astype(int)


def match(detected, reference, tolerance):
    detected = np.sort(np.asarray(detected))
    reference = np.sort(np.asarray(reference))
    if detected.size == 0 or reference.size == 0:
        return 0, detected.size, reference.size

    used = np.zeros(detected.size, dtype=bool)
    tp = 0
    j = 0
    for r in reference:
        while j < detected.size and detected[j] < r - tolerance:
            j += 1
        k = j
        best, best_d = -1, tolerance + 1
        while k < detected.size and detected[k] <= r + tolerance:
            if not used[k] and abs(detected[k] - r) < best_d:
                best, best_d = k, abs(detected[k] - r)
            k += 1
        if best >= 0:
            used[best] = True
            tp += 1
    return tp, int(detected.size - tp), int(reference.size - tp)


def evaluate_record(record, db, annotator, fs_in, tolerance):
    try:
        rec = wfdb.rdrecord(str(C.DATA_RAW / db / record))
        y = resample_to(wavelet_denoise(pick_lead_ii(rec)), fs_in, C.TARGET_FS)
        detected = pan_tompkins(clip_artifacts(y), C.TARGET_FS)
        ref = reference_beats(record, db, annotator, fs_in)
        ref = ref[(ref >= 0) & (ref < y.size)]
        tp, fp, fn = match(detected, ref, tolerance)
        se = 100 * tp / max(tp + fn, 1)
        pp = 100 * tp / max(tp + fp, 1)
        return {"record": record, "reference": int(ref.size),
                "detected": int(detected.size), "tp": tp, "fp": fp, "fn": fn,
                "sensitivity": se, "positive_predictivity": pp}
    except Exception as exc:
        return {"record": record, "error": str(exc)}


def _totals(rows):
    tp = sum(r["tp"] for r in rows)
    fp = sum(r["fp"] for r in rows)
    fn = sum(r["fn"] for r in rows)
    return {"records": len(rows), "reference_beats": tp + fn,
            "tp": tp, "fp": fp, "fn": fn,
            "sensitivity": 100 * tp / max(tp + fn, 1),
            "positive_predictivity": 100 * tp / max(tp + fp, 1)}


def main(tolerance_ms=150, n_jobs=-1, out=None):
    tol = int(round(tolerance_ms * C.TARGET_FS / 1000))
    jobs = [(r, C.INCART_DB, "atr", C.INCART_FS)
            for v in C.CAD_RECORDS.values() for r in v]
    jobs += [(r, C.FANTASIA_DB, "ecg", C.FANTASIA_FS) for r in C.NORMAL_RECORDS]

    print(f"Validating R-peak detection on {len(jobs)} records "
          f"(tolerance {tolerance_ms} ms = {tol} samples) ...", flush=True)
    with parallel_backend("loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=n_jobs, verbose=5)(
            delayed(evaluate_record)(r, db, ann, fs, tol) for r, db, ann, fs in jobs)

    good = [r for r in rows if "error" not in r]
    failed = [r for r in rows if "error" in r]

    summary = {"tolerance_ms": tolerance_ms, "records": rows, "by_class": {}}
    for name, prefix in (("CAD (INCART)", "I"), ("Normal (Fantasia)", "f")):
        sub = [r for r in good if r["record"].startswith(prefix)]
        if sub:
            summary["by_class"][name] = _totals(sub)
    summary["overall"] = _totals(good)

    out = Path(out) if out else C.TABLES / "detector_validation.json"
    out.write_text(json.dumps(summary, indent=2))

    print()
    for name, s in summary["by_class"].items():
        print(f"{name:20} Se = {s['sensitivity']:6.2f} %   "
              f"+P = {s['positive_predictivity']:6.2f} %   "
              f"({s['reference_beats']:,} reference beats)")
    o = summary["overall"]
    print(f"{'Overall':20} Se = {o['sensitivity']:6.2f} %   "
          f"+P = {o['positive_predictivity']:6.2f} %   "
          f"({o['reference_beats']:,} reference beats)")
    for f in failed:
        print("  FAILED", f)
    print(f"\nSaved -> {out}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance-ms", type=float, default=150)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(args.tolerance_ms, args.n_jobs, args.out)
