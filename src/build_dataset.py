import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import wfdb
from joblib import Parallel, delayed, parallel_backend

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from energy_operators import OPERATORS
from features import beat_features_batch
from preprocessing import pick_lead_ii, record_to_beats
from scale_space import estimate_nvm
from vmd import vmd_batch

SHARD_DIR = C.DATA_PROC / "beats"
STAGE_DIR = C.DATA_PROC / "_staging"
SHARD_DIR.mkdir(parents=True, exist_ok=True)

SUBJECT_OF_RECORD = {r: s for s, rs in C.CAD_RECORDS.items() for r in rs}


def beats_to_features(beats, n_modes=None, estimate_nvms=True,
                      nvm_stride=1, batch=256):
    n_modes = C.N_MODES if n_modes is None else n_modes
    B = beats.shape[0]
    feats = {op: np.empty((B, C.N_FEATURES), dtype=np.float32)
             for op in C.ENERGY_OPERATORS}
    nvms = []

    if estimate_nvms:
        for i in range(0, B, max(1, nvm_stride)):
            nvms.append(estimate_nvm(beats[i].astype(float),
                                     C.SS_NUM_SCALES, C.SS_SCALE_STEP,
                                     C.NVM_MIN, C.NVM_MAX))

    for lo in range(0, B, batch):
        hi = min(lo + batch, B)
        chunk = beats[lo:hi].astype(float)

        modes, _, _ = vmd_batch(chunk, alpha=C.VMD_ALPHA, tau=C.VMD_TAU,
                                K=n_modes, DC=C.VMD_DC, init=C.VMD_INIT,
                                tol=C.VMD_TOL, max_iter=C.VMD_MAX_ITER)

        for op in C.ENERGY_OPERATORS:
            per_mode = OPERATORS[op](modes.reshape(-1, modes.shape[-1]))
            per_mode = per_mode.reshape(modes.shape)
            feats[op][lo:hi] = beat_features_batch(per_mode,
                                                   per_mode.sum(axis=1))

    return feats, np.asarray(nvms, dtype=np.int16)


def process_record(record, db, fs_in, label, nvm_stride=1,
                   max_beats=None, overwrite=False):
    shard = SHARD_DIR / f"{record}.npz"
    if shard.exists() and not overwrite:
        with np.load(shard) as z:
            return {"record": record, "status": "cached",
                    "n_beats": int(z["label"].size)}

    t0 = time.time()
    try:
        rec = wfdb.rdrecord(str(C.DATA_RAW / db / record))
        sig = pick_lead_ii(rec)
        beats, _, rpeaks = record_to_beats(sig, fs_in)

        if max_beats is not None and beats.shape[0] > max_beats:
            idx = np.linspace(0, beats.shape[0] - 1, max_beats).astype(int)
            beats = beats[idx]

        if beats.shape[0] == 0:
            return {"record": record, "status": "empty", "n_beats": 0}

        feats, nvms = beats_to_features(beats, nvm_stride=nvm_stride)

        payload = {f"X_{op}": feats[op] for op in C.ENERGY_OPERATORS}
        payload["label"] = np.full(beats.shape[0], label, dtype=np.int8)
        payload["nvm"] = nvms
        payload["record"] = np.array([record])
        payload["subject"] = np.array([SUBJECT_OF_RECORD.get(record, record)])
        payload["exemplar"] = beats[: min(5, beats.shape[0])]

        # write to a staging folder first (savez appends .npz to other names),
        # then move into place so half-written shards are never picked up
        STAGE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STAGE_DIR / f"{record}.npz"
        np.savez_compressed(tmp, **payload)
        tmp.replace(shard)

        return {"record": record, "status": "ok", "n_beats": beats.shape[0],
                "n_rpeaks": int(np.size(rpeaks)), "secs": round(time.time() - t0, 1)}
    except Exception as exc:
        traceback.print_exc()
        return {"record": record, "status": f"error: {exc}", "n_beats": 0}


def build(n_jobs=-1, nvm_stride=1, max_beats=None, overwrite=False,
          records=None):
    jobs = []
    for subj_recs in C.CAD_RECORDS.values():
        for r in subj_recs:
            jobs.append((r, C.INCART_DB, C.INCART_FS, 1))
    for r in C.NORMAL_RECORDS:
        jobs.append((r, C.FANTASIA_DB, C.FANTASIA_FS, 0))

    if records:
        wanted = set(records)
        jobs = [j for j in jobs if j[0] in wanted]

    print(f"Processing {len(jobs)} records with n_jobs={n_jobs} ...", flush=True)
    # one thread per worker, otherwise the workers fight over the same cores
    with parallel_backend("loky", inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_record)(r, db, fs, lab, nvm_stride, max_beats,
                                    overwrite)
            for r, db, fs, lab in jobs
        )

    ok = [r for r in results if r["status"] in ("ok", "cached")]
    total = sum(r["n_beats"] for r in ok)
    print(f"\nDone: {len(ok)}/{len(jobs)} records, {total} beats total")
    for r in results:
        if r["status"] not in ("ok", "cached"):
            print("  FAILED", r)
    return results


def assemble(out=None):
    out = Path(out) if out else C.DATA_PROC / "dataset.npz"
    shards = sorted(SHARD_DIR.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"no shards in {SHARD_DIR}; run build() first")

    X = {op: [] for op in C.ENERGY_OPERATORS}
    y, groups, nvm_cad, nvm_norm = [], [], [], []

    unreadable = []
    for s in shards:
        try:
            z = np.load(s, allow_pickle=False)
        except Exception as exc:
            unreadable.append((s.name, str(exc)))
            continue
        with z:
            lab = z["label"]
            if lab.size == 0:
                continue
            for op in C.ENERGY_OPERATORS:
                X[op].append(z[f"X_{op}"])
            y.append(lab)
            groups.append(np.full(lab.size, str(z["subject"][0])))
            (nvm_cad if lab[0] == 1 else nvm_norm).append(z["nvm"])

    data = {f"X_{op}": np.concatenate(X[op]) for op in C.ENERGY_OPERATORS}
    data["y"] = np.concatenate(y)
    data["groups"] = np.concatenate(groups)
    data["nvm_cad"] = np.concatenate(nvm_cad) if nvm_cad else np.array([], np.int16)
    data["nvm_normal"] = np.concatenate(nvm_norm) if nvm_norm else np.array([], np.int16)

    if unreadable:
        raise RuntimeError(
            f"{len(unreadable)} shard(s) could not be read - is a build still "
            f"running? " + ", ".join(n for n, _ in unreadable[:5]))

    np.savez_compressed(out, **data)
    n_cad = int((data["y"] == 1).sum())
    n_norm = int((data["y"] == 0).sum())
    print(f"Assembled -> {out}\n  CAD beats   : {n_cad}\n  Normal beats: {n_norm}"
          f"\n  Features    : {data['X_TKEO'].shape[1]} per operator")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--nvm-stride", type=int, default=1,
                    help="estimate the NVM on every k-th beat")
    ap.add_argument("--max-beats", type=int, default=None,
                    help="max beats kept per record")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--records", nargs="*", default=None)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    if not args.assemble_only:
        build(n_jobs=args.n_jobs, nvm_stride=args.nvm_stride,
              max_beats=args.max_beats, overwrite=args.overwrite,
              records=args.records)
    assemble()
