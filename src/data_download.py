import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C

BASE = "https://physionet.org/files/{db}/1.0.0/{name}"
EXTS = {C.INCART_DB: (".hea", ".dat", ".atr"),
        C.FANTASIA_DB: (".hea", ".dat", ".ecg")}
REQUIRED = {".hea", ".dat"}


def _flat(d):
    return [r for v in d.values() for r in v]


def _fetch(db, name, dest, retries=3):
    url = BASE.format(db=db, name=name)
    for attempt in range(retries):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                if r.status_code == 404:
                    return name, 0, "absent"
                r.raise_for_status()
                remote = int(r.headers.get("Content-Length", 0))
                if dest.exists() and remote and dest.stat().st_size == remote:
                    return name, dest.stat().st_size, "cached"
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(dest)
                return name, dest.stat().st_size, "ok"
        except Exception as exc:
            if attempt == retries - 1:
                return name, 0, f"error: {exc}"
            time.sleep(2 * (attempt + 1))
    return name, 0, "error"


def download(workers=8, force=False):
    tasks = []
    for db, records in ((C.INCART_DB, _flat(C.CAD_RECORDS)),
                        (C.FANTASIA_DB, C.NORMAL_RECORDS)):
        outdir = C.DATA_RAW / db
        outdir.mkdir(parents=True, exist_ok=True)
        for rec in records:
            for ext in EXTS[db]:
                dest = outdir / f"{rec}{ext}"
                if dest.exists() and not force and ext in REQUIRED:
                    continue
                tasks.append((db, f"{rec}{ext}", dest))

    if not tasks:
        print("All records already present.")
        return

    print(f"Fetching {len(tasks)} files with {workers} workers ...", flush=True)
    total, done, failed = 0, 0, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch, db, name, dest): name
                   for db, name, dest in tasks}
        for fut in as_completed(futures):
            name, nbytes, status = fut.result()
            done += 1
            total += nbytes
            if status.startswith("error"):
                failed.append((name, status))
            if done % 10 == 0 or done == len(tasks):
                print(f"  {done}/{len(tasks)}  ({total / 1e6:.0f} MB)", flush=True)

    print(f"Downloaded {total / 1e6:.1f} MB")
    for name, status in failed:
        print("  FAILED", name, status)


def verify():
    missing = []
    for db, records in ((C.INCART_DB, _flat(C.CAD_RECORDS)),
                        (C.FANTASIA_DB, C.NORMAL_RECORDS)):
        for rec in records:
            for ext in REQUIRED:
                p = C.DATA_RAW / db / f"{rec}{ext}"
                if not p.exists() or p.stat().st_size == 0:
                    missing.append(str(p.relative_to(C.DATA_RAW)))
    if missing:
        print(f"MISSING {len(missing)} file(s):")
        for m in missing[:20]:
            print("  ", m)
    else:
        print("All required .hea/.dat files present.")
    return missing


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if not args.verify_only:
        download(workers=args.workers, force=args.force)
    verify()
