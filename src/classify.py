import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import (AdaBoostClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from bayesopt import Space, minimize
from features import feature_names, segment_column_mask


def metrics_from_counts(tp, tn, fp, fn):
    tp, tn, fp, fn = map(float, (tp, tn, fp, fn))
    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    sen = tp / (tp + fn) if (tp + fn) else 0.0
    spe = tn / (tn + fp) if (tn + fp) else 0.0
    pre = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = tp / (tp + 0.5 * (fp + fn)) if (tp + 0.5 * (fp + fn)) else 0.0
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den > 0 else 0.0
    return {"accuracy": 100 * acc, "sensitivity": 100 * sen,
            "specificity": 100 * spe, "precision": 100 * pre,
            "f1": 100 * f1, "mcc": mcc,
            "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)}


def evaluate(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return metrics_from_counts(tp, tn, fp, fn)


# AdaBoost is left out of the search: sklearn fits it one learner at a time
# and a single fold takes longer than the whole search
ENSEMBLE_SPACE = Space({
    "method": ("cat", ["Bag", "LogitBoost"]),
    "n_estimators": ("log", 30, 300),
    "learn_rate": ("log", 0.01, 1.0),
    "max_num_splits": ("log", 8, 4096),
    "min_leaf_size": ("log", 1, 64),
})


def make_model(params, random_state=C.RANDOM_STATE, n_jobs=-1):
    method = params.get("method", "Bag")
    n_est = int(params.get("n_estimators", 150))
    max_splits = int(params.get("max_num_splits", 1024))
    min_leaf = int(params.get("min_leaf_size", 1))
    lr = float(params.get("learn_rate", 0.1))

    if method == "Bag":
        return RandomForestClassifier(
            n_estimators=n_est, max_leaf_nodes=max_splits + 1,
            min_samples_leaf=min_leaf, n_jobs=n_jobs,
            random_state=random_state, bootstrap=True)

    if method == "AdaBoostM1":
        base = DecisionTreeClassifier(
            max_leaf_nodes=int(np.clip(max_splits + 1, 2, 64)),
            min_samples_leaf=min_leaf, random_state=random_state)
        return AdaBoostClassifier(estimator=base, n_estimators=min(n_est, 150),
                                  learning_rate=lr, random_state=random_state)

    return HistGradientBoostingClassifier(
        max_iter=n_est, learning_rate=lr,
        max_leaf_nodes=int(np.clip(max_splits + 1, 2, 255)),
        min_samples_leaf=max(min_leaf, 1),
        early_stopping=False, random_state=random_state)


def cross_validate(X, y, params, groups=None, n_splits=C.CV_FOLDS,
                   subject_wise=False, n_jobs=-1, return_preds=False,
                   verbose=False):
    X = np.ascontiguousarray(X, dtype=np.float32)
    y = np.asarray(y).astype(int)

    if subject_wise:
        if groups is None:
            raise ValueError("subject-wise CV needs group labels")
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True,
                                        random_state=C.RANDOM_STATE)
        folds = splitter.split(X, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=C.RANDOM_STATE)
        folds = splitter.split(X, y)

    preds = np.zeros_like(y)
    fold_metrics = []
    for k, (tr, te) in enumerate(folds):
        t0 = time.time()
        model = make_model(params, n_jobs=n_jobs)
        model.fit(X[tr], y[tr])
        preds[te] = model.predict(X[te])
        fm = evaluate(y[te], preds[te])
        fold_metrics.append(fm)
        if verbose:
            print(f"    fold {k + 1}/{n_splits}: acc={fm['accuracy']:.3f}%"
                  f"  ({time.time() - t0:.0f}s)", flush=True)

    pooled = evaluate(y, preds)
    pooled["fold_accuracy_mean"] = float(np.mean([f["accuracy"] for f in fold_metrics]))
    pooled["fold_accuracy_std"] = float(np.std([f["accuracy"] for f in fold_metrics]))
    return (pooled, preds) if return_preds else pooled


def optimise(X, y, n_iter=C.BO_ITERATIONS, subsample=C.BO_SUBSAMPLE,
             inner_folds=C.BO_INNER_FOLDS, n_jobs=-1, verbose=True):
    rng = np.random.default_rng(C.RANDOM_STATE)
    if subsample and X.shape[0] > subsample:
        idx = []
        for cls in np.unique(y):
            pool = np.flatnonzero(y == cls)
            take = max(1, int(round(subsample * pool.size / y.size)))
            idx.append(rng.choice(pool, size=min(take, pool.size), replace=False))
        idx = np.sort(np.concatenate(idx))
        Xs, ys = X[idx], y[idx]
    else:
        Xs, ys = X, y

    if verbose:
        print(f"  Bayesian optimisation on {Xs.shape[0]} beats, "
              f"{inner_folds}-fold inner CV, {n_iter} iterations", flush=True)

    def objective(params):
        m = cross_validate(Xs, ys, params, n_splits=inner_folds, n_jobs=n_jobs)
        return 1.0 - m["accuracy"] / 100.0

    best, val, hist = minimize(objective, ENSEMBLE_SPACE, n_iter=n_iter,
                               n_init=8, random_state=C.RANDOM_STATE,
                               verbose=verbose)
    if verbose:
        print(f"  best inner error = {val:.6f}  ->  {best}", flush=True)
    return best, val, hist


def load_dataset(path=None):
    path = Path(path) if path else C.DATA_PROC / "dataset.npz"
    return np.load(path, allow_pickle=False)


def run_experiments(dataset=None, n_jobs=-1, bo_iter=C.BO_ITERATIONS,
                    skip_bo=False, out=None):
    z = load_dataset(dataset)
    y = z["y"].astype(int)
    groups = z["groups"]
    results = {"generated": datetime.now().isoformat(timespec="seconds"),
               "n_cad": int((y == 1).sum()), "n_normal": int((y == 0).sum()),
               "n_beats": int(y.size),
               "n_features": int(z["X_TKEO"].shape[1]),
               "cv_folds": C.CV_FOLDS, "operators": {}}

    names = feature_names(C.N_MODES)
    masks = {"All": np.ones(len(names), dtype=bool)}
    for s in range(len(C.SEG_BOUNDS)):
        masks[f"Seg{s + 1}"] = segment_column_mask(C.N_MODES, s)

    for op in C.ENERGY_OPERATORS:
        X = z[f"X_{op}"]
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"\n=== {op} : {X.shape[0]} beats x {X.shape[1]} features ===",
              flush=True)

        if skip_bo:
            best = {"method": "Bag", "n_estimators": 200,
                    "max_num_splits": 4096, "min_leaf_size": 1,
                    "learn_rate": 0.1}
            bo_hist = []
        else:
            best, _, bo_hist = optimise(X, y, n_iter=bo_iter, n_jobs=n_jobs)

        entry = {"best_params": best,
                 "bo_history": [{"params": p, "error": v} for p, v in bo_hist],
                 "segments": {}}

        for seg_name, mask in masks.items():
            t0 = time.time()
            print(f"  [{op}] {C.CV_FOLDS}-fold CV on {seg_name} "
                  f"({int(mask.sum())} features)", flush=True)
            m, preds = cross_validate(X[:, mask], y, best, n_splits=C.CV_FOLDS,
                                      n_jobs=n_jobs, return_preds=True,
                                      verbose=True)
            m["seconds"] = round(time.time() - t0, 1)
            m["n_features"] = int(mask.sum())
            entry["segments"][seg_name] = m
            print(f"     -> acc={m['accuracy']:.2f}%  sen={m['sensitivity']:.2f}%"
                  f"  spe={m['specificity']:.2f}%  MCC={m['mcc']:.4f}", flush=True)
            if seg_name == "All":
                np.save(C.RESULTS / f"preds_{op}.npy", preds)

        print(f"  [{op}] subject-wise CV", flush=True)
        entry["subject_wise"] = cross_validate(X, y, best, groups=groups,
                                               n_splits=7, subject_wise=True,
                                               n_jobs=n_jobs, verbose=True)
        print(f"     -> acc={entry['subject_wise']['accuracy']:.2f}%", flush=True)

        results["operators"][op] = entry

    out = Path(out) if out else C.TABLES / "classification_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved -> {out}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--bo-iter", type=int, default=C.BO_ITERATIONS)
    ap.add_argument("--skip-bo", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_experiments(dataset=args.dataset, n_jobs=args.n_jobs,
                    bo_iter=args.bo_iter, skip_bo=args.skip_bo, out=args.out)
