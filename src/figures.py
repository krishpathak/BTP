import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wfdb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from energy_operators import improved_energy
from preprocessing import (pan_tompkins, pick_lead_ii, resample_to,
                           segment_beats, wavelet_denoise)
from vmd import vmd

plt.rcParams.update({
    "font.size": 9, "font.family": "serif", "axes.grid": True,
    "grid.alpha": 0.25, "figure.dpi": 220, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})

BLUE, RED, GREEN, GREY = "#1f3b73", "#c1121f", "#0b7a3b", "#555555"


def fig_block_diagram(outdir):
    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.05)
    ax.axis("off")

    boxes = [
        (0.3, 4.05, 2.5, 0.95, "ECG databases\nINCART (CAD)\nFantasia (normal)", "#dbe5f1"),
        (3.3, 4.1, 2.5, 0.85, "Pre-processing\ndb6 denoise + resample", "#dbe5f1"),
        (6.3, 4.1, 3.3, 0.85, "R-peak detection (Pan-Tompkins)\nand 169-sample beat segmentation", "#dbe5f1"),
        (6.3, 2.7, 3.3, 0.85, "Magnitude spectrum + scale-space\nboundaries → data-driven NVM", "#fde8d0"),
        (3.3, 2.7, 2.5, 0.85, "VMD with N = 4\nmono-components", "#fde8d0"),
        (0.3, 2.7, 2.5, 0.85, "Energy operators\nTKEO / AGO / EDO", "#fde8d0"),
        (0.3, 1.3, 2.5, 0.85, "Improved energy\n(aggregate of modes)", "#dff0d8"),
        (3.3, 1.3, 2.5, 0.85, "Seg1 / Seg2 / Seg3\nmean, var, skew, kurt", "#dff0d8"),
        (6.3, 1.3, 3.3, 0.85, "60-dimensional feature vector\nper beat and per operator", "#dff0d8"),
        (3.3, 0.05, 3.3, 0.85, "Bayesian-optimised tree ensemble\n10-fold cross-validation", "#f3d9e2"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                    fc=color, ec="#33415c", lw=0.9))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=7.2, linespacing=1.35)

    def arrow(p, q):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9,
                                     lw=0.9, color="#33415c"))

    arrow((2.8, 4.52), (3.3, 4.52))
    arrow((5.8, 4.52), (6.3, 4.52))
    arrow((7.95, 4.1), (7.95, 3.6))
    arrow((6.3, 3.12), (5.8, 3.12))
    arrow((3.3, 3.12), (2.8, 3.12))
    arrow((1.55, 2.7), (1.55, 2.15))
    arrow((2.8, 1.72), (3.3, 1.72))
    arrow((5.8, 1.72), (6.3, 1.72))
    ax.plot([7.95, 7.95], [1.3, 0.47], lw=0.9, color="#33415c", zorder=0)
    arrow((7.95, 0.47), (6.65, 0.47))

    fig.savefig(Path(outdir) / "fig01_block_diagram.png")
    plt.close(fig)


def _load_beats(record, db, fs_in, n=4000):
    rec = wfdb.rdrecord(str(C.DATA_RAW / db / record))
    raw = pick_lead_ii(rec)
    clean = resample_to(wavelet_denoise(raw), fs_in, C.TARGET_FS)
    raw_rs = resample_to(raw, fs_in, C.TARGET_FS)
    peaks = pan_tompkins(clean, C.TARGET_FS)
    beats = segment_beats(clean, peaks)
    return raw_rs, clean, peaks, beats


def fig_preprocessing(outdir, cad_rec="I01", normal_rec="f1o01"):
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 3.6), sharex=True)
    for ax, (rec, db, fs, title) in zip(
            axes, [(cad_rec, C.INCART_DB, C.INCART_FS, f"CAD record {cad_rec} (INCART)"),
                   (normal_rec, C.FANTASIA_DB, C.FANTASIA_FS, f"Normal record {normal_rec} (Fantasia)")]):
        raw, clean, peaks, _ = _load_beats(rec, db, fs)
        t = np.arange(5 * C.TARGET_FS) / C.TARGET_FS
        seg = slice(10 * C.TARGET_FS, 15 * C.TARGET_FS)
        ax.plot(t, raw[seg] - np.mean(raw[seg]), lw=0.7, color=GREY,
                alpha=0.7, label="raw (mean-removed)")
        ax.plot(t, clean[seg], lw=0.9, color=BLUE, label="db6 denoised")
        pk = peaks[(peaks >= seg.start) & (peaks < seg.stop)] - seg.start
        ax.plot(pk / C.TARGET_FS, clean[seg][pk], "v", ms=4, color=RED,
                label="detected R-peak")
        ax.set_title(title, fontsize=8.5)
        ax.set_ylabel("Amplitude (mV)")
        ax.legend(fontsize=6.5, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig02_preprocessing.png")
    plt.close(fig)


def fig_nvm_pdf(outdir, dataset=None):
    z = np.load(dataset or C.DATA_PROC / "dataset.npz", allow_pickle=False)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
    opt = {}
    for ax, key, title, color in [
            (axes[0], "nvm_cad", "(a) CAD ECG beats", RED),
            (axes[1], "nvm_normal", "(b) Normal ECG beats", BLUE)]:
        v = z[key].astype(float)
        if v.size == 0:
            continue
        bins = np.arange(v.min() - 0.5, v.max() + 1.5)
        ax.hist(v, bins=bins, density=True, color=color, alpha=0.55,
                edgecolor="white", label="normalised histogram")
        mu, sd = v.mean(), max(v.std(), 1e-6)
        xs = np.linspace(v.min() - 1, v.max() + 1, 400)
        ax.plot(xs, norm.pdf(xs, mu, sd), color="k", lw=1.2, label="fitted PDF")
        peak = int(round(mu))
        opt[key] = peak
        ax.axvline(peak, ls="--", color=GREEN, lw=1.1,
                   label=f"optimum NVM = {peak}")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlabel("Number of variational modes")
        ax.legend(fontsize=6)
    axes[0].set_ylabel("Probability density")
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig06_nvm_pdf.png")
    plt.close(fig)
    return opt


def fig_beat_decomposition(outdir, beat, title, fname, operator="TKEO"):
    modes, _, _ = vmd(beat.astype(float), alpha=C.VMD_ALPHA, tau=C.VMD_TAU,
                      K=C.N_MODES, init=C.VMD_INIT, tol=C.VMD_TOL,
                      max_iter=C.VMD_MAX_ITER)
    per_mode, agg = improved_energy(modes, operator)
    beat_energy = improved_energy(beat[None, :].astype(float), operator)[1]

    rows = C.N_MODES + 2
    fig, axes = plt.subplots(rows, 2, figsize=(7.0, 1.0 * rows), sharex=True)
    t = np.arange(beat.size)

    axes[0, 0].plot(t, beat, lw=0.9, color=BLUE)
    axes[0, 0].set_ylabel("(a) beat", fontsize=7)
    axes[0, 1].plot(t, beat_energy, lw=0.9, color=RED)
    axes[0, 1].set_ylabel("(g) energy", fontsize=7)

    for k in range(C.N_MODES):
        axes[k + 1, 0].plot(t, modes[k], lw=0.8, color=GREY)
        axes[k + 1, 0].set_ylabel(f"IMF$_{k + 1}$", fontsize=7)
        axes[k + 1, 1].plot(t, per_mode[k], lw=0.8, color=GREEN)
        axes[k + 1, 1].set_ylabel(f"$\\Psi$[IMF$_{k + 1}$]", fontsize=7)

    axes[-1, 0].plot(t, modes.sum(axis=0), lw=0.9, color=BLUE)
    axes[-1, 0].set_ylabel("aggregate", fontsize=7)
    axes[-1, 1].plot(t, agg, lw=0.9, color=RED)
    axes[-1, 1].set_ylabel("IEO", fontsize=7)

    for a in axes[-1, :]:
        a.set_xlabel("Sample index")
    axes[0, 0].set_title(f"{title}: signal domain", fontsize=8.5)
    axes[0, 1].set_title(f"{title}: {operator} energy", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(Path(outdir) / fname)
    plt.close(fig)


def fig_segmentation(outdir, beat):
    fig, ax = plt.subplots(figsize=(6.2, 2.5))
    t = np.arange(beat.size)
    ax.plot(t, beat, lw=1.1, color=BLUE)
    colors = ["#fde8d0", "#dff0d8", "#dbe5f1"]
    for (a, b), name, col in zip(C.SEG_BOUNDS, C.SEG_NAMES, colors):
        ax.axvspan(a, b - 1, color=col, alpha=0.85, zorder=0)
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.92, name.split(" ")[0],
                ha="center", fontsize=8, weight="bold")
        ax.text((a + b) / 2, ax.get_ylim()[1] * 0.72,
                name.split(" ", 1)[1].strip("()"), ha="center", fontsize=6.5)
    for a, _ in C.SEG_BOUNDS[1:]:
        ax.axvline(a, color="k", ls="--", lw=0.8)
    ax.axvline(C.PRE_R, color=RED, ls=":", lw=1.0)
    ax.annotate("R", (C.PRE_R, beat[C.PRE_R]), textcoords="offset points",
                xytext=(4, 2), color=RED, fontsize=8, weight="bold")
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Segmentation of an ECG beat (169 samples)", fontsize=9)
    ax.set_xlim(0, beat.size - 1)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig09_segmentation.png")
    plt.close(fig)


def fig_class_average_beats(outdir, dataset_dir=None):
    shard_dir = Path(dataset_dir) if dataset_dir else C.DATA_PROC / "beats"
    pools = {1: [], 0: []}
    for s in sorted(shard_dir.glob("*.npz")):
        try:
            with np.load(s) as z:
                if z["label"].size:
                    pools[int(z["label"][0])].append(z["exemplar"])
        except Exception:
            continue
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    for lab, color, name in [(0, BLUE, "Normal"), (1, RED, "CAD")]:
        if not pools[lab]:
            continue
        arr = np.concatenate(pools[lab])
        m, s = arr.mean(0), arr.std(0)
        t = np.arange(m.size)
        ax.plot(t, m, lw=1.2, color=color, label=f"{name} (n={arr.shape[0]})")
        ax.fill_between(t, m - s, m + s, color=color, alpha=0.18)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalised amplitude")
    ax.set_title("Mean beat morphology per class", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig03_class_beats.png")
    plt.close(fig)


def fig_confusion(outdir, results):
    ops = [o for o in C.ENERGY_OPERATORS if o in results["operators"]]
    fig, axes = plt.subplots(1, len(ops), figsize=(2.4 * len(ops), 2.5))
    axes = np.atleast_1d(axes)
    for ax, op, tag in zip(axes, ops, "abcdef"):
        m = results["operators"][op]["segments"]["All"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]], dtype=float)
        pct = 100 * cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(pct, cmap="Blues", vmin=0, vmax=100)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{int(cm[i, j])}\n{pct[i, j]:.2f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if pct[i, j] > 55 else "black")
        ax.set_xticks([0, 1], ["Normal", "CAD"], fontsize=7)
        ax.set_yticks([0, 1], ["Normal", "CAD"], fontsize=7)
        ax.set_xlabel("Predicted", fontsize=7.5)
        ax.set_ylabel("True", fontsize=7.5)
        ax.set_title(f"({tag}) {op}", fontsize=8.5)
        ax.grid(False)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig10_confusion.png")
    plt.close(fig)


def fig_segment_accuracy(outdir, results):
    segs = ["Seg1", "Seg2", "Seg3", "All"]
    labels = ["Seg1\n(P-wave)", "Seg2\n(QRS)", "Seg3\n(T-wave)", "All\nsegments"]
    ops = [o for o in C.ENERGY_OPERATORS if o in results["operators"]]
    x = np.arange(len(segs))
    w = 0.8 / max(len(ops), 1)

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for i, (op, color) in enumerate(zip(ops, [BLUE, RED, GREEN])):
        vals = [results["operators"][op]["segments"][s]["accuracy"] for s in segs]
        b = ax.bar(x + i * w - 0.4 + w / 2, vals, w, label=op, color=color, alpha=0.85)
        ax.bar_label(b, fmt="%.2f", fontsize=5.5, padding=1)
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("10-fold CV accuracy (%)")
    lo = min(results["operators"][o]["segments"][s]["accuracy"]
             for o in ops for s in segs)
    ax.set_ylim(max(0, lo - 3), 100.6)
    ax.legend(fontsize=7, ncol=3)
    ax.set_title("Discriminative power of each energy segment", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig11_segment_accuracy.png")
    plt.close(fig)


def fig_bo_convergence(outdir, results):
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    plotted = False
    for op, color in zip(C.ENERGY_OPERATORS, [BLUE, RED, GREEN]):
        hist = results["operators"].get(op, {}).get("bo_history", [])
        if not hist:
            continue
        err = 100 * np.array([h["error"] for h in hist])
        ax.plot(np.arange(1, err.size + 1), np.minimum.accumulate(err),
                lw=1.3, color=color, label=op)
        ax.scatter(np.arange(1, err.size + 1), err, s=7, color=color, alpha=0.35)
        plotted = True
    if not plotted:
        plt.close(fig)
        return
    ax.set_xlabel("Bayesian-optimisation iteration")
    ax.set_ylabel("Inner-CV classification error (%)")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    ax.set_title("Hyper-parameter search convergence", fontsize=9)
    fig.tight_layout()
    fig.savefig(Path(outdir) / "fig12_bo_convergence.png")
    plt.close(fig)


def fig_feature_importance(outdir, dataset=None, operator="TKEO", top=18):
    from sklearn.ensemble import RandomForestClassifier
    from features import feature_names

    z = np.load(dataset or C.DATA_PROC / "dataset.npz", allow_pickle=False)
    X = np.nan_to_num(z[f"X_{operator}"])
    y = z["y"].astype(int)
    rng = np.random.default_rng(C.RANDOM_STATE)
    idx = rng.choice(X.shape[0], size=min(40000, X.shape[0]), replace=False)

    rf = RandomForestClassifier(n_estimators=150, n_jobs=-1,
                                random_state=C.RANDOM_STATE)
    rf.fit(X[idx], y[idx])
    imp = rf.feature_importances_
    names = feature_names(C.N_MODES)
    order = np.argsort(imp)[::-1][:top][::-1]

    fig, ax = plt.subplots(figsize=(6.2, 0.22 * top + 1.0))
    colors = [RED if "Seg1" in names[i] else GREEN if "Seg2" in names[i] else BLUE
              for i in order]
    ax.barh(range(len(order)), imp[order], color=colors, alpha=0.85)
    ax.set_yticks(range(len(order)), [names[i] for i in order], fontsize=6.5)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Most informative {operator} features", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, alpha=0.85)
               for c in (RED, GREEN, BLUE)]
    ax.legend(handles, ["Seg1 (P-wave)", "Seg2 (QRS)", "Seg3 (T-wave)"],
              fontsize=6.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(outdir) / f"fig13_feature_importance_{operator}.png")
    plt.close(fig)


def main(outdir=None, dataset=None, results_json=None):
    outdir = Path(outdir) if outdir else C.FIGURES
    outdir.mkdir(parents=True, exist_ok=True)

    print("fig01 block diagram")
    fig_block_diagram(outdir)

    try:
        print("fig02 preprocessing")
        fig_preprocessing(outdir)
    except Exception as exc:
        print("  skipped:", exc)

    ds = Path(dataset) if dataset else C.DATA_PROC / "dataset.npz"
    if ds.exists():
        print("fig03 class-average beats")
        fig_class_average_beats(outdir)
        print("fig06 NVM PDFs")
        fig_nvm_pdf(outdir, ds)

        shard_dir = C.DATA_PROC / "beats"
        exemplars = {}
        for s in sorted(shard_dir.glob("*.npz")):
            try:
                with np.load(s) as z:
                    if z["label"].size and int(z["label"][0]) not in exemplars:
                        exemplars[int(z["label"][0])] = z["exemplar"][0]
            except Exception:
                continue
            if len(exemplars) == 2:
                break
        if 0 in exemplars:
            print("fig07 healthy beat decomposition")
            fig_beat_decomposition(outdir, exemplars[0], "Healthy beat",
                                   "fig07_healthy_decomposition.png")
            fig_segmentation(outdir, exemplars[0])
        if 1 in exemplars:
            print("fig08 CAD beat decomposition")
            fig_beat_decomposition(outdir, exemplars[1], "CAD beat",
                                   "fig08_cad_decomposition.png")
        print("fig13 feature importance")
        fig_feature_importance(outdir, ds)

    rj = Path(results_json) if results_json else C.TABLES / "classification_results.json"
    if rj.exists():
        results = json.loads(rj.read_text())
        print("fig10 confusion matrices")
        fig_confusion(outdir, results)
        print("fig11 segment accuracy")
        fig_segment_accuracy(outdir, results)
        print("fig12 BO convergence")
        fig_bo_convergence(outdir, results)

    print(f"\nFigures written to {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    main(args.outdir, args.dataset, args.results)
