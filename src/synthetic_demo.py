import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config as C
from energy_operators import OPERATORS, conventional_energy, improved_energy
from scale_space import detect_boundaries, magnitude_spectrum
from vmd import vmd

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
    "figure.dpi": 200, "savefig.bbox": "tight", "font.family": "serif",
})

FS = 1024
N = 512

P = dict(p0=0.05,
         p1=0.15, p2=-3.287e-5, p3=2.5481e-7,
         p4=0.28, p5=-2.4e-5,
         p6=0.45, p7=3.2875e-5, p8=-2.5481e-7)


def components():
    n = np.arange(N, dtype=float)
    phases = [
        2 * np.pi * (P["p0"] * n),
        2 * np.pi * (P["p1"] * n + P["p2"] * n ** 2 / 2 + P["p3"] * n ** 3 / 3),
        2 * np.pi * (P["p4"] * n + P["p5"] * n ** 2 / 2),
        2 * np.pi * (P["p6"] * n + P["p7"] * n ** 2 / 2 + P["p8"] * n ** 3 / 3),
    ]
    freqs = [
        np.full_like(n, P["p0"]),
        P["p1"] + P["p2"] * n + P["p3"] * n ** 2,
        P["p4"] + P["p5"] * n,
        P["p6"] + P["p7"] * n + P["p8"] * n ** 2,
    ]
    return np.vstack([np.cos(p) for p in phases]), np.vstack(freqs)


def true_total_energy(comps, freqs):
    omega = 2 * np.pi * freqs
    return (omega ** 2).sum(axis=0)


def run(outdir=None):
    outdir = Path(outdir) if outdir else C.FIGURES
    outdir.mkdir(parents=True, exist_ok=True)

    comps, freqs = components()
    ss = comps.sum(axis=0)

    spec = magnitude_spectrum(ss)
    bnds = detect_boundaries(ss, C.SS_NUM_SCALES, C.SS_SCALE_STEP)
    nvm = len(bnds) - 1
    f_axis = np.linspace(0, FS / 2, spec.size)

    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    ax.plot(f_axis, spec, lw=1.0, color="#1f3b73")
    for b in bnds:
        ax.axvline(f_axis[b], ls="--", lw=0.9, color="#c1121f")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Normalised magnitude")
    ax.set_title(f"Magnitude spectrum and detected boundaries  (NVM = {nvm})")
    fig.savefig(outdir / "fig_synth_boundaries.png")
    plt.close(fig)

    modes, _, omega = vmd(ss, alpha=C.VMD_ALPHA, tau=C.VMD_TAU, K=nvm,
                          init=C.VMD_INIT, tol=C.VMD_TOL,
                          max_iter=C.VMD_MAX_ITER)

    # pair each recovered mode with the true component of closest centre frequency
    true_cf = freqs.mean(axis=1)
    pairing = [int(np.argmin(np.abs(true_cf - w))) for w in omega]
    rmse = {}
    for k, j in enumerate(pairing):
        a, b = modes[k], comps[j]
        sign = np.sign(np.dot(a, b)) or 1.0
        rmse[f"ss{j + 1}(t)"] = float(np.sqrt(np.mean((sign * a - b) ** 2)))
    recon_rmse = float(np.sqrt(np.mean((modes.sum(axis=0) - ss) ** 2)))

    fig, axes = plt.subplots(nvm + 2, 2, figsize=(7.2, 1.05 * (nvm + 2)),
                             sharex=True)
    axes[0, 0].plot(ss, lw=0.7, color="#1f3b73")
    axes[0, 0].set_ylabel("ss(t)", fontsize=8)
    axes[0, 1].plot(modes.sum(axis=0), lw=0.7, color="#0b7a3b")
    axes[0, 1].set_ylabel("recon.", fontsize=8)
    for k in range(nvm):
        axes[k + 1, 0].plot(comps[pairing[k]], lw=0.7, color="#444")
        axes[k + 1, 0].set_ylabel(f"ss$_{pairing[k] + 1}$", fontsize=8)
        axes[k + 1, 1].plot(modes[k], lw=0.7, color="#c1121f")
        axes[k + 1, 1].set_ylabel(f"IMF$_{k + 1}$", fontsize=8)
    axes[-1, 0].plot(ss - modes.sum(axis=0), lw=0.7, color="#777")
    axes[-1, 0].set_ylabel("residual", fontsize=8)
    axes[-1, 1].axis("off")
    axes[0, 0].set_title("Actual components", fontsize=9)
    axes[0, 1].set_title("VMD mono-components", fontsize=9)
    for a in axes[-1, :]:
        a.set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(outdir / "fig_synth_decomposition.png")
    plt.close(fig)

    actual = true_total_energy(comps, freqs)
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 5.4), sharex=True, sharey="row")
    err = {}
    for r, opname in enumerate(C.ENERGY_OPERATORS):
        conv = conventional_energy(ss, opname)
        _, impr = improved_energy(modes, opname)

        # operators only match the true energy up to a scale factor
        def _fit(e):
            s = np.dot(e, actual) / max(np.dot(e, e), 1e-30)
            return s * e
        conv_s, impr_s = _fit(conv), _fit(impr)

        axes[r, 0].plot(conv_s, lw=0.7, color="#c1121f", label="conventional")
        axes[r, 0].plot(actual, lw=0.9, color="#1f3b73", ls="--", label="actual")
        axes[r, 1].plot(impr_s, lw=0.7, color="#0b7a3b", label="improved (IEO)")
        axes[r, 1].plot(actual, lw=0.9, color="#1f3b73", ls="--", label="actual")
        axes[r, 0].set_ylabel(opname, fontsize=9)
        for a in axes[r]:
            a.legend(fontsize=6, loc="upper right")

        den = np.sqrt(np.mean(actual ** 2))
        err[opname] = {
            "conventional_nrmse": float(np.sqrt(np.mean((conv_s - actual) ** 2)) / den),
            "improved_nrmse": float(np.sqrt(np.mean((impr_s - actual) ** 2)) / den),
        }
    axes[0, 0].set_title("Conventional energy estimation", fontsize=9)
    axes[0, 1].set_title("IEO-based energy estimation", fontsize=9)
    for a in axes[-1, :]:
        a.set_xlabel("Sample index")
    fig.tight_layout()
    fig.savefig(outdir / "fig_synth_energy.png")
    plt.close(fig)

    summary = {"detected_nvm": nvm,
               "boundaries_hz": [float(f_axis[b]) for b in bnds],
               "mode_centre_freqs_hz": [float(w * FS) for w in omega],
               "component_rmse": rmse,
               "reconstruction_rmse": recon_rmse,
               "energy_nrmse": err}
    (C.TABLES / "synthetic_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import pprint
    pprint.pprint(run())
