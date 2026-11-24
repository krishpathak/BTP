import numpy as np

from config import SEG_BOUNDS, STATS


def _moments(seg):
    m = seg.mean()
    d = seg - m
    var = float(np.mean(d ** 2))
    s = np.sqrt(var)
    if s < 1e-300:
        return m, var, 0.0, 0.0
    skew = float(np.mean(d ** 3) / s ** 3)
    kurt = float(np.mean(d ** 4) / var ** 2)
    return float(m), var, skew, kurt


def beat_features(per_mode_energy, aggregate_energy):
    signals = np.vstack([np.atleast_2d(per_mode_energy),
                         aggregate_energy[None, :]])
    out = np.empty(signals.shape[0] * len(SEG_BOUNDS) * len(STATS),
                   dtype=np.float32)
    i = 0
    for sig in signals:
        for a, b in SEG_BOUNDS:
            for v in _moments(sig[a:b]):
                out[i] = v
                i += 1
    return out


def beat_features_batch(per_mode_energy, aggregate_energy):
    per_mode = np.asarray(per_mode_energy, dtype=float)
    agg = np.asarray(aggregate_energy, dtype=float)
    sig = np.concatenate([per_mode, agg[:, None, :]], axis=1)

    B, S, _ = sig.shape
    out = np.empty((B, S, len(SEG_BOUNDS), len(STATS)), dtype=np.float64)

    for g, (a, b) in enumerate(SEG_BOUNDS):
        seg = sig[:, :, a:b]
        m = seg.mean(axis=-1)
        d = seg - m[..., None]
        var = np.mean(d ** 2, axis=-1)
        s = np.sqrt(var)
        safe = s > 1e-150
        with np.errstate(divide="ignore", invalid="ignore"):
            skew = np.where(safe, np.mean(d ** 3, axis=-1) / np.where(safe, s ** 3, 1.0), 0.0)
            kurt = np.where(safe, np.mean(d ** 4, axis=-1) / np.where(safe, var ** 2, 1.0), 0.0)
        out[:, :, g, 0] = m
        out[:, :, g, 1] = var
        out[:, :, g, 2] = skew
        out[:, :, g, 3] = kurt

    return out.reshape(B, -1).astype(np.float32)


def feature_names(n_modes):
    sig_names = [f"IMF{k + 1}" for k in range(n_modes)] + ["Aggregate"]
    seg_names = [f"Seg{j + 1}" for j in range(len(SEG_BOUNDS))]
    return [f"{s}_{g}_{m}" for s in sig_names for g in seg_names for m in STATS]


def segment_column_mask(n_modes, seg_index):
    tag = f"Seg{seg_index + 1}_"
    return np.array([tag in n for n in feature_names(n_modes)])
