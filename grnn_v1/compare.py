"""Representational comparison of model and mouse V1 (numpy only).

Mouse: VISp units from Visual Coding Neuropixels sessions (processed by
C:/pianshi/invivo/process_nwb.py), 118 natural scenes x 50 repeats, spike counts
0-250 ms. Model: 118 scenes x 10 repeats, mean rate over 240 ms.

Geometry measures (all on condition means unless stated):
  rdm      : 1 - Pearson r between z-scored population vectors of two images
  rsa      : Spearman rho between upper triangles of model and mouse RDMs
  rsa_dis  : rsa / sqrt(rel_model * rel_mouse); rel = Spearman-Brown corrected
             split-half (repeat) reliability of each RDM
  cka      : linear CKA of image x unit matrices (centred over images)
  pr       : participation ratio of the image-response covariance, matched unit count
  sparse   : lifetime sparseness (Vinje & Gallant) per unit
  decode   : nearest-centroid image decoding from single repeats, matched unit count
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from scipy import stats

PROC = Path('C:/pianshi/invivo/processed')


# ------------------------------------------------------------------ mouse data
def load_mouse(min_rate_hz: float = 0.1):
    """Returns dict with trials [n_rep_min, 118, U] (Hz), mean [118, U], rs mask."""
    trials, rs = [], []
    files = sorted(PROC.glob('ns_*.npz'))
    for f in files:
        d = np.load(f)
        fr, c = d['frames'], d['counts'].astype(np.float32) / 0.25
        per_img = [c[fr == i] for i in range(118)]
        n = min(len(x) for x in per_img)
        t = np.stack([x[:n] for x in per_img], 1)               # [n, 118, U]
        trials.append(t[:48])
        rs.append(d['waveform_duration_ms'] > 0.4)
    n = min(t.shape[0] for t in trials)
    tr = np.concatenate([t[:n] for t in trials], axis=2)
    rs = np.concatenate(rs)
    keep = tr.mean((0, 1)) >= min_rate_hz
    return {'trials': tr[:, :, keep], 'mean': tr[:, :, keep].mean(0), 'rs': rs[keep], 'n_sessions': len(files)}


# ------------------------------------------------------------------ measures
def zscore_units(m: np.ndarray) -> np.ndarray:
    sd = m.std(0, keepdims=True)
    ok = sd[0] > 1e-9
    return (m[:, ok] - m[:, ok].mean(0, keepdims=True)) / sd[:, ok]


def rdm(m: np.ndarray) -> np.ndarray:
    z = zscore_units(m)
    return 1 - np.corrcoef(z)


def upper(x):
    i, j = np.triu_indices(len(x), 1)
    return x[i, j]


def rdm_reliability(trials: np.ndarray, n_split: int = 20, seed: int = 0) -> float:
    """Spearman-Brown corrected split-half reliability of the RDM over repeats."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_split):
        p = rng.permutation(trials.shape[0])
        a, b = trials[p[::2]].mean(0), trials[p[1::2]].mean(0)
        r = stats.spearmanr(upper(rdm(a)), upper(rdm(b)))[0]
        rs.append(r)
    r = float(np.mean(rs))
    return 2 * r / (1 + r) if r > 0 else np.nan


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(0); y = y - y.mean(0)
    hsic = np.linalg.norm(x.T @ y) ** 2
    return float(hsic / (np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)))


def participation_ratio(m: np.ndarray, n_units: int, reps: int = 20, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        idx = rng.choice(m.shape[1], min(n_units, m.shape[1]), replace=False)
        ev = np.clip(np.linalg.eigvalsh(np.cov(m[:, idx].T)), 0, None)
        out.append(ev.sum() ** 2 / (ev ** 2).sum())
    return float(np.mean(out))


def lifetime_sparseness(m: np.ndarray) -> np.ndarray:
    n = m.shape[0]
    a = (m.mean(0) ** 2) / np.maximum((m ** 2).mean(0), 1e-12)
    return (1 - a) / (1 - 1 / n)


def decode(trials: np.ndarray, n_units: int, reps: int = 10, seed: int = 0) -> float:
    """Leave-one-repeat-out nearest-centroid decoding of image identity (chance 1/118)."""
    rng = np.random.default_rng(seed)
    acc = []
    R = trials.shape[0]
    for _ in range(reps):
        idx = rng.choice(trials.shape[2], min(n_units, trials.shape[2]), replace=False)
        t = trials[:, :, idx]
        mu, sd = t.mean((0, 1)), t.std((0, 1)) + 1e-6
        t = (t - mu) / sd
        k = rng.integers(R)
        cent = np.delete(t, k, axis=0).mean(0)                  # [118, u]
        test = t[k]                                             # [118, u]
        d = ((test[:, None, :] - cent[None]) ** 2).sum(-1)
        acc.append(float((d.argmin(1) == np.arange(len(test))).mean()))
    return float(np.mean(acc))


def ks_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 3 or len(b) < 3:
        return np.nan
    return float(1 - stats.ks_2samp(a, b).statistic)


# ------------------------------------------------------------------ one network
def compare_network(ev: Dict[str, np.ndarray], mouse: Dict, targets: Dict, n_units: int = 200,
                    min_rate_hz: float = 0.1, seed: int = 0) -> Dict[str, float]:
    ns = ev['ns'].astype(np.float32)                           # [rep, 118, N]
    types = ev['types']
    out = {}
    active = ns.mean((0, 1)) >= min_rate_hz
    out['frac_active'] = float(active.mean())
    for label, sel_model, sel_mouse in [('all', active, np.ones(mouse['mean'].shape[1], bool)),
                                        ('exc', active & (types == 'Pyr'), mouse['rs'])]:
        if sel_model.sum() < 10:
            continue
        mt, mm = ns[:, :, sel_model], ns[:, :, sel_model].mean(0)
        xt, xm = mouse['trials'][:, :, sel_mouse], mouse['mean'][:, sel_mouse]
        rho = stats.spearmanr(upper(rdm(mm)), upper(rdm(xm)))[0]
        rel_m, rel_x = rdm_reliability(mt, seed=seed), rdm_reliability(xt, seed=seed)
        out[f'rsa_{label}'] = float(rho)
        out[f'rsa_dis_{label}'] = float(rho / np.sqrt(rel_m * rel_x)) if rel_m > 0 and rel_x > 0 else np.nan
        out[f'rel_model_{label}'] = rel_m
        out[f'cka_{label}'] = linear_cka(zscore_units(mm), zscore_units(xm))
        out[f'pr_{label}'] = participation_ratio(zscore_units(mm), n_units, seed=seed)
        out[f'pr_mouse_{label}'] = participation_ratio(zscore_units(xm), n_units, seed=seed)
        out[f'sparse_med_{label}'] = float(np.median(lifetime_sparseness(mm)))
        out[f'sparse_sim_{label}'] = ks_similarity(lifetime_sparseness(mm), lifetime_sparseness(xm))
        out[f'decode_{label}'] = decode(mt, n_units, seed=seed)
        out[f'decode_mouse_{label}'] = decode(xt[:mt.shape[0]], n_units, seed=seed)
    # tuning statistics vs in vivo (per type), 1 - KS
    for t in ('Pyr', 'PV', 'SST', 'VIP'):
        i = types == t
        tg = targets[t]
        resp = ev['gr_pref'][i] >= 0.5
        out[f'sim_spont_{t}'] = ks_similarity(ev['gr_spont'][i], tg['spont'])
        out[f'sim_pref_{t}'] = ks_similarity(ev['gr_pref'][i], tg['pref'])
        out[f'sim_osi_{t}'] = ks_similarity(ev['gr_osi'][i][resp], tg['osi'])
        out[f'sim_dsi_{t}'] = ks_similarity(ev['gr_dsi'][i][resp], tg['dsi'])
        out[f'med_pref_{t}'] = float(np.median(ev['gr_pref'][i]))
        out[f'med_osi_{t}'] = float(np.median(ev['gr_osi'][i][resp])) if resp.any() else np.nan
    out['sim_tuning_mean'] = float(np.nanmean([out[k] for k in out if k.startswith('sim_') and k.split('_')[1] in ('spont', 'pref', 'osi', 'dsi')]))
    return out


def mouse_ceilings(mouse: Dict, n_units: int = 200, seed: int = 0) -> Dict[str, float]:
    """Mouse-vs-mouse reference: RSA between two disjoint halves of units."""
    rng = np.random.default_rng(seed)
    rs_list, cka_list = [], []
    U = mouse['mean'].shape[1]
    for _ in range(20):
        p = rng.permutation(U)
        a, b = mouse['mean'][:, p[:U // 2]], mouse['mean'][:, p[U // 2:]]
        rs_list.append(stats.spearmanr(upper(rdm(a)), upper(rdm(b)))[0])
        cka_list.append(linear_cka(zscore_units(a), zscore_units(b)))
    return {'rsa_unit_halves': float(np.mean(rs_list)), 'cka_unit_halves': float(np.mean(cka_list)),
            'rel_mouse_all': rdm_reliability(mouse['trials'], seed=seed), 'n_units': U,
            'pr_mouse_all': participation_ratio(zscore_units(mouse['mean']), n_units, seed=seed),
            'decode_mouse_all': decode(mouse['trials'][:10], n_units, seed=seed)}
