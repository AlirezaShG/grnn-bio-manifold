"""Evaluation protocols shared by every condition (no gradients).

natural_scenes: the 118 Brain Observatory images, 240 ms each, back-to-back in a
random order per repeat (as in the Neuropixels recordings), after 500 ms gray.
gratings: 8 directions, 2 s each (0.04 cpd, 2 Hz, 80%), after 500 ms gray;
plus 2 s gray trials for spontaneous rates.
"""
from __future__ import annotations

import numpy as np
import torch

from .core import grating_movie
from .learn import DIRECTIONS, tuning_stats

IMG_BINS, WARM = 12, 25


@torch.no_grad()
def natural_scenes(circuit, lgn, scenes: np.ndarray, n_rep: int = 10, seed: int = 0, chunk: int = 5):
    """Returns responses [n_rep, 118, N] (mean rate, Hz, over each 240 ms presentation)."""
    rng = np.random.default_rng(seed)
    orders = np.stack([rng.permutation(len(scenes)) for _ in range(n_rep)])
    out = np.zeros((n_rep, len(scenes), circuit.N), np.float32)
    for s in range(0, n_rep, chunk):
        o = orders[s:s + chunk]
        T = WARM + len(scenes) * IMG_BINS
        x = np.zeros((len(o), T, 48, 64), np.float32)
        for b, order in enumerate(o):
            x[b, WARM:] = np.repeat(scenes[order], IMG_BINS, axis=0)
        lg = lgn(torch.tensor(x, device=circuit.device))
        r, _ = circuit.run(lg, noise_seed=seed * 1000 + s)
        r = r[:, WARM:].reshape(len(o), len(scenes), IMG_BINS, -1).mean(2).cpu().numpy()
        for b, order in enumerate(o):
            out[s + b, order] = r[b]
    return out


@torch.no_grad()
def gratings(circuit, lgn, n_rep: int = 5, seed: int = 0, bins: int = 100):
    rng = np.random.default_rng(seed)
    movies, labels = [], []
    for d in DIRECTIONS:
        for _ in range(n_rep):
            m = np.zeros((WARM + bins, 48, 64), np.float32)
            m[WARM:] = grating_movie(d, bins, phase=rng.uniform(0, 2 * np.pi))
            movies.append(m); labels.append(d)
    for _ in range(n_rep):
        movies.append(np.zeros((WARM + bins, 48, 64), np.float32)); labels.append(-1)
    x = torch.tensor(np.stack(movies), device=circuit.device)
    r, _ = circuit.run(lgn(x), noise_seed=seed + 77)
    r = r[:, WARM:].mean(1)                                      # [B, N]
    labels = np.asarray(labels)
    R = torch.stack([r[labels == d].mean(0) for d in DIRECTIONS], 1)   # [N, 8]
    spont = r[labels == -1].mean(0)
    tun = tuning_stats(R)
    return {'R': R.cpu().numpy(), 'spont': spont.cpu().numpy(),
            **{k: v.cpu().numpy() for k, v in tun.items()}}
