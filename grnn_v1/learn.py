"""Objectives, the two credit-assignment rules, and Exponentiated Adam.

Both learners optimise the same objective with the same optimizer. They differ
only in how the gradient for recurrent magnitudes is obtained:
  bptt   : exact gradient through all network paths and the true readout;
  local  : e-prop. dL/d|W_ij| ~ sum_t L_i,t dr_i,t/d|W_ij| following cell i's own
           filters and spike-history loop only; L = direct dLoss/dr_i (local terms)
           + fixed random feedback B_i * dLoss/dy (task error).
  local_nobeta : as local, without the spike-history (beta) term in the trace.
"""
from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch

from .core import TYPES, DT_MS, Circuit, LGN, grating_movie

DIRECTIONS = np.arange(0, 360, 45.0)


# ------------------------------------------------------------------ in vivo targets
def invivo_targets() -> Dict[str, Dict[str, np.ndarray]]:
    """Neuropixels V1 statistics (Ito et al. 2026 curated set, Allen Visual Coding).

    Pyr <- L2/3 Exc, PV <- L2/3 PV (V1 only); SST, VIP pooled over layers (and,
    as in the source, over visual areas). OSI/DSI only for units with preferred
    rate >= 0.5 Hz, as in the source.
    """
    d = pd.read_csv('C:/pianshi/invivo/ito_osi_dsi_neuropixels_v4.csv', sep=' ')
    sel = {'Pyr': d.cell_type == 'L2/3 Exc', 'PV': d.cell_type == 'L2/3 PV',
           'SST': d.cell_type.str.contains('SST'), 'VIP': d.cell_type.str.contains('VIP')}
    out = {}
    for t, m in sel.items():
        x = d[m]
        resp = x['max_mean_rate(Hz)'] >= 0.5
        fin = lambda v: v[np.isfinite(v)]
        out[t] = {'spont': fin(x['Spont_Rate(Hz)'].to_numpy(float)), 'pref': fin(x['max_mean_rate(Hz)'].to_numpy(float)),
                  'osi': fin(x.loc[resp, 'OSI'].to_numpy(float)), 'dsi': fin(x.loc[resp, 'DSI'].to_numpy(float))}
    return out


def tuning_stats(R: torch.Tensor) -> Dict[str, torch.Tensor]:
    """R [N, 8] mean rate per direction -> pref rate, OSI, DSI (Ito et al. Fig. 2 form)."""
    pref_idx = R.argmax(1)
    th = torch.tensor(np.radians(DIRECTIONS), dtype=R.dtype, device=R.device)
    dth = th[None, :] - th[pref_idx][:, None]
    tot = R.sum(1).clamp_min(1e-6)
    return {'pref': R.max(1).values, 'osi': (R * torch.cos(2 * dth)).sum(1) / tot,
            'dsi': (R * torch.cos(dth)).sum(1) / tot}


def quantile_loss(model: torch.Tensor, target: np.ndarray) -> torch.Tensor:
    """Squared 1-D Wasserstein-2 between a model sample and a fixed target sample."""
    if model.numel() < 2 or len(target) < 2:
        return model.sum() * 0
    q = (torch.arange(model.numel(), device=model.device, dtype=model.dtype) + 0.5) / model.numel()
    tq = torch.tensor(np.quantile(target, q.cpu().numpy()), device=model.device, dtype=model.dtype)
    return ((torch.sort(model).values - tq) ** 2).mean()


# ------------------------------------------------------------------ objectives
class DataObjective:
    """Match in vivo spontaneous/preferred rates, OSI and DSI per cell type (Ito-style).

    One batch: n_gray gray trials and 8 directions x n_rep drifting-grating trials
    (0.04 cpd, 2 Hz, 80% contrast), each 10 warm-up bins + 25 scored bins (500 ms).
    """
    name = 'data'

    def __init__(self, circuit: Circuit, lgn: LGN, device, n_gray=4, n_rep=2, warm=10, bins=25):
        self.c, self.lgn, self.device = circuit, lgn, device
        self.n_gray, self.n_rep, self.warm, self.bins = n_gray, n_rep, warm, bins
        self.targets = invivo_targets()
        self.type_idx = {t: torch.tensor(np.flatnonzero(circuit.types == t), device=device) for t in TYPES}

    def batch(self, rng: np.random.Generator):
        T = self.warm + self.bins
        movies = [np.zeros((T, 48, 64), np.float32) for _ in range(self.n_gray)]
        for d in DIRECTIONS:
            for _ in range(self.n_rep):
                m = np.zeros((T, 48, 64), np.float32)
                m[self.warm:] = grating_movie(d, self.bins, phase=rng.uniform(0, 2 * np.pi))
                movies.append(m)
        x = torch.tensor(np.stack(movies), device=self.device)
        with torch.no_grad():
            lgn = self.lgn(x)
        return {'lgn': lgn}

    def stats(self, rates: torch.Tensor):
        r = rates[:, self.warm:].mean(1)                     # [B, N]
        spont = r[:self.n_gray].mean(0)
        R = r[self.n_gray:].reshape(len(DIRECTIONS), self.n_rep, -1).mean(1).T   # [N, 8]
        return spont, tuning_stats(R)

    def loss(self, rates: torch.Tensor, y: torch.Tensor, batch) -> Dict[str, torch.Tensor]:
        spont, tun = self.stats(rates)
        terms = []
        for t in TYPES:
            idx, tg = self.type_idx[t], self.targets[t]
            terms.append(quantile_loss(torch.log1p(spont[idx]), np.log1p(tg['spont'])))
            terms.append(quantile_loss(torch.log1p(tun['pref'][idx]), np.log1p(tg['pref'])))
            resp = tun['pref'][idx] >= 0.5
            terms.append(quantile_loss(tun['osi'][idx][resp], tg['osi']))
            terms.append(quantile_loss(tun['dsi'][idx][resp], tg['dsi']))
        direct = torch.stack(terms).sum()
        return {'direct': direct, 'task': y.sum() * 0, 'total': direct}

    def metric(self, rates, y, batch) -> float:
        return float(self.loss(rates, y, batch)['total'])


class ChangeObjective:
    """Go/No-Go image change detection (Allen Visual Behavior structure).

    gray 100 ms, image A 240 ms, gray 500 ms, image B 240 ms, gray 260 ms. Target 1
    if B != A. Lick logit is averaged 100-500 ms after B onset. Images are natural
    movie frames; the 118 natural scenes used for evaluation are never trained on.
    A weak homeostatic term keeps each cell's mean rate near its calibrated baseline.
    """
    name = 'change'

    def __init__(self, circuit: Circuit, lgn: LGN, device, frames: np.ndarray, batch_size=32,
                 homeo=1e-3):
        self.c, self.lgn, self.device, self.frames = circuit, lgn, device, frames
        self.bs, self.homeo = batch_size, homeo
        self.on_a, self.len_img, self.gap = 5, 12, 25
        self.on_b = self.on_a + self.len_img + self.gap
        self.T = self.on_b + self.len_img + 13
        self.win = slice(self.on_b + 5, self.on_b + 25)
        self.base = torch.tensor(np.asarray(circuit.calibration['achieved_rate_hz']), device=device, dtype=torch.float32)

    def batch(self, rng: np.random.Generator, frame_ids: Optional[np.ndarray] = None):
        ids = np.arange(len(self.frames)) if frame_ids is None else frame_ids
        a = rng.choice(ids, self.bs)
        change = np.arange(self.bs) % 2 == 1
        rng.shuffle(change)
        b = a.copy()
        for k in np.flatnonzero(change):
            while b[k] == a[k]:
                b[k] = rng.choice(ids)
        x = np.zeros((self.bs, self.T, 48, 64), np.float32)
        x[:, self.on_a:self.on_a + self.len_img] = self.frames[a][:, None]
        x[:, self.on_b:self.on_b + self.len_img] = self.frames[b][:, None]
        with torch.no_grad():
            lgn = self.lgn(torch.tensor(x, device=self.device))
        return {'lgn': lgn, 'target': torch.tensor(change.astype(np.float32), device=self.device)}

    def loss(self, rates, y, batch):
        logit = y[:, self.win, 0].mean(1)
        task = torch.nn.functional.binary_cross_entropy_with_logits(logit, batch['target'])
        direct = self.homeo * ((rates.mean((0, 1)) - self.base) ** 2).mean()
        return {'direct': direct, 'task': task, 'total': task + direct}

    def metric(self, rates, y, batch) -> float:
        logit = y[:, self.win, 0].mean(1)
        return float(((logit > 0).float() == batch['target']).float().mean())


# ------------------------------------------------------------------ gradients
def compute_gradients(circuit: Circuit, objective, batch, rule: str, noise_seed: int,
                      feedback: str = 'random'):
    """Fill .grad of circuit.mag, w_out, b_out according to the learning rule."""
    params = [circuit.mag, circuit.w_out, circuit.b_out]
    for p in params:
        p.grad = None
    mode = {'bptt': 'bptt', 'local': 'local', 'local_nobeta': 'local_nobeta', 'readout': 'local'}[rule]
    rates, y = circuit.run(batch['lgn'], mode=mode, noise_seed=noise_seed)
    L = objective.loss(rates, y, batch)
    if rule == 'bptt':
        L['total'].backward()
        return L, rates, y
    # readout: exact gradient of the task term (local to the readout synapses)
    if L['task'].requires_grad:
        g_out = torch.autograd.grad(L['task'], [circuit.w_out, circuit.b_out], retain_graph=True, allow_unused=True)
        circuit.w_out.grad = g_out[0] if g_out[0] is not None else torch.zeros_like(circuit.w_out)
        circuit.b_out.grad = g_out[1] if g_out[1] is not None else torch.zeros_like(circuit.b_out)
    if rule == 'readout':
        circuit.mag.grad = torch.zeros_like(circuit.mag)
        return L, rates, y
    # learning signal per cell and time bin
    sig = torch.zeros_like(rates)
    if L['direct'].requires_grad:
        sig = sig + torch.autograd.grad(L['direct'], rates, retain_graph=True)[0]
    if L['task'].requires_grad:
        gy = torch.autograd.grad(L['task'], y, retain_graph=True)[0]          # [B, T, n_out]
        B = circuit.feedback if feedback == 'random' else (circuit.w_out * circuit.readout_mask).detach() / 1.0
        sig = sig + gy @ B.T / 100.0                                          # rate units (R0)
    surrogate = (sig.detach() * rates).sum()
    circuit.mag.grad = torch.autograd.grad(surrogate, circuit.mag)[0]
    return L, rates, y


class ExpAdam:
    """Exponentiated Adam (Ito et al. 2026): |W| <- |W| exp(-lr m_hat / (sqrt(v_hat) + eps)).

    Sign-preserving and multiplicative for the recurrent magnitudes; plain Adam for
    the readout. Identical for every learning rule.
    """

    def __init__(self, circuit: Circuit, lr_rec=5e-3, lr_out=1e-2, b1=0.9, b2=0.999, eps=1e-11):
        self.c, self.lr_rec, self.lr_out, self.b1, self.b2, self.eps = circuit, lr_rec, lr_out, b1, b2, eps
        self.state = {}
        self.t = 0

    @torch.no_grad()
    def step(self):
        self.t += 1
        for name, p, lr, mult in [('mag', self.c.mag, self.lr_rec, True), ('w_out', self.c.w_out, self.lr_out, False),
                                  ('b_out', self.c.b_out, self.lr_out, False)]:
            if p.grad is None:
                continue
            g = p.grad * (self.c.mask if name == 'mag' else 1.0)
            m, v = self.state.get(name, (torch.zeros_like(p), torch.zeros_like(p)))
            m = self.b1 * m + (1 - self.b1) * g
            v = self.b2 * v + (1 - self.b2) * g * g
            self.state[name] = (m, v)
            u = (m / (1 - self.b1 ** self.t)) / (torch.sqrt(v / (1 - self.b2 ** self.t)) + self.eps)
            if mult:
                p.mul_(torch.exp(-lr * u.clamp(-5, 5)))
            else:
                p.sub_(lr * u)
