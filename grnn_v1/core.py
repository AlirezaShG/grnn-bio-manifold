"""Stimuli, LGN front end and the GFR V1 microcircuit (PyTorch, GPU).

Units follow the GFR bank: current in pA, rates in Hz, dt = 20 ms. The intrinsic
parameters of every cell are frozen. Only recurrent magnitudes |W| (sign fixed by
presynaptic type, Dale's law) and the task readout are plastic.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
TYPES = ('Pyr', 'PV', 'SST', 'VIP')
DT_MS = 20.0
R0 = 100.0  # Hz, rate normalisation for synaptic drive

# ---------------------------------------------------------------- stimuli
GRID_H, GRID_W, DEG_PER_PX = 48, 64, 2.0   # 96 x 128 deg visual field


def images_to_grid(images: np.ndarray) -> np.ndarray:
    """uint8 [n, H, W] -> contrast in [-1, 1] on the model grid (area-averaged)."""
    from PIL import Image
    out = np.empty((len(images), GRID_H, GRID_W), np.float32)
    for i, im in enumerate(images):
        small = Image.fromarray(im).resize((GRID_W, GRID_H), Image.BOX)
        out[i] = np.asarray(small, np.float32) / 127.5 - 1.0
    return out


def load_natural_scenes() -> np.ndarray:
    from PIL import Image
    d = Path('C:/pianshi/invivo/natural_scenes')
    ims = np.stack([np.asarray(Image.open(d / f'natural_scene_{i}.tiff')) for i in range(118)])
    return images_to_grid(ims)


def load_movie_frames(step: int = 5) -> np.ndarray:
    # Allen ships this template as a NumPy array despite the .h5 suffix: [frames, H, W] uint8.
    frames = np.load('C:/pianshi/invivo/movies/natural_movie_1.h5', mmap_mode='r')[::step]
    return images_to_grid(np.asarray(frames, np.uint8))


def grating_movie(direction_deg: float, n_bins: int, sf_cpd: float = 0.04, tf_hz: float = 2.0,
                  contrast: float = 0.8, phase: float = 0.0) -> np.ndarray:
    y, x = np.mgrid[0:GRID_H, 0:GRID_W] * DEG_PER_PX
    th = math.radians(direction_deg)
    proj = x * math.cos(th) + y * math.sin(th)
    t = np.arange(n_bins)[:, None, None] * DT_MS / 1000.0
    return (contrast * np.sin(2 * np.pi * (sf_cpd * proj[None] - tf_hz * t) + phase)).astype(np.float32)


# ---------------------------------------------------------------- LGN
class LGN:
    """Linear-nonlinear LGN: ON/OFF centre-surround x sustained/transient channels.

    Spatial DoG (centre 2 deg, surround 6 deg) on a stride-2 grid; temporal
    low-pass (sustained, tau 40 ms) or band-pass (tau 20 - tau 80 ms, transient);
    rectified with a spontaneous baseline. Output is [B, T, n_lgn].
    """

    def __init__(self, device, stride: int = 2):
        self.device = device
        self.stride = stride
        r = 9
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float32)
        def g(s):
            k = np.exp(-(xx ** 2 + yy ** 2) / (2 * s ** 2))
            return k / k.sum()
        dog = g(1.0) - 0.85 * g(3.0)
        self.kernel = torch.tensor(dog, device=device)[None, None]
        self.h = len(range(0, GRID_H, stride))
        self.w = len(range(0, GRID_W, stride))
        self.n = 4 * self.h * self.w
        self.pos = np.stack(np.meshgrid(np.arange(0, GRID_W, stride), np.arange(0, GRID_H, stride)), -1).reshape(-1, 2)

    def __call__(self, movie: torch.Tensor) -> torch.Tensor:
        b, t, hh, ww = movie.shape
        x = torch.nn.functional.conv2d(movie.reshape(b * t, 1, hh, ww), self.kernel, padding=self.kernel.shape[-1] // 2)
        x = x[:, 0, ::self.stride, ::self.stride].reshape(b, t, -1)
        a40, a20, a80 = math.exp(-DT_MS / 40), math.exp(-DT_MS / 20), math.exp(-DT_MS / 80)
        s = torch.zeros_like(x[:, 0]); f = torch.zeros_like(s); sl = torch.zeros_like(s)
        sus, tra = [], []
        for k in range(t):
            s = a40 * s + (1 - a40) * x[:, k]
            f = a20 * f + (1 - a20) * x[:, k]
            sl = a80 * sl + (1 - a80) * x[:, k]
            sus.append(s); tra.append(f - sl)
        sus = torch.stack(sus, 1); tra = torch.stack(tra, 1)
        gain = 8.0
        out = torch.cat([torch.relu(0.1 + gain * sus), torch.relu(0.1 + gain * -sus),
                         torch.relu(0.1 + 2 * gain * tra), torch.relu(0.1 + 2 * gain * -tra)], dim=-1)
        return out  # [B, T, 4 * h * w]: sON, sOFF, tON, tOFF


# ---------------------------------------------------------------- cells
@dataclass
class CellBank:
    a: np.ndarray; beta: np.ndarray; ds: np.ndarray; max_current: np.ndarray
    gamma_hz: np.ndarray; poly: np.ndarray; threshold: np.ndarray
    cell_ids: np.ndarray; types: np.ndarray


def sample_bank(cohort, counts: Dict[str, int], rng: np.random.Generator) -> CellBank:
    """Sample fitted cells per type WITH replacement (a 1,000-cell circuit from ~300 fits)."""
    from grnn_bio.data import parameters_from_rows
    rows = []
    for t in TYPES:
        pool = cohort[cohort.cell_type == t]
        rows.append(pool.iloc[rng.choice(len(pool), counts[t], replace=True)])
    import pandas as pd
    rows = pd.concat(rows, ignore_index=True)
    p = parameters_from_rows(rows)
    return CellBank(p.a, p.beta, p.ds, p.max_current, p.gamma_hz, p.poly_coeff, p.threshold,
                    p.cell_ids, rows.cell_type.to_numpy())


# ---------------------------------------------------------------- connectivity
def v1dd_pmax() -> np.ndarray:
    """Peak connection probability P[post, pre] for L2/3 Exc, PV, SST, VIP (Allen V1DD)."""
    import pandas as pd
    d = pd.read_csv('C:/pianshi/invivo/pmax_matrix_v1dd.csv', sep=' ', index_col=0)
    names = ['L2/3_Exc', 'L2/3_PV', 'L2/3_SST', 'L2/3_VIP']
    pre_post = d.loc[names, names].to_numpy(float)   # rows pre, cols post
    return pre_post.T                                   # -> [post, pre]


def psp_medians() -> np.ndarray:
    """Relative median 90th-percentile PSP (mV) [post, pre] from Allen synaptic physiology."""
    import pandas as pd
    d = pd.read_csv('C:/pianshi/invivo/psp_characterization.csv', sep=' ')
    key = {'Pyr': 'l23_pyr', 'PV': 'pvalb', 'SST': 'sst', 'VIP': 'vip'}
    m = np.full((4, 4), np.nan)
    for i, post in enumerate(TYPES):
        for j, pre in enumerate(TYPES):
            r = d[(d.pre == key[pre]) & (d.post == key[post])]
            if len(r) and str(r.median_90th.iloc[0]) not in ('[]', 'nan'):
                m[i, j] = float(r.median_90th.iloc[0])
    m[np.isnan(m)] = np.nanmedian(m)
    return m


@dataclass
class CircuitConfig:
    counts: Dict[str, int] = field(default_factory=lambda: {'Pyr': 800, 'PV': 100, 'SST': 60, 'VIP': 40})
    rec_gain: float = 1.0          # total recurrent drive scale (in units of S_i)
    inh_gain: float = 1.0          # extra factor on inhibitory sources
    lognormal_sigma: float = 0.8   # heavy-tailed magnitudes
    like_to_like_beta: float = 0.5 # Ito et al. E->E orientation modulation
    lgn_gain: float = 1.0          # feedforward drive scale
    lgn_type_gain: Dict[str, float] = field(default_factory=lambda: {'Pyr': 1.0, 'PV': 0.35, 'SST': 0.25, 'VIP': 0.25})
    noise_std: float = 0.15        # background current noise (drive units)
    baseline_fraction: float = 0.02
    input_scale_fraction: float = 0.25
    n_out: int = 1


class Circuit:
    """GFR V1 microcircuit. Construction is fully determined by `seed`."""

    def __init__(self, cohort, cfg: CircuitConfig, seed: int, device):
        from grnn_bio.model import GFRParameters, calibrate_current
        self.cfg, self.seed, self.device = cfg, seed, device
        rng = np.random.default_rng(seed)
        bank = sample_bank(cohort, cfg.counts, rng)
        self.types = bank.types
        self.N = len(self.types)
        tid = np.array([TYPES.index(t) for t in self.types])
        self.type_id = tid
        params = GFRParameters(bank.a, bank.beta, bank.ds, bank.max_current, bank.gamma_hz, bank.poly,
                               bank.threshold, bank.cell_ids)
        offset, scale, info = calibrate_current(params, cfg.baseline_fraction, cfg.input_scale_fraction)
        self.calibration = info
        T = lambda x: torch.tensor(np.asarray(x), dtype=torch.float32, device=device)
        self.a, self.beta = T(bank.a), T(bank.beta)
        self.decay = T(1.0 - bank.ds[0])          # shared filter grid
        self.max_current, self.gamma = T(bank.max_current), T(bank.gamma_hz)
        self.poly2, self.threshold = T(bank.poly ** 2), T(bank.threshold)
        self.offset, self.scale = T(offset), T(scale)
        self.K = bank.a.shape[1]
        # orientation preference, RF position (deg) and LGN weights
        self.theta = rng.uniform(0, 180, self.N)
        self.rf_xy = np.stack([rng.uniform(28, 100, self.N), rng.uniform(20, 76, self.N)], 1)
        # recurrent mask: V1DD p_max by type, like-to-like for E->E (Ito et al. eq. 2)
        P = v1dd_pmax()[tid[:, None], tid[None, :]]
        b = cfg.like_to_like_beta
        b1, b2 = 2 / (1 + b), b * 2 / (1 + b)
        dphi = np.abs(self.theta[:, None] - self.theta[None, :]); dphi = np.minimum(dphi, 180 - dphi)
        ee = (tid[:, None] == 0) & (tid[None, :] == 0)
        P = np.where(ee, P * (b1 + (b2 - b1) / 90 * dphi), P)
        mask = rng.random((self.N, self.N)) < np.clip(P, 0, 1)
        np.fill_diagonal(mask, False)
        rel = psp_medians()[tid[:, None], tid[None, :]]
        mag = rel * rng.lognormal(-0.5 * cfg.lognormal_sigma ** 2, cfg.lognormal_sigma, (self.N, self.N))
        sign = np.where(tid == 0, 1.0, -1.0)
        mag = mag * np.where(sign[None, :] < 0, cfg.inh_gain, 1.0)
        # normalise expected recurrent input per postsynaptic cell type
        indeg = np.maximum(mask.sum(1, keepdims=True), 1)
        mag = mag * cfg.rec_gain * 30.0 / indeg
        self.mask = T(mask)
        self.sign = T(sign)
        self.mag = T(np.where(mask, mag, 0.0)).requires_grad_(True)  # positive magnitudes |W|
        self.w_out = torch.zeros(self.N, cfg.n_out, device=device, requires_grad=True)
        self.b_out = torch.zeros(cfg.n_out, device=device, requires_grad=True)
        self.readout_mask = T((tid == 0).astype(np.float32))[:, None]
        g = torch.Generator(device='cpu').manual_seed(int(seed) + 7)
        self.feedback = (torch.randn(self.N, cfg.n_out, generator=g) / math.sqrt(self.N)).to(device)
        self.w_in = None

    # ------------------------------------------------------------ input weights
    def build_lgn_weights(self, lgn: LGN):
        """Fixed LGN->V1 weights: Gabor subfields (Pyr), broad/less-selective (interneurons).

        ON subfield <- ON channels, OFF subfield <- OFF channels; each subfield draws on
        sustained or transient channels at random, which yields direction preference
        (spatially offset sustained/transient inputs).
        """
        rng = np.random.default_rng(self.seed + 11)
        pos = lgn.pos.astype(np.float32) * DEG_PER_PX          # [n_pos, 2] deg
        n_pos = len(pos)
        W = np.zeros((self.N, 4 * n_pos), np.float32)
        for i in range(self.N):
            dx, dy = pos[:, 0] - self.rf_xy[i, 0], pos[:, 1] - self.rf_xy[i, 1]
            th = math.radians(self.theta[i])
            u = dx * math.cos(th) + dy * math.sin(th)
            v = -dx * math.sin(th) + dy * math.cos(th)
            pyr = self.type_id[i] == 0
            sig = rng.uniform(6, 10) if pyr else rng.uniform(10, 16)
            env = np.exp(-(u ** 2 + (v / 1.5) ** 2) / (2 * sig ** 2))
            gab = env * np.cos(2 * np.pi * 0.04 * u + rng.uniform(0, 2 * np.pi))
            if not pyr:   # interneurons: mostly untuned Gaussian drive plus a weak Gabor
                gab = 0.35 * gab + 0.65 * env * rng.choice([-1, 1])
            on, off = np.maximum(gab, 0), np.maximum(-gab, 0)
            s_on = rng.random() < 0.5; s_off = rng.random() < 0.5
            W[i, (0 if s_on else 2) * n_pos:(1 if s_on else 3) * n_pos] += on
            W[i, (1 if s_off else 3) * n_pos:(2 if s_off else 4) * n_pos] += off
            W[i] /= max(W[i].sum(), 1e-6)
            W[i] *= self.cfg.lgn_type_gain[TYPES[self.type_id[i]]]
        self.w_in = torch.tensor(W * self.cfg.lgn_gain * 6.0, device=self.device)

    # ------------------------------------------------------------ dynamics
    def W(self) -> torch.Tensor:
        return self.mag * self.mask * self.sign[None, :]

    def activation(self, hbar: torch.Tensor) -> torch.Tensor:
        x = (hbar - self.threshold) / self.max_current
        c = self.poly2
        poly = c[:, -1].expand_as(x)
        for d in range(c.shape[1] - 2, -1, -1):
            poly = poly * x + c[:, d]
        return self.gamma * torch.relu(torch.tanh(poly))

    def run(self, lgn_rates: torch.Tensor, mode: str = 'bptt', noise_seed: Optional[int] = None,
            W: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simulate. lgn_rates [B, T, n_lgn]. Returns rates [B, T, N] (Hz) and readout [B, T, n_out].

        mode='bptt'  : full graph (exact gradient).
        mode='local' : synaptic input from other cells is detached, so autograd follows only
                       each cell's own filters and spike-history loop (e-prop eligibility).
        mode='local_nobeta': also detaches the cell's own spike-history (beta) path.
        Forward values are identical in every mode.
        """
        B, T, _ = lgn_rates.shape
        W = self.W() if W is None else W
        ff = lgn_rates @ self.w_in.T
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(noise_seed if noise_seed is not None else 0))
        noise = torch.randn(B, T, self.N, generator=gen, device=self.device) * self.cfg.noise_std
        h = torch.zeros(B, self.N, self.K, device=self.device)
        r = torch.zeros(B, self.N, device=self.device)
        rates = []
        for t in range(T):
            r_syn = r.detach() if mode != 'bptt' else r
            r_own = r.detach() if mode == 'local_nobeta' else r
            drive = ff[:, t] - self.ff_baseline + (r_syn / R0) @ W.T + noise[:, t]
            current = self.offset + self.scale * drive
            h = self.decay * h + self.a * current[:, :, None] + self.beta * r_own[:, :, None]
            r = self.activation(h.mean(-1))
            rates.append(r)
        rates = torch.stack(rates, 1)
        y = (rates / R0) @ (self.w_out * self.readout_mask) + self.b_out
        return rates, y

    def set_ff_baseline(self, lgn_gray: torch.Tensor):
        """Feedforward drive on a gray screen is subtracted, so gray -> calibrated baseline."""
        self.ff_baseline = (lgn_gray @ self.w_in.T).mean(dim=(0, 1))
