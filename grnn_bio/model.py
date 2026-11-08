"""Frozen Allen GFR cell bank. Units: pA, ms, and Hz.

Matches AllenInstitute/GRNN commit 835a535f6831b88339c6a5c17180fdbc504f7448.
Saved a and beta already include the discrete-time scaling: DO NOT multiply
by dt again. beta multiplies rates in Hz. Negative fitted taps are retained.
"""
from __future__ import annotations
from dataclasses import dataclass, fields
import hashlib
import numpy as np


@dataclass(frozen=True)
class GFRParameters:
    a: np.ndarray
    beta: np.ndarray
    ds: np.ndarray
    max_current: np.ndarray
    gamma_hz: np.ndarray
    poly_coeff: np.ndarray
    threshold: np.ndarray
    cell_ids: np.ndarray
    dt_ms: int = 20

    def __post_init__(self):
        if (isinstance(self.dt_ms, (bool, np.bool_))
                or not isinstance(self.dt_ms, (int, np.integer)) or self.dt_ms <= 0):
            raise ValueError('dt_ms must be a finite positive integer')
        object.__setattr__(self, 'dt_ms', int(self.dt_ms))
        for f in fields(self):
            if f.name == 'dt_ms':
                continue
            dtype = np.int64 if f.name == 'cell_ids' else np.float64
            arr = np.array(getattr(self, f.name), dtype=dtype, copy=True)
            if not np.all(np.isfinite(arr)):
                raise ValueError(f'Nonfinite parameter {f.name}')
            arr.setflags(write=False)
            object.__setattr__(self, f.name, arr)
        if self.a.ndim != 2 or min(self.a.shape) < 1:
            raise ValueError('a must have nonempty [cells, filters] axes')
        n, k = self.a.shape
        if self.beta.shape != (n, k) or self.ds.shape != (n, k):
            raise ValueError('a, beta, ds must have the same [cells, filters] shape')
        for name in ['max_current', 'gamma_hz', 'threshold', 'cell_ids']:
            if getattr(self, name).shape != (n,):
                raise ValueError(f'{name} must have shape [cells]')
        if (self.poly_coeff.ndim != 2 or self.poly_coeff.shape[0] != n
                or self.poly_coeff.shape[1] < 1):
            raise ValueError('poly_coeff must have nonempty shape [cells, degree + 1]')
        if np.any(self.max_current <= 0) or np.any(self.gamma_hz <= 0):
            raise ValueError('Current scales and rate ceilings must be positive')
        if np.any(self.ds <= 0) or np.any(self.ds > 1):
            raise ValueError('Discrete decay coefficients must lie in (0,1]')

    @property
    def n(self):
        return self.a.shape[0]

    @property
    def k(self):
        return self.a.shape[1]

    def fingerprint(self):
        h = hashlib.sha256()
        for f in fields(self):
            v = getattr(self, f.name)
            h.update(f.name.encode())
            if isinstance(v, np.ndarray):
                h.update(str(v.shape).encode())
                h.update(v.tobytes())
            else:
                h.update(str(v).encode())
        return h.hexdigest()


def activation_and_grad(z, params: GFRParameters):
    """Return firing rate (Hz) and derivative (Hz/pA); ReLU derivative at 0 is 0."""
    z = np.asarray(z, dtype=np.float64)
    x = (z - params.threshold) / params.max_current
    # Simultaneous Horner evaluation handles constant and arbitrary-degree polynomials.
    coeff = params.poly_coeff ** 2
    poly = np.broadcast_to(coeff[:, -1], x.shape).copy()
    deriv = np.zeros_like(x)
    for d in range(coeff.shape[1] - 2, -1, -1):
        deriv = deriv * x + poly
        poly = poly * x + coeff[:, d]
    tanh = np.tanh(poly)
    rates = params.gamma_hz * np.maximum(tanh, 0.)
    dr = params.gamma_hz * (1. - tanh * tanh) * deriv / params.max_current
    dr = np.where(poly > 0., dr, 0.)
    return rates, dr


def cell_step(params, h_prev, rates_prev_hz, current_pa):
    h = (1. - params.ds) * h_prev + params.a * np.asarray(current_pa)[..., None]
    h = h + params.beta * np.asarray(rates_prev_hz)[..., None]
    rates, dr = activation_and_grad(h.mean(axis=-1), params)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(rates)):
        raise FloatingPointError('GFR dynamics became nonfinite')
    return h, rates, dr


def isolated_response(params, currents_pa):
    """Run [batch,time,cells] current-clamp probes from exact zero state."""
    currents = np.asarray(currents_pa, dtype=float)
    if currents.ndim != 3 or currents.shape[2] != params.n:
        raise ValueError('currents must be [batch, time, cells]')
    h = np.zeros((currents.shape[0], params.n, params.k))
    r = np.zeros((currents.shape[0], params.n))
    rates = np.empty_like(currents)
    for t in range(currents.shape[1]):
        h, r, _ = cell_step(params, h, r, currents[:, t])
        rates[:, t] = r
    return rates


def calibrate_current(params, baseline_fraction=.05, input_scale_fraction=.25,
                      n_grid=33, probe_steps=150):
    """Label-free operating-current calibration; never alters a fitted coefficient.

    Sweep currents from 0 to each fit's max_current. Choose the discrete grid
    point closest to baseline_fraction*gamma, using the last third of a 3 s
    isolated probe. This is a model operating-point choice, not new patch-clamp
    validation. If target is unreachable, retain and report the residual.
    """
    if (not np.isfinite(baseline_fraction) or not np.isfinite(input_scale_fraction)
            or not (0 <= baseline_fraction < 1) or input_scale_fraction <= 0):
        raise ValueError('Invalid calibration fractions')
    for name, value in [('n_grid', n_grid), ('probe_steps', probe_steps)]:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 3):
            raise ValueError(f'Calibration {name} must be an integer of at least 3')
    grid = np.linspace(0., 1., n_grid)[:, None] * params.max_current
    current = np.broadcast_to(grid[:, None, :], (n_grid, probe_steps, params.n))
    rates = isolated_response(params, current)
    steady = rates[:, probe_steps * 2 // 3:, :].mean(axis=1)
    target = baseline_fraction * params.gamma_hz
    chosen = np.argmin(np.abs(steady - target), axis=0)
    offset = grid[chosen, np.arange(params.n)]
    actual = steady[chosen, np.arange(params.n)]
    info = {
        'target_rate_hz': target.tolist(), 'achieved_rate_hz': actual.tolist(),
        'absolute_error_hz': np.abs(actual - target).tolist(),
        'at_grid_boundary': ((chosen == 0) | (chosen == n_grid - 1)).tolist(),
        'probe_steps': probe_steps, 'probe_ms': int(params.dt_ms * probe_steps),
        'grid_points': n_grid, 'input_scale_fraction': input_scale_fraction,
        'uses_task_labels': False,
    }
    return offset, input_scale_fraction * params.max_current, info
