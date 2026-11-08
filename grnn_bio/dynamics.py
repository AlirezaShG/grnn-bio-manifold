"""Fixed-input state diagnostics for the unmodified GFR recurrent map.

The Markov state is the complete [cells, filters] hidden state, not firing
rates alone. These functions apply to an autonomous, deterministic circuit
under a declared constant input, after rates are consistent with the state.
They do not apply to rate clamps, injected noise, or changing task inputs.
An eigenvalue calculation at a transient point is not an attractor test.
"""
from __future__ import annotations

import numpy as np
from .model import activation_and_grad, cell_step


def _state_and_input(circuit, state, input_vector):
    h = np.asarray(state, dtype=float)
    x = np.zeros(circuit.in_dim) if input_vector is None else np.asarray(input_vector, dtype=float)
    if h.shape != (circuit.N, circuit.K) or not np.all(np.isfinite(h)):
        raise ValueError('state must be finite [cells, filters]')
    if x.shape != (circuit.in_dim,) or not np.all(np.isfinite(x)):
        raise ValueError('input_vector must be finite [input channels]')
    return h, x


def _positive_integer(value, name):
    if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
            or value < 1):
        raise ValueError(f'{name} must be a positive integer')
    return int(value)


def _guard_state_dimension(dimension, max_state_dim):
    limit = _positive_integer(max_state_dim, 'max_state_dim')
    if dimension > limit:
        raise ValueError(f'Full Jacobian dimension {dimension} exceeds max_state_dim={limit}')


def constant_input_step(circuit, state, input_vector=None):
    """One deterministic step with r=g(mean(h)) consistent with its state.

    Circuit.initial_state intentionally initializes rates to zero even if
    g(0)>0. This autonomous map concerns subsequent consistent states and
    should not be used to redefine that documented initialization.
    """
    h, x = _state_and_input(circuit, state, input_vector)
    rates, _ = activation_and_grad(h.mean(axis=-1), circuit.params)
    drive = circuit.input_weights @ x + circuit.W_rec @ (rates / circuit.rate_scale_hz)
    current = circuit.current_offset + circuit.current_scale * drive
    next_h, _, _ = cell_step(circuit.params, h, rates, current)
    return next_h


def fixed_point_residual(circuit, state, input_vector=None):
    """Return F(h; x)-h in state-current units without fitting or projecting."""
    h, x = _state_and_input(circuit, state, input_vector)
    return constant_input_step(circuit, h, x) - h


def full_state_jacobian(circuit, state, input_vector=None, *, max_state_dim=1024):
    """Dense full NK-by-NK Jacobian, including recurrent and intrinsic history.

    Flattening order is (cell, filter), with the filter index changing fastest.
    A resource guard prevents accidental large dense allocations. The matrix
    differentiates the full circuit map, not only cell-local eligibility.
    """
    h, _ = _state_and_input(circuit, state, input_vector)
    dimension = circuit.N * circuit.K
    _guard_state_dimension(dimension, max_state_dim)
    _, dr = activation_and_grad(h.mean(axis=-1), circuit.params)
    # dI_i / dh_j,l = scale_i W_ij gprime_j / (K reference_rate_j).
    coupling = (circuit.current_scale[:, None] * circuit.W_rec
                * (dr / circuit.rate_scale_hz)[None, :] / circuit.K)
    jac = np.repeat(np.repeat(coupling, circuit.K, axis=0), circuit.K, axis=1)
    jac *= circuit.a.reshape(dimension, 1)
    diag = np.arange(dimension)
    jac[diag, diag] += (1. - circuit.ds).reshape(dimension)
    for i in range(circuit.N):
        sl = slice(i * circuit.K, (i + 1) * circuit.K)
        jac[sl, sl] += circuit.beta[i, :, None] * dr[i] / circuit.K
    return jac


def fixed_point_diagnostics(circuit, state, input_vector=None, *, atol=1e-7,
                            rtol=1e-7, kink_tolerance=1e-10,
                            max_state_dim=1024):
    """Residual and full-state spectral radius at a candidate fixed point.

    `linearly_stable_fixed_point` is None when the residual is too large or
    the activation is at a ReLU kink. True is only a local, fixed-input
    linearization statement; it establishes neither global attraction nor
    a task-relevant memory attractor. A radius near one is marginal evidence.
    """
    h, x = _state_and_input(circuit, state, input_vector)
    if not all(np.isfinite(v) and v >= 0 for v in (atol, rtol, kink_tolerance)):
        raise ValueError('Diagnostic tolerances must be finite and nonnegative')
    residual = fixed_point_residual(circuit, h, x)
    residual_inf = float(np.max(np.abs(residual)))
    scale = max(1., float(np.max(np.abs(h))))
    threshold = float(atol + rtol * scale)
    stationary = bool(residual_inf <= threshold)
    normalized = (h.mean(axis=-1) - circuit.params.threshold) / circuit.params.max_current
    # Independent direct powers are useful here to locate the nonsmooth kink.
    powers = np.arange(circuit.params.poly_coeff.shape[1])
    polynomial = np.sum(circuit.params.poly_coeff ** 2 * normalized[:, None] ** powers, axis=1)
    kink_margin = float(np.min(np.abs(polynomial)))
    differentiable = bool(kink_margin > kink_tolerance)
    jacobian = full_state_jacobian(circuit, h, x, max_state_dim=max_state_dim)
    eigenvalues = np.linalg.eigvals(jacobian)
    radius = float(np.max(np.abs(eigenvalues)))
    return {
        'state_dimension': int(h.size),
        'constant_input': x.tolist(),
        'residual_inf': residual_inf,
        'residual_rms': float(np.sqrt(np.mean(residual ** 2))),
        'residual_relative_inf': residual_inf / scale,
        'residual_tolerance': threshold,
        'is_approximate_fixed_point': stationary,
        'activation_polynomial_kink_margin': kink_margin,
        'differentiable_at_candidate': differentiable,
        'full_state_spectral_radius': radius,
        'linearly_stable_fixed_point': bool(radius < 1.) if stationary and differentiable else None,
        'spectral_margin_to_unit_circle': 1. - radius,
        'global_attractor_claimed': False,
        'input_regime': 'deterministic constant input; no perturbations or rate clamps',
    }


def find_fixed_point(circuit, initial_state, input_vector=None, *, atol=1e-7,
                     rtol=1e-7, solver_xtol=1e-9, max_evaluations=2000,
                     kink_tolerance=1e-10, max_state_dim=1024):
    """Solve F(h; x)-h=0 with SciPy hybr and its full analytic Jacobian J-I.

    The default input is the five-channel zero vector, NOT the final task
    input. One starting point finds at most one candidate: it cannot establish
    uniqueness or attraction. ``success`` requires both the solver's success
    flag and an independently recomputed residual passing the stated tolerance.
    A solver's termination flag alone can never certify a transient state.

    The returned JSON-safe dictionary includes the solver status, independently
    recomputed full-state diagnostics, and the candidate state as nested lists.
    Numerical solver failure remains explicit; a nonfinite final iterate uses
    the best finite evaluated state only for diagnostic reporting, not success.
    """
    from scipy.optimize import root

    h, x = _state_and_input(circuit, initial_state, input_vector)
    h = h.copy()
    _guard_state_dimension(h.size, max_state_dim)
    max_evaluations = _positive_integer(max_evaluations, 'max_evaluations')
    if not np.isfinite(solver_xtol) or solver_xtol <= 0:
        raise ValueError('solver_xtol must be finite and positive')
    if not all(np.isfinite(v) and v >= 0 for v in (atol, rtol, kink_tolerance)):
        raise ValueError('Diagnostic tolerances must be finite and nonnegative')

    shape = h.shape
    identity = np.eye(h.size)
    initial_residual = fixed_point_residual(circuit, h, x)
    if not np.all(np.isfinite(initial_residual)):
        raise FloatingPointError('Initial fixed-point residual is nonfinite')
    initial_norm = float(np.max(np.abs(initial_residual)))
    best_state, best_norm = h.copy(), initial_norm
    evaluations, jacobian_evaluations = 0, 0

    def residual(flat):
        nonlocal best_state, best_norm, evaluations
        evaluations += 1
        state = np.asarray(flat, dtype=float).reshape(shape)
        if not np.all(np.isfinite(state)):
            raise FloatingPointError('Solver proposed a nonfinite state')
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            value = fixed_point_residual(circuit, state, x)
        if not np.all(np.isfinite(value)):
            raise FloatingPointError('Solver residual became nonfinite')
        norm = float(np.max(np.abs(value)))
        if norm < best_norm:
            best_state, best_norm = state.copy(), norm
        return value.ravel()

    def residual_jacobian(flat):
        nonlocal jacobian_evaluations
        jacobian_evaluations += 1
        state = np.asarray(flat, dtype=float).reshape(shape)
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            jac = full_state_jacobian(circuit, state, x, max_state_dim=max_state_dim) - identity
        if not np.all(np.isfinite(jac)):
            raise FloatingPointError('Solver Jacobian became nonfinite')
        return jac

    candidate_source = 'solver_final_state'
    try:
        result = root(residual, h.ravel(), jac=residual_jacobian, method='hybr',
                      options={'xtol': float(solver_xtol), 'maxfev': max_evaluations})
        candidate = np.asarray(result.x, dtype=float).reshape(shape)
        solver_success = bool(result.success)
        solver_status = int(result.status)
        solver_message = str(result.message)
        if not np.all(np.isfinite(candidate)):
            candidate = best_state
            candidate_source = 'best_finite_evaluated_state'
            solver_success = False
            solver_message += '; nonfinite final iterate rejected'
    except (FloatingPointError, OverflowError, np.linalg.LinAlgError) as exc:
        candidate = best_state
        candidate_source = 'best_finite_evaluated_state'
        solver_success = False
        solver_status = -1
        solver_message = f'Numerical solver failure: {type(exc).__name__}: {exc}'

    diagnostics = fixed_point_diagnostics(circuit, candidate, x, atol=atol, rtol=rtol,
                                          kink_tolerance=kink_tolerance,
                                          max_state_dim=max_state_dim)
    return {
        **diagnostics,
        'success': bool(solver_success and diagnostics['is_approximate_fixed_point']),
        'solver_success': solver_success,
        'solver_status': solver_status,
        'solver_message': solver_message,
        'solver_method': 'scipy.optimize.root/hybr',
        'solver_jacobian': 'full_state_jacobian - identity',
        'solver_xtol': float(solver_xtol),
        'max_function_evaluations': max_evaluations,
        'function_evaluations': int(evaluations),
        'jacobian_evaluations': int(jacobian_evaluations),
        'initial_residual_inf': initial_norm,
        'candidate_source': candidate_source,
        'state': candidate.tolist(),
        'uniqueness_claimed': False,
    }
