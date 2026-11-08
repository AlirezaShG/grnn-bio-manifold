"""Auditable learning rules for a frozen-physiology, current-based GFR circuit.

The only plastic parameters are recurrent magnitudes, readout weights and bias.
``local_eligibility_gradients`` is a forward-time local approximation; it is not
the exact network gradient or a claim to implement biological e-prop literally.
All routines use the same decision-bin-normalized binary cross-entropy.  See
``docs/LEARNING_RULES.md`` for the equations, approximations and cost accounting.

NumPy is the only dependency.  Parameter updates are deliberately separate from
gradient estimation, so every estimate is evaluated at one fixed parameter set.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _batch_arrays(batch):
    """Validate a binary trial batch and return normalized decision weights."""
    x = np.asarray(batch.X, dtype=np.float64)
    y = np.asarray(batch.y, dtype=np.float64)
    mask = np.asarray(batch.decision_mask, dtype=np.float64)
    if x.ndim != 3 or x.shape[0] == 0 or x.shape[1] == 0:
        raise ValueError("X must have nonempty shape [batch, time, input].")
    if y.shape != (x.shape[0],) or not np.all(np.isin(y, [-1.0, 1.0])):
        raise ValueError("Training labels must have shape [batch] and be -1 or +1.")
    if mask.shape != x.shape[:2] or not np.all(np.isfinite(mask)) or np.any(mask < 0):
        raise ValueError("decision_mask must be finite, nonnegative and match [batch,time].")
    counts = mask.sum(axis=1)
    if np.any(counts <= 0):
        raise ValueError("Every trial must contain at least one decision bin.")
    if not np.all(np.isfinite(x)):
        raise ValueError("Inputs contain nonfinite values.")
    return x, y, mask / counts[:, None]


def _noise_array(noise, shape):
    if noise is None:
        return None
    noise = np.asarray(noise, dtype=np.float64)
    if noise.shape != shape or not np.all(np.isfinite(noise)):
        raise ValueError(f"Normalized current noise must be finite and have shape {shape}.")
    return noise


def _readout(circuit):
    return np.asarray(circuit.W_out) * np.asarray(circuit.readout_mask)


def _time_loss_and_delta(logit, y):
    """Stable BCE for +/-1 labels and its derivative with respect to logit."""
    margin = y * logit
    return np.logaddexp(0.0, -margin), -y * np.exp(-np.logaddexp(0.0, margin))


def _metrics(logits, y, weights):
    losses = np.sum(np.logaddexp(0.0, -y[:, None] * logits) * weights, axis=1)
    decision = np.sum(logits * weights, axis=1)
    prediction = np.where(decision >= 0.0, 1.0, -1.0)
    delta = -y[:, None] * np.exp(-np.logaddexp(0.0, y[:, None] * logits)) * weights
    return losses, float(np.mean(prediction == y)), delta


def _forward_cache(circuit, x, noise=None):
    """Independent explicit forward kernel used by the analytical BPTT audit.

    Rates and activation slopes are in Hz.  The fitted a and beta coefficients
    are already those used by upstream's discrete recurrence: no extra dt is
    applied.  The initial state must be independent of plastic parameters.
    """
    batch_size, n_time, _ = x.shape
    h, r = circuit.initial_state(batch_size)
    h, r = np.array(h, dtype=np.float64, copy=True), np.array(r, dtype=np.float64, copy=True)
    rate_scale = np.asarray(circuit.rate_scale_hz)
    normalized = np.zeros((batch_size, n_time + 1, circuit.N), dtype=np.float64)
    slopes = np.zeros((batch_size, n_time, circuit.N), dtype=np.float64)
    logits = np.zeros((batch_size, n_time), dtype=np.float64)
    normalized[:, 0] = r / rate_scale
    w = circuit.W_rec
    out = _readout(circuit)
    for t in range(n_time):
        drive = x[:, t] @ circuit.input_weights.T + normalized[:, t] @ w.T
        if noise is not None:
            drive = drive + noise[:, t]
        current = circuit.current_offset + circuit.current_scale * drive
        h = (1.0 - circuit.ds) * h + circuit.a * current[:, :, None] + circuit.beta * r[:, :, None]
        r, slopes[:, t] = circuit.activation_and_grad(h.mean(axis=-1))
        normalized[:, t + 1] = r / rate_scale
        logits[:, t] = normalized[:, t + 1] @ out + float(circuit.b_out)
    if not (np.all(np.isfinite(h)) and np.all(np.isfinite(normalized)) and np.all(np.isfinite(slopes))):
        raise FloatingPointError("GFR trajectory or derivatives became nonfinite.")
    return normalized, slopes, logits


def _stream_trial_losses(circuit, x, y, weights):
    """Clean baseline evaluated without storing a time-indexed trajectory."""
    h, r = circuit.initial_state(x.shape[0])
    h, r = np.array(h, dtype=np.float64, copy=True), np.array(r, dtype=np.float64, copy=True)
    loss_per_trial = np.zeros(x.shape[0])
    w, out = circuit.W_rec, _readout(circuit)
    for t in range(x.shape[1]):
        drive = x[:, t] @ circuit.input_weights.T + (r / circuit.rate_scale_hz) @ w.T
        current = circuit.current_offset + circuit.current_scale * drive
        h = (1.0 - circuit.ds) * h + circuit.a * current[:, :, None] + circuit.beta * r[:, :, None]
        r, _ = circuit.activation_and_grad(h.mean(axis=-1))
        logit = (r / circuit.rate_scale_hz) @ out + float(circuit.b_out)
        loss_per_trial += weights[:, t] * np.logaddexp(0.0, -y * logit)
    if not np.all(np.isfinite(h)) or not np.all(np.isfinite(loss_per_trial)):
        raise FloatingPointError("Baseline trajectory became nonfinite.")
    return loss_per_trial


def _result(circuit, rec, out, bias, loss, accuracy, diagnostics):
    rec = np.asarray(rec, dtype=np.float64) * np.asarray(circuit.mask)
    out = np.asarray(out, dtype=np.float64) * np.asarray(circuit.readout_mask)
    bias = float(bias)
    if not (np.all(np.isfinite(rec)) and np.all(np.isfinite(out)) and np.isfinite(bias)
            and np.isfinite(loss) and np.isfinite(accuracy)):
        raise FloatingPointError("Learning rule produced nonfinite loss or gradients.")
    return {"rec": rec, "out": out, "bias": bias, "loss": float(loss),
            "accuracy": float(accuracy), "diagnostics": diagnostics}


def exact_bptt_gradients(circuit, batch, *, noise=None):
    """Full analytical network gradient, including intrinsic feedback and loops.

    ``noise`` is optional, fixed normalized-current noise [B,T,N].  BPTT treats
    that draw as an exogenous constant; passing it permits matched-noise controls.
    No parameter is mutated by this function.
    """
    x, y, weights = _batch_arrays(batch)
    b, t_max, _ = x.shape
    noise = _noise_array(noise, (b, t_max, circuit.N))
    normalized, slopes, logits = _forward_cache(circuit, x, noise)
    losses, accuracy, delta = _metrics(logits, y, weights)
    delta /= b
    grad_out = np.einsum("bt,btn->n", delta, normalized[:, 1:])
    grad_bias = float(delta.sum())
    grad_rec = np.zeros_like(circuit.W_abs, dtype=np.float64)
    dh_future = np.zeros((b, circuit.N, circuit.K), dtype=np.float64)
    dr_future = np.zeros((b, circuit.N), dtype=np.float64)
    temporal_norms = np.zeros(t_max, dtype=np.float64)
    out = _readout(circuit)
    w = circuit.W_rec
    for t in range(t_max - 1, -1, -1):
        dr = delta[:, t, None] * out / circuit.rate_scale_hz + dr_future
        dh = dh_future + (dr * slopes[:, t])[:, :, None] / circuit.K
        temporal_norms[t] = np.sqrt(np.mean(dh ** 2))
        d_current = np.sum(dh * circuit.a, axis=-1)
        d_drive = d_current * circuit.current_scale
        grad_rec += (d_drive.T @ normalized[:, t]) * circuit.signs[None, :]
        dr_future = (d_drive @ w) / circuit.rate_scale_hz + np.sum(dh * circuit.beta, axis=-1)
        dh_future = dh * (1.0 - circuit.ds)
    return _result(circuit, grad_rec, grad_out, grad_bias, losses.mean(), accuracy, {
        "method": "bptt", "gradient_kind": "exact deterministic derivative",
        "forward_rollouts": 1, "backward_sweeps": 1, "simulated_trial_bins": b * t_max,
        "stored_time_bins": t_max, "temporal_gradient_rms": temporal_norms.tolist(),
        "training_current_noise": noise is not None,
    })


def _feedback_vector(circuit, feedback_mode):
    if feedback_mode == "random":
        feedback = np.asarray(circuit.feedback)
    elif feedback_mode == "symmetric":
        feedback = _readout(circuit)
    else:
        raise ValueError("feedback_mode must be 'random' or 'symmetric'.")
    if feedback.shape != (circuit.N,) or not np.all(np.isfinite(feedback)):
        raise ValueError("Feedback must be a finite vector of length N.")
    return feedback


def local_eligibility_gradients(circuit, batch, *, include_beta=True,
                                feedback_mode="random", noise=None,
                                implementation="adjoint"):
    """Cell-local e-prop gradient for GFR neurons.

    The estimate is sum_t L_i,t * dr_i,t/dw_ij, where the derivative follows
    only neuron i's own state (filters h and spike-history feedback beta) and
    treats every other neuron's activity as fixed input. L_i,t is the output
    error broadcast through a fixed feedback vector.

    ``implementation='forward'`` keeps per-synapse eligibility traces and runs
    forward in time, as an online synapse would. ``'adjoint'`` computes the same
    quantity by one reverse sweep over each neuron's own dynamics (the network
    gradient with all paths through other neurons removed). The two are equal up
    to floating-point rounding (tested); the adjoint is ~50x faster in NumPy.
    """
    if implementation == "forward":
        return _local_forward(circuit, batch, include_beta=include_beta,
                              feedback_mode=feedback_mode, noise=noise)
    if implementation != "adjoint":
        raise ValueError("implementation must be 'adjoint' or 'forward'.")
    x, y, weights = _batch_arrays(batch)
    b, t_max, _ = x.shape
    noise = _noise_array(noise, (b, t_max, circuit.N))
    feedback = _feedback_vector(circuit, feedback_mode)
    normalized, slopes, logits = _forward_cache(circuit, x, noise)
    losses, accuracy, delta = _metrics(logits, y, weights)
    delta /= b
    grad_out = np.einsum("bt,btn->n", delta, normalized[:, 1:])
    grad_bias = float(delta.sum())
    grad_rec = np.zeros_like(circuit.W_abs, dtype=np.float64)
    dh_future = np.zeros((b, circuit.N, circuit.K), dtype=np.float64)
    dr_own_future = np.zeros((b, circuit.N), dtype=np.float64)
    learning_signal = feedback / circuit.rate_scale_hz
    for t in range(t_max - 1, -1, -1):
        dr = delta[:, t, None] * learning_signal + dr_own_future
        dh = dh_future + (dr * slopes[:, t])[:, :, None] / circuit.K
        d_drive = np.sum(dh * circuit.a, axis=-1) * circuit.current_scale
        grad_rec += (d_drive.T @ normalized[:, t]) * circuit.signs[None, :]
        # Only the neuron's own spike-history path; no term through W.
        dr_own_future = np.sum(dh * circuit.beta, axis=-1) if include_beta else 0.0 * dr_own_future
        dh_future = dh * (1.0 - circuit.ds)
    return _result(circuit, grad_rec, grad_out, grad_bias, losses.mean(), accuracy, {
        "method": "local_beta" if include_beta else "local_no_beta",
        "gradient_kind": "cell-local approximation", "feedback_mode": feedback_mode,
        "implementation": "adjoint (equal to forward eligibility traces)",
        "forward_rollouts": 1, "backward_sweeps": 1, "simulated_trial_bins": b * t_max,
        "stored_time_bins": t_max, "eligibility_channels": int(circuit.K),
        "training_current_noise": noise is not None,
    })


def _local_forward(circuit, batch, *, include_beta=True,
                   feedback_mode="random", noise=None):
    """Forward-time GFR eligibility with an approximate instantaneous signal.

    Per-edge K-vector traces retain the exact isolated-cell beta Jacobian while
    ignoring derivatives through other neurons' recurrent activities.  Random
    feedback supplies a fixed teaching direction to every postsynaptic class.
    ``include_beta=False`` removes ONLY that Jacobian term: the same K channels,
    dynamics, readout, graph and parameter support remain in the control.

    No trajectory is retained and no backward-time loop is used.  Weights are
    held fixed through the trial batch and updated separately at its end.
    """
    x, y, weights = _batch_arrays(batch)
    b, t_max, _ = x.shape
    noise = _noise_array(noise, (b, t_max, circuit.N))
    post, pre = np.asarray(circuit.edge_post), np.asarray(circuit.edge_pre)
    n_edges = len(post)
    feedback = _feedback_vector(circuit, feedback_mode)
    h, r = circuit.initial_state(b)
    h, r = np.array(h, dtype=np.float64, copy=True), np.array(r, dtype=np.float64, copy=True)
    previous_slope = np.zeros((b, circuit.N), dtype=np.float64)
    eligibility = np.zeros((b, n_edges, circuit.K), dtype=np.float64)
    edge_gradient = np.zeros(n_edges, dtype=np.float64)
    grad_out = np.zeros(circuit.N, dtype=np.float64)
    grad_bias = 0.0
    loss_per_trial = np.zeros(b, dtype=np.float64)
    decision_logit = np.zeros(b, dtype=np.float64)
    w, out = circuit.W_rec, _readout(circuit)
    decay = 1.0 - circuit.ds[post]
    direct_gain = circuit.a[post] * circuit.current_scale[post, None] * circuit.signs[pre, None]
    for t in range(t_max):
        normalized_pre = r / circuit.rate_scale_hz
        local_feedback = 0.0
        if include_beta:
            local_feedback = (circuit.beta[post] * previous_slope[:, post, None]
                              * eligibility.mean(axis=-1, keepdims=True))
        eligibility = (eligibility * decay
                       + normalized_pre[:, pre, None] * direct_gain
                       + local_feedback)
        drive = x[:, t] @ circuit.input_weights.T + normalized_pre @ w.T
        if noise is not None:
            drive = drive + noise[:, t]
        current = circuit.current_offset + circuit.current_scale * drive
        h = (1.0 - circuit.ds) * h + circuit.a * current[:, :, None] + circuit.beta * r[:, :, None]
        r, slope = circuit.activation_and_grad(h.mean(axis=-1))
        normalized = r / circuit.rate_scale_hz
        logit = normalized @ out + float(circuit.b_out)
        time_loss, time_delta = _time_loss_and_delta(logit, y)
        delta = time_delta * weights[:, t] / b
        loss_per_trial += weights[:, t] * time_loss
        decision_logit += weights[:, t] * logit
        grad_out += delta @ normalized
        grad_bias += float(delta.sum())
        rate_eligibility = (slope[:, post] / circuit.rate_scale_hz[post]) * eligibility.mean(axis=-1)
        edge_gradient += np.sum(delta[:, None] * feedback[post] * rate_eligibility, axis=0)
        previous_slope = slope
    if not np.all(np.isfinite(h)):
        raise FloatingPointError("GFR hidden state became nonfinite.")
    grad_rec = np.zeros_like(circuit.W_abs, dtype=np.float64)
    grad_rec[post, pre] = edge_gradient
    accuracy = np.mean(np.where(decision_logit >= 0.0, 1.0, -1.0) == y)
    return _result(circuit, grad_rec, grad_out, grad_bias, loss_per_trial.mean(), accuracy, {
        "method": "local_beta" if include_beta else "local_no_beta",
        "gradient_kind": "cell-local approximation", "feedback_mode": feedback_mode,
        "implementation": "forward eligibility traces",
        "forward_rollouts": 1, "backward_sweeps": 0, "simulated_trial_bins": b * t_max,
        "stored_time_bins": 0, "eligibility_elements": int(eligibility.size),
        "eligibility_channels": int(circuit.K), "training_current_noise": noise is not None,
    })


def readout_only_gradients(circuit, batch, *, noise=None):
    """Frozen recurrent reservoir; direct readout delta rule, no hidden gradient."""
    x, y, weights = _batch_arrays(batch)
    b, t_max, _ = x.shape
    noise = _noise_array(noise, (b, t_max, circuit.N))
    h, r = circuit.initial_state(b)
    h, r = np.array(h, dtype=np.float64, copy=True), np.array(r, dtype=np.float64, copy=True)
    grad_out = np.zeros(circuit.N)
    grad_bias = 0.0
    losses = np.zeros(b)
    decisions = np.zeros(b)
    w, out = circuit.W_rec, _readout(circuit)
    for t in range(t_max):
        drive = x[:, t] @ circuit.input_weights.T + (r / circuit.rate_scale_hz) @ w.T
        if noise is not None:
            drive = drive + noise[:, t]
        current = circuit.current_offset + circuit.current_scale * drive
        h = (1.0 - circuit.ds) * h + circuit.a * current[:, :, None] + circuit.beta * r[:, :, None]
        r, _ = circuit.activation_and_grad(h.mean(axis=-1))
        normalized = r / circuit.rate_scale_hz
        logit = normalized @ out + float(circuit.b_out)
        loss, delta = _time_loss_and_delta(logit, y)
        losses += loss * weights[:, t]
        decisions += logit * weights[:, t]
        delta = delta * weights[:, t] / b
        grad_out += delta @ normalized
        grad_bias += float(delta.sum())
    if not np.all(np.isfinite(h)):
        raise FloatingPointError("GFR hidden state became nonfinite.")
    accuracy = np.mean(np.where(decisions >= 0, 1.0, -1.0) == y)
    return _result(circuit, np.zeros_like(circuit.W_abs), grad_out, grad_bias,
                   losses.mean(), accuracy, {
                       "method": "readout_only", "gradient_kind": "exact readout derivative",
                       "recurrent_plastic": False,
                       "forward_rollouts": 1, "backward_sweeps": 0,
                       "simulated_trial_bins": b * t_max, "stored_time_bins": 0,
                       "training_current_noise": noise is not None,
                   })


def node_perturbation_gradients(circuit, batch, *, rng, sigma=0.1, baseline="clean"):
    """Scalar-loss-modulated, local Gaussian current-perturbation score estimator.

    For each trial, accumulate e_ij = sum_t xi_i,t / sigma^2 * s_j * rnorm_j,t-1,
    then multiply by that trial's loss minus an independent baseline.  This is
    an unbiased score estimator of expected *noisy-trajectory* loss.  It is not
    claimed to equal deterministic BPTT at finite sigma.  Readout derivatives
    are exact conditional on the noisy trajectory.  The default clean baseline
    costs an additional complete rollout and is reported in diagnostics.
    """
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Perturbation sigma must be strictly positive and finite.")
    if baseline not in ("clean", "zero"):
        raise ValueError("baseline must be 'clean' or 'zero'.")
    x, y, weights = _batch_arrays(batch)
    b, t_max, _ = x.shape
    if baseline == "clean":
        baseline_losses = _stream_trial_losses(circuit, x, y, weights)
        forward_count = 2
    else:
        baseline_losses = np.zeros(b)
        forward_count = 1
    post, pre = np.asarray(circuit.edge_post), np.asarray(circuit.edge_pre)
    score = np.zeros((b, len(post)))
    last_decision = np.max(np.where(weights > 0, np.arange(t_max)[None, :], -1), axis=1)
    h, r = circuit.initial_state(b)
    h, r = np.array(h, dtype=np.float64, copy=True), np.array(r, dtype=np.float64, copy=True)
    grad_out = np.zeros(circuit.N)
    grad_bias = 0.0
    loss_per_trial = np.zeros(b)
    decisions = np.zeros(b)
    w, out = circuit.W_rec, _readout(circuit)
    for t in range(t_max):
        normalized_pre = r / circuit.rate_scale_hz
        perturbation = rng.normal(0.0, sigma, size=(b, circuit.N))
        # Noise after a trial's final response cannot affect its loss. Including
        # its zero-mean score would add unnecessary variance for padded trials.
        score += ((t <= last_decision)[:, None] * perturbation[:, post] / (sigma ** 2)
                  * circuit.signs[pre] * normalized_pre[:, pre])
        drive = x[:, t] @ circuit.input_weights.T + normalized_pre @ w.T + perturbation
        current = circuit.current_offset + circuit.current_scale * drive
        h = (1.0 - circuit.ds) * h + circuit.a * current[:, :, None] + circuit.beta * r[:, :, None]
        r, _ = circuit.activation_and_grad(h.mean(axis=-1))
        normalized = r / circuit.rate_scale_hz
        logit = normalized @ out + float(circuit.b_out)
        loss, delta = _time_loss_and_delta(logit, y)
        loss_per_trial += weights[:, t] * loss
        decisions += weights[:, t] * logit
        delta = weights[:, t] * delta / b
        grad_out += delta @ normalized
        grad_bias += float(delta.sum())
    if not np.all(np.isfinite(h)):
        raise FloatingPointError("GFR hidden state became nonfinite.")
    advantage = loss_per_trial - baseline_losses
    edge_gradient = np.mean(advantage[:, None] * score, axis=0)
    grad_rec = np.zeros_like(circuit.W_abs)
    grad_rec[post, pre] = edge_gradient
    accuracy = np.mean(np.where(decisions >= 0, 1.0, -1.0) == y)
    return _result(circuit, grad_rec, grad_out, grad_bias, loss_per_trial.mean(), accuracy, {
        "method": "node_perturbation", "gradient_kind": "Gaussian score estimator",
        "target_objective": "expected noisy-trajectory decision BCE",
        "forward_rollouts": forward_count, "backward_sweeps": 0,
        "simulated_trial_bins": forward_count * b * t_max,
        "stored_time_bins": 0,
        "eligibility_elements": int(score.size), "noise_std_normalized_current": float(sigma),
        "baseline": baseline, "advantage_std": float(np.std(advantage)),
    })


@dataclass
class ProjectedOptimizer:
    """Shared constrained SGD/Adam optimizer, independent of the learning rule.

    Adam is an optimization convenience, not a claim about biological plasticity.
    Use ``kind='sgd'`` for the more restrictive update comparison.  Positive
    recurrent magnitudes and the immutable support impose Dale's law; input and
    intrinsic parameters are never touched.  Global clipping is disclosed.
    """

    kind: str = "adam"
    lr_rec: float = 1e-3
    lr_out: float = 1e-2
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    max_grad_norm: float | None = 1.0
    max_weight: float | None = 2.0
    weight_decay: float = 0.0
    steps: int = 0
    _moments: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self):
        # 'rmsprop': each parameter divides its own update by a running RMS of its
        # own past updates. No momentum and (if max_grad_norm is None) no global
        # clipping, so every quantity is local to the synapse. It is an
        # optimizer, not a model of homeostatic synaptic scaling.
        if self.kind not in ("adam", "sgd", "rmsprop"):
            raise ValueError("Optimizer kind must be 'adam', 'sgd' or 'rmsprop'.")
        if not all(np.isfinite(v) and v >= 0 for v in (self.lr_rec, self.lr_out, self.weight_decay)):
            raise ValueError("Learning rates and weight decay must be finite and nonnegative.")
        if not (0 <= self.beta1 < 1 and 0 <= self.beta2 < 1 and self.epsilon > 0):
            raise ValueError("Invalid Adam coefficients.")
        if any(v is not None and (not np.isfinite(v) or v <= 0)
               for v in (self.max_grad_norm, self.max_weight)):
            raise ValueError("Gradient and magnitude bounds must be positive or None.")

    def step(self, circuit, result):
        grads = {"rec": np.array(result["rec"], dtype=np.float64, copy=True),
                 "out": np.array(result["out"], dtype=np.float64, copy=True),
                 "bias": np.asarray(float(result["bias"]))}
        if grads["rec"].shape != circuit.W_abs.shape or grads["out"].shape != circuit.W_out.shape:
            raise ValueError("Gradient shapes do not match circuit parameters.")
        grads["rec"] *= circuit.mask
        grads["out"] *= circuit.readout_mask
        if not all(np.all(np.isfinite(g)) for g in grads.values()):
            raise FloatingPointError("Refusing a nonfinite optimizer update.")
        norm = float(np.sqrt(sum(np.sum(g * g) for g in grads.values())))
        scale = min(1.0, self.max_grad_norm / max(norm, 1e-300)) if self.max_grad_norm else 1.0
        for name, g in grads.items():
            if name in self._moments:
                m, v = self._moments[name]
                if m.shape != g.shape or v.shape != g.shape:
                    raise ValueError("Optimizer checkpoint shape does not match circuit.")
        self.steps += 1
        updates = {}
        for name, g in grads.items():
            g = g * scale
            if self.kind in ("adam", "rmsprop"):
                m, v = self._moments.get(name, (np.zeros_like(g), np.zeros_like(g)))
                if m.shape != g.shape or v.shape != g.shape:
                    raise ValueError("Optimizer checkpoint shape does not match circuit.")
                v = self.beta2 * v + (1.0 - self.beta2) * g * g
                if self.kind == "adam":
                    m = self.beta1 * m + (1.0 - self.beta1) * g
                    first = m / (1.0 - self.beta1 ** self.steps)
                else:
                    first = g
                self._moments[name] = (m, v)
                g = first / (np.sqrt(v / (1.0 - self.beta2 ** self.steps)) + self.epsilon)
            updates[name] = g
        frozen_recurrent = not result.get("diagnostics", {}).get("recurrent_plastic", True)
        if frozen_recurrent:
            proposed = np.array(circuit.W_abs, copy=True)
        else:
            proposed = ((1.0 - self.lr_rec * self.weight_decay) * circuit.W_abs
                        - self.lr_rec * updates["rec"])
        upper = np.inf if self.max_weight is None else self.max_weight
        clipped = proposed if frozen_recurrent else np.clip(proposed, 0.0, upper)
        changed = (clipped != proposed) & np.asarray(circuit.mask, dtype=bool)
        circuit.W_abs[...] = clipped * circuit.mask
        circuit.W_out[...] = ((1.0 - self.lr_out * self.weight_decay) * circuit.W_out
                              - self.lr_out * updates["out"]) * circuit.readout_mask
        circuit.b_out = float(circuit.b_out) - self.lr_out * float(updates["bias"])
        return {"gradient_norm": norm, "gradient_clip_factor": scale,
                "projection_fraction": float(changed.sum() / max(1, np.count_nonzero(circuit.mask))),
                "optimizer_step": self.steps}

    def state_dict(self):
        state = {key: getattr(self, key) for key in (
            "kind", "lr_rec", "lr_out", "beta1", "beta2", "epsilon", "max_grad_norm",
            "max_weight", "weight_decay", "steps")}
        for name, (m, v) in self._moments.items():
            state["m_" + name], state["v_" + name] = np.array(m, copy=True), np.array(v, copy=True)
        return state

    def load_state_dict(self, state):
        for key in ("kind", "lr_rec", "lr_out", "beta1", "beta2", "epsilon", "max_grad_norm",
                    "max_weight", "weight_decay", "steps"):
            if key not in state:
                raise ValueError(f"Optimizer checkpoint is missing {key}.")
            setattr(self, key, state[key])
        self.steps = int(self.steps)
        if self.steps < 0:
            raise ValueError("Optimizer step counter must be nonnegative.")
        self.__post_init__()
        self._moments = {}
        for name in ("rec", "out", "bias"):
            if "m_" + name in state or "v_" + name in state:
                if "m_" + name not in state or "v_" + name not in state:
                    raise ValueError(f"Incomplete Adam state for {name}.")
                m = np.array(state["m_" + name], dtype=np.float64, copy=True)
                v = np.array(state["v_" + name], dtype=np.float64, copy=True)
                if m.shape != v.shape or not np.all(np.isfinite(m)) or not np.all(np.isfinite(v)) or np.any(v < 0):
                    raise ValueError(f"Invalid Adam state for {name}.")
                self._moments[name] = m, v
        if self.kind in ("adam", "rmsprop") and self.steps and set(self._moments) != {"rec", "out", "bias"}:
            raise ValueError("A nonempty Adam checkpoint must include all moment buffers.")


def finite_difference_audit(circuit, batch, *, epsilon=1e-5, n_edges=24,
                            seed=0, noise=None, rtol=1e-4, atol=1e-7):
    """Central finite differences for several recurrent edges and every readout.

    The sample includes excitatory/inhibitory pre- AND postsynaptic classes and
    the strongest gradient in addition to seeded random edges.  A flat recurrent
    objective is explicitly ``inconclusive``, never an automatic zero-error pass.
    Returns full checked coordinates and numerical/analytical values for audit.
    """
    if not np.isfinite(epsilon) or epsilon <= 0 or n_edges <= 0:
        raise ValueError("epsilon and n_edges must be positive.")
    x, y, weights = _batch_arrays(batch)
    noise = _noise_array(noise, (x.shape[0], x.shape[1], circuit.N))
    analytical = exact_bptt_gradients(circuit, batch, noise=noise)
    edges = np.column_stack((circuit.edge_post, circuit.edge_pre))
    if len(edges) == 0:
        raise ValueError("Cannot audit recurrent gradients on a graph with no edges.")
    rng = np.random.default_rng(seed)
    chosen = set(map(int, rng.choice(len(edges), min(n_edges, len(edges)), replace=False)))
    edge_values = analytical["rec"][edges[:, 0], edges[:, 1]]
    chosen.add(int(np.argmax(np.abs(edge_values))))
    for axis in (0, 1):
        for sign in (-1, 1):
            candidates = np.flatnonzero(circuit.signs[edges[:, axis]] == sign)
            if len(candidates):
                chosen.add(int(candidates[np.argmax(np.abs(edge_values[candidates]))]))

    def loss():
        _, _, logits = _forward_cache(circuit, x, noise)
        return float(_metrics(logits, y, weights)[0].mean())

    checks = []
    for e in sorted(chosen):
        i, j = map(int, edges[e])
        original = float(circuit.W_abs[i, j])
        step = epsilon * max(1.0, abs(original))
        try:
            circuit.W_abs[i, j] = original + step
            plus = loss()
            circuit.W_abs[i, j] = original - step
            minus = loss()
        finally:
            circuit.W_abs[i, j] = original
        checks.append({"parameter": "rec", "post": i, "pre": j,
                       "analytical": float(analytical["rec"][i, j]),
                       "numerical": (plus - minus) / (2.0 * step)})
    for i in np.flatnonzero(circuit.readout_mask):
        original = float(circuit.W_out[i])
        step = epsilon * max(1.0, abs(original))
        try:
            circuit.W_out[i] = original + step
            plus = loss()
            circuit.W_out[i] = original - step
            minus = loss()
        finally:
            circuit.W_out[i] = original
        checks.append({"parameter": "out", "index": int(i),
                       "analytical": float(analytical["out"][i]),
                       "numerical": (plus - minus) / (2.0 * step)})
    original = float(circuit.b_out)
    step = epsilon * max(1.0, abs(original))
    try:
        circuit.b_out = original + step
        plus = loss()
        circuit.b_out = original - step
        minus = loss()
    finally:
        circuit.b_out = original
    checks.append({"parameter": "bias", "analytical": analytical["bias"],
                   "numerical": (plus - minus) / (2.0 * step)})
    analytical_vector = np.array([r["analytical"] for r in checks])
    numerical_vector = np.array([r["numerical"] for r in checks])
    passed = bool(np.allclose(analytical_vector, numerical_vector, rtol=rtol, atol=atol))
    recurrent_nonzero = sum(r["parameter"] == "rec" and max(abs(r["numerical"]), abs(r["analytical"])) > atol
                            for r in checks)
    error = analytical_vector - numerical_vector
    return {"status": ("pass" if recurrent_nonzero else "inconclusive") if passed else "fail",
            "epsilon": epsilon, "rtol": rtol, "atol": atol,
            "relative_l2_error": float(np.linalg.norm(error) / max(np.linalg.norm(numerical_vector), 1e-15)),
            "max_absolute_error": float(np.max(np.abs(error))),
            "checked_recurrent_edges": len(chosen), "nonzero_recurrent_checks": recurrent_nonzero,
            "checks": checks}
