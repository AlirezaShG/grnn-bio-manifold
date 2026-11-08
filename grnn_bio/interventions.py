"""Paired in-silico rate silencing and current-pulse assays.

Rate multipliers are idealized interventions on model output. They are not an
opsin, photocurrent, dendrite or experimental-light model. Control and treated
trajectories receive the identical realized dimensionless drive-noise tensor.
Physical pA pulses are converted through the circuit's fixed current scale.
"""

from typing import Optional, Sequence

import numpy as np

from .analysis import perturbation_recovery, phase_average
from .task import TrialBatch, behavior_metrics, decision_values, phase_mask


def make_rate_multiplier(
    batch: TrialBatch,
    n_neurons: int,
    indices: Sequence[int],
    phase: str,
    dose: float,
) -> np.ndarray:
    """One means intact output; zero is complete idealized rate silencing."""
    if not np.isfinite(dose) or not 0 <= dose <= 1:
        raise ValueError("silencing dose must lie in [0,1]")
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 1 or not len(indices) or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= n_neurons):
        raise ValueError("indices must be a nonempty sequence of distinct valid neurons")
    active = phase_mask(batch, phase)
    if not np.any(active):
        raise ValueError(f"phase {phase!r} contains no intervention timesteps")
    multiplier = np.ones((batch.n_trials, batch.n_steps, n_neurons))
    multiplier[:, :, indices] = 1.0 - dose * active[:, :, None]
    return multiplier


def evaluate_rate_interventions(
    circuit,
    batch: TrialBatch,
    cell_types: Optional[Sequence[str]] = None,
    *,
    current_noise: Optional[np.ndarray] = None,
    phases: Sequence[str] = ("stimulus", "delay", "decision"),
    doses: Sequence[float] = (1.0,),
    target_types: Sequence[str] = ("PV", "SST", "VIP"),
    control_seed: int = 0,
    include_matched_controls: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Return summary rows and trial-level predictions for a paired assay.

    ``current_noise`` uses the circuit's dimensionless drive units, not pA.
    Matched controls silence the same number of different neurons. For an
    inhibitory class, other inhibitory cells are preferred when enough exist;
    otherwise all nontarget cells form the pool and that fallback is explicit.
    These size controls do not match each cell's firing rate or connectivity.
    Sample alternative control subsets in the planned replicated experiment.
    """
    types = np.asarray(circuit.cell_types if cell_types is None else cell_types)
    if types.shape != (circuit.N,):
        raise ValueError("one cell-type label is required for every circuit cell")
    if current_noise is not None:
        current_noise = np.asarray(current_noise, dtype=float)
        if current_noise.shape != (batch.n_trials, batch.n_steps, circuit.N) or not np.all(np.isfinite(current_noise)):
            raise ValueError("current_noise must be a finite, fixed [trial,time,neuron] tensor")
    baseline = circuit.forward(batch.X, current_noise=current_noise)
    base_metrics = behavior_metrics(baseline.logits, batch)
    base_values = decision_values(baseline.logits, batch)
    base_predictions = np.where(base_values >= 0, 1, -1)
    base_mean_rate = float(phase_average(baseline.rates_hz, batch, "all").mean())
    summary_rows = [{"intervention": "intact", "target_type": "none", "phase": "none", "dose": 0.0, "n_targeted": 0, "target_indices": [], "control_pool": "not_applicable", "baseline_accuracy": base_metrics["accuracy"], "delta_accuracy": 0.0, "choice_flip_fraction": 0.0, "mean_rate_hz": base_mean_rate, "background_noise_shared": True, **base_metrics}]
    trial_rows = []

    def append_trials(label: str, values: np.ndarray, predictions: np.ndarray) -> None:
        for trial in range(batch.n_trials):
            trial_rows.append({
                "intervention": label,
                "trial_index": trial,
                "target": float(batch.y[trial]),
                "valid_choice": bool(batch.y[trial] != 0),
                "context_rule": int(batch.meta["context_rule"][trial]),
                "congruent": bool(batch.meta["is_congruent"][trial]),
                "decision_value": float(values[trial]),
                "prediction": int(predictions[trial]),
                "baseline_prediction": int(base_predictions[trial]),
            })

    append_trials("intact", base_values, base_predictions)
    rng = np.random.default_rng(control_seed)
    for cell_type in target_types:
        selected = np.flatnonzero(types == cell_type)
        if not len(selected):
            raise ValueError(f"target cell type {cell_type!r} is absent")
        target_sets = [("targeted", selected, "target_type")]
        if include_matched_controls:
            if cell_type in {"PV", "SST", "VIP"}:
                pool = np.flatnonzero(np.isin(types, ("PV", "SST", "VIP")) & (types != cell_type))
                pool_label = "other_inhibitory_cells"
            else:
                pool = np.flatnonzero(types != cell_type)
                pool_label = "other_cells"
            if len(pool) < len(selected):
                pool = np.flatnonzero(types != cell_type)
                pool_label = "all_other_cells_fallback"
            if len(pool) < len(selected):
                raise ValueError(f"cannot construct a same-size nontarget control for {cell_type!r}")
            matched = np.sort(rng.choice(pool, size=len(selected), replace=False))
            target_sets.append(("matched_size_control", matched, pool_label))
        for phase in phases:
            for dose in doses:
                for kind, indices, pool_label in target_sets:
                    label = f"{cell_type}:{kind}:{phase}:dose{float(dose):g}"
                    multiplier = make_rate_multiplier(batch, circuit.N, indices, phase, float(dose))
                    treated = circuit.forward(batch.X, current_noise=current_noise, rate_multiplier=multiplier)
                    metrics = behavior_metrics(treated.logits, batch)
                    values = decision_values(treated.logits, batch)
                    predictions = np.where(values >= 0, 1, -1)
                    summary_rows.append({
                        "intervention": label,
                        "target_type": cell_type,
                        "intervention_kind": kind,
                        "phase": phase,
                        "dose": float(dose),
                        "n_targeted": len(indices),
                        "target_indices": indices.tolist(),
                        "control_pool": pool_label,
                        "baseline_accuracy": base_metrics["accuracy"],
                        "delta_accuracy": metrics["accuracy"] - base_metrics["accuracy"],
                        "choice_flip_fraction": float(np.mean(predictions != base_predictions)),
                        "mean_rate_hz": float(phase_average(treated.rates_hz, batch, "all").mean()),
                        "target_rate_during_phase_hz": float(phase_average(treated.rates_hz, batch, phase)[:, indices].mean()),
                        "background_noise_shared": True,
                        **metrics,
                    })
                    append_trials(label, values, predictions)
    return summary_rows, trial_rows


def evaluate_pulse_recovery(
    circuit,
    batch: TrialBatch,
    *,
    pulse_std_pa: float = 20.0,
    pulse_seed: int = 0,
    current_noise: Optional[np.ndarray] = None,
    neuron_indices: Optional[Sequence[int]] = None,
) -> dict:
    """Mid-delay additive current pulse with rate *and* hidden-state distance.

    Each trial receives one random pulse in the middle of its own delay. The
    endpoint is the last silent-delay bin, before context/go inputs resume.
    ``current_noise`` is background noise in dimensionless drive units. The
    pulse itself is sampled in pA and converted using each fixed current scale.
    A random current direction is reported as such; no orthogonality claim is
    made. Retain the no-recurrence control: intrinsic filters can also recover.
    """
    if not np.isfinite(pulse_std_pa) or pulse_std_pa < 0:
        raise ValueError("pulse_std_pa must be finite and nonnegative")
    if np.any(batch.delay_mask.sum(axis=1) < 2):
        raise ValueError("a pulse assay requires at least two delay bins per trial")
    ids = np.arange(circuit.N) if neuron_indices is None else np.asarray(neuron_indices, dtype=int)
    if ids.ndim != 1 or not len(ids) or len(np.unique(ids)) != len(ids) or np.any(ids < 0) or np.any(ids >= circuit.N):
        raise ValueError("neuron_indices must be distinct valid neurons")
    shape = (batch.n_trials, batch.n_steps, circuit.N)
    noise = np.zeros(shape) if current_noise is None else np.asarray(current_noise, dtype=float)
    if noise.shape != shape or not np.all(np.isfinite(noise)):
        raise ValueError("current_noise must match [trial,time,neuron]")
    pulse_mask = np.zeros(batch.X.shape[:2], dtype=bool)
    endpoint_mask = np.zeros_like(pulse_mask)
    for trial in range(batch.n_trials):
        delay = np.flatnonzero(batch.delay_mask[trial])
        pulse_mask[trial, delay[(len(delay) - 1) // 2]] = True
        endpoint_mask[trial, delay[-1]] = True
    direction = np.zeros((batch.n_trials, circuit.N))
    direction[:, ids] = np.random.default_rng(pulse_seed).normal(0, pulse_std_pa, size=(batch.n_trials, len(ids)))
    pulse_pa = pulse_mask[:, :, None] * direction[:, None, :]
    current_scale = np.asarray(circuit.current_scale, dtype=float)
    if current_scale.shape != (circuit.N,) or np.any(current_scale <= 0) or not np.all(np.isfinite(current_scale)):
        raise ValueError("circuit.current_scale must be positive pA per unit drive for every neuron")
    pulse_drive = pulse_pa / current_scale[None, None, :]
    baseline = circuit.forward(batch.X, current_noise=noise, record_states=True)
    treated = circuit.forward(batch.X, current_noise=noise + pulse_drive, record_states=True)
    if baseline.states is None or treated.states is None:
        raise ValueError("circuit must return full hidden states for the pulse assay")
    before, after = behavior_metrics(baseline.logits, batch), behavior_metrics(treated.logits, batch)
    base_pred = decision_values(baseline.logits, batch) >= 0
    treated_pred = decision_values(treated.logits, batch) >= 0
    return {
        "assay": "finite_horizon_random_current_pulse",
        "attractor_claim_supported": False,
        "pulse_std_pa": float(pulse_std_pa),
        "pulse_seed": int(pulse_seed),
        "background_noise_units": "dimensionless_drive",
        "target_indices": ids.tolist(),
        "pulse_steps": np.argmax(pulse_mask, axis=1).tolist(),
        "endpoint_steps": np.argmax(endpoint_mask, axis=1).tolist(),
        "background_noise_shared": True,
        "rate_displacement": perturbation_recovery(baseline.rates_hz, treated.rates_hz, pulse_mask, endpoint_mask),
        "hidden_state_displacement": perturbation_recovery(baseline.states, treated.states, pulse_mask, endpoint_mask),
        "baseline_accuracy": before["accuracy"],
        "perturbed_accuracy": after["accuracy"],
        "delta_accuracy": after["accuracy"] - before["accuracy"],
        "choice_flip_fraction": float(np.mean(base_pred != treated_pred)),
    }
