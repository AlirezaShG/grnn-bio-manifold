"""Balanced context-dependent delayed decisions with explicit trial phase masks.

This is a synthetic task inspired by context-dependent decision experiments, not
a replication of a particular animal dataset. Input channels are sensory A,
sensory B, context A, context B, and response/go. Only the sensory channels carry
evidence about the correct sign. Input timing and padding never encode choice.
"""

from dataclasses import dataclass
from itertools import product
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class TaskConfig:
    bin_size_ms: int = 20
    t_fixation: int = 5
    t_stimulus: int = 10
    t_decision: int = 3
    delay_bins: tuple[int, ...] = (5, 10, 15)
    coherence_levels: tuple[float, ...] = (0.25, 0.5, 1.0)
    noise_std: float = 0.25
    context_timing: str = "early"
    t_context: int = 3
    distractor_std: float = 0.0
    distractor_bins: int = 2
    # 'context': context-dependent decision (v7). 'dms': delayed match-to-sample
    # (v11 Task 2). In 'dms', channel 0 is the sample, channel 1 a simultaneous
    # distractor stream, channels 2/3 a post-delay probe (+1/-1) and the target
    # is sign(sample) * probe, so nothing before the probe predicts the choice.
    task_type: str = "context"

    def __post_init__(self) -> None:
        if self.task_type not in {"context", "dms"}:
            raise ValueError("task_type must be 'context' or 'dms'")
        if self.task_type == "dms" and self.context_timing != "late":
            raise ValueError("dms presents its probe after the delay: use context_timing='late'")
        for name in ("bin_size_ms", "t_stimulus", "t_decision", "t_context"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(self.t_fixation, (int, np.integer)) or self.t_fixation < 0:
            raise ValueError("t_fixation must be a nonnegative integer")
        if not isinstance(self.distractor_bins, (int, np.integer)) or self.distractor_bins < 1:
            raise ValueError("distractor_bins must be a positive integer")
        _positive_delays(self.delay_bins)
        levels = np.asarray(self.coherence_levels, dtype=float)
        if levels.ndim != 1 or not len(levels) or not np.all(np.isfinite(levels)) or np.any(levels <= 0):
            raise ValueError("coherence_levels must be finite, strictly positive magnitudes")
        if self.context_timing not in {"early", "late"}:
            raise ValueError("context_timing must be 'early' or 'late'")
        for name in ("noise_std", "distractor_std"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _positive_delays(values: Any) -> np.ndarray:
    a = np.asarray(values)
    if a.ndim == 0:
        a = a.reshape(1)
    if a.ndim != 1 or not len(a) or not np.all(np.isfinite(a)):
        raise ValueError("delay_bins must be a positive integer or a nonempty sequence")
    if np.any(a < 1) or np.any(a != np.floor(a)):
        raise ValueError("delay_bins must contain positive integers")
    return a.astype(np.int64)


@dataclass
class TrialBatch:
    X: np.ndarray
    y: np.ndarray
    meta: dict[str, Any]
    decision_mask: np.ndarray
    delay_mask: np.ndarray
    stimulus_mask: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None
    context_mask: Optional[np.ndarray] = None
    fixation_mask: Optional[np.ndarray] = None
    distractor_mask: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        self.X = np.asarray(self.X, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.X.ndim != 3 or self.X.shape[2] != 5:
            raise ValueError("X must have shape [trials, time, 5]")
        shape = self.X.shape[:2]
        if not shape[0] or not shape[1] or self.y.shape != (shape[0],):
            raise ValueError("nonempty trials/time and one target per trial are required")
        if not np.all(np.isfinite(self.X)) or not np.all(np.isin(self.y, (-1.0, 0.0, 1.0))):
            raise ValueError("inputs must be finite and targets must be -1, 0, or +1")
        for name in ("decision_mask", "delay_mask", "stimulus_mask", "valid_mask", "context_mask", "fixation_mask", "distractor_mask"):
            mask = getattr(self, name)
            if mask is None:
                mask = np.ones(shape, dtype=bool) if name == "valid_mask" else np.zeros(shape, dtype=bool)
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != shape:
                raise ValueError(f"{name} must have shape [trials, time]")
            setattr(self, name, mask)
        if np.any(self.decision_mask.sum(axis=1) == 0):
            raise ValueError("every trial must have a decision window")
        if np.any(self.decision_mask & self.delay_mask):
            raise ValueError("decision and delay masks may not overlap")
        for name in ("decision_mask", "delay_mask", "stimulus_mask", "context_mask", "fixation_mask", "distractor_mask"):
            if np.any(getattr(self, name) & ~self.valid_mask):
                raise ValueError(f"{name} contains padded timesteps")
        self.meta = dict(self.meta)
        provided = np.asarray(self.meta.get("valid_choice", self.y != 0), dtype=bool)
        if provided.shape != self.y.shape or np.any(provided != (self.y != 0)):
            raise ValueError("valid_choice must mark exactly the nonzero targets")
        self.meta["valid_choice"] = provided

    @property
    def n_trials(self) -> int:
        return self.X.shape[0]

    @property
    def n_steps(self) -> int:
        return self.X.shape[1]


def generate_batch(
    cfg: TaskConfig,
    n_trials: int,
    rng: np.random.Generator,
    *,
    delay_bins: Optional[Sequence[int] | int] = None,
    noise_std: Optional[float] = None,
    distractor_std: Optional[float] = None,
    relevant_coherence: Optional[float] = None,
    context_rule: Optional[int] = None,
) -> TrialBatch:
    """Draw a reproducible batch using only the supplied RNG.

    Context, relevant sign and irrelevant sign are jointly balanced (eight
    cells; counts differ by at most one). Delay is sampled independently of
    choice from the requested support. For paired algorithms, generate a batch
    once and reuse it. Split RNG creation is deliberately the caller's job.

    ``relevant_coherence`` fixes *signed* relevant evidence for psychometrics;
    zero evidence yields target zero and valid_choice=False, so its choices can
    be measured without inventing a correct response. Never train on these
    ambiguous targets. ``context_rule`` can fix either context for a curve.
    """
    if not isinstance(n_trials, (int, np.integer)) or n_trials <= 0:
        raise ValueError("n_trials must be a positive integer")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("pass an explicit numpy.random.Generator")
    delays = _positive_delays(cfg.delay_bins if delay_bins is None else delay_bins)
    sigma = cfg.noise_std if noise_std is None else float(noise_std)
    distract_sigma = cfg.distractor_std if distractor_std is None else float(distractor_std)
    if not np.isfinite(sigma) or sigma < 0 or not np.isfinite(distract_sigma) or distract_sigma < 0:
        raise ValueError("noise standard deviations must be finite and nonnegative")
    if context_rule is not None and context_rule not in (0, 1):
        raise ValueError("context_rule must be 0, 1, or None")
    if relevant_coherence is not None and not np.isfinite(relevant_coherence):
        raise ValueError("relevant_coherence must be finite")

    contexts = (0, 1) if context_rule is None else (int(context_rule),)
    signs = (-1, 1) if relevant_coherence is None else (1,)
    design = np.asarray(list(product(contexts, signs, (-1, 1))), dtype=int)
    # Randomize the partial block, then the whole batch. A partial block never
    # systematically favors context 0 or the negative response.
    full_blocks, remainder = divmod(n_trials, len(design))
    cells = np.concatenate([np.tile(design, (full_blocks, 1)), design[rng.permutation(len(design))[:remainder]]], axis=0)
    cells = cells[rng.permutation(n_trials)]
    context = cells[:, 0]
    relevant = cells[:, 1] * rng.choice(cfg.coherence_levels, size=n_trials)
    if relevant_coherence is not None:
        relevant = np.full(n_trials, float(relevant_coherence), dtype=np.float64)
    irrelevant = cells[:, 2] * rng.choice(cfg.coherence_levels, size=n_trials)
    if cfg.task_type == "dms":
        # 'context' slot carries the probe identity: 0 -> probe +1, 1 -> probe -1.
        motion, color = relevant, irrelevant
        probe = np.where(context == 0, 1.0, -1.0)
        y = np.sign(relevant) * probe
    else:
        motion = np.where(context == 0, relevant, irrelevant)
        color = np.where(context == 1, relevant, irrelevant)
        y = np.sign(relevant)
    trial_delays = rng.choice(delays, size=n_trials)
    stimulus_start = cfg.t_fixation
    stimulus_end = cfg.t_fixation + cfg.t_stimulus
    context_duration = cfg.t_context if cfg.context_timing == "late" else 0
    decision_start = stimulus_end + trial_delays + context_duration
    trial_end = decision_start + cfg.t_decision
    T = int(np.max(trial_end))
    times = np.arange(T)[None, :]
    valid_mask = times < trial_end[:, None]
    fixation_mask = np.broadcast_to(times < stimulus_start, (n_trials, T)).copy()
    stimulus_mask = np.broadcast_to((times >= stimulus_start) & (times < stimulus_end), (n_trials, T)).copy()
    delay_mask = (times >= stimulus_end) & (times < (stimulus_end + trial_delays)[:, None])
    decision_mask = (times >= decision_start[:, None]) & valid_mask
    if cfg.context_timing == "early":
        context_mask = np.broadcast_to(times < stimulus_end, (n_trials, T)).copy()
    else:
        context_mask = (times >= (stimulus_end + trial_delays)[:, None]) & (times < decision_start[:, None])

    X = np.zeros((n_trials, T, 5), dtype=np.float64)
    X[:, stimulus_start:stimulus_end, 0] = motion[:, None] + rng.normal(0.0, sigma, (n_trials, cfg.t_stimulus))
    X[:, stimulus_start:stimulus_end, 1] = color[:, None] + rng.normal(0.0, sigma, (n_trials, cfg.t_stimulus))
    X[:, :, 2] = context_mask * (context == 0)[:, None]
    X[:, :, 3] = context_mask * (context == 1)[:, None]
    X[:, :, 4] = decision_mask

    # A short, zero-mean sensory distractor in the middle of the memory delay.
    # Its realization is independent of the target; both streams are perturbed.
    distractor_mask = np.zeros((n_trials, T), dtype=bool)
    if distract_sigma > 0:
        widths = np.minimum(trial_delays, cfg.distractor_bins)
        starts = stimulus_end + (trial_delays - widths) // 2
        distractor_mask = (times >= starts[:, None]) & (times < (starts + widths)[:, None])
        distractors = rng.normal(0.0, distract_sigma, (n_trials, T, 2))
        X[:, :, :2] += distractors * distractor_mask[:, :, None]

    meta = {
        "task_type": cfg.task_type,
        "context_rule": context,
        "relevant_coherence": relevant,
        "irrelevant_coherence": irrelevant,
        "motion_coh": motion,
        "color_coh": color,
        "motion_dir": np.sign(motion),
        "color_dir": np.sign(color),
        "is_congruent": (np.sign(motion) == np.sign(color)) & (y != 0),
        "relevant_dir": np.sign(relevant),
        "valid_choice": y != 0,
        "delay_bins": trial_delays,
        "decision_start": decision_start,
        "trial_end": trial_end,
        "noise_std": sigma,
        "distractor_std": distract_sigma,
        "context_timing": cfg.context_timing,
        "bin_size_ms": cfg.bin_size_ms,
    }
    return TrialBatch(X, y, meta, decision_mask, delay_mask, stimulus_mask, valid_mask, context_mask, fixation_mask, distractor_mask)


def decision_values(logits: np.ndarray, batch: TrialBatch) -> np.ndarray:
    """Mean readout over each trial's response window, excluding padding."""
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim == 3 and values.shape[-1] == 1:
        values = values[:, :, 0]
    if values.shape != batch.decision_mask.shape or not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite [trials,time] values matching the batch")
    return (values * batch.decision_mask).sum(axis=1) / batch.decision_mask.sum(axis=1)


def behavior_metrics(logits: np.ndarray, batch: TrialBatch) -> dict[str, float | int]:
    """Fractions (not percentages), counts, and decision-window MSE.

    Ties choose +1 deterministically and their frequency is reported. Coherence
    zero contributes to choice probabilities but never accuracy or target loss.
    Empty subgroups return NaN, not a fabricated zero accuracy.
    """
    values = decision_values(logits, batch)
    prediction = np.where(values >= 0, 1.0, -1.0)
    valid = np.asarray(batch.meta["valid_choice"], dtype=bool)
    correct = prediction == batch.y

    def acc(mask: np.ndarray) -> float:
        selection = valid & mask
        return float(np.mean(correct[selection])) if np.any(selection) else float("nan")

    context = np.asarray(batch.meta.get("context_rule", np.full(batch.n_trials, -1)))
    congruent = np.asarray(batch.meta.get("is_congruent", np.zeros(batch.n_trials)), dtype=bool)
    window_logits = np.asarray(logits)
    if window_logits.ndim == 3:
        window_logits = window_logits[:, :, 0]
    per_trial_mse = 0.5 * (((window_logits - batch.y[:, None]) ** 2) * batch.decision_mask).sum(axis=1) / batch.decision_mask.sum(axis=1)
    class_accuracies = [acc(batch.y == label) for label in (-1.0, 1.0)]
    return {
        "n_trials": batch.n_trials,
        "n_scored": int(valid.sum()),
        "accuracy": acc(np.ones(batch.n_trials, dtype=bool)),
        "balanced_accuracy": float(np.mean(class_accuracies)),
        "accuracy_congruent": acc(congruent),
        "accuracy_incongruent": acc(~congruent),
        "n_congruent": int((valid & congruent).sum()),
        "n_incongruent": int((valid & ~congruent).sum()),
        "accuracy_context0": acc(context == 0),
        "accuracy_context1": acc(context == 1),
        "decision_mse": float(per_trial_mse[valid].mean()) if valid.any() else float("nan"),
        "choice_probability_plus": float(np.mean(prediction > 0)),
        "tie_fraction": float(np.mean(values == 0)),
    }


def phase_mask(batch: TrialBatch, phase: str) -> np.ndarray:
    names = {"all": "valid_mask", "fixation": "fixation_mask", "stimulus": "stimulus_mask", "delay": "delay_mask", "context": "context_mask", "decision": "decision_mask", "distractor": "distractor_mask"}
    if phase not in names:
        raise ValueError(f"unknown phase {phase!r}; choose one of {tuple(names)}")
    return getattr(batch, names[phase])
