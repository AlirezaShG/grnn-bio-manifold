"""Held-out population geometry and finite-horizon dynamical diagnostics.

Fitted axes, feature scaling and readouts use analysis-fit trials only. Their
test statistics use independent analysis-evaluation trials. Trial-time bins are
never treated as independent biological or network replicates.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .task import TrialBatch, phase_mask


def phase_average(rates: np.ndarray, batch: TrialBatch, phase: str = "delay") -> np.ndarray:
    """Return [trial,neuron] averages with per-trial timing and no padding."""
    x = np.asarray(rates, dtype=np.float64)
    if x.ndim != 3 or x.shape[:2] != batch.X.shape[:2] or not np.all(np.isfinite(x)):
        raise ValueError("rates must be finite [trial,time,neuron] values aligned to batch inputs")
    mask = phase_mask(batch, phase)
    counts = mask.sum(axis=1)
    if np.any(counts == 0):
        raise ValueError(f"phase {phase!r} is empty for one or more trials")
    return np.einsum("btn,bt->bn", x, mask) / counts[:, None]


def participation_ratio(features: np.ndarray) -> float:
    """Descriptive linear dimension of between-trial covariance, in [1,N].

    Constant activity has undefined dimension and returns NaN. A single trial
    cannot estimate trial covariance. No significance test is implied.
    """
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("features must be finite [trials,features] with at least two trials")
    s = np.linalg.svd(x - x.mean(axis=0), full_matrices=False, compute_uv=False)
    variances = s * s
    if not np.any(variances > np.finfo(float).eps):
        return float("nan")
    return float(variances.sum() ** 2 / np.square(variances).sum())


@dataclass
class RidgeProbe:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float

    @classmethod
    def fit(cls, features: np.ndarray, targets: np.ndarray, alpha: float = 1.0) -> "RidgeProbe":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if x.ndim != 2 or y.shape != (len(x),) or len(x) < 2:
            raise ValueError("probe fitting requires matching features/targets and at least two trials")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("features/targets must be finite and ridge alpha strictly positive")
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale = np.where(scale > 1e-12, scale, 1.0)
        z = (x - mean) / scale
        intercept = float(y.mean())
        # The objective is mean squared error + alpha * ||w||^2. This keeps
        # alpha's meaning independent of the number of fit trials.
        gram = z.T @ z / len(x) + alpha * np.eye(x.shape[1])
        coefficients = np.linalg.solve(gram, z.T @ (y - intercept) / len(x))
        return cls(mean, scale, coefficients, intercept, float(alpha))

    def predict(self, features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.all(np.isfinite(x)):
            raise ValueError("prediction features do not match the fitted probe")
        return (x - self.mean) / self.scale @ self.coefficients + self.intercept


def _class_axis(features: np.ndarray, y: np.ndarray) -> np.ndarray:
    if not np.any(y > 0) or not np.any(y < 0):
        raise ValueError("both classes are needed to fit a population axis")
    return features[y > 0].mean(axis=0) - features[y < 0].mean(axis=0)


def _signed_dprime(projection: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    positive, negative = projection[y > 0], projection[y < 0]
    if len(positive) < 2 or len(negative) < 2:
        return float("nan"), float("nan")
    pooled_sd = float(np.sqrt(0.5 * (positive.var(ddof=1) + negative.var(ddof=1))))
    dprime = float((positive.mean() - negative.mean()) / pooled_sd) if pooled_sd > 1e-12 else float("nan")
    return dprime, pooled_sd


@dataclass
class PopulationFit:
    phase: str
    neuron_indices: np.ndarray
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_eigenvalues: np.ndarray
    choice_axis: np.ndarray
    context_axis: np.ndarray
    choice_probe: RidgeProbe
    context_probe: RidgeProbe
    n_fit_trials: int
    fit_participation_ratio: float
    # v11: sign of each input stream (motion/sample = channel 0, color/
    # distractor = channel 1), fit on analysis-fit trials only.
    stream_probes: Optional[dict] = None

    def evaluate(self, rates: np.ndarray, batch: TrialBatch, readout: Optional[np.ndarray] = None) -> dict:
        features = phase_average(rates, batch, self.phase)[:, self.neuron_indices]
        choice_valid = np.asarray(batch.meta["valid_choice"], dtype=bool)
        y = batch.y[choice_valid]
        context = np.asarray(batch.meta["context_rule"], dtype=int)
        if context.shape != (len(features),) or not np.all(np.isin(context, (0, 1))):
            raise ValueError("context_rule metadata must contain one 0/1 value per trial")
        axis_norm = np.linalg.norm(self.choice_axis)
        unit = self.choice_axis / axis_norm if axis_norm > 1e-12 else np.zeros_like(self.choice_axis)
        dprime, pooled_sd = _signed_dprime(features[choice_valid] @ unit, y)
        if axis_norm <= 1e-12:
            dprime = float("nan")
        choice_score = self.choice_probe.predict(features[choice_valid])
        choice_pred = np.where(choice_score >= 0, 1.0, -1.0)
        context_pred = np.where(self.context_probe.predict(features) >= 0, 1, 0)
        ctx_norm = np.linalg.norm(self.context_axis)
        angle = float("nan")
        if axis_norm > 1e-12 and ctx_norm > 1e-12:
            cosine = abs(float(self.choice_axis @ self.context_axis / (axis_norm * ctx_norm)))
            angle = float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))

        fit_total = self.pca_eigenvalues.sum()
        centered = features - self.pca_mean
        projected = centered @ self.pca_components.T
        reconstruction = projected @ self.pca_components
        total_energy = float(np.square(centered).sum())
        explained_eval = 1.0 - float(np.square(centered - reconstruction).sum()) / total_energy if total_energy > 1e-12 else float("nan")
        alignment = float("nan")
        if readout is not None:
            w = np.asarray(readout, dtype=np.float64).reshape(-1)
            if len(w) == rates.shape[-1]:
                w = w[self.neuron_indices]
            if w.shape != self.pca_mean.shape:
                raise ValueError("readout must span all cells or the selected neuron subset")
            if np.linalg.norm(w) > 1e-12:
                alignment = float(np.linalg.norm(self.pca_components @ w) / np.linalg.norm(w))
        return {
            "phase": self.phase,
            "n_fit_trials": self.n_fit_trials,
            "n_eval_trials": len(features),
            "n_eval_scored": int(choice_valid.sum()),
            "n_neurons": len(self.neuron_indices),
            "participation_ratio_fit": self.fit_participation_ratio,
            "participation_ratio_eval_descriptive": participation_ratio(features),
            "pca_fit_explained_fraction": float(self.pca_eigenvalues[:len(self.pca_components)].sum() / fit_total) if fit_total > 1e-12 else float("nan"),
            "pca_eval_reconstruction_fraction": explained_eval,
            "choice_dprime_heldout": dprime,
            "choice_projection_pooled_sd": pooled_sd,
            "choice_decoder_accuracy": float(np.mean(choice_pred == y)) if len(y) else float("nan"),
            "context_decoder_accuracy": float(np.mean(context_pred == context)),
            "context_choice_axis_angle_deg_fit": angle,
            "readout_pca_alignment": alignment,
            **self._stream_scores(features, batch),
        }

    def _stream_scores(self, features: np.ndarray, batch: TrialBatch) -> dict:
        scores = {}
        for name, probe in (self.stream_probes or {}).items():
            target = np.sign(np.asarray(batch.meta[name], dtype=float))
            keep = target != 0
            if probe is None or not np.any(keep):
                scores[f"{name}_decoder_accuracy"] = float("nan")
                continue
            pred = np.where(probe.predict(features[keep]) >= 0, 1.0, -1.0)
            scores[f"{name}_decoder_accuracy"] = float(np.mean(pred == target[keep]))
        return scores

    def condition_trajectories(self, rates: np.ndarray, batch: TrialBatch) -> dict[str, list]:
        """Descriptive held-out paths in fitted PC coordinates.

        Time is absolute, so variable-delay trials may leave a condition at
        different times. Counts per time bin accompany each mean; NaN denotes
        no remaining trials. PCA axes carry no assumed choice/context labels.
        """
        x = np.asarray(rates, dtype=np.float64)
        if x.ndim != 3 or x.shape[:2] != batch.X.shape[:2]:
            raise ValueError("rates must align with trial time")
        projected = (x[:, :, self.neuron_indices] - self.pca_mean) @ self.pca_components.T
        result = {}
        for context in (0, 1):
            for choice in (-1, 1):
                trials = (np.asarray(batch.meta["context_rule"]) == context) & (batch.y == choice)
                mask = batch.valid_mask & trials[:, None]
                counts = mask.sum(axis=0)
                summed = np.einsum("btk,bt->tk", projected, mask)
                means = np.full_like(summed, np.nan)
                np.divide(summed, counts[:, None], out=means, where=counts[:, None] > 0)
                result[f"context{context}_choice{choice:+d}"] = {"coordinates": means.tolist(), "n_trials_per_bin": counts.tolist()}
        return result


def fit_population_analysis(
    rates: np.ndarray,
    batch: TrialBatch,
    *,
    neuron_indices: Optional[Sequence[int]] = None,
    phase: str = "delay",
    ridge_alpha: float = 1.0,
    n_components: int = 3,
) -> PopulationFit:
    features_all = phase_average(rates, batch, phase)
    indices = np.arange(features_all.shape[1]) if neuron_indices is None else np.asarray(neuron_indices, dtype=int)
    if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= features_all.shape[1]) or len(np.unique(indices)) != len(indices):
        raise ValueError("neuron_indices must be distinct valid column indices")
    if n_components < 1:
        raise ValueError("n_components must be positive")
    features = features_all[:, indices]
    if len(features) < 2:
        raise ValueError("at least two fitting trials are needed")
    mean = features.mean(axis=0)
    _, s, vt = np.linalg.svd(features - mean, full_matrices=False)
    eigenvalues = s * s / (len(features) - 1)
    choice_valid = np.asarray(batch.meta["valid_choice"], dtype=bool)
    choice_x, y = features[choice_valid], batch.y[choice_valid]
    context = np.asarray(batch.meta["context_rule"], dtype=int)
    if context.shape != (len(features),) or not np.all(np.isin(context, (0, 1))):
        raise ValueError("context_rule metadata must contain 0/1 values")
    context_y = 2.0 * context - 1.0
    stream_probes = {}
    for name in ("motion_dir", "color_dir"):
        if name not in batch.meta:
            continue
        target = np.sign(np.asarray(batch.meta[name], dtype=float))
        keep = target != 0
        stream_probes[name] = (RidgeProbe.fit(features[keep], target[keep], ridge_alpha)
                               if len(np.unique(target[keep])) == 2 else None)
    return PopulationFit(
        phase, indices, mean, vt[:n_components], eigenvalues,
        _class_axis(choice_x, y), _class_axis(features, context_y),
        RidgeProbe.fit(choice_x, y, ridge_alpha), RidgeProbe.fit(features, context_y, ridge_alpha),
        len(features), participation_ratio(features), stream_probes,
    )


def analyze_population(
    rates_fit: np.ndarray,
    batch_fit: TrialBatch,
    rates_eval: np.ndarray,
    batch_eval: TrialBatch,
    *,
    neuron_indices: Optional[Sequence[int]] = None,
    phase: str = "delay",
    readout: Optional[np.ndarray] = None,
    ridge_alpha: float = 1.0,
    n_components: int = 3,
) -> dict:
    """Convenience wrapper; independent fit and evaluation trials are required."""
    if rates_fit is rates_eval or batch_fit is batch_eval:
        raise ValueError("use independent analysis-fit and analysis-evaluation trials")
    fit = fit_population_analysis(rates_fit, batch_fit, neuron_indices=neuron_indices, phase=phase, ridge_alpha=ridge_alpha, n_components=n_components)
    return fit.evaluate(rates_eval, batch_eval, readout=readout)


def perturbation_recovery(
    baseline: np.ndarray,
    perturbed: np.ndarray,
    pulse_mask: np.ndarray,
    endpoint_mask: np.ndarray,
) -> dict:
    """Unclipped finite-horizon displacement after a declared perturbation.

    Inputs may be rates [B,T,N] or full hidden states [B,T,N,K]; all state
    dimensions are flattened. Initial displacement is measured at the final
    pulse bin. Endpoint distance is averaged over the declared later window.
    Values above one indicate expansion; a zero initial displacement produces
    an undefined ratio, not a perfect recovery. This is not an attractor test.
    """
    a, b = np.asarray(baseline, dtype=float), np.asarray(perturbed, dtype=float)
    pulse, endpoint = np.asarray(pulse_mask, dtype=bool), np.asarray(endpoint_mask, dtype=bool)
    if a.shape != b.shape or a.ndim < 3 or pulse.shape != a.shape[:2] or endpoint.shape != a.shape[:2]:
        raise ValueError("matched trajectories and [trial,time] masks are required")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("trajectories must be finite")
    if np.any(pulse.sum(axis=1) == 0) or np.any(endpoint.sum(axis=1) == 0):
        raise ValueError("every trial needs a pulse and an endpoint window")
    times = np.arange(a.shape[1])[None, :]
    last_pulse = np.where(pulse, times, -1).max(axis=1)
    first_end = np.where(endpoint, times, a.shape[1]).min(axis=1)
    if np.any(first_end <= last_pulse):
        raise ValueError("the endpoint window must follow the pulse")
    distances = np.linalg.norm((b - a).reshape(a.shape[0], a.shape[1], -1), axis=-1)
    initial = distances[np.arange(len(a)), last_pulse]
    final = (distances * endpoint).sum(axis=1) / endpoint.sum(axis=1)
    nonzero = initial > 1e-12
    ratios = np.full(len(a), np.nan)
    ratios[nonzero] = final[nonzero] / initial[nonzero]
    ratio_of_means = float(final.mean() / initial.mean()) if initial.mean() > 1e-12 else float("nan")
    return {
        "initial_distance_mean": float(initial.mean()),
        "endpoint_distance_mean": float(final.mean()),
        "endpoint_to_initial_ratio": ratio_of_means,
        "recovery_fraction_unclipped": 1.0 - ratio_of_means,
        "n_trials": len(a),
        "n_zero_initial_displacement": int((~nonzero).sum()),
        "initial_distance_per_trial": initial.tolist(),
        "endpoint_distance_per_trial": final.tolist(),
        "ratio_per_trial": ratios.tolist(),
    }


def temporal_generalization(
    rates_fit: np.ndarray,
    batch_fit: TrialBatch,
    rates_eval: np.ndarray,
    batch_eval: TrialBatch,
    *,
    n_bins: int = 5,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Held-out choice decoding between normalized delay positions.

    Each phase is split into n_bins contiguous nonempty pieces per trial. This
    accommodates variable delays without counting padded time as activity.
    Shorter delays than n_bins are rejected. Ridge scaling is fit at each
    training-time bin using fitting trials only.
    """
    if rates_fit is rates_eval or batch_fit is batch_eval:
        raise ValueError("temporal decoding requires independent fit/evaluation trials")
    if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    def aligned(rates: np.ndarray, batch: TrialBatch) -> np.ndarray:
        rates = np.asarray(rates, dtype=float)
        if rates.ndim != 3 or rates.shape[:2] != batch.X.shape[:2] or not np.all(np.isfinite(rates)):
            raise ValueError("rates must align with the batch")
        if np.any(batch.delay_mask.sum(axis=1) < n_bins):
            raise ValueError("each delay must contain at least n_bins timesteps")
        out = np.empty((batch.n_trials, n_bins, rates.shape[2]))
        for trial, mask in enumerate(batch.delay_mask):
            for b, ids in enumerate(np.array_split(np.flatnonzero(mask), n_bins)):
                out[trial, b] = rates[trial, ids].mean(axis=0)
        return out

    fit_x, eval_x = aligned(rates_fit, batch_fit), aligned(rates_eval, batch_eval)
    fit_valid, eval_valid = batch_fit.y != 0, batch_eval.y != 0
    if len(np.unique(batch_fit.y[fit_valid])) != 2 or not np.any(eval_valid):
        raise ValueError("fitting requires both response classes and evaluation needs scored trials")
    matrix = np.empty((n_bins, n_bins))
    for fit_bin in range(n_bins):
        probe = RidgeProbe.fit(fit_x[fit_valid, fit_bin], batch_fit.y[fit_valid], ridge_alpha)
        for eval_bin in range(n_bins):
            prediction = np.where(probe.predict(eval_x[eval_valid, eval_bin]) >= 0, 1.0, -1.0)
            matrix[fit_bin, eval_bin] = np.mean(prediction == batch_eval.y[eval_valid])
    return matrix
