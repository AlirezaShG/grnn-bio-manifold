"""Copy of grnn_bio/statistics.py for Python 3.8 (the v11 file is hash-locked and left untouched)."""
from __future__ import annotations


from itertools import product
from typing import Iterable, Optional, Sequence

import numpy as np


def _finite_vector(values: Sequence[float], name: str) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not len(x) or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional sequence")
    return x


def mean_interval(
    values: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Percentile bootstrap CI for a mean, resampling independent runs.

    With fewer than two runs, only a point estimate is returned. With very few
    runs the interval has weak coverage and is explicitly marked as a pilot.
    """
    x = _finite_vector(values, "values")
    if not 0 < confidence < 1 or not isinstance(n_resamples, (int, np.integer)) or n_resamples < 100:
        raise ValueError("confidence must be in (0,1) and n_resamples >= 100")
    result = {"n": len(x), "mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else None, "ci_low": None, "ci_high": None, "confidence": float(confidence), "interval_method": "network-bootstrap-percentile", "small_sample": len(x) < 10}
    if len(x) > 1:
        rng = np.random.default_rng(seed)
        bootstrap = x[rng.integers(0, len(x), size=(n_resamples, len(x)))].mean(axis=1)
        q = (1.0 - confidence) / 2.0
        low, high = np.quantile(bootstrap, [q, 1 - q])
        result["ci_low"], result["ci_high"] = float(low), float(high)
    return result


def paired_comparison(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    seed: int = 0,
    exact_max_pairs: int = 16,
) -> dict:
    """Effect A minus B, paired-run bootstrap CI and two-sided sign-flip p.

    The order of pairs must be identical and known to the caller. With one run,
    no p value or confidence interval is reported. Zero-variance nonzero
    differences have undefined standardized effect, so dz is None.
    """
    a, b = _finite_vector(values_a, "values_a"), _finite_vector(values_b, "values_b")
    if a.shape != b.shape:
        raise ValueError("paired vectors must have the same length")
    differences = a - b
    interval = mean_interval(differences, confidence=confidence, n_resamples=n_resamples, seed=seed)
    result = {
        "n_pairs": len(a),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_difference_a_minus_b": float(differences.mean()),
        "difference_sd": interval["sd"],
        "ci_low": interval["ci_low"],
        "ci_high": interval["ci_high"],
        "confidence": confidence,
        "interval_method": interval["interval_method"],
        "effect_dz": None,
        "p_value": None,
        "test_method": "not_estimable_one_pair",
        "small_sample": interval["small_sample"],
        "differences": differences.tolist(),
    }
    if len(a) < 2:
        return result
    sd = float(differences.std(ddof=1))
    if sd > 0:
        result["effect_dz"] = float(differences.mean() / sd)
    elif np.all(differences == 0):
        result["effect_dz"] = 0.0
    observed = abs(float(differences.mean()))
    tolerance = np.finfo(float).eps * max(1.0, observed) * 16
    if len(a) <= exact_max_pairs:
        # Enumerate every sign assignment; include observed and global negation.
        signs = np.asarray(list(product((-1.0, 1.0), repeat=len(a))))
        null_statistics = abs(signs @ differences / len(a))
        result["p_value"] = float(np.mean(null_statistics >= observed - tolerance))
        result["test_method"] = "exact_paired_sign_flip_two_sided"
        result["n_sign_assignments"] = len(signs)
    else:
        rng = np.random.default_rng(seed + 1)
        signs = rng.choice((-1.0, 1.0), size=(n_resamples, len(a)))
        null_statistics = abs(signs @ differences / len(a))
        # +1 correction for Monte Carlo inference; p can never be zero.
        result["p_value"] = float((1 + np.count_nonzero(null_statistics >= observed - tolerance)) / (1 + n_resamples))
        result["test_method"] = "monte_carlo_paired_sign_flip_two_sided"
        result["n_sign_assignments"] = n_resamples
    return result


def holm_adjust(p_values: Sequence[Optional[float]]) -> list[Optional[float]]:
    """Holm family-wise correction, retaining declared unavailable contrasts.

    Missing values remain None. They still count toward family size, rather
    than making the correction less conservative after a failed experiment.
    """
    p = np.asarray([np.nan if value is None else value for value in p_values], dtype=float)
    if p.ndim != 1 or np.any(np.isfinite(p) & ((p < 0) | (p > 1))) or np.any(np.isinf(p)):
        raise ValueError("p_values must lie in [0,1] or be unavailable")
    available = np.flatnonzero(np.isfinite(p))
    order = available[np.argsort(p[available], kind="stable")]
    output: list[Optional[float]] = [None] * len(p)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p) - rank) * float(p[index]))
        output[int(index)] = min(1.0, running)
    return output


def paired_condition_table(
    rows: Iterable[dict],
    baseline: str,
    metric: str,
    *,
    condition_key: str = "condition",
    pair_key: str = "seed",
    seed: int = 0,
    n_resamples: int = 10_000,
) -> list[dict]:
    """Compare each condition against a reference using exact matching run IDs.

    Filter rows to one prespecified evaluation split and metric family before
    calling. Duplicate/missing pairs raise errors; none are silently dropped.
    Holm correction is applied over the returned condition-vs-reference family.
    """
    groups: dict[str, dict[object, float]] = {}
    for row in rows:
        condition, pair_id = row[condition_key], row[pair_key]
        group = groups.setdefault(condition, {})
        if pair_id in group:
            raise ValueError(f"duplicate run {pair_id!r} in condition {condition!r}; filter to one evaluation split")
        value = float(row[metric])
        if not np.isfinite(value):
            raise ValueError(f"metric {metric!r} is not finite for {condition!r}/{pair_id!r}")
        group[pair_id] = value
    if baseline not in groups:
        raise ValueError("baseline condition is absent")
    pair_ids = sorted(groups[baseline], key=str)
    output = []
    for index, condition in enumerate(sorted(groups)):
        if condition == baseline:
            continue
        if set(groups[condition]) != set(pair_ids):
            raise ValueError(f"condition {condition!r} does not have the same network run IDs as the baseline")
        result = paired_comparison([groups[condition][s] for s in pair_ids], [groups[baseline][s] for s in pair_ids], seed=seed + index, n_resamples=n_resamples)
        output.append({"condition": condition, "reference": baseline, "metric": metric, "pair_ids": pair_ids, **result})
    for row, adjusted in zip(output, holm_adjust([x["p_value"] for x in output])):
        row["p_holm"] = adjusted
        row["family_size"] = len(output)
    return output


def network_level_cell_association(
    rows: Iterable[dict],
    x_key: str,
    y_key: str,
    *,
    network_key: str = "seed",
    cell_key: str = "cell_id",
    seed: int = 0,
) -> dict:
    """Descriptive cell correlations per network and network-bootstrap summary.

    Repeated appearances of one fitted Allen cell within a network are first
    averaged. Network-specific correlations are combined in Fisher-z space.
    There is deliberately no pooled-cell p value. Generalization is conditional
    on this finite fitted-cell dataset; crossed cell/network mixed models are a
    separate extension for biological population inference.
    """
    groups: dict[object, dict[object, list[tuple[float, float]]]] = {}
    for row in rows:
        x, y = float(row[x_key]), float(row[y_key])
        if not np.isfinite(x) or not np.isfinite(y):
            raise ValueError("association input values must be finite")
        groups.setdefault(row[network_key], {}).setdefault(row[cell_key], []).append((x, y))
    per_network = []
    z_values = []
    for network_id in sorted(groups, key=str):
        features = np.asarray([np.mean(observations, axis=0) for observations in groups[network_id].values()])
        valid = len(features) >= 4 and np.std(features[:, 0]) > 0 and np.std(features[:, 1]) > 0
        r = float(np.corrcoef(features.T)[0, 1]) if valid else None
        per_network.append({network_key: network_id, "n_unique_cells": len(features), "correlation": r})
        if r is not None:
            z_values.append(float(np.arctanh(np.clip(r, -1 + 1e-12, 1 - 1e-12))))
    summary = {"n_networks_total": len(groups), "n_networks_valid": len(z_values), "mean_correlation_fisher": None, "ci_low": None, "ci_high": None, "unit_of_inference": "network_run_conditional_on_cell_dataset", "pooled_cell_p_value": None, "per_network": per_network}
    if z_values:
        interval = mean_interval(z_values, seed=seed)
        summary["mean_correlation_fisher"] = float(np.tanh(interval["mean"]))
        if interval["ci_low"] is not None:
            summary["ci_low"] = float(np.tanh(interval["ci_low"]))
            summary["ci_high"] = float(np.tanh(interval["ci_high"]))
        summary["small_sample"] = interval["small_sample"]
    return summary
