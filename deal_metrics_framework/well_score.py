"""Well-level scoring utilities: WRS and P_high.

This module implements well-level pieces of the canonical workflow in `run_demo.py`:
- Step 5: Standardize per-well bootstrap feature draws using global refs and compute WRS_raw as a weighted sum of standardized features.
- Step 6: Percentile-clip and scale WRS_raw -> 0-100 and compute per-well summaries (median WRS).
- Step 7: Compute P_high from scaled WRS draws (fraction >= tau).
"""
from typing import Dict, List, Sequence, Tuple
import numpy as np
import pandas as pd


def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    s = sum(weights.values())
    if s == 0:
        raise ValueError("Sum of weights must not be zero")
    return {k: float(v) / s for k, v in weights.items()}


def compute_wrs_raw(z_features: Dict[str, float], v_weights: Dict[str, float]) -> float:
    return sum(v_weights[k] * z_features[k] for k in v_weights.keys())


def scale_to_0_100(raw: float, raw_min: float, raw_max: float) -> float:
    if raw_max - raw_min < 1e-12:
        return 50.0
    scaled = 100.0 * (raw - raw_min) / (raw_max - raw_min)
    return float(min(max(scaled, 0.0), 100.0))


def wrs_from_point(row: pd.Series, refs: Dict[str, Tuple[float, float]], features: Sequence[str], v_weights_raw: Dict[str, float], raw_min: float, raw_max: float) -> float:
    v_weights = normalize_weights(v_weights_raw)
    z = {f: (row[f] - refs[f][0]) / refs[f][1] for f in features}
    raw = compute_wrs_raw(z, v_weights)
    return scale_to_0_100(raw, raw_min, raw_max)


def wrs_from_draws(draws: np.ndarray, refs: Dict[str, Tuple[float, float]], feature_names: Sequence[str], v_weights_raw: Dict[str, float], raw_min: float, raw_max: float) -> np.ndarray:
    """Compute scaled WRS for an array of draws.
    draws: (n_draws, n_features)
    returns: (n_draws,) scaled WRS values
    """
    v_weights = normalize_weights(v_weights_raw)
    # standardize
    zs = (draws - np.array([refs[f][0] for f in feature_names])) / np.array([refs[f][1] for f in feature_names])
    raw = zs.dot(np.array([v_weights[f] for f in feature_names]))
    scaled = 100.0 * (raw - raw_min) / (raw_max - raw_min) if raw_max - raw_min >= 1e-12 else np.full_like(raw, 50.0)
    scaled = np.clip(scaled, 0.0, 100.0)
    return scaled


def p_high_from_wrs_draws(wrs_draws: np.ndarray, tau: float) -> float:
    """Fraction of draws >= tau."""
    return float(np.mean(wrs_draws >= float(tau)))
