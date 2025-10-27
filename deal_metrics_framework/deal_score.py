"""
Deal-level scoring utilities.

This module contains the pure functions used by the demo to standardize deal features, compute raw DRS, rescale to 0-100, and map the final DRS_100 to a recommended underwriting percentile (`P_rec`). Each function is annotated with the workflow step it implements so the code is auditable against the design doc.

Workflow annotations (in-file):
- Step 10: standardize deal-level features using pooled historical refs (mu/sigma)
- Step 11: compute DRS_raw as weighted sum of standardized deal features
- Step 12: scale DRS_raw -> DRS_100 via percentile clipping or historical range
- Step 13: logistic_map maps DRS_100 -> P_rec (recommended underwriting percentile)
"""
from typing import Dict, Sequence
import numpy as np


def standardize_deal_features(deal_features: Dict[str, float], refs: Dict[str, tuple], feature_list: Sequence[str]) -> Dict[str, float]:
    """Step 10: Standardize deal-level features.

    Inputs:
    - deal_features: mapping feature_name -> raw value for a single deal
    - refs: mapping feature_name -> (mu, sigma) from historical deals
    - feature_list: ordered list of features to produce standardized outputs

    Output: mapping feature_name -> z_score
    """
    out = {}
    for f in feature_list:
        mu, sigma = refs[f]
        sigma = sigma if sigma > 0 else 1e-9
        out[f] = (deal_features[f] - mu) / sigma
    return out


def compute_drs_raw(z_features: Dict[str, float], w_weights: Dict[str, float]) -> float:
    """Step 11: Compute raw DRS as weighted sum of standardized deal features.

    This function normalizes raw weights to sum to 1 and returns the scalar
    DRS_raw value (unscaled). Keep weights explicit and auditable.
    """
    s = sum(w_weights.values())
    if s == 0:
        raise ValueError("Sum of weights is zero")
    w_norm = {k: float(v) / s for k, v in w_weights.items()}
    return float(sum(w_norm[k] * z_features[k] for k in w_norm.keys()))


def scale_raw_to_0_100(raw: float, raw_min: float, raw_max: float) -> float:
    """Step 12: Rescale a raw scalar to the 0-100 range using raw_min/raw_max.

    Notes:
    - Uses linear mapping between raw_min -> 0 and raw_max -> 100.
    - Clamps results to [0,100]. If raw_max == raw_min returns midpoint 50.0.
    """
    if raw_max - raw_min < 1e-12:
        return 50.0
    val = 100.0 * (raw - raw_min) / (raw_max - raw_min)
    return float(min(max(val, 0.0), 100.0))


def logistic_map(drs_100: float, gamma: float = 0.05, theta: float = 50.0, p50: float = 50.0, p_max: float = 80.0) -> float:
    """Step 13: Map DRS_100 (0-100) to P_rec (underwriting percentile) using logistic.

    Parameters:
    - gamma: slope/steepness of logistic curve
    - theta: midpoint in DRS_100 space where sigmoid = 0.5
    - p50: lower asymptote (baseline percentile)
    - p_max: upper cap for recommended percentile

    Output: percentile value (e.g. 50..80). Adjust gamma/theta to reflect risk appetite.
    """
    x = gamma * (drs_100 - theta)
    s = 1.0 / (1.0 + np.exp(-x))
    return float(p50 + (p_max - p50) * s)
