"""Preprocessing utilities: standardization and reference snapshots.

This module implements the preprocessing pieces referenced in the canonical workflow (see `run_demo.py`). Mapping to the canonical steps:
- Step 4: Pool bootstrap draws across wells to form global feature references
    (mu_k, sigma_k) used for Z-scoring.
- Step 10: Utility functions used when standardizing deal-level or well-level
    features (shared z-score logic).
"""
from typing import Dict, Sequence, Tuple
import numpy as np
import pandas as pd


def compute_global_refs(df: pd.DataFrame, features: Sequence[str]) -> Dict[str, Tuple[float, float]]:
    """Compute mu and sigma (ddof=0) for features. Guard sigma with eps."""
    refs = {}
    for f in features:
        s = df[f].dropna()
        mu = float(s.mean()) if len(s) > 0 else 0.0
        sigma = float(s.std(ddof=0)) if len(s) > 1 else 0.0
        sigma = sigma if sigma > 0 else 1e-9
        refs[f] = (mu, sigma)
    return refs


def z_score(v: float, mu: float, sigma: float) -> float:
    return (v - mu) / sigma


def standardize_row(row: pd.Series, refs: Dict[str, Tuple[float, float]], features: Sequence[str]) -> Dict[str, float]:
    out = {}
    for f in features:
        mu, sigma = refs[f]
        out[f] = z_score(row[f], mu, sigma)
    return out
