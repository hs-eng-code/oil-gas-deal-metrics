"""
Portfolio-level utilities.

Responsibilities mapped to the canonical workflow in `run_demo.py`:
- Step 2: (support) functions for working with joint PV draws produced by the factor simulation (note: the demo implements the factor simulation inline in `run_demo.py`, this module contains helpers for alternate/simple sims).
- Step 8: Compute portfolio-level statistics (P50, mean, std, Var%, VaR_q, CVaR_q) from the joint PV draws for a deal (preserving dependence).
- Step 8/9: Compute effective correlation (rho_eff) and MCVaR contributions used in deal-level features and prioritization.

All functions expect `pv_draws` shaped (n_draws, n_wells) and return pure numerical results suitable for downstream standardization and scoring.
"""
from typing import Dict, List, Tuple
import numpy as np


def corr_matrix_from_draws(pv_draws: np.ndarray) -> np.ndarray:
    # pv_draws: (n_draws, n_wells)
    return np.corrcoef(pv_draws.T)


def rho_eff_from_draws(pv_draws: np.ndarray) -> float:
    """Step 7: Compute effective correlation (rho_eff).

    Return the average off-diagonal correlation coefficient across well PVs.
    """
    C = corr_matrix_from_draws(pv_draws)
    n = C.shape[0]
    if n <= 1:
        return 0.0
    off_diag_sum = C.sum() - np.trace(C)
    return float(off_diag_sum / (n * (n - 1)))


def portfolio_stats_from_draws(pv_draws: np.ndarray, q: float = 0.2) -> Dict[str, float]:
    """Step 7: Compute portfolio-level statistics from joint draws.

    Inputs:
    - pv_draws: array shape (n_draws, n_wells) containing correlated PV draws
    - q: tail quantile used for VaR/CVaR (default 0.20)

    Returns dict with keys: P50, mean, std, Var%, VaR_q, CVaR_q
    """
    # pv_draws: (n_draws, n_wells)
    port = pv_draws.sum(axis=1)
    p50 = float(np.median(port))
    mean = float(np.mean(port))
    std = float(np.std(port, ddof=0))
    var_pct = std / mean if mean != 0 else 0.0
    var_q = float(np.quantile(port, q))
    cvar_q = float(np.mean(port[port <= var_q]))
    return {"P50": p50, "mean": mean, "std": std, "Var%": var_pct, "VaR_q": var_q, "CVaR_q": cvar_q}


def simple_factor_simulation(p50s: List[float], beta_price: List[float], beta_basin: List[float], sigma_idio: List[float], n_draws: int = 10000, price_sigma: float = 0.1, basin_sigma: float = 0.05) -> np.ndarray:
    """Simulate PV draws per well using a simple two-factor Gaussian model for percent changes.
    Returns pv_draws shape (n_draws, n_wells)
    """
    n_wells = len(p50s)
    price_shocks = np.random.normal(0.0, price_sigma, size=n_draws)
    basin_shocks = np.random.normal(0.0, basin_sigma, size=(n_draws, 1))
    # expand basin_shocks to all wells (simple single-basin demo)
    pct_changes = np.outer(price_shocks, beta_price) + basin_shocks.dot(np.array([beta_basin]))
    idio = np.random.normal(0.0, 1.0, size=(n_draws, n_wells)) * np.array(sigma_idio)
    pct_changes = pct_changes + idio
    pv_draws = np.maximum(1e-9, (1.0 + pct_changes)) * np.array(p50s)
    return pv_draws


def mcvare_contributions(pv_draws: np.ndarray, q: float = 0.2, top_n: int = None) -> Dict[int, float]:
    """Step 9: Compute marginal CVaR (MCVaR) contributions for wells.

    For each well i (or top_n wells by proxy) compute:
      MCVaR_i = CVaR_q(port) - CVaR_q(port without well i)

    Returns mapping index -> MCVaR in same units as PV (dollars).
    """
    n_draws, n_wells = pv_draws.shape
    port = pv_draws.sum(axis=1)
    var_q = np.quantile(port, q)
    cvar_full = np.mean(port[port <= var_q])
    indices = range(n_wells)
    if top_n is not None:
        # proxy ranking by mean PV contribution (fast filter)
        means = pv_draws.mean(axis=0)
        order = np.argsort(-means)[:top_n]
        indices = order
    results = {}
    for i in indices:
        port_wo = port - pv_draws[:, i]
        var_q_wo = np.quantile(port_wo, q)
        cvar_wo = np.mean(port_wo[port_wo <= var_q_wo])
        results[int(i)] = float(cvar_full - cvar_wo)
    return results
