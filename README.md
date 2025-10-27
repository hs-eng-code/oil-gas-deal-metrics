# Deal Metrics Framework

This folder contains a modular Python demo for evaluating oil & gas deals using Deal Risk Score (DRS).

Files added:
- `deal_metrics_framework/:` Small package with modules: preprocessing, well_score, portfolio, deal_score, utils, deal_metrics_viz.
- `run_demo.py:` Runnable example that builds synthetic wells, computes WRS/P_high, runs a factor simulation and computes DRS and P_rec.

How to run:

```
python run_demo.py
```

Notes:
- Replace synthetic draws with your posterior draw CSVs and wire the real global refs for production use.

# Workflow Analysis

## Purpose and Context
- `run_demo.py` stitches together the 13-step canonical workflow described in the inline header comments to exercise the modular deal metrics package on synthetic data.
- The script fabricates a portfolio of deals, simulates correlated PV draws, derives well-level and deal-level features, scores them, and emits diagnostics plus visualizations. It relies on utilities in `deal_metrics_framework` for scoring, portfolio statistics, and plotting.

## High-Level Workflow
1. Build synthetic well metadata per deal (Step 1).
2. Simulate correlated PV draws via a factor model (Step 2).
3. Bootstrap per-well features from the draws (Step 3).
4. Pool feature draws to compute global references (Step 4).
5. Standardize features and compute well risk score draws (Step 5).
6. Percentile clip and scale WRS draws, summarize per well (Step 6).
7. Select a global threshold tau and compute `P_high` probabilities (Step 7).
8. Aggregate per deal, compute portfolio stats and MCVaR (Step 8).
9. Build deal-level reference stats (Step 9).
10. Standardize deal features and get raw DRS (Step 10).
11. Scale DRS to 0-100 and map to recommended percentile (Step 11).
12. Emit diagnostics about scaling and scores (Step 12).
13. Produce visualization outputs for isorisk surfaces and deal deep dives (Step 13).

## Detailed Step-by-Step Mechanics

### Step 1: Synthetic Deal Universe
- `make_synthetic_wells` fabricates P50 PVs, months-on production, categorical basin/operator flags, factor betas, and idiosyncratic CV proxies using a seeded `RandomState` (`run_demo.py`).
- `main` loops over `n_deals = 30` with `wells_per_deal = 200`, concatenating the results so that each deal shares a deterministic seed offset (`run_demo.py`).
- Output is a DataFrame with 6k wells, each labeled `deal_id`/`well_id`, and no standardization is applied yet.

### Step 2: Factor-Based PV Simulation
- Student-t shocks with 4 degrees of freedom emulate heavy-tailed behavior for price, basin, and operator factors plus idiosyncratic residuals (`run_demo.py`).
- Each well's percent change is assembled as `beta_price*price_shock + beta_basin*basin_shock + beta_operator*operator_shock + idio`, then scaled by the well's P50 (guarded by `np.maximum(1e-9, ...)`) to form `pv_draws` shaped `(n_draws, n_wells)` with `n_draws = 2000`.

### Step 3: Bootstrap Feature Extraction
- Finite-difference price and production shocks are prepared (`delta_price = 0.01`, `delta_prod = 0.05`) to approximate sensitivities (`run_demo.py`).
- For each well, the script draws 1000 resampled draws (`n_draws_resample = n_draws * 0.5`) per bootstrap iteration `B = 50`, computing CV of PV, log-decline volatility, price/production derivatives, and a monotonic `-log1p(months_on)` transform (`run_demo.py`).
- `feature_draws` is only used to determine feature keys when pooling results, while the actual values live in `per_well_feature_boot`.

### Step 4: Global Feature References
- Bootstrap samples are concatenated across wells to compute population mean and std pairs `(mu, sigma)` per feature with an epsilon guard on small variance (`run_demo.py`).
- This mirrors what `preprocessing.compute_global_refs` would offer (`deal_metrics_framework/preprocessing.py`), albeit implemented inline.

### Step 5: Well Risk Score per Draw
- For each well, bootstrap features are stacked, standardized using the reference stats, and combined with normalized weights `v_weights_raw` to yield raw WRS draws (`run_demo.py`).
- Weighted contributions emphasize `CV_res` (40%) and months-on (20%), with other features filling the remainder.

### Step 6: Scaling and Medians
- Global pooled WRS draws determine 1st/99th percentile bounds for clipping; each well's WRS draws are rescaled to 0-100 and median values stored on the `wells` DataFrame (`run_demo.py`).

### Step 7: High-Probability Wells
- Tau is the 75th percentile of all scaled WRS draws, and each well's `P_high` is the share of its draws that meet or exceed tau (`run_demo.py`).

### Step 8: Deal Portfolio Metrics & MCVaR
- For every `deal_id`, the code slices `pv_draws` to keep factor dependence, computes portfolio statistics via `portfolio.portfolio_stats_from_draws` and `rho_eff_from_draws` (`run_demo.py`, `deal_metrics_framework/portfolio.py`).
- High-risk share weighs each well's `P_high` by its PV, `es_gap` measures the gap between P50 and CVaR, and `portfolio.mcvare_contributions` returns marginal tail impacts for the top contributors (`deal_metrics_framework/portfolio.py`).

### Step 9: Deal-Level References
- Means and population stds across simulated deals create reference tuples for Var%, log size, high-risk share, ES gap, and rho (`run_demo.py`).

### Step 10: Raw Deal Risk Score
- Each deal's features are z-scored via `deal_score.standardize_deal_features` and combined with normalized weights (0.35 on Var%, 0.30 on log size, etc.) to form `drs_raw` values (`run_demo.py`, `deal_metrics_framework/deal_score.py`).

### Step 11: Scaling and Mapping to `P_rec`
- Raw scores are percentile-scaled to `[0, 100]` using `deal_score.scale_raw_to_0_100`, then mapped to underwriting recommendations with the logistic curve in `deal_score.logistic_map` (`run_demo.py`, `deal_metrics_framework/deal_score.py`).
- The logistic parameters `gamma=0.05`, `theta=50`, `p50=50`, `p_max=80` produce a gentle sigmoid anchored at mid-risk.

### Step 12: Diagnostics
- The script reports the deals processed, scaling window, and per-deal metrics plus derived scores to stdout for quick calibration review (`run_demo.py`).

### Step 13: Visualization Stage
- Visual outputs are requested inside a `try` block: 2D contour, 3D surface, and combined surface + contour (all writing under `outputs/`) using helpers in `deal_metrics_framework/deal_metrics_viz.py` (`run_demo.py`).
- On plot failure, the exception path attempts only the DRS-to-P_rec curve with `viz.plot_drs_to_prec`; as written the logistic plot never runs when the earlier figures succeed because it is nested inside the `except` block (`run_demo.py`).
- Regardless of earlier errors, the code tries to build a per-deal WRS density plus MCVaR waterfall for the first deal (`run_demo.py`, `deal_metrics_framework/deal_metrics_viz.py`).

## Module Roles and Usage
- `deal_metrics_framework/deal_score.py` houses the reusable z-scoring, weighting, scaling, and logistic mapping used in Steps 10-11 (`deal_metrics_framework/deal_score.py`).
- `deal_metrics_framework/portfolio.py` provides dependence-aware portfolio metrics and marginal CVaR analysis invoked during Step 8 (`deal_metrics_framework/portfolio.py`).
- `deal_metrics_framework/deal_metrics_viz.py` centralizes visualization logic for isorisk overlays, DRS mapping, and deal diagnostics (`deal_metrics_framework/deal_metrics_viz.py`).
- `deal_metrics_framework/preprocessing.py` and `deal_metrics_framework/well_score.py` expose general-purpose helpers for global references and well-level scoring (`deal_metrics_framework/preprocessing.py`, `deal_metrics_framework/well_score.py`), although the demo re-implements equivalent logic inline and never calls them directly.

## Outputs and Diagnostics
- Console diagnostics enumerate deal counts and key per-deal metrics to help verify scaling decisions before inspecting visuals (`run_demo.py`).
- On success the script writes `outputs/isorisk.png`, `outputs/isorisk_3d.png`, `outputs/isorisk_combined.png`, optional `outputs/drs_to_prec.png`, and `outputs/deal_D1_wrs_mcvare.png` (first deal id), all created via the visualization module (`run_demo.py`, `deal_metrics_framework/deal_metrics_viz.py`).
- Returned objects (`wells`, `pv_draws`, `deal_summaries`, `drs_raw_list`, `drs_values`) give callers access to the full simulated datasets for further inspection (`run_demo.py`).