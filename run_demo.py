"""
Runnable demo that exercises the modular deal metrics package with synthetic data.

This script implements the pipeline described in the guide in numbered sequential
steps. Each major block below is annotated with the corresponding step number
from the canonical workflow so reviewers can trace code -> doc easily.

Workflow step mapping (high-level):
1) Generate synthetic well metadata (per-well P50, months_on, basin/operator, factor betas, cv_res_proxy) and assemble the input DataFrame.
2) Simulate per-well posterior-like PV draws via a factor model (price, basin, operator common shocks plus idiosyncratic Student-t noise) -> matrix (n_draws x n_wells).
3) Compute per-well bootstrap feature draws (B resamples) to estimate CV_res, sigma_decline (std of log declines), finite-difference sensitivities dPV/dProd and dPV/dPrice, and transformed months_on.
4) Pool bootstrap draws across wells to form global feature references (mu_k, sigma_k) used for Z-scoring.
5) Standardize per-well bootstrap draws using refs and compute WRS_raw per draw as a weighted sum of standardized features; collect the pooled raw distribution for percentile clipping.
6) Percentile-clip the pooled WRS_raw (1st/99th) and scale per-well draws to 0-100; compute per-well summaries (median WRS).
7) Choose tau (empirical percentile, default 75th) from pooled scaled WRS draws and compute per-well P_high = fraction of draws >= tau (alternative: retain continuous P_high).
8) For each deal: slice the joint PV_draws (preserving dependence), compute portfolio stats (P50, Var%, VaR_q, CVaR_q), compute rho_eff, HR_share, ES_gap, and estimate MCVaR contributions (top-N wells).
9) Build deal-level reference statistics (mu, sigma) across generated deals for features: Var%, log_size, hr_share, es_gap, rho_eff.
10) Standardize deal-level features using deal refs and compute DRS_raw as a weighted sum of standardized deal features.
11) Percentile-clip/scale DRS_raw -> DRS_100 (0-100) and map DRS_100 -> recommended underwriting percentile P_rec via the logistic_map (parameters configurable).
12) Diagnostics and logging: print ranges, per-deal summaries, and scaling diagnostics to inspect choices and results.
13) Produce visual outputs: 2D/3D isorisk surface(s), DRS->P_rec mapping plot (with deal overlays), and per-deal WRS density + MCVaR waterfall.

Notes:
- See `deal_metrics_framework.deal_score` for standardization, DRS computation and mapping.
- See `deal_metrics_framework.portfolio` for portfolio-level metrics and MCVaR computation.
"""
import numpy as np
import pandas as pd
from deal_metrics_framework import preprocessing, well_score, portfolio, deal_score, deal_metrics_viz as viz


def make_synthetic_wells(n_wells=100, seed=0, deal_id=1, n_basins=3, n_ops=10):
    rng = np.random.RandomState(seed + deal_id)
    p50 = rng.lognormal(mean=2.0, sigma=0.5, size=n_wells)
    months_on = rng.randint(1, 120, size=n_wells)
    # assign basins and operators
    basin = rng.randint(0, n_basins, size=n_wells)
    operator = rng.randint(0, n_ops, size=n_wells)
    # per-well factor sensitivities
    beta_price = rng.normal(0.5, 0.1, size=n_wells)
    beta_basin = rng.normal(0.1, 0.05, size=n_wells)
    beta_operator = rng.normal(0.05, 0.02, size=n_wells)
    # idiosyncratic volatility proxy
    cv_res = rng.uniform(0.08, 0.6, size=n_wells)

    df = pd.DataFrame({
        "deal_id": [f"D{deal_id}"] * n_wells,
        "well_id": [f"D{deal_id}_W{i+1}" for i in range(n_wells)],
        "p50_pv": p50,
        "months_on": months_on,
        "basin": basin,
        "operator": operator,
        "beta_price": beta_price,
        "beta_basin": beta_basin,
        "beta_operator": beta_operator,
        "cv_res_proxy": cv_res,
    })
    return df


def main():
    # (Doc step 1) Generate synthetic inputs
    #--------------STEP 1--------------
    # Step 1: Generate synthetic well metadata (inputs)
    # - Create a synthetic universe of wells with per-well P50, months_on, basin/operator
    # - This is the input dataset used to build posterior-like draws and per-well features
    # Note: replace `make_synthetic_wells` with a loader for real data when available
    # create synthetic deals: default 100 deals for a larger-sample test; change this variable for other tests
    #-----------------------------------
    n_deals = 30
    wells_per_deal = 200
    wells_list = []
    for d in range(1, n_deals + 1):
        wells_list.append(make_synthetic_wells(n_wells=wells_per_deal, seed=100, deal_id=d))
    wells = pd.concat(wells_list, ignore_index=True)
    
    # compute global refs across the entire universe (all deals)
    # we'll compute feature refs after deriving feature draws from posterior-like simulations
    refs = None
    
    # (Doc step 2) Simulate posterior-like PV draws (factor model)
    #--------------STEP 2--------------
    # Step 2: Simulate per-well posterior-like PV draws (factor model)
    # - Build common shocks (price, basin, operator) and idiosyncratic shocks per well
    # - Combine factor exposures and idiosyncratic terms to produce a matrix of PV draws
    #   with shape (n_draws, n_wells). These draws preserve dependence via shared shocks.
    # Parameters for simulation
    #-----------------------------------
    n_draws = 2000
    # use heavy-tailed shocks (Student-t) for realism
    df_t = 4.0
    price_sigma = 0.12
    basin_sigma = 0.06
    op_sigma = 0.04

    # build common shocks
    rng = np.random.RandomState(2025)
    price_shocks = rng.standard_t(df_t, size=n_draws) * price_sigma

    # Basin shocks for each basin id: to impose correlated behavior among wells that share a basin
    n_basins = int(wells["basin"].max()) + 1
    basin_shocks = {b: rng.standard_t(df_t, size=n_draws) * basin_sigma for b in range(n_basins)}

    # Operator shocks for each operator id: operator-specific performance swings applied to every well that belongs to the same operator during the synthetic PV simulation
    n_ops = int(wells["operator"].max()) + 1
    op_shocks = {o: rng.standard_t(df_t, size=n_draws) * op_sigma for o in range(n_ops)}

    # simulate per-well posterior PV draws matrix (n_draws x n_wells)
    all_p50s = wells["p50_pv"].values
    n_wells = len(all_p50s)
    pv_draws = np.zeros((n_draws, n_wells))
    for i, row in wells.iterrows():
        bp = row["beta_price"]
        bb = row["beta_basin"]
        bo = row["beta_operator"]
        b = int(row["basin"])
        o = int(row["operator"])
        # Idiosyncratic shock: per-well residual noise term
        idio_sigma = max(0.05, row["cv_res_proxy"]) * 0.6
        idio = rng.standard_t(df_t, size=n_draws) * idio_sigma
        perc = bp * price_shocks + bb * basin_shocks[b] + bo * op_shocks[o] + idio
        pv_draws[:, i] = np.maximum(1e-9, (1.0 + perc)) * row["p50_pv"]

    # (Doc step 3) Bootstrap per-well feature draws
    #--------------STEP 3--------------
    # Step 3: Compute per-well bootstrap feature draws
    # - For each well, resample PV draws to estimate per-well features repeatedly
    # - Features: CV_res (std/mean), sigma_decline (std of log declines), finite-diff
    #   sensitivities dPV/dProd and dPV/dPrice, and months_on transform
    # - These B bootstrap samples propagate per-well uncertainty into WRS
    # now compute per-well bootstrap feature draws (B samples each)
    #-----------------------------------
    B = 50
    feature_draws = {"CV_res": [], "sigma_decline": [], "dPV_dProd": [], "dPV_dPrice": [], "months_on": []}
    # finite difference delta for price and production sensitivity
    delta_price = 0.01
    delta_prod = 0.05

    # precompute price-up draws
    price_shocks_up = price_shocks + delta_price
    pv_draws_price_up = np.zeros_like(pv_draws)
    for i, row in wells.iterrows():
        bp = row["beta_price"]
        bb = row["beta_basin"]
        bo = row["beta_operator"]
        b = int(row["basin"])
        o = int(row["operator"])
        idio_sigma = max(0.05, row["cv_res_proxy"]) * 0.6
        # reuse same idio shocks distribution but re-sample to avoid perfect correlation
        idio_up = rng.standard_t(df_t, size=n_draws) * idio_sigma
        perc_up = bp * price_shocks_up + bb * basin_shocks[b] + bo * op_shocks[o] + idio_up
        pv_draws_price_up[:, i] = np.maximum(1e-9, (1.0 + perc_up)) * row["p50_pv"]

    # production-up draws (scale perc by (1+delta_prod))
    pv_draws_prod_up = pv_draws * (1.0 + delta_prod)

    # compute bootstrap samples per well
    rng_boot = np.random.RandomState(54321)
    n_draws_resample = int(n_draws * 0.5)
    per_well_feature_boot = {i: {"CV_res": [], "sigma_decline": [], "dPV_dProd": [], "dPV_dPrice": [], "months_on": []} for i in range(n_wells)}
    for i in range(n_wells):
        base = pv_draws[:, i]
        up_price = pv_draws_price_up[:, i]
        up_prod = pv_draws_prod_up[:, i]
        for bidx in range(B):
            idxs = rng_boot.randint(0, n_draws, size=n_draws_resample)
            sample = base[idxs]
            # CV_res: std/mean
            mu_s = sample.mean()
            sigma_s = sample.std(ddof=0)
            cv = sigma_s / mu_s if mu_s != 0 else 0.0
            # sigma_decline proxy: use std of log declines approximated from consecutive percent changes
            # approximate by std of -log(PV) differences across a small sliding window
            declines = -np.diff(np.log(sample + 1e-12)) if len(sample) > 1 else np.array([0.0])
            sigma_decl = float(np.std(declines)) if len(declines) > 0 else 0.0
            # sensitivities via finite diff on means
            dPV_price = (up_price[idxs].mean() - sample.mean()) / delta_price
            dPV_prod = (up_prod[idxs].mean() - sample.mean()) / (sample.mean() * delta_prod) if sample.mean() != 0 else 0.0
            # transform months_on per guidance: use -log1p(months_on) so higher vintage -> smaller value
            months_on_raw = wells.at[i, "months_on"]
            months_on_val = -np.log1p(float(months_on_raw))
            per_well_feature_boot[i]["CV_res"].append(cv)
            per_well_feature_boot[i]["sigma_decline"].append(sigma_decl)
            per_well_feature_boot[i]["dPV_dProd"].append(dPV_prod)
            per_well_feature_boot[i]["dPV_dPrice"].append(dPV_price)
            per_well_feature_boot[i]["months_on"].append(months_on_val)

    # (Doc step 4) Pool bootstrap draws to form global feature refs
    #--------------STEP 4--------------
    # Step 4: Aggregate bootstrap draws across all wells to compute global refs
    # - Pool the B draws per well into a global reference distribution for each feature
    # - Compute mu and sigma for each feature; these refs are used to standardize features
    # aggregate all bootstrap draws across wells to form global refs
    #-----------------------------------
    all_feature_values = {k: [] for k in feature_draws.keys()}
    for i in range(n_wells):
        for k in all_feature_values.keys():
            all_feature_values[k].extend(per_well_feature_boot[i][k])
    # compute refs mu/sigma for each feature
    refs = {}
    for k, vals in all_feature_values.items():
        arr = np.array(vals)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=0))
        sigma = sigma if sigma > 1e-9 else 1e-9
        refs[k] = (mu, sigma)

    # (Doc step 5) Standardize draws and compute WRS_raw per-draw
    #--------------STEP 5--------------
    # Step 5: Compute per-well WRS_raw draws using standardized features and weights
    # - Standardize each bootstrap feature draw using refs (Z = (x - mu)/sigma)
    # - Combine standardized features with weights v_k to produce WRS_raw per draw
    # - Collect global raw distribution for percentile-based scaling
    # compute WRS draws for each well from its bootstrap feature draws
    #-----------------------------------
    v_weights_raw = {"CV_res": 0.4, "sigma_decline": 0.18, "dPV_dProd": 0.14, "dPV_dPrice": 0.08, "months_on": 0.2}
    # build list of all WRS_raw values to compute global raw min/max (percentile-clipped)
    all_wrs_raw = []
    per_well_wrs_draws = {}
    for i in range(n_wells):
        feat_mat = np.vstack([per_well_feature_boot[i]["CV_res"], per_well_feature_boot[i]["sigma_decline"], per_well_feature_boot[i]["dPV_dProd"], per_well_feature_boot[i]["dPV_dPrice"], per_well_feature_boot[i]["months_on"]]).T
        # standardize using refs
        zs = (feat_mat - np.array([refs[k][0] for k in ["CV_res", "sigma_decline", "dPV_dProd", "dPV_dPrice", "months_on"]])) / np.array([refs[k][1] for k in ["CV_res", "sigma_decline", "dPV_dProd", "dPV_dPrice", "months_on"]])
        weights = np.array([v_weights_raw[k] for k in ["CV_res", "sigma_decline", "dPV_dProd", "dPV_dPrice", "months_on"]])
        weights = weights / weights.sum()
        wrs_raw = zs.dot(weights)
        per_well_wrs_draws[i] = wrs_raw
        all_wrs_raw.extend(list(wrs_raw))

    # populate per-well CV_res (median of bootstrap draws) into wells DataFrame for downstream use
    cv_medians = [np.median(per_well_feature_boot[i]["CV_res"]) for i in range(n_wells)]
    wells["CV_res"] = cv_medians

    wrs_raw_min = float(np.percentile(all_wrs_raw, 1.0))
    wrs_raw_max = float(np.percentile(all_wrs_raw, 99.0))
    if wrs_raw_max - wrs_raw_min < 1e-12:
        wrs_raw_min = float(min(all_wrs_raw))
        wrs_raw_max = float(max(all_wrs_raw))

    # (Doc step 6) Percentile-clip and scale WRS_raw -> 0-100; compute medians
    #--------------STEP 6--------------
    # Step 6: Scale WRS draws to 0-100 and compute per-well summaries
    # - Use global 1st/99th percentile clipping of raw WRS to map raw->0-100
    # - Store per-well median WRS for reporting and diagnostics
    # compute per-well scaled WRS medians and P_high from bootstrap feature draws
    #-----------------------------------
    wells["WRS_median"] = 0.0
    wells["P_high"] = 0.0
    # scale draws to 0-100
    per_well_wrs_scaled = {}
    for i in range(n_wells):
        wrs_scaled = 100.0 * (per_well_wrs_draws[i] - wrs_raw_min) / (wrs_raw_max - wrs_raw_min)
        wrs_scaled = np.clip(wrs_scaled, 0.0, 100.0)
        per_well_wrs_scaled[i] = wrs_scaled
        wells.at[i, "WRS_median"] = float(np.median(wrs_scaled))

    # (Doc step 7) Choose tau and compute per-well P_high
    #--------------STEP 7--------------
    # Step 7: Compute tau (empirical threshold) and P_high for each well
    # - Tau is chosen as a global percentile (default 75th) of pooled scaled WRS draws
    # - P_high_i = fraction of scaled draws for well i that exceed tau (a probability in [0,1])
    # - Alternative: keep continuous P_high without threshold to avoid binarization
    # set tau to an empirical percentile (e.g., 75th) of the global scaled WRS distribution to avoid a magic number
    #-----------------------------------
    global_wrs_all = np.concatenate([per_well_wrs_scaled[i] for i in range(n_wells)])
    tau_percentile = 75.0
    tau = float(np.percentile(global_wrs_all, tau_percentile))
    for i in range(n_wells):
        wells.at[i, "P_high"] = float(np.mean(per_well_wrs_scaled[i] >= tau))

    # (Doc step 8) Slice joint draws per deal and compute portfolio stats + MCVaR
    #--------------STEP 8--------------
    # Step 8: For each deal compute portfolio-level statistics and MCVaR
    # - Slice the joint PV_draws matrix to preserve dependence for portfolio-level metrics
    # - Compute P50, Var%, VaR_q, CVaR_q, rho_eff, ES_gap
    # - Compute MCVaR per well by recomputing CVaR with that well removed (top_n optimization allowed)
    # now compute per-deal metrics and collect deal-level raw features
    #-----------------------------------
    deal_summaries = []
    # Use the dependent pv_draws we already simulated above. We'll slice columns for each deal
    # so portfolio metrics respect the joint dependence produced earlier.
    for deal_id, group in wells.groupby("deal_id"):
        indices = list(group.index)
        # slice the global pv_draws matrix for these wells
        pv_draws_deal = pv_draws[:, indices]
        stats = portfolio.portfolio_stats_from_draws(pv_draws_deal)
        rho_eff = portfolio.rho_eff_from_draws(pv_draws_deal)
        # follow the doc: use Var%, log(Size), HR_share, ES_gap, rho_eff
        size_mm = float(group["p50_pv"].sum())
        log_size = float(np.log(size_mm + 1e-9))
        hr_share = float((group["P_high"] * group["p50_pv"]).sum() / (group["p50_pv"].sum() + 1e-12))
        es_gap = (stats["P50"] - stats["CVaR_q"]) / (stats["P50"] + 1e-12)
        # compute MCVaR contributions for top 10 wells and map to well ids
        mcv_idx = portfolio.mcvare_contributions(pv_draws_deal, q=0.2, top_n=10)
        # map integer column indices (relative to pv_draws_deal) to well ids
        mcv = {}
        deal_well_ids = list(group["well_id"].values)
        for idx_rel, val in mcv_idx.items():
            try:
                well_id = deal_well_ids[int(idx_rel)]
            except Exception:
                well_id = f"unknown_{idx_rel}"
            mcv[well_id] = val
        deal_summaries.append({"deal_id": deal_id, "n_wells": len(deal_well_ids), "size_mm": size_mm, "log_size": log_size, "Var%": stats["Var%"], "hr_share": hr_share, "es_gap": es_gap, "rho_eff": rho_eff, "mcv": mcv})
    
    # (Doc step 9) Build deal-level reference statistics (mu, sigma)
    #--------------STEP 9--------------
    # Step 9: Build deal-level references (mu, sigma) across generated deals
    # - Compute mean and population std for each deal-level feature to standardize later
    #-----------------------------------
    deal_feature_names = ["Var%", "log_size", "hr_share", "es_gap", "rho_eff"]
    # compute mu/sigma across deals
    deal_refs = {}
    for f in deal_feature_names:
        vals = np.array([d[f] for d in deal_summaries])
        mu = float(vals.mean())
        sigma = float(vals.std(ddof=0))
        sigma = sigma if sigma > 1e-9 else 1e-9
        deal_refs[f] = (mu, sigma)

    # (Doc step 10) Standardize deal features and compute DRS_raw
    #--------------STEP 10--------------
    # Step 10: Compute DRS_raw per deal using standardized deal features and weights
    # - Standardize deal features (Var%, log_size, HR_share, ES_gap, rho_eff)
    # - Combine standardized features with weights w_m to produce DRS_raw
    # scale DRS_raw to 0-100 using observed min/max across generated deals (avoid placeholder saturation)
    # build deal-level refs (mu, sigma) from generated deals (ddof=0 population std as in doc)
    #-----------------------------------
    # compute drs_raw for each deal using standardized deal features and weights
    w_weights = {"Var%": 0.35, "log_size": 0.30, "hr_share": 0.15, "es_gap": 0.15, "rho_eff": 0.05}
    drs_raw_list = []
    for d in deal_summaries:
        feat = {f: d[f] for f in deal_feature_names}
        z = deal_score.standardize_deal_features(feat, deal_refs, deal_feature_names)
        drs_raw = deal_score.compute_drs_raw(z, w_weights)
        d["drs_raw"] = drs_raw
        drs_raw_list.append(drs_raw)

    # scale DRS_raw to 0-100 using percentile clipping (1st/99th) to avoid outliers
    drs_min = float(np.percentile(drs_raw_list, 1.0))
    drs_max = float(np.percentile(drs_raw_list, 99.0))
    if drs_max - drs_min < 1e-12:
        drs_min = float(min(drs_raw_list))
        drs_max = float(max(drs_raw_list))

    # (Doc step 11) Scale DRS_raw -> DRS_100 and map to P_rec
    #--------------STEP 11--------------
    # Step 11: Scale DRS_raw -> DRS_100 and map to P_rec
    # - Use percentile clipping (1st/99th) to scale raw to 0-100
    # - Map DRS_100 to a recommended underwriting percentile P_rec via logistic_map
    # assign DRS_100 and P_rec into each deal record
    #-----------------------------------
    drs_values = []
    for d in deal_summaries:
        drs_100 = deal_score.scale_raw_to_0_100(d["drs_raw"], drs_min, drs_max)
        p_rec = deal_score.logistic_map(drs_100, gamma=0.05, theta=50.0, p50=50.0, p_max=80.0)
        d["DRS_100"] = drs_100
        d["P_rec"] = p_rec
        drs_values.append(drs_100)

    # (Doc step 12) Diagnostics and logging
    #--------------STEP 12--------------
    # Step 12: Diagnostics and logging
    # - Print ranges and a short summary per deal to help inspect scaling and mapping choices
    # diagnostic: how many deals computed + report DRS_raw and DRS_100
    #-----------------------------------
    print(f"Computed {len(deal_summaries)} deals: {[d['deal_id'] for d in deal_summaries]}")
    print(f"DRS raw range used for scaling: drs_min={drs_min:.6f}, drs_max={drs_max:.6f}")
    for d in deal_summaries:
        drs_raw = float(d["drs_raw"])
        drs_100 = deal_score.scale_raw_to_0_100(drs_raw, drs_min, drs_max)
        p_rec = deal_score.logistic_map(drs_100, gamma=0.05, theta=50.0, p50=50.0, p_max=80.0)
        print(f"Deal {d['deal_id']}: wells={d['n_wells']}, P50_sum={d['size_mm']:.2f}, Var%={d['Var%']:.3f}, HR_share={d['hr_share']:.3f}, rho_eff={d['rho_eff']:.3f}, DRS_raw={drs_raw:.6f}, DRS_100={drs_100:.2f}, P_rec={p_rec:.1f}")

    # (Doc step 13) Produce visual outputs
    #--------------STEP 13--------------
    # Step 13: Visualization outputs
    # - Produce executive isorisk plots (2D, 3D, combined), the DRS->P_rec mapping plot,
    #   and per-deal WRS density + MCVaR waterfall for a deep-dive slide.
    # save isorisk plot
    #-----------------------------------
    try:
        p2d = viz.plot_isorisk(deal_summaries, out_path="outputs/isorisk.png")
        print(f"Saved isorisk plot to {p2d}")
    except Exception as e:
        print("Failed to save 2D isorisk plot:", e)

    try:
        p3d = viz.plot_isorisk_3d(deal_summaries, out_path="outputs/isorisk_3d.png")
        print(f"Saved 3D isorisk plot to {p3d}")
    except Exception as e:
        print("Failed to save 3D isorisk plot:", e)

    try:
        p_comb = viz.plot_isorisk_combined(deal_summaries, out_path="outputs/isorisk_combined.png", use_log_size=True, grid_res=120)
        print(f"Saved combined isorisk figure to {p_comb}")
    except Exception as e:
        print("Failed to save combined isorisk figure:", e)

    try:
        p_map = viz.plot_drs_to_prec(deal_summaries, out_path="outputs/drs_to_prec.png", gamma=0.05, theta=50.0, p50=50.0, p_max=80.0)
        print(f"Saved DRS->P_rec mapping plot to {p_map}")
    except Exception as e:
        print("Failed to save DRS->P_rec mapping plot:", e)

    # create per-deal WRS density + MCVaR for the first deal
    try:
        first = deal_summaries[0]
        out_file = viz.plot_deal_wrs_and_mcvare(wells, per_well_wrs_scaled, first, out_dir="outputs", top_n=10)
        print(f"Saved deal WRS+MCVaR plot to {out_file}")
    except Exception as e:
        print("Failed to save deal WRS+MCVaR plot:", e)
    
    return wells, pv_draws, deal_summaries, drs_raw_list, drs_values


if __name__ == "__main__":
    wells, pv_draws, deal_summaries, drs_raw_list, drs_values = main()
