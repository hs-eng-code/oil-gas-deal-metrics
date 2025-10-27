# Deal Risk Score Workflow Flowchart

```mermaid
flowchart TD
    subgraph WellLevel[**EACH DEAL-LEVEL WELL PIPELINE**]
        A1["STEP 1<br/>**Well data & metadata** (well_id, basin_id, months_on, p50_pv, posterior_draws (PV or production))"] --> A2["STEP 2<br/>**Construct perturbed PV matrices from price and production for finite-difference sensitivities** (dPV_dProd, dPV_dPrice)"]
        A2 --> A3["STEP 3<br/>**Bootstrap per-well features: Z_k_j(i) for each feature k**<br/>(1. CV_res, 2. sigma_decline (std of log declines), 3. dPV_dProd, 4. dPV_dPrice, 5. months_on (-log1p(months_on) so higher vintage -> smaller value))"]
        A3 --> A4["STEP 4<br/>**Global feature refs (mu, sigma)**"]
        A4 --> A5["STEP 5<br/>**a. Standardize well features** (Z(x) = (x - mu_x) / sigma_x) <br/>**b. Compute WRS_raw draws** (WRS_raw_j(i) = sum_k (v_k * Z_k_j(i)), where v_k is the weight given to each feature Z_k, totaling to 1)"]
        A5 --> A6["STEP 6<br/>**Scale WRS (-> 0-100) & WRS_median**"]
        A6 --> A7["STEP 7<br/>**Compute P_high**<br/>(fraction of draws where the well's risk metric exceeds the global threshold) per well"]
    end

    subgraph DealLevel[**GLOBAL DEALS AGGREGATION-LEVEL PIPELINE**]
        B1["STEP 8<br/>**Slice PV draws per deal**<br/>(CV, log of deal size, expected high-risk P50 share for the deal (HR_share), expected shortfall gap (es_gap), effective correlation (rho_eff))"] --> B2["STEP 9<br/>**Deal feature reference stats (mu, sigma)**"]
        B2 --> B3["STEP 10<br/>**a. Standardize deal features** (Z(x) = (x - mu_x) / sigma_x)<br/>**b. Compute DRS_raw** (Z_m(deal): DRS_raw = sum_m w_m * Z_m(deal) (i.e., w1Z1 + w2Z2 + ...))"]
        B3 --> B4["STEP 11<br/>**a. Scale DRS -> DRS_100**<br/>**b. Logistic map to P_rec**"]
    end

    A7 -->|Augmented wells DataFrame| B1
    A2 -->|Joint PV_draws matrix| B1
```

- Well-level pipeline produces enriched `wells` data (WRS medians, `P_high`) and shared `pv_draws` for each deal.
- Deal-level pipeline ingests well outputs to compute deal summaries, standardized scores, and recommended underwriting percentiles.
