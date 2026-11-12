# Pre-implementation inventory (2026-10-08)

Read-only sources: `../machine-learning-trade-sector-prediction.html`, all three files in its named asset folder, `../4-hs-data-analysis/build_model_comparison.py` (actual analytical implementation), saved audits, diagnostics and results; the reference partial-adjustment script in the supplied external archive. The named asset folder contains animation only, not fitted models. Design references: Canada/provinces HTML, CSS and app JS; collection index; Erdős–Rényi cream/card and MathJax conventions.

## Exact names and order

1. Dynamic linear regression: OLS on log1p next-year dollars, 14 standardized economic/geographic/product inputs, HS2 dummies (drop first), plus standardized log1p current trade. Full fit uses 2023 inputs / 2024 target. Dollar inverse is clipped expm1 multiplied by training-total calibration. Missing manufacturing is training-median imputed. HC3 and dyad-cluster diagnostics are descriptive, not identification of causal effects.
2. Elastic Net regression: contemporaneous log1p trade, same 14 inputs, all HS2 dummies, standardized; objective SSE/(2n) + alpha [r L1 + (1-r)L2/2]. Nested leave-exporter-out tuning over alpha .01/.1, r .2/.8, equal-exporter mean log MSE. max_iter 5000, tol 1e-4. Final 2024 fit.
3. Random Forest regression: same contemporaneous log1p target/features; 120 bootstrap squared-error trees, max_depth 16, min_samples_leaf 8, max_features .7. Final 2024 fit.
4. Multilayer Perceptron: same contemporaneous log1p target/features; tanh 32/16, linear output, Adam, L2 alpha .1, batch 512, learning rate .001, max_iter 180, early stopping with random internal 15% validation, patience 15. Original repair replaced unstable ReLU extrapolation using pre-2025 information.
5. Single-stage gradient boosting: histogram gradient boosting on log1p trade including zero, 150 iterations, at most 7 leaves, learning rate .05, scikit-learn automatic early stopping defaults. Final 2024 fit. No classification or lag blending.
6. Dynamic two-stage boosted partial adjustment: histogram binary log-loss classifier on positive/not-positive; histogram squared-error regressor on log(V) ONLY positive flows; separate preprocessing. Positive-dollar calibration c=sum(V+)/sum(exp(f)). Structural potential V*=P(V>0|x) c exp(f(x)). Lambda estimated through origin from 2022 exporter-OOF potential gap and 2022→2023 log1p change, clipped to [0,1]. 2023 exporter-OOF potential and frozen lambda forecast 2024. Final structural stages fit 2024; forecast log1p(V2025)=log1p(V2024)+lambda[log1p(V*2024)-log1p(V2024)]. Same 150/7/.05 tree budget. This is an empirical gravity-feature model, not an estimated structural-gravity PPML system with multilateral resistance or general-equilibrium closure.

All seeds 338. Models 1–5 share training-only dollar-total calibration. No model logs zero with log(V), except the positive-only stage where zero rows are deliberately excluded from that regression and retained in classification.

## Inputs and sources

log exporter/importer GDP (current USD), log population (persons), manufacturing % GDP, internet % population, log population-weighted GeoDist distw km, contig, comlang_off, log1p outside-network exporter-HS4 supply / importer-HS4 demand / world-HS4 demand, and categorical HS2. Outside network means exclude EVERY flow for which both endpoints are in the seven-country network. No tariffs or legal text predictors.

Original network CAN CHN GBR JPN KOR MEX USA (South Korea, not Germany). BACI HS2022 V202601 has 2022–2024 only. Values v*1000, HS6 sums to HS4, absence within valid grid filled zero. WDI copy updated July 2026; manufacturing missing for Canada and USA. GeoDist time invariant. Retrospective years, not reconstructed release vintages.

Original final target is Alberta domestic exports in 2025, StatCan current CAD. 185 eligible destinations × 1229 headings plus disclosed persistence fallbacks; national Canadian macro/geography/supply proxy Alberta; provincial trade lag only in Models 1 and 6. Reporting FX 1.3698 CAD/USD fixed from 2024; realized 2025 1.3978 only ex-post PDF diagnostic. Actual-only cells enter evaluation with prediction zero. Forecasts frozen and hashed before reading Alberta 2025.

## Exact original metrics

Cell MAE=mean abs(pred-actual); Cell WAPE=sum abs(cell error)/sum actual; RMSLE=root mean square log1p difference. Destination WAPE first sums cells by destination then takes absolute errors (cancellation possible). Spearman destination rank correlation; top-N destination overlap (validation N=3, final N=10); mean/median absolute predicted-top10 destination error. Model 6 OOF classification log loss and Brier score. No original RMSE/MSE table; add these explicitly as teaching metrics, with dollars/dollars-squared units.

## Educational decisions

Use preferred Germany instead of South Korea and two sectors after raw-subset inspection. Main table fixed 336 rows. Observed outcomes only 2022–2024; 2025 observed column blank, illustrative scenario separate and flagged. No claims of actual 2025 evaluation or an observed 2025 follow-through. Use 2024 observed temporal validation alongside 2025 scenario comparisons. Keep full feature family. Pool 2022–2024 for final static models (252); dynamic OLS final two transitions (168). Validate OLS on 2023→2024 after fitting 2022→2023 (84); static validate 2024 using 2023 inputs after fitting 2022/23 (168). Elastic selection uses 2022→2023 only. Model 6 retains original OOF lambda chronology and final structural 2024 (84). Smaller 8/4 tanh MLP, fixed max_iter 2000, no random early stopping; this explicit teaching change preserves neural structure and avoids a misleading random temporal validation. Other tree budgets retained. No 2025 covariates or scenario targets in fitting. Do not execute original builder because it writes the protected page.
