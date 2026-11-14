# Wine as an Alternative Asset in Emerging Markets

MATLAB code for:

**Alfeus, M., & Faff, R.** "Benchmarking Emerging-Market Fine-Wine Indices against the Liv-ex 100: Risk, Dependence, and Portfolio Value." *Quantitative Finance* (QMF 2025 Special Issue, under review).

The paper builds monthly, lot-level auction-based price indices for four Southern Hemisphere fine-wine markets — South Africa, Australia, Argentina and Chile — and evaluates their volatility, time-varying dependence, connectedness and portfolio role alongside the Liv-ex Fine Wine 100 and conventional equity, bond and gold benchmarks.

## Data availability

This repository contains **code only**. The underlying auction transaction records are third-party and partly proprietary (South Africa: Strauss & Co. Fine Wine Auctions; Australia/Argentina/Chile: compiled auction records supplied by Dr. Gertjan Verdickt, University of Queensland) and are not redistributed here. De-identified research extracts are available on reasonable request to the corresponding author.

## Requirements

MATLAB with the Econometrics Toolbox (GARCH-family estimation, `estimate`), Optimization Toolbox (`fmincon`, `quadprog`, `linprog`) and Statistics and Machine Learning Toolbox.

## Pipeline

Scripts are grouped below by analytical stage; most file names are self-describing, and the `_4country`/`_18` and `_18`/`_long` suffixes distinguish the primary four-country panel (South Africa, Australia, Argentina, Chile, 2019–2024) from the three-country extended panel (Australia, Argentina, Chile, 2013–2024).

**1. Index construction.** `fogartyJonesHybridRS.m` (hybrid hedonic/repeat-sales estimator), `normalize_lot_currency.m` (lot-level currency conversion), `Build_RS_Hybrid_AllCountries.m` (builds the country indices from raw lots), `Build_MasterData_USD_v2.m` (assembles the panel with benchmark series), `Build_Extra_FX.m`, `Data_Processing.m`, `Compare_RS_Hybrid_AllCountries.m` (repeat-sales vs. hybrid comparison), `Index_Identification_Diagnostics.m`, `Sparse_Month_Disclosure.m`, `AvgPrice_ByCountry.m`, `AuctionHouse_ByCountry.m`, `Monthly_Turnover_LotLevel.m`, `Slippage_PriceImpact_Proxy.m`, `Hedonic_Coefficients_Appendix.m`, `build_PIT_wine_index.m` (point-in-time, no-look-ahead index reconstruction for the backtest).

**2. Volatility model selection.** `fit_auto_vol_models.m` and `fit_tb.m` (GARCH/GJR-GARCH/EGARCH/FIGARCH candidates with a BIC-and-significance selection rule), `figarch_nll_gauss.m` (FIGARCH QML likelihood), `fitIndices_autoModels_v2.m` / `fitIndices_autoModels_v2_4country.m` (model selection driver), `write_full_model_comparison.m`, `write_figarch_only_table.m`, `write_latex_model_table.m` / `write_latex_model_table_all.m`, `gen_modelsel_table.m`, `FIGARCH_Robustness_Check.m` / `_4country.m` (multi-start, independent-Hessian robustness check).

**3. Dynamic dependence (DCC/ADCC).** `rebuild_residualstd.m` / `_4country.m` (standardized residuals from the selected volatility models), `DCC_ADCC_Parameters_4country.m`, `DCC_ADCC_Parameters.m` (extended panel), `DCC_ADCC_Parameters_ARGCHLexcluded_4country.m`, `Generate_DCC_ADCC_Figures*.m`, `unconditional_corr.m`, `Unsmoothing_Robustness_4country.m` (Geltner AR(1) unsmoothing check).

**4. Connectedness (spillover).** `Spillover_Analysis_4country.m`, `Spillover_Analysis.m` (extended panel), `Spillover_Analysis_ARGCHLexcluded_4country.m`, `Spillover_Analysis_SmallSystem_4country.m` / `_LONG.m` / `_HedonicAUS.m` / `_ChainedHedonicAUS.m` (economically motivated small-system connectedness with a moving-block bootstrap).

**5. Portfolio optimization and backtesting.** `Portfolio_Optimization_4country.m` / `_18.m` (static in-sample allocations: GMV, Tangency, Risk Parity, CVaR, Equal-Weight), `Portfolio_Backtest_PointInTime_4country.m` / `_18.m` (rolling, quarterly-rebalanced out-of-sample backtest with point-in-time wine returns), `Compare_WithWithoutWine_4country.m` / `_18.m`, `WithWithoutWine_HighCost_18.m`, `CVaR_Sensitivity_4country.m`, `Capacity_Cap_Sensitivity_4country.m`, `CapacityCap_RollingBacktest_4country.m`, `TransactionCost_Sensitivity_18.m`, `Rebalance_vs_BuyHold_Bootstrap.m`, `SharpeComparison_RiskParity_vs_EqualWeight_18.m`.

**6. Alternative index constructions (Australia).** `Build_HedonicAUS_MonthlyIndex.m` (pooled time-dummy hedonic index), `Build_ChainedHedonicAUS_MonthlyIndex.m` (adjacent-period chained hedonic index), `Apply_ChainedHedonicAUS_ToPanel.m`, `Robustness_HedonicAUS_ModelSelect.m` / `_DCC.m` / `_Spillover.m`, `Robustness_ChainedHedonicAUS_ModelSelect.m` / `_DCC.m`, `Australia_Only_Robustness.m`, `Bilateral_AUSLivex_AltConstruction.m`.

**7. Other diagnostics and utilities.** `Decompose_Interpolation_Currency_Volatility.m`, `Diagnose_MERVAL_Dec2023.m` / `_4country.m`, `SHORT_Window_Interp_Check.m`, `CheckDateRange.m`, `Inspect_Currency_Fields.m` / `2.m`, `Acker_Currency_Sensitivity.m`, `DataCoverage_Liquidity_2019_2024.m`, `descriptiveStats.m`, `realizedVar5.m` / `realizedVar30.m`, `softbox.m`, `Main.m`.

A few early scripts (`Index_Comparizon.m`, `Main_model_check.m`, `DCC_Analysis.m`, `portfolio_paths_tc.m`, `cumulative_returns.m`) predate later corrections to the index construction and are not the source of any table or figure in the paper; they are kept only for provenance. `Build_MasterData_USD.m` and `DCC_Analysis.m` carry an in-file notice identifying their replacement (`Build_MasterData_USD_v2.m` and the scripts in groups 2–4 above, respectively); `Build_MasterData_USD.m` raises an error if run, to avoid overwriting the corrected master dataset.

## Contact

Mesias Alfeus — mesias@sun.ac.za
