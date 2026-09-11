# P01 Environment-Correction Audit

## 1. Executive Summary

P01 decision: **RED**. The analysis used 29 F97 Pro field stations with valid station-labelled EEM and chlorophyll. Raw fluorescence MAE was 7.273; salinity-corrected fluorescence MAE was 8.283, a relative change of -13.88%. The 1000-repeat salinity permutation p-value was 0.6434.

## 2. Research Question

Whether synchronous environmental state can explain and correct drift in limited-channel fluorescence features. This is a measurement-compensation audit, not a field algae-classification study.

## 3. Data Inventory

The source workbook contained 30 field station rows. The frozen manifest is `data_manifest/field_station_manifest.csv`. Raw Excel/EEM files remain outside the repository. F-4700 laboratory files were not merged into this analysis.

## 4. Evidence Boundary

Chlorophyll is an independent validation endpoint. It is not an algae-composition truth label. No result here proves algae species or algae-group classification accuracy.

## 5. Feature Construction

Candidate channels used Ex 370, 470, 525, 570, 590, 600 nm and Em 580, 640, 680, 730 nm, with Em > Ex + 20 nm, finite non-negative values, and a fixed saturation ceiling. Families were raw intensities, total-intensity-normalized shape features, and mechanistic log-ratio features. The same feature rules were applied before LOSO; model scaling and imputation were fitted inside each fold.

## 6. Validation Protocol

Every station was held out once. The training fold alone determined missing-value medians, standardization, RidgeCV alpha, linear environment-drift coefficients, and training-median reference state. The test station was used only for prediction scoring.

## 7. Leakage Audit

- Station-level holdout is explicit and no replicate rows were treated as independent stations.
- The explicit correction fits one linear drift function per channel in the training fold only.
- Reference state `z0` is the training-fold median only.
- The chlorophyll endpoint is never included as a feature or environment variable.
- Environment-only and concatenation models are reported as diagnostics, not patent implementations.
- Channel sensitivity is a descriptive mechanism diagnostic and is not used for feature selection or decision tuning.

## 8. Baselines

`results/baseline_model_comparison.csv` retains fluorescence-only, fluorescence-plus-environment, explicit correction, and environment-only models for all requested environment variables and feature families.

## 9. Environment Variable Screening

The screening table is `results/environment_variable_screening.csv`. Salinity is evaluated as the prespecified primary candidate, not selected after inspecting the endpoint.

## 10. Explicit Environment Correction

For each channel, the fitted relation was `F_j = g_j(z) + residual`, with `F_corr = F - (g_j(z) - g_j(z0))`. Both single-variable and multi-variable correction code paths are represented in the reproducible implementation; the primary gate reports salinity correction and the screening reports the six prespecified single variables.

## 11. Permutation Test

Station salinity labels were permuted 1000 times while EEM and chlorophyll pairing remained fixed. Observed ΔMAE was compared with the complete LOSO correction null distribution. The result is p = 0.6434; it is not treated as significant unless below 0.05.

## 12. Extreme Station Sensitivity

`results/extreme_station_sensitivity.csv` reports removal of the low/high 1–3 stations for chlorophyll, salinity, turbidity, and pH, followed by a new LOSO run. The positive-improvement fraction was 0.3333333333333333.

## 13. Channel-Level Mechanism Analysis

`results/channel_salinity_sensitivity.csv` reports per-channel salinity slopes and standardized coefficients. This is a descriptive diagnostic of possible amplitude, channel-specific, spectral-shape, or ratio drift. It does not establish causality.

## 14. Negative Results

The environment-only salinity MAE was 18.88. The difference between environment-only and corrected fluorescence MAE was 10.59; a small or negative gap would weaken the compensation interpretation.

## 15. Patent-Relevance Assessment

Only a simple, fold-fitted correction operator with stable independent-endpoint improvement can support a P02 implementation study. Feature concatenation alone is not evidence of the claimed compensation mechanism.

## 16. Limitations

The sample is small, field station coverage is limited, EEM instrument response is not cross-calibrated to F-4700, and no synchronous algae-composition truth is available. The station manifest and all exclusions must be preserved with future data revisions.

## 17. P01 Decision

**RED** under the frozen gates in the repository README and config. This status is a data-audit result, not a patentability opinion.

## 18. Recommended P02

If GREEN, freeze the operator, channels, reference-state definition, and implementation examples. If YELLOW, collect independent stations and preregister a confirmatory run before drafting a main claim. If RED, stop the environment-compensation route and retain the negative evidence.

## Claims We CAN Make

- The specified field EEM and environment records can be audited with station-level holdout.
- The data support or do not support a measurable association between environment state and the fluorescence-to-chlorophyll mapping, as quantified above.
- A training-fold-only explicit correction operator was implemented and tested.

## Claims We CANNOT Make

- We cannot claim that environment compensation improves algae species or algae-group classification accuracy.
- We cannot claim field algae-composition truth, causal ecological effects, or quantitative F97/F-4700 cross-instrument calibration.
- We cannot treat concatenating environment variables with fluorescence as the final patent solution.
