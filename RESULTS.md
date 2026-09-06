# Collateral-side residualization across four model/SAE settings

Companion results for the revision of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Duan, arXiv:2606.08365). All numbers below are read from `results/` by `analysis/make_results.py`; nothing is copied from the paper except where labelled as the paper's value.

Settings included: GPT-2-small, Pythia-70M-deduped, Gemma-2-2B, Llama-3.1-8B.

## Summary

Decoder crowding vs collateral after partialling out the paper's Section 3.8 nuisance set plus firing frequency (partial Spearman, 95% bootstrap CI), and the strongest predictor under that control:

| Setting | Crowding, raw count | Crowding, C-tilde | Strongest predictor (raw count) | Strongest predictor (C-tilde) |
|---|---|---|---|---|
| GPT-2-small | +0.475 [+0.372, +0.567] | +0.425 [+0.314, +0.523] | crowding (+0.48) | crowding (+0.42) |
| Pythia-70M-deduped | -0.004 [-0.124, +0.116] | +0.001 [-0.119, +0.120] | logit_l2 (+0.34) | logit_l2 (+0.31) |
| Gemma-2-2B | +0.242 [+0.123, +0.357] | +0.252 [+0.129, +0.368] | enc_dec_cos (-0.45) | enc_dec_cos (-0.43) |
| Llama-3.1-8B | +0.152 [+0.038, +0.267] | +0.166 [+0.053, +0.276] | logit_top10_mass (-0.29) | logit_top10_mass (-0.30) |

Reading: crowding carries independent signal in GPT-2-small on both metrics. In Pythia-70M it carries none on either metric, and the direct-logit footprint leads instead. In Gemma-2-2B crowding predicts the raw count strongly before controls but its partial correlation shrinks once effect magnitude is held fixed, and encoder-decoder alignment is the strongest predictor after control. The dominant predictor changes with the setting, which is the paper's own Table 2 pattern, now seen on the collateral axis after the control the paper only ran on stability.

## 1. Setup per setting

| Setting | Model | Primary SAE / hook | Downstream SAE / hook | Panel | Eligible features | Mean L0 (primary / downstream) | Contexts, texts dropped | dtype | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2-small | gpt2 | gpt2-small-res-jb `blocks.8.hook_resid_pre` | `blocks.10.hook_resid_pre` | 2048 | 7532 of 24576 | 71.79 / 60.01 | 2048 x 48 tokens, 36 short texts dropped | float32 | 103 s on NVIDIA GB10 |
| Pythia-70M-deduped | pythia-70m-deduped | pythia-70m-deduped-res-sm `blocks.4.hook_resid_post` | `blocks.5.hook_resid_post` | 2048 | 4375 of 32768 | 68.33 / 118.56 | 2048 x 48 tokens, 36 short texts dropped | float32 | 44 s on NVIDIA GB10 |
| Gemma-2-2B | google/gemma-2-2b | gemma-scope-2b-pt-res-canonical `layer_12/width_16k/canonical` at `blocks.12.hook_resid_post` | `layer_16/width_16k/canonical` at `blocks.16.hook_resid_post` | 2048 | 7517 of 16384 | 83.04 / 79.25 | 2048 x 48 tokens, 32 short texts dropped | float32 | 1079 s on NVIDIA GB10 |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | llama_scope_lxr_8x `l16r_8x` at `blocks.16.hook_resid_post` | `l20r_8x` at `blocks.20.hook_resid_post` | 1024 | 2837 of 32768 | 32.18 / 31.42 | 2048 x 48 tokens, 27 short texts dropped | bfloat16 | 891 s on NVIDIA GB10 |

Common: Wikitext-103 train split, 8,000 texts, 300 features sampled with seed 0 from the final-token firing-frequency band [0.002, 0.50], 16 contexts per type (top / random / low), additive steering alpha = 1.0 at the final token, tau = 0.05, epsilon_fire = 1e-6, crowding = top-20 mean absolute cosine, protocol v2 (no pad final tokens, whitespace-normalised dedup).

## 2. Decoder crowding vs collateral, with and without controls

Spearman rho; partial correlations are rank-based (both sides OLS-residualised on the ranked controls). 95% CIs from 10000 bootstrap resamples of the 300 features. Controls: **primary** = effect magnitude E_f, intervention value (constant, dropped), natural activation (mean final-token activation), firing frequency; **robust** = frequency and activation magnitude only.

### Target: raw downstream count C_{f,0.05}

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.482 [+0.381, +0.571] | +0.517 [+0.417, +0.605] | +0.475 [+0.372, +0.567] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | -0.022 [-0.136, +0.091] | -0.026 [-0.147, +0.096] | -0.004 [-0.124, +0.116] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.512 [+0.411, +0.600] | +0.468 [+0.360, +0.567] | +0.242 [+0.123, +0.357] | encoder norm, signed stability, rho = 0.350 |
| Llama-3.1-8B | +0.222 [+0.108, +0.330] | +0.182 [+0.066, +0.293] | +0.152 [+0.038, +0.267] | direct-logit L2, downstream count, rho = 0.464 |

### Target: C-tilde = C / (E_f + eps)

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.344 [+0.234, +0.447] | +0.355 [+0.246, +0.457] | +0.425 [+0.314, +0.523] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | +0.031 [-0.082, +0.145] | +0.011 [-0.105, +0.127] | +0.001 [-0.119, +0.120] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.069 [-0.051, +0.189] | +0.052 [-0.071, +0.177] | +0.252 [+0.129, +0.368] | encoder norm, signed stability, rho = 0.350 |
| Llama-3.1-8B | +0.201 [+0.092, +0.305] | +0.183 [+0.070, +0.291] | +0.166 [+0.053, +0.276] | direct-logit L2, downstream count, rho = 0.464 |

## 3. Strongest predictors under the primary control

Top three predictors by absolute partial rho after the primary control, per setting and metric.

### Target: raw downstream count C_{f,0.05}

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.475 [+0.372, +0.567] | 3.7e-18 |
| GPT-2-small | enc_dec_cos | -0.401 [-0.499, -0.294] | 6.5e-13 |
| GPT-2-small | crowd_max | +0.333 [+0.217, +0.441] | 4.1e-09 |
| Pythia-70M-deduped | logit_l2 | +0.343 [+0.229, +0.447] | 1.3e-09 |
| Pythia-70M-deduped | act_mean_firing | -0.207 [-0.315, -0.081] | 3.3e-04 |
| Pythia-70M-deduped | coact_entropy | -0.202 [-0.302, -0.084] | 4.8e-04 |
| Gemma-2-2B | enc_dec_cos | -0.454 [-0.552, -0.342] | 1.6e-16 |
| Gemma-2-2B | coact_entropy | -0.299 [-0.389, -0.196] | 1.6e-07 |
| Gemma-2-2B | crowding | +0.242 [+0.123, +0.357] | 2.5e-05 |
| Llama-3.1-8B | logit_top10_mass | -0.293 [-0.390, -0.188] | 2.6e-07 |
| Llama-3.1-8B | logit_entropy | +0.243 [+0.128, +0.351] | 2.3e-05 |
| Llama-3.1-8B | act_entropy | +0.159 [+0.051, +0.254] | 5.9e-03 |

### Target: C-tilde = C / (E_f + eps)

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.425 [+0.314, +0.523] | 1.9e-14 |
| GPT-2-small | enc_dec_cos | -0.399 [-0.495, -0.294] | 9.1e-13 |
| GPT-2-small | coact_entropy | -0.345 [-0.443, -0.232] | 9.9e-10 |
| Pythia-70M-deduped | logit_l2 | +0.311 [+0.193, +0.421] | 4.2e-08 |
| Pythia-70M-deduped | coact_count | +0.198 [+0.085, +0.307] | 6.1e-04 |
| Pythia-70M-deduped | act_mean_firing | -0.171 [-0.275, -0.057] | 3.2e-03 |
| Gemma-2-2B | enc_dec_cos | -0.427 [-0.520, -0.325] | 1.5e-14 |
| Gemma-2-2B | coact_entropy | -0.311 [-0.402, -0.201] | 4.4e-08 |
| Gemma-2-2B | crowding | +0.252 [+0.129, +0.368] | 1.1e-05 |
| Llama-3.1-8B | logit_top10_mass | -0.298 [-0.394, -0.192] | 1.7e-07 |
| Llama-3.1-8B | logit_entropy | +0.231 [+0.115, +0.344] | 6.1e-05 |
| Llama-3.1-8B | crowding | +0.166 [+0.053, +0.276] | 4.2e-03 |

## 4. Table B3 analog: predictor sets on the residualized collateral target

The collateral label is OLS-residualised against the primary control set, then predicted with ridge (alpha = 1, standardised predictors) under 5-fold cross-validation; score = Spearman between held-out predictions and residualised label, mean over folds (sd in parentheses).

### Target: raw downstream count C_{f,0.05}

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.015 (0.08) | -0.035 (0.06) | +0.458 (0.06) | +0.208 (0.10) | +0.182 (0.11) | +0.504 (0.08) | +0.494 (0.09) |
| Pythia-70M-deduped | -0.018 (0.14) | -0.113 (0.10) | +0.011 (0.08) | +0.493 (0.08) | +0.184 (0.09) | +0.518 (0.05) | +0.522 (0.05) |
| Gemma-2-2B | -0.144 (0.27) | -0.106 (0.29) | +0.468 (0.16) | +0.137 (0.15) | +0.168 (0.09) | +0.591 (0.11) | +0.589 (0.11) |
| Llama-3.1-8B | -0.031 (0.11) | -0.080 (0.13) | +0.099 (0.04) | +0.300 (0.13) | +0.114 (0.05) | +0.263 (0.12) | +0.268 (0.07) |

### Target: C-tilde = C / (E_f + eps)

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.027 (0.08) | -0.029 (0.05) | +0.447 (0.07) | +0.216 (0.10) | +0.176 (0.11) | +0.500 (0.09) | +0.492 (0.09) |
| Pythia-70M-deduped | -0.118 (0.09) | -0.116 (0.11) | +0.044 (0.09) | +0.426 (0.09) | +0.162 (0.09) | +0.475 (0.05) | +0.478 (0.05) |
| Gemma-2-2B | -0.127 (0.24) | -0.095 (0.27) | +0.429 (0.16) | +0.104 (0.14) | +0.144 (0.14) | +0.579 (0.10) | +0.586 (0.11) |
| Llama-3.1-8B | -0.037 (0.11) | -0.075 (0.09) | +0.160 (0.06) | +0.309 (0.13) | +0.164 (0.08) | +0.312 (0.07) | +0.294 (0.06) |

## 5. Feature-sample robustness (crowding)

Same contexts, different random sample of 300 features (seeds 0, 1, 2). Values are crowding vs collateral rho.

| Setting | Target | Control | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|---|---|
| Gemma-2-2B | collateral_ctilde | none | +0.069 | +0.099 | +0.087 | +0.085 | 0.015 |
| Gemma-2-2B | collateral_ctilde | primary | +0.252 | +0.209 | +0.222 | +0.228 | 0.022 |
| Gemma-2-2B | collateral_ctilde | robust | +0.052 | +0.086 | +0.128 | +0.088 | 0.038 |
| Gemma-2-2B | collateral_raw | none | +0.512 | +0.531 | +0.462 | +0.502 | 0.036 |
| Gemma-2-2B | collateral_raw | primary | +0.242 | +0.217 | +0.233 | +0.231 | 0.013 |
| Gemma-2-2B | collateral_raw | robust | +0.468 | +0.468 | +0.439 | +0.458 | 0.017 |
| GPT-2-small | collateral_ctilde | none | +0.344 | +0.313 | +0.289 | +0.315 | 0.028 |
| GPT-2-small | collateral_ctilde | primary | +0.425 | +0.365 | +0.351 | +0.380 | 0.039 |
| GPT-2-small | collateral_ctilde | robust | +0.355 | +0.306 | +0.289 | +0.316 | 0.034 |
| GPT-2-small | collateral_raw | none | +0.482 | +0.450 | +0.458 | +0.463 | 0.016 |
| GPT-2-small | collateral_raw | primary | +0.475 | +0.413 | +0.405 | +0.431 | 0.039 |
| GPT-2-small | collateral_raw | robust | +0.517 | +0.472 | +0.482 | +0.491 | 0.024 |
| Pythia-70M-deduped | collateral_ctilde | none | +0.031 | +0.045 | -0.062 | +0.004 | 0.058 |
| Pythia-70M-deduped | collateral_ctilde | primary | +0.001 | +0.002 | -0.074 | -0.023 | 0.044 |
| Pythia-70M-deduped | collateral_ctilde | robust | +0.011 | -0.018 | -0.112 | -0.040 | 0.065 |
| Pythia-70M-deduped | collateral_raw | none | -0.022 | +0.059 | +0.028 | +0.021 | 0.041 |
| Pythia-70M-deduped | collateral_raw | primary | -0.004 | +0.006 | -0.081 | -0.027 | 0.048 |
| Pythia-70M-deduped | collateral_raw | robust | -0.026 | +0.051 | +0.019 | +0.015 | 0.039 |
| Llama-3.1-8B | | seed 0 only so far | | | | | |

## 6. Regression gate against the published GPT-2-small notebook

`src/run_setting.py --protocol v1` reproduces the Kaggle notebook's context construction exactly. Same seed, same eligible-feature count (7469). The remaining differences (at most 0.02) come from a different GPU (T4 vs GB10) and different TransformerLens / SAELens versions; the eligible set and hence the feature sample are the same.

| Statistic | Kaggle notebook (Aug 7 version) | this code, protocol v1 |
|---|---|---|
| rho_crowding__collateral_raw | +0.548 | +0.554 |
| partial_crowding__collateral_raw__given_freq_actmag | +0.569 | +0.572 |
| rho_crowding__collateral_ctilde | +0.421 | +0.403 |
| partial_crowding__collateral_ctilde__given_freq_actmag | +0.416 | +0.403 |
| rho_frequency__collateral_raw | +0.248 | +0.229 |
| rho_frequency__collateral_ctilde | +0.146 | +0.124 |
| rho_act_mag__collateral_raw | +0.230 | +0.221 |
| crowd_rho_lowfreq_half | +0.651 | +0.658 |
| crowd_rho_highfreq_half | +0.449 | +0.450 |

Protocol v1 leaves 33 of 2048 contexts with a pad token in the final position (short texts are right-padded). Protocol v2, used for every reported setting, drops texts shorter than 48 tokens instead. On GPT-2-small this moves crowding vs raw count from +0.554 (v1) to +0.482 (v2) and the frequency baseline from +0.229 to +0.110; the feature sample also changes because the eligible set changes.

## 6b. Cross-machine check (Kaggle T4, different library stack)

The same `src/run_setting.py`, protocol v2, seed 0, re-run on Kaggle, Tesla T4 with transformer-lens 2.18.0, sae-lens 5.11.0, transformers 4.57.6, torch 2.10.0+cu128, numpy 1.26.4 (GB10 runs: transformer-lens 3.8.1, sae-lens 6.50.0, transformers 5.16.1, torch 2.12.1). Eligible-feature counts and mean L0 matched exactly on both machines.

| Setting | Statistic | GB10 (this repo) | Kaggle T4 |
|---|---|---|---|
| GPT-2-small | rho_crowding__collateral_raw | +0.482 | +0.487 |
| GPT-2-small | partial_crowding__collateral_raw__given_freq_actmag | +0.517 | +0.509 |
| GPT-2-small | rho_crowding__collateral_ctilde | +0.344 | +0.359 |
| GPT-2-small | partial_crowding__collateral_ctilde__given_freq_actmag | +0.355 | +0.361 |
| GPT-2-small | rho_frequency__collateral_raw | +0.110 | +0.028 |
| Pythia-70M-deduped | rho_crowding__collateral_raw | -0.022 | -0.077 |
| Pythia-70M-deduped | partial_crowding__collateral_raw__given_freq_actmag | -0.026 | -0.080 |
| Pythia-70M-deduped | rho_crowding__collateral_ctilde | +0.031 | -0.089 |
| Pythia-70M-deduped | partial_crowding__collateral_ctilde__given_freq_actmag | +0.011 | -0.077 |
| Pythia-70M-deduped | rho_frequency__collateral_raw | +0.175 | +0.111 |

GPT-2-small's crowding statistics agree to within 0.02 across machines. Pythia's move by up to 0.12, but every Pythia value on both machines sits inside the roughly +/-0.12 bootstrap interval around zero, so the stable finding there is the null itself, not any particular value. The weak frequency baselines also shift between machines for the same reason.

## 8. Robustness variants

## Label reliability (split-half over contexts)

Same 300 features, labels measured once on the even-indexed contexts (A) and once on the odd-indexed contexts (B). Spearman between A and B is the test-retest reliability; the attenuation-corrected crowding correlation is rho / sqrt(reliability).

| Setting | Label | reliability rho(A, B) | crowding rho (A) | crowding rho (B) | corrected, raw count |
|---|---|---|---|---|---|
| GPT-2-small | collateral_raw | +0.960 | +0.489 | +0.496 | +0.492 |
| GPT-2-small | collateral_ctilde | +0.938 | +0.353 | +0.357 |  |
| GPT-2-small | effect_l2 | +0.895 |  | |  |
| GPT-2-small | stab_signed | +0.902 |  | |  |
| GPT-2-small | stab_abs | +0.903 |  | |  |
| GPT-2-small | kl_mean | +0.617 |  | |  |
| Pythia-70M-deduped | collateral_raw | +0.906 | -0.022 | -0.028 | -0.024 |
| Pythia-70M-deduped | collateral_ctilde | +0.759 | -0.016 | -0.025 |  |
| Pythia-70M-deduped | effect_l2 | +0.946 |  | |  |
| Pythia-70M-deduped | stab_signed | +0.937 |  | |  |
| Pythia-70M-deduped | stab_abs | +0.937 |  | |  |
| Pythia-70M-deduped | kl_mean | +0.781 |  | |  |
| Gemma-2-2B | collateral_raw | +0.925 | +0.526 | +0.542 | +0.533 |
| Gemma-2-2B | collateral_ctilde | +0.788 | +0.191 | +0.100 |  |
| Gemma-2-2B | effect_l2 | +0.800 |  | |  |
| Gemma-2-2B | stab_signed | +0.664 |  | |  |
| Gemma-2-2B | stab_abs | +0.602 |  | |  |
| Gemma-2-2B | kl_mean | +0.711 |  | |  |
| Llama-3.1-8B | collateral_raw | +0.946 | +0.240 | +0.213 | +0.228 |
| Llama-3.1-8B | collateral_ctilde | +0.897 | +0.222 | +0.163 |  |
| Llama-3.1-8B | effect_l2 | +0.854 |  | |  |
| Llama-3.1-8B | stab_signed | +0.694 |  | |  |
| Llama-3.1-8B | stab_abs | +0.520 |  | |  |
| Llama-3.1-8B | kl_mean | +0.375 |  | |  |

### Steering coefficient

Crowding vs collateral under different additive coefficients. alpha = 1.0 is the paper's value; q95 adds alpha times the feature's 95th-percentile natural activation, so the perturbation is scaled to each feature.

| Setting | Variant | raw count: rho | primary partial | C-tilde: rho | primary partial | median intervention value |
|---|---|---|---|---|---|---|
| GPT-2-small | alpha 1.0 (reported) | +0.482 | +0.475 | +0.344 | +0.425 | 1 |
| GPT-2-small | alpha0.5 | +0.487 | +0.493 | +0.471 | +0.493 | 0.5 |
| GPT-2-small | alpha2 | +0.460 | +0.457 | +0.145 | +0.364 | 2 |
| GPT-2-small | alpha4 | +0.347 | +0.382 | -0.015 | +0.323 | 4 |
| GPT-2-small | q95 | +0.352 | +0.264 | -0.234 | +0.266 | 4.16 |
| Pythia-70M-deduped | alpha 1.0 (reported) | -0.022 | -0.004 | +0.031 | +0.001 | 1 |
| Pythia-70M-deduped | alpha0.5 | +0.102 | +0.136 | +0.146 | +0.119 | 0.5 |
| Pythia-70M-deduped | alpha2 | -0.046 | -0.048 | +0.056 | -0.008 | 2 |
| Pythia-70M-deduped | alpha4 | -0.036 | -0.076 | +0.048 | -0.055 | 4 |
| Pythia-70M-deduped | q95 | +0.214 | +0.037 | +0.198 | +0.110 | 0.792 |
| Gemma-2-2B | alpha 1.0 (reported) | +0.512 | +0.242 | +0.069 | +0.252 | 1 |
| Llama-3.1-8B | alpha 1.0 (reported) | +0.222 | +0.152 | +0.201 | +0.166 | 1 |

### Random-direction control

300 random directions with norms drawn from the decoder norms, steered on 48 random contexts each. Crowding is the same top-20 mean absolute cosine to the dictionary. Frequency and activation controls do not exist for random directions, so the control is effect magnitude only; the SAE-feature row uses the same control for comparison.

| Setting | Steered vectors | raw count: rho [95% CI] | partial given E_f | C-tilde: rho [95% CI] | partial given E_f | median collateral |
|---|---|---|---|---|---|---|
| GPT-2-small | SAE features | +0.482 [+0.379, +0.573] | +0.444 | +0.344 [+0.235, +0.446] | +0.398 | 11.5 |
| GPT-2-small | random directions | +0.021 [-0.096, +0.135] | +0.020 | +0.016 [-0.099, +0.135] | +0.020 | 9.3 |
| Pythia-70M-deduped | SAE features | -0.022 [-0.132, +0.093] | +0.008 | +0.031 [-0.083, +0.145] | +0.016 | 22.1 |
| Pythia-70M-deduped | random directions | -0.014 [-0.126, +0.099] | -0.017 | -0.024 [-0.144, +0.090] | -0.021 | 22.8 |
| Gemma-2-2B | SAE features | +0.512 [+0.413, +0.601] | +0.267 | +0.069 [-0.051, +0.193] | +0.266 | 4.3 |
| Gemma-2-2B | random directions | -0.120 [-0.231, -0.010] | -0.130 | -0.118 [-0.226, -0.002] | -0.082 | 3.2 |
| Llama-3.1-8B | SAE features | +0.222 [+0.107, +0.332] | +0.161 | +0.201 [+0.090, +0.305] | +0.171 | 2.7 |
| Llama-3.1-8B | random directions | +0.020 [-0.095, +0.132] | +0.015 | +0.010 [-0.107, +0.126] | +0.004 | 2.0 |

### Dense-panel exclusion and residual-space collateral

Collateral recomputed without downstream panel features that fire in more than 10% of contexts, and the downstream residual-stream change itself (its norm, and the share of it the downstream SAE reconstructs).

| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) | crowding vs residual change norm | mean SAE-explained share of the change |
|---|---|---|---|---|---|---|
| GPT-2-small | 2042 of 2048 | +0.482 | +0.486 | +0.477 | +0.328 | 0.845 |
| Pythia-70M-deduped | 1969 of 2048 | -0.022 | -0.093 | -0.093 | +0.144 | 0.778 |
| Gemma-2-2B | 2011 of 2048 | +0.512 | +0.515 | +0.196 | +0.626 | 2.289 |
| Llama-3.1-8B | 1005 of 1024 | +0.222 | +0.242 | +0.144 | +0.172 | 1.186 |


## 9. Output-level side effect: KL shift

The same analysis with the mean KL divergence between clean and steered next-token distributions as the label.

| Setting | crowding vs KL: raw | partial, primary set | strongest predictor after control |
|---|---|---|---|
| GPT-2-small | +0.290 [+0.177, +0.397] | +0.270 [+0.160, +0.374] | logit_l2 (-0.30, Holm p 2.0e-06) |
| Pythia-70M-deduped | +0.135 [+0.016, +0.248] | +0.086 [-0.046, +0.217] | enc_norm (+0.29, Holm p 5.6e-06) |
| Gemma-2-2B | +0.487 [+0.392, +0.575] | +0.105 [-0.020, +0.228] | logit_entropy (-0.24, Holm p 6.5e-04) |
| Llama-3.1-8B | +0.075 [-0.044, +0.191] | -0.102 [-0.211, +0.014] | logit_entropy (+0.16, Holm p 1.3e-01) |

## 7. Assumptions and conventions

- natural activation a-bar_f := act_mag (mean final-token activation over all contexts).
- intervention value c_f is constant under fixed_global_add (alpha = 1.0) and is dropped.
- ridge alpha = 1.0 on standardised predictors, 5-fold KFold shuffle seed 0.
- Feature sampling band [0.002, 0.50] on final-token firing frequency; the paper says only "the non-degenerate range".
- Downstream panel = the most frequently active downstream features on the clean contexts (2048; 1024 for Llama).
- Model loading per setting: GPT-2-small in float32 with TransformerLens weight processing on (default); Pythia-70M-deduped in float32 with TransformerLens weight processing on (default); Gemma-2-2B in float32 with TransformerLens weight processing on (default); Llama-3.1-8B in bfloat16 with TransformerLens weight processing off (from_pretrained_no_processing). Weight processing never changes the residual stream the SAEs read (TransformerLens centres writing weights only for LayerNorm models); the mean L0 column above is the empirical check that each SAE sees the activations it was trained on.
- Direct-logit predictors, effect magnitude and stability use the unembedding as loaded: centred for GPT-2 and Pythia, uncentred for Gemma-2 (logit softcap) and for any setting loaded without processing.
- Effect magnitude E_f and the stability cosines are computed on final-token logit differences in float32.

## Files

- `results/<setting>/per_feature.csv`: one row per sampled feature, all predictors and labels.
- `results/<setting>/selection.json`: sampled feature indices and downstream panel indices.
- `results/<setting>/meta.json`: resolved config, sizes, versions, timings, headline statistics.
- `results/analysis/partial_correlations.csv`, `residualized_cv_ridge.csv`, `seed_summary.csv`, `summary.md`, `crowding_partials.png`.
- `results/analysis/table_collateral_residualized.tex`: LaTeX rows for the appendix.
