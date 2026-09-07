# Collateral-side residualization across four model/SAE settings

Companion results for the revision of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Duan, arXiv:2606.08365). All numbers below are read from `results/` by `analysis/make_results.py`; nothing is copied from the paper except where labelled as the paper's value.

Settings included: GPT-2-small, Pythia-70M-deduped, Gemma-2-2B, Llama-3.1-8B.

## Summary

Decoder crowding vs collateral after partialling out the paper's Section 3.8 nuisance set plus firing frequency (partial Spearman, 95% bootstrap CI), and the strongest predictor under that control:

| Setting | Crowding, raw count | Crowding, C-tilde | Strongest predictor (raw count) | Strongest predictor (C-tilde) |
|---|---|---|---|---|
| GPT-2-small | +0.475 [+0.373, +0.567] | +0.425 [+0.317, +0.524] | crowding (+0.48) | crowding (+0.42) |
| Pythia-70M-deduped | -0.004 [-0.124, +0.114] | +0.001 [-0.121, +0.124] | logit_l2 (+0.34) | logit_l2 (+0.31) |
| Gemma-2-2B | +0.242 [+0.123, +0.362] | +0.252 [+0.130, +0.364] | enc_dec_cos (-0.45) | enc_dec_cos (-0.43) |
| Llama-3.1-8B | +0.152 [+0.039, +0.266] | +0.166 [+0.052, +0.275] | logit_top10_mass (-0.29) | logit_top10_mass (-0.30) |

Reading: crowding carries independent signal in GPT-2-small on both metrics. In Pythia-70M the crowding estimate is near zero with intervals spanning zero on both metrics, and the direct-logit footprint leads instead. In Gemma-2-2B crowding predicts the raw count strongly before controls but its partial correlation shrinks once effect magnitude is held fixed, and encoder-decoder alignment is the strongest predictor after control. The dominant predictor changes with the setting, which is the paper's own Table 2 pattern, now seen on the collateral axis after the control the paper only ran on stability.

Intervals are pointwise. Llama crowding has Holm-adjusted p = 0.122 for raw count and 0.063 for C-tilde. Neither passes the 0.05 threshold after that correction.

## 1. Setup per setting

| Setting | Model | Primary SAE / hook | Downstream SAE / hook | Panel | Eligible features | Mean L0 (primary / downstream) | Contexts, texts dropped | dtype | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2-small | gpt2 | gpt2-small-res-jb `blocks.8.hook_resid_pre` | `blocks.10.hook_resid_pre` | 2048 | 7532 of 24576 | 71.79 / 60.01 | 2048 x 48 tokens, 36 short texts dropped | float32 | 142 s on NVIDIA GB10 |
| Pythia-70M-deduped | pythia-70m-deduped | pythia-70m-deduped-res-sm `blocks.4.hook_resid_post` | `blocks.5.hook_resid_post` | 2048 | 4375 of 32768 | 68.33 / 118.56 | 2048 x 48 tokens, 36 short texts dropped | float32 | 66 s on NVIDIA GB10 |
| Gemma-2-2B | google/gemma-2-2b | gemma-scope-2b-pt-res-canonical `layer_12/width_16k/canonical` at `blocks.12.hook_resid_post` | `layer_16/width_16k/canonical` at `blocks.16.hook_resid_post` | 2048 | 7517 of 16384 | 83.04 / 79.25 | 2048 x 48 tokens, 32 short texts dropped | float32 | 1925 s on NVIDIA GB10 |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B | llama_scope_lxr_8x `l16r_8x` at `blocks.16.hook_resid_post` | `l20r_8x` at `blocks.20.hook_resid_post` | 1024 | 2837 of 32768 | 32.18 / 31.42 | 2048 x 48 tokens, 27 short texts dropped | bfloat16 | 1750 s on NVIDIA GB10 |

Common: Wikitext-103 train split, 8,000 texts, 300 features sampled with seed 0 from the final-token firing-frequency band [0.002, 0.50], 16 contexts per type (top / random / low), additive steering alpha = 1.0 at the final token, tau = 0.05, epsilon_fire = 1e-6, crowding = top-20 mean absolute cosine, protocol v2 (no pad final tokens, whitespace-normalised dedup). Audited-run wall clocks include the paired random control and output/checkpoint saving.

## 2. Decoder crowding vs collateral, with and without controls

Spearman rho; partial correlations are rank-based (both sides OLS-residualised on the ranked controls). 95% CIs from 10000 bootstrap resamples of the 300 features. Controls: **primary** = effect magnitude E_f, intervention value (constant, dropped), natural activation (mean final-token activation), firing frequency; **robust** = frequency and activation magnitude only.

### Target: raw downstream count C_{f,0.05}

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.482 [+0.381, +0.571] | +0.517 [+0.418, +0.604] | +0.475 [+0.373, +0.567] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | -0.022 [-0.137, +0.092] | -0.026 [-0.148, +0.098] | -0.004 [-0.124, +0.114] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.512 [+0.411, +0.605] | +0.468 [+0.366, +0.566] | +0.242 [+0.123, +0.362] | encoder norm, signed stability, rho = 0.350 |
| Llama-3.1-8B | +0.222 [+0.109, +0.332] | +0.182 [+0.067, +0.298] | +0.152 [+0.039, +0.266] | direct-logit L2, downstream count, rho = 0.464 |

### Target: C-tilde = C / (E_f + eps)

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.344 [+0.233, +0.445] | +0.355 [+0.248, +0.454] | +0.425 [+0.317, +0.524] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | +0.031 [-0.084, +0.143] | +0.011 [-0.108, +0.127] | +0.001 [-0.121, +0.124] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.069 [-0.054, +0.192] | +0.052 [-0.073, +0.177] | +0.252 [+0.130, +0.364] | encoder norm, signed stability, rho = 0.350 |
| Llama-3.1-8B | +0.201 [+0.092, +0.307] | +0.183 [+0.070, +0.293] | +0.166 [+0.052, +0.275] | direct-logit L2, downstream count, rho = 0.464 |

## 3. Strongest predictors under the primary control

Top three predictors by absolute partial rho after the primary control, per setting and metric. Holm and BH adjust within each setting/target/control family. Ranking by magnitude does not test whether one predictor is stronger than another.

### Target: raw downstream count C_{f,0.05}

| Setting | Predictor | partial rho [95% CI] | p | Holm p | BH q |
|---|---|---|---|---|---|
| GPT-2-small | crowding | +0.475 [+0.373, +0.567] | 3.7e-18 | 6.3e-17 | 6.3e-17 |
| GPT-2-small | enc_dec_cos | -0.401 [-0.498, -0.294] | 6.5e-13 | 1.0e-11 | 5.5e-12 |
| GPT-2-small | crowd_max | +0.333 [+0.215, +0.440] | 4.1e-09 | 6.2e-08 | 2.3e-08 |
| Pythia-70M-deduped | logit_l2 | +0.343 [+0.230, +0.448] | 1.3e-09 | 2.3e-08 | 2.3e-08 |
| Pythia-70M-deduped | act_mean_firing | -0.207 [-0.313, -0.088] | 3.3e-04 | 5.3e-03 | 2.7e-03 |
| Pythia-70M-deduped | coact_entropy | -0.202 [-0.301, -0.086] | 4.8e-04 | 7.1e-03 | 2.7e-03 |
| Gemma-2-2B | enc_dec_cos | -0.454 [-0.550, -0.344] | 1.6e-16 | 2.8e-15 | 2.8e-15 |
| Gemma-2-2B | coact_entropy | -0.299 [-0.388, -0.194] | 1.6e-07 | 2.5e-06 | 1.3e-06 |
| Gemma-2-2B | crowding | +0.242 [+0.123, +0.362] | 2.5e-05 | 3.7e-04 | 1.4e-04 |
| Llama-3.1-8B | logit_top10_mass | -0.293 [-0.392, -0.191] | 2.6e-07 | 4.4e-06 | 4.4e-06 |
| Llama-3.1-8B | logit_entropy | +0.243 [+0.126, +0.352] | 2.3e-05 | 3.7e-04 | 2.0e-04 |
| Llama-3.1-8B | act_entropy | +0.159 [+0.051, +0.255] | 5.9e-03 | 8.8e-02 | 3.3e-02 |

### Target: C-tilde = C / (E_f + eps)

| Setting | Predictor | partial rho [95% CI] | p | Holm p | BH q |
|---|---|---|---|---|---|
| GPT-2-small | crowding | +0.425 [+0.317, +0.524] | 1.9e-14 | 3.3e-13 | 3.3e-13 |
| GPT-2-small | enc_dec_cos | -0.399 [-0.499, -0.290] | 9.1e-13 | 1.5e-11 | 7.7e-12 |
| GPT-2-small | coact_entropy | -0.345 [-0.443, -0.234] | 9.9e-10 | 1.5e-08 | 5.6e-09 |
| Pythia-70M-deduped | logit_l2 | +0.311 [+0.195, +0.423] | 4.2e-08 | 7.2e-07 | 7.2e-07 |
| Pythia-70M-deduped | coact_count | +0.198 [+0.085, +0.304] | 6.1e-04 | 9.8e-03 | 5.2e-03 |
| Pythia-70M-deduped | act_mean_firing | -0.171 [-0.275, -0.059] | 3.2e-03 | 4.8e-02 | 1.8e-02 |
| Gemma-2-2B | enc_dec_cos | -0.427 [-0.519, -0.321] | 1.5e-14 | 2.5e-13 | 2.5e-13 |
| Gemma-2-2B | coact_entropy | -0.311 [-0.403, -0.203] | 4.4e-08 | 7.1e-07 | 3.8e-07 |
| Gemma-2-2B | crowding | +0.252 [+0.130, +0.364] | 1.1e-05 | 1.6e-04 | 6.1e-05 |
| Llama-3.1-8B | logit_top10_mass | -0.298 [-0.394, -0.193] | 1.7e-07 | 2.9e-06 | 2.9e-06 |
| Llama-3.1-8B | logit_entropy | +0.231 [+0.115, +0.339] | 6.1e-05 | 9.7e-04 | 5.2e-04 |
| Llama-3.1-8B | crowding | +0.166 [+0.052, +0.275] | 4.2e-03 | 6.3e-02 | 2.4e-02 |

## 4. Table B3 analog: predictor sets on the residualized collateral target

Within each training fold, the collateral label is OLS-residualised against the primary control set. The fitted nuisance model is applied to that fold's held-out features, and ridge predicts those residuals (alpha = 1, predictor scaling fitted on training features) under 5-fold cross-validation; score = Spearman between held-out predictions and residualised label, mean over all five folds (sd in parentheses). Constant predictions give undefined Spearman; a baseline made only of nuisance variables has no linear training-residual signal, so numerical noise is not scored.

### Target: raw downstream count C_{f,0.05}

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | undefined | undefined | +0.456 (0.07) | +0.201 (0.11) | +0.198 (0.11) | +0.512 (0.08) | +0.499 (0.09) |
| Pythia-70M-deduped | undefined | undefined | +0.004 (0.10) | +0.500 (0.07) | +0.202 (0.12) | +0.521 (0.05) | +0.528 (0.06) |
| Gemma-2-2B | undefined | undefined | +0.469 (0.17) | +0.152 (0.13) | +0.204 (0.09) | +0.594 (0.12) | +0.591 (0.11) |
| Llama-3.1-8B | undefined | undefined | +0.142 (0.06) | +0.319 (0.12) | +0.131 (0.13) | +0.289 (0.10) | +0.294 (0.06) |

### Target: C-tilde = C / (E_f + eps)

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | undefined | undefined | +0.451 (0.08) | +0.213 (0.10) | +0.191 (0.11) | +0.514 (0.09) | +0.507 (0.10) |
| Pythia-70M-deduped | undefined | undefined | +0.047 (0.11) | +0.429 (0.09) | +0.184 (0.13) | +0.482 (0.05) | +0.487 (0.05) |
| Gemma-2-2B | undefined | undefined | +0.439 (0.16) | +0.109 (0.14) | +0.164 (0.15) | +0.593 (0.10) | +0.595 (0.11) |
| Llama-3.1-8B | undefined | undefined | +0.187 (0.07) | +0.318 (0.13) | +0.177 (0.12) | +0.325 (0.08) | +0.313 (0.06) |

## 5. Feature-sample robustness (crowding)

Same 2,048-context pool, different random sample of 300 features and random-context selections (seeds 0, 1, 2). These are not independently trained SAE seeds. Values are crowding vs collateral rho.

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

`src/run_setting.py --protocol v1` reproduces the Kaggle notebook's context construction exactly. Same seed, same eligible-feature count (7469). The remaining differences (at most 0.02) occur with a different GPU (T4 vs GB10) and different TransformerLens / SAELens versions. Equal eligible counts alone do not establish identical feature IDs across machines.

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

GPT-2-small's crowding statistics agree to within 0.02 across machines. Pythia's move by up to 0.12, but every Pythia value on both machines sits inside the roughly +/-0.12 bootstrap interval around zero, so the stable finding there is the absence of a detectable association, with point estimates that vary across machines. The weak frequency baselines also shift between machines.

## 7. Robustness variants

### Context-pool reproducibility

The same 300 features and clean-activation predictors are retained. Labels use 48 mixed contexts selected from the 1,024 even-indexed contexts (A) or the 1,024 odd-indexed contexts (B) of the same ordered corpus pool. The halves are disjoint, but are not independent corpus draws; neighboring texts can be related. These are split-pool Spearman correlations, conditional on one dictionary and feature sample. No predictor ceiling or attenuation-corrected correlation is claimed. Correlation is undefined when one draw gives a constant label. Percentile intervals resample matched feature rows and are undefined if a draw is degenerate; JSON records the valid-draw count.

| Setting | Label | rho(A, B) [95% CI] | crowding rho (A) | crowding rho (B) |
|---|---|---|---|---|
| GPT-2-small | collateral_raw | +0.960 [+0.944, +0.969] | +0.489 | +0.496 |
| GPT-2-small | collateral_ctilde | +0.938 [+0.919, +0.952] | +0.353 | +0.357 |
| GPT-2-small | effect_l2 | +0.895 [+0.863, +0.920] |  |  |
| GPT-2-small | stab_signed | +0.902 [+0.868, +0.927] |  |  |
| GPT-2-small | stab_abs | +0.903 [+0.870, +0.927] |  |  |
| GPT-2-small | kl_mean | +0.617 [+0.526, +0.697] |  |  |
| GPT-2-small | stab_anti_frac | +0.093 [-0.060, +0.264] |  |  |
| GPT-2-small | effect_pc1_ratio | +0.241 [+0.129, +0.347] |  |  |
| Pythia-70M-deduped | collateral_raw | +0.906 [+0.877, +0.928] | -0.022 | -0.028 |
| Pythia-70M-deduped | collateral_ctilde | +0.759 [+0.696, +0.811] | -0.016 | -0.025 |
| Pythia-70M-deduped | effect_l2 | +0.946 [+0.920, +0.962] |  |  |
| Pythia-70M-deduped | stab_signed | +0.937 [+0.913, +0.954] |  |  |
| Pythia-70M-deduped | stab_abs | +0.937 [+0.914, +0.954] |  |  |
| Pythia-70M-deduped | kl_mean | +0.781 [+0.722, +0.831] |  |  |
| Pythia-70M-deduped | stab_anti_frac | undefined (constant label) |  |  |
| Pythia-70M-deduped | effect_pc1_ratio | +0.440 [+0.341, +0.532] |  |  |
| Gemma-2-2B | collateral_raw | +0.925 [+0.898, +0.943] | +0.526 | +0.542 |
| Gemma-2-2B | collateral_ctilde | +0.788 [+0.733, +0.833] | +0.191 | +0.100 |
| Gemma-2-2B | effect_l2 | +0.800 [+0.744, +0.847] |  |  |
| Gemma-2-2B | stab_signed | +0.664 [+0.575, +0.746] |  |  |
| Gemma-2-2B | stab_abs | +0.602 [+0.507, +0.685] |  |  |
| Gemma-2-2B | kl_mean | +0.711 [+0.635, +0.775] |  |  |
| Gemma-2-2B | stab_anti_frac | +0.261 [+0.148, +0.371] |  |  |
| Gemma-2-2B | effect_pc1_ratio | +0.205 [+0.091, +0.316] |  |  |
| Llama-3.1-8B | collateral_raw | +0.946 [+0.928, +0.958] | +0.240 | +0.213 |
| Llama-3.1-8B | collateral_ctilde | +0.897 [+0.864, +0.922] | +0.222 | +0.163 |
| Llama-3.1-8B | effect_l2 | +0.854 [+0.810, +0.888] |  |  |
| Llama-3.1-8B | stab_signed | +0.694 [+0.614, +0.762] |  |  |
| Llama-3.1-8B | stab_abs | +0.520 [+0.419, +0.611] |  |  |
| Llama-3.1-8B | kl_mean | +0.375 [+0.265, +0.481] |  |  |
| Llama-3.1-8B | stab_anti_frac | +0.105 [-0.007, +0.218] |  |  |
| Llama-3.1-8B | effect_pc1_ratio | +0.103 [-0.011, +0.215] |  |  |

### Steering coefficient

Only GPT-2 and Pythia have coefficient sweeps. q95 is the 95th percentile conditional on firing (activation > 1e-6), multiplied by alpha; it is not the unconditional activation percentile. The primary control includes the varying intervention value for q95. These are coefficient sensitivity checks, without evidence that the selected scale improves generated behavior.

| Setting | Variant | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] | median coefficient |
|---|---|---|---|---|---|---|
| GPT-2-small | alpha1 | +0.482 | +0.475 [+0.375, +0.567] | +0.344 | +0.425 [+0.313, +0.524] | 1 |
| GPT-2-small | alpha0.5 | +0.487 | +0.493 [+0.391, +0.583] | +0.471 | +0.493 [+0.391, +0.584] | 0.5 |
| GPT-2-small | alpha2 | +0.460 | +0.457 [+0.355, +0.549] | +0.145 | +0.364 [+0.248, +0.467] | 2 |
| GPT-2-small | alpha4 | +0.347 | +0.382 [+0.271, +0.486] | -0.015 | +0.323 [+0.206, +0.424] | 4 |
| GPT-2-small | q95 firing | +0.352 | +0.264 [+0.156, +0.370] | -0.234 | +0.266 [+0.145, +0.366] | 4.16 |
| Pythia-70M-deduped | alpha1 | -0.022 | -0.004 [-0.121, +0.118] | +0.031 | +0.001 [-0.117, +0.123] | 1 |
| Pythia-70M-deduped | alpha0.5 | +0.102 | +0.136 [+0.017, +0.255] | +0.146 | +0.119 [-0.005, +0.243] | 0.5 |
| Pythia-70M-deduped | alpha2 | -0.046 | -0.048 [-0.167, +0.070] | +0.056 | -0.008 [-0.134, +0.120] | 2 |
| Pythia-70M-deduped | alpha4 | -0.036 | -0.076 [-0.199, +0.053] | +0.048 | -0.055 [-0.185, +0.079] | 4 |
| Pythia-70M-deduped | q95 firing | +0.214 | +0.037 [-0.085, +0.166] | +0.198 | +0.110 [-0.015, +0.226] | 0.792 |

### Paired random-direction control

Each of 300 isotropic Gaussian directions is normalized to unit length, then scaled to the corresponding sampled SAE decoder norm. Each pair uses identical 48-context indices, the same coefficient, downstream panel, and clean forward pass. This retains the nonunit Llama decoder scales. Direction seed is 2000; vectors and context indices are saved in each *_random_paired directory. Frequency and natural activation are undefined for random directions, so both arms use effect-only adjustment. The older *_random files used unmatched random contexts and are retained as exploratory artifacts. Within-arm correlations and pointwise CIs do not establish a difference between correlations, a causal role for crowding, or improved behavioral utility.

| Setting | Vectors | raw count: rho [95% CI] | partial given E_f [95% CI] | C-tilde: rho [95% CI] | partial given E_f [95% CI] | median count | median effect L2 | median C-tilde |
|---|---|---|---|---|---|---|---|---|
| GPT-2-small | SAE features | +0.482 [+0.382, +0.573] | +0.444 [+0.340, +0.539] | +0.344 [+0.237, +0.447] | +0.398 [+0.290, +0.499] | 11.5 | 3.86 | 2.89 |
| GPT-2-small | paired random directions | -0.063 [-0.172, +0.048] | -0.064 [-0.174, +0.047] | -0.077 [-0.187, +0.035] | -0.075 [-0.186, +0.036] | 8.7 | 3.52 | 2.47 |
| Pythia-70M-deduped | SAE features | -0.022 [-0.139, +0.092] | +0.008 [-0.105, +0.124] | +0.031 [-0.085, +0.143] | +0.016 [-0.096, +0.130] | 22.1 | 35.4 | 0.614 |
| Pythia-70M-deduped | paired random directions | +0.009 [-0.104, +0.123] | +0.007 [-0.109, +0.122] | +0.021 [-0.098, +0.139] | +0.013 [-0.106, +0.131] | 21.7 | 33.3 | 0.659 |
| Gemma-2-2B | SAE features | +0.512 [+0.411, +0.603] | +0.267 [+0.152, +0.379] | +0.069 [-0.054, +0.190] | +0.266 [+0.143, +0.378] | 4.3 | 5.48 | 0.759 |
| Gemma-2-2B | paired random directions | -0.012 [-0.127, +0.102] | -0.011 [-0.128, +0.105] | -0.006 [-0.126, +0.113] | -0.015 [-0.123, +0.097] | 3.2 | 5.01 | 0.626 |
| Llama-3.1-8B | SAE features | +0.222 [+0.108, +0.329] | +0.161 [+0.051, +0.270] | +0.201 [+0.090, +0.303] | +0.171 [+0.062, +0.280] | 2.7 | 14 | 0.184 |
| Llama-3.1-8B | paired random directions | -0.058 [-0.172, +0.060] | -0.089 [-0.195, +0.021] | -0.070 [-0.184, +0.051] | -0.091 [-0.197, +0.019] | 1.5 | 12.8 | 0.115 |

### Dense-panel exclusion

Panel features firing in more than 10% of clean contexts are excluded. The remaining panel is still selected by frequency; this check does not provide coverage of all sparse downstream features.

| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) [95% CI] |
|---|---|---|---|---|
| GPT-2-small | 2042 of 2048 | +0.482 | +0.486 | +0.477 [+0.373, +0.568] |
| Pythia-70M-deduped | 1969 of 2048 | -0.022 | -0.093 | -0.093 [-0.212, +0.030] |
| Gemma-2-2B | 2011 of 2048 | +0.512 | +0.515 | +0.196 [+0.073, +0.315] |
| Llama-3.1-8B | 1005 of 1024 | +0.222 | +0.242 | +0.144 [+0.032, +0.256] |

### Residual-space change

Let dh be the downstream hidden-state change and dr the change in its full SAE reconstruction. The unrepresented change is de = dh - dr. For each feature we report 1 - sum_context ||de||^2 / sum_context ||dh||^2, a reconstruction score relative to predicting zero change. It can be negative when reconstruction change is a worse prediction than zero. It is not a bounded variance partition: dr and de need not be orthogonal. The reconstruction norm ratio mean_context ||dr||/||dh|| is reported separately and is not an explained fraction. Neither measure proves behavioral harm or residual-mediated recovery. Per-context dh and dr are saved locally as residual_deltas.npz and verified against the CSV; these large arrays are excluded from git and can be regenerated with --save-residual-deltas.

| Setting | crowding vs ||dh|| [95% CI] | mean reconstruction norm ratio | mean explained-change score | median score | features with score < 0 |
|---|---|---|---|---|---|
| GPT-2-small | +0.328 [+0.220, +0.427] | 0.845 | -0.377 | -0.234 | 78.3% |
| Pythia-70M-deduped | +0.144 [+0.029, +0.260] | 0.778 | +0.466 | +0.452 | 0.0% |
| Gemma-2-2B | +0.626 [+0.536, +0.703] | 2.289 | -13.766 | -13.581 | 100.0% |
| Llama-3.1-8B | +0.172 [+0.056, +0.283] | 1.186 | -2.658 | -2.579 | 100.0% |

### Additional stability summaries

PC1 is the top eigenvalue divided by the trace of the centered logit-change Gram matrix over 48 contexts. Anti-aligned fraction counts contexts whose change has negative cosine with that feature's mean change, including the context in the mean. These are descriptive summaries inspired by FEGA and steering-reliability work; they do not reproduce feature removal or behavior-score multiplier slopes.

| Setting | median anti-aligned context fraction | median centered PC1 ratio |
|---|---|---|
| GPT-2-small | 0.000 | 0.200 |
| Pythia-70M-deduped | 0.000 | 0.336 |
| Gemma-2-2B | 0.104 | 0.448 |
| Llama-3.1-8B | 0.021 | 0.180 |


## 8. Next-token distribution change: KL shift

The same analysis with mean KL(clean || steered) for the next-token distribution as the label. Distribution change does not establish harmful side effects, loss of fluency, or success at an intended behavior.

| Setting | crowding vs KL: raw | partial, primary set | strongest predictor after control |
|---|---|---|---|
| GPT-2-small | +0.290 [+0.177, +0.395] | +0.270 [+0.158, +0.375] | logit_l2 (-0.30, Holm p 2.0e-06) |
| Pythia-70M-deduped | +0.135 [+0.013, +0.249] | +0.086 [-0.047, +0.220] | enc_norm (+0.29, Holm p 5.6e-06) |
| Gemma-2-2B | +0.487 [+0.389, +0.573] | +0.105 [-0.026, +0.227] | logit_entropy (-0.24, Holm p 6.5e-04) |
| Llama-3.1-8B | +0.075 [-0.044, +0.191] | -0.102 [-0.211, +0.012] | logit_entropy (+0.16, Holm p 1.3e-01) |

## 9. Assumptions and limitations

- natural activation a-bar_f := act_mag (mean final-token activation over all contexts).
- intervention value c_f is constant under fixed_global_add (alpha = 1.0) and is dropped.
- ridge alpha = 1.0 on standardised predictors, 5-fold KFold shuffle seed 0.
- nuisance regression and predictor scaling are fitted inside each training fold.
- CV Spearman is undefined for constant or numerically zero predictions; this commonly occurs when a baseline uses only nuisance variables.
- partial rho is undefined when controls exhaust a ranked variable; percentile CIs are undefined if any bootstrap draw is degenerate (n_boot_valid records usable draws).
- bootstrap draws and nuisance fits are shared within each complete predictor family.
- bootstrap RNG is seeded per setting with SeedSequence([seed, first four SHA256 bytes of setting name as little-endian integer]); results do not depend on setting execution order.
- Holm and BH adjustments apply within each setting/target/control family (up to 19 predictors); they do not test whether the top-ranked predictor exceeds the runner-up.
- Feature sampling band [0.002, 0.50] on final-token firing frequency; the paper says only "the non-degenerate range".
- Downstream panel = the most frequently active downstream features on the clean contexts (2048; 1024 for Llama).
- Model loading per setting: GPT-2-small in float32 with TransformerLens weight processing on (default); Pythia-70M-deduped in float32 with TransformerLens weight processing on (default); Gemma-2-2B in float32 with TransformerLens weight processing on (default); Llama-3.1-8B in bfloat16 with TransformerLens weight processing off (from_pretrained_no_processing). Mean L0 is a loading sanity check, not proof that preprocessing matches the SAE training pipeline. Model, dictionary, width, layer, dtype and loading choices vary together, so differences are setting-dependent rather than isolated architecture effects.
- Direct-logit predictors, effect magnitude and stability use the unembedding as loaded: centred for GPT-2 and Pythia, uncentred for Gemma-2 (logit softcap) and for any setting loaded without processing.
- Effect magnitude E_f and the stability cosines are computed on final-token logit differences in float32.
- The collateral count measures downstream SAE activation changes, not independently labeled unrelated behaviors. Reconstruction-residual changes require the separate measurements in Section 7.
- The context-pool split estimates conditional reproducibility; independent corpus draws and independently trained SAE seeds remain untested.
- Coefficient sweeps cover only GPT-2 and Pythia. Fixed-alpha results for Gemma and Llama do not establish coefficient robustness.
- Feature bootstrap intervals condition on the shared dictionary, panel and corpus pool. They do not account for uncertainty from drawing a different model or dictionary.

## 10. Relation to recent work

The literature review was refreshed on 7 September 2026, including papers submitted through 4 September. The claim is a four-setting collateral residualization and robustness study. Pre-intervention side-effect forecasting also appears at the behavioral level in [Ong et al.](https://arxiv.org/abs/2608.11227), and decoder geometry is connected to safety/language intervention costs in [Upadhyaya and Sikdar](https://arxiv.org/abs/2608.29936). We did not locate an exact duplicate of this controlled per-feature analysis; this does not establish universal novelty. See [RELATED_WORK.md](RELATED_WORK.md) and [RECENT_RESEARCH_2026-09-07.md](RECENT_RESEARCH_2026-09-07.md) for source-specific comparisons.

## Files

- `results/<setting>/per_feature.csv`: one row per sampled feature, all predictors and labels.
- `results/<setting>/selection.json`: sampled feature indices, downstream panel indices, and per-feature context indices for the verified reruns.
- `results/<setting>/meta.json`: resolved config, sizes, versions, timings, headline statistics.
- `results/analysis/partial_correlations.csv`, `residualized_cv_ridge.csv`, `seed_summary.csv`, `summary.md`, `crowding_partials.png`.
- `results/analysis/table_collateral_residualized.tex`: LaTeX rows for the appendix.
- `results/analysis/variants.json` and `variants.md`: the robustness block; `audit_gate.json` and `artifact_verification.json`: rerun and independent artifact validation.
