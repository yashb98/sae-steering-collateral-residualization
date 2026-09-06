# Collateral-side residualization across four model/SAE settings

Companion results for the revision of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Duan, arXiv:2606.08365). All numbers below are read from `results/` by `analysis/make_results.py`; nothing is copied from the paper except where labelled as the paper's value.

Settings included: GPT-2-small, Pythia-70M-deduped, Gemma-2-2B. Still to run: Llama-3.1-8B.

## Summary

Decoder crowding vs collateral after partialling out the paper's Section 3.8 nuisance set plus firing frequency (partial Spearman, 95% bootstrap CI), and the strongest predictor under that control:

| Setting | Crowding, raw count | Crowding, C-tilde | Strongest predictor (raw count) | Strongest predictor (C-tilde) |
|---|---|---|---|---|
| GPT-2-small | +0.475 [+0.373, +0.568] | +0.425 [+0.316, +0.524] | crowding (+0.48) | crowding (+0.42) |
| Pythia-70M-deduped | -0.004 [-0.124, +0.114] | +0.001 [-0.119, +0.122] | logit_l2 (+0.34) | logit_l2 (+0.31) |
| Gemma-2-2B | +0.242 [+0.123, +0.357] | +0.252 [+0.133, +0.368] | enc_dec_cos (-0.45) | enc_dec_cos (-0.43) |

Reading: crowding carries independent signal in GPT-2-small on both metrics. In Pythia-70M it carries none on either metric, and the direct-logit footprint leads instead. In Gemma-2-2B crowding predicts the raw count strongly before controls but its partial correlation shrinks once effect magnitude is held fixed, and encoder-decoder alignment is the strongest predictor after control. The dominant predictor changes with the setting, which is the paper's own Table 2 pattern, now seen on the collateral axis after the control the paper only ran on stability.

## 1. Setup per setting

| Setting | Model | Primary SAE / hook | Downstream SAE / hook | Panel | Eligible features | Mean L0 (primary / downstream) | Contexts, texts dropped | dtype | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2-small | gpt2 | gpt2-small-res-jb `blocks.8.hook_resid_pre` | `blocks.10.hook_resid_pre` | 2048 | 7532 of 24576 | 71.79 / 60.01 | 2048 x 48 tokens, 36 short texts dropped | float32 | 104 s on NVIDIA GB10 |
| Pythia-70M-deduped | pythia-70m-deduped | pythia-70m-deduped-res-sm `blocks.4.hook_resid_post` | `blocks.5.hook_resid_post` | 2048 | 4375 of 32768 | 68.33 / 118.56 | 2048 x 48 tokens, 36 short texts dropped | float32 | 89 s on NVIDIA GB10 |
| Gemma-2-2B | google/gemma-2-2b | gemma-scope-2b-pt-res-canonical `layer_12/width_16k/canonical` at `blocks.12.hook_resid_post` | `layer_16/width_16k/canonical` at `blocks.16.hook_resid_post` | 2048 | 7517 of 16384 | 83.04 / 79.25 | 2048 x 48 tokens, 32 short texts dropped | float32 | 1151 s on NVIDIA GB10 |

Common: Wikitext-103 train split, 8,000 texts, 300 features sampled with seed 0 from the final-token firing-frequency band [0.002, 0.50], 16 contexts per type (top / random / low), additive steering alpha = 1.0 at the final token, tau = 0.05, epsilon_fire = 1e-6, crowding = top-20 mean absolute cosine, protocol v2 (no pad final tokens, whitespace-normalised dedup).

## 2. Decoder crowding vs collateral, with and without controls

Spearman rho; partial correlations are rank-based (both sides OLS-residualised on the ranked controls). 95% CIs from 10000 bootstrap resamples of the 300 features. Controls: **primary** = effect magnitude E_f, intervention value (constant, dropped), natural activation (mean final-token activation), firing frequency; **robust** = frequency and activation magnitude only.

### Target: raw downstream count C_{f,0.05}

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.482 [+0.381, +0.574] | +0.517 [+0.418, +0.605] | +0.475 [+0.373, +0.568] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | -0.022 [-0.136, +0.095] | -0.026 [-0.148, +0.097] | -0.004 [-0.124, +0.114] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.512 [+0.411, +0.600] | +0.468 [+0.360, +0.567] | +0.242 [+0.123, +0.357] | encoder norm, signed stability, rho = 0.350 |

### Target: C-tilde = C / (E_f + eps)

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.344 [+0.234, +0.447] | +0.355 [+0.245, +0.457] | +0.425 [+0.316, +0.524] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | +0.031 [-0.083, +0.142] | +0.011 [-0.104, +0.129] | +0.001 [-0.119, +0.122] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |
| Gemma-2-2B | +0.069 [-0.052, +0.189] | +0.052 [-0.071, +0.177] | +0.252 [+0.133, +0.368] | encoder norm, signed stability, rho = 0.350 |

## 3. Strongest predictors under the primary control

Top three predictors by absolute partial rho after the primary control, per setting and metric.

### Target: raw downstream count C_{f,0.05}

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.475 [+0.373, +0.568] | 3.7e-18 |
| GPT-2-small | enc_dec_cos | -0.401 [-0.499, -0.295] | 6.5e-13 |
| GPT-2-small | crowd_max | +0.333 [+0.218, +0.444] | 4.1e-09 |
| Pythia-70M-deduped | logit_l2 | +0.343 [+0.227, +0.450] | 1.3e-09 |
| Pythia-70M-deduped | act_mean_firing | -0.207 [-0.314, -0.084] | 3.3e-04 |
| Pythia-70M-deduped | coact_entropy | -0.202 [-0.297, -0.082] | 4.8e-04 |
| Gemma-2-2B | enc_dec_cos | -0.454 [-0.552, -0.342] | 1.6e-16 |
| Gemma-2-2B | coact_entropy | -0.299 [-0.389, -0.196] | 1.6e-07 |
| Gemma-2-2B | crowding | +0.242 [+0.123, +0.357] | 2.5e-05 |

### Target: C-tilde = C / (E_f + eps)

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.425 [+0.316, +0.524] | 1.9e-14 |
| GPT-2-small | enc_dec_cos | -0.399 [-0.498, -0.290] | 9.1e-13 |
| GPT-2-small | coact_entropy | -0.345 [-0.439, -0.232] | 9.9e-10 |
| Pythia-70M-deduped | logit_l2 | +0.311 [+0.195, +0.424] | 4.2e-08 |
| Pythia-70M-deduped | coact_count | +0.198 [+0.087, +0.302] | 6.1e-04 |
| Pythia-70M-deduped | act_mean_firing | -0.171 [-0.273, -0.058] | 3.2e-03 |
| Gemma-2-2B | enc_dec_cos | -0.427 [-0.520, -0.320] | 1.5e-14 |
| Gemma-2-2B | coact_entropy | -0.311 [-0.401, -0.204] | 4.4e-08 |
| Gemma-2-2B | crowding | +0.252 [+0.133, +0.368] | 1.1e-05 |

## 4. Table B3 analog: predictor sets on the residualized collateral target

The collateral label is OLS-residualised against the primary control set, then predicted with ridge (alpha = 1, standardised predictors) under 5-fold cross-validation; score = Spearman between held-out predictions and residualised label, mean over folds (sd in parentheses).

### Target: raw downstream count C_{f,0.05}

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.015 (0.08) | -0.035 (0.06) | +0.458 (0.06) | +0.208 (0.10) | +0.182 (0.11) | +0.502 (0.08) | +0.493 (0.09) |
| Pythia-70M-deduped | -0.018 (0.14) | -0.113 (0.10) | +0.011 (0.08) | +0.493 (0.08) | +0.184 (0.09) | +0.517 (0.05) | +0.522 (0.05) |
| Gemma-2-2B | -0.144 (0.27) | -0.106 (0.29) | +0.468 (0.16) | +0.137 (0.15) | +0.168 (0.09) | +0.588 (0.11) | +0.590 (0.11) |

### Target: C-tilde = C / (E_f + eps)

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.027 (0.08) | -0.029 (0.05) | +0.447 (0.07) | +0.216 (0.10) | +0.176 (0.11) | +0.498 (0.09) | +0.489 (0.10) |
| Pythia-70M-deduped | -0.118 (0.09) | -0.116 (0.11) | +0.044 (0.09) | +0.426 (0.09) | +0.162 (0.09) | +0.477 (0.05) | +0.482 (0.05) |
| Gemma-2-2B | -0.127 (0.24) | -0.095 (0.27) | +0.429 (0.16) | +0.104 (0.14) | +0.144 (0.14) | +0.578 (0.11) | +0.582 (0.11) |

## 5. Feature-sample robustness (crowding)

Same contexts, different random sample of 300 features (seeds 0, 1, 2). Values are crowding vs collateral rho.

| Setting | Target | Control | seed 0 | seed 1 | seed 2 | mean | sd |
|---|---|---|---|---|---|---|---|
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
| Gemma-2-2B | | seed 0 only so far | | | | | |

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

## 7. Assumptions and conventions

- natural activation a-bar_f := act_mag (mean final-token activation over all contexts).
- intervention value c_f is constant under fixed_global_add (alpha = 1.0) and is dropped.
- ridge alpha = 1.0 on standardised predictors, 5-fold KFold shuffle seed 0.
- Feature sampling band [0.002, 0.50] on final-token firing frequency; the paper says only "the non-degenerate range".
- Downstream panel = the most frequently active downstream features on the clean contexts (2048; 1024 for Llama).
- Models loaded with TransformerLens default weight processing plus each SAE's `model_from_pretrained_kwargs`, in float32. TransformerLens centres writing weights only for LayerNorm models, so Gemma-2 and Llama residual streams match the Hugging Face values the Gemma Scope and Llama Scope SAEs were trained on; the mean L0 column above is the empirical check.
- Direct-logit predictors use the model's unembedding as loaded (centred for GPT-2, Pythia and Llama; Gemma-2 keeps its logit softcap, so its unembedding is not centred).
- Effect magnitude E_f and the stability cosines are computed on final-token logit differences in float32.

## Files

- `results/<setting>/per_feature.csv`: one row per sampled feature, all predictors and labels.
- `results/<setting>/selection.json`: sampled feature indices and downstream panel indices.
- `results/<setting>/meta.json`: resolved config, sizes, versions, timings, headline statistics.
- `results/analysis/partial_correlations.csv`, `residualized_cv_ridge.csv`, `seed_summary.csv`, `summary.md`, `crowding_partials.png`.
- `results/analysis/table_collateral_residualized.tex`: LaTeX rows for the appendix.
