# Collateral-side residualization across four model/SAE settings

Companion results for the revision of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Duan, arXiv:2606.08365). All numbers below are read from `results/` by `analysis/make_results.py`; nothing is copied from the paper except where labelled as the paper's value.

Settings included: GPT-2-small, Pythia-70M-deduped. Still to run: Gemma-2-2B, Llama-3.1-8B.

## 1. Setup per setting

| Setting | Model | Primary SAE / hook | Downstream SAE / hook | Panel | Eligible features | Mean L0 (primary / downstream) | Contexts, texts dropped | dtype | Wall clock |
|---|---|---|---|---|---|---|---|---|---|
| GPT-2-small | gpt2 | gpt2-small-res-jb `blocks.8.hook_resid_pre` | `blocks.10.hook_resid_pre` | 2048 | 7532 of 24576 | 71.79 / 60.01 | 2048 x 48 tokens, 36 short texts dropped | float32 | 104 s on NVIDIA GB10 |
| Pythia-70M-deduped | pythia-70m-deduped | pythia-70m-deduped-res-sm `blocks.4.hook_resid_post` | `blocks.5.hook_resid_post` | 2048 | 4375 of 32768 | 68.33 / 118.56 | 2048 x 48 tokens, 36 short texts dropped | float32 | 89 s on NVIDIA GB10 |

Common: Wikitext-103 train split, 8,000 texts, 300 features sampled with seed 0 from the final-token firing-frequency band [0.002, 0.50], 16 contexts per type (top / random / low), additive steering alpha = 1.0 at the final token, tau = 0.05, epsilon_fire = 1e-6, crowding = top-20 mean absolute cosine, protocol v2 (no pad final tokens, whitespace-normalised dedup).

## 2. Decoder crowding vs collateral, with and without controls

Spearman rho; partial correlations are rank-based (both sides OLS-residualised on the ranked controls). 95% CIs from 10000 bootstrap resamples of the 300 features. Controls: **primary** = effect magnitude E_f, intervention value (constant, dropped), natural activation (mean final-token activation), firing frequency; **robust** = frequency and activation magnitude only.

### Target: raw downstream count C_{f,0.05}

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.482 [+0.381, +0.569] | +0.517 [+0.418, +0.604] | +0.475 [+0.372, +0.567] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | -0.022 [-0.137, +0.092] | -0.026 [-0.144, +0.097] | -0.004 [-0.126, +0.116] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |

### Target: C-tilde = C / (E_f + eps)

| Setting | raw Spearman | partial, robust pair | partial, primary set | Paper Table 2 (for orientation) |
|---|---|---|---|---|
| GPT-2-small | +0.344 [+0.232, +0.446] | +0.355 [+0.246, +0.457] | +0.425 [+0.319, +0.527] | decoder crowding, downstream count, rho = 0.466 |
| Pythia-70M-deduped | +0.031 [-0.082, +0.144] | +0.011 [-0.107, +0.128] | +0.001 [-0.118, +0.122] | direct-logit L2, downstream count, rho = 0.391 (crowding leads stability, rho = -0.360) |

## 3. Strongest predictors under the primary control

Top three predictors by absolute partial rho after the primary control, per setting and metric.

### Target: raw downstream count C_{f,0.05}

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.475 [+0.372, +0.567] | 3.7e-18 |
| GPT-2-small | enc_dec_cos | -0.401 [-0.498, -0.293] | 6.5e-13 |
| GPT-2-small | crowd_max | +0.333 [+0.216, +0.442] | 4.1e-09 |
| Pythia-70M-deduped | logit_l2 | +0.343 [+0.230, +0.448] | 1.3e-09 |
| Pythia-70M-deduped | act_mean_firing | -0.207 [-0.314, -0.087] | 3.3e-04 |
| Pythia-70M-deduped | coact_entropy | -0.202 [-0.303, -0.083] | 4.8e-04 |

### Target: C-tilde = C / (E_f + eps)

| Setting | Predictor | partial rho [95% CI] | p |
|---|---|---|---|
| GPT-2-small | crowding | +0.425 [+0.319, +0.527] | 1.9e-14 |
| GPT-2-small | enc_dec_cos | -0.399 [-0.496, -0.292] | 9.1e-13 |
| GPT-2-small | coact_entropy | -0.345 [-0.442, -0.235] | 9.9e-10 |
| Pythia-70M-deduped | logit_l2 | +0.311 [+0.194, +0.420] | 4.2e-08 |
| Pythia-70M-deduped | coact_count | +0.198 [+0.088, +0.305] | 6.1e-04 |
| Pythia-70M-deduped | act_mean_firing | -0.171 [-0.272, -0.059] | 3.2e-03 |

## 4. Table B3 analog: predictor sets on the residualized collateral target

The collateral label is OLS-residualised against the primary control set, then predicted with ridge (alpha = 1, standardised predictors) under 5-fold cross-validation; score = Spearman between held-out predictions and residualised label, mean over folds (sd in parentheses).

### Target: raw downstream count C_{f,0.05}

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.015 (0.08) | -0.035 (0.06) | +0.458 (0.06) | +0.208 (0.10) | +0.182 (0.11) | +0.502 (0.08) | +0.493 (0.09) |
| Pythia-70M-deduped | -0.018 (0.14) | -0.113 (0.10) | +0.011 (0.08) | +0.493 (0.08) | +0.184 (0.09) | +0.517 (0.05) | +0.522 (0.05) |

### Target: C-tilde = C / (E_f + eps)

| Setting | frequency_only | actmag_only | geometry_only | direct_logit_only | coactivation_only | full_no_magnitude | full_all |
|---|---|---|---|---|---|---|---|
| GPT-2-small | -0.027 (0.08) | -0.029 (0.05) | +0.447 (0.07) | +0.216 (0.10) | +0.176 (0.11) | +0.498 (0.09) | +0.489 (0.10) |
| Pythia-70M-deduped | -0.118 (0.09) | -0.116 (0.11) | +0.044 (0.09) | +0.426 (0.09) | +0.162 (0.09) | +0.477 (0.05) | +0.482 (0.05) |

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

## 6. Regression gate against the published GPT-2-small notebook

`src/run_setting.py --protocol v1` reproduces the Kaggle notebook's context construction exactly. Same seed, same eligible-feature count (7469). Differences are GPU floating-point noise (T4 vs GB10).

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
