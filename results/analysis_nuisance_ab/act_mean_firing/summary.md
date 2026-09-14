# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.512 | [+0.411, +0.605] | 1.8e-21 | - |
| gemma_2_2b | primary | +0.246 | [+0.130, +0.362] | 1.8e-05 | effect_l2,act_mean_firing,frequency |
| gemma_2_2b | robust | +0.476 | [+0.372, +0.574] | 2.7e-18 | frequency,act_mean_firing |
| gemma_2_2b | effect_only | +0.267 | [+0.150, +0.382] | 2.7e-06 | effect_l2 |
| gpt2_small | none | +0.482 | [+0.381, +0.571] | 7.6e-19 | - |
| gpt2_small | primary | +0.493 | [+0.392, +0.582] | 1.2e-19 | effect_l2,act_mean_firing,frequency |
| gpt2_small | robust | +0.532 | [+0.436, +0.617] | 3.2e-23 | frequency,act_mean_firing |
| gpt2_small | effect_only | +0.444 | [+0.340, +0.540] | 7.5e-16 | effect_l2 |
| llama_3_1_8b | none | +0.222 | [+0.109, +0.332] | 1.1e-04 | - |
| llama_3_1_8b | primary | +0.157 | [+0.044, +0.269] | 6.8e-03 | effect_l2,act_mean_firing,frequency |
| llama_3_1_8b | robust | +0.179 | [+0.066, +0.293] | 1.9e-03 | frequency,act_mean_firing |
| llama_3_1_8b | effect_only | +0.161 | [+0.051, +0.269] | 5.2e-03 | effect_l2 |
| pythia_70m_deduped | none | -0.022 | [-0.137, +0.092] | 7.0e-01 | - |
| pythia_70m_deduped | primary | +0.019 | [-0.101, +0.138] | 7.5e-01 | effect_l2,act_mean_firing,frequency |
| pythia_70m_deduped | robust | -0.015 | [-0.136, +0.107] | 8.0e-01 | frequency,act_mean_firing |
| pythia_70m_deduped | effect_only | +0.008 | [-0.108, +0.119] | 9.0e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.453 | [-0.548, -0.345] | 2.0e-16 | 3.4e-15 |
| gemma_2_2b | coact_entropy | -0.295 | [-0.383, -0.190] | 2.3e-07 | 3.7e-06 |
| gemma_2_2b | crowding | +0.246 | [+0.130, +0.362] | 1.8e-05 | 2.6e-04 |
| gpt2_small | crowding | +0.493 | [+0.392, +0.582] | 1.2e-19 | 2.1e-18 |
| gpt2_small | enc_dec_cos | -0.472 | [-0.558, -0.375] | 6.8e-18 | 1.1e-16 |
| gpt2_small | crowd_max | +0.360 | [+0.247, +0.464] | 1.6e-10 | 2.4e-09 |
| llama_3_1_8b | logit_top10_mass | -0.293 | [-0.391, -0.189] | 2.7e-07 | 4.7e-06 |
| llama_3_1_8b | logit_entropy | +0.241 | [+0.124, +0.351] | 2.6e-05 | 4.2e-04 |
| llama_3_1_8b | crowding | +0.157 | [+0.044, +0.269] | 6.8e-03 | 1.0e-01 |
| pythia_70m_deduped | logit_l2 | +0.344 | [+0.231, +0.450] | 1.1e-09 | 1.9e-08 |
| pythia_70m_deduped | coact_entropy | -0.244 | [-0.342, -0.134] | 2.1e-05 | 3.4e-04 |
| pythia_70m_deduped | act_mag | +0.185 | [+0.072, +0.285] | 1.4e-03 | 2.1e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +0.253 | 0.105 | +0.172 |
| gemma_2_2b | geometry_only | +0.485 | 0.165 | +0.514 |
| gemma_2_2b | direct_logit_only | +0.158 | 0.137 | +0.140 |
| gemma_2_2b | coactivation_only | +0.186 | 0.070 | +0.183 |
| gemma_2_2b | full_no_magnitude | +0.580 | 0.120 | +0.601 |
| gemma_2_2b | full_all | +0.586 | 0.111 | +0.602 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.052 | 0.054 | +0.023 |
| gpt2_small | geometry_only | +0.463 | 0.080 | +0.465 |
| gpt2_small | direct_logit_only | +0.228 | 0.103 | +0.226 |
| gpt2_small | coactivation_only | +0.183 | 0.126 | +0.175 |
| gpt2_small | full_no_magnitude | +0.517 | 0.089 | +0.513 |
| gpt2_small | full_all | +0.514 | 0.094 | +0.526 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | -0.141 | 0.068 | -0.044 |
| llama_3_1_8b | geometry_only | +0.150 | 0.072 | +0.183 |
| llama_3_1_8b | direct_logit_only | +0.318 | 0.130 | +0.310 |
| llama_3_1_8b | coactivation_only | +0.159 | 0.126 | +0.140 |
| llama_3_1_8b | full_no_magnitude | +0.301 | 0.100 | +0.315 |
| llama_3_1_8b | full_all | +0.278 | 0.065 | +0.295 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | +0.097 | 0.145 | +0.061 |
| pythia_70m_deduped | geometry_only | +0.010 | 0.110 | +0.020 |
| pythia_70m_deduped | direct_logit_only | +0.498 | 0.065 | +0.502 |
| pythia_70m_deduped | coactivation_only | +0.196 | 0.134 | +0.213 |
| pythia_70m_deduped | full_no_magnitude | +0.523 | 0.051 | +0.545 |
| pythia_70m_deduped | full_all | +0.529 | 0.061 | +0.542 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.054, +0.192] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.249 | [+0.128, +0.360] | 1.4e-05 | effect_l2,act_mean_firing,frequency |
| gemma_2_2b | robust | +0.038 | [-0.082, +0.160] | 5.1e-01 | frequency,act_mean_firing |
| gemma_2_2b | effect_only | +0.266 | [+0.147, +0.378] | 3.1e-06 | effect_l2 |
| gpt2_small | none | +0.344 | [+0.233, +0.445] | 9.2e-10 | - |
| gpt2_small | primary | +0.439 | [+0.334, +0.535] | 2.0e-15 | effect_l2,act_mean_firing,frequency |
| gpt2_small | robust | +0.371 | [+0.265, +0.468] | 3.9e-11 | frequency,act_mean_firing |
| gpt2_small | effect_only | +0.398 | [+0.289, +0.499] | 8.1e-13 | effect_l2 |
| llama_3_1_8b | none | +0.201 | [+0.092, +0.307] | 4.5e-04 | - |
| llama_3_1_8b | primary | +0.168 | [+0.055, +0.275] | 3.8e-03 | effect_l2,act_mean_firing,frequency |
| llama_3_1_8b | robust | +0.182 | [+0.071, +0.290] | 1.6e-03 | frequency,act_mean_firing |
| llama_3_1_8b | effect_only | +0.171 | [+0.061, +0.278] | 3.0e-03 | effect_l2 |
| pythia_70m_deduped | none | +0.031 | [-0.084, +0.143] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.020 | [-0.102, +0.142] | 7.3e-01 | effect_l2,act_mean_firing,frequency |
| pythia_70m_deduped | robust | +0.032 | [-0.087, +0.148] | 5.8e-01 | frequency,act_mean_firing |
| pythia_70m_deduped | effect_only | +0.016 | [-0.097, +0.130] | 7.8e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.416 | [-0.505, -0.312] | 7.9e-14 | 1.3e-12 |
| gemma_2_2b | coact_entropy | -0.303 | [-0.395, -0.196] | 1.1e-07 | 1.7e-06 |
| gemma_2_2b | crowding | +0.249 | [+0.128, +0.360] | 1.4e-05 | 2.2e-04 |
| gpt2_small | enc_dec_cos | -0.456 | [-0.545, -0.355] | 1.1e-16 | 1.8e-15 |
| gpt2_small | crowding | +0.439 | [+0.334, +0.535] | 2.0e-15 | 3.3e-14 |
| gpt2_small | coact_entropy | -0.366 | [-0.461, -0.257] | 7.5e-11 | 1.1e-09 |
| llama_3_1_8b | logit_top10_mass | -0.298 | [-0.394, -0.192] | 1.7e-07 | 3.0e-06 |
| llama_3_1_8b | logit_entropy | +0.230 | [+0.115, +0.338] | 6.5e-05 | 1.0e-03 |
| llama_3_1_8b | crowding | +0.168 | [+0.055, +0.275] | 3.8e-03 | 5.7e-02 |
| pythia_70m_deduped | logit_l2 | +0.312 | [+0.195, +0.423] | 3.8e-08 | 6.5e-07 |
| pythia_70m_deduped | coact_entropy | -0.195 | [-0.317, -0.061] | 7.5e-04 | 1.2e-02 |
| pythia_70m_deduped | coact_count | +0.173 | [+0.060, +0.277] | 2.9e-03 | 4.3e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +0.158 | 0.219 | +0.147 |
| gemma_2_2b | geometry_only | +0.454 | 0.160 | +0.460 |
| gemma_2_2b | direct_logit_only | +0.106 | 0.128 | +0.085 |
| gemma_2_2b | coactivation_only | +0.158 | 0.131 | +0.144 |
| gemma_2_2b | full_no_magnitude | +0.580 | 0.101 | +0.589 |
| gemma_2_2b | full_all | +0.583 | 0.102 | +0.589 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.045 | 0.046 | +0.020 |
| gpt2_small | geometry_only | +0.461 | 0.080 | +0.458 |
| gpt2_small | direct_logit_only | +0.233 | 0.093 | +0.239 |
| gpt2_small | coactivation_only | +0.169 | 0.135 | +0.165 |
| gpt2_small | full_no_magnitude | +0.517 | 0.097 | +0.513 |
| gpt2_small | full_all | +0.519 | 0.099 | +0.530 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | +0.047 | 0.104 | +0.128 |
| llama_3_1_8b | geometry_only | +0.199 | 0.070 | +0.215 |
| llama_3_1_8b | direct_logit_only | +0.299 | 0.120 | +0.308 |
| llama_3_1_8b | coactivation_only | +0.173 | 0.090 | +0.161 |
| llama_3_1_8b | full_no_magnitude | +0.342 | 0.088 | +0.351 |
| llama_3_1_8b | full_all | +0.317 | 0.083 | +0.340 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | -0.131 | 0.111 | -0.104 |
| pythia_70m_deduped | geometry_only | +0.056 | 0.111 | +0.045 |
| pythia_70m_deduped | direct_logit_only | +0.427 | 0.090 | +0.434 |
| pythia_70m_deduped | coactivation_only | +0.181 | 0.128 | +0.196 |
| pythia_70m_deduped | full_no_magnitude | +0.483 | 0.051 | +0.507 |
| pythia_70m_deduped | full_all | +0.483 | 0.050 | +0.504 |
