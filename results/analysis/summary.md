# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.512 | [+0.411, +0.600] | 1.8e-21 | - |
| gemma_2_2b | primary | +0.242 | [+0.123, +0.357] | 2.5e-05 | effect_l2,act_mag,frequency |
| gemma_2_2b | robust | +0.468 | [+0.360, +0.567] | 1.2e-17 | frequency,act_mag |
| gemma_2_2b | effect_only | +0.267 | [+0.151, +0.382] | 2.7e-06 | effect_l2 |
| gpt2_small | none | +0.482 | [+0.381, +0.571] | 7.6e-19 | - |
| gpt2_small | primary | +0.475 | [+0.372, +0.567] | 3.7e-18 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.517 | [+0.417, +0.605] | 8.2e-22 | frequency,act_mag |
| gpt2_small | effect_only | +0.444 | [+0.340, +0.540] | 7.5e-16 | effect_l2 |
| llama_3_1_8b | none | +0.222 | [+0.108, +0.330] | 1.1e-04 | - |
| llama_3_1_8b | primary | +0.152 | [+0.038, +0.267] | 8.7e-03 | effect_l2,act_mag,frequency |
| llama_3_1_8b | robust | +0.182 | [+0.066, +0.293] | 1.6e-03 | frequency,act_mag |
| llama_3_1_8b | effect_only | +0.161 | [+0.050, +0.268] | 5.2e-03 | effect_l2 |
| pythia_70m_deduped | none | -0.022 | [-0.136, +0.091] | 7.0e-01 | - |
| pythia_70m_deduped | primary | -0.004 | [-0.124, +0.116] | 9.4e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | -0.026 | [-0.147, +0.096] | 6.6e-01 | frequency,act_mag |
| pythia_70m_deduped | effect_only | +0.008 | [-0.106, +0.124] | 9.0e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.454 | [-0.552, -0.342] | 1.6e-16 | 2.8e-15 |
| gemma_2_2b | coact_entropy | -0.299 | [-0.389, -0.196] | 1.6e-07 | 2.5e-06 |
| gemma_2_2b | crowding | +0.242 | [+0.123, +0.357] | 2.5e-05 | 3.7e-04 |
| gpt2_small | crowding | +0.475 | [+0.372, +0.567] | 3.7e-18 | 6.3e-17 |
| gpt2_small | enc_dec_cos | -0.401 | [-0.499, -0.294] | 6.5e-13 | 1.0e-11 |
| gpt2_small | crowd_max | +0.333 | [+0.217, +0.441] | 4.1e-09 | 6.2e-08 |
| llama_3_1_8b | logit_top10_mass | -0.293 | [-0.390, -0.188] | 2.6e-07 | 4.4e-06 |
| llama_3_1_8b | logit_entropy | +0.243 | [+0.128, +0.351] | 2.3e-05 | 3.7e-04 |
| llama_3_1_8b | act_entropy | +0.159 | [+0.051, +0.254] | 5.9e-03 | 8.8e-02 |
| pythia_70m_deduped | logit_l2 | +0.343 | [+0.229, +0.447] | 1.3e-09 | 2.3e-08 |
| pythia_70m_deduped | act_mean_firing | -0.207 | [-0.315, -0.081] | 3.3e-04 | 5.3e-03 |
| pythia_70m_deduped | coact_entropy | -0.202 | [-0.302, -0.084] | 4.8e-04 | 7.1e-03 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | -0.144 | 0.268 | -0.186 |
| gemma_2_2b | actmag_only | -0.106 | 0.292 | -0.179 |
| gemma_2_2b | geometry_only | +0.468 | 0.156 | +0.483 |
| gemma_2_2b | direct_logit_only | +0.137 | 0.149 | +0.106 |
| gemma_2_2b | coactivation_only | +0.168 | 0.090 | +0.156 |
| gemma_2_2b | full_no_magnitude | +0.591 | 0.108 | +0.600 |
| gemma_2_2b | full_all | +0.589 | 0.108 | +0.600 |
| gpt2_small | frequency_only | -0.015 | 0.084 | -0.138 |
| gpt2_small | actmag_only | -0.035 | 0.058 | -0.131 |
| gpt2_small | geometry_only | +0.458 | 0.064 | +0.445 |
| gpt2_small | direct_logit_only | +0.208 | 0.102 | +0.192 |
| gpt2_small | coactivation_only | +0.182 | 0.106 | +0.166 |
| gpt2_small | full_no_magnitude | +0.504 | 0.080 | +0.495 |
| gpt2_small | full_all | +0.494 | 0.088 | +0.501 |
| llama_3_1_8b | frequency_only | -0.031 | 0.111 | -0.185 |
| llama_3_1_8b | actmag_only | -0.080 | 0.128 | -0.191 |
| llama_3_1_8b | geometry_only | +0.099 | 0.038 | +0.061 |
| llama_3_1_8b | direct_logit_only | +0.300 | 0.134 | +0.255 |
| llama_3_1_8b | coactivation_only | +0.114 | 0.048 | +0.037 |
| llama_3_1_8b | full_no_magnitude | +0.263 | 0.116 | +0.246 |
| llama_3_1_8b | full_all | +0.268 | 0.070 | +0.256 |
| pythia_70m_deduped | frequency_only | -0.018 | 0.140 | -0.185 |
| pythia_70m_deduped | actmag_only | -0.113 | 0.099 | -0.204 |
| pythia_70m_deduped | geometry_only | +0.011 | 0.081 | -0.020 |
| pythia_70m_deduped | direct_logit_only | +0.493 | 0.083 | +0.495 |
| pythia_70m_deduped | coactivation_only | +0.184 | 0.095 | +0.180 |
| pythia_70m_deduped | full_no_magnitude | +0.518 | 0.050 | +0.537 |
| pythia_70m_deduped | full_all | +0.522 | 0.051 | +0.536 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.051, +0.189] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.252 | [+0.129, +0.368] | 1.1e-05 | effect_l2,act_mag,frequency |
| gemma_2_2b | robust | +0.052 | [-0.071, +0.177] | 3.7e-01 | frequency,act_mag |
| gemma_2_2b | effect_only | +0.266 | [+0.149, +0.376] | 3.1e-06 | effect_l2 |
| gpt2_small | none | +0.344 | [+0.234, +0.447] | 9.2e-10 | - |
| gpt2_small | primary | +0.425 | [+0.314, +0.523] | 1.9e-14 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.355 | [+0.246, +0.457] | 2.9e-10 | frequency,act_mag |
| gpt2_small | effect_only | +0.398 | [+0.291, +0.497] | 8.1e-13 | effect_l2 |
| llama_3_1_8b | none | +0.201 | [+0.092, +0.305] | 4.5e-04 | - |
| llama_3_1_8b | primary | +0.166 | [+0.053, +0.276] | 4.2e-03 | effect_l2,act_mag,frequency |
| llama_3_1_8b | robust | +0.183 | [+0.070, +0.291] | 1.5e-03 | frequency,act_mag |
| llama_3_1_8b | effect_only | +0.171 | [+0.059, +0.276] | 3.0e-03 | effect_l2 |
| pythia_70m_deduped | none | +0.031 | [-0.082, +0.145] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.001 | [-0.119, +0.120] | 9.8e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | +0.011 | [-0.105, +0.127] | 8.5e-01 | frequency,act_mag |
| pythia_70m_deduped | effect_only | +0.016 | [-0.097, +0.128] | 7.8e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.427 | [-0.520, -0.325] | 1.5e-14 | 2.5e-13 |
| gemma_2_2b | coact_entropy | -0.311 | [-0.402, -0.201] | 4.4e-08 | 7.1e-07 |
| gemma_2_2b | crowding | +0.252 | [+0.129, +0.368] | 1.1e-05 | 1.6e-04 |
| gpt2_small | crowding | +0.425 | [+0.314, +0.523] | 1.9e-14 | 3.3e-13 |
| gpt2_small | enc_dec_cos | -0.399 | [-0.495, -0.294] | 9.1e-13 | 1.5e-11 |
| gpt2_small | coact_entropy | -0.345 | [-0.443, -0.232] | 9.9e-10 | 1.5e-08 |
| llama_3_1_8b | logit_top10_mass | -0.298 | [-0.394, -0.192] | 1.7e-07 | 2.9e-06 |
| llama_3_1_8b | logit_entropy | +0.231 | [+0.115, +0.344] | 6.1e-05 | 9.7e-04 |
| llama_3_1_8b | crowding | +0.166 | [+0.053, +0.276] | 4.2e-03 | 6.3e-02 |
| pythia_70m_deduped | logit_l2 | +0.311 | [+0.193, +0.421] | 4.2e-08 | 7.2e-07 |
| pythia_70m_deduped | coact_count | +0.198 | [+0.085, +0.307] | 6.1e-04 | 9.8e-03 |
| pythia_70m_deduped | act_mean_firing | -0.171 | [-0.275, -0.057] | 3.2e-03 | 4.8e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | -0.127 | 0.245 | -0.137 |
| gemma_2_2b | actmag_only | -0.095 | 0.266 | -0.135 |
| gemma_2_2b | geometry_only | +0.429 | 0.165 | +0.430 |
| gemma_2_2b | direct_logit_only | +0.104 | 0.140 | +0.065 |
| gemma_2_2b | coactivation_only | +0.144 | 0.141 | +0.123 |
| gemma_2_2b | full_no_magnitude | +0.579 | 0.105 | +0.584 |
| gemma_2_2b | full_all | +0.586 | 0.112 | +0.591 |
| gpt2_small | frequency_only | -0.027 | 0.077 | -0.132 |
| gpt2_small | actmag_only | -0.029 | 0.046 | -0.129 |
| gpt2_small | geometry_only | +0.447 | 0.075 | +0.437 |
| gpt2_small | direct_logit_only | +0.216 | 0.100 | +0.206 |
| gpt2_small | coactivation_only | +0.176 | 0.114 | +0.159 |
| gpt2_small | full_no_magnitude | +0.500 | 0.089 | +0.495 |
| gpt2_small | full_all | +0.492 | 0.094 | +0.502 |
| llama_3_1_8b | frequency_only | -0.037 | 0.108 | -0.172 |
| llama_3_1_8b | actmag_only | -0.075 | 0.094 | -0.184 |
| llama_3_1_8b | geometry_only | +0.160 | 0.055 | +0.143 |
| llama_3_1_8b | direct_logit_only | +0.309 | 0.130 | +0.277 |
| llama_3_1_8b | coactivation_only | +0.164 | 0.084 | +0.090 |
| llama_3_1_8b | full_no_magnitude | +0.312 | 0.073 | +0.312 |
| llama_3_1_8b | full_all | +0.294 | 0.063 | +0.297 |
| pythia_70m_deduped | frequency_only | -0.118 | 0.090 | -0.187 |
| pythia_70m_deduped | actmag_only | -0.116 | 0.112 | -0.182 |
| pythia_70m_deduped | geometry_only | +0.044 | 0.092 | +0.005 |
| pythia_70m_deduped | direct_logit_only | +0.426 | 0.094 | +0.426 |
| pythia_70m_deduped | coactivation_only | +0.162 | 0.088 | +0.157 |
| pythia_70m_deduped | full_no_magnitude | +0.475 | 0.046 | +0.501 |
| pythia_70m_deduped | full_all | +0.478 | 0.052 | +0.499 |
