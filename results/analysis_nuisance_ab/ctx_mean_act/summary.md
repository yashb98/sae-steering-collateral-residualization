# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.512 | [+0.411, +0.605] | 1.8e-21 | - |
| gemma_2_2b | primary | +0.240 | [+0.119, +0.359] | 2.9e-05 | effect_l2,ctx_mean_act,frequency |
| gemma_2_2b | robust | +0.466 | [+0.363, +0.564] | 1.8e-17 | frequency,ctx_mean_act |
| gemma_2_2b | effect_only | +0.267 | [+0.150, +0.382] | 2.7e-06 | effect_l2 |
| gpt2_small | none | +0.482 | [+0.381, +0.571] | 7.6e-19 | - |
| gpt2_small | primary | +0.482 | [+0.381, +0.572] | 1.1e-18 | effect_l2,ctx_mean_act,frequency |
| gpt2_small | robust | +0.523 | [+0.425, +0.608] | 2.8e-22 | frequency,ctx_mean_act |
| gpt2_small | effect_only | +0.444 | [+0.340, +0.540] | 7.5e-16 | effect_l2 |
| llama_3_1_8b | none | +0.222 | [+0.109, +0.332] | 1.1e-04 | - |
| llama_3_1_8b | primary | +0.157 | [+0.045, +0.271] | 6.7e-03 | effect_l2,ctx_mean_act,frequency |
| llama_3_1_8b | robust | +0.184 | [+0.069, +0.299] | 1.4e-03 | frequency,ctx_mean_act |
| llama_3_1_8b | effect_only | +0.161 | [+0.051, +0.269] | 5.2e-03 | effect_l2 |
| pythia_70m_deduped | none | -0.022 | [-0.137, +0.092] | 7.0e-01 | - |
| pythia_70m_deduped | primary | +0.003 | [-0.116, +0.122] | 9.6e-01 | effect_l2,ctx_mean_act,frequency |
| pythia_70m_deduped | robust | -0.026 | [-0.148, +0.098] | 6.6e-01 | frequency,ctx_mean_act |
| pythia_70m_deduped | effect_only | +0.008 | [-0.108, +0.119] | 9.0e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.450 | [-0.548, -0.338] | 3.3e-16 | 5.9e-15 |
| gemma_2_2b | coact_entropy | -0.297 | [-0.386, -0.192] | 1.9e-07 | 3.2e-06 |
| gemma_2_2b | crowding | +0.240 | [+0.119, +0.359] | 2.9e-05 | 4.7e-04 |
| gpt2_small | crowding | +0.482 | [+0.381, +0.572] | 1.1e-18 | 1.9e-17 |
| gpt2_small | enc_dec_cos | -0.424 | [-0.517, -0.321] | 2.1e-14 | 3.5e-13 |
| gpt2_small | crowd_max | +0.344 | [+0.228, +0.449] | 1.1e-09 | 1.8e-08 |
| llama_3_1_8b | logit_top10_mass | -0.293 | [-0.391, -0.190] | 2.6e-07 | 4.8e-06 |
| llama_3_1_8b | logit_entropy | +0.242 | [+0.125, +0.352] | 2.5e-05 | 4.2e-04 |
| llama_3_1_8b | act_mag | +0.165 | [-0.037, +0.292] | 4.4e-03 | 7.0e-02 |
| pythia_70m_deduped | logit_l2 | +0.344 | [+0.231, +0.450] | 1.2e-09 | 2.1e-08 |
| pythia_70m_deduped | coact_entropy | -0.216 | [-0.313, -0.103] | 1.7e-04 | 2.9e-03 |
| pythia_70m_deduped | act_mean_firing | -0.182 | [-0.277, -0.076] | 1.7e-03 | 2.7e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | -0.101 | 0.094 | -0.136 |
| gemma_2_2b | geometry_only | +0.482 | 0.145 | +0.506 |
| gemma_2_2b | direct_logit_only | +0.146 | 0.114 | +0.140 |
| gemma_2_2b | coactivation_only | +0.097 | 0.056 | +0.089 |
| gemma_2_2b | full_no_magnitude | +0.526 | 0.139 | +0.550 |
| gemma_2_2b | full_all | +0.534 | 0.134 | +0.561 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.007 | 0.084 | +0.025 |
| gpt2_small | geometry_only | +0.454 | 0.072 | +0.456 |
| gpt2_small | direct_logit_only | +0.202 | 0.110 | +0.206 |
| gpt2_small | coactivation_only | +0.198 | 0.105 | +0.191 |
| gpt2_small | full_no_magnitude | +0.513 | 0.085 | +0.511 |
| gpt2_small | full_all | +0.502 | 0.093 | +0.514 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | -0.100 | 0.103 | +0.002 |
| llama_3_1_8b | geometry_only | +0.137 | 0.098 | +0.161 |
| llama_3_1_8b | direct_logit_only | +0.309 | 0.128 | +0.291 |
| llama_3_1_8b | coactivation_only | +0.062 | 0.125 | +0.024 |
| llama_3_1_8b | full_no_magnitude | +0.273 | 0.092 | +0.285 |
| llama_3_1_8b | full_all | +0.279 | 0.080 | +0.289 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | -0.049 | 0.132 | -0.024 |
| pythia_70m_deduped | geometry_only | +0.018 | 0.086 | +0.027 |
| pythia_70m_deduped | direct_logit_only | +0.498 | 0.065 | +0.501 |
| pythia_70m_deduped | coactivation_only | +0.221 | 0.124 | +0.241 |
| pythia_70m_deduped | full_no_magnitude | +0.523 | 0.054 | +0.546 |
| pythia_70m_deduped | full_all | +0.533 | 0.049 | +0.545 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.054, +0.192] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.249 | [+0.127, +0.362] | 1.4e-05 | effect_l2,ctx_mean_act,frequency |
| gemma_2_2b | robust | +0.049 | [-0.079, +0.175] | 4.0e-01 | frequency,ctx_mean_act |
| gemma_2_2b | effect_only | +0.266 | [+0.147, +0.378] | 3.1e-06 | effect_l2 |
| gpt2_small | none | +0.344 | [+0.233, +0.445] | 9.2e-10 | - |
| gpt2_small | primary | +0.430 | [+0.324, +0.528] | 9.1e-15 | effect_l2,ctx_mean_act,frequency |
| gpt2_small | robust | +0.361 | [+0.254, +0.459] | 1.4e-10 | frequency,ctx_mean_act |
| gpt2_small | effect_only | +0.398 | [+0.289, +0.499] | 8.1e-13 | effect_l2 |
| llama_3_1_8b | none | +0.201 | [+0.092, +0.307] | 4.5e-04 | - |
| llama_3_1_8b | primary | +0.170 | [+0.056, +0.279] | 3.3e-03 | effect_l2,ctx_mean_act,frequency |
| llama_3_1_8b | robust | +0.186 | [+0.073, +0.295] | 1.2e-03 | frequency,ctx_mean_act |
| llama_3_1_8b | effect_only | +0.171 | [+0.061, +0.278] | 3.0e-03 | effect_l2 |
| pythia_70m_deduped | none | +0.031 | [-0.084, +0.143] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.007 | [-0.115, +0.128] | 9.1e-01 | effect_l2,ctx_mean_act,frequency |
| pythia_70m_deduped | robust | +0.019 | [-0.099, +0.134] | 7.5e-01 | frequency,ctx_mean_act |
| pythia_70m_deduped | effect_only | +0.016 | [-0.097, +0.130] | 7.8e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.425 | [-0.518, -0.316] | 1.9e-14 | 3.4e-13 |
| gemma_2_2b | coact_entropy | -0.309 | [-0.400, -0.200] | 5.7e-08 | 9.7e-07 |
| gemma_2_2b | crowding | +0.249 | [+0.127, +0.362] | 1.4e-05 | 2.2e-04 |
| gpt2_small | crowding | +0.430 | [+0.324, +0.528] | 9.1e-15 | 1.6e-13 |
| gpt2_small | enc_dec_cos | -0.417 | [-0.513, -0.310] | 6.7e-14 | 1.1e-12 |
| gpt2_small | coact_entropy | -0.352 | [-0.448, -0.242] | 4.5e-10 | 7.2e-09 |
| llama_3_1_8b | logit_top10_mass | -0.298 | [-0.394, -0.192] | 1.7e-07 | 3.1e-06 |
| llama_3_1_8b | logit_entropy | +0.230 | [+0.115, +0.338] | 6.3e-05 | 1.1e-03 |
| llama_3_1_8b | crowding | +0.170 | [+0.056, +0.279] | 3.3e-03 | 5.3e-02 |
| pythia_70m_deduped | logit_l2 | +0.312 | [+0.196, +0.424] | 3.9e-08 | 7.0e-07 |
| pythia_70m_deduped | coact_count | +0.190 | [+0.078, +0.295] | 1.0e-03 | 1.7e-02 |
| pythia_70m_deduped | coact_entropy | -0.170 | [-0.291, -0.036] | 3.3e-03 | 5.3e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | -0.101 | 0.049 | -0.112 |
| gemma_2_2b | geometry_only | +0.460 | 0.129 | +0.462 |
| gemma_2_2b | direct_logit_only | +0.110 | 0.103 | +0.092 |
| gemma_2_2b | coactivation_only | +0.053 | 0.081 | +0.034 |
| gemma_2_2b | full_no_magnitude | +0.542 | 0.122 | +0.550 |
| gemma_2_2b | full_all | +0.550 | 0.125 | +0.555 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.001 | 0.065 | +0.013 |
| gpt2_small | geometry_only | +0.451 | 0.076 | +0.450 |
| gpt2_small | direct_logit_only | +0.214 | 0.105 | +0.223 |
| gpt2_small | coactivation_only | +0.188 | 0.118 | +0.183 |
| gpt2_small | full_no_magnitude | +0.513 | 0.094 | +0.511 |
| gpt2_small | full_all | +0.508 | 0.100 | +0.523 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | +0.001 | 0.103 | +0.051 |
| llama_3_1_8b | geometry_only | +0.178 | 0.091 | +0.197 |
| llama_3_1_8b | direct_logit_only | +0.282 | 0.123 | +0.280 |
| llama_3_1_8b | coactivation_only | +0.100 | 0.106 | +0.080 |
| llama_3_1_8b | full_no_magnitude | +0.327 | 0.102 | +0.340 |
| llama_3_1_8b | full_all | +0.309 | 0.104 | +0.333 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | -0.064 | 0.121 | -0.033 |
| pythia_70m_deduped | geometry_only | +0.038 | 0.089 | +0.043 |
| pythia_70m_deduped | direct_logit_only | +0.429 | 0.098 | +0.435 |
| pythia_70m_deduped | coactivation_only | +0.207 | 0.121 | +0.224 |
| pythia_70m_deduped | full_no_magnitude | +0.489 | 0.052 | +0.507 |
| pythia_70m_deduped | full_all | +0.483 | 0.035 | +0.508 |
