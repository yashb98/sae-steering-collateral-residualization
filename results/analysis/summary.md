# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.512 | [+0.411, +0.605] | 1.8e-21 | - |
| gemma_2_2b | primary | +0.242 | [+0.123, +0.362] | 2.5e-05 | effect_l2,act_mag,frequency |
| gemma_2_2b | robust | +0.468 | [+0.366, +0.566] | 1.2e-17 | frequency,act_mag |
| gemma_2_2b | effect_only | +0.267 | [+0.150, +0.382] | 2.7e-06 | effect_l2 |
| gpt2_small | none | +0.482 | [+0.381, +0.571] | 7.6e-19 | - |
| gpt2_small | primary | +0.475 | [+0.373, +0.567] | 3.7e-18 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.517 | [+0.418, +0.604] | 8.2e-22 | frequency,act_mag |
| gpt2_small | effect_only | +0.444 | [+0.340, +0.540] | 7.5e-16 | effect_l2 |
| llama_3_1_8b | none | +0.222 | [+0.109, +0.332] | 1.1e-04 | - |
| llama_3_1_8b | primary | +0.152 | [+0.039, +0.266] | 8.7e-03 | effect_l2,act_mag,frequency |
| llama_3_1_8b | robust | +0.182 | [+0.067, +0.298] | 1.6e-03 | frequency,act_mag |
| llama_3_1_8b | effect_only | +0.161 | [+0.051, +0.269] | 5.2e-03 | effect_l2 |
| pythia_70m_deduped | none | -0.022 | [-0.137, +0.092] | 7.0e-01 | - |
| pythia_70m_deduped | primary | -0.004 | [-0.124, +0.114] | 9.4e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | -0.026 | [-0.148, +0.098] | 6.6e-01 | frequency,act_mag |
| pythia_70m_deduped | effect_only | +0.008 | [-0.108, +0.119] | 9.0e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.454 | [-0.550, -0.344] | 1.6e-16 | 2.8e-15 |
| gemma_2_2b | coact_entropy | -0.299 | [-0.388, -0.194] | 1.6e-07 | 2.5e-06 |
| gemma_2_2b | crowding | +0.242 | [+0.123, +0.362] | 2.5e-05 | 3.7e-04 |
| gpt2_small | crowding | +0.475 | [+0.373, +0.567] | 3.7e-18 | 6.3e-17 |
| gpt2_small | enc_dec_cos | -0.401 | [-0.498, -0.294] | 6.5e-13 | 1.0e-11 |
| gpt2_small | crowd_max | +0.333 | [+0.215, +0.440] | 4.1e-09 | 6.2e-08 |
| llama_3_1_8b | logit_top10_mass | -0.293 | [-0.392, -0.191] | 2.6e-07 | 4.4e-06 |
| llama_3_1_8b | logit_entropy | +0.243 | [+0.126, +0.352] | 2.3e-05 | 3.7e-04 |
| llama_3_1_8b | act_entropy | +0.159 | [+0.051, +0.255] | 5.9e-03 | 8.8e-02 |
| pythia_70m_deduped | logit_l2 | +0.343 | [+0.230, +0.448] | 1.3e-09 | 2.3e-08 |
| pythia_70m_deduped | act_mean_firing | -0.207 | [-0.313, -0.088] | 3.3e-04 | 5.3e-03 |
| pythia_70m_deduped | coact_entropy | -0.202 | [-0.301, -0.086] | 4.8e-04 | 7.1e-03 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +nan | nan | +nan |
| gemma_2_2b | geometry_only | +0.469 | 0.173 | +0.498 |
| gemma_2_2b | direct_logit_only | +0.152 | 0.128 | +0.147 |
| gemma_2_2b | coactivation_only | +0.204 | 0.091 | +0.194 |
| gemma_2_2b | full_no_magnitude | +0.594 | 0.119 | +0.611 |
| gemma_2_2b | full_all | +0.591 | 0.113 | +0.609 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +nan | nan | +nan |
| gpt2_small | geometry_only | +0.456 | 0.073 | +0.457 |
| gpt2_small | direct_logit_only | +0.201 | 0.110 | +0.207 |
| gpt2_small | coactivation_only | +0.198 | 0.105 | +0.192 |
| gpt2_small | full_no_magnitude | +0.512 | 0.083 | +0.511 |
| gpt2_small | full_all | +0.499 | 0.093 | +0.512 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | +nan | nan | +nan |
| llama_3_1_8b | geometry_only | +0.142 | 0.056 | +0.162 |
| llama_3_1_8b | direct_logit_only | +0.319 | 0.120 | +0.306 |
| llama_3_1_8b | coactivation_only | +0.131 | 0.127 | +0.111 |
| llama_3_1_8b | full_no_magnitude | +0.289 | 0.098 | +0.294 |
| llama_3_1_8b | full_all | +0.294 | 0.059 | +0.311 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | +nan | nan | +nan |
| pythia_70m_deduped | geometry_only | +0.004 | 0.098 | +0.018 |
| pythia_70m_deduped | direct_logit_only | +0.500 | 0.068 | +0.502 |
| pythia_70m_deduped | coactivation_only | +0.202 | 0.122 | +0.224 |
| pythia_70m_deduped | full_no_magnitude | +0.521 | 0.053 | +0.543 |
| pythia_70m_deduped | full_all | +0.528 | 0.059 | +0.540 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.054, +0.192] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.252 | [+0.130, +0.364] | 1.1e-05 | effect_l2,act_mag,frequency |
| gemma_2_2b | robust | +0.052 | [-0.073, +0.177] | 3.7e-01 | frequency,act_mag |
| gemma_2_2b | effect_only | +0.266 | [+0.147, +0.378] | 3.1e-06 | effect_l2 |
| gpt2_small | none | +0.344 | [+0.233, +0.445] | 9.2e-10 | - |
| gpt2_small | primary | +0.425 | [+0.317, +0.524] | 1.9e-14 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.355 | [+0.248, +0.454] | 2.9e-10 | frequency,act_mag |
| gpt2_small | effect_only | +0.398 | [+0.289, +0.499] | 8.1e-13 | effect_l2 |
| llama_3_1_8b | none | +0.201 | [+0.092, +0.307] | 4.5e-04 | - |
| llama_3_1_8b | primary | +0.166 | [+0.052, +0.275] | 4.2e-03 | effect_l2,act_mag,frequency |
| llama_3_1_8b | robust | +0.183 | [+0.070, +0.293] | 1.5e-03 | frequency,act_mag |
| llama_3_1_8b | effect_only | +0.171 | [+0.061, +0.278] | 3.0e-03 | effect_l2 |
| pythia_70m_deduped | none | +0.031 | [-0.084, +0.143] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.001 | [-0.121, +0.124] | 9.8e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | +0.011 | [-0.108, +0.127] | 8.5e-01 | frequency,act_mag |
| pythia_70m_deduped | effect_only | +0.016 | [-0.097, +0.130] | 7.8e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.427 | [-0.519, -0.321] | 1.5e-14 | 2.5e-13 |
| gemma_2_2b | coact_entropy | -0.311 | [-0.403, -0.203] | 4.4e-08 | 7.1e-07 |
| gemma_2_2b | crowding | +0.252 | [+0.130, +0.364] | 1.1e-05 | 1.6e-04 |
| gpt2_small | crowding | +0.425 | [+0.317, +0.524] | 1.9e-14 | 3.3e-13 |
| gpt2_small | enc_dec_cos | -0.399 | [-0.499, -0.290] | 9.1e-13 | 1.5e-11 |
| gpt2_small | coact_entropy | -0.345 | [-0.443, -0.234] | 9.9e-10 | 1.5e-08 |
| llama_3_1_8b | logit_top10_mass | -0.298 | [-0.394, -0.193] | 1.7e-07 | 2.9e-06 |
| llama_3_1_8b | logit_entropy | +0.231 | [+0.115, +0.339] | 6.1e-05 | 9.7e-04 |
| llama_3_1_8b | crowding | +0.166 | [+0.052, +0.275] | 4.2e-03 | 6.3e-02 |
| pythia_70m_deduped | logit_l2 | +0.311 | [+0.195, +0.423] | 4.2e-08 | 7.2e-07 |
| pythia_70m_deduped | coact_count | +0.198 | [+0.085, +0.304] | 6.1e-04 | 9.8e-03 |
| pythia_70m_deduped | act_mean_firing | -0.171 | [-0.275, -0.059] | 3.2e-03 | 4.8e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +nan | nan | +nan |
| gemma_2_2b | geometry_only | +0.439 | 0.165 | +0.444 |
| gemma_2_2b | direct_logit_only | +0.109 | 0.136 | +0.083 |
| gemma_2_2b | coactivation_only | +0.164 | 0.147 | +0.148 |
| gemma_2_2b | full_no_magnitude | +0.593 | 0.103 | +0.598 |
| gemma_2_2b | full_all | +0.595 | 0.109 | +0.602 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +nan | nan | +nan |
| gpt2_small | geometry_only | +0.451 | 0.076 | +0.451 |
| gpt2_small | direct_logit_only | +0.213 | 0.103 | +0.223 |
| gpt2_small | coactivation_only | +0.191 | 0.113 | +0.185 |
| gpt2_small | full_no_magnitude | +0.514 | 0.091 | +0.512 |
| gpt2_small | full_all | +0.507 | 0.096 | +0.520 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | +nan | nan | +nan |
| llama_3_1_8b | geometry_only | +0.187 | 0.073 | +0.210 |
| llama_3_1_8b | direct_logit_only | +0.318 | 0.126 | +0.318 |
| llama_3_1_8b | coactivation_only | +0.177 | 0.121 | +0.155 |
| llama_3_1_8b | full_no_magnitude | +0.325 | 0.082 | +0.334 |
| llama_3_1_8b | full_all | +0.313 | 0.055 | +0.321 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | +nan | nan | +nan |
| pythia_70m_deduped | geometry_only | +0.047 | 0.108 | +0.037 |
| pythia_70m_deduped | direct_logit_only | +0.429 | 0.090 | +0.435 |
| pythia_70m_deduped | coactivation_only | +0.184 | 0.127 | +0.204 |
| pythia_70m_deduped | full_no_magnitude | +0.482 | 0.046 | +0.507 |
| pythia_70m_deduped | full_all | +0.487 | 0.048 | +0.508 |
