# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.512 | [+0.411, +0.605] | 1.8e-21 | - |
| gemma_2_2b | primary | +0.261 | [+0.144, +0.377] | 5.1e-06 | effect_l2,act_max,frequency |
| gemma_2_2b | robust | +0.508 | [+0.409, +0.601] | 5.3e-21 | frequency,act_max |
| gemma_2_2b | effect_only | +0.267 | [+0.150, +0.382] | 2.7e-06 | effect_l2 |
| gpt2_small | none | +0.482 | [+0.381, +0.571] | 7.6e-19 | - |
| gpt2_small | primary | +0.482 | [+0.379, +0.572] | 1.1e-18 | effect_l2,act_max,frequency |
| gpt2_small | robust | +0.522 | [+0.425, +0.608] | 3.0e-22 | frequency,act_max |
| gpt2_small | effect_only | +0.444 | [+0.340, +0.540] | 7.5e-16 | effect_l2 |
| llama_3_1_8b | none | +0.222 | [+0.109, +0.332] | 1.1e-04 | - |
| llama_3_1_8b | primary | +0.160 | [+0.050, +0.270] | 5.8e-03 | effect_l2,act_max,frequency |
| llama_3_1_8b | robust | +0.196 | [+0.082, +0.309] | 6.5e-04 | frequency,act_max |
| llama_3_1_8b | effect_only | +0.161 | [+0.051, +0.269] | 5.2e-03 | effect_l2 |
| pythia_70m_deduped | none | -0.022 | [-0.137, +0.092] | 7.0e-01 | - |
| pythia_70m_deduped | primary | +0.006 | [-0.112, +0.125] | 9.2e-01 | effect_l2,act_max,frequency |
| pythia_70m_deduped | robust | -0.018 | [-0.139, +0.103] | 7.6e-01 | frequency,act_max |
| pythia_70m_deduped | effect_only | +0.008 | [-0.108, +0.119] | 9.0e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.476 | [-0.569, -0.368] | 3.6e-18 | 6.1e-17 |
| gemma_2_2b | coact_entropy | -0.313 | [-0.400, -0.209] | 3.6e-08 | 5.8e-07 |
| gemma_2_2b | crowding | +0.261 | [+0.144, +0.377] | 5.1e-06 | 7.6e-05 |
| gpt2_small | crowding | +0.482 | [+0.379, +0.572] | 1.1e-18 | 1.9e-17 |
| gpt2_small | enc_dec_cos | -0.438 | [-0.531, -0.335] | 2.2e-15 | 3.5e-14 |
| gpt2_small | crowd_max | +0.347 | [+0.233, +0.451] | 8.1e-10 | 1.2e-08 |
| llama_3_1_8b | logit_top10_mass | -0.292 | [-0.391, -0.189] | 2.9e-07 | 4.9e-06 |
| llama_3_1_8b | logit_entropy | +0.242 | [+0.124, +0.350] | 2.6e-05 | 4.1e-04 |
| llama_3_1_8b | crowding | +0.160 | [+0.050, +0.270] | 5.8e-03 | 8.7e-02 |
| pythia_70m_deduped | logit_l2 | +0.345 | [+0.232, +0.450] | 1.0e-09 | 1.8e-08 |
| pythia_70m_deduped | coact_entropy | -0.222 | [-0.319, -0.113] | 1.1e-04 | 1.8e-03 |
| pythia_70m_deduped | coact_count | +0.162 | [+0.051, +0.269] | 5.1e-03 | 7.6e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +0.230 | 0.096 | +0.147 |
| gemma_2_2b | geometry_only | +0.481 | 0.169 | +0.506 |
| gemma_2_2b | direct_logit_only | +0.162 | 0.131 | +0.144 |
| gemma_2_2b | coactivation_only | +0.167 | 0.067 | +0.167 |
| gemma_2_2b | full_no_magnitude | +0.576 | 0.122 | +0.595 |
| gemma_2_2b | full_all | +0.583 | 0.120 | +0.597 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.044 | 0.063 | +0.025 |
| gpt2_small | geometry_only | +0.460 | 0.072 | +0.460 |
| gpt2_small | direct_logit_only | +0.218 | 0.113 | +0.219 |
| gpt2_small | coactivation_only | +0.193 | 0.115 | +0.185 |
| gpt2_small | full_no_magnitude | +0.515 | 0.087 | +0.510 |
| gpt2_small | full_all | +0.513 | 0.089 | +0.519 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | -0.121 | 0.083 | -0.035 |
| llama_3_1_8b | geometry_only | +0.157 | 0.063 | +0.185 |
| llama_3_1_8b | direct_logit_only | +0.325 | 0.130 | +0.307 |
| llama_3_1_8b | coactivation_only | +0.139 | 0.142 | +0.128 |
| llama_3_1_8b | full_no_magnitude | +0.306 | 0.101 | +0.313 |
| llama_3_1_8b | full_all | +0.289 | 0.067 | +0.301 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | -0.028 | 0.167 | +0.028 |
| pythia_70m_deduped | geometry_only | +0.004 | 0.109 | +0.016 |
| pythia_70m_deduped | direct_logit_only | +0.497 | 0.069 | +0.499 |
| pythia_70m_deduped | coactivation_only | +0.197 | 0.135 | +0.216 |
| pythia_70m_deduped | full_no_magnitude | +0.520 | 0.054 | +0.541 |
| pythia_70m_deduped | full_all | +0.529 | 0.065 | +0.540 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.054, +0.192] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.264 | [+0.142, +0.376] | 4.0e-06 | effect_l2,act_max,frequency |
| gemma_2_2b | robust | +0.059 | [-0.065, +0.182] | 3.1e-01 | frequency,act_max |
| gemma_2_2b | effect_only | +0.266 | [+0.147, +0.378] | 3.1e-06 | effect_l2 |
| gpt2_small | none | +0.344 | [+0.233, +0.445] | 9.2e-10 | - |
| gpt2_small | primary | +0.431 | [+0.326, +0.529] | 6.9e-15 | effect_l2,act_max,frequency |
| gpt2_small | robust | +0.362 | [+0.255, +0.461] | 1.2e-10 | frequency,act_max |
| gpt2_small | effect_only | +0.398 | [+0.289, +0.499] | 8.1e-13 | effect_l2 |
| llama_3_1_8b | none | +0.201 | [+0.092, +0.307] | 4.5e-04 | - |
| llama_3_1_8b | primary | +0.170 | [+0.059, +0.276] | 3.3e-03 | effect_l2,act_max,frequency |
| llama_3_1_8b | robust | +0.191 | [+0.080, +0.298] | 9.5e-04 | frequency,act_max |
| llama_3_1_8b | effect_only | +0.171 | [+0.061, +0.278] | 3.0e-03 | effect_l2 |
| pythia_70m_deduped | none | +0.031 | [-0.084, +0.143] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.008 | [-0.113, +0.129] | 8.9e-01 | effect_l2,act_max,frequency |
| pythia_70m_deduped | robust | +0.018 | [-0.101, +0.133] | 7.6e-01 | frequency,act_max |
| pythia_70m_deduped | effect_only | +0.016 | [-0.097, +0.130] | 7.8e-01 | effect_l2 |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p | Holm p |
|---|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.442 | [-0.530, -0.340] | 1.2e-15 | 2.1e-14 |
| gemma_2_2b | coact_entropy | -0.320 | [-0.410, -0.213] | 1.7e-08 | 2.7e-07 |
| gemma_2_2b | crowding | +0.264 | [+0.142, +0.376] | 4.0e-06 | 6.1e-05 |
| gpt2_small | crowding | +0.431 | [+0.326, +0.529] | 6.9e-15 | 1.2e-13 |
| gpt2_small | enc_dec_cos | -0.430 | [-0.525, -0.322] | 8.6e-15 | 1.4e-13 |
| gpt2_small | coact_entropy | -0.358 | [-0.453, -0.248] | 2.1e-10 | 3.1e-09 |
| llama_3_1_8b | logit_top10_mass | -0.297 | [-0.394, -0.192] | 1.8e-07 | 3.1e-06 |
| llama_3_1_8b | logit_entropy | +0.230 | [+0.115, +0.338] | 6.5e-05 | 1.0e-03 |
| llama_3_1_8b | crowding | +0.170 | [+0.059, +0.276] | 3.3e-03 | 5.0e-02 |
| pythia_70m_deduped | logit_l2 | +0.313 | [+0.196, +0.423] | 3.8e-08 | 6.4e-07 |
| pythia_70m_deduped | coact_count | +0.179 | [+0.066, +0.285] | 2.0e-03 | 3.1e-02 |
| pythia_70m_deduped | coact_entropy | -0.172 | [-0.292, -0.041] | 2.9e-03 | 4.4e-02 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | +nan | nan | +nan |
| gemma_2_2b | actmag_only | +0.239 | 0.103 | +0.159 |
| gemma_2_2b | geometry_only | +0.446 | 0.160 | +0.450 |
| gemma_2_2b | direct_logit_only | +0.106 | 0.126 | +0.089 |
| gemma_2_2b | coactivation_only | +0.159 | 0.117 | +0.138 |
| gemma_2_2b | full_no_magnitude | +0.585 | 0.097 | +0.589 |
| gemma_2_2b | full_all | +0.595 | 0.101 | +0.596 |
| gpt2_small | frequency_only | +nan | nan | +nan |
| gpt2_small | actmag_only | +0.039 | 0.048 | +0.022 |
| gpt2_small | geometry_only | +0.453 | 0.079 | +0.453 |
| gpt2_small | direct_logit_only | +0.228 | 0.108 | +0.236 |
| gpt2_small | coactivation_only | +0.179 | 0.125 | +0.176 |
| gpt2_small | full_no_magnitude | +0.512 | 0.097 | +0.514 |
| gpt2_small | full_all | +0.515 | 0.098 | +0.526 |
| llama_3_1_8b | frequency_only | +nan | nan | +nan |
| llama_3_1_8b | actmag_only | +0.046 | 0.102 | +0.128 |
| llama_3_1_8b | geometry_only | +0.194 | 0.065 | +0.214 |
| llama_3_1_8b | direct_logit_only | +0.296 | 0.114 | +0.305 |
| llama_3_1_8b | coactivation_only | +0.159 | 0.103 | +0.149 |
| llama_3_1_8b | full_no_magnitude | +0.337 | 0.083 | +0.343 |
| llama_3_1_8b | full_all | +0.334 | 0.081 | +0.350 |
| pythia_70m_deduped | frequency_only | +nan | nan | +nan |
| pythia_70m_deduped | actmag_only | -0.119 | 0.113 | -0.081 |
| pythia_70m_deduped | geometry_only | +0.047 | 0.106 | +0.041 |
| pythia_70m_deduped | direct_logit_only | +0.426 | 0.092 | +0.433 |
| pythia_70m_deduped | coactivation_only | +0.192 | 0.131 | +0.203 |
| pythia_70m_deduped | full_no_magnitude | +0.474 | 0.047 | +0.502 |
| pythia_70m_deduped | full_all | +0.480 | 0.046 | +0.504 |
