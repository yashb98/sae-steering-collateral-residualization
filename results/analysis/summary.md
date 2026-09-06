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
| gpt2_small | none | +0.482 | [+0.381, +0.574] | 7.6e-19 | - |
| gpt2_small | primary | +0.475 | [+0.373, +0.568] | 3.7e-18 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.517 | [+0.418, +0.605] | 8.2e-22 | frequency,act_mag |
| pythia_70m_deduped | none | -0.022 | [-0.136, +0.095] | 7.0e-01 | - |
| pythia_70m_deduped | primary | -0.004 | [-0.124, +0.114] | 9.4e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | -0.026 | [-0.148, +0.097] | 6.6e-01 | frequency,act_mag |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p |
|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.454 | [-0.552, -0.342] | 1.6e-16 |
| gemma_2_2b | coact_entropy | -0.299 | [-0.389, -0.196] | 1.6e-07 |
| gemma_2_2b | crowding | +0.242 | [+0.123, +0.357] | 2.5e-05 |
| gpt2_small | crowding | +0.475 | [+0.373, +0.568] | 3.7e-18 |
| gpt2_small | enc_dec_cos | -0.401 | [-0.499, -0.295] | 6.5e-13 |
| gpt2_small | crowd_max | +0.333 | [+0.218, +0.444] | 4.1e-09 |
| pythia_70m_deduped | logit_l2 | +0.343 | [+0.227, +0.450] | 1.3e-09 |
| pythia_70m_deduped | act_mean_firing | -0.207 | [-0.314, -0.084] | 3.3e-04 |
| pythia_70m_deduped | coact_entropy | -0.202 | [-0.297, -0.082] | 4.8e-04 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | -0.144 | 0.268 | -0.186 |
| gemma_2_2b | actmag_only | -0.106 | 0.292 | -0.179 |
| gemma_2_2b | geometry_only | +0.468 | 0.156 | +0.483 |
| gemma_2_2b | direct_logit_only | +0.137 | 0.149 | +0.106 |
| gemma_2_2b | coactivation_only | +0.168 | 0.090 | +0.156 |
| gemma_2_2b | full_no_magnitude | +0.588 | 0.110 | +0.598 |
| gemma_2_2b | full_all | +0.590 | 0.106 | +0.600 |
| gpt2_small | frequency_only | -0.015 | 0.084 | -0.139 |
| gpt2_small | actmag_only | -0.035 | 0.058 | -0.131 |
| gpt2_small | geometry_only | +0.458 | 0.064 | +0.445 |
| gpt2_small | direct_logit_only | +0.208 | 0.102 | +0.192 |
| gpt2_small | coactivation_only | +0.182 | 0.106 | +0.166 |
| gpt2_small | full_no_magnitude | +0.502 | 0.078 | +0.493 |
| gpt2_small | full_all | +0.493 | 0.091 | +0.500 |
| pythia_70m_deduped | frequency_only | -0.018 | 0.140 | -0.185 |
| pythia_70m_deduped | actmag_only | -0.113 | 0.099 | -0.204 |
| pythia_70m_deduped | geometry_only | +0.011 | 0.081 | -0.020 |
| pythia_70m_deduped | direct_logit_only | +0.493 | 0.083 | +0.495 |
| pythia_70m_deduped | coactivation_only | +0.184 | 0.095 | +0.180 |
| pythia_70m_deduped | full_no_magnitude | +0.517 | 0.052 | +0.537 |
| pythia_70m_deduped | full_all | +0.522 | 0.053 | +0.536 |

## Target: collateral_ctilde

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gemma_2_2b | none | +0.069 | [-0.052, +0.189] | 2.3e-01 | - |
| gemma_2_2b | primary | +0.252 | [+0.133, +0.368] | 1.1e-05 | effect_l2,act_mag,frequency |
| gemma_2_2b | robust | +0.052 | [-0.071, +0.177] | 3.7e-01 | frequency,act_mag |
| gpt2_small | none | +0.344 | [+0.234, +0.447] | 9.2e-10 | - |
| gpt2_small | primary | +0.425 | [+0.316, +0.524] | 1.9e-14 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.355 | [+0.245, +0.457] | 2.9e-10 | frequency,act_mag |
| pythia_70m_deduped | none | +0.031 | [-0.083, +0.142] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.001 | [-0.119, +0.122] | 9.8e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | +0.011 | [-0.104, +0.129] | 8.5e-01 | frequency,act_mag |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p |
|---|---|---|---|---|
| gemma_2_2b | enc_dec_cos | -0.427 | [-0.520, -0.320] | 1.5e-14 |
| gemma_2_2b | coact_entropy | -0.311 | [-0.401, -0.204] | 4.4e-08 |
| gemma_2_2b | crowding | +0.252 | [+0.133, +0.368] | 1.1e-05 |
| gpt2_small | crowding | +0.425 | [+0.316, +0.524] | 1.9e-14 |
| gpt2_small | enc_dec_cos | -0.399 | [-0.498, -0.290] | 9.1e-13 |
| gpt2_small | coact_entropy | -0.345 | [-0.439, -0.232] | 9.9e-10 |
| pythia_70m_deduped | logit_l2 | +0.311 | [+0.195, +0.424] | 4.2e-08 |
| pythia_70m_deduped | coact_count | +0.198 | [+0.087, +0.302] | 6.1e-04 |
| pythia_70m_deduped | act_mean_firing | -0.171 | [-0.273, -0.058] | 3.2e-03 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
| gemma_2_2b | frequency_only | -0.127 | 0.245 | -0.137 |
| gemma_2_2b | actmag_only | -0.095 | 0.266 | -0.136 |
| gemma_2_2b | geometry_only | +0.429 | 0.165 | +0.430 |
| gemma_2_2b | direct_logit_only | +0.104 | 0.140 | +0.065 |
| gemma_2_2b | coactivation_only | +0.144 | 0.141 | +0.123 |
| gemma_2_2b | full_no_magnitude | +0.578 | 0.105 | +0.583 |
| gemma_2_2b | full_all | +0.582 | 0.110 | +0.590 |
| gpt2_small | frequency_only | -0.027 | 0.077 | -0.132 |
| gpt2_small | actmag_only | -0.029 | 0.046 | -0.129 |
| gpt2_small | geometry_only | +0.447 | 0.075 | +0.437 |
| gpt2_small | direct_logit_only | +0.216 | 0.100 | +0.206 |
| gpt2_small | coactivation_only | +0.176 | 0.114 | +0.159 |
| gpt2_small | full_no_magnitude | +0.498 | 0.089 | +0.493 |
| gpt2_small | full_all | +0.489 | 0.095 | +0.500 |
| pythia_70m_deduped | frequency_only | -0.118 | 0.090 | -0.187 |
| pythia_70m_deduped | actmag_only | -0.116 | 0.112 | -0.182 |
| pythia_70m_deduped | geometry_only | +0.044 | 0.092 | +0.005 |
| pythia_70m_deduped | direct_logit_only | +0.426 | 0.094 | +0.426 |
| pythia_70m_deduped | coactivation_only | +0.162 | 0.088 | +0.157 |
| pythia_70m_deduped | full_no_magnitude | +0.477 | 0.046 | +0.501 |
| pythia_70m_deduped | full_all | +0.482 | 0.052 | +0.500 |
