# Collateral-side residualization: summary

n_boot = 10000, bootstrap over features with replacement, 95% percentile CIs.
Controls: primary = effect_l2 + intervention_value (dropped if constant) + act_mag (natural activation, assumption) + frequency; robust = frequency + act_mag; none = raw Spearman.

## Target: collateral_raw

### Decoder crowding

| setting | control | rho | 95% CI | p | controls |
|---|---|---|---|---|---|
| gpt2_small | none | +0.482 | [+0.381, +0.569] | 7.6e-19 | - |
| gpt2_small | primary | +0.475 | [+0.372, +0.567] | 3.7e-18 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.517 | [+0.418, +0.604] | 8.2e-22 | frequency,act_mag |
| pythia_70m_deduped | none | -0.022 | [-0.137, +0.092] | 7.0e-01 | - |
| pythia_70m_deduped | primary | -0.004 | [-0.126, +0.116] | 9.4e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | -0.026 | [-0.144, +0.097] | 6.6e-01 | frequency,act_mag |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p |
|---|---|---|---|---|
| gpt2_small | crowding | +0.475 | [+0.372, +0.567] | 3.7e-18 |
| gpt2_small | enc_dec_cos | -0.401 | [-0.498, -0.293] | 6.5e-13 |
| gpt2_small | crowd_max | +0.333 | [+0.216, +0.442] | 4.1e-09 |
| pythia_70m_deduped | logit_l2 | +0.343 | [+0.230, +0.448] | 1.3e-09 |
| pythia_70m_deduped | act_mean_firing | -0.207 | [-0.314, -0.087] | 3.3e-04 |
| pythia_70m_deduped | coact_entropy | -0.202 | [-0.303, -0.083] | 4.8e-04 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
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
| gpt2_small | none | +0.344 | [+0.232, +0.446] | 9.2e-10 | - |
| gpt2_small | primary | +0.425 | [+0.319, +0.527] | 1.9e-14 | effect_l2,act_mag,frequency |
| gpt2_small | robust | +0.355 | [+0.246, +0.457] | 2.9e-10 | frequency,act_mag |
| pythia_70m_deduped | none | +0.031 | [-0.082, +0.144] | 6.0e-01 | - |
| pythia_70m_deduped | primary | +0.001 | [-0.118, +0.122] | 9.8e-01 | effect_l2,act_mag,frequency |
| pythia_70m_deduped | robust | +0.011 | [-0.107, +0.128] | 8.5e-01 | frequency,act_mag |

### Strongest predictor per setting under the primary control (by |rho|)

| setting | predictor | partial rho | 95% CI | p |
|---|---|---|---|---|
| gpt2_small | crowding | +0.425 | [+0.319, +0.527] | 1.9e-14 |
| gpt2_small | enc_dec_cos | -0.399 | [-0.496, -0.292] | 9.1e-13 |
| gpt2_small | coact_entropy | -0.345 | [-0.442, -0.235] | 9.9e-10 |
| pythia_70m_deduped | logit_l2 | +0.311 | [+0.194, +0.420] | 4.2e-08 |
| pythia_70m_deduped | coact_count | +0.198 | [+0.088, +0.305] | 6.1e-04 |
| pythia_70m_deduped | act_mean_firing | -0.171 | [-0.272, -0.059] | 3.2e-03 |

### Table B3 analog: CV ridge Spearman on the residualized target (primary control)

| setting | predictor set | CV Spearman (mean over folds) | sd | pooled OOF |
|---|---|---|---|---|
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
