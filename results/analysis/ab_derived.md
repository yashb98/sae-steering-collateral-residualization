# Derived A/B analyses (2026-09-13, branch ab-battery-2026-09-13)

Inputs: canonical per-feature CSVs, alpha-variant CSVs, variants.json,
partial_correlations.csv. No new GPU measurements.

## gpt2_small

Measurement ceiling (ctxA/ctxB split-half, Spearman-Brown) and attenuation-corrected crowding partial rho:

| target | split-half r | SB reliability R | partial rho | corrected |
|---|---|---|---|---|
| collateral_raw | +0.960 | 0.979 | +0.475 | +0.480 |
| collateral_ctilde | +0.938 | 0.968 | +0.425 | +0.432 |

Top-decile precision (fraction of predictor-top-decile features in collateral top decile; baseline 0.10):

| target | predictor | precision | n flagged |
|---|---|---|---|
| collateral_raw | crowding | 0.400 | 30 |
| collateral_raw | crowd_max | 0.267 | 30 |
| collateral_raw | logit_l2 | 0.067 | 30 |
| collateral_raw | enc_dec_cos | 0.067 | 30 |
| collateral_ctilde | crowding | 0.267 | 30 |
| collateral_ctilde | crowd_max | 0.233 | 30 |
| collateral_ctilde | logit_l2 | 0.067 | 30 |
| collateral_ctilde | enc_dec_cos | 0.033 | 30 |

Max-vs-mean crowding aggregation (primary control):

| target | crowding (mean top-20) | crowd_max (max) |
|---|---|---|
| collateral_raw | +0.475 [+0.373, +0.567] | +0.333 [+0.215, +0.440] |
| collateral_ctilde | +0.425 [+0.317, +0.524] | +0.300 [+0.177, +0.417] |

Orthogonal-selection demo (median raw collateral within effect_l2 terciles):

| effect tercile | low crowding | high crowding |
|---|---|---|
| low | 9.09 | 11.94 |
| mid | 10.77 | 12.98 |
| high | 11.16 | 15.31 |

Dose-response over alphas [0.5, 1.0, 2.0, 4.0] (n=300 common features):

| target | median collateral per alpha | per-feature rho(collateral, alpha) median [IQR] |
|---|---|---|
| collateral_raw | ['3.06', '11.49', '19.66', '24.55'] | +1.000 [+1.000, +1.000] |
| collateral_ctilde | ['1.54', '2.89', '2.50', '1.55'] | -0.400 [-0.400, +0.200] |

## pythia_70m_deduped

Measurement ceiling (ctxA/ctxB split-half, Spearman-Brown) and attenuation-corrected crowding partial rho:

| target | split-half r | SB reliability R | partial rho | corrected |
|---|---|---|---|---|
| collateral_raw | +0.906 | 0.951 | -0.004 | -0.005 |
| collateral_ctilde | +0.759 | 0.863 | +0.001 | +0.002 |

Top-decile precision (fraction of predictor-top-decile features in collateral top decile; baseline 0.10):

| target | predictor | precision | n flagged |
|---|---|---|---|
| collateral_raw | crowding | 0.133 | 30 |
| collateral_raw | crowd_max | 0.100 | 30 |
| collateral_raw | logit_l2 | 0.167 | 30 |
| collateral_raw | enc_dec_cos | 0.067 | 30 |
| collateral_ctilde | crowding | 0.100 | 30 |
| collateral_ctilde | crowd_max | 0.067 | 30 |
| collateral_ctilde | logit_l2 | 0.167 | 30 |
| collateral_ctilde | enc_dec_cos | 0.200 | 30 |

Max-vs-mean crowding aggregation (primary control):

| target | crowding (mean top-20) | crowd_max (max) |
|---|---|---|
| collateral_raw | -0.004 [-0.124, +0.114] | +0.015 [-0.106, +0.132] |
| collateral_ctilde | +0.001 [-0.121, +0.124] | +0.014 [-0.107, +0.134] |

Orthogonal-selection demo (median raw collateral within effect_l2 terciles):

| effect tercile | low crowding | high crowding |
|---|---|---|
| low | 18.93 | 18.41 |
| mid | 22.74 | 22.94 |
| high | 23.79 | 24.82 |

Dose-response over alphas [0.5, 1.0, 2.0, 4.0] (n=300 common features):

| target | median collateral per alpha | per-feature rho(collateral, alpha) median [IQR] |
|---|---|---|
| collateral_raw | ['6.52', '22.05', '42.25', '62.66'] | +1.000 [+1.000, +1.000] |
| collateral_ctilde | ['0.35', '0.61', '0.62', '0.48'] | +0.200 [+0.200, +0.400] |

## gemma_2_2b

Measurement ceiling (ctxA/ctxB split-half, Spearman-Brown) and attenuation-corrected crowding partial rho:

| target | split-half r | SB reliability R | partial rho | corrected |
|---|---|---|---|---|
| collateral_raw | +0.925 | 0.961 | +0.242 | +0.247 |
| collateral_ctilde | +0.788 | 0.882 | +0.252 | +0.269 |

Top-decile precision (fraction of predictor-top-decile features in collateral top decile; baseline 0.10):

| target | predictor | precision | n flagged |
|---|---|---|---|
| collateral_raw | crowding | 0.267 | 30 |
| collateral_raw | crowd_max | 0.100 | 30 |
| collateral_raw | logit_l2 | 0.200 | 30 |
| collateral_raw | enc_dec_cos | 0.000 | 30 |
| collateral_ctilde | crowding | 0.033 | 30 |
| collateral_ctilde | crowd_max | 0.167 | 30 |
| collateral_ctilde | logit_l2 | 0.033 | 30 |
| collateral_ctilde | enc_dec_cos | 0.100 | 30 |

Max-vs-mean crowding aggregation (primary control):

| target | crowding (mean top-20) | crowd_max (max) |
|---|---|---|
| collateral_raw | +0.242 [+0.123, +0.362] | +0.136 [+0.018, +0.256] |
| collateral_ctilde | +0.252 [+0.130, +0.364] | +0.158 [+0.039, +0.272] |

Orthogonal-selection demo (median raw collateral within effect_l2 terciles):

| effect tercile | low crowding | high crowding |
|---|---|---|
| low | 2.74 | 3.80 |
| mid | 3.93 | 5.21 |
| high | 5.07 | 5.71 |

Dose-response over alphas [0.5, 1.0, 2.0] (n=300 common features):

| target | median collateral per alpha | per-feature rho(collateral, alpha) median [IQR] |
|---|---|---|
| collateral_raw | ['0.81', '4.31', '14.97'] | +1.000 [+1.000, +1.000] |
| collateral_ctilde | ['0.28', '0.76', '1.36'] | +1.000 [+1.000, +1.000] |

## llama_3_1_8b

Measurement ceiling (ctxA/ctxB split-half, Spearman-Brown) and attenuation-corrected crowding partial rho:

| target | split-half r | SB reliability R | partial rho | corrected |
|---|---|---|---|---|
| collateral_raw | +0.946 | 0.972 | +0.152 | +0.154 |
| collateral_ctilde | +0.897 | 0.946 | +0.166 | +0.170 |

Top-decile precision (fraction of predictor-top-decile features in collateral top decile; baseline 0.10):

| target | predictor | precision | n flagged |
|---|---|---|---|
| collateral_raw | crowding | 0.300 | 30 |
| collateral_raw | crowd_max | 0.300 | 30 |
| collateral_raw | logit_l2 | 0.433 | 30 |
| collateral_raw | enc_dec_cos | 0.033 | 30 |
| collateral_ctilde | crowding | 0.233 | 30 |
| collateral_ctilde | crowd_max | 0.267 | 30 |
| collateral_ctilde | logit_l2 | 0.233 | 30 |
| collateral_ctilde | enc_dec_cos | 0.133 | 30 |

Max-vs-mean crowding aggregation (primary control):

| target | crowding (mean top-20) | crowd_max (max) |
|---|---|---|
| collateral_raw | +0.152 [+0.039, +0.266] | +0.086 [-0.049, +0.218] |
| collateral_ctilde | +0.166 [+0.052, +0.275] | +0.077 [-0.053, +0.206] |

Orthogonal-selection demo (median raw collateral within effect_l2 terciles):

| effect tercile | low crowding | high crowding |
|---|---|---|
| low | 2.08 | 2.31 |
| mid | 2.47 | 2.98 |
| high | 3.24 | 3.65 |

Dose-response over alphas [0.5, 1.0, 2.0] (n=300 common features):

| target | median collateral per alpha | per-feature rho(collateral, alpha) median [IQR] |
|---|---|---|
| collateral_raw | ['1.15', '2.68', '6.18'] | +1.000 [+1.000, +1.000] |
| collateral_ctilde | ['0.10', '0.18', '0.28'] | +1.000 [+1.000, +1.000] |
