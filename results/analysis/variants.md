# Robustness variants

## Label reliability (split-half over contexts)

Same 300 features, labels measured once on the even-indexed contexts (A) and once on the odd-indexed contexts (B). Spearman between A and B is the test-retest reliability; the attenuation-corrected crowding correlation is rho / sqrt(reliability).

| Setting | Label | reliability rho(A, B) | crowding rho (A) | crowding rho (B) | corrected, raw count |
|---|---|---|---|---|---|
| GPT-2-small | collateral_raw | +0.960 | +0.489 | +0.496 | +0.492 |
| GPT-2-small | collateral_ctilde | +0.938 | +0.353 | +0.357 |  |
| GPT-2-small | effect_l2 | +0.895 |  | |  |
| GPT-2-small | stab_signed | +0.902 |  | |  |
| GPT-2-small | stab_abs | +0.903 |  | |  |
| GPT-2-small | kl_mean | +0.617 |  | |  |
| Pythia-70M-deduped | collateral_raw | +0.906 | -0.022 | -0.028 | -0.024 |
| Pythia-70M-deduped | collateral_ctilde | +0.759 | -0.016 | -0.025 |  |
| Pythia-70M-deduped | effect_l2 | +0.946 |  | |  |
| Pythia-70M-deduped | stab_signed | +0.937 |  | |  |
| Pythia-70M-deduped | stab_abs | +0.937 |  | |  |
| Pythia-70M-deduped | kl_mean | +0.781 |  | |  |
| Gemma-2-2B | collateral_raw | +0.925 | +0.526 | +0.542 | +0.533 |
| Gemma-2-2B | collateral_ctilde | +0.788 | +0.191 | +0.100 |  |
| Gemma-2-2B | effect_l2 | +0.800 |  | |  |
| Gemma-2-2B | stab_signed | +0.664 |  | |  |
| Gemma-2-2B | stab_abs | +0.602 |  | |  |
| Gemma-2-2B | kl_mean | +0.711 |  | |  |
| Llama-3.1-8B | collateral_raw | +0.946 | +0.240 | +0.213 | +0.228 |
| Llama-3.1-8B | collateral_ctilde | +0.897 | +0.222 | +0.163 |  |
| Llama-3.1-8B | effect_l2 | +0.854 |  | |  |
| Llama-3.1-8B | stab_signed | +0.694 |  | |  |
| Llama-3.1-8B | stab_abs | +0.520 |  | |  |
| Llama-3.1-8B | kl_mean | +0.375 |  | |  |

## Steering coefficient

Crowding vs collateral under different additive coefficients. alpha = 1.0 is the paper's value; q95 adds alpha times the feature's 95th-percentile natural activation, so the perturbation is scaled to each feature.

| Setting | Variant | raw count: rho | primary partial | C-tilde: rho | primary partial | median intervention value |
|---|---|---|---|---|---|---|
| GPT-2-small | alpha 1.0 (reported) | +0.482 | +0.475 | +0.344 | +0.425 | 1 |
| GPT-2-small | alpha0.5 | +0.487 | +0.493 | +0.471 | +0.493 | 0.5 |
| GPT-2-small | alpha2 | +0.460 | +0.457 | +0.145 | +0.364 | 2 |
| GPT-2-small | alpha4 | +0.347 | +0.382 | -0.015 | +0.323 | 4 |
| GPT-2-small | q95 | +0.352 | +0.264 | -0.234 | +0.266 | 4.16 |
| Pythia-70M-deduped | alpha 1.0 (reported) | -0.022 | -0.004 | +0.031 | +0.001 | 1 |
| Pythia-70M-deduped | alpha0.5 | +0.102 | +0.136 | +0.146 | +0.119 | 0.5 |
| Pythia-70M-deduped | alpha2 | -0.046 | -0.048 | +0.056 | -0.008 | 2 |
| Pythia-70M-deduped | alpha4 | -0.036 | -0.076 | +0.048 | -0.055 | 4 |
| Pythia-70M-deduped | q95 | +0.214 | +0.037 | +0.198 | +0.110 | 0.792 |
| Gemma-2-2B | alpha 1.0 (reported) | +0.512 | +0.242 | +0.069 | +0.252 | 1 |
| Llama-3.1-8B | alpha 1.0 (reported) | +0.222 | +0.152 | +0.201 | +0.166 | 1 |

## Random-direction control

300 random directions with norms drawn from the decoder norms, steered on 48 random contexts each. Crowding is the same top-20 mean absolute cosine to the dictionary. Frequency and activation controls do not exist for random directions, so the control is effect magnitude only; the SAE-feature row uses the same control for comparison.

| Setting | Steered vectors | raw count: rho [95% CI] | partial given E_f | C-tilde: rho [95% CI] | partial given E_f | median collateral |
|---|---|---|---|---|---|---|
| GPT-2-small | SAE features | +0.482 [+0.379, +0.573] | +0.444 | +0.344 [+0.235, +0.446] | +0.398 | 11.5 |
| GPT-2-small | random directions | +0.021 [-0.096, +0.135] | +0.020 | +0.016 [-0.099, +0.135] | +0.020 | 9.3 |
| Pythia-70M-deduped | SAE features | -0.022 [-0.132, +0.093] | +0.008 | +0.031 [-0.083, +0.145] | +0.016 | 22.1 |
| Pythia-70M-deduped | random directions | -0.014 [-0.126, +0.099] | -0.017 | -0.024 [-0.144, +0.090] | -0.021 | 22.8 |
| Gemma-2-2B | SAE features | +0.512 [+0.413, +0.601] | +0.267 | +0.069 [-0.051, +0.193] | +0.266 | 4.3 |
| Gemma-2-2B | random directions | -0.120 [-0.231, -0.010] | -0.130 | -0.118 [-0.226, -0.002] | -0.082 | 3.2 |
| Llama-3.1-8B | SAE features | +0.222 [+0.107, +0.332] | +0.161 | +0.201 [+0.090, +0.305] | +0.171 | 2.7 |
| Llama-3.1-8B | random directions | +0.020 [-0.095, +0.132] | +0.015 | +0.010 [-0.107, +0.126] | +0.004 | 2.0 |

## Dense-panel exclusion and residual-space collateral

Collateral recomputed without downstream panel features that fire in more than 10% of contexts, and the downstream residual-stream change itself (its norm, and the share of it the downstream SAE reconstructs).

| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) | crowding vs residual change norm | mean SAE-explained share of the change |
|---|---|---|---|---|---|---|
| GPT-2-small | 2042 of 2048 | +0.482 | +0.486 | +0.477 | +0.328 | 0.845 |
| Pythia-70M-deduped | 1969 of 2048 | -0.022 | -0.093 | -0.093 | +0.144 | 0.778 |
| Gemma-2-2B | 2011 of 2048 | +0.512 | +0.515 | +0.196 | +0.626 | 2.289 |
| Llama-3.1-8B | 1005 of 1024 | +0.222 | +0.242 | +0.144 | +0.172 | 1.186 |
