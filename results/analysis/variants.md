# Robustness variants

## Context-pool reproducibility

The same 300 features and clean-activation predictors are retained. Labels use 48 mixed contexts selected from the 1,024 even-indexed contexts (A) or the 1,024 odd-indexed contexts (B) of the same ordered corpus pool. The halves are disjoint, but are not independent corpus draws; neighboring texts can be related. These are split-pool Spearman correlations, conditional on one dictionary and feature sample. No predictor ceiling or attenuation-corrected correlation is claimed. Correlation is undefined when one draw gives a constant label. Percentile intervals resample matched feature rows and are undefined if a draw is degenerate; JSON records the valid-draw count.

| Setting | Label | rho(A, B) [95% CI] | crowding rho (A) | crowding rho (B) |
|---|---|---|---|---|
| GPT-2-small | collateral_raw | +0.960 [+0.944, +0.969] | +0.489 | +0.496 |
| GPT-2-small | collateral_ctilde | +0.938 [+0.919, +0.952] | +0.353 | +0.357 |
| GPT-2-small | effect_l2 | +0.895 [+0.863, +0.920] |  |  |
| GPT-2-small | stab_signed | +0.902 [+0.868, +0.927] |  |  |
| GPT-2-small | stab_abs | +0.903 [+0.870, +0.927] |  |  |
| GPT-2-small | kl_mean | +0.617 [+0.526, +0.697] |  |  |
| GPT-2-small | stab_anti_frac | +0.093 [-0.060, +0.264] |  |  |
| GPT-2-small | effect_pc1_ratio | +0.241 [+0.129, +0.347] |  |  |
| Pythia-70M-deduped | collateral_raw | +0.906 [+0.877, +0.928] | -0.022 | -0.028 |
| Pythia-70M-deduped | collateral_ctilde | +0.759 [+0.696, +0.811] | -0.016 | -0.025 |
| Pythia-70M-deduped | effect_l2 | +0.946 [+0.920, +0.962] |  |  |
| Pythia-70M-deduped | stab_signed | +0.937 [+0.913, +0.954] |  |  |
| Pythia-70M-deduped | stab_abs | +0.937 [+0.914, +0.954] |  |  |
| Pythia-70M-deduped | kl_mean | +0.781 [+0.722, +0.831] |  |  |
| Pythia-70M-deduped | stab_anti_frac | undefined (constant label) |  |  |
| Pythia-70M-deduped | effect_pc1_ratio | +0.440 [+0.341, +0.532] |  |  |
| Gemma-2-2B | collateral_raw | +0.925 [+0.898, +0.943] | +0.526 | +0.542 |
| Gemma-2-2B | collateral_ctilde | +0.788 [+0.733, +0.833] | +0.191 | +0.100 |
| Gemma-2-2B | effect_l2 | +0.800 [+0.744, +0.847] |  |  |
| Gemma-2-2B | stab_signed | +0.664 [+0.575, +0.746] |  |  |
| Gemma-2-2B | stab_abs | +0.602 [+0.507, +0.685] |  |  |
| Gemma-2-2B | kl_mean | +0.711 [+0.635, +0.775] |  |  |
| Gemma-2-2B | stab_anti_frac | +0.261 [+0.148, +0.371] |  |  |
| Gemma-2-2B | effect_pc1_ratio | +0.205 [+0.091, +0.316] |  |  |
| Llama-3.1-8B | collateral_raw | +0.946 [+0.928, +0.958] | +0.240 | +0.213 |
| Llama-3.1-8B | collateral_ctilde | +0.897 [+0.864, +0.922] | +0.222 | +0.163 |
| Llama-3.1-8B | effect_l2 | +0.854 [+0.810, +0.888] |  |  |
| Llama-3.1-8B | stab_signed | +0.694 [+0.614, +0.762] |  |  |
| Llama-3.1-8B | stab_abs | +0.520 [+0.419, +0.611] |  |  |
| Llama-3.1-8B | kl_mean | +0.375 [+0.265, +0.481] |  |  |
| Llama-3.1-8B | stab_anti_frac | +0.105 [-0.007, +0.218] |  |  |
| Llama-3.1-8B | effect_pc1_ratio | +0.103 [-0.011, +0.215] |  |  |

## Steering coefficient

GPT-2 and Pythia have alpha in {0.5, 1, 2, 4} and a q95 firing variant; Gemma and Llama have alpha in {0.5, 1, 2}. q95 is the 95th percentile conditional on firing (activation > 1e-6), multiplied by alpha; it is not the unconditional activation percentile. The primary control includes the varying intervention value for q95. These are coefficient sensitivity checks, without evidence that the selected scale improves generated behavior.

| Setting | Variant | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] | median coefficient |
|---|---|---|---|---|---|---|
| GPT-2-small | alpha1 | +0.482 | +0.475 [+0.375, +0.567] | +0.344 | +0.425 [+0.313, +0.524] | 1 |
| GPT-2-small | alpha0.5 | +0.487 | +0.493 [+0.391, +0.583] | +0.471 | +0.493 [+0.391, +0.584] | 0.5 |
| GPT-2-small | alpha2 | +0.460 | +0.457 [+0.355, +0.549] | +0.145 | +0.364 [+0.248, +0.467] | 2 |
| GPT-2-small | alpha4 | +0.347 | +0.382 [+0.271, +0.486] | -0.015 | +0.323 [+0.206, +0.424] | 4 |
| GPT-2-small | q95 firing | +0.352 | +0.264 [+0.156, +0.370] | -0.234 | +0.266 [+0.145, +0.366] | 4.16 |
| Pythia-70M-deduped | alpha1 | -0.022 | -0.004 [-0.121, +0.118] | +0.031 | +0.001 [-0.117, +0.123] | 1 |
| Pythia-70M-deduped | alpha0.5 | +0.102 | +0.136 [+0.017, +0.255] | +0.146 | +0.119 [-0.005, +0.243] | 0.5 |
| Pythia-70M-deduped | alpha2 | -0.046 | -0.048 [-0.167, +0.070] | +0.056 | -0.008 [-0.134, +0.120] | 2 |
| Pythia-70M-deduped | alpha4 | -0.036 | -0.076 [-0.199, +0.053] | +0.048 | -0.055 [-0.185, +0.079] | 4 |
| Pythia-70M-deduped | q95 firing | +0.214 | +0.037 [-0.085, +0.166] | +0.198 | +0.110 [-0.015, +0.226] | 0.792 |
| Gemma-2-2B | alpha1 | +0.512 | +0.242 [+0.121, +0.361] | +0.069 | +0.252 [+0.131, +0.366] | 1 |
| Gemma-2-2B | alpha0.5 | +0.555 | +0.261 [+0.149, +0.375] | +0.317 | +0.262 [+0.151, +0.374] | 0.5 |
| Gemma-2-2B | alpha2 | +0.413 | +0.181 [+0.060, +0.304] | -0.318 | +0.178 [+0.041, +0.305] | 2 |
| Llama-3.1-8B | alpha1 | +0.222 | +0.152 [+0.039, +0.265] | +0.201 | +0.166 [+0.055, +0.279] | 1 |
| Llama-3.1-8B | alpha0.5 | +0.251 | +0.179 [+0.069, +0.287] | +0.241 | +0.179 [+0.066, +0.288] | 0.5 |
| Llama-3.1-8B | alpha2 | +0.193 | +0.111 [-0.005, +0.229] | +0.060 | +0.108 [-0.005, +0.221] | 2 |

## Feature-selection band

The canonical band keeps features with final-token firing frequency in [0.002, 0.50]; the bandb variant uses [0.005, 0.30]. The eligible pool and the sampled 300 features differ between bands, so these are separate feature samples, not the same features under two filters. GPT-2 and Pythia only.

| Setting | Band | eligible | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] |
|---|---|---|---|---|---|---|
| GPT-2-small | [0.002, 0.50] | 7532 | +0.482 | +0.475 [+0.374, +0.569] | +0.344 | +0.425 [+0.317, +0.524] |
| GPT-2-small | [0.005, 0.30] | 3546 | +0.460 | +0.423 [+0.309, +0.524] | +0.244 | +0.372 [+0.265, +0.470] |
| Pythia-70M-deduped | [0.002, 0.50] | 4375 | -0.022 | -0.004 [-0.120, +0.116] | +0.031 | +0.001 [-0.117, +0.123] |
| Pythia-70M-deduped | [0.005, 0.30] | 2945 | -0.023 | -0.053 [-0.166, +0.062] | +0.016 | -0.052 [-0.173, +0.072] |

## Collateral threshold

The canonical threshold is tau = 0.05 on absolute downstream SAE activation change (Section 3.4). The sweep repeats the full pipeline at tau in {0.02, 0.1, 0.2} with identical settings otherwise; each tau resamples nothing, so within a setting the 300 features coincide across rows and only the labels change. C-tilde inherits the threshold through the count. These are threshold sensitivity checks on the labeling rule, not evidence for any particular threshold.

| Setting | tau | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] | median count |
|---|---|---|---|---|---|---|
| GPT-2-small | 0.05 | +0.482 | +0.475 [+0.374, +0.569] | +0.344 | +0.425 [+0.315, +0.522] | 11.5 |
| GPT-2-small | 0.02 | +0.434 | +0.432 [+0.330, +0.527] | +0.089 | +0.345 [+0.226, +0.453] | 21.6 |
| GPT-2-small | 0.1 | +0.487 | +0.493 [+0.391, +0.585] | +0.471 | +0.495 [+0.393, +0.587] | 3.0 |
| GPT-2-small | 0.2 | +0.447 | +0.464 [+0.357, +0.559] | +0.442 | +0.466 [+0.361, +0.563] | 0.2 |
| Pythia-70M-deduped | 0.05 | -0.022 | -0.004 [-0.124, +0.115] | +0.031 | +0.001 [-0.116, +0.121] | 22.1 |
| Pythia-70M-deduped | 0.02 | -0.029 | -0.034 [-0.149, +0.083] | +0.057 | +0.010 [-0.114, +0.136] | 48.4 |
| Pythia-70M-deduped | 0.1 | +0.110 | +0.147 [+0.028, +0.264] | +0.152 | +0.124 [-0.001, +0.247] | 6.2 |
| Pythia-70M-deduped | 0.2 | +0.165 | +0.175 [+0.058, +0.289] | +0.199 | +0.180 [+0.062, +0.298] | 0.9 |
| Gemma-2-2B | 0.05 | +0.512 | +0.242 [+0.123, +0.362] | +0.069 | +0.252 [+0.133, +0.369] | 4.3 |
| Gemma-2-2B | 0.02 | +0.380 | +0.156 [+0.040, +0.278] | -0.415 | +0.138 [-0.001, +0.272] | 19.2 |
| Gemma-2-2B | 0.1 | +0.553 | +0.260 [+0.149, +0.374] | +0.280 | +0.263 [+0.151, +0.372] | 0.9 |
| Gemma-2-2B | 0.2 | +0.478 | +0.201 [+0.089, +0.316] | +0.203 | +0.217 [+0.105, +0.326] | 0.3 |
| Llama-3.1-8B | 0.05 | +0.222 | +0.152 [+0.041, +0.261] | +0.201 | +0.166 [+0.054, +0.276] | 2.7 |
| Llama-3.1-8B | 0.02 | +0.203 | +0.126 [+0.012, +0.241] | +0.059 | +0.122 [+0.007, +0.237] | 8.1 |
| Llama-3.1-8B | 0.1 | +0.251 | +0.184 [+0.072, +0.288] | +0.229 | +0.185 [+0.070, +0.291] | 1.1 |
| Llama-3.1-8B | 0.2 | +0.245 | +0.159 [+0.053, +0.265] | +0.232 | +0.157 [+0.044, +0.267] | 0.5 |

The GPT-2, Gemma and Llama primary partials change little across the sweep, so the crowding result in those settings is not an artifact of the 0.05 choice. Pythia, null at the canonical threshold, shows a small positive partial at tau = 0.1 and 0.2 with intervals that exclude zero; at those thresholds the median Pythia count falls below 7 of 2,048 panel features, so the label is sparse and the shift should be read with that caution.


## Llama precision: bfloat16 vs float32

The canonical Llama run loads in bfloat16 with TransformerLens weight processing off; the retry loads the same checkpoint in float32. Both have 2,837 eligible features, but only 88 of the 300 sampled features coincide: dtype shifts final-token frequencies enough to move features across the eligibility band edges, so these are different feature samples, not the same features at two precisions.

| Run | dtype | feature overlap | raw count: rho | primary partial [95% CI] | C-tilde: rho | primary partial [95% CI] |
|---|---|---|---|---|---|---|
| canonical | bfloat16 | 300 of 300 | +0.222 | +0.152 [+0.039, +0.264] | +0.201 | +0.166 [+0.053, +0.277] |
| float32 retry | float32 | 88 of 300 | +0.221 | +0.142 [+0.020, +0.263] | +0.144 | +0.160 [+0.042, +0.273] |

## Paired random-direction control

Each of 300 isotropic Gaussian directions is normalized to unit length, then scaled to the corresponding sampled SAE decoder norm. Each pair uses identical 48-context indices, the same coefficient, downstream panel, and clean forward pass. This retains the nonunit Llama decoder scales. Direction seed is 2000; vectors and context indices are saved in each *_random_paired directory. Frequency and natural activation are undefined for random directions, so both arms use effect-only adjustment. The older *_random files used unmatched random contexts and are retained as exploratory artifacts. Within-arm correlations and pointwise CIs do not establish a difference between correlations, a causal role for crowding, or improved behavioral utility.

| Setting | Vectors | raw count: rho [95% CI] | partial given E_f [95% CI] | C-tilde: rho [95% CI] | partial given E_f [95% CI] | median count | median effect L2 | median C-tilde |
|---|---|---|---|---|---|---|---|---|
| GPT-2-small | SAE features | +0.482 [+0.379, +0.571] | +0.444 [+0.338, +0.537] | +0.344 [+0.232, +0.446] | +0.398 [+0.292, +0.498] | 11.5 | 3.86 | 2.89 |
| GPT-2-small | paired random directions | -0.063 [-0.173, +0.048] | -0.064 [-0.178, +0.050] | -0.077 [-0.187, +0.035] | -0.075 [-0.189, +0.038] | 8.7 | 3.52 | 2.47 |
| Pythia-70M-deduped | SAE features | -0.022 [-0.136, +0.093] | +0.008 [-0.108, +0.122] | +0.031 [-0.083, +0.142] | +0.016 [-0.095, +0.130] | 22.1 | 35.4 | 0.614 |
| Pythia-70M-deduped | paired random directions | +0.009 [-0.102, +0.123] | +0.007 [-0.109, +0.124] | +0.021 [-0.091, +0.133] | +0.013 [-0.106, +0.131] | 21.7 | 33.3 | 0.659 |
| Gemma-2-2B | SAE features | +0.512 [+0.411, +0.604] | +0.267 [+0.150, +0.382] | +0.069 [-0.048, +0.189] | +0.266 [+0.148, +0.379] | 4.3 | 5.48 | 0.759 |
| Gemma-2-2B | paired random directions | -0.012 [-0.128, +0.102] | -0.011 [-0.128, +0.105] | -0.006 [-0.119, +0.109] | -0.015 [-0.125, +0.095] | 3.2 | 5.01 | 0.626 |
| Llama-3.1-8B | SAE features | +0.222 [+0.109, +0.329] | +0.161 [+0.049, +0.270] | +0.201 [+0.092, +0.304] | +0.171 [+0.063, +0.278] | 2.7 | 14 | 0.184 |
| Llama-3.1-8B | paired random directions | -0.058 [-0.177, +0.058] | -0.089 [-0.197, +0.022] | -0.070 [-0.189, +0.048] | -0.091 [-0.198, +0.021] | 1.5 | 12.8 | 0.115 |

## Dense-panel exclusion

Panel features firing in more than 10% of clean contexts are excluded. The remaining panel is still selected by frequency; this check does not provide coverage of all sparse downstream features.

| Setting | panel kept | crowding vs count (all) | crowding vs count (no dense) | primary partial (no dense) [95% CI] |
|---|---|---|---|---|
| GPT-2-small | 2042 of 2048 | +0.482 | +0.486 | +0.477 [+0.373, +0.569] |
| Pythia-70M-deduped | 1969 of 2048 | -0.022 | -0.093 | -0.093 [-0.213, +0.030] |
| Gemma-2-2B | 2011 of 2048 | +0.512 | +0.515 | +0.196 [+0.078, +0.316] |
| Llama-3.1-8B | 1005 of 1024 | +0.222 | +0.242 | +0.144 [+0.030, +0.257] |

## Residual-space change

Let dh be the downstream hidden-state change and dr the change in its full SAE reconstruction. The unrepresented change is de = dh - dr. For each feature we report 1 - sum_context ||de||^2 / sum_context ||dh||^2, a reconstruction score relative to predicting zero change. It can be negative when reconstruction change is a worse prediction than zero. It is not a bounded variance partition: dr and de need not be orthogonal. The reconstruction norm ratio mean_context ||dr||/||dh|| is reported separately and is not an explained fraction. Neither measure proves behavioral harm or residual-mediated recovery. Per-context dh and dr are saved locally as residual_deltas.npz and verified against the CSV; these large arrays are excluded from git and can be regenerated with --save-residual-deltas.

| Setting | crowding vs ||dh|| [95% CI] | mean reconstruction norm ratio | mean explained-change score | median score | features with score < 0 |
|---|---|---|---|---|---|
| GPT-2-small | +0.328 [+0.222, +0.427] | 0.845 | -0.377 | -0.234 | 78.3% |
| Pythia-70M-deduped | +0.144 [+0.027, +0.261] | 0.778 | +0.466 | +0.452 | 0.0% |
| Gemma-2-2B | +0.626 [+0.539, +0.702] | 2.289 | -13.766 | -13.581 | 100.0% |
| Llama-3.1-8B | +0.172 [+0.054, +0.283] | 1.186 | -2.658 | -2.579 | 100.0% |

## Additional stability summaries

PC1 is the top eigenvalue divided by the trace of the centered logit-change Gram matrix over 48 contexts. Anti-aligned fraction counts contexts whose change has negative cosine with that feature's mean change, including the context in the mean. These are descriptive summaries inspired by FEGA and steering-reliability work; they do not reproduce feature removal or behavior-score multiplier slopes.

| Setting | median anti-aligned context fraction | median centered PC1 ratio |
|---|---|---|
| GPT-2-small | 0.000 | 0.200 |
| Pythia-70M-deduped | 0.000 | 0.336 |
| Gemma-2-2B | 0.104 | 0.448 |
| Llama-3.1-8B | 0.021 | 0.180 |
