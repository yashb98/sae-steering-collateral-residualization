# Collateral-side residualization for SAE steering side effects

Code, configs and per-feature results extending the collateral analysis of *Pre-Intervention Prediction of Sparse Autoencoder Steering Side Effects* (Evan Duan, [arXiv:2606.08365](https://arxiv.org/abs/2606.08365)) from GPT-2-small to all four of the paper's model/SAE settings.

The paper residualizes its stability labels against effect magnitude, intervention value and natural activation (Section 3.8) but does not apply that control to its collateral labels, and firing frequency is not in the nuisance set. This repository runs that control. For each setting it measures steering collateral from scratch and asks which pre-intervention predictors, decoder crowding in particular, still predict collateral after partialling out the nuisance variables and firing frequency. Both collateral metrics of the paper are reported everywhere: the raw downstream count C_{f,0.05} (the target of Table 2) and C-tilde = C / (E_f + eps) (the primary metric of Section 3.4).

The GPT-2-small version of this analysis was first published as a Kaggle notebook, [SAE Steering Side Effects Reproduced](https://www.kaggle.com/code/yashbishnoi98/sae-steering-side-effects-reproduced). `src/run_setting.py` is that notebook generalised to a per-setting config; with `--protocol v1` it reproduces the notebook's GPT-2-small numbers (`results/gpt2_small_v1/meta.json`).

## Settings

Configuration follows the paper's Appendix Tables A1, A2 and F1 and the author's confirmation of two values the paper leaves symbolic: top-k for crowding = 20 and downstream panel size = 2048 (1024 for Llama).

| Setting | Model | SAE release (SAELens) | Primary hook | Downstream hook | Panel |
|---|---|---|---|---|---|
| gpt2_small | gpt2 | gpt2-small-res-jb | blocks.8.hook_resid_pre | blocks.10.hook_resid_pre | 2048 |
| pythia_70m_deduped | EleutherAI/pythia-70m-deduped | pythia-70m-deduped-res-sm | blocks.4.hook_resid_post | blocks.5.hook_resid_post | 2048 |
| gemma_2_2b | google/gemma-2-2b | gemma-scope-2b-pt-res-canonical, layer_12 and layer_16, width_16k | blocks.12.hook_resid_post | blocks.16.hook_resid_post | 2048 |
| llama_3_1_8b | meta-llama/Llama-3.1-8B | llama_scope_lxr_8x, l16r_8x and l20r_8x | blocks.16.hook_resid_post | blocks.20.hook_resid_post | 1024 |

Common to all: Wikitext-103 train split, 8,000 texts, 2,048 contexts of 48 tokens, 300 sampled features, 16 contexts per type (top-activating, random, low-activating), additive steering alpha = 1.0 at the final token, collateral threshold tau = 0.05, firing threshold 1e-6.

## Install and run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
huggingface-cli login            # gemma-2-2b and Llama-3.1-8B are gated

python src/run_setting.py --config configs/pythia_70m_deduped.yaml   # one setting
python src/run_setting.py --config configs/gpt2_small.yaml --smoke   # pipeline check on tiny sizes
python analysis/residualize.py --n-boot 10000                        # analysis over every results/<setting>/
python analysis/make_results.py                                      # regenerate RESULTS.md and the LaTeX table
pytest -q                                                            # tests
```

`--protocol v1` reproduces the Kaggle notebook's context construction exactly (regression gate). The default `v2` drops texts shorter than 48 tokens, so the final token is never a pad token, and deduplicates texts after whitespace normalisation.

## Outputs

`results/<setting>/per_feature.csv` has one row per sampled feature: the pre-intervention predictors (decoder geometry, activation statistics, co-activation, direct-logit) and the steering labels (collateral count, effect magnitude, C-tilde, signed and absolute stability, KL shift). `results/analysis/` holds `partial_correlations.csv` (every predictor, both metrics, three control sets, bootstrap CIs), `residualized_cv_ridge.csv` (the Table B3 analog), `seed_summary.csv` and `summary.md`. `RESULTS.md` is the write-up, generated from these files.

## Assumptions recorded in the outputs

- Natural activation is the mean final-token activation over all 2,048 contexts. Intervention value is constant under fixed_global_add with alpha = 1.0 and is dropped from the regression.
- Feature sampling uses seed 0 over features with final-token firing frequency in [0.002, 0.50]; the paper says only "the non-degenerate range".
- The downstream panel is the most frequently active downstream features on clean contexts.
- Models are loaded in float32 with TransformerLens default weight processing plus the kwargs stored in each SAE's SAELens config.

## License

MIT. See `LICENSE`.
