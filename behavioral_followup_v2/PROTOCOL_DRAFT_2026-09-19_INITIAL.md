# Behavioral follow-up v2 protocol: sensitive preservation benchmark

Written 19 September 2026 before any v2 development generations or held-out outcomes were observed. This is a locally timestamped exploratory protocol, not an external preregistration. It is a separate experiment from the completed September 18 follow-up; no v2 prompt, feature selection or outcome may reuse v1 held-out data, and no v1 held-out outcome informs any rule below.

## Motivation from v1

The September 18 experiment failed the joint behavioral-transfer criterion only on the factual-preservation gate: factual accuracy was identical across all arms (96.9% GPT-2-small, 100.0% Gemma-2-2B), yielding a zero-width difference interval whose lower bound cannot exceed zero. The benchmark was saturated — answers were supplied in the prompt and decoding was deterministic — so no steering arm could degrade accuracy and no preservation advantage was demonstrable. Sentiment noninferiority passed in both models. This version keeps the question and the joint criterion but replaces the preservation benchmark and the coefficient-selection rule so that collateral damage, if present, is measurable.

Question (unchanged): does selecting SAE features using the existing collateral predictor preserve unrelated factual-answer accuracy while achieving comparable positive-sentiment steering on new review continuations?

## Scope and separation

- Run GPT-2-small and Gemma-2-2B at the original primary SAE hooks, using the original float32 loading conventions and the frozen collateral predictor from v1 (300 canonical raw-collateral labels, unsteered feature statistics as inputs). Do not refit the predictor on any v2 data.
- Reuse the v1 discovery activations and candidate pool exclusions (union of canonical and feature-seed run features excluded). Candidates remain unlabeled in score training. Sentiment discovery may add new paired sentences only if the firing-frequency eligibility band and the 3% positive-context firing rule are unchanged; report pool size.
- Fresh prompts everywhere: new discovery sentences (if any), new development prompts, new classifier-calibration sentences and new held-out test prompts. All v1 development and test prompts are retired from v2 evaluation. Test prompts and generation seeds are fixed before any development generation. Test outcomes cannot alter feature selection, strength, or scoring rules.

## Sensitive preservation benchmark

- Replace in-context answer supply with **parametric short-answer retrieval**: the model must answer from its own knowledge (TriviaQA-style factual questions with short gold answers), deterministic decoding, 8 continuation tokens, EOS respected. Exact scoring reads the first answer segment and requires the expected answer at its start with a word boundary; aliases are frozen before development.
- Sensitivity gate, checked on development data before selection: unsteered exact-answer accuracy must lie between 40% and 85%. Below 40% the benchmark is at floor (nothing to preserve); above 85% it is at ceiling (no headroom to detect degradation, the v1 failure mode). If outside the band, adjust question difficulty once using a frozen difficulty-mixing rule (swap in pre-listed easier/harder items), recheck once, and if still outside the band report an insensitive benchmark and do not claim preservation evidence.
- Held-out factual set: 128 new questions per feature or baseline (double v1), disjoint from development.

## Development and selection

- Sentiment steering development as in v1: 12 new neutral review prefixes, 32 continuation tokens, fixed seed, temperature 0.8, top-k 50; SiEBERT scoring verified on a new separate balanced 32-sentence calibration set; degenerate definition unchanged (fewer than eight words or repeated-trigram fraction above 0.20).
- Candidate evaluation at additive coefficients {1, 2, 4, 8}, decoder direction applied at the last real prompt position and each generated position, native decoder norm preserved.
- **Coefficient regime rule (replaces the v1 smallest-six-pairs rule):** choose the smallest coefficient at which the pooled random-control arm's development factual accuracy falls at least 5 percentage points below unsteered development accuracy. This places evaluation in a regime where collateral damage is measurable, so a preservation advantage can in principle be observed. If no coefficient in {1, 2, 4, 8} produces a 5-point random-control drop, use coefficient 8 and report that the damage regime was not reached; do not interpret absence of a difference as preservation.
- Effective-candidate eligibility unchanged: development positive-probability gain at least 0.05 over unsteered and at most 25% degenerate continuations.
- Median split on predicted raw-collateral score; match low/high pairs on development sentiment gain, discovery selectivity and decoder norm; sentiment-gain difference at most 0.10 per pair. Take up to **10 pairs** (v1 took 6) at the coefficient chosen above. Report a failed gate if fewer than 4 pairs are available; do not claim selector validation in that case.
- Add ten randomly sampled remaining sentiment-associated candidates as controls, plus unsteered generation. Random-feature controls are distinct from random-vector controls.
- Save and hash all candidate scores, development outputs and final selections before held-out generation.

## Held-out evaluation

- 128 new review prefixes with two seeds each; 128 new factual questions with deterministic decoding. Continuations: 32 tokens (reviews), 8 tokens (factual). Match prompts and sampling uniforms across arms; retain every output including failures and empty continuations.
- Primary intended outcome: mean classifier positive probability. Primary unwanted outcome: factual exact-answer accuracy loss relative to the same unsteered prompts; also report broken-originally-correct and repaired-originally-wrong fractions.
- Primary comparison: matched low-score versus high-score features. Report both differences; lower collateral counts alone cannot count as behavioral success.
- Bootstrap matched feature pairs and held-out prompt identities independently with replacement (5,000 draws), keeping both review seeds together per prompt. Report 95% pointwise intervals, pair counts and prompt counts.
- **Joint transfer criterion (unchanged form):** a promising transfer requires the lower 95% bound above -0.05 for low-minus-high mean positive probability and above zero for low-minus-high factual accuracy. Exploratory decision rule, not a multiplicity-corrected claim.
- Power note: with 128 factual prompts per feature and at least 4 matched pairs, the paired design detects a low-minus-high factual advantage of roughly 5 percentage points when per-prompt accuracy is near 70%; with 10 pairs the detectable difference is roughly half that. Report the observed pair count and the corresponding detectable difference alongside the intervals.

## Secondary blinded assessment

Repeat the v1 secondary protocol unchanged on new fixed continuations: blinded Qwen3-4B judge, hash identifiers and shuffled jobs, positive/coherence/relevance criteria, calibration on a new separate 32-sentence set. Descriptive validity check only; never a selection criterion; no tuning in response.

## Reporting and limits

Same discipline as v1: source hashes, model revisions, package versions, prompt/selection hashes, raw token IDs and text, independent scores, summaries and a reproducible report; null and adverse results retained; no retuning or re-selection from held-out outcomes. Additional known limits for v2: parametric short-answer QA still does not establish preservation of reasoning, safety or broad knowledge; the coefficient-regime rule selects on development random-control behavior, so conclusions are specific to the chosen regime; repeated generation-time steering still differs from the original final-token intervention.

References: v1 follow-up at ../behavioral_followup (PROTOCOL.md, RESULTS.md, STATUS_2026-09-18.md); original collateral repository at ../../../../../../sae-steering-collateral-residualization; https://arxiv.org/html/2606.08365v1; https://huggingface.co/siebert/sentiment-roberta-large-english.
