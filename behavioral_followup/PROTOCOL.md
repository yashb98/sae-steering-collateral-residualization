# Behavioral follow-up protocol

Written 18 September 2026 before development generations or held-out outcomes were observed. This is a locally timestamped exploratory protocol, not an external preregistration.

Question: does selecting SAE features using our existing collateral predictor preserve unrelated factual-answer accuracy while achieving comparable positive-sentiment steering on new review continuations?

## Scope and separation

- Run GPT-2-small and Gemma-2-2B at the original primary SAE hooks, using the original float32 loading conventions.
- Fit the existing full-no-magnitude standardized ridge (alpha 1) to the 300 canonical raw-collateral labels. Also save a predictor trained on the primary-control residualized label as a secondary diagnostic. All score inputs are unsteered feature statistics.
- Discover sentiment-associated features using paired positive/negative sentences, without intervening. Exclude the union of features in canonical and feature-seed runs from the candidate pool. Keep the original Wikitext final-token frequency eligibility band. Take up to 48 features with the most positive standardized activation contrasts, requiring positive-context firing in at least 3% of discovery sentences (three of 96). The available eligible pool can be smaller; report its actual size.
- Recompute the original predictor definitions on the same 2,048-context protocol and verify them against the canonical feature rows. Candidate features have no labels in score training.
- Separate discovery sentences, development prompts, classifier-calibration sentences and final test prompts. Test prompts and generation seeds are fixed before development. Test outcomes cannot alter feature selection, strength or scoring rules.

## Development and selection

- Use 12 neutral review prefixes, 32 continuation tokens and one fixed sampling seed, temperature 0.8, top-k 50.
- Evaluate all candidates at additive coefficients 1, 2, 4 and 8. This expands the previous strength range to establish a behavioral effect; conclusions remain specific to this generation protocol.
- Apply the SAE decoder direction at the last real prompt position and each newly generated token position. Earlier prompt positions remain clean. Preserve each decoder's native norm.
- Independently score generated continuations with SiEBERT (siebert/sentiment-roberta-large-english). Verify its label mapping and performance on a separate balanced 32-sentence calibration set. Report classifier uncertainty and limitations; these are automated labels, not human judgments.
- Eligible effective candidates improve mean positive probability over unsteered development continuations by at least 0.05, with no more than 25% degenerate continuations. Degenerate means fewer than eight words or repeated-trigram fraction greater than 0.20.
- Divide all candidates at the median predicted raw-collateral score. At each coefficient, match effective low-score and high-score candidates using development sentiment gain, discovery selectivity and decoder norm. Require sentiment-gain difference at most 0.10 per pair. Choose the smallest coefficient permitting six pairs. If none permits six, choose the coefficient with the largest number (up to six) of qualifying pairs, breaking ties toward smaller coefficients. Report a failed gate if fewer than three pairs are available; do not claim selector validation in that case.
- Add six randomly sampled remaining sentiment-associated candidates, without conditioning on their development effects, plus unsteered generation. Random-feature controls are distinct from random-vector controls.
- Check unrelated factual-answer prompts on development data. If unsteered exact-answer accuracy is below 70%, report an insensitive preservation benchmark; do not interpret failure to detect degradation as preservation.
- Save and hash all candidate scores, development outputs and final selections before held-out generation.

## Held-out evaluation

- 64 new review prefixes with two seeds each; 64 new factual-retrieval questions with deterministic decoding. Continuations are 32 tokens for reviews and 8 for factual answers, with EOS respected.
- Factual questions supply an answer in a short fact and use fixed few-shot examples. Exact scoring reads the first answer segment and requires the expected answer at its start with a word boundary. It does not give credit for an answer appearing later in an irrelevant continuation.
- Match prompts and sampling uniforms across steering arms. Retain every output, including failures and empty continuations.
- Primary intended outcome: mean classifier positive probability; also report positive-label rate, nondegenerate positive rate, and repetition/length diagnostics.
- Primary unwanted outcome: factual exact-answer accuracy loss relative to the same unsteered prompts. Also report the fraction of originally correct answers broken by steering and originally wrong answers repaired.
- Primary comparison: matched low-score versus high-score features. Report both intended-success and factual-accuracy differences; lower collateral counts alone cannot count as behavioral success.
- Bootstrap matched feature pairs and held-out prompt identities independently with replacement (5,000 draws), keeping both review seeds together per prompt. Report 95% pointwise intervals, the number of feature pairs and prompts. These conditional intervals do not generalize across dictionaries, model seeds or all behaviors.
- A promising transfer requires a lower 95% confidence bound above -0.05 for low-minus-high mean positive probability and above zero for low-minus-high factual accuracy. This is an exploratory decision rule, not a multiplicity-corrected claim of universal superiority.

## Reporting and limits

Save source hashes, model revisions, package versions, prompt/selection hashes, raw generated token IDs and text, independent scores, summaries and a reproducible report. Retain null and adverse results. Development matching is not guaranteed to equalize held-out success, and single-model sentiment classification cannot establish semantic quality or safety. This evaluates one behavior and one synthetic capability-preservation benchmark. It extends one-token activation measurements to repeated generation-time interventions; failure to transfer may reflect that protocol change.

References: original collateral repository at ../../../../../../sae-steering-collateral-residualization (resolved explicitly by the runner); https://arxiv.org/html/2606.08365v1; https://huggingface.co/siebert/sentiment-roberta-large-english.
