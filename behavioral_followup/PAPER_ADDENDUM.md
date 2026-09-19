# Behavioral-transfer addendum for the revision

The completed pilot does not establish a behavioral-preservation advantage for selecting low predicted collateral. This is supplementary exploratory evidence; it does not alter the four-setting activation-collateral findings.

## Proposed manuscript text

We evaluated transfer of the collateral predictor to positive-sentiment steering and unrelated synthetic factual retrieval in GPT-2-small and Gemma-2-2B. The predictor was fitted to the 300 canonical activation-collateral labels in each setting, using only unsteered feature statistics as inputs. Candidate features were excluded from all three original feature samples. Development generations selected six low/high predicted-collateral feature pairs matched on sentiment gain, with additive strengths 1 and 2 in GPT-2 and Gemma, respectively. Six remaining sentiment-associated features were sampled as controls. Held-out evaluation used 64 review prompts with two sampling seeds and 64 factual prompts per feature or unsteered baseline, yielding 3,648 outputs per model.

| Setting | Low-minus-high positive probability, pp [95% CI] | Low-minus-high factual accuracy, pp | Factual accuracy in every arm |
|---|---:|---:|---:|
| GPT-2-small | +4.2 [-0.8, +9.2] | 0.0 | 96.9% |
| Gemma-2-2B | +0.6 [-2.4, +3.9] | 0.0 | 100.0% |

Intervals resample matched feature pairs and prompt identities, retaining the two review seeds together. Both settings satisfy the exploratory -5 percentage point sentiment noninferiority margin, but neither shows a factual-preservation advantage: correctness is identical across steering arms for every factual prompt. The joint transfer criterion is therefore unmet. In Gemma, neither selected arm improves mean held-out positive probability relative to the unsteered baseline, despite qualifying development gains. These results do not establish useful behavioral steering from this selector.

A secondary blinded Qwen3-4B assessment covered four fixed product-review prompts per feature and baseline, using one generation seed (76 outputs per model). Only 50% of GPT-2 low- and high-score outputs jointly passed positive-sentiment, coherence and relevance criteria; both Gemma groups scored 100% on this subset. These automated ratings have no human validation and cover a single prefix template. They illustrate why a sentiment score alone cannot establish generation quality.

The factual benchmark shows no intervention-induced correctness changes, so its zero-width empirical difference intervals do not establish general equivalence or preserved capability. The experiment uses one behavior, two fixed dictionaries, six feature pairs per setting and repeated generation-time steering; it is not a general behavioral validation of the original final-token activation measure. Any harder preservation benchmark or new selection rule should be evaluated in a separately frozen follow-up with fresh held-out prompts.

## Evidence

- [Full results and exportable figures](RESULTS.md).
- [Frozen protocol](PROTOCOL.md) and [secondary assessment protocol](SECONDARY_JUDGE_PROTOCOL.md).
- [Artifact verification](VERIFICATION.json) and [recovery/audit corrections](AMENDMENTS.md).
