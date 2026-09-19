# Secondary blinded assessment

Written before held-out generation on 18 September 2026. In addition to the primary fixed SiEBERT and exact-answer scores, assess a fixed subset with the cached Qwen3-4B model. Use every evaluated feature and unsteered baseline, the first generation seed, and review prompt IDs 00, 08, 16 and 24. Do not select examples based on outputs.

Score positive sentiment, coherence and topical relevance separately, using restricted A/B next-token probabilities. The judge receives only the prompt, continuation and criterion. Hash identifiers and shuffle jobs; omit all arm labels, feature IDs, coefficients and original scores. Calibrate its sentiment criterion on the same separate 32-sentence balanced reference set. Report agreement with SiEBERT and positive-plus-coherent-plus-on-topic rates. This is an exploratory automated validity check, not human annotation or an additional feature-selection criterion. Do not tune primary selection or scores in response.
