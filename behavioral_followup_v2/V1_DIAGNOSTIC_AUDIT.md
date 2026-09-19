# Diagnostic audit of the completed behavioral study

19 September 2026. Post-hoc exploratory analysis of saved outputs; no new model generations. These analyses were not preregistered and do not replace the original criterion or results.

The key distinction is intended steering versus unsteered performance, separately from low/high noninferiority. Intervals below resample selected features and prompt identities (5,000 draws), retaining review seeds together. They are pointwise, conditional intervals, with no multiplicity correction or guarantee of generalization to new feature populations.

| Model | Arm | Development gain | Test gain vs unsteered, 95% CI | Features with test gain >=5 pp |
|---|---|---:|---:|---:|
| gpt2_small | low | +8.90 pp | +4.78 pp [-1.89, +11.48] | 3/6 |
| gpt2_small | high | +9.66 pp | +0.57 pp [-5.44, +6.72] | 1/6 |
| gpt2_small | random | -6.94 pp | +2.02 pp [-4.26, +8.62] | 0/6 |
| gemma_2_2b | low | +7.90 pp | -0.38 pp [-3.90, +3.21] | 0/6 |
| gemma_2_2b | high | +8.22 pp | -1.02 pp [-5.09, +2.95] | 0/6 |
| gemma_2_2b | random | -6.22 pp | -1.36 pp [-5.44, +2.51] | 0/6 |

Development gains are selected estimates; their change on test mixes selection optimism, prompt differences and sampling variation. This audit does not identify their separate causes.

## gpt2_small

Factual correctness changed in 0/1152 feature/prompt comparisons. Full continuation text changed in 6 comparisons; this includes text after the scored answer and is not a harm measure.

The secondary judge covered 4 prompt IDs and 1/8 review templates. 34/76 judged outputs passed the simple nondegeneration filter but were rated incoherent. The judge is another model, not human ground truth.

## gemma_2_2b

Factual correctness changed in 0/1152 feature/prompt comparisons. Full continuation text changed in 5 comparisons; this includes text after the scored answer and is not a harm measure.

The secondary judge covered 4 prompt IDs and 1/8 review templates. 0/76 judged outputs passed the simple nondegeneration filter but were rated incoherent. The judge is another model, not human ground truth.

## Consequences for the next study

- Establish intended benefit against an unsteered baseline on fresh validation data, in addition to low/high noninferiority.
- Validate that an independently chosen control intervention produces measurable factual errors at usable output quality. A perfect unsteered baseline does not prevent measuring degradation.
- Cover distinct prompt templates and add blinded human review before making semantic-quality claims.
- Measure downstream activation collateral and behavioral errors under matched prompts and intervention schedules. The saved outputs cannot establish that link without new activation measurements.
- Plan sample size using pilot variance across independent feature pairs and prompt families. Repeated generations are not independent feature replications.

Verification: both original primary sentiment contrasts and feature-level sentiment gains reproduce within 1e-12; all 73 checked v1 files remained byte-identical. Full input hashes and descriptive template results are in [V1_DIAGNOSTIC_AUDIT.json](V1_DIAGNOSTIC_AUDIT.json).
