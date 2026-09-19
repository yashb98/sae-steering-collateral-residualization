# Revision materials, 19 September 2026

This repository contains the independent four-setting collateral residualization and robustness study, plus a completed exploratory behavioral follow-up. The manuscript text is proposed for incorporation into the paper; the research-update PDF is a supplement.

## Four-setting contribution

Read [RESULTS.md](RESULTS.md), [the proposed manuscript paragraph and limitations](PAPER_INSERT_2026-09-15.md), and [the LaTeX table](results/analysis/table_collateral_residualized.tex). Per-feature measurements, configurations and analysis code are included. The current results incorporate the September 14 coefficient, precision, feature-selection and threshold checks and the September 15 interpretation corrections.

Crowding's primary partial correlations with raw collateral are +0.475 in GPT-2, -0.004 in Pythia, +0.242 in Gemma and +0.152 in Llama. The Llama crowding result does not pass Holm correction. Feature-sampling seeds and context-pool halves are distinct from dictionary reseeds and independent corpora; those broader generalization checks remain untested. Exact matching to the author's original experiment still depends on his per-feature files, feature/panel selections and nuisance definitions.

## Behavioral follow-up

Read the [six-page PDF](behavioral_followup/RESEARCH_UPDATE_2026-09-18.pdf), [full results](behavioral_followup/RESULTS.md) and [proposed addendum](behavioral_followup/PAPER_ADDENDUM.md). There are 3,648 held-out outputs per model, six matched low/high feature pairs and six random candidate controls. The secondary blinded assessment covers 76 fixed continuations per model.

The low-minus-high sentiment differences are +4.2 percentage points in GPT-2 (95% CI -0.8 to +9.2) and +0.6 in Gemma (-2.4 to +3.9). Factual correctness is identical across every arm in each setting: 96.9% in GPT-2 and 100% in Gemma. Neither meets the joint behavioral-transfer criterion. These outcomes were retained without retuning the frozen experiment.

## Verification

The publication checks cover the 17 original tests and seven behavioral tests, independent canonical-artifact checks, and both behavioral audits. See [canonical verification](results/analysis/artifact_verification.json) and [behavioral verification](behavioral_followup/VERIFICATION.json). The original dated behavioral audit is retained separately. [Behavioral reproduction instructions](behavioral_followup/README.md) explain the recorded environment and portable checkout paths.

Generated outputs, frozen prompts, selections, historical source snapshots and the September 18 PDF are preserved. Runtime logs, model weights and regenerable activation caches are excluded. The publication changes to the behavioral runner affect local path resolution; generation and analysis functions are unchanged.
