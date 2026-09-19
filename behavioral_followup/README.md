# Behavioral follow-up to collateral prediction

[Read the September 18 research update PDF](RESEARCH_UPDATE_2026-09-18.pdf), compiled on September 19 from the verified results and proposed manuscript addendum. Full-paper integration is pending.

Start with [RESULTS.md](RESULTS.md), the [manuscript addendum](PAPER_ADDENDUM.md), [PROTOCOL.md](PROTOCOL.md) and the [documented amendments](AMENDMENTS.md). The secondary local-model assessment has a [separate frozen protocol](SECONDARY_JUDGE_PROTOCOL.md).

This experiment evaluates transfer of the existing collateral predictor to generated positive-sentiment reviews and preservation of factual-retrieval accuracy. Both GPT-2-small and Gemma-2-2B are complete, with 7,296 held-out outputs. Neither meets the joint transfer criterion. The original four-setting activation-collateral findings remain separate.

## Reproduce and verify

Install the repository requirements and pytest, then run from the repository root:

```bash
python -m pytest -q tests behavioral_followup/test_followup.py
python analysis/verify_artifacts.py --reference 4a639d4
python behavioral_followup/verify.py
python behavioral_followup/analyze.py
```

The strict behavioral audit checks the recorded package versions in each setting's `manifest.json`, reconstructs predictor scores, verifies complete development/test grids and scores, and audits the blinded assessment. Match those versions for the original numerical audit. `VERIFICATION_2026-09-18.json` preserves the original local audit; `VERIFICATION.json` records the publication-checkout audit. Historical source snapshots retain the exact generation and judge code.

For GPU generation in a separate copy, use the original recorded environment, a CUDA GPU, and authorized access to the Gemma checkpoint:

```bash
python behavioral_followup/run.py --setting gpt2_small --stage all
python behavioral_followup/run.py --setting gemma_2_2b --stage all --batch-size 64
python behavioral_followup/blind_judge.py
python behavioral_followup/analyze.py
python behavioral_followup/verify.py
```

The runner reads canonical inputs from the repository root. Set `DUAN_REPO_ROOT` to use another checkout. Generation resumes completed JSONL rows. Preserve the frozen protocol, prompts and selections; retain the published artifacts in a separate copy before any independent rerun. Generated text, token IDs, scores, model coefficients and verification records are included. Regenerable activation caches and runtime logs are excluded.

SiEBERT provides primary sentiment scores; Qwen3-4B supplies the secondary assessment on a fixed subset. Both run locally. These are automated assessments, not human annotation.

## PDF and figures

The report includes primary matched comparisons and secondary-judge tables. Exportable PNG, PDF and SVG figures are in `results/figures/`. To rebuild the six-page PDF with an installed Chromium browser:

```bash
pip install -r behavioral_followup/requirements-pdf.txt
python behavioral_followup/build_pdf.py --browser /path/to/chromium
```

The builder also recognizes `CHROMIUM`, browser executables on PATH, and cached Playwright headless Chromium. The archived September 18 PDF and numerical results are unchanged in this publication. The completed run and checkpoint recovery are documented in [STATUS_2026-09-18.md](STATUS_2026-09-18.md), [RESUME_2026-09-18.json](RESUME_2026-09-18.json) and [AMENDMENTS.md](AMENDMENTS.md).

Publication on September 19 changes local path resolution and PDF browser discovery. Generation, scoring, selection and statistical functions are unchanged. The resume audit checks the archived original runner hash; the current portable runner has its own source hash. Historical notes retain their original dates and describe the state before publication.
