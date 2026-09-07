#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
for setting in gpt2_small pythia_70m_deduped gemma_2_2b llama_3_1_8b; do
    python src/run_setting.py --config "configs/$setting.yaml" --paired-random-control --save-residual-deltas
    for split in A B; do
        python src/run_setting.py --config "configs/$setting.yaml" --context-split "$split"
    done
done
for setting in gpt2_small pythia_70m_deduped; do
    for alpha in 0.5 2 4; do
        python src/run_setting.py --config "configs/$setting.yaml" --alpha "$alpha"
    done
    python src/run_setting.py --config "configs/$setting.yaml" --alpha-mode q95
done
python analysis/residualize.py --n-boot 10000
python analysis/variants.py --n-boot 10000
python analysis/seed_summary.py
python analysis/make_results.py
python analysis/verify_artifacts.py
python -m pytest -q
python kaggle/make_notebook.py
