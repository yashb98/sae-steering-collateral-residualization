#!/usr/bin/env bash
# Full reproduction: four settings, then the analysis. Needs a GPU and a Hugging Face login.
set -euo pipefail
cd "$(dirname "$0")"
for s in pythia_70m_deduped gpt2_small gemma_2_2b llama_3_1_8b; do
  python src/run_setting.py --config "configs/$s.yaml" 2>&1 | tee "results/logs/$s.log"
done
python analysis/residualize.py --n-boot 10000
