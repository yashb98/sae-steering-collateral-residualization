#!/usr/bin/env bash
# A/B battery, branch ab-battery-2026-09-13. Serialized GPU jobs; each step
# logs and continues on failure. Main-branch results under results/<setting>/
# are inputs only; every output here lands in a new results/ directory.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
PY="$HOME/venvs/duan/bin/python"
LOG=results/logs/ab_battery_2026-09-13.log
mkdir -p results/logs

step() {
    local label="$1"; shift
    echo "=== START $label $(date -Is)" | tee -a "$LOG"
    if "$@" >> "$LOG" 2>&1; then
        echo "=== OK    $label $(date -Is)" | tee -a "$LOG"
    else
        echo "=== FAIL  $label $(date -Is) (exit $?) - continuing" | tee -a "$LOG"
    fi
}

# Power A/B: Llama seeds 1 and 2 (main branch has seed 0 only)
step llama_seed1 "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml --seed 1 --out results/llama_3_1_8b_seed1 --trace-steering
step llama_seed2 "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml --seed 2 --out results/llama_3_1_8b_seed2 --trace-steering

# Precision A/B: Llama float32 retry, no other GPU load this time
step llama_fp32 "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml --dtype float32 --out results/llama_3_1_8b_fp32 --trace-steering

# Coefficient robustness gap: alpha 0.5 and 2 for Gemma and Llama
step gemma_alpha0.5 "$PY" src/run_setting.py --config configs/gemma_2_2b.yaml --alpha 0.5 --out results/gemma_2_2b_alpha0.5
step gemma_alpha2   "$PY" src/run_setting.py --config configs/gemma_2_2b.yaml --alpha 2   --out results/gemma_2_2b_alpha2
step llama_alpha0.5 "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml --alpha 0.5 --out results/llama_3_1_8b_alpha0.5
step llama_alpha2   "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml --alpha 2   --out results/llama_3_1_8b_alpha2

# Selection A/B: narrower frequency band for the two small settings
step gpt2_bandb   "$PY" src/run_setting.py --config configs/gpt2_small_bandb.yaml
step pythia_bandb "$PY" src/run_setting.py --config configs/pythia_70m_deduped_bandb.yaml

echo "=== BATTERY DONE $(date -Is)" | tee -a "$LOG"
