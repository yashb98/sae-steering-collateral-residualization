#!/usr/bin/env bash
# Collateral-threshold sweep, branch ab-battery-2026-09-13. Tau 0.05 is the
# canonical run; this adds 0.02, 0.1, 0.2 for all four settings. Outputs land
# in results/<setting>_tau<tau>/ via the --tau suffix; canonical dirs untouched.
set -uo pipefail
cd "$(dirname "$0")"
export HF_HUB_DISABLE_XET=1
PY="$HOME/venvs/duan/bin/python"
LOG=results/logs/tau_sweep_2026-09-14.log
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

for tau in 0.02 0.1 0.2; do
    step gpt2_tau$tau   "$PY" src/run_setting.py --config configs/gpt2_small.yaml         --tau $tau
    step pythia_tau$tau "$PY" src/run_setting.py --config configs/pythia_70m_deduped.yaml --tau $tau
    step gemma_tau$tau  "$PY" src/run_setting.py --config configs/gemma_2_2b.yaml         --tau $tau
    step llama_tau$tau  "$PY" src/run_setting.py --config configs/llama_3_1_8b.yaml       --tau $tau
done

echo "=== SWEEP DONE $(date -Is)" | tee -a "$LOG"
