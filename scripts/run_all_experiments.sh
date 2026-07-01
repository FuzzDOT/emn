#!/usr/bin/env bash
# ==============================================================================
# EMN: run_all_experiments.sh
# Master orchestration script — runs all 3 experiments end-to-end.
#
# Usage:
#   bash scripts/run_all_experiments.sh                    # full run
#   bash scripts/run_all_experiments.sh --fast-test        # smoke test
#   bash scripts/run_all_experiments.sh --skip-exp 2 3     # only exp1
#   SEEDS="42 43 44" bash scripts/run_all_experiments.sh  # custom seeds
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ── Defaults ────────────────────────────────────────────────────────────────
FAST_TEST=0
SKIP_EXPS=()
SEEDS="${SEEDS:-42 43 44}"
DEVICE="${DEVICE:-cpu}"
RESULTS_DIR="${RESULTS_DIR:-results}"
FIGURES_DIR="${FIGURES_DIR:-figures}"
MEMORY_AGENT_BENCH_REPO="https://github.com/HUST-AI-HYZ/MemoryAgentBench.git"
MEMORY_AGENT_BENCH_PATH="${REPO_ROOT}/external/MemoryAgentBench"
TINYLLAMA_MODEL="TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# ── Argument parsing ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --fast-test)   FAST_TEST=1 ;;
    --skip-exp)    shift; SKIP_EXPS+=("$1") ;;
    --device)      shift; DEVICE="$1" ;;
    --results-dir) shift; RESULTS_DIR="$1" ;;
    --figures-dir) shift; FIGURES_DIR="$1" ;;
    --seeds)       shift; SEEDS="$1" ;;
    --model)       shift; TINYLLAMA_MODEL="$1" ;;
    *)             echo "Unknown argument: $1"; exit 1 ;;
  esac
  shift
done

# ── Helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log()     { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
fail()    { echo -e "${RED}✗${NC} $*"; exit 1; }

should_skip() {
  local exp=$1
  for skip in "${SKIP_EXPS[@]:-}"; do
    [[ "$skip" == "$exp" ]] && return 0
  done
  return 1
}

# ── Banner ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║   Epistemic Memory Networks (EMN) — Full Experiment Run  ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
log "Repo root:    ${REPO_ROOT}"
log "Results dir:  ${RESULTS_DIR}"
log "Figures dir:  ${FIGURES_DIR}"
log "Seeds:        ${SEEDS}"
log "Device:       ${DEVICE}"
log "Fast test:    ${FAST_TEST}"
echo ""

# ── Check Python environment ─────────────────────────────────────────────────
log "Checking Python environment..."
python -c "import torch; print(f'  PyTorch {torch.__version__}')" || \
  fail "PyTorch not found. Run: conda env create -f environment.yml && conda activate emn"
python -c "import emn; print(f'  EMN package found (version {emn.__version__})')" 2>/dev/null || {
  warn "EMN not installed as package; adding src/ to PYTHONPATH"
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
}
python -c "import emn; print(f'  EMN {emn.__version__}')"

mkdir -p "${RESULTS_DIR}" "${FIGURES_DIR}" data paper/tables paper/figures

# ── Clone MemoryAgentBench ────────────────────────────────────────────────────
log "Checking MemoryAgentBench..."
if [[ -d "${MEMORY_AGENT_BENCH_PATH}" ]]; then
  success "MemoryAgentBench already present at ${MEMORY_AGENT_BENCH_PATH}"
else
  log "Cloning MemoryAgentBench..."
  mkdir -p external
  git clone --depth 1 "${MEMORY_AGENT_BENCH_REPO}" "${MEMORY_AGENT_BENCH_PATH}" || \
    fail "Failed to clone MemoryAgentBench. Check your internet connection."
  success "MemoryAgentBench cloned."
fi
export MEMORY_AGENT_BENCH_PATH

# ── Experiment 1: Selective Forgetting ───────────────────────────────────────
if should_skip 1; then
  warn "Skipping Experiment 1"
else
  echo ""
  echo "──────────────────────────────────────────────────────────"
  log "Experiment 1: Selective Forgetting (MemoryAgentBench)"
  echo "──────────────────────────────────────────────────────────"

  EXP1_ARGS=(
    "--seeds" ${SEEDS}
    "--device" "${DEVICE}"
    "--results-dir" "${RESULTS_DIR}"
    "--figures-dir" "${FIGURES_DIR}"
  )
  [[ $FAST_TEST -eq 1 ]] && EXP1_ARGS+=("--max-items" "20")

  python experiments/exp1_selective_forgetting.py "${EXP1_ARGS[@]}"
  success "Experiment 1 complete."
fi

# ── Experiment 2: Continual Learning ─────────────────────────────────────────
if should_skip 2; then
  warn "Skipping Experiment 2"
else
  echo ""
  echo "──────────────────────────────────────────────────────────"
  log "Experiment 2: Continual Learning (Split-CIFAR100)"
  echo "──────────────────────────────────────────────────────────"

  EXP2_ARGS=(
    "--seeds" ${SEEDS}
    "--device" "${DEVICE}"
    "--results-dir" "${RESULTS_DIR}"
    "--figures-dir" "${FIGURES_DIR}"
  )
  [[ $FAST_TEST -eq 1 ]] && EXP2_ARGS+=("--fast-test")

  python experiments/exp2_continual_learning.py "${EXP2_ARGS[@]}"
  success "Experiment 2 complete."
fi

# ── Experiment 3: Confabulation ───────────────────────────────────────────────
if should_skip 3; then
  warn "Skipping Experiment 3"
else
  echo ""
  echo "──────────────────────────────────────────────────────────"
  log "Experiment 3: Confabulation Benchmark (${TINYLLAMA_MODEL})"
  echo "──────────────────────────────────────────────────────────"

  EXP3_ARGS=(
    "--model-name" "${TINYLLAMA_MODEL}"
    "--seed" "42"
    "--device" "${DEVICE}"
    "--results-dir" "${RESULTS_DIR}"
    "--figures-dir" "${FIGURES_DIR}"
  )
  [[ $FAST_TEST -eq 1 ]] && EXP3_ARGS+=("--fast-test")

  python experiments/exp3_confabulation.py "${EXP3_ARGS[@]}"
  success "Experiment 3 complete."
fi

# ── Generate all figures ──────────────────────────────────────────────────────
if [[ $FAST_TEST -eq 0 ]]; then
  echo ""
  log "Generating all publication figures..."
  python -c "
import sys; sys.path.insert(0, 'src')
from emn.utils.plotting import generate_all_figures
generate_all_figures(results_dir='${RESULTS_DIR}', output_dir='${FIGURES_DIR}')
"
  success "All figures generated in ${FIGURES_DIR}/"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║                    All experiments done!                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
log "Results:  ${RESULTS_DIR}/"
log "Figures:  ${FIGURES_DIR}/"
log "Tables:   paper/tables/"
echo ""
echo "Output files:"
find "${RESULTS_DIR}" -name "*.json" -o -name "*.csv" -o -name "*.tex" 2>/dev/null | \
  sort | sed 's/^/  /'
find "${FIGURES_DIR}" -name "*.pdf" -o -name "*.png" 2>/dev/null | \
  sort | sed 's/^/  /'
echo ""
success "Done. Ready to write the paper."
