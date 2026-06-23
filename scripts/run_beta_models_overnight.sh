#!/usr/bin/env bash
# Run all three beta-GLMM comparison pipelines sequentially.
#
# Usage:
#   ./scripts/run_beta_models_overnight.sh --tmux          # detached tmux session (recommended overnight)
#   ./scripts/run_beta_models_overnight.sh                 # foreground in current shell
#   ./scripts/run_beta_models_overnight.sh --run-id 20260529_manual
#   BETA_SMOKE=1 ./scripts/run_beta_models_overnight.sh --tmux   # minimal MCMC smoke test (all pipelines)
#   ./scripts/run_beta_models_overnight.sh --smoke --run-id smoke_test
#
# Environment (optional):
#   BETA_DATA_DIR          default: <repo>/data/sully_og
#   BETA_COMPARISON_ROOT   default: <repo>/data/sully_og/output/comparison_runs
#   BETA_SMOKE=1           minimal MCMC + skip heavy post-processing in Rmds
#   BETA_SKIP=my1,preprocessed,reparam   comma-separated skip list

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NATIVE_R="${REPO_ROOT}/src/native_r"
DATA_DIR="${BETA_DATA_DIR:-${REPO_ROOT}/data/sully_og}"
COMPARISON_ROOT="${BETA_COMPARISON_ROOT:-${DATA_DIR}/output/comparison_runs}"

RUN_ID=""
USE_TMUX=0
SKIP_LIST=""
SMOKE=0

usage() {
  sed -n '2,14p' "$0"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tmux)
      USE_TMUX=1
      shift
      ;;
    --smoke)
      SMOKE=1
      shift
      ;;
    --run-id)
      RUN_ID="${2:?--run-id requires a value}"
      shift 2
      ;;
    --skip)
      SKIP_LIST="${2:?--skip requires a comma-separated list}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

should_skip() {
  local name="$1"
  [[ ",${SKIP_LIST}," == *",${name},"* ]]
}

if [[ "${USE_TMUX}" -eq 1 ]]; then
  SESSION="beta_models_overnight"
  if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session '${SESSION}' already exists."
    echo "  attach:  tmux attach -t ${SESSION}"
    echo "  kill:    tmux kill-session -t ${SESSION}"
    exit 1
  fi
  TMUX_CMD=(bash "$(printf '%q' "$0")")
  [[ "${SMOKE}" -eq 1 ]] && TMUX_CMD+=(--smoke)
  [[ -n "${RUN_ID}" ]] && TMUX_CMD+=(--run-id "$(printf '%q' "$RUN_ID")")
  [[ -n "${SKIP_LIST}" ]] && TMUX_CMD+=(--skip "$(printf '%q' "$SKIP_LIST")")
  tmux new-session -d -s "${SESSION}" "${TMUX_CMD[*]}"
  echo "Started overnight run in tmux session: ${SESSION}"
  echo "  attach:  tmux attach -t ${SESSION}"
  echo "  logs:    tail -f ${COMPARISON_ROOT}/<run_id>/00_master/run.log"
  exit 0
fi

if [[ "${SMOKE}" -eq 1 || "${BETA_SMOKE:-0}" == "1" ]]; then
  export BETA_SMOKE=1
  export RPARAM_SMOKE=1
  export BETA_SMOKE_MAX_SITES="${BETA_SMOKE_MAX_SITES:-12}"
  export BETA_SMOKE_N_BURNIN="${BETA_SMOKE_N_BURNIN:-5}"
  export BETA_SMOKE_N_ITER="${BETA_SMOKE_N_ITER:-20}"
  export BETA_SMOKE_N_CHAINS="${BETA_SMOKE_N_CHAINS:-2}"
  echo "BETA_SMOKE=1: subsample=${BETA_SMOKE_MAX_SITES} sites, burnin=${BETA_SMOKE_N_BURNIN}, iter=${BETA_SMOKE_N_ITER}"
fi

if [[ -z "${RUN_ID}" ]]; then
  if [[ "${BETA_SMOKE:-0}" == "1" ]]; then
    RUN_ID="smoke_$(date +%Y%m%d_%H%M%S)"
  else
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
  fi
fi

RUN_ROOT="${COMPARISON_ROOT}/${RUN_ID}"
OUT_MY1="${RUN_ROOT}/01_my1_original"
OUT_PRE="${RUN_ROOT}/02_preprocessed"
OUT_REP="${RUN_ROOT}/03_reparam"
MASTER_DIR="${RUN_ROOT}/00_master"

mkdir -p "${MASTER_DIR}" "${OUT_MY1}/logs" "${OUT_PRE}/logs" "${OUT_REP}/logs"

MASTER_LOG="${MASTER_DIR}/run.log"
exec > >(tee -a "${MASTER_LOG}") 2>&1

write_readme() {
  local path="$1"
  cat > "${path}"
}

write_master_readme() {
  local status="${1:-running}"
  write_readme "${MASTER_DIR}/README.md" <<EOF
# Beta-GLMM comparison run: ${RUN_ID}

**Status:** ${status}
**Started:** $(date -Iseconds)
**Host:** $(hostname)
**Repo:** ${REPO_ROOT}
**Data:** ${DATA_DIR}

## Pipelines (run order)

| Dir | Source | Description |
|-----|--------|-------------|
| \`01_my1_original/\` | \`src/native_r/my_1_run_the_beta_model.Rmd\` | Paper pipeline from \`data.csv\` + shapefiles |
| \`02_preprocessed/\` | \`src/native_r/run_beta_model_from_preprocessed.Rmd\` | Centered model on \`data_for_maps.csv\` |
| \`03_reparam/\` | \`src/native_r/run_beta_model_reparam.R\` | Non-centered reparameterization (no intercept in X) |

Each subdirectory has its own \`README.md\` and \`logs/run.log\`.

## Monitor

\`\`\`bash
tmux attach -t beta_models_overnight   # if launched with --tmux
tail -f ${MASTER_LOG}
\`\`\`

## Compare coefficients

\`\`\`bash
# After all three finish:
ls ${RUN_ROOT}/01_my1_original/beta_est.csv
ls ${RUN_ROOT}/02_preprocessed/beta_est_from_preprocessed.csv
ls ${RUN_ROOT}/03_reparam/beta_est_reparam.csv
\`\`\`

Note: \`01_my1_original\` uses \`beta[2:14]\` (intercept in X); \`03_reparam\` uses \`beta[1:13]\` (no intercept).
EOF
}

write_my1_readme() {
  write_readme "${OUT_MY1}/README.md" <<EOF
# 01 — Original paper pipeline (\`my_1_run_the_beta_model.Rmd\`)

**Source:** \`${NATIVE_R}/my_1_run_the_beta_model.Rmd\`
**Output directory:** \`${OUT_MY1}\`
**Data input:** \`${DATA_DIR}/data.csv\` (+ shapefiles, diversity from \`data_for_maps.csv\` lookup)

## What this runs

- Full R Markdown knit (filtering, corrplot, JAGS, coefficient plot, maps/exports at end)
- **JAGS fit 1:** \`jags.parallel\`, 3 chains, burn-in 4000, iter 15000, thin 10
- **JAGS fit 2 (large):** 6 chains, burn-in 10000, iter 20000, thin 10 — saved as \`saved_parallel_model_large.RData\`

## Key outputs

- \`beta_est.csv\`, \`coefficient_diagnostics/coefficient_posterior_forest.png\`
- \`GLMM_coral_cover.txt\`, \`logs/parallel_large_*.png\`
- \`data_for_maps.csv\` (written here, not to shared \`output/\`)

## Known caveats

- Passes observation-level \`diversity.standardized\` to JAGS (length N); model expects length R.
- Includes intercept in design matrix (\`beta[1]\`); coefficient plot uses \`beta[2:14]\` for slopes.

## Log

\`${OUT_MY1}/logs/run.log\`
EOF
}

write_preprocessed_readme() {
  write_readme "${OUT_PRE}/README.md" <<EOF
# 02 — Preprocessed data pipeline (\`run_beta_model_from_preprocessed.Rmd\`)

**Source:** \`${NATIVE_R}/run_beta_model_from_preprocessed.Rmd\`
**Output directory:** \`${OUT_PRE}\`
**Data input:** \`${DATA_DIR}/data_for_maps.csv\` (override with \`BETA_DATA_PATH\`)

## What this runs

- Centered hierarchical beta-GLMM (same JAGS structure as original, with intercept in X)
- Region-level \`diversity\` vector (length R), dense site/region remapping
- **JAGS:** \`jags\` then \`jags.parallel\`, 3 chains, burn-in 4000, iter 15000, thin 10

## Key outputs

- \`beta_est_from_preprocessed.csv\`, \`Beta_coeff_plot_from_preprocessed.png\`
- \`GLMM_coral_cover_preprocessed.txt\`, \`logs/*.png\`

## Log

\`${OUT_PRE}/logs/run.log\`
EOF
}

write_reparam_readme() {
  write_readme "${OUT_REP}/README.md" <<EOF
# 03 — Reparameterized pipeline (\`run_beta_model_reparam.R\`)

**Source:** \`${NATIVE_R}/run_beta_model_reparam.R\`
**Output directory:** \`${OUT_REP}\`
**Data input:** \`data.csv\` + shapefile pipeline via \`load_model_data_from_pipeline()\`

## What this runs

- No intercept in X (\`~ 0 + ...\`); global baseline via \`mu_global\`
- Non-centered site and ecoregion random effects
- Region-level diversity vector (length R)
- **JAGS:** \`jags.parallel\`, 3 chains, burn-in 10000, iter 20000, thin 10
- Smoke mode if \`BETA_SMOKE=1\` or \`RPARAM_SMOKE=1\`

## Key outputs

- \`beta_est_reparam.csv\`, \`Beta_coeff_plot_reparam.png\`
- \`convergence_diagnostics.csv\`, \`hyperparameter_summary.csv\`
- \`logs/reparam_*.png\`

## Log

\`${OUT_REP}/logs/run.log\`
EOF
}

run_step() {
  local name="$1"
  local out_dir="$2"
  local log_file="${out_dir}/logs/run.log"
  shift 2
  local start_ts end_ts rc
  start_ts="$(date +%s)"
  echo ""
  echo "======================================================================"
  echo "[$(date -Iseconds)] START ${name}"
  echo "  output: ${out_dir}"
  echo "======================================================================"
  set +e
  (
    set -euo pipefail
    export BETA_DATA_DIR="${DATA_DIR}"
    export BETA_OUTPUT_DIR="${out_dir}"
    export BETA_SMOKE="${BETA_SMOKE:-0}"
    export RPARAM_SMOKE="${RPARAM_SMOKE:-0}"
    export BETA_NATIVE_R="${NATIVE_R}"
    "$@"
  ) 2>&1 | tee -a "${log_file}"
  rc=${PIPESTATUS[0]}
  set -e
  end_ts="$(date +%s)"
  local elapsed=$(( end_ts - start_ts ))
  if [[ "${rc}" -eq 0 ]]; then
    echo "[$(date -Iseconds)] OK ${name} (${elapsed}s)"
    echo "${name}: OK (${elapsed}s)" >> "${MASTER_DIR}/status.txt"
  else
    echo "[$(date -Iseconds)] FAIL ${name} exit=${rc} (${elapsed}s)" >&2
    echo "${name}: FAIL exit=${rc} (${elapsed}s)" >> "${MASTER_DIR}/status.txt"
  fi
  return "${rc}"
}

write_master_readme "running"
write_my1_readme
write_preprocessed_readme
write_reparam_readme

echo "Run root: ${RUN_ROOT}"
echo "Master log: ${MASTER_LOG}"
: > "${MASTER_DIR}/status.txt"

FAILURES=0

if should_skip "my1"; then
  echo "Skipping my1 (--skip)"
else
  run_step "my1_original" "${OUT_MY1}" \
    Rscript -e "rmarkdown::render(input='${NATIVE_R}/my_1_run_the_beta_model.Rmd', encoding='UTF-8', quiet=FALSE, clean=identical(Sys.getenv('BETA_SMOKE'), '1'))" \
    || FAILURES=$((FAILURES + 1))
fi

if should_skip "preprocessed"; then
  echo "Skipping preprocessed (--skip)"
else
  run_step "preprocessed" "${OUT_PRE}" \
    Rscript -e "rmarkdown::render(input='${NATIVE_R}/run_beta_model_from_preprocessed.Rmd', encoding='UTF-8', quiet=FALSE, clean=identical(Sys.getenv('BETA_SMOKE'), '1'))" \
    || FAILURES=$((FAILURES + 1))
fi

if should_skip "reparam"; then
  echo "Skipping reparam (--skip)"
else
  run_step "reparam" "${OUT_REP}" \
    Rscript "${NATIVE_R}/run_beta_model_reparam.R" \
    || FAILURES=$((FAILURES + 1))
fi

if [[ "${FAILURES}" -eq 0 ]]; then
  FINAL_STATUS="completed successfully"
else
  FINAL_STATUS="finished with ${FAILURES} failure(s) — see status.txt"
fi

write_master_readme "${FINAL_STATUS}"
echo ""
echo "======================================================================"
echo "Done: ${FINAL_STATUS}"
echo "Run root: ${RUN_ROOT}"
echo "======================================================================"
cat "${MASTER_DIR}/status.txt"

exit "${FAILURES}"
