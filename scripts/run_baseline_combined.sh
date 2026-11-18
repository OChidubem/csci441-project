#!/usr/bin/env bash
# Train, evaluate, and report the RQ1 baseline and combined-augmentation runs.
#
#   scripts/run_baseline_combined.sh /path/to/gtsrb            # both runs
#   scripts/run_baseline_combined.sh /path/to/gtsrb baseline   # one run
#   scripts/run_baseline_combined.sh /path/to/gtsrb combined
#
# /path/to/gtsrb is the Kaggle GTSRB extract (Train/, Test/, Train.csv, Test.csv).
# For each run this trains, scores the validation split under all five
# evaluation.yaml conditions, writes figures, and copies the run to shared/<name>/.
# When both runs exist it also writes the RQ1 comparison tables.
set -euo pipefail

DATA_ROOT="${1:?Usage: $0 /path/to/gtsrb [baseline|combined|all]}"
WHICH="${2:-all}"
CONDITIONS=(clean rotation zoom brightness lowres)

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
if [[ -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
fi

# Keep a Mac awake for the whole run (each run takes over an hour on CPU).
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1
  exec caffeinate -i "$SCRIPT" "$DATA_ROOT" "$WHICH"
fi

cd "$(dirname "$SCRIPT")/.."
for required in Train Test; do
  if [[ ! -d "$DATA_ROOT/$required" ]]; then
    echo "error: $DATA_ROOT/$required not found. Point this at the Kaggle GTSRB extract." >&2
    exit 1
  fi
done

expected_hash="e32ea0680262e73b1c733bc90b7c90c5c343d71dd038adaff9aaec6880fe98d7"
actual_hash="$(shasum -a 256 configs/splits/gtsrb_v1.json | cut -d' ' -f1)"
if [[ "$actual_hash" != "$expected_hash" ]]; then
  echo "error: configs/splits/gtsrb_v1.json does not match the team manifest." >&2
  echo "       Restore it with: git checkout -- configs/splits/gtsrb_v1.json" >&2
  exit 1
fi

if [[ -n "$(git status --porcelain -- src scripts configs baseline.yaml evaluation.yaml)" ]]; then
  echo "warning: uncommitted changes in code or configs; metadata.json will record git_dirty: true." >&2
  echo "         Commit first so teammates can reproduce the run." >&2
  sleep 5
fi

uv sync

run_one() {
  local name="$1" config="$2"
  echo "==> [$name] training ($config)"
  uv run python scripts/train.py --config "$config" --data-root "$DATA_ROOT"

  echo "==> [$name] evaluating: ${CONDITIONS[*]}"
  uv run python scripts/evaluate.py \
    --checkpoint "artifacts/$name/model.keras" \
    --split validation \
    --conditions "${CONDITIONS[@]}" \
    --data-root "$DATA_ROOT"

  echo "==> [$name] figures"
  uv run python scripts/report.py run --run "artifacts/$name" --data-root "$DATA_ROOT"

  echo "==> [$name] copying to shared/$name"
  rm -rf "shared/$name"
  cp -R "artifacts/$name" "shared/$name"
}

case "$WHICH" in
  baseline) run_one baseline baseline.yaml ;;
  combined) run_one combined configs/experiments/combined.yaml ;;
  all)
    run_one baseline baseline.yaml
    run_one combined configs/experiments/combined.yaml
    ;;
  *) echo "error: second argument must be baseline, combined, or all" >&2; exit 1 ;;
esac

if [[ -f artifacts/baseline/model.keras && -f artifacts/combined/model.keras ]]; then
  echo "==> RQ1 comparison: baseline vs combined"
  uv run python scripts/report.py compare --type augmentation \
    --runs artifacts/baseline artifacts/combined
  uv run python scripts/report.py compare --type augmentation \
    --runs artifacts/baseline artifacts/combined \
    --conditions "${CONDITIONS[@]}"
  echo "Tables and plots are under artifacts/comparisons/augmentation/."
fi

echo "Done. Commit shared/baseline and/or shared/combined when you are happy with them."
