#!/usr/bin/env bash
# Train, evaluate, and report the RQ1 baseline and combined-augmentation runs.
#
#   scripts/run_baseline_combined.sh /path/to/gtsrb                    # both runs
#   scripts/run_baseline_combined.sh /path/to/gtsrb baseline           # one run
#   scripts/run_baseline_combined.sh /path/to/gtsrb combined
#   scripts/run_baseline_combined.sh /path/to/gtsrb baseline --force   # replace a finished run
#
# /path/to/gtsrb is the Kaggle GTSRB extract (Train/, Test/, Train.csv, Test.csv).
# For each run this trains, scores the validation split under all five
# evaluation.yaml conditions, writes figures, and copies the run to shared/<name>/.
# When both runs exist it also writes the RQ1 comparison tables.
#
# Existing results are never replaced without --force. With --force, the old
# artifacts/<name> folder is moved to artifacts/<name>.backup-<timestamp> first.
set -euo pipefail

usage="Usage: $0 /path/to/gtsrb [baseline|combined|all] [--force]"
DATA_ROOT=""
WHICH="all"
FORCE=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    baseline|combined|all) WHICH="$arg" ;;
    -h|--help) echo "$usage"; exit 0 ;;
    -*) echo "error: unknown option $arg" >&2; echo "$usage" >&2; exit 1 ;;
    *)
      if [[ -z "$DATA_ROOT" ]]; then
        DATA_ROOT="$arg"
      else
        echo "error: unexpected argument '$arg'. Choose baseline, combined, or all." >&2
        echo "$usage" >&2
        exit 1
      fi
      ;;
  esac
done
if [[ -z "$DATA_ROOT" ]]; then
  echo "$usage" >&2
  exit 1
fi

case "$WHICH" in
  baseline) RUNS=(baseline) ;;
  combined) RUNS=(combined) ;;
  all) RUNS=(baseline combined) ;;
esac
CONDITIONS=(clean rotation zoom brightness lowres)

config_for() {
  case "$1" in
    baseline) echo baseline.yaml ;;
    combined) echo configs/experiments/combined.yaml ;;
  esac
}

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
if [[ -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
fi

# Keep a Mac awake for the whole run (each run takes over an hour on CPU).
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1
  forward=("$DATA_ROOT" "$WHICH")
  [[ "$FORCE" == 1 ]] && forward+=(--force)
  exec caffeinate -i "$SCRIPT" "${forward[@]}"
fi

cd "$(dirname "$SCRIPT")/.."

# --- Checks that need nothing installed -------------------------------------

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

# Refuse to replace finished results unless --force. Every requested run is
# checked here, so "all" cannot train baseline and then stop at combined.
existing=()
for name in "${RUNS[@]}"; do
  [[ -e "artifacts/$name" ]] && existing+=("artifacts/$name")
  [[ -e "shared/$name" ]] && existing+=("shared/$name")
done
if (( ${#existing[@]} > 0 )) && [[ "$FORCE" != 1 ]]; then
  echo "error: these results already exist and would be replaced:" >&2
  printf '         %s\n' "${existing[@]}" >&2
  echo "       Re-run with --force to replace them. The old artifacts/ folder is kept as a backup." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain -- src scripts configs baseline.yaml evaluation.yaml)" ]]; then
  echo "warning: uncommitted changes in code or configs; metadata.json will record git_dirty: true." >&2
  echo "         Commit first so teammates can reproduce the run." >&2
  sleep 5
fi

# --- Environment and config compatibility -----------------------------------

uv sync

# Only `experiment` and `augmentation` may differ between the two configs, the
# baseline must have no augmentation, and combined must enable all three.
uv run python - baseline.yaml configs/experiments/combined.yaml <<'PY'
import sys
from traffic_signs.utils.config import ConfigError, load_config

try:
    base, comb = (load_config(p) for p in sys.argv[1:3])
except ConfigError as exc:
    sys.exit(f"error: {exc}")

problems = []
for key in base:
    if key in ("experiment", "augmentation"):
        continue
    if base[key] != comb[key]:
        problems.append(f"{key}: baseline {base[key]!r} vs combined {comb[key]!r}")
if base["experiment"] != "baseline":
    problems.append(f"baseline.yaml experiment is {base['experiment']!r}, expected 'baseline'")
if comb["experiment"] != "combined":
    problems.append(f"combined.yaml experiment is {comb['experiment']!r}, expected 'combined'")
if any(v != 0 for v in base["augmentation"].values()):
    problems.append(f"baseline augmentation must be all 0, got {base['augmentation']}")
off = [k for k, v in comb["augmentation"].items() if v == 0]
if off:
    problems.append(f"combined augmentation leaves {', '.join(off)} at 0")

if problems:
    print("error: baseline.yaml and combined.yaml are not comparable:", file=sys.stderr)
    for p in problems:
        print(f"         {p}", file=sys.stderr)
    sys.exit(1)
print("Configs are comparable: only experiment and augmentation differ.")
PY

# --- Runs -------------------------------------------------------------------

run_one() {
  local name="$1" config
  config="$(config_for "$name")"

  if [[ -e "artifacts/$name" ]]; then
    local backup="artifacts/$name.backup-$(date +%Y%m%d_%H%M%S)"
    echo "==> [$name] moving old results to $backup"
    mv "artifacts/$name" "$backup"
  fi

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

  # shared/<name> is only replaced after the new run has fully succeeded.
  echo "==> [$name] copying to shared/$name"
  rm -rf "shared/$name"
  cp -R "artifacts/$name" "shared/$name"
}

for name in "${RUNS[@]}"; do
  run_one "$name"
done

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