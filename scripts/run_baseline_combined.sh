#!/usr/bin/env bash
# Train, evaluate, and report the RQ1 baseline and combined-augmentation runs.
#
#   bash scripts/run_baseline_combined.sh /path/to/gtsrb
#   bash scripts/run_baseline_combined.sh /path/to/gtsrb baseline
#   bash scripts/run_baseline_combined.sh /path/to/gtsrb combined
#   bash scripts/run_baseline_combined.sh /path/to/gtsrb all --force
#
# Existing experiment outputs are never replaced without --force. With --force,
# the old artifacts/<name> is backed up before training begins. A shared run is
# only replaced after training, evaluation, and reporting all succeed.
set -euo pipefail

usage="Usage: $0 /path/to/gtsrb [baseline|combined|all] [--force]"
DATA_ROOT=""
WHICH="all"
FORCE=0
selected_mode=0
for arg in "$@"; do
  case "$arg" in
    --force) FORCE=1 ;;
    baseline|combined|all)
      if (( selected_mode )); then
        echo "error: specify only one run selection" >&2
        echo "$usage" >&2
        exit 1
      fi
      WHICH="$arg"
      selected_mode=1
      ;;
    -h|--help) echo "$usage"; exit 0 ;;
    -*) echo "error: unknown option '$arg'" >&2; echo "$usage" >&2; exit 1 ;;
    *)
      if [[ -z "$DATA_ROOT" ]]; then
        DATA_ROOT="$arg"
      else
        echo "error: unexpected argument '$arg'" >&2
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
    baseline) printf '%s\n' baseline.yaml ;;
    combined) printf '%s\n' configs/experiments/combined.yaml ;;
  esac
}

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
DATA_ROOT="${DATA_ROOT/#\~/$HOME}"
if [[ -d "$DATA_ROOT" ]]; then
  DATA_ROOT="$(cd "$DATA_ROOT" && pwd)"
fi

# Prevent macOS from sleeping. Invoke bash explicitly: executable permission
# on the script is not required.
if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]] && command -v caffeinate >/dev/null; then
  export CAFFEINATED=1
  forward=("$DATA_ROOT" "$WHICH")
  (( FORCE )) && forward+=(--force)
  exec caffeinate -i bash "$SCRIPT" "${forward[@]}"
fi

cd "$(dirname "$SCRIPT")/.."

for required in Train Test; do
  if [[ ! -d "$DATA_ROOT/$required" ]]; then
    echo "error: $DATA_ROOT/$required not found. Point this at the Kaggle GTSRB extract." >&2
    exit 1
  fi
done

manifest=configs/splits/gtsrb_v1.json
if [[ ! -f "$manifest" ]]; then
  echo "error: missing shared split manifest: $manifest" >&2
  exit 1
fi
expected_hash="e32ea0680262e73b1c733bc90b7c90c5c343d71dd038adaff9aaec6880fe98d7"
actual_hash="$(shasum -a 256 "$manifest" | cut -d ' ' -f1)"
if [[ "$actual_hash" != "$expected_hash" ]]; then
  echo "error: $manifest does not match the team manifest." >&2
  echo "       Restore the agreed file before training." >&2
  exit 1
fi

# Avoid concurrent invocations replacing the same results. An atomic mkdir is
# sufficient for team members running the script within one working tree.
mkdir -p artifacts shared
LOCK_DIR=artifacts/.run_baseline_combined.lock
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "error: another runner may be active (lock: $LOCK_DIR)." >&2
  echo "       Check for an active run before manually removing a stale lock." >&2
  exit 1
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

# Display saved completion status without requiring uv or project dependencies.
# If python3 is unavailable, report that status cannot be read.
show_status() {
  local dir="$1"
  if [[ ! -f "$dir/metadata.json" ]]; then
    printf '%s: no metadata.json (possibly incomplete)\n' "$dir"
  elif command -v python3 >/dev/null 2>&1; then
    python3 - "$dir/metadata.json" <<'PYTHON_STATUS'
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as handle:
        status = json.load(handle).get("status", "unknown")
except (OSError, ValueError, TypeError):
    status = "unreadable metadata"
print(f"{sys.argv[1].removesuffix('/metadata.json')}: {status}")
PYTHON_STATUS
  else
    printf '%s: metadata.json present (python3 unavailable to read status)\n' "$dir"
  fi
}

existing=()
for name in "${RUNS[@]}"; do
  [[ -e "artifacts/$name" ]] && existing+=("artifacts/$name")
  [[ -e "shared/$name" ]] && existing+=("shared/$name")
done
if (( ${#existing[@]} > 0 )) && (( ! FORCE )); then
  echo "error: existing results would be replaced:" >&2
  for dir in "${existing[@]}"; do
    show_status "$dir" >&2
  done
  echo "       Use --force to replace them; local artifacts are backed up." >&2
  exit 1
fi

if [[ -n "$(git status --porcelain -- src scripts configs baseline.yaml evaluation.yaml)" ]]; then
  echo "warning: uncommitted changes in code or configs; metadata will record git_dirty: true." >&2
  echo "         Commit first if teammates need an exact reproduction." >&2
fi

uv sync

# Validate configurations before modifying any experiment outputs.
uv run python - baseline.yaml configs/experiments/combined.yaml <<'PY'
import sys
from traffic_signs.utils.config import load_config
try:
    base, comb = (load_config(p) for p in sys.argv[1:3])
except Exception as exc:
    sys.exit(f"error: could not load experiment configs: {exc}")

problems = []
for key in sorted(set(base) | set(comb)):
    if key in ("experiment", "augmentation"):
        continue
    if base.get(key) != comb.get(key):
        problems.append(f"{key}: baseline {base.get(key)!r} vs combined {comb.get(key)!r}")
if base.get("experiment") != "baseline":
    problems.append("baseline.yaml must have experiment: baseline")
if comb.get("experiment") != "combined":
    problems.append("combined.yaml must have experiment: combined")
if any(value != 0 for value in base.get("augmentation", {}).values()):
    problems.append("baseline augmentation must be entirely disabled")
expected_aug = {"rotation_degrees", "brightness", "zoom"}
if set(comb.get("augmentation", {})) != expected_aug:
    problems.append(f"combined augmentation must specify {sorted(expected_aug)}")
for field in sorted(expected_aug):
    if comb.get("augmentation", {}).get(field, 0) <= 0:
        problems.append(f"combined augmentation must enable {field}")
if problems:
    print("error: configurations are not comparable:", file=sys.stderr)
    for issue in problems:
        print(f"  {issue}", file=sys.stderr)
    sys.exit(1)
print("Configs comparable: only experiment name and augmentation differ.")
PY

# The training pipeline opens files using manifest paths; its CSV annotations
# are consumed during manifest preparation, not training/evaluation.
uv run python - "$manifest" "$DATA_ROOT" <<'PY'
import json, sys
from pathlib import Path
manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
root = Path(sys.argv[2])
missing = [
    sample["id"]
    for split in ("train", "validation")
    for sample in manifest["samples"][split]
    if not (root / sample["id"]).is_file()
]
if missing:
    print(f"error: {len(missing)} manifest images are missing in {root}", file=sys.stderr)
    for sample in missing[:5]:
        print(f"       {sample}", file=sys.stderr)
    sys.exit(1)
count = sum(len(manifest["samples"][s]) for s in ("train", "validation"))
print(f"Verified {count} train/validation images.")
PY

# Unique, non-colliding backup destination. The candidate is reserved with
# mktemp before being used as a renamed directory.
backup_existing() {
  local src="$1" backup
  [[ -e "$src" ]] || return 0
  backup="$(mktemp -d "${src}.backup-$(date +%Y%m%d_%H%M%S)-XXXXXX")"
  rmdir "$backup"
  echo "==> Backing up $src to $backup"
  mv "$src" "$backup"
}

# Preserve the previous shared run until a complete new copy exists. The old
# copy is kept under artifacts/ (ignored by git), not shared/, so it can never
# be committed by accident. A stale staging folder is not deleted.
publish_shared() {
  local name="$1" staging previous=""
  staging="$(mktemp -d "shared/.${name}.tmp.XXXXXX")"
  cp -R "artifacts/$name/." "$staging/"
  if [[ -e "shared/$name" ]]; then
    previous="$(mktemp -d "artifacts/shared-${name}.previous-$(date +%Y%m%d_%H%M%S)-XXXXXX")"
    rmdir "$previous"
    mv "shared/$name" "$previous"
  fi
  if ! mv "$staging" "shared/$name"; then
    if [[ -n "$previous" && -e "$previous" && ! -e "shared/$name" ]]; then
      mv "$previous" "shared/$name"
    fi
    echo "error: could not publish shared/$name (staging: $staging)" >&2
    return 1
  fi
  if [[ -n "$previous" ]]; then
    echo "==> Previous shared results retained at $previous"
  fi
}

run_one() {
  local name="$1" config
  config="$(config_for "$name")"
  if [[ -e "artifacts/$name" ]]; then
    backup_existing "artifacts/$name"
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

  echo "==> [$name] publishing finished run to shared/$name"
  publish_shared "$name"
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
  echo "Tables and figures are under artifacts/comparisons/augmentation/."
fi

echo "Done. Review shared/baseline and/or shared/combined before committing."