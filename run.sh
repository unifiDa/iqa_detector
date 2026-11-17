#!/usr/bin/env bash
# run.sh - runs the evaluation pipeline on ONE dataset, described by a config file.
#
#   ./run.sh <dataset>                  all the steps the config enables, in pipeline order
#   ./run.sh <dataset> quality prompt   only the listed steps (always in pipeline order)
#   ./run.sh <dataset> --list           show the steps this config enables and exit
#
# <dataset> is the name of datasets/<dataset>.env, or the path of any .env file.
# To add a dataset copy datasets/template.env and fill it in; no code changes are needed.
#
# Steps: prepare, prepare_orig, coherence, padding, quality, prompts, prompt, mps, dataset, age, merge, filter
# A step only exists if the config provides what it needs (e.g. `dataset` needs ORIGINALS).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
[ "$#" -ge 1 ] || { sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2; }
case "$1" in
  */*|*.env) CONF=$(realpath "$1") ;;
  *)         CONF=$HERE/datasets/$1.env ;;
esac
[ -f "$CONF" ] || { echo "config not found: $CONF" >&2; exit 1; }
shift
cd "$HERE"

# ---- defaults (every one can be overridden in the config) -------------------------------------
NAME=$(basename "$CONF" .env)   # dataset name, stored in summary.json
INPAINTS=""                     # REQUIRED: folder with the images to evaluate
OUT=""                          # REQUIRED: results folder
MASKS=""                        # folder with the masks (optional)
PREPARE=faces                   # faces = prepare.py (insightface); generic = prepare_generic.py (no faces)
PREPARE_ARGS=()                 # extra arguments for the prepare step, e.g. (--mask-regex '...' --mask-template '...')
ORIGINALS=""                    # folder with the original images: enables prepare_orig and dataset (FID/KID)
OUT_ORIG=""                     # results of the originals (default: <OUT>-original/)
REF_DIR=""                      # FID/KID reference crops (default: <OUT_ORIG>/crops/face)
DATASET_ARGS=()                 # extra arguments for the dataset step, e.g. (--kind mask)
PROMPT_JSON=""                  # folder of prompt JSON files: enables the prompts step (make_prompts.py)
PROMPT_JSON_ARGS=()             # extra arguments for make_prompts.py, e.g. (--id-template '{name}_{i}_inpainted')
PROMPTS=""                      # prompts file (jsonl/csv/tsv): enables prompt (CLIP); default <OUT>/prompts.jsonl with PROMPT_JSON
PROMPT_COLS=()                  # if the prompts file is not {record_id, prompt}: (--id-key ID --prompt-key TEXT)
QUALITY_METRICS=()              # quality metrics (default: all of b_quality.py)
MPS_CKPT=""                     # MPS checkpoint: with PROMPTS enables the mps step
MPS_REPO=""                     # cloned Kwai-Kolors/MPS repo (the folder that contains trainer/)
AGE=0                           # 1 = enable the age step (e_age.py)
AGE_ARGS=()                     # extra arguments for e_age.py, e.g. (--crops)
FILTER=0                        # 1 = enable the filter step (colour images without minors; needs AGE=1)
BY_FRAME=1                      # filter: a minor discards the whole frame
FRAME_REGEX=""                  # filter: how to recognise a frame in the file name (default frame_<number>)
FILTER_ARGS=()                  # extra arguments for filter_images.py, e.g. (--max-color-dev 0.02)
ENV_NAME=gpu                    # pixi environment for the torch steps: gpu | cpu
DEVICE=""                       # default cuda:0 (cpu with ENV_NAME=cpu)

# shellcheck disable=SC1090
source "$CONF"

[ -n "$INPAINTS" ] || { echo "$CONF: INPAINTS is not set" >&2; exit 1; }
[ -n "$OUT" ] || { echo "$CONF: OUT is not set" >&2; exit 1; }
[ -n "$DEVICE" ] || { [ "$ENV_NAME" = cpu ] && DEVICE=cpu || DEVICE=cuda:0; }
MPS_ENV=mps; [ "$ENV_NAME" = cpu ] && MPS_ENV=mps-cpu
[ -n "$OUT_ORIG" ] || OUT_ORIG=${OUT%/}-original/
[ -n "$REF_DIR" ] || REF_DIR=${OUT_ORIG%/}/crops/face
[ -z "$PROMPT_JSON" ] || [ -n "$PROMPTS" ] || PROMPTS=${OUT%/}/prompts.jsonl
if [ "$FILTER" = 1 ] && [ "$AGE" != 1 ]; then echo "$CONF: FILTER=1 needs AGE=1" >&2; exit 1; fi

# Expands an array that may be empty (safe with `set -u` on old bash): $(x ARR) is not usable, so every use
# is written ${ARR[@]+"${ARR[@]}"}.

# ---- steps ------------------------------------------------------------------------------------
# step 0 on the images to evaluate: mask / face -> index.csv and crops
step_prepare() {
  local base=(--images "$INPAINTS" --out "$OUT"); [ -z "$MASKS" ] || base+=(--masks "$MASKS")
  local task=prepare; [ "$PREPARE" = generic ] && task=prepare_generic
  pixi run -e "$ENV_NAME" "$task" "${base[@]}" ${PREPARE_ARGS[@]+"${PREPARE_ARGS[@]}"}
}
# step 0 on the originals (reference for FID/KID)
step_prepare_orig() {
  local task=prepare; [ "$PREPARE" = generic ] && task=prepare_generic
  pixi run -e "$ENV_NAME" "$task" --images "$ORIGINALS" --out "$OUT_ORIG"
}
# A. region/surroundings coherence (uses the image and the mask, not the crops)
step_coherence() { pixi run coherence --out "$OUT"; }
# replicated pixels in the frame (flag "edge padding") and crops outside the frame (must be 0)
step_padding() { pixi run padding --out "$OUT"; }
# B. no-reference quality
step_quality() {
  local m=(); [ "${#QUALITY_METRICS[@]}" -eq 0 ] || m=(--metrics "${QUALITY_METRICS[@]}")
  pixi run -e "$ENV_NAME" quality --device "$DEVICE" ${m[@]+"${m[@]}"} --out "$OUT"
}
# prompt JSON files -> prompts.jsonl
step_prompts() {
  mkdir -p "$OUT"
  pixi run make_prompts --json-dir "$PROMPT_JSON" --out "$PROMPTS" --images "$INPAINTS" ${PROMPT_JSON_ARGS[@]+"${PROMPT_JSON_ARGS[@]}"}
}
# C. CLIP score
step_prompt() {
  pixi run -e "$ENV_NAME" prompt --device "$DEVICE" --out "$OUT" --prompts "$PROMPTS" ${PROMPT_COLS[@]+"${PROMPT_COLS[@]}"}
}
# C. MPS score (dedicated environment, transformers 4.27.3); completes c_prompt.csv without redoing CLIP
step_mps() {
  pixi run -e "$MPS_ENV" python eval_suite/c_prompt.py --out "$OUT" --prompts "$PROMPTS" \
    --metrics mps --mps-ckpt "$MPS_CKPT" --mps-repo "$MPS_REPO" ${PROMPT_COLS[@]+"${PROMPT_COLS[@]}"}
}
# D. FID / KID against the crops of the originals
step_dataset() {
  pixi run -e "$ENV_NAME" dataset --device "$DEVICE" --out "$OUT" --ref "$REF_DIR" ${DATASET_ARGS[@]+"${DATASET_ARGS[@]}"}
}
# E. age of the faces (one row per face -> e_age.csv)
step_age() {
  pixi run -e "$ENV_NAME" age --device "$DEVICE" --out "$OUT" ${AGE_ARGS[@]+"${AGE_ARGS[@]}"}
}
# merges everything into results.csv and summary.json
step_merge() { pixi run merge --out "$OUT" --name "$NAME"; }
# keeps the colour images without minors -> $OUT/filtered/
step_filter() {
  local f=(); [ "$BY_FRAME" != 1 ] || f+=(--by-frame); [ -z "$FRAME_REGEX" ] || f+=(--frame-regex "$FRAME_REGEX")
  pixi run filter --out "$OUT" ${f[@]+"${f[@]}"} ${FILTER_ARGS[@]+"${FILTER_ARGS[@]}"}
}

# ---- which steps this config enables -----------------------------------------------------------
ALL=(prepare)
[ -z "$ORIGINALS" ] || ALL+=(prepare_orig)
ALL+=(coherence padding quality)
[ -z "$PROMPT_JSON" ] || ALL+=(prompts)
[ -z "$PROMPTS" ] || ALL+=(prompt)
{ [ -z "$PROMPTS" ] || [ -z "$MPS_CKPT" ]; } || ALL+=(mps)
[ -z "$ORIGINALS" ] || ALL+=(dataset)
[ "$AGE" != 1 ] || ALL+=(age)
ALL+=(merge)
[ "$FILTER" != 1 ] || ALL+=(filter)

if [ "${1:-}" = "--list" ]; then echo "$NAME: ${ALL[*]}"; exit 0; fi

if [ "$#" -eq 0 ]; then
  WANTED=("${ALL[@]}")
else
  for s in "$@"; do
    case " ${ALL[*]} " in *" $s "*) ;; *) echo "step not available for $NAME: $s (enabled by $CONF: ${ALL[*]})" >&2; exit 2 ;; esac
  done
  WANTED=("$@")
fi

# ---- checks ------------------------------------------------------------------------------------
need() { [ -e "$1" ] || { echo "not found: $1  (check the paths in $CONF)" >&2; exit 1; }; }
has() { case " ${WANTED[*]} " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
if has prepare; then need "$INPAINTS"; [ -z "$MASKS" ] || need "$MASKS"; fi
if has prepare_orig; then need "$ORIGINALS"; fi
if has prompts; then need "$PROMPT_JSON"; need "$INPAINTS"; fi
if has mps; then need "$MPS_CKPT"; need "$MPS_REPO"; fi

# ---- run ---------------------------------------------------------------------------------------
echo "Dataset $NAME ($CONF) -> $OUT"
for s in "${ALL[@]}"; do
  if has "$s"; then
    echo; echo "==================== $s ===================="
    "step_$s"
  fi
done
echo; echo "Done: ${WANTED[*]}"
