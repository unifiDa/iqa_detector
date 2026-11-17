#!/usr/bin/env bash
# lamarr.sh - full pipeline (with faces) on the fooocus-cinematic dataset.
#
#   ./lamarr.sh                       all the steps, in order
#   ./lamarr.sh quality prompt        only the listed steps (always in pipeline order)
#
# Steps: prepare, prepare_orig, coherence, padding, quality, prompt, mps, dataset, age, merge, filter
# Keep it in the project root (the folder that contains pixi.toml and eval_suite/).
set -euo pipefail
cd "$(dirname "$0")"

# ---- paths ------------------------------------------------------------------------------------
BASE=/Scratch/Shullani/Dataset
INPAINTS=$BASE/fooocus-cinematic/inpaints/
MASKS=$BASE/fooocus-cinematic/masks/
ORIGINALS=$BASE/fooocus-cinematic/original/
PROMPTS=$BASE/fooocus-cinematic/prompts.jsonl
OUT=$BASE/results/
OUT_ORIG=$BASE/results-original/

MPS_ROOT=/data/staff/shullani/Documents/Research/IAPP/DETECTOR/iqa_detector
MPS_CKPT=$MPS_ROOT/Weights/MPS_overall_checkpoint.pth
MPS_REPO=$MPS_ROOT/MPS/

DEVICE=cuda:0

# ---- steps ------------------------------------------------------------------------------------
# prepare with faces, on the inpainted images
step_prepare() {
  pixi run -e gpu prepare --images "$INPAINTS" --out "$OUT" --masks "$MASKS"
}

# prepare on the originals (reference for FID/KID)
step_prepare_orig() {
  pixi run -e gpu prepare --images "$ORIGINALS" --out "$OUT_ORIG"
}

# A. region/surroundings coherence: only uses the original image and the mask, not the crops
step_coherence() {
  pixi run coherence --out "$OUT"
}

# replicated pixels in the frame (flag "edge padding") and crops outside the frame (must be 0)
step_padding() {
  pixi run padding --out "$OUT"
}

# B. topiq_nr-face, musiq, clipiqa, ...
step_quality() {
  pixi run -e gpu quality --device "$DEVICE" --out "$OUT"
}

# C. CLIP score
step_prompt() {
  pixi run -e gpu prompt --device "$DEVICE" --out "$OUT" --prompts "$PROMPTS"
}

# C. MPS score (dedicated environment, transformers 4.27.3); completes c_prompt.csv without redoing CLIP
step_mps() {
  pixi run -e mps mps "$OUT" "$PROMPTS" "$MPS_CKPT" "$MPS_REPO"
}

# D. FID / KID against the faces of the originals
step_dataset() {
  pixi run -e gpu dataset --device "$DEVICE" --out "$OUT" --ref "$OUT_ORIG/crops/face"
}

# E. age estimation of the faces, on the full frames (one row per face -> e_age.csv)
step_age() {
  pixi run -e gpu age --device "$DEVICE" --out "$OUT"
}

# merges everything into results.csv and summary.json
step_merge() {
  pixi run merge --out "$OUT"
}

# keeps the colour images without minors (a minor discards the whole frame) -> $OUT/filtered/
step_filter() {
  pixi run filter --out "$OUT" --by-frame
}

# ---- run --------------------------------------------------------------------------------------
ALL=(prepare prepare_orig coherence padding quality prompt mps dataset age merge filter)
if [ "$#" -eq 0 ]; then
  WANTED=("${ALL[@]}")
else
  for s in "$@"; do
    case " ${ALL[*]} " in *" $s "*) ;; *) echo "unknown step: $s (valid: ${ALL[*]})" >&2; exit 2 ;; esac
  done
  WANTED=("$@")
fi

for s in "${ALL[@]}"; do
  case " ${WANTED[*]} " in
    *" $s "*)
      echo; echo "==================== $s ===================="
      "step_$s"
      ;;
  esac
done
echo; echo "Done: ${WANTED[*]}"
