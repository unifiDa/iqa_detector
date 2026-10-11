#!/usr/bin/env bash
# lamarr.sh - pipeline completa (con volti) sul dataset fooocus-cinematic.
#
#   ./lamarr.sh                       tutti i passi, nell'ordine
#   ./lamarr.sh quality prompt        solo i passi indicati (sempre nell'ordine della pipeline)
#
# Passi: prepare, prepare_orig, coherence, quality, prompt, mps, dataset, merge
# Va messo nella radice del progetto (la cartella che contiene pixi.toml e eval_suite/).
set -euo pipefail
cd "$(dirname "$0")"

# ---- percorsi ---------------------------------------------------------------------------------
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

# ---- passi ------------------------------------------------------------------------------------
# prepare con volti, sulle immagini inpaintate
step_prepare() {
  pixi run -e gpu prepare --images "$INPAINTS" --out "$OUT" --masks "$MASKS"
}

# prepare sugli originali (riferimento per FID/KID)
step_prepare_orig() {
  pixi run -e gpu prepare --images "$ORIGINALS" --out "$OUT_ORIG"
}

# A. coerenza regione/contorno: usa solo l'immagine originale e la maschera, non i ritagli
step_coherence() {
  pixi run coherence --out "$OUT"
}

# B. topiq_nr-face, musiq, clipiqa, ...
step_quality() {
  pixi run -e gpu quality --device "$DEVICE" --out "$OUT"
}

# C. CLIP score
step_prompt() {
  pixi run -e gpu prompt --device "$DEVICE" --out "$OUT" --prompts "$PROMPTS"
}

# C. MPS score (ambiente dedicato, transformers 4.27.3); completa c_prompt.csv senza rifare CLIP
step_mps() {
  pixi run -e mps mps "$OUT" "$PROMPTS" "$MPS_CKPT" "$MPS_REPO"
}

# D. FID / KID contro i volti degli originali
step_dataset() {
  pixi run -e gpu dataset --device "$DEVICE" --out "$OUT" --ref "$OUT_ORIG/crops/face"
}

# unisce tutto in results.csv e summary.json
step_merge() {
  pixi run merge --out "$OUT"
}

# ---- esecuzione -------------------------------------------------------------------------------
ALL=(prepare prepare_orig coherence quality prompt mps dataset merge)
if [ "$#" -eq 0 ]; then
  WANTED=("${ALL[@]}")
else
  for s in "$@"; do
    case " ${ALL[*]} " in *" $s "*) ;; *) echo "passo sconosciuto: $s (validi: ${ALL[*]})" >&2; exit 2 ;; esac
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
echo; echo "Finito: ${WANTED[*]}"