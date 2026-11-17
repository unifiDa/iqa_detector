#!/usr/bin/env bash
# flux.sh - evaluation pipeline on the btb-dresden dataset inpainted with FLUX.1-Fill.
#
#   ./flux.sh                         all the steps, in order
#   ./flux.sh prompt mps merge        only the listed steps (always in pipeline order)
#
# Steps: prepare, coherence, padding, quality, prompts, prompt, mps, merge
# Keep it in the project root (the folder that contains pixi.toml and eval_suite/).
set -euo pipefail
cd "$(dirname "$0")"

# ---- paths ---------------------------------------------------------------------------------
NAME=flux
DATA=/home/dasara/Research/btb-dresden-all
INPAINTS=$DATA/flux_inpaint/
MASKS=$DATA/btb_masks/
PROMPT_JSON=$DATA/miniInternVL_btb_prompts/
OUT=/home/dasara/Research/results/$NAME/
PROMPTS=$OUT/prompts.jsonl

CODE=/home/dasara/Research/code/iqa_detector
MPS_CKPT=$CODE/Weights/MPS_overall_checkpoint.pth
MPS_REPO=$CODE/MPS/

DEVICE=cuda:0
QUALITY_METRICS=(musiq clipiqa brisque niqe maniqa)

# ---- steps ------------------------------------------------------------------------------------
# crops on the mask (scheme mask_<image>_<large|medium|small>.png)
step_prepare() {
    pixi run -e gpu prepare_generic --images "$INPAINTS" --out "$OUT" --masks "$MASKS"
}

# A. region/surroundings coherence
step_coherence() {
    pixi run coherence --out "$OUT"
}

# replicated pixels in the frame and crops outside the frame (must be 0)
step_padding() {
    pixi run padding --out "$OUT"
}

# B. no-reference quality
step_quality() {
    pixi run -e gpu quality --device "$DEVICE" --metrics "${QUALITY_METRICS[@]}" --out "$OUT"
}

# prompt JSON -> prompts.jsonl (id = inpainted file name without extension)
step_prompts() {
    mkdir -p "$OUT"
    pixi run make_prompts --json-dir "$PROMPT_JSON" --out "$PROMPTS" --images "$INPAINTS"
}

# C. CLIP score
step_prompt() {
    pixi run -e gpu prompt --device "$DEVICE" --out "$OUT" --prompts "$PROMPTS"
}

# C. MPS score (dedicated environment, transformers 4.27.3); completes c_prompt.csv without redoing CLIP
step_mps() {
    pixi run -e mps mps "$OUT" "$PROMPTS" "$MPS_CKPT" "$MPS_REPO"
}

# merges everything into results.csv and summary.json
step_merge() {
    pixi run merge --out "$OUT" --name "$NAME"
}

# ---- checks --------------------------------------------------------------------------------
need() { [ -e "$1" ] || { echo "not found: $1  (check the paths at the top of $0)" >&2; exit 1; }; }

# ---- run -------------------------------------------------------------------------------
ALL=(prepare coherence padding quality prompts prompt mps merge)
if [ "$#" -eq 0 ]; then
    WANTED=("${ALL[@]}")
else
    for s in "$@"; do
        case " ${ALL[*]} " in *" $s "*) ;; *) echo "unknown step: $s (valid: ${ALL[*]})" >&2; exit 2 ;; esac
    done
    WANTED=("$@")
fi

case " ${WANTED[*]} " in *" prepare "*) need "$INPAINTS"; need "$MASKS" ;; esac
case " ${WANTED[*]} " in *" prompts "*) need "$PROMPT_JSON"; need "$INPAINTS" ;; esac
case " ${WANTED[*]} " in *" mps "*) need "$MPS_CKPT"; need "$MPS_REPO" ;; esac

echo "Dataset $NAME -> $OUT"
for s in "${ALL[@]}"; do
    case " ${WANTED[*]} " in
        *" $s "*)
            echo; echo "==================== $s ===================="
            "step_$s"
            ;;
    esac
done
echo; echo "Finito ($NAME): ${WANTED[*]}"