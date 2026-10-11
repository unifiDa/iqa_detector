#!/usr/bin/env bash
# jagg.sh - pipeline di valutazione sul dataset btb-dresden inpaintato con jagg.
#
#   ./jagg.sh                         tutti i passi, nell'ordine
#   ./jagg.sh prompt mps merge        solo i passi indicati (sempre nell'ordine della pipeline)
#
# Passi: prepare, coherence, padding, quality, prompts, prompt, mps, merge
# Va messo nella radice del progetto (la cartella che contiene pixi.toml e eval_suite/).
set -euo pipefail
cd "$(dirname "$0")"

# ---- percorsi ---------------------------------------------------------------------------------
NAME=jagg
DATA=/home/dasara/Research/btb-dresden-all
INPAINTS=$DATA/jagg_inpaint/
MASKS=$DATA/btb_masks/
PROMPT_JSON=$DATA/miniInternVL_btb_prompts/
OUT=/home/dasara/Research/results/$NAME/
PROMPTS=$OUT/prompts.jsonl

CODE=/home/dasara/Research/code/iqa_detector
MPS_CKPT=$CODE/Weights/MPS_overall_checkpoint.pth
MPS_REPO=$CODE/MPS/

DEVICE=cuda:0
QUALITY_METRICS=(musiq clipiqa brisque niqe maniqa)

# ---- passi ------------------------------------------------------------------------------------
# ritagli sulla maschera (schema mask_<immagine>_<large|medium|small>.png)
step_prepare() {
    pixi run -e gpu prepare_generic --images "$INPAINTS" --out "$OUT" --masks "$MASKS"
}

# A. coerenza regione/contorno
step_coherence() {
    pixi run coherence --out "$OUT"
}

# pixel replicati nel frame e ritagli fuori dal frame (deve dare 0)
step_padding() {
    pixi run padding --out "$OUT"
}

# B. qualita' no-reference
step_quality() {
    pixi run -e gpu quality --device "$DEVICE" --metrics "${QUALITY_METRICS[@]}" --out "$OUT"
}

# JSON di prompt -> prompts.jsonl (id = nome del file inpaintato senza estensione)
step_prompts() {
    mkdir -p "$OUT"
    pixi run make_prompts --json-dir "$PROMPT_JSON" --out "$PROMPTS" --images "$INPAINTS"
}

# C. CLIP score
step_prompt() {
    pixi run -e gpu prompt --device "$DEVICE" --out "$OUT" --prompts "$PROMPTS"
}

# C. MPS score (ambiente dedicato, transformers 4.27.3); completa c_prompt.csv senza rifare CLIP
step_mps() {
    pixi run -e mps mps "$OUT" "$PROMPTS" "$MPS_CKPT" "$MPS_REPO"
}

# unisce tutto in results.csv e summary.json
step_merge() {
    pixi run merge --out "$OUT" --name "$NAME"
}

# ---- controlli --------------------------------------------------------------------------------
need() { [ -e "$1" ] || { echo "non trovato: $1  (controlla i percorsi in cima a $0)" >&2; exit 1; }; }

# ---- esecuzione -------------------------------------------------------------------------------
ALL=(prepare coherence padding quality prompts prompt mps merge)
if [ "$#" -eq 0 ]; then
    WANTED=("${ALL[@]}")
else
    for s in "$@"; do
        case " ${ALL[*]} " in *" $s "*) ;; *) echo "passo sconosciuto: $s (validi: ${ALL[*]})" >&2; exit 2 ;; esac
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