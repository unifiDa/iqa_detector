#!/usr/bin/env python3
"""
c_prompt.py - C. aderenza al prompt e preferenza umana (serve il file dei prompt).

  clip_score  100 x coseno(embedding immagine, embedding testo), CLIP ViT-L/14.
              Valori tipici 20-30; confronta solo immagini con lo stesso prompt.
  mps_score   Multi-dimensional Preference Score (Kwai-Kolors/MPS, CVPR 2024), score "overall" per
              singola immagine (logit_scale x coseno testo-immagine condizionato). Piu' alto = piu'
              preferita, ma la scala ha senso solo a parita' di prompt.

Il prompt di ogni immagine si cerca per nome file, nome senza estensione, poi prefisso piu' lungo
(es. id 'frame_000758_track001' per 'frame_000758_track001_inpainted-0.png').

  python c_prompt.py --out res_d1 --prompts prompts.jsonl
  python c_prompt.py --out res_d1 --prompts prompts.jsonl --metrics mps --mps-ckpt X.pth --mps-repo MPS/
"""
import argparse
import sys
import traceback
from pathlib import Path

import cv2

from common import (STAGE_CSV, StageTable, describe, load_index, load_prompts, lookup_prompt,
                    resolve_device, run_crop_metrics)

CLIP_NAME = "openai/clip-vit-large-patch14"
MPS_PROC = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MPS_CONDITION = ("light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, "
                 "structure, instance, texture, quantity, attributes, position, number, location, word, things.")
COLUMN = {"clip": "clip_score", "mps": "mps_score"}


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP_NAME).to(device).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)


def clip_score(model_proc, crop_bgr, prompt, device):
    import torch
    model, proc = model_proc
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(text=[prompt], images=rgb, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img * txt).sum() * 100)


def load_mps(ckpt: Path, repo, device):
    """Carica il checkpoint MPS 'overall'. ATTENZIONE: il file e' un modello serializzato con pickle
    (torch.load con weights_only=False) e quindi esegue codice al caricamento: scaricalo solo dal link
    ufficiale del repo Kwai-Kolors/MPS.
    Il pickle contiene l'intero oggetto, quindi al caricamento servono le stesse classi con cui e' stato
    salvato: trainer.models.* dal repo clonato (--mps-repo, nel percorso di import; il suo
    requirements.txt NON va installato) e le classi interne di transformers 4.27.3 (CLIPTextTransformer,
    ecc., rimosse nelle versioni recenti). Per questo MPS gira nell'ambiente pixi 'mps'."""
    import transformers
    if transformers.__version__ != "4.27.3":
        print(f"[warn] transformers {transformers.__version__}: il checkpoint MPS e' stato salvato con 4.27.3 "
              f"e con versioni diverse puo' non caricarsi. Usa: pixi run -e mps mps ... (o -e mps-cpu senza GPU)")
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor
    if repo:
        sys.path.insert(0, str(repo))
    proc = CLIPImageProcessor.from_pretrained(MPS_PROC)
    tok = AutoTokenizer.from_pretrained(MPS_PROC)
    try:
        model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:                                      # torch < 1.13
        model = torch.load(str(ckpt), map_location="cpu")
    # il checkpoint e' in fp16: va bene su GPU, ma su CPU molte operazioni fp16 non esistono
    # o sono lentissime, quindi su CPU si converte tutto in fp32 (stesso risultato, piu' memoria)
    model = model.eval().to(device)
    if not str(device).startswith("cuda"):
        model = model.float()
    dtype = next(model.parameters()).dtype
    print(f"[mps] caricato su {device} ({dtype})")
    return model, proc, tok, device


def mps_score(mps, crop_bgr, prompt):
    """Replica infer_one_sample() di eval_overall_mps_on_hpdv2.py del repo MPS."""
    import torch
    from PIL import Image
    model, proc, tok, device = mps

    def ids(text):
        return tok(text, max_length=tok.model_max_length, padding="max_length", truncation=True,
                   return_tensors="pt").input_ids.to(device)

    dtype = next(model.parameters()).dtype                 # fp16 su GPU, fp32 su CPU (vedi load_mps)
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pix = proc(img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        text_f, text_eos = model.model.get_text_features(ids(prompt))
        image_f = model.model.get_image_features(pix.to(dtype))
        cond_f, _ = model.model.get_text_features(ids(MPS_CONDITION))
        sim = torch.einsum("b i d, b j d -> b j i", text_f, cond_f)
        sim = sim.max(dim=1, keepdim=True)[0]
        sim = sim / sim.max()
        mask = torch.where(sim > 0.3, 0, float("-inf")).repeat(1, image_f.shape[1], 1)
        feat = model.cross_model(image_f, text_f, mask.to(dtype))[:, 0, :]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        txt = text_eos / text_eos.norm(dim=-1, keepdim=True)
        return float((model.logit_scale.exp() * txt @ feat.T)[0, 0])


def build_parser():
    ap = argparse.ArgumentParser(description="C. aderenza al prompt (CLIP, MPS)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="cartella creata da prepare.py")
    ap.add_argument("--prompts", required=True, type=Path, help="JSONL/CSV/TSV con id e prompt")
    ap.add_argument("--id-key", default="record_id")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--metrics", nargs="+", choices=["clip", "mps"], default=["clip"])
    ap.add_argument("--mps-ckpt", type=Path, default=None, help="MPS_overall_checkpoint.pth")
    ap.add_argument("--mps-repo", type=Path, default=None, help="cartella del repo Kwai-Kolors/MPS clonato")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-every", type=int, default=25)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if "mps" in args.metrics and not args.mps_ckpt:
        sys.exit("--metrics mps richiede --mps-ckpt (checkpoint MPS overall) e di solito --mps-repo")

    device = resolve_device(args.device)
    index = load_index(args.out)
    prompts = load_prompts(args.prompts, args.id_key, args.prompt_key)
    prompt_of = {f: lookup_prompt(prompts, f) for f in index["file"]}
    missing = [f for f, p in prompt_of.items() if p is None]
    if missing:
        print(f"[warn] {len(missing)}/{len(index)} immagini senza prompt, es. {missing[:2]}; "
              f"id nel file prompts, es. {list(prompts)[:2]}")

    table = StageTable(args.out / STAGE_CSV["c"], args.overwrite, args.save_every)
    metrics = {n: COLUMN[n] for n in args.metrics}
    clip = load_clip(device) if "clip" in metrics else None
    mps = None
    if "mps" in metrics:
        try:
            mps = load_mps(args.mps_ckpt, args.mps_repo, device)
        except Exception as e:  # noqa: BLE001
            print("\n" + "=" * 70)
            print(f"[ERRORE] MPS non caricato: {type(e).__name__}: {e}")
            traceback.print_exc()
            if "Can't get attribute" in str(e) and "transformers" in str(e):
                print("Il checkpoint richiede le classi di transformers 4.27.3: lancia MPS nell'ambiente "
                      "dedicato, es.  pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/")
            elif isinstance(e, ModuleNotFoundError) and "trainer" in str(e):
                print("--mps-repo deve puntare alla radice del repo Kwai-Kolors/MPS clonato "
                      "(quella che contiene la cartella 'trainer/').")
            else:
                print("Controlla: --mps-ckpt punta al file .pth; --mps-repo punta alla radice del repo clonato "
                      "(quella che contiene la cartella 'trainer/').")
            print("=" * 70 + "\n")
            metrics.pop("mps")
    if not metrics:
        sys.exit("Nessuna metrica disponibile.")

    def score(name, crop, row):
        prompt = prompt_of[row["file"]]
        if name == "clip":
            return clip_score(clip, crop, prompt, device)
        try:
            return mps_score(mps, crop, prompt)
        except Exception:
            print("\n[ERRORE] MPS fallito sulla prima immagine: lo disattivo per questa esecuzione.")
            traceback.print_exc()
            metrics.pop("mps", None)
            raise

    def applicable(name, row):
        return prompt_of.get(row["file"]) is not None

    def extra(row):
        return {"prompt": prompt_of.get(row["file"])}

    run_crop_metrics(args.out, index, table, metrics, score, applicable, args.limit, extra)
    print(f"\nSalvato {table.path}")
    describe(table.df(), list(COLUMN.values()))


if __name__ == "__main__":
    main()
