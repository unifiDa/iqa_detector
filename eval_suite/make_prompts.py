#!/usr/bin/env python3
"""
make_prompts.py - converte i JSON di prompt (uno per immagine originale) nel prompts.jsonl
letto da c_prompt.py.

Ogni JSON e' una lista di stringhe del tipo
    "Rationale: ...\nReplace with: Snow-covered Fence Wall"
Il prompt viene estratto ESATTAMENTE come nello script di inpainting FLUX: il testo dopo
"Replace with: " (con lo spazio), senza togliere virgolette, cosi' CLIP/MPS valutano la stessa
stringa che ha ricevuto FLUX. Lo script FLUX salva  {image_name}_{size}_{i % 5}_inpainted.png
e, per via del "if i == 0: break", genera solo il primo prompt di ogni JSON: per questo di
default si esporta solo i=0 (--all per esportarli tutti).

L'id di ogni riga si costruisce con --id-template, che puo' usare:
    {name}  nome del JSON senza .json            es. Sony_DSC-W170_1_51977.JPG_large
            (= {image_name}_{size} dello script FLUX)
    {base}  {name} senza _large e senza estensione es. Sony_DSC-W170_1_51977
    {i}     indice del prompt nella lista (0, 1, ...)

  python make_prompts.py --json-dir prompts_json/ --out prompts.jsonl
  python make_prompts.py --json-dir prompts_json/ --out prompts.jsonl --images res_d1/crops
"""
import argparse
import json
import re
import sys
from pathlib import Path

MARKER = "Replace with: "
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def extract_prompt(text: str):
    """Stessa logica dello script FLUX: prompt_text[idx + len(marker):]."""
    idx = text.find(MARKER)
    if idx == -1:
        return None
    return text[idx + len(MARKER):] or None


def base_of(name: str) -> str:
    name = re.sub(r"_large$", "", name)
    return Path(name).stem if Path(name).suffix.lower() in IMG_EXT else name


def main(argv=None):
    ap = argparse.ArgumentParser(description="JSON di prompt -> prompts.jsonl per c_prompt.py",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--id-template", default="{name}_{i}_inpainted",
                    help="nome file inpaintato senza estensione (FLUX usa i %% 5)")
    ap.add_argument("--all", action="store_true",
                    help="esporta tutti i prompt, non solo il primo (FLUX genera solo i=0)")
    ap.add_argument("--images", type=Path, default=None,
                    help="cartella delle immagini inpaintate: controlla quali id hanno un'immagine")
    args = ap.parse_args(argv)

    files = sorted(args.json_dir.glob("*.json"))
    if not files:
        sys.exit(f"Nessun .json in {args.json_dir}")

    rows, bad = [], []
    for f in files:
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            bad.append(f"{f.name}: JSON non valido ({e})")
            continue
        if not isinstance(items, list):
            bad.append(f"{f.name}: atteso una lista, trovato {type(items).__name__}")
            continue
        name = f.name[:-len(".json")]
        for i, text in enumerate(items if args.all else items[:1]):
            prompt = extract_prompt(text) if isinstance(text, str) else None
            if prompt is None:
                bad.append(f"{f.name}[{i}]: 'Replace with:' non trovato")
                continue
            rid = args.id_template.format(name=name, base=base_of(name), i=i % 5)
            rows.append({"record_id": rid, "prompt": prompt, "source": f.name, "index": i,
                         "rationale": text.split(MARKER)[0].replace("Rationale:", "").strip()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Scritte {len(rows)} righe da {len(files)} JSON in {args.out}")
    for r in rows[:3]:
        print(f"  {r['record_id']}  ->  {r['prompt']}")
    if bad:
        print(f"[warn] {len(bad)} elementi saltati:")
        for b in bad[:10]:
            print("  " + b)

    if args.images:
        stems = {p.stem for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXT}
        ids = [r["record_id"] for r in rows]
        hit = [i for i in ids if i in stems or any(s.startswith(i) for s in stems)]
        print(f"\nControllo immagini: {len(hit)}/{len(ids)} id trovano un'immagine in {args.images}")
        if len(hit) < len(ids):
            miss = [i for i in ids if i not in hit][:3]
            print(f"  es. id senza immagine: {miss}")
            print(f"  es. nomi immagine:     {sorted(stems)[:3]}")
            print("  -> se i nomi non tornano, cambia --id-template")


if __name__ == "__main__":
    main()