#!/usr/bin/env python3
"""
check_padding.py - cerca PIXEL REPLICATI sui bordi (le strisce uniformi in verticale/orizzontale che si vedono
quando un'immagine e' stata estesa copiando l'ultima riga o colonna).

Due controlli, separati:

  FRAME   (frame_*)  euristica sul frame originale (image_path in index.csv). Un lato ha una banda se le prime
          righe/colonne dal bordo sono quasi uguali alla riga/colonna di bordo (differenza media < --eps livelli
          di grigio) E la riga di bordo ha texture (dev. std > --min-std; altrimenti e' una banda uniforme,
          letterbox, che e' un'altra cosa) E lo spessore supera --min-frac del lato.
          Colonne: frame_<lato>_pad (px), frame_<lato>_pad_frac, frame_pad_sides (lati con banda).
          La banda e' gia' nell'immagine data in pasto allo strumento (padding del generatore, ecc.): falsa
          anche le metriche di coerenza, che lavorano sul frame con la maschera. Flag "pixel_replicati".

  RITAGLIO (crop_outside_sides)  verifica ESATTA, senza euristiche: lati su cui il box del ritaglio
          (crop_x0..y1 in index.csv) esce dal frame. Con il common.py corretto e' sempre vuoto; se non lo e', quei
          ritagli hanno pixel replicati certi (prodotti con il vecchio common.py) e vanno rifatti. Flag
          "ritaglio_con_padding".
          Non si controlla il ritaglio con l'euristica del frame: confonde contenuto verticale (pareti, sfondi
          scuri) e interpolazione con bande replicate.

Scrive <out>/padding.csv (colonna "file"); load_merged (common.py) lo unisce agli altri risultati e aggiunge i
flag a "flags", quindi compaiono in results.csv, summary.json e nel viewer.

  python check_padding.py --out res_d1
  python check_padding.py --out res_d1 --min-frac 0.02 --eps 2.0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import STAGE_CSV, load_index

SIDES = ("top", "bottom", "left", "right")


def band_thickness(gray, eps, min_std):
    """Righe consecutive dal bordo superiore quasi uguali alla riga di bordo (0 se la riga di bordo e' piatta)."""
    first = gray[0]
    if float(first.std()) < min_std:
        return 0
    diff = np.abs(gray - first[None, :]).mean(axis=1)          # differenza media di ogni riga dalla riga di bordo
    bad = np.flatnonzero(diff >= eps)
    return int(bad[0]) if len(bad) else gray.shape[0]          # prima riga che si discosta = spessore


def padding(bgr, eps, min_std):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    views = {"top": g, "bottom": g[::-1], "left": g.T, "right": g[:, ::-1].T}
    out = {}
    for side, v in views.items():
        t = band_thickness(np.ascontiguousarray(v), eps, min_std)
        out[f"{side}_pad"] = t if t > 1 else 0                  # 1 = solo la riga di bordo stessa
        out[f"{side}_pad_frac"] = (t if t > 1 else 0) / v.shape[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Cerca pixel replicati sui bordi",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="cartella creata da prepare.py")
    ap.add_argument("--eps", type=float, default=1.5, help="differenza media (livelli di grigio) per dire 'uguale'")
    ap.add_argument("--min-std", type=float, default=3.0, help="texture minima della riga di bordo (esclude le bande uniformi)")
    ap.add_argument("--min-frac", type=float, default=0.03, help="spessore minimo (frazione del lato del frame) per la banda")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    index = load_index(args.out)
    rows = []
    for i, r in enumerate(index.to_dict("records")[:args.limit], 1):
        row = {"file": r["file"]}
        img = cv2.imread(str(r["image_path"]), cv2.IMREAD_COLOR) if isinstance(r.get("image_path"), str) else None
        if img is not None:
            for k, v in padding(img, args.eps, args.min_std).items():
                row[f"frame_{k}"] = v
            row["frame_pad_sides"] = ",".join(s for s in SIDES if row[f"frame_{s}_pad_frac"] >= args.min_frac)
        box = [r.get(k) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")]
        if all(pd.notna(v) for v in box) and pd.notna(r.get("width")) and pd.notna(r.get("height")):
            x0, y0, x1, y1 = (int(v) for v in box)
            outside = {"top": y0 < 0, "bottom": y1 > int(r["height"]), "left": x0 < 0, "right": x1 > int(r["width"])}
            row["crop_outside_sides"] = ",".join(k for k, v in outside.items() if v)
        rows.append(row)
        if i % 200 == 0:
            print(f"[{i}]")

    df = pd.DataFrame(rows)
    path = args.out / STAGE_CSV["p"]
    df.to_csv(path, index=False)
    print(f"Salvato {path} ({len(df)} immagini)")
    if "frame_pad_sides" in df:
        has = df["frame_pad_sides"].fillna("") != ""
        print(f"frame con banda replicata (>= {args.min_frac:.0%} del lato): {int(has.sum())}/{len(df)}  -> flag edge padding")
        if has.any():
            print("   lati:", df.loc[has, "frame_pad_sides"].str.split(",").explode().value_counts().to_dict())
    if "crop_outside_sides" in df:
        out = df["crop_outside_sides"].fillna("") != ""
        print(f"ritagli con box FUORI dal frame (padding certo): {int(out.sum())}/{len(df)}  -> flag padded crop"
              + ("" if out.any() else "  (ok)"))
        if out.any():
            print("   i ritagli sono vecchi: cancella --out e rifai prepare con il common.py corretto")


if __name__ == "__main__":
    main()