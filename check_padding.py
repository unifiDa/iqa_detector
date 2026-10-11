#!/usr/bin/env python3
"""
check_padding.py - cerca bande di PIXEL REPLICATI sui bordi (le strisce uniformi in verticale/orizzontale che
si vedono quando un'immagine e' stata estesa copiando l'ultima riga o colonna).

Per ogni immagine controlla due cose e le tiene separate:
  frame_pad_*  il frame originale (image_path in index.csv): la banda c'e' gia' nell'immagine che
               hai dato in pasto allo strumento (padding del generatore, frame del video esteso, ...);
  crop_pad_*   il ritaglio salvato da prepare.py: se la banda c'e' solo qui, la causa e' il ritaglio
               (ritagli prodotti con il vecchio common.py, che usciva dall'immagine e replicava i bordi).

Un lato ha padding se le prime righe/colonne dal bordo sono quasi uguali alla riga/colonna di bordo
(differenza media < --eps livelli di grigio) E la riga di bordo ha una texture (dev. std > --min-std),
altrimenti sarebbe una banda nera/uniforme (letterbox), che e' un'altra cosa.
Lo spessore e' in pixel; <side>_pad_frac e' lo spessore diviso per il lato dell'immagine.

Scrive <out>/padding.csv (colonna "file" per il merge) e stampa un riepilogo.

  python check_padding.py --out res_d1
  python check_padding.py --out res_d1 --min-frac 0.02 --eps 2.0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from eval_suite.common import load_index

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
    ap = argparse.ArgumentParser(description="Cerca bande di pixel replicati sui bordi",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="cartella creata da prepare.py")
    ap.add_argument("--eps", type=float, default=1.5, help="differenza media (livelli di grigio) per dire 'uguale'")
    ap.add_argument("--min-std", type=float, default=3.0, help="texture minima della riga di bordo (esclude le bande uniformi)")
    ap.add_argument("--min-frac", type=float, default=0.03, help="spessore minimo (frazione del lato) per il flag")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    index = load_index(args.out)
    rows = []
    for i, r in enumerate(index.to_dict("records")[:args.limit], 1):
        row = {"file": r["file"]}
        for tag, path in (("frame", r.get("image_path")), ("crop", args.out / r["crop_file"])):
            img = cv2.imread(str(path), cv2.IMREAD_COLOR) if isinstance(path, (str, Path)) else None
            if img is None:
                continue
            for k, v in padding(img, args.eps, args.min_std).items():
                row[f"{tag}_{k}"] = v
        for tag in ("frame", "crop"):
            cols = [f"{tag}_{s}_pad_frac" for s in SIDES]
            if all(c in row for c in cols):
                row[f"{tag}_pad_sides"] = ",".join(s for s, c in zip(SIDES, cols) if row[c] >= args.min_frac)
        rows.append(row)
        if i % 200 == 0:
            print(f"[{i}]")

    df = pd.DataFrame(rows)
    path = args.out / "padding.csv"
    df.to_csv(path, index=False)
    print(f"Salvato {path} ({len(df)} immagini)")
    for tag, label in (("frame", "nel frame originale"), ("crop", "nel ritaglio")):
        col = f"{tag}_pad_sides"
        if col in df:
            has = df[col].fillna("") != ""
            print(f"{label}: {int(has.sum())}/{len(df)} con banda replicata (>= {args.min_frac:.0%} del lato)")
            if has.any():
                sides = df.loc[has, col].str.split(",").explode().value_counts().to_dict()
                print("   lati:", sides)
    if "frame_pad_sides" in df and "crop_pad_sides" in df:
        only_crop = (df["crop_pad_sides"].fillna("") != "") & (df["frame_pad_sides"].fillna("") == "")
        print(f"banda SOLO nel ritaglio (colpa del ritaglio, rifai prepare --overwrite): {int(only_crop.sum())}")


if __name__ == "__main__":
    main()