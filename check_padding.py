#!/usr/bin/env python3
"""
check_padding.py - cerca bande di PIXEL REPLICATI sui bordi (le strisce uniformi in verticale/orizzontale che
si vedono quando un'immagine e' stata estesa copiando l'ultima riga o colonna).

Per ogni immagine controlla due cose e le tiene separate:
  frame_pad_*  il frame originale (image_path in index.csv): la banda c'e' gia' nell'immagine che
               hai dato in pasto allo strumento (padding del generatore, frame del video esteso, ...);
  crop_pad_*   il ritaglio salvato da prepare.py: se la banda c'e' solo qui, la causa e' il ritaglio
               (ritagli prodotti con il vecchio common.py, che usciva dall'immagine e replicava i bordi);
  crop_outside_sides  verifica ESATTA, senza euristiche: lati su cui il box del ritaglio (crop_x0..y1 in
               index.csv) esce dal frame. Con il common.py corretto e' sempre vuoto; se non lo e', quei
               ritagli hanno pixel replicati sicuri e vanno rifatti (prepare --overwrite).

Un lato ha padding se le prime righe/colonne dal bordo sono quasi uguali alla riga/colonna di bordo
(differenza media < --eps livelli di grigio) E la riga di bordo ha una texture (dev. std > --min-std),
altrimenti sarebbe una banda nera/uniforme (letterbox), che e' un'altra cosa.
Lo spessore e' in pixel; <side>_pad_frac e' lo spessore diviso per il lato dell'immagine. Nei ritagli lo
spessore va anche confrontato in pixel del FRAME (spessore / upscale, --min-src-px): un ritaglio molto
ingrandito ha righe vicine quasi uguali per semplice interpolazione, senza nessun padding.

Scrive <out>/padding.csv (colonna "file" per il merge) e stampa un riepilogo.

  python check_padding.py --out res_d1
  python check_padding.py --out res_d1 --min-frac 0.02 --eps 2.0
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import load_index

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
    ap.add_argument("--min-src-px", type=float, default=4.0,
                    help="solo ritagli: spessore minimo in pixel del frame (spessore / upscale) per il flag")
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
        up = float(r.get("upscale") or 1.0)
        for tag in ("frame", "crop"):
            cols = [f"{tag}_{s}_pad_frac" for s in SIDES]
            if all(c in row for c in cols):
                min_px = args.min_src_px if tag == "crop" else 0.0
                row[f"{tag}_pad_sides"] = ",".join(
                    s for s, c in zip(SIDES, cols)
                    if row[c] >= args.min_frac and row[f"{tag}_{s}_pad"] / (up if tag == "crop" else 1.0) >= min_px)
        box = [r.get(k) for k in ("crop_x0", "crop_y0", "crop_x1", "crop_y1")]
        if all(pd.notna(v) for v in box) and pd.notna(r.get("width")) and pd.notna(r.get("height")):
            x0, y0, x1, y1 = (int(v) for v in box)
            outside = {"top": y0 < 0, "bottom": y1 > int(r["height"]), "left": x0 < 0, "right": x1 > int(r["width"])}
            row["crop_outside_sides"] = ",".join(k for k, v in outside.items() if v)
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
    if "crop_outside_sides" in df:
        out = df["crop_outside_sides"].fillna("") != ""
        print(f"ritagli con box FUORI dal frame (padding certo, vecchio common.py): {int(out.sum())}/{len(df)}")
    if "frame_pad_sides" in df and "crop_pad_sides" in df:
        only_crop = (df["crop_pad_sides"].fillna("") != "") & (df["frame_pad_sides"].fillna("") == "")
        print(f"banda SOLO nel ritaglio (colpa del ritaglio, rifai prepare --overwrite): {int(only_crop.sum())}")


if __name__ == "__main__":
    main()