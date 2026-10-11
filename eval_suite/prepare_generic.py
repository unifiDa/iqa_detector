#!/usr/bin/env python3
"""
prepare_generic.py - passo 0 per inpainting GENERICO (non solo volti): niente rilevamento volti.

  - MASCHERA: e' la regione da valutare (region_src=mask). Senza maschera la regione e' l'immagine
    intera (region_src=none) e il ritaglio e' l'immagine intera;
  - RITAGLIO (crop_kind): mask = quadrato 512x512 attorno al bbox della maschera, full = immagine intera
    (lato lungo <= 768). Mai "face".

Scrive lo stesso <out>/index.csv e <out>/crops/<crop_kind>/<nome>.png di prepare.py, con le colonne
del volto (n_faces, face_found, det_score, face_in_mask, face_x0..y1, face_px) vuote (NaN), cosi' gli
script successivi funzionano senza modifiche. Richiede lo stesso common.py di prepare.py.

  python prepare_generic.py --images imgs/ --masks masks/ --out res_g1
  python prepare_generic.py --images imgs/ --masks masks/ --out res_g1 --mask-template "{stem}_m" --invert
  python prepare_generic.py --images imgs2/ --out res_g2                  # senza maschere: tutta l'immagine
  python prepare_generic.py --images imgs/ --masks masks/ --out res_g3 --crop full
"""
import argparse
import traceback
from pathlib import Path

import cv2

from common import (VERSION, INDEX_CSV, StageTable, bbox_of, choose_crop, find_mask_generic, list_images, load_mask)

NAN = float("nan")


def put_box(row, prefix, box):
    for k, v in zip(("x0", "y0", "x1", "y1"), box if box is not None else (None,) * 4):
        row[f"{prefix}_{k}"] = int(v) if v is not None else NAN


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--masks", type=Path, default=None, help="cartella delle maschere (consigliata)")
    ap.add_argument("--invert", action="store_true", help="maschera con nero = zona modificata")
    ap.add_argument("--crop", choices=["auto", "full"], default="auto",
                    help="auto: bbox della maschera (o intera se manca); full: sempre l'immagine intera")
    ap.add_argument("--min-area-frac", type=float, default=0.0,
                    help="maschere con area/immagine sotto questa soglia vengono trattate come assenti")
    ap.add_argument("--limit", type=int, default=None, help="al massimo N nuove immagini")
    ap.add_argument("--overwrite", action="store_true", help="ricalcola tutto (default: riprende)")
    ap.add_argument("--save-every", type=int, default=10)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"prepare_generic {VERSION}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = StageTable(out / INDEX_CSV, args.overwrite, args.save_every)

    files = [p for p in list_images(args.images) if table.get(p.name) is None]
    if args.limit is not None:
        files = files[:args.limit]
    print(f"{len(files)} immagini nuove da preparare")

    errors = no_mask = 0
    for i, path in enumerate(files, 1):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[skip] {path.name}: immagine illeggibile")
                continue
            h, w = bgr.shape[:2]

            mask, mask_path, mask_box = None, None, None
            if args.masks:
                mp = find_mask_generic(args.masks, path.stem)
                mask = load_mask(mp, bgr.shape, args.invert) if mp else None
                if mask is not None and mask.any() and mask.mean() >= args.min_area_frac:
                    mask_path, mask_box = mp, bbox_of(mask)
                else:
                    mask, no_mask = None, no_mask + 1

            region_src = "mask" if mask is not None else "none"
            # fbox=None: choose_crop ricade su maschera > intera
            crop, kind, cbox, src_px, scale = choose_crop(bgr, None, mask_box, args.crop)
            crop_file = Path("crops") / kind / f"{path.stem}.png"
            (out / crop_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / crop_file), crop)

            row = {"file": path.name, "image_path": str(path.resolve()),
                   "mask_path": str(Path(mask_path).resolve()) if mask_path else "",
                   "mask_invert": int(args.invert), "width": w, "height": h,
                   "region_src": region_src,
                   "region_area_frac": float(mask.mean()) if mask is not None else NAN,
                   "n_faces": NAN, "face_found": NAN, "det_score": NAN, "face_in_mask": NAN,
                   "crop_kind": kind, "crop_file": str(crop_file), "src_px": int(src_px), "upscale": float(scale),
                   "face_px": NAN}
            put_box(row, "face", None)
            put_box(row, "mask", mask_box)
            put_box(row, "crop", cbox)
            table.put(row)
            print(f"[{i}/{len(files)}] {path.name}  {kind}/{region_src}")
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[errore] {path.name}: {type(e).__name__}: {e}")
            if errors == 1:
                traceback.print_exc()

    table.save()
    df = table.df()
    if no_mask:
        print(f"[info] {no_mask} immagini senza maschera valida: regione = immagine intera")
    if errors:
        print(f"[info] {errors} immagini saltate per errore")
    if not df.empty:
        print(f"\nSalvato {out / INDEX_CSV} ({len(df)} immagini)")
        print("crop_kind:", df["crop_kind"].value_counts().to_dict())
        print("region_src:", df["region_src"].value_counts().to_dict())


if __name__ == "__main__":
    main()