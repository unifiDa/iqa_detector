#!/usr/bin/env python3
"""
prepare_generic.py - step 0 for GENERIC inpainting (not only faces): no face detection.

  - MASK (name: mask_<name up to large|medium|small>.<ext>, see find_mask_generic in common.py):
    it is the region to evaluate (region_src=mask). Without a mask the region is the whole
    image (region_src=none) and the crop is the whole image;
  - CROP (crop_kind): mask = 512x512 square around the mask bbox, full = whole image
    (long side <= 768). Never "face".

Writes the same <out>/index.csv and <out>/crops/<crop_kind>/<name>.png as prepare.py, with the face
columns (n_faces, face_found, det_score, face_in_mask, face_x0..y1, face_px) empty (NaN), so the
following scripts work unchanged. Needs the same common.py as prepare.py.

  python prepare_generic.py --images imgs/ --masks masks/ --out res_g1
  python prepare_generic.py --images imgs/ --masks masks/ --out res_g1 --invert      # black = edited area
  python prepare_generic.py --images imgs2/ --out res_g2                  # without masks: whole image
  python prepare_generic.py --images imgs/ --masks masks/ --out res_g3 --crop full
"""
import argparse
import traceback
from pathlib import Path

import cv2

from common import (VERSION, INDEX_CSV, StageTable, bbox_of, check_mask_args, choose_crop, find_mask_generic,
                    find_mask_regex, list_images, load_mask)

NAN = float("nan")


def put_box(row, prefix, box):
    for k, v in zip(("x0", "y0", "x1", "y1"), box if box is not None else (None,) * 4):
        row[f"{prefix}_{k}"] = int(v) if v is not None else NAN


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--masks", type=Path, default=None, help="mask folder (recommended)")
    ap.add_argument("--mask-regex", default=None,
                    help="regex on the image name (without extension) to use another mask naming scheme than "
                         "mask_<name up to large|medium|small>; needs --mask-template")
    ap.add_argument("--mask-template", default=None,
                    help="mask name (without extension) built from the groups of --mask-regex, "
                         "e.g. 'mask_{base}' (see find_mask_regex in common.py)")
    ap.add_argument("--invert", action="store_true", help="mask with black = edited area")
    ap.add_argument("--crop", choices=["auto", "full"], default="auto",
                    help="auto: mask bbox (or whole image if missing); full: always the whole image")
    ap.add_argument("--min-area-frac", type=float, default=0.0,
                    help="masks with area/image below this threshold are treated as absent")
    ap.add_argument("--limit", type=int, default=None, help="at most N new images")
    ap.add_argument("--overwrite", action="store_true", help="recompute everything (default: resume)")
    ap.add_argument("--save-every", type=int, default=10)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    check_mask_args(ap, args.mask_regex, args.mask_template)
    print(f"prepare_generic {VERSION}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = StageTable(out / INDEX_CSV, args.overwrite, args.save_every)

    files = [p for p in list_images(args.images) if table.get(p.name) is None]
    if args.limit is not None:
        files = files[:args.limit]
    print(f"{len(files)} new images to prepare")

    errors = no_mask = 0
    for i, path in enumerate(files, 1):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[skip] {path.name}: unreadable image")
                continue
            h, w = bgr.shape[:2]

            mask, mask_path, mask_box = None, None, None
            if args.masks:
                mp = (find_mask_regex(args.masks, path.stem, args.mask_regex, args.mask_template) if args.mask_regex
                      else find_mask_generic(args.masks, path.stem))
                mask = load_mask(mp, bgr.shape, args.invert) if mp else None
                if mask is not None and mask.any() and mask.mean() >= args.min_area_frac:
                    mask_path, mask_box = mp, bbox_of(mask)
                else:
                    mask, no_mask = None, no_mask + 1

            region_src = "mask" if mask is not None else "none"
            # fbox=None: choose_crop falls back to mask > whole image
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
            print(f"[error] {path.name}: {type(e).__name__}: {e}")
            if errors == 1:
                traceback.print_exc()

    table.save()
    df = table.df()
    if no_mask:
        print(f"[info] {no_mask} images without a valid mask: region = whole image")
    if errors:
        print(f"[info] {errors} images skipped because of errors")
    if not df.empty:
        print(f"\nSaved {out / INDEX_CSV} ({len(df)} images)")
        print("crop_kind:", df["crop_kind"].value_counts().to_dict())
        print("region_src:", df["region_src"].value_counts().to_dict())


if __name__ == "__main__":
    main()