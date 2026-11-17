#!/usr/bin/env python3
"""
check_padding.py - looks for REPLICATED PIXELS on the borders (the uniform vertical/horizontal stripes that
appear when an image has been extended by copying the last row or column).

Two separate checks:

  FRAME   (frame_*)  heuristic on the original frame (image_path in index.csv). A side has a band if the first
          rows/columns from the border are almost equal to the border row/column (mean difference < --eps grey
          levels) AND the border row has texture (std dev > --min-std; otherwise it is a uniform band,
          letterbox, which is something else) AND the thickness exceeds --min-frac of the side.
          Columns: frame_<side>_pad (px), frame_<side>_pad_frac, frame_pad_sides (sides with a band).
          The band is already in the image fed to the tool (generator padding, etc.): it also biases the
          coherence metrics, which work on the frame with the mask. Flag "edge padding".

  CROP (crop_outside_sides)  EXACT check, no heuristics: sides on which the crop box
          (crop_x0..y1 in index.csv) goes outside the frame. With the fixed common.py it is always empty; if it is
          not, those crops have certain replicated pixels (produced with the old common.py) and must be redone.
          Flag "padded crop".
          The crop is not checked with the frame heuristic: it confuses vertical content (walls, dark
          backgrounds) and interpolation with replicated bands.

Writes <out>/padding.csv ("file" column); load_merged (common.py) merges it with the other results and adds the
flags to "flags", so they appear in results.csv, summary.json and in the viewer.

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
    """Consecutive rows from the top border almost equal to the border row (0 if the border row is flat)."""
    first = gray[0]
    if float(first.std()) < min_std:
        return 0
    diff = np.abs(gray - first[None, :]).mean(axis=1)          # mean difference of every row from the border row
    bad = np.flatnonzero(diff >= eps)
    return int(bad[0]) if len(bad) else gray.shape[0]          # first row that departs = thickness


def padding(bgr, eps, min_std):
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    views = {"top": g, "bottom": g[::-1], "left": g.T, "right": g[:, ::-1].T}
    out = {}
    for side, v in views.items():
        t = band_thickness(np.ascontiguousarray(v), eps, min_std)
        out[f"{side}_pad"] = t if t > 1 else 0                  # 1 = only the border row itself
        out[f"{side}_pad_frac"] = (t if t > 1 else 0) / v.shape[0]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Looks for replicated pixels on the borders",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder created by prepare.py")
    ap.add_argument("--eps", type=float, default=1.5, help="mean difference (grey levels) to call two rows 'equal'")
    ap.add_argument("--min-std", type=float, default=3.0, help="minimum texture of the border row (excludes uniform bands)")
    ap.add_argument("--min-frac", type=float, default=0.03, help="minimum thickness (fraction of the frame side) for the band")
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
    print(f"Saved {path} ({len(df)} images)")
    if "frame_pad_sides" in df:
        has = df["frame_pad_sides"].fillna("") != ""
        print(f"frames with a replicated band (>= {args.min_frac:.0%} of the side): {int(has.sum())}/{len(df)}  -> flag edge padding")
        if has.any():
            print("   sides:", df.loc[has, "frame_pad_sides"].str.split(",").explode().value_counts().to_dict())
    if "crop_outside_sides" in df:
        out = df["crop_outside_sides"].fillna("") != ""
        print(f"crops with the box OUTSIDE the frame (certain padding): {int(out.sum())}/{len(df)}  -> flag padded crop"
              + ("" if out.any() else "  (ok)"))
        if out.any():
            print("   the crops are old: delete --out and re-run prepare with the fixed common.py")


if __name__ == "__main__":
    main()