#!/usr/bin/env python3
"""
a_coherence.py - A. coherence of the region (mask or face ellipse) with its surroundings.

Compares the region with a ring around it (--ref ring) or with the rest of the frame (--ref frame).
The ring is more local: it reduces false positives from vignetting, shadows and different backgrounds.
Erosion, safety gap, seam thickness and ring width are proportional to the size of the region, so they
work on small and large faces alike.

1. BLUR (blur_logratio) = ln(sharpness_inside / sharpness_outside). Sharpness = 95th percentile of the
   gradient (Sobel) of L after a sigma-1 blur that suppresses grain. Negative = region blurrier
   ("blurry"), positive = sharper ("oversharp"). A face naturally has more edges than a smooth
   background: a small positive value is normal, look at the distribution.
   lapvar_crop = variance of the Laplacian of the crop (512 px): an absolute measure, available even
   without a region; it grows with sharpness AND with grain.

2. TINT (Lab, perceptually uniform: ~2-3 units = barely visible). Distances (a, b) from the median
   tint of the surroundings:
     ab_shift           distance between the median tint inside and outside;
     color_dev_frac_in  fraction of region pixels whose tint differs from that of the surroundings;
     ab_spread_out      how "monochrome" the surroundings are (low = B/W, sepia, toning).
   Flag "color mismatch": monochrome surroundings BUT a region of another colour.

3. SEAM (mask only): jump in lightness (seam_L, L units 0-100) and in tint (seam_ab) between a band
   just inside and one just outside the mask contour (medians over the whole contour).

The flags are recomputed at every run from the columns already saved: to try different thresholds
just re-run with --blur-thr / --seam-thr / ... (the images are not processed again).

  python a_coherence.py --out res_d1
  python a_coherence.py --out res_d1 --blur-thr 0.7 --seam-thr 8      # new flags only
"""
import argparse
import math
import traceback
from pathlib import Path

import cv2
import numpy as np

from common import (STAGE_CSV, StageTable, box_from_row, face_region, is_num, load_index, load_mask)

EPS = 1e-6
MIN_PX = 200                 # minimum pixels in region and reference to compute the statistics
MIN_RING_PX = 400            # below this the ring is replaced by the rest of the frame
COLOR_D = 8.0                # Lab (a,b) distance beyond which a pixel is "of another colour"
AB_SHIFT_THR = 6.0           # distance between median tints beyond which the region is "of another tint"


def region_geometry(region, ref_mode):
    """Comparison zones, with distances proportional to the size of the region."""
    reg = region.astype(np.uint8)
    s = math.sqrt(max(int(reg.sum()), 1))                  # "side" of the region
    erode_px = max(2, round(0.04 * s))                     # inside, away from the border
    gap_px = max(6, round(0.15 * s))                       # safety gap outside the border
    seam_px = max(2, round(0.03 * s))                      # thickness of the seam bands
    ring_px = max(12, round(0.60 * s))                     # width of the reference ring
    d_in = cv2.distanceTransform(reg, cv2.DIST_L2, 3)
    d_out = cv2.distanceTransform(1 - reg, cv2.DIST_L2, 3)
    inner = d_in > erode_px
    frame_out = d_out > gap_px
    ring = frame_out & (d_out <= gap_px + ring_px)
    if ref_mode == "ring" and ring.sum() >= MIN_RING_PX:
        ref, used = ring, "ring"
    else:
        ref, used = frame_out, "frame"
    return dict(inner=inner, ref=ref, ref_used=used,
                band_in=region & (d_in <= seam_px), band_out=(~region) & (d_out <= seam_px))


def gradient_magnitude(L):
    g = cv2.GaussianBlur(L, (0, 0), 1.0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.magnitude(gx, gy)


def coherence(bgr, region, ref_mode, with_seam):
    geo = region_geometry(region, ref_mode)
    inner, ref = geo["inner"], geo["ref"]
    if inner.sum() < MIN_PX or ref.sum() < MIN_PX:
        return {}
    lab = cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2Lab)    # L 0-100, a/b ~ -127..127
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    out = {"ref_used": geo["ref_used"]}

    grad = gradient_magnitude(L)
    b_in, b_out = float(np.percentile(grad[inner], 95)), float(np.percentile(grad[ref], 95))
    out.update(blur_in=b_in, blur_out=b_out, blur_logratio=math.log((b_in + EPS) / (b_out + EPS)))

    valid = (L > 10) & (L < 98)                            # a/b unstable on very dark or burnt pixels
    v_in, v_ref = inner & valid, ref & valid
    if v_in.sum() >= MIN_PX // 2 and v_ref.sum() >= MIN_PX // 2:
        a_o, b_o = float(np.median(A[v_ref])), float(np.median(B[v_ref]))
        a_i, b_i = float(np.median(A[v_in])), float(np.median(B[v_in]))
        d_ref = np.hypot(A[v_ref] - a_o, B[v_ref] - b_o)
        d_reg = np.hypot(A[v_in] - a_o, B[v_in] - b_o)
        out.update(
            chroma_in=float(np.median(np.hypot(A[v_in], B[v_in]))),
            chroma_out=float(np.median(np.hypot(A[v_ref], B[v_ref]))),
            ab_shift=math.hypot(a_i - a_o, b_i - b_o),
            ab_spread_out=float(np.percentile(d_ref, 95)),
            color_dev_frac_in=float((d_reg > COLOR_D).mean()),
        )

    if with_seam and geo["band_in"].any() and geo["band_out"].any():
        bi, bo = geo["band_in"], geo["band_out"]
        out["seam_L"] = abs(float(np.median(L[bi])) - float(np.median(L[bo])))
        vi, vo = bi & valid, bo & valid
        if vi.any() and vo.any():
            out["seam_ab"] = math.hypot(float(np.median(A[vi])) - float(np.median(A[vo])),
                                        float(np.median(B[vi])) - float(np.median(B[vo])))
    return out


def coherence_flags(r, a):
    """Indicative thresholds: tune them on the distributions of your dataset."""
    f = []
    v = lambda k: r.get(k) if is_num(r.get(k)) else None   # noqa: E731
    if v("blur_logratio") is not None:
        if r["blur_logratio"] < -a.blur_thr:
            f.append("blurry")
        if r["blur_logratio"] > a.blur_thr:
            f.append("oversharp")
    if v("ab_spread_out") is not None and r["ab_spread_out"] < a.mono_spread \
            and ((v("color_dev_frac_in") or 0) > a.color_frac or (v("ab_shift") or 0) > AB_SHIFT_THR):
        f.append("color mismatch")
    if (v("seam_L") or 0) > a.seam_thr or (v("seam_ab") or 0) > a.seam_thr:
        f.append("visible seam")
    return ",".join(f)


def lapvar(crop_bgr):
    g = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    s = 512 / max(g.shape[:2])
    if s != 1:
        g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def region_for(row, bgr):
    """Rebuilds the region used by prepare.py: mask, otherwise face ellipse."""
    if row.get("region_src") == "mask" and isinstance(row.get("mask_path"), str) and row["mask_path"]:
        m = load_mask(row["mask_path"], bgr.shape, bool(row.get("mask_invert", 0)))
        if m is not None and m.any():
            return m, True
    fbox = box_from_row(row, "face")
    if fbox is not None:
        return face_region(bgr.shape, fbox), False
    return None, False


def build_parser():
    ap = argparse.ArgumentParser(description="A. region vs. surroundings coherence",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder created by prepare.py")
    ap.add_argument("--ref", choices=["ring", "frame"], default="ring",
                    help="reference: ring around the region or rest of the frame")
    ap.add_argument("--blur-thr", type=float, default=0.5, help="|blur_logratio| beyond which the flag is raised")
    ap.add_argument("--seam-thr", type=float, default=6.0, help="seam_L / seam_ab beyond which 'visible seam' is raised")
    ap.add_argument("--color-frac", type=float, default=0.10, help="fraction of pixels of another colour for 'color mismatch'")
    ap.add_argument("--mono-spread", type=float, default=8.0,
                    help="ab_spread_out below which the surroundings are considered monochrome")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true", help="recompute everything (e.g. after changing --ref)")
    ap.add_argument("--save-every", type=int, default=25)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.out
    index = load_index(out)
    table = StageTable(out / STAGE_CSV["a"], args.overwrite, args.save_every)

    todo = [r for r in index.to_dict("records") if table.get(r["file"]) is None]
    if args.limit is not None:
        todo = todo[:args.limit]
    print(f"{len(todo)} images to analyse")

    errors = 0
    for i, r in enumerate(todo, 1):
        try:
            row = {"file": r["file"], "ref_mode": args.ref}
            crop = cv2.imread(str(out / r["crop_file"]), cv2.IMREAD_COLOR)
            if crop is not None:
                row["lapvar_crop"] = lapvar(crop)
            if r.get("region_src") in ("mask", "face"):
                bgr = cv2.imread(str(r["image_path"]), cv2.IMREAD_COLOR)
                if bgr is None:
                    raise FileNotFoundError(r["image_path"])
                region, with_seam = region_for(r, bgr)
                if region is not None:
                    row.update(coherence(bgr, region, args.ref, with_seam))
            table.put(row)
            if i % 50 == 0 or i == len(todo):
                print(f"[{i}/{len(todo)}] {r['file']}")
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[error] {r['file']}: {type(e).__name__}: {e}")
            if errors == 1:
                traceback.print_exc()

    # flags recomputed on all the rows with the current thresholds
    for row in table.rows.values():
        row["flags"] = coherence_flags(row, args)
    table.save()

    df = table.df()
    if errors:
        print(f"[info] {errors} images skipped because of errors")
    if not df.empty:
        cnt = df["flags"].fillna("").str.split(",").explode()
        print(f"\nSaved {table.path} ({len(df)} rows)")
        print("flag:", cnt[cnt != ""].value_counts().to_dict())
        cols = [c for c in ("blur_logratio", "ab_shift", "color_dev_frac_in", "ab_spread_out",
                            "seam_L", "seam_ab", "lapvar_crop") if c in df]
        print(df[cols].describe(percentiles=[.1, .5, .9]).T[["count", "mean", "10%", "50%", "90%"]].round(3))


if __name__ == "__main__":
    main()
