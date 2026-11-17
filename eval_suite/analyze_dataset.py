#!/usr/bin/env python3
"""
analyze_dataset.py - describes the final dataset: resolution and the range of every quality metric.

The dataset is the set of images in <out>/filtered/kept.csv (what filter_images.py / reject_images.py left); use
--table to describe another set, e.g. discarded.csv. The numbers are read from <out>/results.csv (merge.py), so
metrics computed after the filter are included.

Writes, in <dest> (default <table folder>/analysis):
  report.md          the whole analysis as text
  metrics.csv        one row per metric: n, min, p05, p25, median, p75, p95, max, mean, std (+ which direction is better)
  resolution.csv     the most frequent frame resolutions
  summary.json       everything above, machine readable
  metrics_hist.png   histogram of every metric (median dashed, 5-95% range shaded)
  resolution.png     frame size, megapixels and face size

RESOLUTION
  frame        width x height of the image the region was found in (the frame given to prepare.py)
  region side  src_px: side in px of the face (crop_kind=face) or of the mask bbox (mask) or of the image (full)
  upscale      how much the 512 px crop is enlarged with respect to the source: > 1 means the crop is interpolated
QUALITY: per metric the min-max range and the percentiles. Metrics are no-reference and only comparable across
images of similar resolution, so use --group-by crop_kind (or any other column) to also get the ranges per group.

  python analyze_dataset.py --out res_d1
  python analyze_dataset.py --out res_d1 --group-by crop_kind
  python analyze_dataset.py --out res_d1 --table res_d1/filtered/discarded.csv --dest res_d1/analysis_discarded
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import FILTER_DIR, RESULTS_CSV  # noqa: E402
from merge import SKIP_STATS  # noqa: E402

LOWER_IS_BETTER = {"brisque", "niqe"}
HIGHER_IS_BETTER = {"topiq_nr-face", "musiq", "clipiqa", "maniqa", "clip_score", "mps_score"}
RESOLUTION_COLS = {"width", "height", "src_px", "face_px", "upscale", "region_area_frac", "n_faces"}
NOT_METRICS = {"n_est", "minor", "face_found", "mask_invert"}
QUANT = {"p05": .05, "p25": .25, "median": .5, "p75": .75, "p95": .95}
ORDER = ["topiq_nr-face", "musiq", "clipiqa", "brisque", "niqe", "maniqa", "clip_score", "mps_score"]


def direction(metric):
    return "lower = better" if metric in LOWER_IS_BETTER else "higher = better" if metric in HIGHER_IS_BETTER else ""


def stats(x):
    x = pd.to_numeric(x, errors="coerce").dropna()
    if x.empty:
        return None
    s = {"n": int(len(x)), "min": float(x.min())}
    s.update({k: float(x.quantile(q)) for k, q in QUANT.items()})
    s.update({"max": float(x.max()), "mean": float(x.mean()), "std": float(x.std()) if len(x) > 1 else 0.0})
    return s


def metric_columns(df):
    cols = [c for c in df.select_dtypes("number").columns
            if c not in SKIP_STATS and c not in RESOLUTION_COLS and c not in NOT_METRICS and df[c].notna().any()]
    return sorted(cols, key=lambda c: (ORDER.index(c) if c in ORDER else len(ORDER), c))


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return ""
    if isinstance(v, (float, np.floating)):
        return f"{v:.0f}" if abs(v) >= 100 else f"{v:.{nd}g}"       # 1920 px, not 1.92e+03
    return str(v)


def md_table(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) if not isinstance(v, str) else v for v in row) + " |")
    return "\n".join(lines)


def resolution_info(df, top):
    r = {"n_images": int(len(df))}
    out = {}
    if {"width", "height"} <= set(df.columns):
        w, h = df["width"].astype(float), df["height"].astype(float)
        mp = (w * h) / 1e6
        res = (w.astype(int).astype(str) + "x" + h.astype(int).astype(str))
        counts = res.value_counts()
        table = pd.DataFrame({"resolution": counts.index, "n": counts.values,
                              "percent": (100 * counts.values / len(df)).round(1)}).head(top)
        out["frame_resolutions"] = table
        r["n_distinct_resolutions"] = int(counts.size)
        r["width"], r["height"], r["megapixels"] = stats(w), stats(h), stats(mp)
        r["short_side"], r["long_side"] = stats(np.minimum(w, h)), stats(np.maximum(w, h))
        ar = (w / h).round(2).value_counts().head(5)
        r["aspect_ratio_top"] = {str(k): int(v) for k, v in ar.items()}
    if "crop_kind" in df:
        r["crop_kind"] = {str(k): int(v) for k, v in df["crop_kind"].value_counts().items()}
    for c in ("src_px", "face_px", "upscale", "region_area_frac"):
        if c in df and df[c].notna().any():
            r[c] = stats(df[c])
    if "src_px" in df and df["src_px"].notna().any():
        b = pd.cut(df["src_px"], [0, 64, 128, 256, 512, np.inf], right=False,
                   labels=["<64", "64-127", "128-255", "256-511", ">=512"])
        r["region_side_buckets"] = {str(k): int(v) for k, v in b.value_counts(sort=False).items()}
    if "upscale" in df and df["upscale"].notna().any():
        u = df["upscale"].dropna()
        r["share_upscaled"] = float((u > 1.0001).mean())
        r["share_upscaled_over_2x"] = float((u > 2).mean())
    return r, out


def plot_metrics(df, cols, path):
    if not cols:
        return
    n = len(cols)
    ncol = min(4, n)
    nrow = -(-n // ncol)
    fig, axs = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow), squeeze=False)
    for ax, c in zip(axs.ravel(), cols):
        x = df[c].dropna()
        ax.hist(x, bins=40, color="#7a8aa0")
        lo, hi, med = x.quantile(.05), x.quantile(.95), x.median()
        ax.axvspan(lo, hi, color="#d1495b", alpha=.10)
        ax.axvline(med, color="black", ls="--", lw=1)
        ax.set_title(f"{c}  (n={len(x)})\nmedian {med:.3g}, 5-95%: {lo:.3g} .. {hi:.3g}", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_yticks([])
    for ax in axs.ravel()[n:]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_resolution(df, path):
    panels = []
    if {"width", "height"} <= set(df.columns):
        panels.append("frame")
    if "src_px" in df and df["src_px"].notna().any():
        panels.append("region")
    if not panels:
        return
    fig, axs = plt.subplots(1, len(panels) + (1 if "frame" in panels else 0), figsize=(4.2 * (len(panels) + 1), 3.6),
                            squeeze=False)
    axs = list(axs[0])
    if "frame" in panels:
        ax = axs.pop(0)
        g = df.groupby(["width", "height"]).size().reset_index(name="n")
        ax.scatter(g["width"], g["height"], s=10 + 400 * g["n"] / g["n"].max(), alpha=.6, color="#7a8aa0")
        ax.set_xlabel("width (px)"); ax.set_ylabel("height (px)"); ax.set_title("frame size (area = n images)", fontsize=9)
        ax = axs.pop(0)
        ax.hist(df["width"] * df["height"] / 1e6, bins=40, color="#7a8aa0")
        ax.set_xlabel("megapixels"); ax.set_yticks([]); ax.set_title("frame megapixels", fontsize=9)
    if "region" in panels:
        ax = axs.pop(0)
        ax.hist(df["src_px"].dropna(), bins=40, color="#7a8aa0")
        ax.set_xlabel("region side in the source (px)"); ax.set_yticks([]); ax.set_title("face / mask size", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True, help="results folder (reads results.csv and filtered/kept.csv)")
    ap.add_argument("--table", type=Path, default=None, help="images to describe (default: <out>/filtered/kept.csv)")
    ap.add_argument("--dest", type=Path, default=None, help="output folder (default: <table folder>/analysis)")
    ap.add_argument("--group-by", nargs="*", default=[], help="also compute the metric ranges per value of these columns")
    ap.add_argument("--top", type=int, default=15, help="how many frame resolutions to list")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    table = a.table or a.out / FILTER_DIR / "kept.csv"
    results = a.out / RESULTS_CSV
    for p in (table, results):
        if not p.exists():
            sys.exit(f"{p} not found (run merge.py and filter_images.py first)")
    sel = pd.read_csv(table)
    df = pd.read_csv(results)
    df = df[df["file"].isin(set(sel["file"]))].copy()
    extra = [c for c in ("p_minor_max", "n_est") if c in sel and c not in df]
    if extra:
        df = df.merge(sel[["file"] + extra], on="file", how="left")
    if df.empty:
        sys.exit(f"None of the {len(sel)} images of {table} is in {results}.")
    dest = a.dest or table.parent / "analysis"
    dest.mkdir(parents=True, exist_ok=True)

    res, res_tables = resolution_info(df, a.top)
    cols = metric_columns(df)
    rows = []
    for c in cols:
        s = stats(df[c])
        rows.append({"metric": c, "direction": direction(c), **s})
    mt = pd.DataFrame(rows)
    mt.to_csv(dest / "metrics.csv", index=False)
    if "frame_resolutions" in res_tables:
        res_tables["frame_resolutions"].to_csv(dest / "resolution.csv", index=False)
    groups = {}
    for g in a.group_by:
        if g not in df:
            print(f"[warn] --group-by {g}: no such column, skipped")
            continue
        gr = [{"group": f"{g}={k}", "metric": c, **s} for k, sub in df.groupby(g) for c in cols
              if (s := stats(sub[c])) is not None]
        pd.DataFrame(gr).to_csv(dest / f"metrics_by_{g}.csv", index=False)
        groups[g] = gr

    flags = {}
    if "flags" in df:
        cnt = df["flags"].fillna("").astype(str).str.split(",").explode()
        flags = {str(k): int(v) for k, v in cnt[cnt != ""].value_counts().items()}

    summary = {"table": str(table), "n_images": int(len(df)), "resolution": res,
               "metrics": {r["metric"]: {k: v for k, v in r.items() if k != "metric"} for r in rows}, "flags": flags}
    (dest / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    plot_metrics(df, cols, dest / "metrics_hist.png")
    plot_resolution(df, dest / "resolution.png")

    # ---- report ----------------------------------------------------------------------------------
    L = [f"# Dataset analysis: {a.out.name}", "", f"{len(df)} images from `{table}`.", "", "## Resolution", ""]
    if "megapixels" in res:
        f = lambda k, key: fmt(res[k][key])  # noqa: E731
        L += [f"- frame width {f('width', 'min')}-{f('width', 'max')} px (median {f('width', 'median')}), "
              f"height {f('height', 'min')}-{f('height', 'max')} px (median {f('height', 'median')})",
              f"- megapixels {f('megapixels', 'min')}-{f('megapixels', 'max')} (median {f('megapixels', 'median')})",
              f"- {res['n_distinct_resolutions']} distinct frame resolutions; most common aspect ratios: {res['aspect_ratio_top']}"]
    if "crop_kind" in res:
        L.append(f"- crops: {res['crop_kind']}")
    if "src_px" in res:
        L.append(f"- region side in the source (face / mask / image): {fmt(res['src_px']['min'])}-{fmt(res['src_px']['max'])} px, "
                 f"median {fmt(res['src_px']['median'])}; buckets {res['region_side_buckets']}")
    if "share_upscaled" in res:
        L.append(f"- crops enlarged to 512 px: {100 * res['share_upscaled']:.1f}% of the images "
                 f"({100 * res['share_upscaled_over_2x']:.1f}% by more than 2x); upscale median {fmt(res['upscale']['median'])}")
    if "frame_resolutions" in res_tables:
        L += ["", f"Most frequent frame resolutions (top {a.top}):", "", md_table(res_tables["frame_resolutions"])]
    L += ["", "## Quality metrics", "",
          "Range and percentiles of every metric over this set. No-reference metrics are only comparable between "
          "images of similar resolution (see the groups below, if requested).", ""]
    if not mt.empty:
        L.append(md_table(mt[["metric", "direction", "n", "min", "p05", "p25", "median", "p75", "p95", "max", "mean", "std"]]))
    for g, gr in groups.items():
        L += ["", f"### By {g}", "", md_table(pd.DataFrame(gr)[["group", "metric", "n", "min", "median", "max", "mean"]])]
    if flags:
        L += ["", "## Flags", "", ", ".join(f"{k}: {v}" for k, v in flags.items())]
    L += ["", "## Files", "", "`metrics.csv`, `resolution.csv`, `summary.json`, `metrics_hist.png`, `resolution.png`", ""]
    (dest / "report.md").write_text("\n".join(L), encoding="utf-8")

    print("\n".join(L[:L.index("## Quality metrics")]))
    print(mt[["metric", "n", "min", "median", "max"]].round(3).to_string(index=False) if not mt.empty else "no metrics found")
    print(f"\n-> {dest}/ (report.md, metrics.csv, resolution.csv, summary.json, metrics_hist.png, resolution.png)")


if __name__ == "__main__":
    main()