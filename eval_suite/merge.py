#!/usr/bin/env python3
"""
merge.py - merges index.csv, a_coherence.csv, b_quality.csv, c_prompt.csv (on "file") and d_dataset.json
into <out>/results.csv and <out>/summary.json. It can be run at any time, even with missing stages.
(e_age.csv has one row per face, not per image, so it is not merged here: filter_images.py combines it.)

  python merge.py --out res_d1 --name dataset1
"""
import argparse
import json
from pathlib import Path

from common import D_JSON, RESULTS_CSV, VERSION, describe, load_merged

SKIP_STATS = {"width", "height", "mask_invert", "face_x0", "face_y0", "face_x1", "face_y1",
              "mask_x0", "mask_y0", "mask_x1", "mask_y1", "crop_x0", "crop_y0", "crop_x1", "crop_y1",
              *[f"frame_{side}_pad{suffix}" for side in ("top", "bottom", "left", "right") for suffix in ("", "_frac")]}


def summarize(df, name, d_entries):
    s = {"name": name, "version": VERSION, "n_images": int(len(df))}
    for c in ("crop_kind", "region_src", "ref_used"):
        if c in df:
            s[c] = {str(k): int(v) for k, v in df[c].value_counts().items()}
    s["metrics"] = {}
    for c in df.select_dtypes("number").columns:
        if c in SKIP_STATS:
            continue
        x = df[c].dropna()
        if len(x):
            s["metrics"][c] = {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
                               "p10": float(x.quantile(.1)), "p90": float(x.quantile(.9))}
    if "flags" in df:
        cnt = df["flags"].fillna("").astype(str).str.split(",").explode()
        s["flags"] = {str(k): int(v) for k, v in cnt[cnt != ""].value_counts().items()}
    if d_entries:
        s["fid_kid"] = d_entries
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Merges the results of stages A-D")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default=None, help="dataset name (default: folder name)")
    args = ap.parse_args(argv)

    df = load_merged(args.out)
    d_path = args.out / D_JSON
    d_entries = json.loads(d_path.read_text()) if d_path.exists() else []
    df.to_csv(args.out / RESULTS_CSV, index=False)
    summary = summarize(df, args.name or args.out.name, d_entries)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"Saved {args.out / RESULTS_CSV} ({len(df)} images) and {args.out / 'summary.json'}")
    for c in ("crop_kind", "region_src"):
        if c in summary:
            print(f"{c}: {summary[c]}")
    describe(df, list(summary["metrics"]))
    if summary.get("flags"):
        print("\nFlags:", summary["flags"])
    for e in d_entries:
        print(f"\nFID {e['fid']:.2f}  KID {e['kid']:.4f}  ({e['kind']}: {e['n_gen']} vs {e['n_ref']}, ref {e['ref_dir']})")


if __name__ == "__main__":
    main()