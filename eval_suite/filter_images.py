#!/usr/bin/env python3
"""
filter_images.py - keeps the colour images without minors: the last step of the pipeline.

Inputs (all in --out, unless overridden):
  results.csv   merged table written by merge.py (needs the column color_dev_frac_in from stage A)
  e_age.csv     age estimates written by e_age.py (columns file, age_bin, age_conf[, p_minor])
The two tables are linked on the file base name.

A row is kept only if ALL these conditions are true:
  * color_dev_frac_in <= --max-color-dev (fraction of colour deviation)
  * at least one face has an age estimate (otherwise a minor cannot be ruled out)
  * no face has a bin in --minor-bins, nor p_minor >= --max-p-minor
With --by-frame a minor face also discards all the other rows of the same frame
(frame = the "frame_<number>" prefix of the file name).

Writes <dest>/kept.csv and <dest>/discarded.csv (default dest: <out>/filtered), the latter with the
discard `reason`. Look at them with eval_suite/viewer_filter.ipynb.

  python filter_images.py --out res_d1 --by-frame
  python filter_images.py --out res_d1 --max-color-dev 0.02 --dest res_d1/filtered_strict
  python filter_images.py --meta other/results.csv --ages other/e_age.csv --dest filtered/
"""
import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from common import AGE_CSV, FILTER_DIR, RESULTS_CSV

DEFAULT_MINOR_BINS = ["0-2", "3-9", "10-19"]


def frame_id(name: str) -> str:
    m = re.match(r"(frame_\d+)", name)
    return m.group(1) if m else name


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None,
                    help=f"results folder: reads {RESULTS_CSV} and {AGE_CSV} from it")
    ap.add_argument("--meta", type=Path, default=None, help=f"metadata table (default: <out>/{RESULTS_CSV})")
    ap.add_argument("--ages", type=Path, default=None, help=f"age table (default: <out>/{AGE_CSV})")
    ap.add_argument("--dest", type=Path, default=None, help=f"destination folder (default: <out>/{FILTER_DIR})")
    ap.add_argument("--max-color-dev", type=float, default=0.0,
                    help="discard if color_dev_frac_in > threshold (default 0: null values only)")
    ap.add_argument("--minor-bins", nargs="+", default=DEFAULT_MINOR_BINS)
    ap.add_argument("--max-p-minor", type=float, default=0.2,
                    help="discard if p_minor >= threshold (ignored if the column is missing)")
    ap.add_argument("--by-frame", action="store_true",
                    help="a minor discards the whole frame, not just the row")
    ap.add_argument("--keep-no-face", action="store_true",
                    help="also keep the rows without an estimated face (not recommended)")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    meta_path = a.meta or (a.out / RESULTS_CSV if a.out else None)
    ages_path = a.ages or (a.out / AGE_CSV if a.out else None)
    dest = a.dest or (a.out / FILTER_DIR if a.out else Path("filtered"))
    if meta_path is None or ages_path is None:
        sys.exit("Give --out, or both --meta and --ages.")
    for p in (meta_path, ages_path):
        if not p.exists():
            sys.exit(f"{p} not found (pipeline order: prepare, coherence, e_age, merge, filter_images)")

    meta = pd.read_csv(meta_path)
    ages = pd.read_csv(ages_path)
    if "color_dev_frac_in" not in meta:
        sys.exit(f"{meta_path} has no color_dev_frac_in column: run stage A (coherence) and merge first.")
    ages["key"] = ages["file"].map(lambda p: Path(p).name)
    meta["key"] = meta["file"].map(lambda p: Path(p).name)
    if "p_minor" not in ages:
        ages["p_minor"] = float("nan")

    # per-image state: worst case among the estimated faces (rows without a face have an empty age_bin)
    minor_bins = {b.replace(" ", "") for b in a.minor_bins}
    ages["is_minor_bin"] = ages["age_bin"].astype(str).str.replace(" ", "").isin(minor_bins)
    ages["is_minor_p"] = ages["p_minor"].ge(a.max_p_minor)
    ages["is_minor"] = ages["is_minor_bin"] | ages["is_minor_p"]
    ages["has_est"] = ages["age_bin"].notna()
    per_img = ages.groupby("key").agg(n_est=("has_est", "sum"), minor=("is_minor", "any"),
                                      p_minor_max=("p_minor", "max"))
    df = meta.merge(per_img, on="key", how="left")
    if df["n_est"].isna().all():
        sys.exit(f"No file of {meta_path} matches a file of {ages_path}: check that the two tables "
                 f"refer to the same images (they are linked on the file base name).")
    df["n_est"] = df["n_est"].fillna(0).astype(int)
    df["minor"] = df["minor"].fillna(False).astype(bool)

    if a.by_frame:
        df["frame"] = df["key"].map(frame_id)
        df["minor"] = df.groupby("frame")["minor"].transform("any")

    reason = pd.Series("", index=df.index)
    reason[df.color_dev_frac_in.isna()] = "missing color_dev_frac_in"
    reason[(reason == "") & (df.color_dev_frac_in > a.max_color_dev)] = "color deviation"
    reason[(reason == "") & df.minor] = "possible minor"
    if not a.keep_no_face:
        reason[(reason == "") & (df.n_est == 0)] = "no face estimated"
    df["reason"] = reason

    dest.mkdir(parents=True, exist_ok=True)
    keep, drop = df[df.reason == ""], df[df.reason != ""]
    keep.to_csv(dest / "kept.csv", index=False)
    drop.to_csv(dest / "discarded.csv", index=False)
    print(f"total {len(df)} | kept {len(keep)} | discarded {len(drop)}")
    print(drop.reason.value_counts().to_string())
    print(f"-> {dest}/kept.csv, discarded.csv")


if __name__ == "__main__":
    main()
