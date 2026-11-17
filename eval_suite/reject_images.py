#!/usr/bin/env python3
"""
reject_images.py - updates kept.csv / discarded.csv with the images you rejected by eye.

Give it a text file with the names of the images to discard, one per line (blank lines and lines starting
with # are ignored). A name can be the file name, the name without extension or a full path, so you can paste
lines from kept.txt (export_images.py) or from the file names you read in the PDF. The matching rows move from
kept.csv to discarded.csv with reason "manual review".

The names are also appended to <dest>/manual_rejects.txt, and filter_images.py applies that file every time it
runs: re-running the filter (e.g. after a new metric or a new threshold) does NOT bring those images back.
To bring an image back, delete its line from manual_rejects.txt and re-run filter_images.py.

  python reject_images.py --out res_d1 --list rejected.txt
  python reject_images.py --out res_d1 --names frame_000755_track001_inpainted-0.png frame_000812_track002_inpainted-0
  python reject_images.py --out res_d1 --list rejected.txt --dry-run         # only show what would change

After it, re-run export_images.py to refresh kept.pdf / kept.txt, and analyze_dataset.py for the statistics.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from common import FILTER_DIR, MANUAL_REASON, MANUAL_REJECTS, read_name_list


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help=f"results folder (the tables are in <out>/{FILTER_DIR}/)")
    ap.add_argument("--dest", type=Path, default=None,
                    help="folder with kept.csv / discarded.csv (default: <out>/filtered)")
    ap.add_argument("--list", type=Path, default=None, help="text file with the names to discard, one per line")
    ap.add_argument("--names", nargs="*", default=[], help="names to discard, on the command line")
    ap.add_argument("--dry-run", action="store_true", help="show what would change without writing anything")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    dest = a.dest or (a.out / FILTER_DIR if a.out else None)
    if dest is None:
        ap.error("give --out or --dest")
    if not a.list and not a.names:
        ap.error("give --list and/or --names")
    kept_p, disc_p, rej_p = dest / "kept.csv", dest / "discarded.csv", dest / MANUAL_REJECTS
    for p in (kept_p, disc_p):
        if not p.exists():
            sys.exit(f"{p} not found: run filter_images.py first")

    entries = ([] if not a.list else read_name_list(a.list)) + [Path(n).name for n in a.names]
    entries = list(dict.fromkeys(entries))                          # no duplicates, order kept
    if not entries:
        sys.exit("The list of names is empty.")

    kept, disc = pd.read_csv(kept_p), pd.read_csv(disc_p)
    kept_stem, disc_stem = kept["file"].map(lambda f: Path(f).stem), disc["file"].map(lambda f: Path(f).stem)

    moved_names, already, unknown = [], [], []
    for e in entries:
        s = Path(e).stem
        if (kept_stem == s).any():
            moved_names.append(e)
        elif (disc_stem == s).any():
            already.append(e)
        else:
            unknown.append(e)

    hit = kept_stem.isin({Path(e).stem for e in moved_names})
    moved = kept[hit].assign(reason=MANUAL_REASON)
    print(f"{len(entries)} names | {len(moved)} images moved from kept to discarded | "
          f"{len(already)} already discarded | {len(unknown)} not found")
    if unknown:
        print("  not found in kept.csv nor discarded.csv (typo? different dataset?):")
        for e in unknown[:20]:
            print(f"    {e}")
        if len(unknown) > 20:
            print(f"    ... and {len(unknown) - 20} more")
    if a.dry_run:
        print("[dry-run] nothing written")
        return

    pd.concat([disc, moved], ignore_index=True).to_csv(disc_p, index=False)
    kept[~hit].to_csv(kept_p, index=False)

    # remember them, so that re-running filter_images.py keeps them discarded
    known = read_name_list(rej_p) if rej_p.exists() else []
    new = [e for e in moved_names + already if e not in known]
    if new or not rej_p.exists():
        with rej_p.open("a", encoding="utf-8") as fh:
            fh.writelines(f"{e}\n" for e in new)
    print(f"kept {len(kept) - len(moved)} | discarded {len(disc) + len(moved)}")
    print(f"-> {kept_p}, {disc_p}; {len(new)} new names added to {rej_p}")


if __name__ == "__main__":
    main()