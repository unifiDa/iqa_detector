#!/usr/bin/env python3
"""
d_dataset.py - D. DATASET-level metrics with clean-fid (they say nothing about a single image).

  FID  Frechet Inception Distance between the Inception feature distributions of the two sets.
       Lower = more similar. Unstable with fewer than ~2048 images per set.
  KID  Kernel Inception Distance: like FID but unbiased with few samples; preferable below
       a few thousand images.

Compare against a SENSIBLE reference: e.g. crops of real faces from the same archive
(by also running prepare.py on the originals), not colour, high-resolution FFHQ.

Results accumulate in <out>/d_dataset.json (one entry per crops/reference pair;
re-running with the same reference replaces the entry).

  python d_dataset.py --out res_d1 --ref res_originali/crops/face
  python d_dataset.py --out res_d1 --ref res_d2/crops/face --kind face
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from common import D_JSON, INDEX_CSV, list_images, resolve_device


def fid_kid(gen_dir: Path, ref_dir: Path, device, batch_size):
    import torch
    from cleanfid import fid
    n_gen, n_ref = len(list_images(gen_dir)), len(list_images(ref_dir))
    res = {"gen_dir": str(gen_dir.resolve()), "ref_dir": str(ref_dir.resolve()), "n_gen": n_gen, "n_ref": n_ref}
    kw = dict(mode="clean", device=torch.device(device), batch_size=batch_size, num_workers=0)
    res["kid"] = float(fid.compute_kid(str(gen_dir), str(ref_dir), **kw))
    res["fid"] = float(fid.compute_fid(str(gen_dir), str(ref_dir), **kw))
    if min(n_gen, n_ref) < 2048:
        res["warning"] = ("FID is unreliable with fewer than ~2048 images per set "
                          "(2048-dimensional covariance): prefer KID.")
    return res


def build_parser():
    ap = argparse.ArgumentParser(description="D. FID / KID with clean-fid",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder created by prepare.py")
    ap.add_argument("--ref", required=True, type=Path, help="folder of reference images")
    ap.add_argument("--kind", choices=["face", "mask", "full"], default=None,
                    help="which crops to compare (default: the most frequent kind in index.csv)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    kind = args.kind
    if kind is None:
        idx = args.out / INDEX_CSV
        kind = str(pd.read_csv(idx)["crop_kind"].value_counts().idxmax()) if idx.exists() else "face"
    gen_dir = args.out / "crops" / kind
    if not gen_dir.is_dir() or not list_images(gen_dir):
        raise SystemExit(f"No crops in {gen_dir}")

    device = resolve_device(args.device)
    print(f"[clean-fid] {gen_dir} against {args.ref} ({device}) ...")
    res = dict(kind=kind, **fid_kid(gen_dir, args.ref, device, args.batch))
    print(json.dumps(res, indent=2, ensure_ascii=False))

    path = args.out / D_JSON
    entries = json.loads(path.read_text()) if path.exists() else []
    entries = [e for e in entries if not (e.get("gen_dir") == res["gen_dir"] and e.get("ref_dir") == res["ref_dir"])]
    entries.append(res)
    path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
