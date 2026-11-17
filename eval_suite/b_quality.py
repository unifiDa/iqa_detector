#!/usr/bin/env python3
"""
b_quality.py - B. perceived quality of the crop (no-reference, pyiqa).

  topiq_nr-face  quality of a FACE, learned from human judgements. 0-1, higher = better.
                 Only on crop_kind=face crops, empty by construction otherwise.
  musiq          general multi-scale quality (noise, blur, compression). ~0-100.
  clipiqa        how much the image looks like a "good photo" to CLIP, without a prompt. 0-1.
  (also brisque, niqe, maniqa: for brisque and niqe lower = better)

They are no-reference: compare images with each other and at equal face size (src_px, upscale);
absolute values on archive material are low by nature.

  python b_quality.py --out res_d1
  python b_quality.py --out res_d1 --metrics musiq                 # adds/completes musiq only
"""
import argparse
from pathlib import Path

from common import STAGE_CSV, StageTable, describe, load_index, resolve_device, run_crop_metrics

METRICS = ["topiq_nr-face", "musiq", "clipiqa", "brisque", "niqe", "maniqa"]
FACE_ONLY = {"topiq_nr-face"}


def to_tensor(bgr, device):
    import cv2
    import numpy as np
    import torch
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)


def build_parser():
    ap = argparse.ArgumentParser(description="B. crop quality (pyiqa)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder created by prepare.py")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=METRICS)#["topiq_nr-face", "musiq", "clipiqa"])
    ap.add_argument("--device", default=None, help="cpu/cuda (default: cuda if available)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-every", type=int, default=25)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    import pyiqa
    import torch

    device = resolve_device(args.device)
    index = load_index(args.out)
    table = StageTable(args.out / STAGE_CSV["b"], args.overwrite, args.save_every)
    print(f"device: {device}; metrics: {args.metrics}")
    models = {n: pyiqa.create_metric(n, device=device) for n in args.metrics}

    def score(name, crop, row):
        with torch.no_grad():
            return float(models[name](to_tensor(crop, device)))

    def applicable(name, row):
        return name not in FACE_ONLY or row.get("crop_kind") == "face"

    run_crop_metrics(args.out, index, table, {n: n for n in args.metrics}, score, applicable, args.limit)
    print(f"\nSaved {table.path}")
    describe(table.df(), args.metrics)


if __name__ == "__main__":
    main()
