#!/usr/bin/env python3
"""
b_quality.py - B. qualita' percepita del ritaglio (no-reference, pyiqa).

  topiq_nr-face  qualita' di un VOLTO, appresa da giudizi umani. 0-1, piu' alto = meglio.
                 Solo sui ritagli crop_kind=face, altrimenti vuoto per costruzione.
  musiq          qualita' generale multi-scala (rumore, sfocatura, compressione). ~0-100.
  clipiqa        quanto l'immagine somiglia a una "foto buona" per CLIP, senza prompt. 0-1.
  (anche brisque, niqe, maniqa: per brisque e niqe piu' basso = meglio)

Sono no-reference: confronta immagini tra loro e a parita' di dimensione del volto (src_px, upscale);
i valori assoluti su materiale d'archivio sono bassi per natura.

  python b_quality.py --out res_d1
  python b_quality.py --out res_d1 --metrics musiq                 # aggiunge/completa solo musiq
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
    ap = argparse.ArgumentParser(description="B. qualita' del ritaglio (pyiqa)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="cartella creata da prepare.py")
    ap.add_argument("--metrics", nargs="+", choices=METRICS, default=["topiq_nr-face", "musiq", "clipiqa"])
    ap.add_argument("--device", default=None, help="cpu/cuda (default: cuda se disponibile)")
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
    print(f"device: {device}; metriche: {args.metrics}")
    models = {n: pyiqa.create_metric(n, device=device) for n in args.metrics}

    def score(name, crop, row):
        with torch.no_grad():
            return float(models[name](to_tensor(crop, device)))

    def applicable(name, row):
        return name not in FACE_ONLY or row.get("crop_kind") == "face"

    run_crop_metrics(args.out, index, table, {n: n for n in args.metrics}, score, applicable, args.limit)
    print(f"\nSalvato {table.path}")
    describe(table.df(), args.metrics)


if __name__ == "__main__":
    main()
