#!/usr/bin/env python3
"""
prepare.py - passo 0: per ogni immagine trova la regione da valutare e salva il ritaglio.

  - MASCHERA opzionale: se c'e', e' la regione (region_src=mask);
  - VOLTO opzionale (insightface): con maschera si sceglie il volto con piu' area dentro la maschera,
    senza maschera il volto piu' grande; se manca la maschera la regione e' l'ellisse del volto
    (region_src=face);
  - RITAGLIO (crop_kind): face = quadrato attorno al volto, mask = quadrato attorno al bbox della
    maschera, full = immagine intera (lato lungo <= 768). I quadrati sono 512x512.

Scrive <out>/index.csv e <out>/crops/<crop_kind>/<nome>.png. Tutti gli altri script leggono solo questi.
Colonne utili: image_path, mask_path, face_x0..y1, mask_x0..y1, crop_x0..y1 (box del ritaglio in
coordinate dell'immagine originale), src_px (lato della regione in px), upscale (512 / lato del ritaglio).

  python prepare.py --images imgs/ --masks masks/ --out res_d1
  python prepare.py --images imgs2/ --out res_d2                  # senza maschere
  python prepare.py --images imgs3/ --out res_d3 --crop full --no-face-det
"""
import argparse
import traceback
from pathlib import Path

import cv2

from common import (VERSION, INDEX_CSV, StageTable, bbox_of, choose_crop, find_mask, list_images, load_mask)


def onnx_providers(use_gpu):
    """Provider onnxruntime da usare. Con onnxruntime-gpu le librerie CUDA/cuDNN 12 vengono prese dai
    pacchetti nvidia-* installati insieme a torch (preload_dlls, onnxruntime >= 1.21)."""
    import onnxruntime as ort
    available = ort.get_available_providers()
    if use_gpu and "CUDAExecutionProvider" in available:
        try:
            ort.preload_dlls()
        except Exception:  # noqa: BLE001  (versioni vecchie o librerie gia' nel sistema)
            pass
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def load_face_detector(det_thresh, use_gpu=True):
    try:
        from insightface.app import FaceAnalysis
        providers = onnx_providers(use_gpu)
        app = FaceAnalysis(name="buffalo_l", allowed_modules=["detection"], providers=providers)
        app.prepare(ctx_id=0 if providers[0].startswith("CUDA") else -1,
                    det_size=(640, 640), det_thresh=det_thresh)   # soglia bassa: volti piccoli/sgranati
        active = app.models["detection"].session.get_providers()
        print(f"[face-det] insightface su {active[0]}")
        if use_gpu and providers[0].startswith("CUDA") and not active[0].startswith("CUDA"):
            print("[warn] CUDA richiesto ma onnxruntime e' ricaduto sulla CPU (librerie CUDA 12/cuDNN 9 non "
                  "trovate o GPU non supportata): il rilevamento funziona lo stesso, solo piu' lento.")
        return app
    except Exception as e:  # noqa: BLE001
        print(f"[warn] insightface non disponibile ({type(e).__name__}: {e}): niente rilevamento volti.")
        return None


def detect_faces(app, bgr):
    return [(tuple(int(v) for v in f.bbox), float(f.det_score)) for f in app.get(bgr)]


def overlap_in_mask(face_box, mask_box):
    """Quota dell'area del volto che cade dentro il bbox della maschera (0-1)."""
    ix0, iy0 = max(face_box[0], mask_box[0]), max(face_box[1], mask_box[1])
    ix1, iy1 = min(face_box[2], mask_box[2]), min(face_box[3], mask_box[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area = max(1.0, (face_box[2] - face_box[0]) * (face_box[3] - face_box[1]))
    return inter / area


def pick_face(faces, mask_box, min_overlap):
    """Ritorna (bbox, det_score, quota_in_maschera); bbox None se nessun volto adatto."""
    nan = float("nan")
    if mask_box is not None:
        cands = [(overlap_in_mask(b, mask_box), s, b) for b, s in faces]
        cands = [c for c in cands if c[0] >= min_overlap]
        if not cands:
            return None, nan, nan
        ov, s, b = max(cands, key=lambda c: (c[0], c[1]))
        return b, s, ov
    if not faces:
        return None, nan, nan
    b, s = max(faces, key=lambda f: (f[0][2] - f[0][0]) * (f[0][3] - f[0][1]))
    return b, s, nan


def put_box(row, prefix, box):
    for k, v in zip(("x0", "y0", "x1", "y1"), box if box is not None else (None,) * 4):
        row[f"{prefix}_{k}"] = int(v) if v is not None else float("nan")


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--masks", type=Path, default=None, help="cartella delle maschere (opzionale)")
    ap.add_argument("--mask-template", default="{prefix}mask",
                    help="nome della maschera (senza estensione); campi {prefix} e {stem}")
    ap.add_argument("--invert", action="store_true", help="maschera con nero = zona modificata")
    ap.add_argument("--no-face-det", action="store_true", help="salta insightface")
    ap.add_argument("--det-thresh", type=float, default=0.3)
    ap.add_argument("--cpu", action="store_true", help="rilevamento volti su CPU anche se c'e' la GPU")
    ap.add_argument("--min-overlap", type=float, default=0.5,
                    help="quota minima del volto rilevato dentro la maschera (solo con maschera)")
    ap.add_argument("--crop", choices=["auto", "full"], default="auto",
                    help="auto: volto > maschera > intera; full: sempre l'immagine intera")
    ap.add_argument("--limit", type=int, default=None, help="al massimo N nuove immagini")
    ap.add_argument("--overwrite", action="store_true", help="ricalcola tutto (default: riprende)")
    ap.add_argument("--save-every", type=int, default=10)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    print(f"prepare {VERSION}")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    table = StageTable(out / INDEX_CSV, args.overwrite, args.save_every)

    files = [p for p in list_images(args.images) if table.get(p.name) is None]
    if args.limit is not None:
        files = files[:args.limit]
    print(f"{len(files)} immagini nuove da preparare")
    detector = load_face_detector(args.det_thresh, not args.cpu) if files and not args.no_face_det else None

    errors = no_mask = 0
    for i, path in enumerate(files, 1):
        try:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[skip] {path.name}: immagine illeggibile")
                continue
            h, w = bgr.shape[:2]

            mask, mask_path, mask_box = None, None, None
            if args.masks:
                mp = find_mask(args.masks, path.stem, args.mask_template)
                mask = load_mask(mp, bgr.shape, args.invert) if mp else None
                if mask is not None and mask.any():
                    mask_path, mask_box = mp, bbox_of(mask)
                else:
                    mask, no_mask = None, no_mask + 1

            nan = float("nan")
            faces = detect_faces(detector, bgr) if detector else []
            fbox, det, fov = pick_face(faces, mask_box, args.min_overlap) if detector else (None, nan, nan)
            region_src = "mask" if mask is not None else ("face" if fbox is not None else "none")

            crop, kind, cbox, src_px, scale = choose_crop(bgr, fbox, mask_box, args.crop)
            crop_file = Path("crops") / kind / f"{path.stem}.png"
            (out / crop_file).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / crop_file), crop)

            row = {"file": path.name, "image_path": str(path.resolve()),
                   "mask_path": str(Path(mask_path).resolve()) if mask_path else "",
                   "mask_invert": int(args.invert), "width": w, "height": h,
                   "region_src": region_src,
                   "region_area_frac": float(mask.mean()) if mask is not None else nan,
                   "n_faces": len(faces) if detector else nan,
                   "face_found": int(fbox is not None) if detector else nan,
                   "det_score": det, "face_in_mask": fov,
                   "crop_kind": kind, "crop_file": str(crop_file), "src_px": int(src_px), "upscale": float(scale),
                   "face_px": int(src_px) if kind == "face" else nan}
            put_box(row, "face", fbox)
            put_box(row, "mask", mask_box)
            put_box(row, "crop", cbox)
            table.put(row)
            print(f"[{i}/{len(files)}] {path.name}  {kind}/{region_src}")
        except Exception as e:  # noqa: BLE001
            errors += 1
            print(f"[errore] {path.name}: {type(e).__name__}: {e}")
            if errors == 1:
                traceback.print_exc()

    table.save()
    df = table.df()
    if args.masks and no_mask:
        print(f"[info] {no_mask} immagini senza maschera: regione dal volto o nessuna")
    if errors:
        print(f"[info] {errors} immagini saltate per errore")
    if not df.empty:
        print(f"\nSalvato {out / INDEX_CSV} ({len(df)} immagini)")
        print("crop_kind:", df["crop_kind"].value_counts().to_dict())
        print("region_src:", df["region_src"].value_counts().to_dict())


if __name__ == "__main__":
    main()
