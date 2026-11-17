#!/usr/bin/env python3
"""
e_age.py - E. age estimation of the faces (used by filter_images.py to keep minors out of the dataset).

Evolution of the PyImageSearch age-detection tutorial:

| step  | 2020 (tutorial) | here                                                         |
|-------|-----------------|--------------------------------------------------------------|
| faces | Caffe SSD res10 | YuNet (OpenCV FaceDetectorYN, ONNX)                          |
| age   | Caffe CNN, 8 bins | Hugging Face ViT (9 bins), replaceable with --model        |

Three input modes (output always <out>/e_age.csv, ONE ROW PER FACE):

  default      the full frames listed in <out>/index.csv (image_path): YuNet finds the faces, then the
               age classifier is run on each of them. Best choice: a minor anywhere in the frame is seen.
  --crops      the face crops saved by prepare.py (crops/face): the detector is skipped and each crop is
               classified as a whole. Faster, but images without a face crop get no estimate.
  --input P    stand-alone: P is an image or a folder, index.csv is not needed (add --crops if the
               images are already face crops). "file" is then the base name of each image.

The output is an age BIN with its probability, not a number: for a numeric age use a regression model
(e.g. MiVOLO) via --model, if compatible. Any Hugging Face `image-classification` model whose labels are
age bins works (e.g. a fine-tuned SigLIP2).

Images with no detected face get a row with empty age_bin, so they are not processed again on resume
(rows already present are skipped; --overwrite starts again). The first run downloads YuNet into
models/ and the Hugging Face model (network needed).

  python e_age.py --out res_d1
  python e_age.py --out res_d1 --crops
  python e_age.py --out res_age --input photos/                 # stand-alone
  python e_age.py --out res_d1 --model <hf_repo>                # another age model
"""
import argparse
import traceback
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import AGE_CSV, IMG_EXT, load_index, resolve_device

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_MODEL = "nateraw/vit-age-classifier"
COLUMNS = ["file", "image_path", "x1", "y1", "x2", "y2", "det_score", "age_bin", "age_conf"]
ANNOTATED_DIR = "age_annotated"


def get_yunet(path: Path, score: float) -> cv2.FaceDetectorYN:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading YuNet -> {path}")
        urllib.request.urlretrieve(YUNET_URL, path)
    return cv2.FaceDetectorYN.create(str(path), "", (320, 320), score, 0.3, 5000)


def detect_faces(det: cv2.FaceDetectorYN, bgr: np.ndarray, margin: float):
    """Returns [(x1, y1, x2, y2, score)] with a margin around the face."""
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    out = []
    for f in faces if faces is not None else []:
        x, y, bw, bh = f[:4]
        m = margin * max(bw, bh)
        out.append((
            int(max(0, x - m)), int(max(0, y - m)),
            int(min(w, x + bw + m)), int(min(h, y + bh + m)),
            float(f[-1]),
        ))
    return out


def predict(model, proc, device, crops):
    """Top-1 (label, probability) for each PIL crop."""
    import torch
    with torch.inference_mode():
        inputs = proc(images=crops, return_tensors="pt").to(device)
        probs = model(**inputs).logits.softmax(-1).cpu()
    id2label = model.config.id2label
    return [(id2label[int(p.argmax())], float(p.max())) for p in probs]


def list_inputs(path: Path):
    if path.is_dir():
        return sorted(p for p in path.rglob("*") if p.suffix.lower() in IMG_EXT)
    return [path]


def collect_jobs(args):
    """[(file, path_to_read)]: what to process, depending on the input mode."""
    if args.input:
        return [(p.name, p) for p in list_inputs(args.input)]
    index = load_index(args.out)
    if args.crops:
        faces = index[index["crop_kind"] == "face"]
        return [(r.file, args.out / r.crop_file) for r in faces.itertuples()]
    return [(r.file, Path(r.image_path)) for r in index.itertuples()]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, type=Path,
                    help="results folder (created by prepare.py); e_age.csv is written here")
    ap.add_argument("--input", type=Path, default=None,
                    help="stand-alone mode: an image or a folder, instead of the images in index.csv")
    ap.add_argument("--crops", action="store_true",
                    help="inputs are face crops (no detection): crops/face of index.csv, or --input as is")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face age-classification model")
    ap.add_argument("--yunet", type=Path, default=Path("models/yunet.onnx"))
    ap.add_argument("--det-score", type=float, default=0.7, help="minimum YuNet face score")
    ap.add_argument("--margin", type=float, default=0.15, help="margin around the face bbox (fraction)")
    ap.add_argument("--device", default=None, help="cpu/cuda (default: cuda if available)")
    ap.add_argument("--batch", type=int, default=16, help="faces per classifier batch")
    ap.add_argument("--no-draw", action="store_true", help=f"do not save annotated images ({ANNOTATED_DIR}/)")
    ap.add_argument("--limit", type=int, default=None, help="at most N new images")
    ap.add_argument("--overwrite", action="store_true", help="recompute everything (default: resume)")
    ap.add_argument("--save-every", type=int, default=50)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModelForImageClassification

    args.out.mkdir(parents=True, exist_ok=True)
    csv_path = args.out / AGE_CSV
    rows, done = [], set()
    if csv_path.exists() and not args.overwrite:
        prev = pd.read_csv(csv_path)
        rows, done = prev.to_dict("records"), set(prev["file"])
        print(f"[resume] {len(done)} images already present in {csv_path}")

    jobs = [j for j in collect_jobs(args) if j[0] not in done]
    if args.limit is not None:
        jobs = jobs[:args.limit]
    print(f"{len(jobs)} new images to analyse")
    if not jobs:
        if args.crops and not args.input:
            print("[info] nothing to do (--crops only uses the crop_kind=face rows of index.csv)")
        return

    device = resolve_device(args.device)
    proc = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForImageClassification.from_pretrained(args.model).to(device).eval()
    detector = None if args.crops else get_yunet(args.yunet, args.det_score)
    draw = not args.crops and not args.no_draw
    if draw:
        (args.out / ANNOTATED_DIR).mkdir(exist_ok=True)

    def save():
        pd.DataFrame(rows, columns=COLUMNS).to_csv(csv_path, index=False)

    errors = 0
    try:
        for i, (name, path) in enumerate(jobs, 1):
            try:
                bgr = cv2.imread(str(path))
                if bgr is None:
                    print(f"[skip] {name}: unreadable image ({path})")
                    continue
                h, w = bgr.shape[:2]
                boxes = [(0, 0, w, h, 1.0)] if args.crops else detect_faces(detector, bgr, args.margin)
                if not boxes:
                    rows.append({"file": name, "image_path": str(path.resolve())})     # no face: empty estimate
                    continue

                pil = [Image.fromarray(cv2.cvtColor(bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
                       for x1, y1, x2, y2, _ in boxes]
                preds = []
                for k in range(0, len(pil), args.batch):
                    preds += predict(model, proc, device, pil[k:k + args.batch])

                for (x1, y1, x2, y2, s), (label, conf) in zip(boxes, preds):
                    rows.append({"file": name, "image_path": str(path.resolve()),
                                 "x1": x1, "y1": y1, "x2": x2, "y2": y2, "det_score": round(s, 3),
                                 "age_bin": label, "age_conf": round(conf, 4)})
                    if draw:
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(bgr, f"{label} {conf:.0%}", (x1, max(15, y1 - 8)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                if draw:
                    cv2.imwrite(str(args.out / ANNOTATED_DIR / name), bgr)
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"[error] {name}: {type(e).__name__}: {e}")
                if errors == 1:
                    traceback.print_exc()
            if i % 50 == 0 or i == len(jobs):
                print(f"[{i}/{len(jobs)}] {name}")
            if args.save_every and i % args.save_every == 0:
                save()
    finally:
        save()                                    # also on Ctrl+C or error

    df = pd.DataFrame(rows, columns=COLUMNS)
    if errors:
        print(f"[info] {errors} images skipped because of errors")
    print(f"\nSaved {csv_path} ({df['file'].nunique()} images, {int(df['age_bin'].notna().sum())} faces, "
          f"{int(df.groupby('file')['age_bin'].apply(lambda s: s.notna().sum() == 0).sum())} images without faces)")
    print("age_bin:", df["age_bin"].value_counts().to_dict())


if __name__ == "__main__":
    main()
