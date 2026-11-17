"""
common.py - helpers shared by the evaluation scripts.

Pipeline (every script writes into --out and the tables are linked by the "file" column):

  prepare.py        detect face/mask, save the crops            -> index.csv, crops/<kind>/
  a_coherence.py    A. blur / Lab tint / seam                   -> a_coherence.csv
  b_quality.py      B. topiq_nr-face, musiq, clipiqa (pyiqa)    -> b_quality.csv
  c_prompt.py       C. clip_score, mps_score (needs prompts)    -> c_prompt.csv
  d_dataset.py      D. FID / KID with clean-fid                 -> d_dataset.json
  e_age.py          E. age estimation of the faces              -> e_age.csv   (one row per face)
  merge.py          merges everything                           -> results.csv, summary.json
  filter_images.py  keeps colour images without minors          -> filtered/kept.csv, filtered/discarded.csv

A, B, C, D and E only read index.csv and the crops (E can also read the full frames): they can be run in
any order, on different machines, and re-run without redoing the work already done (rows already present
are skipped and only the missing columns are computed; --overwrite starts the stage again).
"""
import atexit
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VERSION = "v3.0"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MASK_SIZE = ("large", "medium", "small")      # tuple: the order is the priority (with a set it changed at every run)
CROP_SIZE = 512

INDEX_CSV = "index.csv"
STAGE_CSV = {"a": "a_coherence.csv", "b": "b_quality.csv", "c": "c_prompt.csv", "p": "padding.csv"}
D_JSON = "d_dataset.json"
AGE_CSV = "e_age.csv"                          # one row per detected face, so it is NOT in STAGE_CSV / load_merged
RESULTS_CSV = "results.csv"
FILTER_DIR = "filtered"


# ----------------------------------------------------------------------------
# Basic I/O
# ----------------------------------------------------------------------------
def list_images(d: Path):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)


def find_mask(mask_dir: Path, stem: str, template: str):
    """Image 'frame_000755_track001_inpainted-0.png' -> mask 'frame_000755_track001_mask.<ext>'
    with the default template '{prefix}mask'. {prefix} = part of the name before 'inpainted'
    (name + '_' if missing), {stem} = full name without extension."""
    idx = stem.find("inpainted")
    prefix = stem[:idx] if idx > 0 else stem + "_"
    base = template.format(stem=stem, prefix=prefix)
    for ext in IMG_EXT:
        p = Path(mask_dir) / f"{base}{ext}"
        if p.exists():
            return p
    return None



def find_mask_generic(mask_dir: Path, stem: str, prefix="mask_"):
    """'size' scheme: the mask is called <prefix><name up to and including large|medium|small>.<ext>.
    E.g. 'Agfa_DC-504_0_10.JPG_large_0_inpainted-0.png' -> 'mask_Agfa_DC-504_0_10.JPG_large.png'. Returns None
    if the name contains none of the three words or if the file does not exist."""
    for word in MASK_SIZE:
        idx = stem.find(word)
        if idx >= 0:
            base = prefix + stem[:idx + len(word)]
            break
    else:
        return None
    for ext in IMG_EXT:
        p = Path(mask_dir) / f"{base}{ext}"
        if p.exists():
            return p
    return None



def load_mask(path, shape, invert: bool):
    m = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    if m.shape[:2] != shape[:2]:
        m = cv2.resize(m, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    m = m > 127
    return ~m if invert else m


def bbox_of(region):
    ys, xs = np.where(region)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def load_prompts(path: Path, id_key: str, prompt_key: str):
    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".tsv"):
        df = pd.read_csv(path, sep="\t" if suffix == ".tsv" else ",", dtype=str)
    else:
        df = pd.read_json(path, lines=True, dtype=False, convert_dates=False)
    missing = [k for k in (id_key, prompt_key) if k not in df.columns]
    if missing:
        sys.exit(f"The prompts file lacks the columns {missing}; columns found: {list(df.columns)}. "
                 f"Use --id-key and --prompt-key.")
    return {str(i): str(p) for i, p in zip(df[id_key], df[prompt_key]) if pd.notna(p)}


def lookup_prompt(prompts, name: str):
    """File name, then name without extension, then longest prefix
    (e.g. key 'frame_000758_track001' for 'frame_000758_track001_inpainted-0.png')."""
    if not prompts:
        return None
    stem = Path(name).stem
    for k in (name, stem):
        if k in prompts:
            return prompts[k]
    cands = [k for k in prompts if stem.startswith(k)]
    return prompts[max(cands, key=len)] if cands else None


def resolve_device(device=None, verbose=True):
    """'cuda' if available and supported by this torch build, otherwise 'cpu'.
    Prints the GPU and the CUDA version, and warns if torch is not compiled for the GPU architecture
    (e.g. RTX 50xx / Blackwell = sm_120: needs torch >= 2.7 with CUDA >= 12.8)."""
    try:
        import torch
    except ImportError:
        return device or "cpu"
    if device and not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        if device:
            sys.exit(f"--device {device} requested but torch {torch.__version__} cannot see the GPU "
                     f"(CUDA build: {torch.version.cuda}). Use the 'gpu' pixi environment.")
        if verbose:
            print(f"[device] cpu (torch {torch.__version__}, CUDA build: {torch.version.cuda})")
        return "cpu"
    dev = device or "cuda"
    idx = torch.device(dev).index or 0
    major, minor = torch.cuda.get_device_capability(idx)
    arch = f"sm_{major}{minor}"
    if verbose:
        print(f"[device] {dev}: {torch.cuda.get_device_name(idx)} ({arch}), torch {torch.__version__}, "
              f"CUDA {torch.version.cuda}")
    archs = torch.cuda.get_arch_list()
    if archs and arch not in archs and f"compute_{major}{minor}" not in archs:
        print(f"[warn] this torch build is compiled for {archs}, not for {arch}: "
              f"expect 'no kernel image is available'. You need torch >= 2.7 with CUDA >= 12.8.")
    return dev


def is_num(v):
    """True if v is a valid number (not None / NaN)."""
    try:
        return v is not None and not pd.isna(v)
    except (TypeError, ValueError):
        return False


def box_from_row(row, prefix):
    """Reads a bbox stored as <prefix>_x0, _y0, _x1, _y1; None if absent."""
    vals = [row.get(f"{prefix}_{k}") for k in ("x0", "y0", "x1", "y1")]
    if not all(is_num(v) for v in vals):
        return None
    return tuple(int(v) for v in vals)


# ----------------------------------------------------------------------------
# Region and crop
# ----------------------------------------------------------------------------
def face_region(shape, box):
    """Ellipse inscribed in the face bbox."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)),
                (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def square_box(shape, box, margin=1.4):
    """Square centred on the bbox, side = longest side x margin, ALWAYS inside the image.
    If it would stick out of the border it is shifted inwards (the region will no longer be at the centre); if
    it is larger than the image, the side is reduced to the short side. No padding: replicated pixels
    create unnatural stripes that lower MUSIQ/CLIPIQA/TOPIQ and move the crop away from the prompt."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(int(max(x1 - x0, y1 - y0) * margin), 16), h, w)
    sx0 = int(round(min(max(cx - side / 2, 0), w - side)))
    sy0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return sx0, sy0, sx0 + side, sy0 + side


def extract(img, box, size, border=cv2.BORDER_REPLICATE, interp=cv2.INTER_CUBIC):
    """Crops img to the box (even outside the borders, with padding) and resizes to size=(w, h)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=0)
    crop = img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, size, interpolation=interp)


def choose_crop(bgr, fbox, mask_box, mode, max_side=768):
    """Returns (crop, kind, source_box, source_side_px, scale_factor).
    source_box = (x0, y0, x1, y1) in coordinates of the original image: the notebook needs it
    to project the mask onto the crop."""
    h, w = bgr.shape[:2]
    if mode != "full":
        for kind, box in (("face", fbox), ("mask", mask_box)):
            if box is not None:
                sq = square_box(bgr.shape, box)
                side = sq[2] - sq[0]
                crop = extract(bgr, sq, (CROP_SIZE, CROP_SIZE))
                return crop, kind, sq, max(box[2] - box[0], box[3] - box[1]), CROP_SIZE / side
    s = min(1.0, max_side / max(h, w))
    crop = bgr.copy() if s == 1 else cv2.resize(bgr, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    return crop, "full", (0, 0, w, h), max(h, w), s


# ----------------------------------------------------------------------------
# Per-stage tables (automatic resume)
# ----------------------------------------------------------------------------
def load_index(out_dir: Path):
    p = Path(out_dir) / INDEX_CSV
    if not p.exists():
        sys.exit(f"{p} not found: run prepare.py --out {out_dir} first")
    return pd.read_csv(p)


class StageTable:
    """CSV of one stage: one row per 'file'. Automatic resume, periodic saving and saving on exit."""

    def __init__(self, path: Path, overwrite=False, save_every=10):
        self.path, self.save_every, self._n = Path(path), save_every, 0
        self.rows = {}
        if self.path.exists() and not overwrite:
            for r in pd.read_csv(self.path).to_dict("records"):
                self.rows[r["file"]] = r
            print(f"[resume] {len(self.rows)} rows already present in {self.path}")
        elif self.path.exists():
            print(f"[overwrite] {self.path} will be rewritten")
        atexit.register(self.save)              # also saves on Ctrl+C or error

    def get(self, name):
        return self.rows.get(name)

    def put(self, row):
        self.rows[row["file"]] = row
        self._n += 1
        if self.save_every and self._n % self.save_every == 0:
            self.save()

    def missing(self, name, columns):
        """Columns not yet computed for this file."""
        r = self.rows.get(name, {})
        return [c for c in columns if not is_num(r.get(c))]

    def df(self):
        return pd.DataFrame(list(self.rows.values()))

    def save(self):
        if self.rows:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.df().to_csv(self.path, index=False)


def run_crop_metrics(out_dir, index, table, metrics, score, applicable, limit=None, extra=None):
    """Loop shared by B and C: for every row of index, computes on the crop the applicable metrics
    not yet present in table.
      metrics:    {metric_name: column_name}; the dict may lose entries during the loop
                  (e.g. a model disabled after an error)
      score:      f(name, crop_bgr, index_row) -> float
      applicable: f(name, index_row) -> bool
      extra:      f(index_row) -> dict of descriptive columns to save (e.g. the prompt)"""
    out_dir = Path(out_dir)
    done = errors = skipped = 0
    rows = index.to_dict("records")
    for r in rows:
        if limit is not None and done >= limit:
            print(f"[limit] reached the limit of {limit} images")
            break
        names = [n for n in list(metrics) if applicable(n, r)]
        todo = [n for n in names if metrics[n] in table.missing(r["file"], [metrics[n]])]
        if not todo:
            continue
        crop = cv2.imread(str(out_dir / r["crop_file"]), cv2.IMREAD_COLOR)
        if crop is None:
            skipped += 1
            continue
        row = dict(table.get(r["file"]) or {"file": r["file"]})
        if extra:
            row.update(extra(r))
        for n in todo:
            if n not in metrics:                     # disabled during the loop
                continue
            column = metrics[n]
            try:
                row[column] = score(n, crop, r)
            except Exception as e:  # noqa: BLE001
                errors += 1
                row[column] = float("nan")
                print(f"[warn] {r['file']} {n}: {type(e).__name__}: {e}")
        table.put(row)
        done += 1
        if done % 50 == 0:
            print(f"[{done}] {r['file']}")
    table.save()
    if skipped:
        print(f"[info] {skipped} crops not found (re-run prepare.py?)")
    if errors:
        print(f"[info] {errors} computation errors")
    print(f"[info] {done} rows computed or completed")


def describe(df, cols):
    cols = [c for c in cols if c in df]
    if cols:
        print(df[cols].describe(percentiles=[.1, .5, .9]).T[["count", "mean", "10%", "50%", "90%"]].round(3))


def add_flag(df, mask, name):
    """Adds `name` to the "flags" column (comma separated) in the rows where mask is True."""
    flags = df["flags"].fillna("").astype(str) if "flags" in df else pd.Series("", index=df.index)
    df["flags"] = [f"{f},{name}" if (m and f) else (name if m else f) for f, m in zip(flags, mask)]
    return df


def load_merged(out_dir: Path):
    """index.csv + a/b/c/padding merged on 'file' (missing columns stay NaN).
    From the padding.csv written by check_padding.py the flags are added: "edge padding" (replicated band in the
    original frame) and "padded crop" (crop box outside the frame)."""
    out_dir = Path(out_dir)
    df = load_index(out_dir)
    for stage, name in STAGE_CSV.items():
        p = out_dir / name
        if p.exists():
            s = pd.read_csv(p)
            s = s[["file"] + [c for c in s.columns if c not in df.columns]]
            df = df.merge(s, on="file", how="left")
    if "frame_pad_sides" in df:
        df = add_flag(df, df["frame_pad_sides"].fillna("") != "", "edge padding")
    if "crop_outside_sides" in df:
        df = add_flag(df, df["crop_outside_sides"].fillna("") != "", "padded crop")
    return df