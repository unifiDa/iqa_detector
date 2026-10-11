"""
common.py - funzioni condivise dagli script di valutazione.

Pipeline (tutti gli script scrivono in --out, collegati dalla colonna "file"):

  prepare.py     rileva volto/maschera, salva i ritagli    -> index.csv, crops/<kind>/
  a_coherence.py   A. sfocatura / tinta Lab / bordo           -> a_coherence.csv
  b_quality.py     B. topiq_nr-face, musiq, clipiqa (pyiqa)   -> b_quality.csv
  c_prompt.py      C. clip_score, mps_score (serve prompt)    -> c_prompt.csv
  d_dataset.py     D. FID / KID con clean-fid                 -> d_dataset.json
  merge.py         unisce tutto                               -> results.csv, summary.json

A, B, C e D leggono solo index.csv e i ritagli: si possono lanciare in qualunque ordine,
su macchine diverse, e rilanciare senza rifare il lavoro gia' fatto (le righe presenti vengono
saltate e si calcolano solo le colonne mancanti; --overwrite per ricominciare).
"""
import atexit
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

VERSION = "v3.0"
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MASK_SIZE = ("large", "medium", "small")      # tupla: l'ordine e' la priorita' (con un set cambiava a ogni lancio)
CROP_SIZE = 512

INDEX_CSV = "index.csv"
STAGE_CSV = {"a": "a_coherence.csv", "b": "b_quality.csv", "c": "c_prompt.csv"}
D_JSON = "d_dataset.json"


# ----------------------------------------------------------------------------
# I/O di base
# ----------------------------------------------------------------------------
def list_images(d: Path):
    return sorted(p for p in Path(d).iterdir() if p.suffix.lower() in IMG_EXT)


def find_mask(mask_dir: Path, stem: str, template: str):
    """Immagine 'frame_000755_track001_inpainted-0.png' -> maschera 'frame_000755_track001_mask.<ext>'
    con il template di default '{prefix}mask'. {prefix} = parte del nome prima di 'inpainted'
    (se manca, nome + '_'), {stem} = nome completo senza estensione."""
    idx = stem.find("inpainted")
    prefix = stem[:idx] if idx > 0 else stem + "_"
    base = template.format(stem=stem, prefix=prefix)
    for ext in IMG_EXT:
        p = Path(mask_dir) / f"{base}{ext}"
        if p.exists():
            return p
    return None



def find_mask_generic(mask_dir: Path, stem: str, prefix="mask_"):
    """Schema 'size': la maschera si chiama <prefix><nome fino a large|medium|small compresi>.<ext>.
    Es. 'Agfa_DC-504_0_10.JPG_large_0_inpainted-0.png' -> 'mask_Agfa_DC-504_0_10.JPG_large.png'. Ritorna None 
    se il nome non contiene nessuna delle tre parole o se il file non esiste."""
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
        sys.exit(f"Nel file prompts mancano le colonne {missing}; colonne trovate: {list(df.columns)}. "
                 f"Usa --id-key e --prompt-key.")
    return {str(i): str(p) for i, p in zip(df[id_key], df[prompt_key]) if pd.notna(p)}


def lookup_prompt(prompts, name: str):
    """Nome file, poi nome senza estensione, poi prefisso piu' lungo
    (es. chiave 'frame_000758_track001' per 'frame_000758_track001_inpainted-0.png')."""
    if not prompts:
        return None
    stem = Path(name).stem
    for k in (name, stem):
        if k in prompts:
            return prompts[k]
    cands = [k for k in prompts if stem.startswith(k)]
    return prompts[max(cands, key=len)] if cands else None


def resolve_device(device=None, verbose=True):
    """'cuda' se disponibile e supportata da questa build di torch, altrimenti 'cpu'.
    Stampa GPU e versione CUDA, e avvisa se torch non e' compilato per l'architettura della GPU
    (es. RTX 50xx / Blackwell = sm_120: serve torch >= 2.7 con CUDA >= 12.8)."""
    try:
        import torch
    except ImportError:
        return device or "cpu"
    if device and not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        if device:
            sys.exit(f"--device {device} richiesto ma torch {torch.__version__} non vede la GPU "
                     f"(build CUDA: {torch.version.cuda}). Usa l'ambiente pixi 'gpu'.")
        if verbose:
            print(f"[device] cpu (torch {torch.__version__}, build CUDA: {torch.version.cuda})")
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
        print(f"[warn] questa build di torch e' compilata per {archs}, non per {arch}: "
              f"aspettati 'no kernel image is available'. Serve torch >= 2.7 con CUDA >= 12.8.")
    return dev


def is_num(v):
    """True se v e' un numero valido (non None / NaN)."""
    try:
        return v is not None and not pd.isna(v)
    except (TypeError, ValueError):
        return False


def box_from_row(row, prefix):
    """Legge un bbox salvato come <prefix>_x0, _y0, _x1, _y1; None se assente."""
    vals = [row.get(f"{prefix}_{k}") for k in ("x0", "y0", "x1", "y1")]
    if not all(is_num(v) for v in vals):
        return None
    return tuple(int(v) for v in vals)


# ----------------------------------------------------------------------------
# Regione e ritaglio
# ----------------------------------------------------------------------------
def face_region(shape, box):
    """Ellisse inscritta nel bbox del volto."""
    h, w = shape[:2]
    m = np.zeros((h, w), np.uint8)
    x0, y0, x1, y1 = box
    cv2.ellipse(m, (int((x0 + x1) / 2), int((y0 + y1) / 2)),
                (max(1, (x1 - x0) // 2), max(1, (y1 - y0) // 2)), 0, 0, 360, 1, -1)
    return m.astype(bool)


def square_box(shape, box, margin=1.4):
    """Quadrato centrato sul bbox, lato = lato maggiore x margin, SEMPRE dentro l'immagine.
    Se sporgerebbe dal bordo lo si sposta verso l'interno (la regione non sara' piu' al centro); se e'
    piu' grande dell'immagine, il lato si riduce al lato corto. Niente padding: i pixel replicati
    creano strisce innaturali che abbassano MUSIQ/CLIPIQA/TOPIQ e allontanano il ritaglio dal prompt."""
    h, w = shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = min(max(int(max(x1 - x0, y1 - y0) * margin), 16), h, w)
    sx0 = int(round(min(max(cx - side / 2, 0), w - side)))
    sy0 = int(round(min(max(cy - side / 2, 0), h - side)))
    return sx0, sy0, sx0 + side, sy0 + side


def extract(img, box, size, border=cv2.BORDER_REPLICATE, interp=cv2.INTER_CUBIC):
    """Ritaglia img nel box (anche fuori dai bordi, con padding) e ridimensiona a size=(w, h)."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    pad = max(0, -x0, -y0, x1 - w, y1 - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, border, value=0)
    crop = img[y0 + pad:y1 + pad, x0 + pad:x1 + pad]
    return cv2.resize(crop, size, interpolation=interp)


def choose_crop(bgr, fbox, mask_box, mode, max_side=768):
    """Ritorna (ritaglio, tipo, box_sorgente, lato_sorgente_px, fattore_di_scala).
    box_sorgente = (x0, y0, x1, y1) in coordinate dell'immagine originale: serve al notebook
    per proiettare la maschera sul ritaglio."""
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
# Tabelle per stadio (ripresa automatica)
# ----------------------------------------------------------------------------
def load_index(out_dir: Path):
    p = Path(out_dir) / INDEX_CSV
    if not p.exists():
        sys.exit(f"{p} non trovato: lancia prima prepare.py --out {out_dir}")
    return pd.read_csv(p)


class StageTable:
    """CSV di uno stadio: una riga per 'file'. Ripresa automatica, salvataggio periodico e all'uscita."""

    def __init__(self, path: Path, overwrite=False, save_every=10):
        self.path, self.save_every, self._n = Path(path), save_every, 0
        self.rows = {}
        if self.path.exists() and not overwrite:
            for r in pd.read_csv(self.path).to_dict("records"):
                self.rows[r["file"]] = r
            print(f"[resume] {len(self.rows)} righe gia' presenti in {self.path}")
        elif self.path.exists():
            print(f"[overwrite] {self.path} verra' riscritto")
        atexit.register(self.save)              # salva anche con Ctrl+C o errore

    def get(self, name):
        return self.rows.get(name)

    def put(self, row):
        self.rows[row["file"]] = row
        self._n += 1
        if self.save_every and self._n % self.save_every == 0:
            self.save()

    def missing(self, name, columns):
        """Colonne non ancora calcolate per questo file."""
        r = self.rows.get(name, {})
        return [c for c in columns if not is_num(r.get(c))]

    def df(self):
        return pd.DataFrame(list(self.rows.values()))

    def save(self):
        if self.rows:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.df().to_csv(self.path, index=False)


def run_crop_metrics(out_dir, index, table, metrics, score, applicable, limit=None, extra=None):
    """Ciclo comune a B e C: per ogni riga di index calcola sul ritaglio le metriche applicabili
    non ancora presenti in table.
      metrics:    {nome_metrica: nome_colonna}; il dict puo' perdere voci durante il ciclo
                  (es. un modello disattivato dopo un errore)
      score:      f(nome, crop_bgr, riga_index) -> float
      applicable: f(nome, riga_index) -> bool
      extra:      f(riga_index) -> dict di colonne descrittive da salvare (es. il prompt)"""
    out_dir = Path(out_dir)
    done = errors = skipped = 0
    rows = index.to_dict("records")
    for r in rows:
        if limit is not None and done >= limit:
            print(f"[limit] raggiunto il limite di {limit} immagini")
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
            if n not in metrics:                     # disattivata durante il ciclo
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
        print(f"[info] {skipped} ritagli non trovati (rilancia prepare.py?)")
    if errors:
        print(f"[info] {errors} errori di calcolo")
    print(f"[info] {done} righe calcolate o completate")


def describe(df, cols):
    cols = [c for c in cols if c in df]
    if cols:
        print(df[cols].describe(percentiles=[.1, .5, .9]).T[["count", "mean", "10%", "50%", "90%"]].round(3))


def load_merged(out_dir: Path):
    """index.csv + a/b/c uniti su 'file' (le colonne mancanti restano NaN)."""
    out_dir = Path(out_dir)
    df = load_index(out_dir)
    for stage, name in STAGE_CSV.items():
        p = out_dir / name
        if p.exists():
            s = pd.read_csv(p)
            s = s[["file"] + [c for c in s.columns if c not in df.columns]]
            df = df.merge(s, on="file", how="left")
    return df