#!/usr/bin/env python3
"""
export_images.py - saves ALL the images of a table into one file you can scroll through, to review them.

For the table it writes, next to it (or in --dest):
  <name>.pdf   contact sheet: every image as a thumbnail with its file name and some metrics, many per page
  <name>.txt   plain list of the image paths, one per line (--no-pdf: only this)

By default <name> is the table name and the table is <out>/filtered/kept.csv (the images that filter_images.py
kept). Use --table to review the discarded ones instead; their discard `reason` is printed under each thumbnail.

  python export_images.py --out res_d1                                   # kept images -> filtered/kept.pdf, kept.txt
  python export_images.py --out res_d1 --table res_d1/filtered/discarded.csv
  python export_images.py --out res_d1 --source crop                      # the 512 px crops instead of the full frames
  python export_images.py --out res_d1 --shuffle --limit 300             # a random sample of 300
  python export_images.py --out res_d1 --cols 8 --rows 10 --tile 192x144 # more, smaller thumbnails per page

The PDF is built page by page (JPEG pages written straight to the file), so thousands of images do not need
much memory. Images that cannot be read appear as a grey tile marked "missing".
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common import FILTER_DIR

LABEL_COLS = ["reason", "color_dev_frac_in", "p_minor_max"]      # shown under the file name when present
LABEL_H = 34                                                      # px of the label area under each thumbnail
PDF_DPI = 150                                                     # page size in the PDF = pixels at this resolution


class JpegPdf:
    """Minimal PDF writer: one JPEG per page, written straight to disk (no PDF library, no page kept in memory)."""

    def __init__(self, path: Path):
        self.f = open(path, "wb")
        self.f.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        self.offsets, self.n_pages = {}, 0

    def _obj(self, num: int, body: bytes):
        self.offsets[num] = self.f.tell()
        self.f.write(f"{num} 0 obj\n".encode() + body + b"\nendobj\n")

    def add_page(self, jpeg: bytes, w: int, h: int):
        page = 3 + 3 * self.n_pages                     # objects: 1 catalog, 2 page tree, then page/content/image
        pw, ph = w * 72 / PDF_DPI, h * 72 / PDF_DPI
        content = f"q {pw:.2f} 0 0 {ph:.2f} 0 0 cm /Im0 Do Q".encode()
        self._obj(page, f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {pw:.2f} {ph:.2f}] /Contents {page + 1} 0 R "
                        f"/Resources << /XObject << /Im0 {page + 2} 0 R >> >> >>".encode())
        self._obj(page + 1, f"<< /Length {len(content)} >>\nstream\n".encode() + content + b"\nendstream")
        self._obj(page + 2, (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
                             f"/BitsPerComponent 8 /Filter /DCTDecode /Length {len(jpeg)} >>\nstream\n").encode()
                  + jpeg + b"\nendstream")
        self.n_pages += 1

    def close(self):
        kids = " ".join(f"{3 + 3 * i} 0 R" for i in range(self.n_pages))
        self._obj(1, b"<< /Type /Catalog /Pages 2 0 R >>")
        self._obj(2, f"<< /Type /Pages /Kids [{kids}] /Count {self.n_pages} >>".encode())
        size = 3 + 3 * self.n_pages
        xref = self.f.tell()
        self.f.write(f"xref\n0 {size}\n0000000000 65535 f \n".encode())
        for n in range(1, size):
            self.f.write(f"{self.offsets[n]:010d} 00000 n \n".encode())
        self.f.write(f"trailer\n<< /Size {size} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode())
        self.f.close()


def text(img, s, y, scale=0.38):
    cv2.putText(img, s, (3, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def make_tile(path, size, lines):
    """Thumbnail letterboxed in `size`, with the label lines under it."""
    w, h = size
    tile = np.full((h + LABEL_H, w, 3), 255, np.uint8)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR) if path else None
    if img is None:
        tile[:h] = 150
        text(tile, "missing", h // 2)
    else:
        s = min(w / img.shape[1], h / img.shape[0])
        small = cv2.resize(img, (max(1, round(img.shape[1] * s)), max(1, round(img.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
        y0, x0 = (h - small.shape[0]) // 2, (w - small.shape[1]) // 2
        tile[:h] = 40
        tile[y0:y0 + small.shape[0], x0:x0 + small.shape[1]] = small
    for i, line in enumerate(lines[:2]):
        text(tile, line, h + 13 + 13 * i)
    cv2.rectangle(tile, (0, 0), (w - 1, h + LABEL_H - 1), (200, 200, 200), 1)
    return tile


SHORT = {"color_dev_frac_in": "cd", "p_minor_max": "pm"}


def label_lines(row, label_cols, width_chars):
    """[file name, "reason cd=0.00 pm=0.12"], both cut to what fits in a tile."""
    extra = []
    for c in label_cols:
        v = row.get(c)
        if v is None or v == "" or (isinstance(v, (float, np.floating)) and np.isnan(v)):
            continue
        if c == "reason":
            extra.append(str(v))
        else:
            extra.append(f"{SHORT.get(c, c)}=" + (f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)))
    return [Path(str(row["file"])).stem[:width_chars], " ".join(extra)[:width_chars]]


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=None, help="results folder (default table: <out>/filtered/kept.csv)")
    ap.add_argument("--table", type=Path, default=None,
                    help="csv with the images to export, e.g. kept.csv or discarded.csv (default: <out>/filtered/kept.csv)")
    ap.add_argument("--dest", type=Path, default=None, help="output folder (default: the folder of the table)")
    ap.add_argument("--source", choices=["image", "crop"], default="image",
                    help="image: the full frames (image_path); crop: the 512 px crops (<out>/crop_file)")
    ap.add_argument("--cols", type=int, default=6, help="thumbnails per row")
    ap.add_argument("--rows", type=int, default=8, help="rows per page")
    ap.add_argument("--tile", default="256x192", help="thumbnail size WxH in px")
    ap.add_argument("--label-cols", nargs="*", default=LABEL_COLS,
                    help="table columns printed under the file name, when present")
    ap.add_argument("--quality", type=int, default=85, help="JPEG quality of the pages")
    ap.add_argument("--limit", type=int, default=None, help="export at most N images")
    ap.add_argument("--shuffle", action="store_true", help="random order (with --limit: a random sample)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-pdf", action="store_true", help="only write the list of paths")
    return ap


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)
    table = a.table or (a.out / FILTER_DIR / "kept.csv" if a.out else None)
    if table is None:
        ap.error("give --out or --table")
    if not table.exists():
        sys.exit(f"{table} not found (run filter_images.py first, or pass --table)")
    if a.source == "crop" and not a.out:
        ap.error("--source crop needs --out (the crops are inside the results folder)")
    try:
        tw, th = (int(v) for v in a.tile.lower().split("x"))
    except ValueError:
        ap.error("--tile must look like 256x192")

    df = pd.read_csv(table)
    col = "image_path" if a.source == "image" else "crop_file"
    if col not in df:
        sys.exit(f"{table} has no {col} column")
    if a.shuffle:
        df = df.sample(frac=1, random_state=a.seed)
    if a.limit is not None:
        df = df.head(a.limit)
    paths = [(a.out / p if a.source == "crop" else Path(p)) if isinstance(p, str) and p else None for p in df[col]]

    dest = a.dest or table.parent
    dest.mkdir(parents=True, exist_ok=True)
    txt = dest / f"{table.stem}.txt"
    txt.write_text("".join(f"{p}\n" for p in paths if p), encoding="utf-8")
    print(f"{len(df)} images from {table} -> {txt}")
    if a.no_pdf or df.empty:
        return

    per_page = a.cols * a.rows
    pdf = JpegPdf(dest / f"{table.stem}.pdf")
    chars = max(8, int(tw / 6.5))                                  # label characters that fit in a tile
    rows = df.to_dict("records")
    n_pages = -(-len(rows) // per_page)
    try:
        for pg in range(n_pages):
            chunk = slice(pg * per_page, (pg + 1) * per_page)
            page = np.full((a.rows * (th + LABEL_H), a.cols * tw, 3), 255, np.uint8)
            for k, (r, p) in enumerate(zip(rows[chunk], paths[chunk])):
                y, x = (k // a.cols) * (th + LABEL_H), (k % a.cols) * tw
                page[y:y + th + LABEL_H, x:x + tw] = make_tile(p, (tw, th), label_lines(r, a.label_cols, chars))
            ok, jpg = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
            pdf.add_page(jpg.tobytes(), page.shape[1], page.shape[0])
            print(f"[page {pg + 1}/{n_pages}]")
    finally:
        pdf.close()
    print(f"-> {dest / (table.stem + '.pdf')} ({n_pages} pages)")


if __name__ == "__main__":
    main()
