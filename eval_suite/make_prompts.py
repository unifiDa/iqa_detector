#!/usr/bin/env python3
"""
make_prompts.py - converts the prompt JSON files (one per original image) into the prompts.jsonl
read by c_prompt.py.

Each JSON is a list of strings such as
    "Rationale: ...\nReplace with: Snow-covered Fence Wall"
The prompt is extracted EXACTLY as in the FLUX inpainting script: the text after
"Replace with: " (with the space), without stripping quotes, so CLIP/MPS score the same
string that FLUX received. The FLUX script saves  {image_name}_{size}_{i % 5}_inpainted.png
and, because of the "if i == 0: break", only generates the first prompt of each JSON: that is why by
default only i=0 is exported (--all exports them all).

The id of each row is built with --id-template, which can use:
    {name}  name of the JSON without .json            e.g. Sony_DSC-W170_1_51977.JPG_large
            (= {image_name}_{size} of the FLUX script)
    {base}  {name} without _large and without extension e.g. Sony_DSC-W170_1_51977
    {i}     index of the prompt in the list (0, 1, ...)

  python make_prompts.py --json-dir prompts_json/ --out prompts.jsonl
  python make_prompts.py --json-dir prompts_json/ --out prompts.jsonl --images res_d1/crops
"""
import argparse
import json
import re
import sys
from pathlib import Path

MARKER = "Replace with: "
IMG_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def extract_prompt(text: str):
    """Same logic as the FLUX script: prompt_text[idx + len(marker):]."""
    idx = text.find(MARKER)
    if idx == -1:
        return None
    return text[idx + len(MARKER):] or None


def base_of(name: str) -> str:
    name = re.sub(r"_large$", "", name)
    return Path(name).stem if Path(name).suffix.lower() in IMG_EXT else name


def main(argv=None):
    ap = argparse.ArgumentParser(description="Prompt JSON files -> prompts.jsonl for c_prompt.py",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--json-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--id-template", default="{name}_{i}_inpainted",
                    help="inpainted file name without extension (FLUX uses i %% 5)")
    ap.add_argument("--all", action="store_true",
                    help="export all the prompts, not just the first (FLUX only generates i=0)")
    ap.add_argument("--images", type=Path, default=None,
                    help="folder of the inpainted images: checks which ids have an image")
    args = ap.parse_args(argv)

    files = sorted(args.json_dir.glob("*.json"))
    if not files:
        sys.exit(f"No .json in {args.json_dir}")

    rows, bad = [], []
    for f in files:
        try:
            items = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            bad.append(f"{f.name}: invalid JSON ({e})")
            continue
        if not isinstance(items, list):
            bad.append(f"{f.name}: expected a list, found {type(items).__name__}")
            continue
        name = f.name[:-len(".json")]
        for i, text in enumerate(items if args.all else items[:1]):
            prompt = extract_prompt(text) if isinstance(text, str) else None
            if prompt is None:
                bad.append(f"{f.name}[{i}]: 'Replace with:' not found")
                continue
            rid = args.id_template.format(name=name, base=base_of(name), i=i % 5)
            rows.append({"record_id": rid, "prompt": prompt, "source": f.name, "index": i,
                         "rationale": text.split(MARKER)[0].replace("Rationale:", "").strip()})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows from {len(files)} JSON files to {args.out}")
    for r in rows[:3]:
        print(f"  {r['record_id']}  ->  {r['prompt']}")
    if bad:
        print(f"[warn] {len(bad)} items skipped:")
        for b in bad[:10]:
            print("  " + b)

    if args.images:
        stems = {p.stem for p in args.images.rglob("*") if p.suffix.lower() in IMG_EXT}
        ids = [r["record_id"] for r in rows]
        hit = [i for i in ids if i in stems or any(s.startswith(i) for s in stems)]
        print(f"\nImage check: {len(hit)}/{len(ids)} ids find an image in {args.images}")
        if len(hit) < len(ids):
            miss = [i for i in ids if i not in hit][:3]
            print(f"  e.g. ids without an image: {miss}")
            print(f"  e.g. image names:          {sorted(stems)[:3]}")
            print("  -> if the names do not match, change --id-template")


if __name__ == "__main__":
    main()