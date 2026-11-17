#!/usr/bin/env python3
"""
c_prompt.py - C. prompt adherence and human preference (needs the prompts file).

  clip_score  100 x cosine(image embedding, text embedding), CLIP ViT-L/14.
              Typical values 20-30; only compare images with the same prompt.
  mps_score   Multi-dimensional Preference Score (Kwai-Kolors/MPS, CVPR 2024), "overall" score per
              single image (logit_scale x conditioned text-image cosine). Higher = more
              preferred, but the scale only makes sense for the same prompt.

The prompt of each image is looked up by file name, name without extension, then longest prefix
(e.g. id 'frame_000758_track001' for 'frame_000758_track001_inpainted-0.png').

  python c_prompt.py --out res_d1 --prompts prompts.jsonl
  python c_prompt.py --out res_d1 --prompts prompts.jsonl --metrics mps --mps-ckpt X.pth --mps-repo MPS/
"""
import argparse
import sys
import traceback
from pathlib import Path

import cv2

from common import (STAGE_CSV, StageTable, describe, load_index, load_prompts, lookup_prompt,
                    resolve_device, run_crop_metrics)

CLIP_NAME = "openai/clip-vit-large-patch14"
MPS_PROC = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
MPS_CONDITION = ("light, color, clarity, tone, style, ambiance, artistry, shape, face, hair, hands, limbs, "
                 "structure, instance, texture, quantity, attributes, position, number, location, word, things.")
COLUMN = {"clip": "clip_score", "mps": "mps_score"}


def load_clip(device):
    from transformers import CLIPModel, CLIPProcessor
    return CLIPModel.from_pretrained(CLIP_NAME).to(device).eval(), CLIPProcessor.from_pretrained(CLIP_NAME)


def clip_score(model_proc, crop_bgr, prompt, device):
    import torch
    model, proc = model_proc
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    inputs = proc(text=[prompt], images=rgb, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = model(**inputs)
    img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return float((img * txt).sum() * 100)


def load_mps(ckpt: Path, repo, device):
    """Loads the MPS 'overall' checkpoint. WARNING: the file is a model serialised with pickle
    (torch.load with weights_only=False) and therefore runs code when loaded: only download it from the
    official link of the Kwai-Kolors/MPS repo.
    The pickle holds the whole object, so loading needs the same classes it was saved with:
    trainer.models.* from the cloned repo (--mps-repo, put on the import path; its
    requirements.txt must NOT be installed) and the internal classes of transformers 4.27.3
    (CLIPTextTransformer, etc., removed in recent versions). That is why MPS runs in the 'mps' pixi environment."""
    import transformers
    if transformers.__version__ != "4.27.3":
        print(f"[warn] transformers {transformers.__version__}: the MPS checkpoint was saved with 4.27.3 "
              f"and may fail to load with other versions. Use: pixi run -e mps mps ... (or -e mps-cpu without a GPU)")
    import torch
    from transformers import AutoTokenizer, CLIPImageProcessor
    if repo:
        sys.path.insert(0, str(repo))
    proc = CLIPImageProcessor.from_pretrained(MPS_PROC)
    tok = AutoTokenizer.from_pretrained(MPS_PROC)
    try:
        model = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    except TypeError:                                      # torch < 1.13
        model = torch.load(str(ckpt), map_location="cpu")
    # the checkpoint is in fp16: fine on GPU, but on CPU many fp16 operations do not exist
    # or are very slow, so on CPU everything is converted to fp32 (same result, more memory)
    model = model.eval().to(device)
    if not str(device).startswith("cuda"):
        model = model.float()
    dtype = next(model.parameters()).dtype
    print(f"[mps] loaded on {device} ({dtype})")
    return model, proc, tok, device


def mps_score(mps, crop_bgr, prompt):
    """Replicates infer_one_sample() of eval_overall_mps_on_hpdv2.py from the MPS repo."""
    import torch
    from PIL import Image
    model, proc, tok, device = mps

    def ids(text):
        return tok(text, max_length=tok.model_max_length, padding="max_length", truncation=True,
                   return_tensors="pt").input_ids.to(device)

    dtype = next(model.parameters()).dtype                 # fp16 on GPU, fp32 on CPU (see load_mps)
    img = Image.fromarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
    pix = proc(img, return_tensors="pt")["pixel_values"].to(device)
    with torch.no_grad():
        text_f, text_eos = model.model.get_text_features(ids(prompt))
        image_f = model.model.get_image_features(pix.to(dtype))
        cond_f, _ = model.model.get_text_features(ids(MPS_CONDITION))
        sim = torch.einsum("b i d, b j d -> b j i", text_f, cond_f)
        sim = sim.max(dim=1, keepdim=True)[0]
        sim = sim / sim.max()
        mask = torch.where(sim > 0.3, 0, float("-inf")).repeat(1, image_f.shape[1], 1)
        feat = model.cross_model(image_f, text_f, mask.to(dtype))[:, 0, :]
        feat = feat / feat.norm(dim=-1, keepdim=True)
        txt = text_eos / text_eos.norm(dim=-1, keepdim=True)
        return float((model.logit_scale.exp() * txt @ feat.T)[0, 0])


def build_parser():
    ap = argparse.ArgumentParser(description="C. prompt adherence (CLIP, MPS)",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out", required=True, type=Path, help="folder created by prepare.py")
    ap.add_argument("--prompts", required=True, type=Path, help="JSONL/CSV/TSV with id and prompt")
    ap.add_argument("--id-key", default="record_id")
    ap.add_argument("--prompt-key", default="prompt")
    ap.add_argument("--metrics", nargs="+", choices=["clip", "mps"], default=["clip"])
    ap.add_argument("--mps-ckpt", type=Path, default=None, help="MPS_overall_checkpoint.pth")
    ap.add_argument("--mps-repo", type=Path, default=None, help="folder of the cloned Kwai-Kolors/MPS repo")
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--save-every", type=int, default=25)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if "mps" in args.metrics and not args.mps_ckpt:
        sys.exit("--metrics mps requires --mps-ckpt (MPS overall checkpoint) and usually --mps-repo")

    device = resolve_device(args.device)
    index = load_index(args.out)
    prompts = load_prompts(args.prompts, args.id_key, args.prompt_key)
    prompt_of = {f: lookup_prompt(prompts, f) for f in index["file"]}
    missing = [f for f, p in prompt_of.items() if p is None]
    if missing:
        print(f"[warn] {len(missing)}/{len(index)} images without a prompt, e.g. {missing[:2]}; "
              f"ids in the prompts file, e.g. {list(prompts)[:2]}")

    table = StageTable(args.out / STAGE_CSV["c"], args.overwrite, args.save_every)
    metrics = {n: COLUMN[n] for n in args.metrics}
    clip = load_clip(device) if "clip" in metrics else None
    mps = None
    if "mps" in metrics:
        try:
            mps = load_mps(args.mps_ckpt, args.mps_repo, device)
        except Exception as e:  # noqa: BLE001
            print("\n" + "=" * 70)
            print(f"[ERROR] MPS not loaded: {type(e).__name__}: {e}")
            traceback.print_exc()
            if "Can't get attribute" in str(e) and "transformers" in str(e):
                print("The checkpoint needs the transformers 4.27.3 classes: run MPS in the dedicated "
                      "environment, e.g.  pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/")
            elif isinstance(e, ModuleNotFoundError) and "trainer" in str(e):
                print("--mps-repo must point to the root of the cloned Kwai-Kolors/MPS repo "
                      "(the one that contains the 'trainer/' folder).")
            else:
                print("Check: --mps-ckpt points to the .pth file; --mps-repo points to the root of the cloned repo "
                      "(the one that contains the 'trainer/' folder).")
            print("=" * 70 + "\n")
            metrics.pop("mps")
    if not metrics:
        sys.exit("No metric available.")

    def score(name, crop, row):
        prompt = prompt_of[row["file"]]
        if name == "clip":
            return clip_score(clip, crop, prompt, device)
        try:
            return mps_score(mps, crop, prompt)
        except Exception:
            print("\n[ERROR] MPS failed on the first image: disabling it for this run.")
            traceback.print_exc()
            metrics.pop("mps", None)
            raise

    def applicable(name, row):
        return prompt_of.get(row["file"]) is not None

    def extra(row):
        return {"prompt": prompt_of.get(row["file"])}

    run_crop_metrics(args.out, index, table, metrics, score, applicable, args.limit, extra)
    print(f"\nSaved {table.path}")
    describe(table.df(), list(COLUMN.values()))


if __name__ == "__main__":
    main()
