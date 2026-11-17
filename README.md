# IQA detector

Image-quality analysis of inpainted / generated images, in a single repository. It measures how well the edited region blends with the rest of the image, the perceived quality of the crops, how well they follow the prompt, how close the dataset is to a real reference, and it finally keeps only the colour images without minors.

| stage | script (`eval_suite/`) | what it measures |
|---|---|---|
| 0 | `prepare.py`, `prepare_generic.py` | finds the edited region (mask / face) and saves the 512 px crops |
| A | `a_coherence.py`, `check_padding.py` | blur, Lab tint and seam of the region against its surroundings; replicated border pixels |
| B | `b_quality.py` | no-reference quality: topiq_nr-face, musiq, clipiqa, brisque, niqe, maniqa |
| C | `c_prompt.py` | prompt adherence: CLIP score and MPS score |
| D | `d_dataset.py` | dataset-level FID / KID against a reference set |
| E | `e_age.py` | age bin of every face (YuNet + Hugging Face ViT) |
| — | `merge.py` | merges A-D into `results.csv` and `summary.json` |
| — | `filter_images.py` | keeps the colour images without minors (`color_dev_frac_in` + age) |

Full reference of the stages, the pixi environments, the MPS setup and the notebooks: [`eval_suite/README.md`](eval_suite/README.md).

## Setup

1. Install pixi:
```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

2. Clone the MPS repo and download the MPS weights (they are needed for the `mps_score` only):
```bash
git clone https://github.com/Kwai-Kolors/MPS.git

wget https://huggingface.co/szh/mps-pth/resolve/main/MPS_overall_checkpoint.pth -O Weights/MPS_overall_checkpoint.pth
```
The file `Weights/MPS_overall_checkpoint.pth` committed in this repository is only a 135-byte Git LFS pointer, not the real checkpoint (about 2.2 GB): download it with the command above, which overwrites it.

3. Install the environments:
```bash
pixi install -e gpu     # or: pixi install -e cpu
pixi install -e mps     # MPS score (or: pixi install -e mps-cpu)
```

## Dataset -- LESC

```
inpaints = "/Prove/Shullani/Archive/fooocus-cinematic/inpaints/"
masks = "/Prove/Shullani/Archive/fooocus-cinematic/masks/"
original = "/Prove/Shullani/Archive/fooocus-cinematic/original/"
```

- prepare with faces
```
pixi run -e gpu prepare --images /Scratch/Shullani/Dataset/fooocus-cinematic/inpaints/ --out /Scratch/Shullani/Dataset/results/ --masks /Scratch/Shullani/Dataset/fooocus-cinematic/masks/
```

- prepare original
```
pixi run -e gpu prepare --images /Scratch/Shullani/Dataset/fooocus-cinematic/original/ --out /Scratch/Shullani/Dataset/results-original/
```

- coherence: `pixi run coherence --out /Scratch/Shullani/Dataset/results/`
- padding: `pixi run padding --out /Scratch/Shullani/Dataset/results/`
- quality: `pixi run -e gpu quality --device cuda:0 --out /Scratch/Shullani/Dataset/results/`
- clip score: `pixi run -e gpu prompt --device cuda:0  --out /Scratch/Shullani/Dataset/results/ --prompts /Scratch/Shullani/Dataset/fooocus-cinematic/prompts.jsonl `
- mps score: `pixi run -e mps mps /Scratch/Shullani/Dataset/results/ /Scratch/Shullani/Dataset/fooocus-cinematic/prompts.jsonl /data/staff/shullani/Documents/Research/IAPP/DETECTOR/iqa_detector/Weights/MPS_overall_checkpoint.pth /data/staff/shullani/Documents/Research/IAPP/DETECTOR/iqa_detector/MPS/`
- FID/KID: `pixi run -e gpu dataset --device cuda:0 --out /Scratch/Shullani/Dataset/results/ --ref /Scratch/Shullani/Dataset/results-original/crops/face`
- age of the faces: `pixi run -e gpu age --device cuda:0 --out /Scratch/Shullani/Dataset/results/`

- merge results: `pixi run merge --out /Scratch/Shullani/Dataset/results/`
- filter (colour images without minors; a minor discards the whole frame): `pixi run filter --out /Scratch/Shullani/Dataset/results/ --by-frame` -> `results/filtered/kept.csv`, `discarded.csv`

- FINAL
RUN ALL: `./lamarr.sh`
Run with no preparation: `./lamarr.sh coherence quality prompt mps dataset age merge filter`


## Dataset -- ITI

```
inpaints_flux = "/home/dasara/Research/btb-dresden-all/flux_inpaint/"
inpaints_jagg = "/home/dasara/Research/btb-dresden-all/jagg_inpaint/"
masks = "/home/dasara/Research/btb-dresden-all/btb_masks/"
prompts = "/home/dasara/Research/btb-dresden-all/miniInternVL_btb_prompts/"
```

- prepare generic
```
pixi run -e gpu prepare_generic --images /home/dasara/Research/btb-dresden-all/flux_inpaint/ --out /home/dasara/Research/results/flux/ --masks /home/dasara/Research/btb-dresden-all/btb_masks/
```

- coherence: `pixi run coherence --out /home/dasara/Research/results/flux/`
- check padding: `pixi run padding --out /home/dasara/Research/results/flux/`
- quality: `pixi run -e gpu quality --device cuda:0 --metrics musiq clipiqa brisque niqe maniqa --out /home/dasara/Research/results/flux/`
- generate prompts.jsonl: `pixi run make_prompts --json-dir /home/dasara/Research/btb-dresden-all/miniInternVL_btb_prompts/ --out /home/dasara/Research/results/flux/prompts.jsonl --images /home/dasara/Research/btb-dresden-all/flux_inpaint/`
- clip score: `pixi run -e gpu prompt --device cuda:0  --out /home/dasara/Research/results/flux/ --prompts /home/dasara/Research/results/flux/prompts.jsonl`
- mps score: `pixi run -e mps mps /home/dasara/Research/results/flux/ /home/dasara/Research/results/flux/prompts.jsonl /home/dasara/Research/code/iqa_detector/Weights/MPS_overall_checkpoint.pth /home/dasara/Research/code/iqa_detector/MPS/`

- merge results: `pixi run merge --out /home/dasara/Research/results/flux/`

The same sequence is wrapped in `./flux.sh` and `./jagg.sh`. These datasets are generic inpainting (no faces), so the age and filter steps are not part of them.

## Age estimation and filtering

The age filter that used to live in the separate `filter_generated_data` repository is now stage E plus the final `filter` step of this pipeline:

```bash
pixi run -e gpu age --out results/            # e_age.csv: one row per face (age bin + confidence)
pixi run merge --out results/                 # results.csv (needs stage A for color_dev_frac_in)
pixi run filter --out results/ --by-frame     # filtered/kept.csv, filtered/discarded.csv
```

See [`eval_suite/README.md`](eval_suite/README.md#e-age-estimation-and-filtering) for the options and the exact filtering rules.

## License

This project is released under the GNU General Public License v3.0, see [`LICENSE`](LICENSE).
