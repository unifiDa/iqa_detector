# eval_suite

The evaluation pipeline, split by component. Every script writes into the same `--out` folder and the tables are linked by the `file` column.

| script | what it does | writes | dependencies | GPU |
|---|---|---|---|---|
| `prepare.py` | step 0: mask, face (insightface), 512 px crop | `index.csv`, `crops/<kind>/` | opencv, insightface (opt.) | yes (onnxruntime) |
| `prepare_generic.py` | step 0 for generic inpainting: mask only, no face detection | `index.csv`, `crops/<kind>/` | opencv | no |
| `make_prompts.py` | prompt JSON files -> `prompts.jsonl` | `prompts.jsonl` | — | no |
| `a_coherence.py` | **A** blur, Lab tint, seam, flags | `a_coherence.csv` | opencv | no, not needed |
| `check_padding.py` | replicated pixels in the frame, crops outside the frame (flags for merge) | `padding.csv` | opencv | no |
| `b_quality.py` | **B** topiq_nr-face, musiq, clipiqa (also brisque, niqe, maniqa) | `b_quality.csv` | pyiqa, torch | yes |
| `c_prompt.py` | **C** clip_score, mps_score | `c_prompt.csv` | transformers, torch; MPS: transformers 4.27.3 + MPS repo | yes |
| `d_dataset.py` | **D** FID / KID | `d_dataset.json` | clean-fid | yes |
| `e_age.py` | **E** age of the faces (YuNet + Hugging Face ViT), one row per face | `e_age.csv`, `age_annotated/` | opencv, torch, transformers | yes |
| `merge.py` | merges A-D | `results.csv`, `summary.json` | pandas | no |
| `filter_images.py` | keeps the colour images without minors | `filtered/kept.csv`, `filtered/discarded.csv` | pandas | no |
| `export_images.py` | saves all the kept (or discarded) images in one PDF contact sheet + a list of paths | `filtered/kept.pdf`, `filtered/kept.txt` | opencv, pandas | no |
| `reject_images.py` | moves the images you rejected by eye from `kept.csv` to `discarded.csv` | updates `filtered/*.csv`, `filtered/manual_rejects.txt` | pandas | no |
| `analyze_dataset.py` | resolution and metric ranges of the final dataset | `filtered/analysis/` | pandas, matplotlib | no |
| `gpu_check.py` | checks that torch and onnxruntime use the GPU | — | torch, onnxruntime | — |
| `viewer_*.ipynb` | notebooks to look at the results (see [Notebooks](#notebooks)) | — | matplotlib, ipywidgets (opt.) | no |

`common.py` must stay in the same folder as the scripts. `pixi.toml` expects the `eval_suite/` folder in the project root.

## Pixi environments

The project is Linux x86_64 only (`platforms = ["linux-64"]`).

| environment | for what | torch | face detection |
|---|---|---|---|
| `default` | stages that only need pandas/OpenCV: `coherence`, `padding`, `merge`, `filter`, `make_prompts`, `prepare_generic` | — | — |
| `gpu` | everything except MPS, on an NVIDIA GPU | PyTorch CUDA 12.8 wheels | `onnxruntime-gpu` |
| `cpu` | everything except MPS, without a GPU | `pytorch-cpu` (conda-forge) | `onnxruntime` |
| `mps` | MPS only, on GPU (fp16) | PyTorch CUDA 12.8 wheels | — |
| `mps-cpu` | MPS only, without a GPU (fp32) | `pytorch-cpu` (conda-forge) | — |

```bash
# curl -fsSL https://pixi.sh/install.sh | sh

pixi install -e gpu       # GPU environment
pixi install -e cpu       # environment without a GPU
pixi install -e mps       # MPS on GPU
pixi install -e mps-cpu   # MPS without a GPU
```

Without `-e`, a task runs in the `default` environment. Stages that use torch (`quality`, `prompt`, `dataset`, `age`) and `prepare` only exist in the `gpu` and `cpu` environments (in `default` pixi says the task does not exist, instead of failing with `No module named 'torch'`): `pixi run -e cpu quality --out results/`. `gpu-check` only exists in `gpu`: `pixi run -e gpu gpu-check`.

### GPU

- **Requirements:** a recent NVIDIA driver. RTX 50xx cards (Blackwell, sm_120) need torch ≥ 2.7 compiled with CUDA ≥ 12.8, so the environment installs torch ≥ 2.8 from the `cu128` wheels.
- **CUDA 13 driver:** fine. A CUDA 13 driver runs programs compiled for CUDA 12.x without problems.
- **Why cu128 and not cu130:** the `cu130` wheels would work for torch. But `onnxruntime-gpu` is built for CUDA 12 and would not find its libraries, so face detection would fall back to the CPU.
- **Check:** before the first run, execute `pixi run -e gpu gpu-check`. The script does an fp16 matrix multiplication on the GPU, opens an onnxruntime session on CUDA and reports whether the torch build does not support the card architecture.
- **Start-up messages:** every script that uses torch prints the GPU in use, for example `[device] cuda: NVIDIA GeForce RTX 5090 (sm_120), torch 2.x, CUDA 12.8`. `prepare.py` prints `[face-det] insightface on CUDAExecutionProvider`. If onnxruntime falls back to the CPU, `prepare.py` says so and detection goes on, slower, on the CPU.
- **Forcing the CPU in the GPU environment:** `--device cpu` for B, C, D and E; `--cpu` for `prepare.py`.

### CPU

- **What works:** all the metrics also run on the CPU, MPS included, only slower. The heaviest are MPS (a ViT-H), B (pyiqa), D (clean-fid) and E on many frames.
- **A, merge and filter:** only use OpenCV and pandas, so they are equally fast in every environment.
- **MPS:** runs in its own environment, `mps-cpu` without a GPU. See below.

## Example with pixi

GPU (`gpu` environment):

```bash
pixi run -e gpu gpu-check
pixi run -e gpu prepare --images imgs/ --masks masks/ --out results/
pixi run coherence --out results/
pixi run padding   --out results/
pixi run -e gpu quality --out results/
pixi run -e gpu prompt  --out results/ --prompts prompts.jsonl            # CLIP
pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/  # MPS
pixi run -e gpu dataset --out results/ --ref originals/crops/face
pixi run -e gpu age     --out results/                                    # age of the faces
pixi run merge  --out results/
pixi run filter --out results/ --by-frame                                 # colour images without minors
```

CPU (`cpu` environment): same commands with `-e cpu`, for example `pixi run -e cpu quality --out results/`. For MPS use `-e mps-cpu` instead of `-e mps`.

The whole sequence for a dataset is wrapped in `run.sh` in the project root, driven by a config file in `datasets/` (`./run.sh lamarr`, or `./run.sh lamarr quality prompt` for only some steps). To use another dataset copy `datasets/template.env`: see the main README, "Adding a new dataset".

### Mask naming

By default `prepare.py` looks for the mask of `<name>_inpainted-N.png` as `<name>_mask.<ext>`, and `prepare_generic.py` for `mask_<name up to large|medium|small>.<ext>`. Any other scheme is described with a regex on the image name (no extension) and a template that uses its groups, in both scripts:

```bash
# image img_0012_edit.png -> mask img_0012.png
python prepare.py --images imgs/ --masks masks/ --out res --mask-regex '^(?P<base>.*)_edit$' --mask-template '{base}'
```

In the template `{name}` is a named group, `{0}`, `{1}`... are the positional groups, `{match}` the whole match and `{stem}` the whole image name.

## Example without pixi

```bash
python prepare.py     --images imgs/ --masks masks/ --out res_d1
python a_coherence.py --out res_d1
python b_quality.py   --out res_d1
python c_prompt.py    --out res_d1 --prompts prompts.jsonl
python c_prompt.py    --out res_d1 --prompts prompts.jsonl --metrics mps --mps-ckpt MPS_overall_checkpoint.pth --mps-repo MPS/   # with transformers 4.27.3
python d_dataset.py   --out res_d1 --ref res_originals/crops/face
python e_age.py       --out res_d1
python merge.py       --out res_d1 --name dataset1
python filter_images.py --out res_d1 --by-frame
```

A, B, C, D and E only read `index.csv` and the crops (E, by default, the full frames), so the order is free. You can also run them on different machines: for example `prepare` and A on a laptop without a GPU, then B, C, D and E on the workstation with the GPU, copying the `--out` folder.

`index.csv` stores the absolute paths of the images and of the masks. A, E (in its default mode) and the notebooks read them again, so if you move the folder to another machine those paths must exist there too. B, C and D only use the crops inside `--out`.

## E. Age estimation and filtering

`e_age.py` is the modern version of the PyImageSearch age-detection tutorial (OpenCV):

| step | 2020 (tutorial) | here |
|---|---|---|
| faces | Caffe SSD res10 | **YuNet** (OpenCV `FaceDetectorYN`, ONNX) |
| age | Caffe CNN, 8 bins | Hugging Face **ViT** (9 bins), replaceable with `--model` |

```bash
pixi run -e gpu age --out results/                  # the full frames of index.csv: YuNet + age classifier
pixi run -e gpu age --out results/ --crops          # the face crops (crops/face): detector skipped
pixi run -e gpu age --out results/ --input photos/  # stand-alone: any image or folder, no index.csv
pixi run -e gpu age --out results/ --model <hf_repo>   # another age model
```

- **Output:** `e_age.csv` has one row per face (`file`, `image_path`, face box, `det_score`, `age_bin`, `age_conf`, `p_minor`). `p_minor` is the sum of the probabilities of the bins in `--minor-bins` (default `0-2`, `3-9`, `10-19`; they must match the model labels). Images with no detected face get a row with an empty `age_bin`, so they are not processed again when you resume. Unless you pass `--no-draw` (or `--crops`), the annotated frames go to `age_annotated/`.
- **Output is an age bin** with a probability, not a number. For a numeric age use a regression model (e.g. MiVOLO) through `--model`, if compatible.
- **First run:** YuNet is downloaded into `models/` and the Hugging Face model from the Hub (network needed).
- **Resume:** images already present in `e_age.csv` are skipped; `--overwrite` starts again.

`filter_images.py` is the last step. It combines `results.csv` (colour check from stage A, `color_dev_frac_in`) with `e_age.csv`, linking them on the file base name. A row is kept only if **all** of these are true:

- `color_dev_frac_in <= --max-color-dev` (default 0: only images with no colour deviation);
- at least one face has an age estimate (otherwise a minor cannot be ruled out; `--keep-no-face` overrides this);
- no face has a bin in `--minor-bins` (default `0-2`, `3-9`, `10-19`), nor `p_minor >= --max-p-minor` (default 0.2). `p_minor` is written by `e_age.py` and is the summed probability of the minor bins, so a face the classifier is unsure about is also caught, not only when a minor bin is the top-1. If `e_age.csv` has no `p_minor` values (written by an older version), the filter warns and only uses the bins: re-run `e_age.py --overwrite`.

With `--by-frame` a minor face also discards all the other rows of the same frame. The frame is recognised by `--frame-regex` (default `(frame_\d+)`, matched at the start of the file name, group 1 or the whole match); a file that does not match is its own frame. The results are `filtered/kept.csv` and `filtered/discarded.csv` (with the `reason`), inspected with `viewer_filter.ipynb`.

## MPS

The MPS checkpoint (`MPS_overall_checkpoint.pth`) does not only hold the weights: it is the whole model serialised with pickle, including the internal classes of `transformers` 4.27.3 (for example `CLIPTextTransformer`), which do not exist any more in recent versions. With a recent `transformers`, loading fails with an error like this:

```
AttributeError: Can't get attribute 'CLIPTextTransformer' on <module 'transformers.models.clip.modeling_clip' ...>
```

The error does not depend on the GPU: it happens at load time, in both the `gpu` and the `cpu` environments. That is why MPS runs in a separate environment, which only holds torch and `transformers==4.27.3`, with `tokenizers` and `huggingface-hub` of the same era. There are two variants:

```bash
# with GPU (fp16)
pixi install -e mps
pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/

# without GPU (fp32)
pixi install -e mps-cpu
pixi run -e mps-cpu mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/
```

- **Arguments:** the output folder, the prompts file, the checkpoint and the root of the cloned [Kwai-Kolors/MPS](https://github.com/Kwai-Kolors/MPS) repo, i.e. the folder that contains `trainer/`. The repo only has to be cloned: its `requirements.txt` must not be installed.
- **Where the result goes:** `mps_score` is added to `c_prompt.csv` next to `clip_score`, without recomputing CLIP, and then the task updates `results.csv`.
- **On GPU** the model runs in fp16, as in the original repo. If it loaded correctly you see `[mps] loaded on cuda (torch.float16)`.
- **Without GPU** (`mps-cpu`): the script converts the model to fp32, because many fp16 operations on the CPU do not exist or are extremely slow. The code of the MPS repo never forces fp16 or CUDA, so it runs normally on the CPU. The result is the same apart from small numerical differences between fp16 and fp32, but it is slow: it is a ViT-H and each image takes a few seconds. If it loaded correctly you see `[mps] loaded on cpu (torch.float32)`.
- **Security:** the checkpoint is loaded with `torch.load(weights_only=False)` and runs code when opened. Only download it from the official link of the repo.

## Resume

Every script resumes from where it stopped. Rows already present are skipped and only the missing columns are computed: `b_quality.py --metrics musiq` adds musiq without redoing the others. Use `--overwrite` to start a stage again.

In `a_coherence.py` the flags are recomputed at every run from the saved columns. To try different thresholds just re-run it with `--blur-thr`, `--seam-thr`, `--color-frac` or `--mono-spread`, and the images are not processed again. If you change `--ref` you need `--overwrite` instead.

## Notebooks

`pixi run lab` opens JupyterLab. In the notebooks set `OUT` / `RESULTS` (the `--out` folder) and, for the metric viewers, `SCRIPTS` (the folder of `common.py`).

- `viewer_filtered.ipynb`, `viewer_dresden.ipynb`: sort the crops by metric. `show(...)` sorts in ascending order and shows the crops under a histogram that marks the displayed values. Modes: quantiles (N images from the minimum to the maximum) and extremes (the N lowest / highest). `viewer_filtered` first filters on `color_dev_frac_in`.
- `viewer_faces.ipynb`: mosaic of the faces for each age bin, lowest confidence first (the most likely mistakes), and the images without a detected face.
- `viewer_filter.ipynb`: kept / discarded images of `filter_images.py`, with the metric distributions to choose the thresholds, mosaics grouped by discard reason and a single-frame inspector.

None of them uses the GPU: they run in any environment.

### Reviewing all the kept images in one file

The notebooks only show a sample. To check every image that survived the filter, `export_images.py` writes a PDF contact sheet (thumbnail, file name and `cd` = `color_dev_frac_in`, `pm` = `p_minor_max` under each image, 48 per page) and a plain list of the paths:

```bash
pixi run export --out results/                                    # filtered/kept.pdf + filtered/kept.txt
pixi run export --out results/ --table results/filtered/discarded.csv   # the discarded ones, with their reason
pixi run export --out results/ --source crop                      # the 512 px crops instead of the full frames
pixi run export --out results/ --shuffle --limit 300              # a random sample
```

The PDF is built page by page, so thousands of images are fine. `--cols`, `--rows` and `--tile` change how many thumbnails go on a page and how big they are.

### Discarding images by eye

After looking at `kept.pdf`, write the names of the images you want to drop in a text file, one per line (file name, name without extension or a full path, so you can paste from `kept.txt`; blank lines and `#` comments are ignored):

```bash
pixi run reject --out results/ --list rejected.txt --dry-run   # see what would change
pixi run reject --out results/ --list rejected.txt             # do it
pixi run export --out results/                                 # refresh kept.pdf / kept.txt
```

The matching rows move from `kept.csv` to `discarded.csv` with reason `manual review`. Names that match nothing are listed (typos, other dataset). The names are also stored in `filtered/manual_rejects.txt`, and `filter_images.py` applies that file at every run: **re-running the filter does not bring those images back**. To bring one back, delete its line from `manual_rejects.txt` and re-run `pixi run filter`. `./run.sh <dataset> filter` re-runs the filter too, so the manual rejects stay applied.

### Analysing the final dataset

```bash
pixi run analyze --out results/                       # the images left in kept.csv
pixi run analyze --out results/ --group-by crop_kind  # also the ranges per kind of crop (face / mask / full)
```

It writes `filtered/analysis/`: `report.md` (the whole analysis), `metrics.csv` (per metric: n, min, p05, p25, median, p75, p95, max, mean, std, and whether higher or lower is better), `resolution.csv`, `summary.json`, and two figures, `metrics_hist.png` and `resolution.png`. The numbers come from `results.csv`, so metrics computed after the filter (for example MPS) are included.

- **Resolution:** width and height of the frames (min, max, median, megapixels, aspect ratios, the most frequent resolutions), the kind of crops, the size in px of the face or mask in the source, and how many 512 px crops are enlarged (`upscale` > 1) and by how much.
- **Quality:** the range of every metric of stages A-C (`topiq_nr-face`, `musiq`, `clipiqa`, `brisque`, `niqe`, `maniqa`, `clip_score`, `mps_score`, the coherence measures) and `p_minor_max`. `topiq_nr-face` only exists for face crops. No-reference metrics are only comparable between images of similar resolution, which is why `--group-by` exists.