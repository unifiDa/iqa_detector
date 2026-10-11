# IQA for detector

1. initialize pixi enviroment
```
wget -qO- https://pixi.sh/install.sh | sh
```
2. clone MPS repo
```
git clone https://github.com/Kwai-Kolors/MPS.git

wget https://huggingface.co/szh/mps-pth/resolve/main/MPS_overall_checkpoint.pth -O Weights/MPS_overall_checkpoint.pth
```

3. install dependencies
```
pixi install
pixi install -e mps
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

- merge results: `pixi run merge --out /Scratch/Shullani/Dataset/results/`

- FINAL
RUN ALL: `./lamarr.sh`
Run with no preparation: `./lamarr.sh coherence quality prompt mps dataset merge`


## Dataset -- ITI

```
inpaints_flux = "/home/dasara/Research/btb-dresden-all/flux_inpaint/"
inpaints_jagg = "/home/dasara/Research/btb-dresden-all/jagg_inpaint/"
masks = "/home/dasara/Research/btb-dresden-all/btb_masks/"
prompts = "/home/dasara/Research/btb-dresden-all/miniInternVL_btb_prompts/"
original = """
```

- prepare generic
```
pixi run -e gpu prepare_generic --images /home/dasara/Research/btb-dresden-all/flux_inpaint/ --out /home/dasara/Research/results/flux/ --masks /home/dasara/Research/btb-dresden-all/btb_masks/
```

- coherence: `pixi run coherence --out /home/dasara/Research/results/flux/`
- check padding: `pixi run -e gpu quality --device cuda:0 --metrics musiq clipiqa brisque niqe maniqa --out /home/dasara/Research/results/flux/`
- quality: `pixi run -e gpu quality --device cuda:0 --metrics musiq clipiqa brisque niqe maniqa --out /home/dasara/Research/results/flux/`