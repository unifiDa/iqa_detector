# IQA for detector

1. initialize pixi enviroment
```
wget -qO- https://pixi.sh/install.sh | sh
```
2. clone MPS repo
```
git clone https://github.com/Kwai-Kolors/MPS.git
```

## Dataset -- LESC

```
inpaints = "/Prove/Shullani/Archive/fooocus-cinematic/inpaints/"
masks = "/Prove/Shullani/Archive/fooocus-cinematic/masks/"
original = "/Prove/Shullani/Archive/fooocus-cinematic/original/"
```

- prepare con facce
```
pixi run -e gpu prepare --images /Scratch/Shullani/Dataset/fooocus-cinematic/inpaints/ --out /Scratch/Shullani/Dataset/results/ --masks /Scratch/Shullani/Dataset/fooocus-cinematic/masks/
```

- coherence: `pixi run coherence --out /Scratch/Shullani/Dataset/results/`
- quality: `pixi run -e gpu quality --device cuda:0 --out /Scratch/Shullani/Dataset/results/`
- clip score: `pixi run -e gpu prompt --device cuda:0  --out /Scratch/Shullani/Dataset/results/ --prompts /Scratch/Shullani/Dataset/fooocus-cinematic/prompts.jsonl `
- mps score: `pixi run -e mps mps /Scratch/Shullani/Dataset/results/ /Scratch/Shullani/Dataset/fooocus-cinematic/prompts.jsonl /data/staff/shullani/Documents/Research/IAPP/DETECTOR/iqa_detector/Weights/MPS_overall_checkpoint.pth /data/staff/shullani/Documents/Research/IAPP/DETECTOR/iqa_detector/MPS/`