# eval_suite

`eval_images.py` diviso per componente. Tutti gli script scrivono nella stessa cartella `--out` e sono collegati dalla colonna `file`.

| script | cosa fa | scrive | dipendenze | GPU |
|---|---|---|---|---|
| `prepare.py` | passo 0: maschera, volto (insightface), ritaglio 512 px | `index.csv`, `crops/<kind>/` | opencv, insightface (opz.) | sì (onnxruntime) |
| `a_coherence.py` | **A** sfocatura, tinta Lab, bordo, flag | `a_coherence.csv` | opencv | no, non serve |
| `b_quality.py` | **B** topiq_nr-face, musiq, clipiqa (anche brisque, niqe, maniqa) | `b_quality.csv` | pyiqa, torch | sì |
| `c_prompt.py` | **C** clip_score, mps_score | `c_prompt.csv` | transformers, torch; MPS: transformers 4.27.3 + repo MPS | sì |
| `d_dataset.py` | **D** FID / KID | `d_dataset.json` | clean-fid | sì |
| `merge.py` | unisce tutto | `results.csv`, `summary.json` | pandas | no |
| `gpu_check.py` | verifica che torch e onnxruntime usino la GPU | — | torch, onnxruntime | — |
| `viewer.ipynb` | ritagli ordinati per metrica | — | matplotlib, ipywidgets (opz.) | no |

`common.py` va tenuto nella stessa cartella degli script. Il `pixi.toml` si aspetta la cartella `eval_suite/` nella radice del progetto.

## Ambienti pixi

Il progetto è solo per Linux x86_64 (`platforms = ["linux-64"]`).

| ambiente | per cosa | torch | rilevamento volti |
|---|---|---|---|
| `default` | tutto tranne MPS, su GPU NVIDIA | ruote PyTorch CUDA 12.8 | `onnxruntime-gpu` |
| `cpu` | tutto tranne MPS, senza GPU | `pytorch-cpu` (conda-forge) | `onnxruntime` |
| `mps` | solo MPS, su GPU (fp16) | ruote PyTorch CUDA 12.8 | — |
| `mps-cpu` | solo MPS, senza GPU (fp32) | `pytorch-cpu` (conda-forge) | — |

```bash
# curl -fsSL https://pixi.sh/install.sh | sh

pixi install              # ambiente default (GPU)
pixi install -e cpu       # ambiente senza GPU
pixi install -e mps       # MPS su GPU
pixi install -e mps-cpu   # MPS senza GPU
```

Senza `-e`, ogni task gira nell'ambiente `default`. Per usare la CPU si aggiunge `-e cpu`: `pixi run -e cpu quality --out results/`.

### GPU

- **Requisiti:** driver NVIDIA recente. Per le RTX 50xx (Blackwell, sm_120) servono torch ≥ 2.7 compilato con CUDA ≥ 12.8, quindi l'ambiente installa torch ≥ 2.8 dalle ruote `cu128`.
- **Driver CUDA 13:** vanno bene. Un driver CUDA 13 esegue senza problemi programmi compilati per CUDA 12.x.
- **Perché cu128 e non cu130:** le ruote `cu130` funzionerebbero per torch. Però `onnxruntime-gpu` è compilato per CUDA 12 e non troverebbe le sue librerie, quindi il rilevamento volti tornerebbe sulla CPU.
- **Verifica:** prima del primo lancio esegui `pixi run gpu-check`. Lo script fa una moltiplicazione di matrici fp16 sulla GPU, apre una sessione onnxruntime su CUDA e segnala se la build di torch non supporta l'architettura della scheda.
- **Messaggi all'avvio:** ogni script che usa torch stampa la GPU in uso, per esempio `[device] cuda: NVIDIA GeForce RTX 5090 (sm_120), torch 2.x, CUDA 12.8`. `prepare.py` stampa `[face-det] insightface su CUDAExecutionProvider`. Se onnxruntime ricade sulla CPU, `prepare.py` lo segnala e il rilevamento continua sulla CPU, più lento.
- **Forzare la CPU nell'ambiente GPU:** `--device cpu` per B, C e D; `--cpu` per `prepare.py`.

### CPU

- **Cosa funziona:** tutte le metriche funzionano anche sulla CPU, MPS compreso, solo più lente. Le più pesanti sono MPS (un ViT-H), B (pyiqa) e D (clean-fid).
- **A e merge:** usano solo OpenCV e pandas, quindi hanno la stessa velocità in tutti e due gli ambienti.
- **MPS:** gira nel suo ambiente, `mps-cpu` senza GPU. Vedi sotto.

## Esempio con pixi

GPU (ambiente `default`):

```bash
pixi run gpu-check
pixi run prepare   --images imgs/ --masks masks/ --out results/
pixi run coherence --out results/
pixi run quality   --out results/
pixi run prompt    --out results/ --prompts prompts.jsonl            # CLIP
pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/  # MPS
pixi run dataset   --out results/ --ref originali/crops/face
pixi run merge     --out results/
```

CPU (ambiente `cpu`): stessi comandi con `-e cpu`, per esempio `pixi run -e cpu quality --out results/`. Per MPS si usa `-e mps-cpu` al posto di `-e mps`.

Pipeline intera, con argomenti posizionali (immagini, uscita, maschere, prompt; per non passare un argomento usa `""`):

```bash
pixi run eval_all imgs/ results/ masks/                    # prepare + A + B + merge, GPU
pixi run eval_all imgs/ results/ masks/ prompts.jsonl      # + C (solo CLIP)
pixi run -e cpu eval_all imgs/ results/ masks/             # tutto su CPU
pixi run quick_all imgs/ results/ masks/                   # solo prepare (senza volti) + A + merge
```

`eval_all` non include MPS né FID/KID: MPS ha il suo ambiente, e FID/KID richiede un insieme di riferimento.

## Esempio senza pixi

```bash
python prepare.py     --images imgs/ --masks masks/ --out res_d1
python a_coherence.py --out res_d1
python b_quality.py   --out res_d1
python c_prompt.py    --out res_d1 --prompts prompts.jsonl
python c_prompt.py    --out res_d1 --prompts prompts.jsonl --metrics mps --mps-ckpt MPS_overall_checkpoint.pth --mps-repo MPS/   # con transformers 4.27.3
python d_dataset.py   --out res_d1 --ref res_originali/crops/face
python merge.py       --out res_d1 --name dataset1
```

A, B, C e D leggono solo `index.csv` e i ritagli, quindi l'ordine è libero. Si possono anche lanciare su macchine diverse: per esempio `prepare` e A su un portatile senza GPU, poi B, C e D sulla workstation con la GPU, copiando la cartella `--out`.

`index.csv` salva i percorsi assoluti delle immagini e delle maschere. A e il notebook li rileggono, quindi se sposti la cartella su un'altra macchina quei percorsi devono esistere anche lì. B, C e D usano solo i ritagli dentro `--out`.

## MPS

Il checkpoint MPS (`MPS_overall_checkpoint.pth`) non contiene solo i pesi: è l'intero modello serializzato con pickle, comprese le classi interne di `transformers` 4.27.3 (per esempio `CLIPTextTransformer`), che nelle versioni recenti non esistono più. Con un `transformers` recente il caricamento fallisce con un errore di questo tipo:

```
AttributeError: Can't get attribute 'CLIPTextTransformer' on <module 'transformers.models.clip.modeling_clip' ...>
```

L'errore non dipende dalla GPU: avviene al caricamento, sia nell'ambiente `default` sia in `cpu`. Per questo MPS gira in un ambiente a parte, che contiene solo torch e `transformers==4.27.3`, con `tokenizers` e `huggingface-hub` della stessa epoca. Ce ne sono due varianti:

```bash
# con GPU (fp16)
pixi install -e mps
pixi run -e mps mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/

# senza GPU (fp32)
pixi install -e mps-cpu
pixi run -e mps-cpu mps results/ prompts.jsonl MPS_overall_checkpoint.pth MPS/
```

- **Argomenti:** la cartella di uscita, il file dei prompt, il checkpoint e la radice del repo [Kwai-Kolors/MPS](https://github.com/Kwai-Kolors/MPS) clonato, cioè la cartella che contiene `trainer/`. Il repo va solo clonato: il suo `requirements.txt` non va installato.
- **Dove finisce il risultato:** `mps_score` viene aggiunto a `c_prompt.csv` accanto a `clip_score`, senza ricalcolare CLIP, e poi il task aggiorna `results.csv`.
- **Su GPU** il modello gira in fp16, come nel repo originale. Se si è caricato correttamente vedi `[mps] caricato su cuda (torch.float16)`.
- **Senza GPU** (`mps-cpu`): lo script converte il modello in fp32, perché molte operazioni fp16 su CPU non esistono o sono lentissime. Il codice del repo MPS non forza mai fp16 né CUDA, quindi su CPU gira normalmente. Il risultato è lo stesso a meno di piccole differenze numeriche tra fp16 e fp32, ma è lento: è un ViT-H e servono alcuni secondi per immagine. Se si è caricato correttamente vedi `[mps] caricato su cpu (torch.float32)`.
- **Sicurezza:** il checkpoint viene caricato con `torch.load(weights_only=False)` ed esegue codice all'apertura. Scaricalo solo dal link ufficiale del repo.

## Ripresa

Ogni script riprende da dove si era fermato. Le righe già presenti vengono saltate e si calcolano solo le colonne mancanti: `b_quality.py --metrics musiq` aggiunge musiq senza rifare le altre. Usa `--overwrite` per ricominciare uno stadio.

In `a_coherence.py` i flag si ricalcolano a ogni lancio dalle colonne salvate. Per provare soglie diverse basta rilanciarlo con `--blur-thr`, `--seam-thr`, `--color-frac` o `--mono-spread`, e le immagini non vengono rielaborate. Se cambi `--ref` serve invece `--overwrite`.

## Notebook

`pixi run lab` apre JupyterLab. In `eval_suite/viewer.ipynb` imposta `SCRIPTS` (la cartella di `common.py`) e `OUT`.

`show(metrica, ...)` ordina in modo crescente e mostra i ritagli con il contorno della maschera sovrapposto, sotto un istogramma che segna i valori visualizzati. Ha due modi:

- `mode="quantili"`: N immagini dal minimo al massimo.
- `mode="pagina"`: N immagini consecutive da `start`.

Per filtrare usa `kind=` (face, mask, full, tutti) e `flag=` (es. `"sfocato"`). `show_frame(file)` mostra il frame intero con la maschera, il volto rilevato e il box del ritaglio. Il notebook non usa la GPU e gira in qualunque ambiente.

## Problemi frequenti

| sintomo | causa | soluzione |
|---|---|---|
| `no kernel image is available for execution on the device` | torch senza supporto per l'architettura della GPU (sm_120 per le RTX 50xx) | ambiente `default` (torch ≥ 2.8 cu128); controlla con `pixi run gpu-check` |
| `[device] cpu` nell'ambiente `default` | torch non vede la GPU (driver assente o troppo vecchio) | `nvidia-smi`, poi `pixi run gpu-check` |
| `insightface su CPUExecutionProvider` con la GPU | onnxruntime non trova le librerie CUDA 12 / cuDNN 9 | il rilevamento funziona lo stesso, solo più lento; `pixi run gpu-check` per i dettagli |
| `Can't get attribute 'CLIPTextTransformer'` | MPS lanciato con un `transformers` recente | `pixi run -e mps mps ...` (o `-e mps-cpu` senza GPU) |
| `No module named 'trainer'` | `--mps-repo` non punta alla radice del repo MPS | passa la cartella che contiene `trainer/` |

## Differenze rispetto a eval_images.py v2.0

- Corretto il bug per cui, se MPS veniva disattivato a metà ciclo, si saltava la metrica successiva.
- Senza un volto valido, `det_score` e `face_in_mask` ora valgono NaN invece di 0.
- `index.csv` salva i box del volto, della maschera e del ritaglio, così il notebook può proiettare la maschera sul ritaglio.
- insightface carica solo il modulo di detection, quindi è più veloce, e usa la GPU se disponibile.
- MPS funziona anche su CPU, in fp32.
- Ogni script stampa all'avvio quale device sta usando e avvisa se la GPU non è supportata.
