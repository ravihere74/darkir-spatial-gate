# A Tiny Spatial Gate for DarkIR

Class project (AI-ML). We test whether a very small "gate" (593 parameters) added to
[DarkIR](https://github.com/cidautai/DarkIR), a low-light image restoration network,
can improve its results on the LOLBlur dataset. The pretrained DarkIR weights stay
frozen and only the gate is trained.

**Short answer:** the gate gives a very small but repeatable improvement in LPIPS
(0.16201 to 0.16191 with L1 loss, 0.16178 with a combined loss). PSNR and SSIM do not
clearly change. The change cannot be seen by eye. Full details are in the paper:
[`paper/AIML_Project.pdf`](paper/AIML_Project.pdf).

## What the gate does

At the last decoder stage of DarkIR, the features before the block (`F_orig`) and after
the block (`F_res`) are mixed with a per-pixel map `G` predicted from the dark input photo:

```
F_out = G * F_res + (1 - G) * F_orig        G = 2 * sigmoid(conv(relu(conv(x))))
```

`G` is between 0 and 2 and starts at exactly 1 (zero-initialised last layer), so the
model is identical to the original DarkIR before training.

## Results (full LOLBlur test set, 1800 images)

| Model | PSNR | SSIM | LPIPS (lower is better) | Params |
|---|---|---|---|---|
| DarkIR-m (original) | 27.00288 | 0.88291 | 0.16201 | 3,321,638 |
| + Gate, L1 loss, seed 0 | 27.00845 | 0.88299 | 0.16191 | 3,322,231 |
| + Gate, L1 loss, seed 1 | 27.00244 | 0.88282 | 0.16193 | 3,322,231 |
| + Gate, combined loss | 26.98631 | 0.88307 | 0.16178 | 3,322,231 |

## Files

| File | What it is |
|---|---|
| `archs/DarkIR_gate.py` | The gate module and the modified model (a subclass of the original `DarkIR`) |
| `train_gate.py` | Training script (control, gate at one or three stages, frozen network, L1 or combined loss) |
| `eval_gate.py` | Evaluation on the test set, saves per-image scores to `results/*.json` |
| `significance.py` | Per-scene comparison of a gate run against the original model |
| `make_subset.py` | Builds our 1500-pair train / 100-pair validation subset from the LOLBlur zips |
| `run_colab.ipynb` | Step by step notebook that runs everything in Google Colab |
| `paper/` | The paper (LaTeX source and PDF) |
| `figures/` | Gate-map figures used in the paper |

Our files are added **on top of** the original DarkIR repository. Nothing in the original
code is modified.

## How to run (easiest way: Google Colab)

1. Open `run_colab.ipynb` in Google Colab (File > Open notebook > GitHub tab, paste this
   repository's link). Choose a GPU runtime (Runtime > Change runtime type > T4 GPU).
2. Put the LOLBlur dataset in your Google Drive. It can be downloaded from the
   [LEDNet project page](https://github.com/sczhou/LEDNet). The notebook has one line at the
   top where you set the folder name.
3. Run the cells from top to bottom. Rough times on a free Colab GPU: baseline evaluation about
   15 minutes, one training run about 20 minutes, one full test evaluation about 15 minutes.

For a quick check that everything works, use the `--limit 50` evaluation cell, which only
scores 50 test images.

## How to run (manually)

```bash
git clone https://github.com/cidautai/DarkIR.git
cd DarkIR
sed -i -E 's/==([0-9]+\.[0-9]+(\.[0-9]+)?)/>=\1/' requirements.txt   # loosen old version pins
pip install -r requirements.txt
pip install lpips ptflops pytorch-msssim huggingface_hub

# download the pretrained DarkIR weights (gives models/DarkIR_384.pt)
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='Cidaut/DarkIR', local_dir='models')"

# copy this project's files into the DarkIR folder
cp /path/to/this/repo/archs/DarkIR_gate.py archs/
cp /path/to/this/repo/{train_gate.py,eval_gate.py,significance.py,make_subset.py} .

# data: test set in /content/data/test/{low_blur_noise,high_sharp_scaled}/<scene>/*.png
python make_subset.py "<folder with LOLBlur train_separate_zip>" /content/data/train_sub

# 1. score the original model
python eval_gate.py --name pretrained_full --mode pretrained --ckpt models/DarkIR_384.pt
# 2. train the gate (L1 loss), then score it
python train_gate.py --mode gate1 --tag _amp2 --freeze_body --seed 0
python eval_gate.py --name gate1_amp2_full --mode gate1 --ckpt checkpoints/gate1_amp2_s0_best.pt
# 3. combined loss (L1 + 0.2*(1-SSIM) + 0.2*LPIPS)
python train_gate.py --mode gate1 --tag _combo --loss combo --freeze_body --seed 0
python eval_gate.py --name gate1_combo_s0_last_full --mode gate1 --ckpt checkpoints/gate1_combo_s0_last.pt
# 4. compare against the original, scene by scene
python significance.py pretrained_full gate1_amp2_full gate1_combo_s0_last_full
```

A GPU is needed (the scripts use CUDA). Other runs from the paper: use `--seed 1` for the second
seed, `--epochs 30` for the longer-training check, and drop `--freeze_body` (with `--gate_amp 1.0
--gate_bias 3.0 --gate_lr 1e-3`, or `--mode control`) for the first-round experiment in which the whole
network was fine-tuned.

## Things to know

- We trained on a 1500-pair subset because we only had free Colab GPUs and the authors' training
  code is not public. The pretrained model has already seen the full training set.
- The gains are tiny. A constant-scale control (a gate with a single number instead of a map)
  was **not** run, so we cannot say that the spatial pattern is what helps. See the paper's
  limitations section.
- Exact numbers can differ a little between runs because of GPU randomness. The seed 1 run in the
  paper was interrupted and resumed in the middle.
- Training was done in full 32-bit precision. Half precision gave NaN values for us.

## Credits

DarkIR: Feijoo, Benito, Garcia, Conde, "DarkIR: Robust Low-Light Image Restoration", CVPR 2025.
Code and weights are from <https://github.com/cidautai/DarkIR> and are not included here.
LOLBlur dataset: Zhou, Li, Loy, "LEDNet: Joint Low-light Enhancement and Deblurring in the Dark", ECCV 2022.

Our own code in this repository is under the MIT license. DarkIR keeps its own license.
