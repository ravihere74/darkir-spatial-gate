"""Evaluate DarkIR / DarkIR+gate on the LOLBlur test set.

Scores (PSNR, SSIM, LPIPS-VGG) are computed per image on full-size, unclamped
outputs, like the original repository's testing code, and saved to a json.

  python eval_gate.py --name pretrained_full --mode pretrained --ckpt models/DarkIR_384.pt
  python eval_gate.py --name gate1_amp2_full  --mode gate1 --ckpt checkpoints/gate1_amp2_s0_best.pt
"""
import argparse, glob, json, os, sys, time
import numpy as np, torch
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from archs.DarkIR import DarkIR
from archs.DarkIR_gate import DarkIRGate
from utils.test_utils import calc_SSIM   # the repo's own SSIM
from lpips import LPIPS                  # LPIPS with VGG, as in the repo

p = argparse.ArgumentParser()
p.add_argument('--name', required=True)                  # label for the results file
p.add_argument('--mode', required=True, choices=['pretrained', 'control', 'gate1', 'gate2'])
p.add_argument('--ckpt', required=True)
p.add_argument('--test', default='/content/data/test')
p.add_argument('--out', default='results')
p.add_argument('--gate_amp', type=float, default=2.0)
p.add_argument('--limit', type=int, default=0)           # 0 = all 1800 images
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
dev = 'cuda'

cfg = dict(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
           enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9], extra_depth_wise=True)
if a.mode in ('pretrained', 'control'):
    model = DarkIR(**cfg)
else:
    model = DarkIRGate(gate_stages=(2,) if a.mode == 'gate1' else (0, 1, 2),
                       gate_amp=a.gate_amp, **cfg)
sd = torch.load(a.ckpt, map_location='cpu', weights_only=False)['params']
print(model.load_state_dict(sd))   # strict: any mismatch raises an error
model.to(dev).eval()
n_params = sum(q.numel() for q in model.parameters())

lows = sorted(glob.glob(f'{a.test}/low_blur_noise/*/*.*'))
if a.limit:
    lows = lows[::max(1, len(lows) // a.limit)][:a.limit]
print('images:', len(lows), '| params:', n_params)

lp = LPIPS(net='vgg', verbose=False).to(dev)
def load(f):
    return torch.from_numpy(np.array(Image.open(f).convert('RGB'))).permute(2, 0, 1).float().div(255)[None].to(dev)

rows, t0, infer = [], time.time(), 0.0
with torch.no_grad():
    for i, f in enumerate(lows):
        lo = load(f)
        hi = load(f.replace('/low_blur_noise/', '/high_sharp_scaled/'))
        torch.cuda.synchronize(); t1 = time.time()
        out = model(lo)                      # NOT clamped, same as the repo's test code
        torch.cuda.synchronize(); infer += time.time() - t1
        mse = torch.mean((hi - out) ** 2)
        r = {'file': f.split('/low_blur_noise/')[1],
             'psnr': (20 * torch.log10(1. / torch.sqrt(mse))).item(),
             'ssim': calc_SSIM(out, hi).item(),
             'lpips': torch.mean(lp(out, hi)).item()}
        if hasattr(model, 'last_gates') and model.last_gates:
            r['gate_mean'] = {str(k): round(v.mean().item(), 4) for k, v in model.last_gates.items()}
        rows.append(r)
        if (i + 1) % 100 == 0:
            print(i + 1, '/', len(lows), f'{(time.time()-t0)/60:.1f} min', flush=True)

summary = {k: float(np.mean([r[k] for r in rows])) for k in ('psnr', 'ssim', 'lpips')}
summary.update({'name': a.name, 'n_images': len(rows), 'params': n_params,
                'sec_per_image': infer / len(rows)})
json.dump({'summary': summary, 'per_image': rows},
          open(f'{a.out}/{a.name}{"_limit"+str(a.limit) if a.limit else ""}.json', 'w'))
print(json.dumps(summary, indent=1))
