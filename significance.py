"""Compare gate runs against the original model, scene by scene.

  python significance.py pretrained_full gate1_amp2_full gate1_amp2_s1_full gate1_combo_s0_last_full

First name = baseline, the others are compared against it. Photos of one scene
are very similar, so we average the per-photo differences inside each scene and
report mean +- standard error over scenes (30 scenes in the LOLBlur test set).
"""
import json, os, sys
import numpy as np

R = os.environ.get('RESULTS_DIR', 'results')   # folder with the json files from eval_gate.py
def load(n):
    return {r['file']: r for r in json.load(open(f'{R}/{n}.json'))['per_image']}

names = sys.argv[1:]
B = load(names[0])
files = sorted(B)
scenes = sorted({f.split('/')[0] for f in files})
print(len(files), 'photos,', len(scenes), 'scenes')
for name in names[1:]:
    A = load(name)
    print(name)
    for k in ('psnr', 'ssim', 'lpips'):
        per = np.array([np.mean([A[f][k] - B[f][k] for f in files if f.startswith(s + '/')]) for s in scenes])
        se = per.std(ddof=1) / np.sqrt(len(per))
        better = (per < 0).mean() if k == 'lpips' else (per > 0).mean()
        print(f'  {k}: diff {per.mean():+.5f} +- {se:.5f} (by scene) | better in {better*100:.0f}% of {len(scenes)} scenes')
