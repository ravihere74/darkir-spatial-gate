"""Build the 1500-pair training subset and the 100-pair validation subset
from the official LOLBlur training zips (same code we used in Colab)."""
import zipfile, os, shutil, sys, numpy as np
from tqdm import tqdm

D = sys.argv[1] if len(sys.argv) > 1 else 'LOLBlur Dataset/train_separate_zip'
OUT = sys.argv[2] if len(sys.argv) > 2 else '/content/data/train_sub'
N_TRAIN, N_VAL = 1500, 100

zl = zipfile.ZipFile(f'{D}/low_blur_noise.zip')
zh = zipfile.ZipFile(f'{D}/high_sharp_scaled.zip')

def rel(names):  # 'low_blur_noise/0223/0034.png' -> '0223/0034.png'
    return {n.split('/', 1)[1]: n for n in names if n.endswith('.png')}
L, H = rel(zl.namelist()), rel(zh.namelist())
common = sorted(set(L) & set(H))

scenes = sorted({r.split('/')[0] for r in common})
val_scenes = set(scenes[::10])  # ~10% of scenes held out for validation
def pick(items, n):
    idx = np.linspace(0, len(items) - 1, n).astype(int)
    return [items[i] for i in idx]
val = pick([r for r in common if r.split('/')[0] in val_scenes], N_VAL)
train = pick([r for r in common if r.split('/')[0] not in val_scenes], N_TRAIN)
print('scenes:', len(scenes), '| train pairs:', len(train), '| val pairs:', len(val))

for split, items in [('train', train), ('val', val)]:
    for kind in ('low', 'high'):
        os.makedirs(f'{OUT}/{split}/{kind}', exist_ok=True)
    for r in tqdm(items, desc=split):
        flat = r.replace('/', '_')
        for kind, z, M in (('low', zl, L), ('high', zh, H)):
            with z.open(M[r]) as src, open(f'{OUT}/{split}/{kind}/{flat}', 'wb') as dst:
                shutil.copyfileobj(src, dst)
