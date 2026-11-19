"""Train the gate (or a control / the whole network) on the LOLBlur subset.

Settings used for the reported results (the defaults below are already these):
  python train_gate.py --mode gate1 --tag _amp2 --freeze_body --seed 0     # L1 loss
  python train_gate.py --mode gate1 --tag _combo --loss combo --freeze_body --seed 0
The first-round experiment (whole network fine-tuned, no --freeze_body) used
--gate_amp 1.0 --gate_bias 3.0 --gate_lr 1e-3, see the README.
"""
import argparse, glob, json, os, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from archs.DarkIR import DarkIR
from archs.DarkIR_gate import DarkIRGate

p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['control', 'gate1', 'gate2'])
p.add_argument('--seed', type=int, default=0)
p.add_argument('--epochs', type=int, default=10)
p.add_argument('--bs', type=int, default=8)
p.add_argument('--patch', type=int, default=256)
p.add_argument('--lr', type=float, default=2e-5)        # for the pretrained network (when not frozen)
p.add_argument('--gate_lr', type=float, default=2e-3)   # for the new gate
p.add_argument('--data', default='/content/data/train_sub')
p.add_argument('--out', default='checkpoints')
p.add_argument('--pretrained', default='models/DarkIR_384.pt')
p.add_argument('--gate_bias', type=float, default=0.0)
p.add_argument('--freeze_body', action='store_true')
p.add_argument('--gate_amp', type=float, default=2.0)
p.add_argument('--loss', default='l1', choices=['l1', 'combo'])
p.add_argument('--w_ssim', type=float, default=0.2)
p.add_argument('--w_lpips', type=float, default=0.2)
p.add_argument('--tag', default='')
p.add_argument('--resume', action='store_true')
a = p.parse_args()

random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
dev = 'cuda'
name = f'{a.mode}{a.tag}_s{a.seed}'
os.makedirs(a.out, exist_ok=True)


class Pairs(Dataset):
    def __init__(self, root, patch, train):
        self.low = sorted(glob.glob(f'{root}/low/*.png'))
        self.high = [q.replace('/low/', '/high/') for q in self.low]
        self.patch, self.train = patch, train
    def __len__(self): return len(self.low)
    def load(self, f):
        return torch.from_numpy(np.array(Image.open(f).convert('RGB'))).permute(2, 0, 1).float() / 255
    def __getitem__(self, i):
        lo, hi = self.load(self.low[i]), self.load(self.high[i])
        _, h, w = lo.shape; q = self.patch
        if self.train:
            y, x = random.randint(0, h - q), random.randint(0, w - q)
        else:
            y, x = (h - q) // 2, (w - q) // 2
        lo, hi = lo[:, y:y+q, x:x+q], hi[:, y:y+q, x:x+q]
        if self.train and random.random() < 0.5:
            lo, hi = lo.flip(-1), hi.flip(-1)
        return lo, hi


loader = DataLoader(Pairs(f'{a.data}/train', a.patch, True), batch_size=a.bs, shuffle=True,
                    num_workers=2, drop_last=True, pin_memory=True)
vloader = DataLoader(Pairs(f'{a.data}/val', 512, False), batch_size=4, shuffle=False, num_workers=2)

cfg = dict(img_channel=3, width=32, middle_blk_num_enc=2, middle_blk_num_dec=2,
           enc_blk_nums=[1, 2, 3], dec_blk_nums=[3, 1, 1], dilations=[1, 4, 9], extra_depth_wise=True)
if a.mode == 'control':
    model = DarkIR(**cfg)
else:
    model = DarkIRGate(gate_stages=(2,) if a.mode == 'gate1' else (0, 1, 2),
                       gate_bias=a.gate_bias, gate_amp=a.gate_amp, **cfg)
ckpt = torch.load(a.pretrained, map_location='cpu', weights_only=False)['params']
print(model.load_state_dict(ckpt, strict=False))
model.to(dev)
if a.freeze_body:
    for n_, q_ in model.named_parameters():
        if not n_.startswith('gates.'):
            q_.requires_grad = False

if a.loss == 'combo':
    from pytorch_msssim import ssim as ssim_fn
    from lpips import LPIPS
    lp_fn = LPIPS(net='vgg', verbose=False).to(dev).eval()
    for q_ in lp_fn.parameters():
        q_.requires_grad = False

gate_p = [q for n, q in model.named_parameters() if n.startswith('gates.')]
body_p = [q for n, q in model.named_parameters() if not n.startswith('gates.') and q.requires_grad]
groups = ([{'params': body_p, 'lr': a.lr}] if body_p else []) + \
         ([{'params': gate_p, 'lr': a.gate_lr}] if gate_p else [])
opt = torch.optim.AdamW(groups, weight_decay=0)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs * len(loader))


@torch.no_grad()
def validate():
    model.eval(); tot, n = 0, 0
    for lo, hi in vloader:
        out = model(lo.to(dev)).clamp(0, 1)
        mse = ((out - hi.to(dev)) ** 2).mean((1, 2, 3))
        tot += (10 * torch.log10(1 / mse)).sum().item(); n += len(lo)
    model.train()
    return tot / n


last, best_path = f'{a.out}/{name}_last.pt', f'{a.out}/{name}_best.pt'
start, best = 0, -1.0
if a.resume and os.path.exists(last):
    s = torch.load(last, map_location='cpu', weights_only=False)
    model.load_state_dict(s['params']); opt.load_state_dict(s['opt'])
    sched.load_state_dict(s['sched'])
    start, best = s['epoch'], s['best']
    print('resumed from epoch', start)

log = open(f'{a.out}/{name}_log.txt', 'a')
def write(d):
    log.write(json.dumps(d) + '\n'); log.flush(); print(d)

write({'args': vars(a)})
if start == 0:
    write({'epoch': 0, 'val_psnr': round(validate(), 4)})

model.train()
for ep in range(start, a.epochs):
    t0, run, skipped = time.time(), 0.0, 0
    for lo, hi in loader:
        lo, hi = lo.to(dev, non_blocking=True), hi.to(dev, non_blocking=True)
        out = model(lo)              # full fp32 (fp16 gave NaN for us)
        loss = F.l1_loss(out, hi)
        if a.loss == 'combo':
            loss = loss + a.w_ssim * (1 - ssim_fn(out, hi, data_range=1.0)) \
                        + a.w_lpips * lp_fn(out, hi, normalize=True).mean()
        if not torch.isfinite(loss):
            skipped += 1
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run += loss.item()
    v = validate()
    if v > best:
        best = v
        torch.save({'params': model.state_dict()}, best_path)
    torch.save({'params': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                'epoch': ep + 1, 'best': best}, last)
    write({'epoch': ep + 1, 'train_loss': round(run / len(loader), 5), 'val_psnr': round(v, 4),
           'skipped': skipped, 'best_val': round(best, 4), 'minutes': round((time.time() - t0) / 60, 2)})
