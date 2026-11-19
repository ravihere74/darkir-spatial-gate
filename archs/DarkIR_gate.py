"""Spatial gate for DarkIR.

A very small CNN looks at the (dark) input photo and predicts a one-channel
map G.  At the chosen decoder stage the features are blended like this:

    F_out = G * F_restored + (1 - G) * F_original

G ranges from 0 to `amp` (we use amp = 2).  The last conv layer starts with
zero weights and zero bias, so G = amp * sigmoid(0) = 1 everywhere at the
start and the model behaves exactly like the pretrained DarkIR.

This file goes into the archs/ folder of the original DarkIR repository
(https://github.com/cidautai/DarkIR). Nothing in the original repo is changed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from archs.DarkIR import DarkIR


class DegradationGate(nn.Module):
    """Tiny CNN: dark input photo -> one-channel map G in (0, amp)."""

    def __init__(self, in_ch=3, hid=16, init_bias=0.0, amp=2.0):
        super().__init__()
        self.amp = amp
        self.conv1 = nn.Conv2d(in_ch, hid, 3, padding=1)
        self.conv2 = nn.Conv2d(hid, 1, 3, padding=1)
        # start-up: G = amp * sigmoid(init_bias). With amp=2 and bias=0 this is exactly 1.
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, init_bias)

    def forward(self, img, f_orig, f_rest):
        small = F.interpolate(img, size=f_rest.shape[-2:], mode='area')
        g = self.amp * torch.sigmoid(self.conv2(F.relu(self.conv1(small))))
        out = g * f_rest + (1 - g) * f_orig
        return out, g


class DarkIRGate(DarkIR):
    """DarkIR with gates at the decoder stages listed in `gate_stages`
    (0, 1, 2 = first, second, last decoder stage)."""

    def __init__(self, *args, gate_stages=(2,), gate_hidden=16,
                 gate_bias=0.0, gate_amp=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate_stages = tuple(gate_stages)
        self.gates = nn.ModuleDict(
            {str(i): DegradationGate(3, gate_hidden, gate_bias, gate_amp)
             for i in self.gate_stages})
        self.last_gates = {}   # saved so the G maps can be plotted later

    def forward(self, input, side_loss=False, use_adapter=None):
        _, _, H, W = input.shape
        input = self.check_image_size(input)
        x = self.intro(input)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x_light = self.middle_blks_enc(x)
        if side_loss:
            out_side = self.side_out(x_light)
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        self.last_gates = {}
        for i, (decoder, up, skip) in enumerate(zip(self.decoders, self.ups, skips[::-1])):
            x = up(x)
            x = x + skip
            if str(i) in self.gates:
                f_orig = x
                f_rest = decoder(x)
                x, g = self.gates[str(i)](input, f_orig, f_rest)
                self.last_gates[i] = g.detach()
            else:
                x = decoder(x)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W]
        if side_loss:
            return out_side, out
        return out
