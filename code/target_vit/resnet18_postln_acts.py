"""Ablation on resnet18_acts.py: add a channel-wise LayerNorm right after the
residual addition, before the (shared) activation -- i.e. `self.act(out +
identity)` becomes `self.act(post_ln(out + identity))`.

Why: a diagnostic pass over the real trained checkpoints (hooks over every
activation site, not shipped here) found the shared curve is invoked
17x/forward in this arch (vs. 3x in
ConvMLPMixer), and the post-residual-addition call sites specifically are
where pre-activation scale compounds with depth (19.6% of ALL activations
land outside the +-1.211 window the PWL variants were fit on; the deepest
post-addition site alone is 36.7% outside). `out + identity` is never
renormalized -- BatchNorm only covers the conv branch (bn2), not the
accumulated residual stream -- unlike ViT, where the FFN's activation is
always fed from a freshly LayerNorm'd input regardless of depth (vit_acts.py:
self.act(self.fc1(x)) follows norm2(x)). This module tests whether adding
that missing renormalization at the ONE place variance was shown to compound
rescues fact_fixed the way LayerNorm already does in the ViT arm.

LayerNorm2d follows the ConvNeXt convention: normalize each spatial position
across channels (permute to channels-last, F.layer_norm, permute back) --
the direct analogue of ViT's LayerNorm(dim) over the token/embedding axis,
not nn.BatchNorm2d's per-channel-over-batch+spatial statistic.

Everything else -- block structure, stem, GAP+fc head, width/layers
parameterization, init, the ONE shared activation module convention -- is
identical to resnet18_acts.py. This is deliberately a separate file (not an
in-place edit) so the canonical resnet18_acts.py and its existing
runs_resnet100k/ results are untouched; this ablation writes to
runs_resnet100k_postln/ instead.

NOTE on equal-capacity: LayerNorm2d has learnable weight/bias per channel
(unlike the parameter-free activations), so this arch has MORE total
parameters than resnet18_acts.py at the same width/layers -- 2*sum(channels)
extra scalars (one LayerNorm per block's addition site), a few hundred at
width=6. That is fine WITHIN this ablation (still identical across all
activations tested here, so the equal-capacity comparison the activation
ranking rests on is intact) but this arch is NOT parameter-matched against
the original resnet18_acts.py arm -- it isn't meant to be; it's a targeted
mechanism test, not a new point in the main capacity-matched sweep.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

LAYERS_DEFAULT = (2, 2, 2, 2)
WIDTH_DEFAULT = 64


def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class LayerNorm2d(nn.Module):
    """Per-spatial-position LayerNorm over the channel axis of a (B,C,H,W)
    tensor -- the ConvNeXt convention, and the direct analogue of ViT's
    LayerNorm(dim) over the token axis."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.num_channels = num_channels
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class BasicBlock(nn.Module):
    """Same as resnet18_acts.BasicBlock, except the residual-addition site
    is followed by LayerNorm2d before the shared activation:
    `act(post_ln(out + identity))` instead of `act(out + identity)`."""
    expansion = 1

    def __init__(self, in_c, out_c, stride, act):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = act
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.post_ln = LayerNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(self.post_ln(out + identity))


class ActResNet18PostLN(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                 layers=LAYERS_DEFAULT, use_cuda_kernel=True):
        super().__init__()
        in_chans = IN_CHANS[dataset]
        n_classes = N_CLASSES[dataset]
        self.act_kind = act_kind

        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.conv1 = conv3x3(in_chans, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.in_c = width
        self.layer1 = self._make_layer(width, layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(width * 8, n_classes)

        self._init_weights()

    def _make_layer(self, out_c, blocks, stride):
        layers = []
        for i in range(blocks):
            layers.append(BasicBlock(self.in_c, out_c,
                                     stride if i == 0 else 1, self.shared_act))
            self.in_c = out_c
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.shared_act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.dropout(x))


def build_resnet18_postln(dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                          layers=LAYERS_DEFAULT, use_cuda_kernel=True):
    return ActResNet18PostLN(dataset, act_kind, dropout=dropout, width=width,
                             layers=layers, use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    act_ids = {id(p) for p in model.shared_act.parameters()}
    seen, ffn = set(), 0
    for stage in (model.layer1, model.layer2, model.layer3, model.layer4):
        for p in stage.parameters():
            if p.requires_grad and id(p) not in act_ids and id(p) not in seen:
                seen.add(id(p))
                ffn += p.numel()
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ffn_trainable": ffn,
        "act_trainable": sum(p.numel() for p in model.shared_act.parameters()
                             if p.requires_grad),
    }
