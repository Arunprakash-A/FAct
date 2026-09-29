"""Ablation on resnet18_acts.py: insert a dedicated channel-wise LayerNorm
(LayerNorm2d, same as resnet18_postln_acts.py's) directly before EVERY call
to the shared activation -- not just the post-residual-addition sites
resnet18_postln_acts.py covers, but the post-BN1 sites too (stem + each
block's first activation).

Why: that same diagnostic pass showed even the post-BN1 sites are not
clean -- 15-21% of their pre-activations already sit outside the PWL fitting
window (stem itself: 17.6%), just not as bad as the worst post-addition
sites (up to 36.7%). resnet18_postln_acts.py only patched the addition sites
(where BatchNorm never reaches, since `out + identity` is never renormalized)
and left the BN1-preceded sites untouched on the theory that BN1 already
does the job. This module tests the literal "conv -> normalization over the
pre-activation -> activation" pattern EVERYWHERE, i.e. does giving the
post-BN1 sites their own additional LayerNorm2d change anything, and (more
importantly) does full coverage behave differently from partial coverage on
CIFAR-100, where resnet18_postln_acts.py's partial fix made things WORSE
despite provably shrinking the post-addition sites' spread as designed.

Every one of the 17 calls (1 stem + 8 blocks x 2) gets its own LayerNorm2d
instance immediately before `self.act(...)` is called on it -- 17 distinct
LayerNorm2d modules total (channel counts tied to each site's feature width),
each with its own learnable affine weight/bias, same as resnet18_postln_acts.

NOTE on equal-capacity: same caveat as resnet18_postln_acts.py -- more total
parameters than the plain resnet18_acts.py arch (now 17 LayerNorm2d instances
instead of 8), consistent across every activation tested here, but not
matched against the original resnet18 arm. A mechanism test, not a new point
in the capacity-matched sweep.
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
    tensor -- same as resnet18_postln_acts.LayerNorm2d."""

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
    """Both activation sites get their own pre-act LayerNorm2d:
    act(pre_ln1(bn1(conv1(x))))  -- site A, was already BN-fed, now ALSO LN'd
    act(pre_ln2(bn2(conv2(out)) + identity))  -- site B, same as postln_acts
    """
    expansion = 1

    def __init__(self, in_c, out_c, stride, act):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.pre_ln1 = LayerNorm2d(out_c)
        self.act = act
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.pre_ln2 = LayerNorm2d(out_c)
        if stride != 1 or in_c != out_c:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c))
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.act(self.pre_ln1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        return self.act(self.pre_ln2(out + identity))


class ActResNet18ActNorm(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                 layers=LAYERS_DEFAULT, use_cuda_kernel=True):
        super().__init__()
        in_chans = IN_CHANS[dataset]
        n_classes = N_CLASSES[dataset]
        self.act_kind = act_kind

        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.conv1 = conv3x3(in_chans, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.stem_ln = LayerNorm2d(width)

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
        x = self.shared_act(self.stem_ln(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x).flatten(1)
        return self.fc(self.dropout(x))


def build_resnet18_actnorm(dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                           layers=LAYERS_DEFAULT, use_cuda_kernel=True):
    return ActResNet18ActNorm(dataset, act_kind, dropout=dropout, width=width,
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
