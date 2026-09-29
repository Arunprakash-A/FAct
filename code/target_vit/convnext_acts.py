"""~100K-parameter ConvNeXt-style network, sized to match the vit100k and
cnn100k (ConvMLPMixer) arms, with the activation injected as a shared,
parameter-free module -- same convention as every other arch in this study.

Faithful to ConvNeXt's actual block design (Liu et al. 2022, "A ConvNet for
the 2020s"), just scaled down:

    stem:      Conv2d(k=4, s=4) -> LayerNorm2d            (ViT-style patchify)
    stage:     N x ConvNeXtBlock(C)
    ConvNeXtBlock(C):
        DWConv2d(k=7, s=1, pad=3, groups=C)  -- spatial mixing ONLY, no
                                                 channel mixing (depthwise)
        -> LayerNorm2d(C)
        -> pointwise conv (1x1, C -> 4C)      -- channel mixing, expand
        -> act
        -> pointwise conv (1x1, 4C -> C)      -- channel mixing, project
        -> + residual
    downsample: LayerNorm2d -> Conv2d(k=2, s=2)            (between stages)
    head:      GAP -> LayerNorm2d -> Linear

Two stages (one ConvNeXtBlock each), matching convmlpmixer_acts.py's 2-stage,
1-conv-per-stage minimalism -- the point of this arch isn't to be a
competitive ConvNeXt, it's to test the study's activation-sharing question on
ConvNeXt's specific structural choices (depthwise+pointwise factorization,
dense pre-MLP LayerNorm, residual connections) at the same ~100K budget the
other two "small" arms use, for a clean 3-way structural comparison.

The one activation call site per block sits INSIDE the MLP (act between the
two pointwise convs), exactly where GELU sits in real ConvNeXt -- there is no
activation on the depthwise conv or anywhere else, matching the real
architecture (ConvNeXt has exactly one activation per block, same as this
study's ConvMLPMixer has one per ChannelMLP).

Channel widths (CHANNELS_DEFAULT) were tuned by direct param-count search
(see the __main__ block) to land in the same ~100-115K band vit100k/cnn100k
occupy, not by any formula -- same approach convmlpmixer_acts.py's
CHANNELS_DEFAULT=(39,78) comment describes for that arch.
"""
import torch
import torch.nn as nn

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

#: Tuned (see module docstring) to land at ~105-115K params, the same band
#: vit100k/cnn100k occupy.
CHANNELS_DEFAULT = (44, 88)
MLP_RATIO_DEFAULT = 4.0
BLOCKS_DEFAULT = (1, 1)


class LayerNorm2d(nn.Module):
    """Per-spatial-position LayerNorm over the channel axis of a (B,C,H,W)
    tensor -- the actual ConvNeXt convention (channels_first LayerNorm in the
    official implementation), same as resnet18_postln_acts.LayerNorm2d."""

    def __init__(self, num_channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.num_channels = num_channels
        self.eps = eps

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = nn.functional.layer_norm(x, (self.num_channels,), self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """DWConv7x7 -> LN -> pointwise(expand) -> act -> pointwise(project) -> +residual."""

    def __init__(self, c, act, mlp_ratio=MLP_RATIO_DEFAULT):
        super().__init__()
        hidden = int(c * mlp_ratio)
        self.dwconv = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
        self.norm = LayerNorm2d(c)
        self.pw1 = nn.Conv2d(c, hidden, kernel_size=1)
        self.act = act
        self.pw2 = nn.Conv2d(hidden, c, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        return x + residual


class Downsample(nn.Module):
    """LN -> strided conv -- ConvNeXt has no pooling ops anywhere; every
    spatial reduction (including the stem) is a strided/patchify conv."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.norm = LayerNorm2d(in_c)
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=2, stride=2)

    def forward(self, x):
        return self.conv(self.norm(x))


class ActConvNeXt(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.1, channels=CHANNELS_DEFAULT,
                 blocks=BLOCKS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                 use_cuda_kernel=True):
        super().__init__()
        in_chans = IN_CHANS[dataset]
        n_classes = N_CLASSES[dataset]
        c1, c2 = channels
        n1, n2 = blocks
        self.act_kind = act_kind

        # ONE activation module, handed to every block by reference -- same
        # convention as every other arch in this study.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        # Patchify stem: k=4, s=4 (ViT-style), same as real ConvNeXt.
        self.stem = nn.Sequential()
        self.stem_conv = nn.Conv2d(in_chans, c1, kernel_size=4, stride=4)
        self.stem_norm = LayerNorm2d(c1)

        self.stage1 = nn.Sequential(*[ConvNeXtBlock(c1, self.shared_act, mlp_ratio)
                                      for _ in range(n1)])
        self.downsample = Downsample(c1, c2)
        self.stage2 = nn.Sequential(*[ConvNeXtBlock(c2, self.shared_act, mlp_ratio)
                                      for _ in range(n2)])

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head_norm = LayerNorm2d(c2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(c2, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem_norm(self.stem_conv(x))
        x = self.stage1(x)
        x = self.downsample(x)
        x = self.stage2(x)
        x = self.gap(x)
        x = self.head_norm(x)
        x = x.flatten(1)
        x = self.dropout(x)
        return self.fc(x)


def build_convnext(dataset, act_kind, dropout=0.1, channels=CHANNELS_DEFAULT,
                   blocks=BLOCKS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                   use_cuda_kernel=True):
    return ActConvNeXt(dataset, act_kind, dropout=dropout, channels=channels,
                       blocks=blocks, mlp_ratio=mlp_ratio,
                       use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    act_ids = {id(p) for p in model.shared_act.parameters()}
    seen, ffn = set(), 0
    for stage in (model.stage1, model.stage2):
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


if __name__ == "__main__":
    # Channel-width search to land near vit100k/cnn100k's ~105-115K band.
    for c1 in (32, 40, 44, 48, 56):
        for ratio in (1.5, 2.0):
            c2 = int(c1 * ratio)
            m = build_convnext("cifar10", "gelu", channels=(c1, c2), use_cuda_kernel=False)
            pc = count_params(m)
            print(f"channels=({c1},{c2})  total={pc['total']:,}")
