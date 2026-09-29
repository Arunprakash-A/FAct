"""U-Net built from ConvMLPMixer blocks, for VOC2012 segmentation.

The point of this arm is to keep ConvMLPMixer's defining structure and change
only the task-shape. That structure is:

    LinearConvBlock : Conv2d + BatchNorm2d, DELIBERATELY NO ACTIVATION
    ChannelMLP      : Linear(C -> 4C) -> act -> Linear(4C -> C), applied
                      position-wise at every spatial location

Every nonlinearity in the network lives inside a ChannelMLP. Convolutions are
purely linear mixing followed by normalisation. Both classes are imported from
convmlpmixer_acts rather than re-implemented, so this cannot drift from the
classification arm it is meant to be comparable to.

    stem  LinearConv(3->c1, stride 2) + mlp      112x112   -> skip1
    pool  -> 56          LinearConv(c1->c2) + mlp          -> skip2
    pool  -> 28          LinearConv(c2->c3) + mlp          -> skip3
    pool  -> 14          LinearConv(c3->c4) + mlp   bottleneck
    up    -> 28   cat skip3   LinearConv(c4+c3->c3) + mlp
    up    -> 56   cat skip2   LinearConv(c3+c2->c2) + mlp
    up    -> 112  cat skip1   LinearConv(c2+c1->c1) + mlp
    up    -> 224              Conv1x1(c1 -> n_classes)

Three design notes, each a choice that could have gone the other way:

  1. STRIDE-2 STEM. A textbook U-Net runs its first stage at full resolution.
     Here that would put a ChannelMLP -- which expands 4x at EVERY spatial
     position -- on 224*224 = 50,176 tokens, ~0.8 GB of activations per layer
     at batch 32 before autograd's saved tensors. The stem halves resolution
     first, so the finest skip is at 112. For context the ViT arm's decoder
     works from a 14x14 grid, so 112 is still four scales finer.

  2. BILINEAR UPSAMPLE + LinearConvBlock, not ConvTranspose2d. Transposed
     convolution produces checkerboard artifacts at stride 2, and a plain
     Upsample keeps the "conv is linear, BN follows, activation only in the
     MLP" convention intact -- a ConvTranspose would be a third kind of layer.

  3. SKIPS CONCATENATE (U-Net's own convention) rather than add, so the
     decoder convolution sees encoder and decoder features separately.

ONE activation module is built and handed by reference to all eight
ChannelMLPs -- tied across depth and across the encoder/decoder boundary,
matching how every other arm of this study shares its activation. All eleven
activations are parameter-free, so the parameter count is identical for all of
them; that is asserted at build time.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from convmlpmixer_acts import ChannelMLP, LinearConvBlock
from fixed_acts import build_act

#: Chosen so the model lands near the vit3m arm's 3,098,199 params, which is
#: the size that trained best on this data -- not so it matches exactly. Cross
#: -ARCHITECTURE parameter equality was never a constraint in this study (the
#: ResNet-18 arm runs at 11.2M); equality across the eleven ACTIVATIONS within
#: an arm is, and that holds exactly.
CHANNELS_DEFAULT = (40, 80, 160, 320)
MLP_RATIO_DEFAULT = 4.0
EXPECTED_TOTAL_PARAMS = 2_877_781


class MixerStage(nn.Module):
    """LinearConvBlock -> ChannelMLP: one ConvMLPMixer stage, minus the pooling
    (the U-Net controls resolution itself)."""

    def __init__(self, in_c, out_c, act, mlp_ratio=MLP_RATIO_DEFAULT, stride=1):
        super().__init__()
        self.conv = LinearConvBlock(in_c, out_c)
        if stride != 1:
            self.conv.conv = nn.Conv2d(in_c, out_c, kernel_size=3,
                                       padding=1, stride=stride)
        self.mlp = ChannelMLP(out_c, mlp_ratio, act)

    def forward(self, x):
        x = self.conv(x)
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)          # (B, HW, C)
        t = self.mlp(t)
        return t.transpose(1, 2).reshape(b, c, h, w)


class UNetMixer(nn.Module):
    def __init__(self, act_kind, n_classes=21, in_chans=3,
                 channels=CHANNELS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                 use_cuda_kernel=True):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.act_kind = act_kind
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)
        a, r = self.shared_act, mlp_ratio

        self.enc1 = MixerStage(in_chans, c1, a, r, stride=2)   # 224 -> 112
        self.enc2 = MixerStage(c1, c2, a, r)                   # at 56
        self.enc3 = MixerStage(c2, c3, a, r)                   # at 28
        self.bottleneck = MixerStage(c3, c4, a, r)             # at 14
        self.pool = nn.MaxPool2d(2)

        self.dec3 = MixerStage(c4 + c3, c3, a, r)
        self.dec2 = MixerStage(c3 + c2, c2, a, r)
        self.dec1 = MixerStage(c2 + c1, c1, a, r)
        self.head = nn.Conv2d(c1, n_classes, kernel_size=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        # convmlpmixer_acts' init, verbatim.
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    @staticmethod
    def _up_cat(x, skip):
        """Upsample to the skip's spatial size and concatenate. Sizing off the
        skip rather than a fixed factor of 2 keeps the model correct for input
        sizes that are not a clean multiple of 16."""
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear",
                          align_corners=False)
        return torch.cat([x, skip], dim=1)

    def forward(self, x):
        out_size = x.shape[-2:]
        s1 = self.enc1(x)                      # 112
        s2 = self.enc2(self.pool(s1))          # 56
        s3 = self.enc3(self.pool(s2))          # 28
        b = self.bottleneck(self.pool(s3))     # 14
        d3 = self.dec3(self._up_cat(b, s3))    # 28
        d2 = self.dec2(self._up_cat(d3, s2))   # 56
        d1 = self.dec1(self._up_cat(d2, s1))   # 112
        logits = self.head(d1)
        return F.interpolate(logits, size=out_size, mode="bilinear",
                             align_corners=False)


def build_unet_mixer(act_kind, n_classes=21, channels=CHANNELS_DEFAULT,
                     mlp_ratio=MLP_RATIO_DEFAULT, use_cuda_kernel=True,
                     check_params=True):
    model = UNetMixer(act_kind, n_classes=n_classes, channels=channels,
                      mlp_ratio=mlp_ratio, use_cuda_kernel=use_cuda_kernel)
    if check_params:
        got = count_params(model)["total"]
        assert got == EXPECTED_TOTAL_PARAMS, (
            f"{act_kind}: {got} params, expected {EXPECTED_TOTAL_PARAMS} -- "
            "every activation under test is parameter-free, so this must be "
            "identical across all eleven arms.")
    return model


def count_params(model):
    act_ids = {id(p) for p in model.shared_act.parameters()}
    ffn = sum(p.numel() for n, m in model.named_children()
              if isinstance(m, MixerStage)
              for p in m.mlp.parameters()
              if id(p) not in act_ids and p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"total": total,
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "ffn_trainable": ffn,
            "act_trainable": sum(p.numel() for p in model.shared_act.parameters()
                                 if p.requires_grad),
            "decoder": sum(p.numel() for m in (model.dec3, model.dec2,
                                               model.dec1, model.head)
                           for p in m.parameters())}
