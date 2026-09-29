"""Conv1d front-end -> LayerNorm MLP, activation injected as a shared module.

Next step in the architecture sweep. The MLP arm established that FAct's
deficit on a plain MLP is a pre-activation SCALE problem: with LayerNorm it
goes from 10/11 to a tie for 1st. But a tie is not the decisive 1/11 it holds
on the ViT and ConvMLPMixer, so normalisation is necessary and not sufficient.
This arm adds the next ingredient those two have and the MLP does not --
convolution, i.e. local weight sharing over the input.

    flatten -> (B, 1, 784) -> Conv1d(1, 100, k, stride) -> pool(8) -> LayerNorm
            -> fc(W) -> LN -> act -> fc(W) -> LN -> act -> fc(W) -> LN -> act
            -> dropout -> head(10)

The convolution carries NO activation of its own, matching
convmlpmixer_acts.LinearConvBlock's "linear convolution" convention (Conv+norm,
nonlinearity only inside the MLP) -- so every nonlinearity in this network is
still the one shared activation module, handed by reference to all three MLP
sites exactly as in every other arm.

A 1D convolution over a FLATTENED 2D image is a deliberate simplification, not
an oversight: a kernel that crosses a row boundary is mixing pixels that are
not neighbours in the image. This arm therefore tests weight sharing and local
receptive fields WITHOUT the 2D spatial prior, which is the isolated variable
the ConvMLPMixer arm cannot provide.

CAPACITY. Conv1d(1, 100, k=9) has 1,000 parameters where the plain MLP's first
layer (784x100) had 78,500 -- that is what weight sharing buys. With the
position code kept at length 8 the model lands at 104,510 parameters against
the mlp_ln arm's 100,310, so the two are on the same ~100K budget and a
difference between them is attributable to the convolution rather than to
capacity.
"""
import torch.nn as nn

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

CONV_CHANNELS_DEFAULT = 100     # "outputs 100 features" = 100 feature maps
KERNEL_DEFAULT = 9
STRIDE_DEFAULT = 4
#: Positions kept after pooling the conv output, BEFORE the MLP.
#:
#: This was 1 (a true global average pool) in the first version and that model
#: is a dead end: pooling 196 positions to one leaves only each filter's mean
#: response over the whole image, and a probe run reached 32% val accuracy in 4
#: epochs where the plain LayerNorm MLP reached 53% in ONE. Comparing eleven
#: activations inside a 32%-accuracy bottleneck measures the bottleneck.
#:
#: 8 keeps a coarse position code (100 maps x 8 positions = 800 numbers into
#: the MLP) and lands the model at 104,510 parameters -- on the study's ~100K
#: budget with the MLP width left at 100, so this arm is directly comparable to
#: mlp_ln (100,310) without needing a widened variant.
POOL_LEN_DEFAULT = 8
WIDTH_DEFAULT = 100
DEPTH_DEFAULT = 3


class ActConv1dMLP(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.1, width=WIDTH_DEFAULT,
                 depth=DEPTH_DEFAULT, conv_channels=CONV_CHANNELS_DEFAULT,
                 kernel_size=KERNEL_DEFAULT, stride=STRIDE_DEFAULT,
                 pool_len=POOL_LEN_DEFAULT, use_cuda_kernel=True):
        super().__init__()
        in_len = IN_CHANS[dataset] * IMG_SIZE[dataset] ** 2
        n_classes = N_CLASSES[dataset]
        self.act_kind = act_kind
        self.in_len = in_len

        # ONE activation module, handed to every site by reference.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.flatten = nn.Flatten()
        self.conv = nn.Conv1d(1, conv_channels, kernel_size=kernel_size,
                              stride=stride, padding=kernel_size // 2)
        # Average-pool position down to pool_len, keeping a coarse position
        # code rather than collapsing it entirely (see POOL_LEN_DEFAULT).
        self.gap = nn.AdaptiveAvgPool1d(pool_len)
        feat = conv_channels * pool_len
        self.conv_norm = nn.LayerNorm(feat)

        dims = [feat] + [width] * depth
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.norms = nn.ModuleList([nn.LayerNorm(width) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x).unsqueeze(1)          # (B, 1, in_len)
        x = self.conv(x)                          # (B, C, L')
        x = self.gap(x).flatten(1)                # (B, C * pool_len)
        x = self.conv_norm(x)
        for fc, nrm in zip(self.hidden, self.norms):
            x = self.shared_act(nrm(fc(x)))
        return self.head(self.dropout(x))


def build_conv1d_mlp(dataset, act_kind, dropout=0.1, width=WIDTH_DEFAULT,
                     depth=DEPTH_DEFAULT, conv_channels=CONV_CHANNELS_DEFAULT,
                     kernel_size=KERNEL_DEFAULT, stride=STRIDE_DEFAULT,
                     pool_len=POOL_LEN_DEFAULT, use_cuda_kernel=True):
    return ActConv1dMLP(dataset, act_kind, dropout=dropout, width=width,
                        depth=depth, conv_channels=conv_channels,
                        kernel_size=kernel_size, stride=stride,
                        pool_len=pool_len, use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    """Same four-field breakdown the other arms return. "ffn_trainable" is the
    conv front-end plus the hidden MLP -- everything but the classifier."""
    act_ids = {id(p) for p in model.shared_act.parameters()}
    mods = [model.conv, model.conv_norm] + list(model.hidden) + list(model.norms)
    ffn = sum(p.numel() for m in mods for p in m.parameters()
              if p.requires_grad and id(p) not in act_ids)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ffn_trainable": ffn,
        "act_trainable": sum(p.numel() for p in model.shared_act.parameters()
                             if p.requires_grad),
    }
