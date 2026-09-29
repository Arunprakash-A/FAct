"""~100K-parameter ConvMLPMixer+GAP with the activation injected as a module.

Backbone is a faithful port of the companion CNN study's ConvMLPMixer:

    LinearConv1(+BN) -> sharedMLP -> maxpool2
    LinearConv2(+BN) -> sharedMLP -> maxpool2
    -> GAP -> fc1 -> act -> dropout -> fc2

channels=(39,78), fc_hidden=128, mlp_ratio=4.0, trunc_normal_(std=0.02) init,
"linear convolution" blocks (Conv2d+BatchNorm, deliberately NO activation) --
all unchanged, so numbers here sit on the same footing as that experiment's.

The one change is the same one vit_acts.py makes to the ViT: the activation is
supplied by fixed_acts.build_act as a PARAMETER-FREE module and shared by
reference across all three activation sites (mlp1, mlp2, fc1), exactly as that
experiment's fact_k2_global / prelu_global / swish_global variants share one
module -- except here the shared module has no learnable state at all. That
keeps every variant's parameter count byte-identical, which is the property
this whole comparison rests on.

Note the contrast with the source experiment's "standard" variant, which builds
an INDEPENDENT nn.GELU() per site. For parameter-free activations independent
and shared instances are the same function, so sharing costs nothing and keeps
one convention across both architectures in this study.
"""
import torch.nn as nn

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
#: 64px for the two add-on datasets, matching cv_data.NEW_DS_IMG_SIZE. Note the
#: parameter count is INDEPENDENT of img_size here -- global average pooling
#: collapses the spatial dimensions before the classifier, so only n_classes
#: moves the total. Wall-clock does scale (the channel MLPs run at every
#: spatial position, so 64x64 carries 4x the tokens of 32x32).
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

#: cnn_fact.CHANNELS -- tuned so cifar10 lands at ~101.6K params.
CHANNELS_DEFAULT = (39, 78)
FC_HIDDEN_DEFAULT = 128
MLP_RATIO_DEFAULT = 4.0


class ChannelMLP(nn.Module):
    """Position-wise 2-layer MLP applied identically at every spatial location:
    Linear(C -> C*ratio) -> act -> Linear(C*ratio -> C)."""

    def __init__(self, c, ratio, act):
        super().__init__()
        hidden = int(c * ratio)
        self.fc1 = nn.Linear(c, hidden)
        self.act = act
        self.fc2 = nn.Linear(hidden, c)

    def forward(self, x):  # x: (B, N, C)
        return self.fc2(self.act(self.fc1(x)))


class LinearConvBlock(nn.Module):
    """'Linear convolution': Conv2d + BatchNorm2d, deliberately no activation --
    every nonlinearity in this network lives in a ChannelMLP or in fc_act."""

    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x):
        return self.bn(self.conv(x))


class ActConvMLPMixer(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.1, channels=CHANNELS_DEFAULT,
                 fc_hidden=FC_HIDDEN_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                 use_cuda_kernel=True):
        super().__init__()
        in_chans = IN_CHANS[dataset]
        n_classes = N_CLASSES[dataset]
        img_size = IMG_SIZE[dataset]
        c1, c2 = channels
        self.act_kind = act_kind

        # ONE activation module, handed to every site by reference.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.conv1 = LinearConvBlock(in_chans, c1)
        self.mlp1 = ChannelMLP(c1, mlp_ratio, self.shared_act)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = LinearConvBlock(c1, c2)
        self.mlp2 = ChannelMLP(c2, mlp_ratio, self.shared_act)
        self.pool2 = nn.MaxPool2d(2)

        assert img_size % 4 == 0, \
            f"img_size={img_size} must be divisible by 4 (two pool-2 stages)"
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(c2, fc_hidden)
        self.fc_act = self.shared_act
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(fc_hidden, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _mix(self, x, mlp):
        b, c, h, w = x.shape
        t = x.flatten(2).transpose(1, 2)
        t = mlp(t)
        return t.transpose(1, 2).reshape(b, c, h, w)

    def forward(self, x):
        x = self.conv1(x)
        x = self._mix(x, self.mlp1)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self._mix(x, self.mlp2)
        x = self.pool2(x)
        x = self.gap(x).flatten(1)
        x = self.fc_act(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def build_convmlpmixer(dataset, act_kind, dropout=0.1, channels=CHANNELS_DEFAULT,
                       fc_hidden=FC_HIDDEN_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                       use_cuda_kernel=True):
    return ActConvMLPMixer(dataset, act_kind, dropout=dropout, channels=channels,
                           fc_hidden=fc_hidden, mlp_ratio=mlp_ratio,
                           use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    """Same four-field breakdown vit_acts.count_params returns, so
    train_static.py's equal-capacity assertion works unchanged across
    architectures. "ffn_trainable" is the channel-mixing MLPs -- this
    architecture's analogue of the ViT's FFN."""
    act_ids = {id(p) for p in model.shared_act.parameters()}
    ffn = sum(p.numel() for m in (model.mlp1, model.mlp2)
              for p in m.parameters() if id(p) not in act_ids and p.requires_grad)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ffn_trainable": ffn,
        "act_trainable": sum(p.numel() for p in model.shared_act.parameters()
                             if p.requires_grad),
    }
