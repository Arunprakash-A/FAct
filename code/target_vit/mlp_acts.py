"""~100K-parameter plain MLP on flattened inputs, activation injected as a
shared module.

The simplest possible member of this study: no convolution, no attention, no
spatial structure at all -- the 28x28 image is flattened to a 784-vector and
the network is four fully-connected layers. That is the point. The ViT arm has
patch embedding and self-attention, the ConvMLPMixer arm has convolutions and
spatial pooling; if the activation ranking those two produce is really a
property of the ACTIVATION rather than of the architecture around it, it should
survive stripping the architecture down to matrix multiplies.

    flatten(784) -> fc1(100) -> [norm] -> act -> fc2(100) -> [norm] -> act
                 -> fc3(100) -> [norm] -> act -> dropout -> fc4(10)

`norm` is one of none / batch / layer. It exists because the plain variant is
the only architecture in this study with NO normalisation anywhere, while the
ViT arm has LayerNorm and the ConvMLPMixer arm has BatchNorm -- so
normalisation is a standing confound in any comparison between them. The norm
sits immediately BEFORE the activation, which is where it bears on the
question: it fixes the scale of the pre-activation distribution, and a
periodic, bounded activation is far more sensitive to that scale than a
monotone unbounded one (the frozen FAct curve has ten turning points on
[-8, 8], so pre-activations that drift wide wrap through several periods).

Normalisation adds 2*width parameters per hidden layer -- 600 on this model,
0.6% of the total. That is identical for all eleven activations, so the
equal-capacity guarantee within each variant is unaffected; only the
plain-vs-normalised comparison carries the 600-parameter difference, and it is
reported rather than hidden by re-tuning the width.

    fc1  784x100 + 100 = 78,500
    fc2  100x100 + 100 = 10,100
    fc3  100x100 + 100 = 10,100
    fc4  100x10  +  10 =  1,010
                         -------
                          99,710 parameters

Width 100 with three hidden layers is chosen to land on ~100K, the same budget
the ViT (105,098 on FMNIST) and the ConvMLPMixer (101,609) were built to, so
the three arms are compared at equal capacity. Three hidden layers rather than
one because the activation is the object of study and a single-hidden-layer net
would give it exactly one site; three sites also matches the "shared across
depth" convention the other arms use.

ONE activation module is built per network and handed by reference to all three
sites -- identical to vit_acts.py and convmlpmixer_acts.py, and to how
fact_k2_global trained the curve being tested (tied across neurons AND depth).
Init is trunc_normal_(std=0.02) with zero biases, matching the two ~100K arms
(NOT the ResNet-18 arm, which needs He init to train 20 layers).

No activation in ACT_KINDS has learnable parameters, so all eleven variants
have a byte-identical parameter count -- asserted per run by train_static.py.
"""
import torch.nn as nn

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

#: Three hidden layers of this width. Unlike the two GAP-based arms, an MLP's
#: parameter count DOES depend on the input size (fc1 is in_features x width),
#: so a non-FMNIST dataset would land well off 100K at this width -- which is
#: why the study instruction scoped this arm to FMNIST.
WIDTH_DEFAULT = 100
DEPTH_DEFAULT = 3
NORMS = ("none", "batch", "layer")


def _make_norm(kind, width):
    if kind == "none":
        return nn.Identity()
    if kind == "batch":
        return nn.BatchNorm1d(width)
    if kind == "layer":
        return nn.LayerNorm(width)
    raise ValueError(f"unknown norm {kind!r} (expected one of {NORMS})")


class ActMLP(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.1, width=WIDTH_DEFAULT,
                 depth=DEPTH_DEFAULT, norm="none", use_cuda_kernel=True):
        super().__init__()
        self.norm_kind = norm
        in_features = IN_CHANS[dataset] * IMG_SIZE[dataset] ** 2
        n_classes = N_CLASSES[dataset]
        self.act_kind = act_kind
        self.in_features = in_features

        # ONE activation module, handed to every site by reference.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.flatten = nn.Flatten()
        dims = [in_features] + [width] * depth
        self.hidden = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(depth)])
        self.norms = nn.ModuleList([_make_norm(norm, width) for _ in range(depth)])
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.flatten(x)
        for fc, nrm in zip(self.hidden, self.norms):
            x = self.shared_act(nrm(fc(x)))
        return self.head(self.dropout(x))


def build_mlp(dataset, act_kind, dropout=0.1, width=WIDTH_DEFAULT,
              depth=DEPTH_DEFAULT, norm="none", use_cuda_kernel=True):
    return ActMLP(dataset, act_kind, dropout=dropout, width=width, depth=depth,
                  norm=norm, use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    """Same four-field breakdown the other arms return, so train_static.py's
    equal-capacity assertion works unchanged. "ffn_trainable" is the three
    hidden layers -- this architecture's analogue of the ViT's FFN and the
    mixer's channel MLPs."""
    act_ids = {id(p) for p in model.shared_act.parameters()}
    ffn = sum(p.numel() for m in list(model.hidden) + list(model.norms)
              for p in m.parameters()
              if p.requires_grad and id(p) not in act_ids)
    return {
        "total": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "ffn_trainable": ffn,
        "act_trainable": sum(p.numel() for p in model.shared_act.parameters()
                             if p.requires_grad),
    }
