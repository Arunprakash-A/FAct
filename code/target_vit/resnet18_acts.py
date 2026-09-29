"""ResNet-18 with the activation injected as a shared, parameter-free module.

Third architecture of this study, after the ~100K ViT (vit_acts.py) and the
~100K ConvMLPMixer+GAP (convmlpmixer_acts.py). Those two were built to a
parameter budget; this one is NOT -- "ResNet-18" names a specific network, and
shrinking it to ~100K would leave something that is no longer ResNet-18. It
therefore runs at its own natural size (11.2M params on CIFAR-10), which makes
this arm a test of whether the activation ranking survives a ~110x jump in
capacity and the addition of residual connections + BatchNorm, not another
point in the equal-capacity series.

The equal-capacity guarantee still holds WITHIN the arm, which is what the
comparison rests on: all 11 activations are parameter-free, so every variant's
ResNet-18 has a byte-identical parameter count and optimizer state (asserted
per run by train_static.py). Only the activation function differs.

Backbone is the standard CIFAR-style ResNet-18 (He et al. 2016 for the block
structure; the 3x3/stride-1 stem with no max-pool is the variant the CIFAR
literature reports against, where the 7x7/stride-2 ImageNet stem would throw
away 32x32 spatial resolution before the first block):

    conv3x3(s1) -> BN -> act
    layer1: 2 x BasicBlock(64,  s1)
    layer2: 2 x BasicBlock(128, s2)
    layer3: 2 x BasicBlock(256, s2)
    layer4: 2 x BasicBlock(512, s2)
    -> GAP -> fc

The stem is used unchanged for the 64px datasets (food101, eurosat) too, so
the architecture is one architecture across all five datasets; only the final
feature map differs (8x8 at 64px vs 4x4 at 32px), and GAP absorbs that -- the
parameter count depends on n_classes alone, exactly as in the ConvMLPMixer arm.

Activation placement is the one design decision, and it follows the original
ResNet exactly: an activation after every BatchNorm, plus one after each
residual addition. Every one of those sites gets THE SAME module by reference
-- the same convention vit_acts.py and convmlpmixer_acts.py use. For
parameter-free activations shared and independent instances are the same
function, so this costs nothing and keeps one convention across all three
architectures.

Init is torchvision's ResNet default (He/kaiming fan_out for convs, ones/zeros
for BN) rather than this study's other two arms' trunc_normal_(std=0.02),
because that init is part of what "ResNet-18" means and 0.02-scale weights
would cripple a 20-layer network. Applied identically to all 11 variants, so
it cannot favour one activation.
"""
import torch.nn as nn

from fixed_acts import build_act

N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100,
             "food101": 101, "eurosat": 10}
IN_CHANS = {"fmnist": 1, "cifar10": 3, "cifar100": 3,
            "food101": 3, "eurosat": 3}
#: Matches cv_data.NEW_DS_IMG_SIZE for the two add-on datasets. Informational
#: only -- unlike the ViT (which patch-embeds a fixed grid), this network is
#: fully convolutional + GAP, so img_size moves neither the parameter count nor
#: any shape assertion. Wall-clock does scale with it.
IMG_SIZE = {"fmnist": 28, "cifar10": 32, "cifar100": 32,
            "food101": 64, "eurosat": 64}

#: Standard ResNet-18: four stages, two BasicBlocks each, widths doubling.
LAYERS_DEFAULT = (2, 2, 2, 2)
WIDTH_DEFAULT = 64


def conv3x3(in_c, out_c, stride=1):
    return nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    """The ResNet-18/34 block. `act` is a module shared with the whole network,
    used at both of the block's activation sites (post-BN1 and post-addition),
    which is where the original places its two ReLUs."""
    expansion = 1

    def __init__(self, in_c, out_c, stride, act):
        super().__init__()
        self.conv1 = conv3x3(in_c, out_c, stride)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = act
        self.conv2 = conv3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm2d(out_c)
        # Projection shortcut only where shape changes (option B in the paper).
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
        return self.act(out + identity)


class ActResNet18(nn.Module):
    def __init__(self, dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                 layers=LAYERS_DEFAULT, use_cuda_kernel=True):
        super().__init__()
        in_chans = IN_CHANS[dataset]
        n_classes = N_CLASSES[dataset]
        self.act_kind = act_kind

        # ONE activation module, handed to every site by reference.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.conv1 = conv3x3(in_chans, width)
        self.bn1 = nn.BatchNorm2d(width)

        self.in_c = width
        self.layer1 = self._make_layer(width, layers[0], stride=1)
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        self.gap = nn.AdaptiveAvgPool2d(1)
        # Kept at 0.0 by default: the reference ResNet-18 has no dropout, and
        # its regularisation comes from BN + weight decay + augmentation.
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


def build_resnet18(dataset, act_kind, dropout=0.0, width=WIDTH_DEFAULT,
                   layers=LAYERS_DEFAULT, use_cuda_kernel=True):
    return ActResNet18(dataset, act_kind, dropout=dropout, width=width,
                       layers=layers, use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    """Same four-field breakdown vit_acts.count_params returns, so
    train_static.py's equal-capacity assertion works unchanged across all three
    architectures. "ffn_trainable" is the four residual stages -- this
    architecture's analogue of the ViT's FFN / the mixer's channel MLPs."""
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
