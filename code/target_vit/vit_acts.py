"""The ~100K-parameter ViT every run in this study trains -- identical for all
11 activations, with the activation module as the single injected difference.

Architecture is the same pre-LN ViT as the main study's cv_vit.py (patchify ->
[CLS] + learnable pos-embed -> depth x [MHSA, MLP] -> LN -> linear head), with
the same weight init (trunc_normal_ std=0.02, zero biases, LN at 1/0), scaled
down to the study's ~100K budget:

    embed_dim=64  depth=2  num_heads=4  mlp_ratio=4.0 (d_ff=256)  patch=4

      fmnist    105,098 params    (28x28 ->  49 patches, 1 chan, 10 classes)
      cifar10   108,106 params    (32x32 ->  64 patches, 3 chan, 10 classes)
      cifar100  113,956 params    (32x32 ->  64 patches, 3 chan, 100 classes)

mlp_ratio stays at the standard ViT 4.0 rather than being shrunk, so the FFN --
where the activation lives -- keeps its usual share of the network.

Why a fork rather than cv_vit.build_vit: cv_vit selects its FFN through
fourier_ffn.build_ffn's `ffn_kind` string, which enumerates FAct/conv/neuron-
tied/MoE variants and hardcodes GELU-or-ReLU for the standard FFN. This study
needs the orthogonal axis -- one fixed FFN shape, eleven activations -- so the
activation arrives as a module, not a variant name. Everything else (block
structure, init, forward) is a faithful port.

ONE activation module is built per network and handed by reference to every
block, matching how fact_k2_global trained the curve being tested (tied across
neurons and across depth). For the ten stateless builtins that is
indistinguishable from a fresh instance per site; for FixedFAct it also avoids
duplicating the coefficient buffers. No variant has learnable activation
parameters, so this choice carries no capacity difference either way.
"""
import torch
import torch.nn as nn

from fixed_acts import build_act

EMBED_DIM_DEFAULT = 64
DEPTH_DEFAULT = 2
NUM_HEADS_DEFAULT = 4
MLP_RATIO_DEFAULT = 4.0

#: Same per-dataset patchification convention as cv_vit.DATASET_CFG.
DATASET_CFG = {
    "fmnist": dict(img_size=28, patch_size=4, in_chans=1, n_classes=10),
    "cifar10": dict(img_size=32, patch_size=4, in_chans=3, n_classes=10),
    "cifar100": dict(img_size=32, patch_size=4, in_chans=3, n_classes=100),
    # 64x64 datasets use patch=8, giving an 8x8 grid = 64 patches -- the SAME
    # sequence length as the 32x32/patch-4 datasets, so attention cost and
    # pos_embed size carry over unchanged and only the patch-projection conv
    # grows (3*8*8*64 vs 3*4*4*64, about +9K params). patch=4 here would
    # instead mean 256 patches: 4x the attention and a 4x larger pos_embed.
    "food101": dict(img_size=64, patch_size=8, in_chans=3, n_classes=101),
    "eurosat": dict(img_size=64, patch_size=8, in_chans=3, n_classes=10),
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MHSA(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        return self.proj_drop(out)


class MLP(nn.Module):
    """Linear -> act -> dropout -> Linear. The activation is passed in, never
    constructed here -- that is the whole point of this file."""

    def __init__(self, d_model, d_ff, act, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = act
        self.drop = nn.Dropout(dropout)

    def forward(self, x, return_acts=False):
        h = self.act(self.fc1(x))
        out = self.fc2(self.drop(h))
        return (out, h) if return_acts else out


class Block(nn.Module):
    def __init__(self, dim, num_heads, d_ff, act, dropout=0.1, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MLP(dim, d_ff, act, dropout=dropout)

    def forward(self, x, return_acts=False):
        """return_acts=True also hands back this block's POST-activation FFN
        tensor -- the single implementation of the block, so the diagnostic
        path can never drift from the training path."""
        x = x + self.attn(self.norm1(x))
        if return_acts:
            y, h = self.ffn(self.norm2(x), return_acts=True)
            return x + y, h
        return x + self.ffn(self.norm2(x))


class ActViT(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 act_kind="gelu", embed_dim=EMBED_DIM_DEFAULT,
                 depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
                 mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
                 use_cuda_kernel=True):
        super().__init__()
        self.act_kind = act_kind
        self.embed_dim = embed_dim
        d_ff = int(embed_dim * mlp_ratio)
        self.d_ff = d_ff

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # Registered as `shared_act` so downstream tooling (checkpoint readers,
        # the continual driver's activation snapshot) finds it in the same
        # place the main study's fact_k2_global models keep theirs.
        self.shared_act = build_act(act_kind, use_cuda_kernel=use_cuda_kernel)

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, d_ff, self.shared_act,
                  dropout=dropout, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, return_acts=False):
        """return_acts=True additionally returns the POST-activation tensor of
        every FFN site, in block order -- what a dead/saturated-unit
        diagnostic measures (that driver is not shipped here). Kept as
        an explicit second return rather than a forward hook so it costs
        nothing on the normal path."""
        acts = [] if return_acts else None
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            if return_acts:
                x, h = blk(x, return_acts=True)
                acts.append(h)
            else:
                x = blk(x)
        x = self.norm(x)
        out = self.head(x[:, 0])
        return (out, acts) if return_acts else out


def build_vit(dataset, act_kind, embed_dim=EMBED_DIM_DEFAULT,
              depth=DEPTH_DEFAULT, num_heads=NUM_HEADS_DEFAULT,
              mlp_ratio=MLP_RATIO_DEFAULT, dropout=0.1, attn_drop=0.0,
              use_cuda_kernel=True):
    cfg = DATASET_CFG[dataset]
    return ActViT(act_kind=act_kind, embed_dim=embed_dim, depth=depth,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                  attn_drop=attn_drop, use_cuda_kernel=use_cuda_kernel, **cfg)


def build_vit_shape(img_size, patch_size, in_chans, n_classes, act_kind,
                    embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
                    num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                    dropout=0.1, attn_drop=0.0, use_cuda_kernel=True):
    """build_vit for shapes that don't come from DATASET_CFG -- the continual
    benchmarks, whose input size and class count are set by the handler."""
    return ActViT(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                  n_classes=n_classes, act_kind=act_kind, embed_dim=embed_dim,
                  depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  dropout=dropout, attn_drop=attn_drop,
                  use_cuda_kernel=use_cuda_kernel)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    act = sum(p.numel() for p in model.shared_act.parameters() if p.requires_grad)
    ffn = 0
    seen = set()
    for blk in model.blocks:
        for p in blk.ffn.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                ffn += p.numel()
    return {"total": total, "trainable": train, "ffn_trainable": ffn,
            "act_trainable": act}
