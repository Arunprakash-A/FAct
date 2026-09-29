"""A small, from-scratch ViT (patchify -> [CLS] + learnable pos-embed ->
pre-LN Transformer encoder -> linear head) -- the CV analogue of
transformer_nmt.TransformerNMT. Architecture (embed_dim=192, depth=6,
num_heads=6, mlp_ratio=4.0, patch/img size per dataset) is copied verbatim
from the earlier convolutional-FFN baseline study's ViT, so results stay
comparable to its already-computed "standard" and "conv_gelu" baselines. The ONLY
change here: the MLP block is built by fourier_ffn.build_ffn (the same function
the authors' earlier NMT study used, not shipped here), not a bespoke per-study
MLP class, so the
same FFN_VARIANTS -- standard, standard_narrow, fact_k2, fact_k2_narrow,
fact_k2_shared, fact_k2_global, conv_gelu, ... -- are available here too.

fact_k2_global: exactly as in transformer_nmt.TransformerNMT, ONE
FourierActivation module is built once in VisionTransformer.__init__ and
handed by reference to every one of the `depth` blocks' FFN.
"""
import torch
import torch.nn as nn

from fourier_ffn import build_ffn
from fourier_layers import (PAU, AconC, FourierActivation, parse_acon_global,
                            parse_embed_k, parse_global_2act_k, parse_global_k,
                            parse_global_phase, parse_global_routed,
                            parse_pau_global)

DEPTH_DEFAULT = 6
NUM_HEADS_DEFAULT = 6
EMBED_DIM_DEFAULT = 192
MLP_RATIO_DEFAULT = 4.0

DATASET_CFG = {
    "fmnist": dict(img_size=28, patch_size=4, in_chans=1, n_classes=10),
    "cifar10": dict(img_size=32, patch_size=4, in_chans=3, n_classes=10),
    "cifar100": dict(img_size=32, patch_size=4, in_chans=3, n_classes=100),
    "tinyimagenet": dict(img_size=64, patch_size=8, in_chans=3, n_classes=200),
    # img_size/patch_size match the standard ViT-Ti/16 ImageNet convention
    # (224x224 -> 14x14=196 patches); embed_dim/depth/num_heads/mlp_ratio
    # stay at this study's usual defaults (see build_vit) rather than scaling
    # up for ImageNet -- the point of this comparison is the activation
    # function, not architecture size, so it's held fixed across all five
    # datasets.
    "imagenet1k": dict(img_size=224, patch_size=16, in_chans=3, n_classes=1000),
    # Food-101 is cached at 64x64 (cv_data.FOOD101_IMG_SIZE), same resolution
    # as tinyimagenet -- patch=8 gives an 8x8=64-patch grid, matching
    # tinyimagenet's sequence length exactly (same attention cost/pos_embed
    # size), the same convention an earlier (much smaller) Food-101 ViT used.
    "food101": dict(img_size=64, patch_size=8, in_chans=3, n_classes=101),
    # Food-101 at the ImageNet geometry (224/16, 197 tokens) -- the only
    # setting where an ImageNet checkpoint's patch_embed/pos_embed/
    # cls_token transfer, so a fine-tune reuses them instead of
    # relearning the input stem from 75,750 images.
    "food101_224": dict(img_size=224, patch_size=16, in_chans=3, n_classes=101),
}


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

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


class Block(nn.Module):
    """Pre-LN Transformer block: x + Attn(LN(x)); x + FFN(LN(x)). `ffn_kind`
    dispatches to fourier_ffn.build_ffn -- the only architectural difference
    from the baseline study's Block, which hardcoded its own MLP/ConvGELUMLP."""

    def __init__(self, dim, num_heads, ffn_kind, d_ff, dropout=0.1,
                 attn_drop=0.0, shared_act=None, shared_ffn=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MHSA(dim, num_heads, attn_drop=attn_drop, proj_drop=dropout)
        self.norm2 = nn.LayerNorm(dim)
        # shared_ffn (fact_kK_global_wshare only): the SAME FFN module
        # (fc1, fc2, and the shared activation) reused by reference across
        # every block, instead of building an independent one here -- ties
        # the FFN's linear weights across depth, not just its activation.
        self.ffn = shared_ffn if shared_ffn is not None else build_ffn(
            ffn_kind, dim, d_ff, dropout, act=shared_act)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, n_classes,
                 embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
                 num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
                 ffn_kind="standard", dropout=0.1, attn_drop=0.0, act_ref="gelu",
                 act_init="true", act_init_scale=1.0, anneal_harmonics=False,
                 anneal_rate=0.5, ntied_zero_init_weights=False,
                 ffn_zero_init_weights=False, ntied_rank=1, ntied_const_init=None,
                 ntied_const_init_all=False, use_cuda_act=False):
        super().__init__()
        self.ffn_kind = ffn_kind
        self.embed_dim = embed_dim
        d_ff_standard = int(embed_dim * mlp_ratio)
        self.d_ff_standard = d_ff_standard

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(dropout)

        # fact_kK_global (any K): ONE FourierActivation, tied across neurons
        # too, shared BY REFERENCE across every block's FFN -- identical
        # convention to transformer_nmt.TransformerNMT.
        shared_act = None
        shared_act_1 = None
        shared_act_2 = None
        shared_ffn = None
        k_global = parse_global_k(ffn_kind)
        k_2act = parse_global_2act_k(ffn_kind)
        if k_global is not None:
            # act_init="random" (small randn coeffs instead of the true GELU
            # Fourier coefficients) isolates whether starting from a
            # deliberate GELU approximation -- known to be a poor global fit
            # away from t=0, see fourier_layers.true_fourier_coeffs -- is
            # itself what hurts the neuron-tied variants, vs. just letting
            # the shared activation find its own shape from near-zero init.
            shared_act = FourierActivation(d_ff_standard, K=k_global, ref=act_ref,
                                           init=act_init, shared=True,
                                           init_scale=act_init_scale,
                                           anneal_harmonics=anneal_harmonics,
                                           anneal_rate=anneal_rate,
                                           route=parse_global_routed(ffn_kind),
                                           phase=parse_global_phase(ffn_kind))
        elif parse_acon_global(ffn_kind):
            # acon_c_global: the SAME construction as above, but the shared
            # activation is ACON-C rather than a Fourier series -- one
            # {p1, p2, beta} triple (shared=True ties them across neurons;
            # handing this one module to every Block ties them across depth)
            # for the entire network. Registered as self.shared_act like the
            # Fourier one so it lands in checkpoints under `shared_act.*` and
            # every downstream per-epoch activation-evolution script finds it
            # in the same place. act_ref/act_init/anneal_* do not apply --
            # ACON-C has no reference activation to fit and no harmonics to
            # anneal; its init is fixed at the paper's beta=p1=1, p2=0.
            shared_act = AconC(d_ff_standard, shared=True)
        elif parse_pau_global(ffn_kind):
            # pau_global: the SAME construction as fact_kK_global/
            # acon_c_global, but the shared activation is a Pade Activation
            # Unit (Molina et al., ICLR 2020 -- P(t)/Q(t), order (5,4),
            # |b_k| denominator -- see PAU, which is NOT the paper's
            # pole-free "safe" form) rather than a Fourier series or
            # ACON-C -- ONE {a, b} coefficient pair (shared=True ties them
            # across neurons too) for the entire network. act_ref/act_init
            # DO apply here (unlike ACON-C): init="true" (the default) fits
            # the Pade rational approximation to `act_ref` (gelu) by
            # least-squares on [-pi, pi], the same "start as a faithful
            # GELU copy" convention as FourierActivation, so this leg is
            # comparable to fact_k2_global at t=0.
            if use_cuda_act:
                # CUDA-kernel PAU is fixed at order (m, n) = (5, 4) (see
                # cuda_pau/pau_module.py) and only supports init in
                # {"true", "random"} -- same supported-subset guard as
                # fact_kK_global's use_cuda_act branch, so a caller sweeping
                # act_init doesn't silently get a different (unsupported)
                # activation than requested.
                assert act_init in ("true", "random"), (
                    "use_cuda_act=True with pau_global requires act_init in "
                    f"('true', 'random'), got {act_init!r}")
                from cuda_pau.pau_module import PAUCUDA
                shared_act = PAUCUDA(d_ff_standard, ref=act_ref, init=act_init, shared=True)
            else:
                shared_act = PAU(d_ff_standard, shared=True, ref=act_ref, init=act_init)
        elif k_2act is not None:
            # fact_kK_global_2act: two INDEPENDENT globally-shared
            # activations, one handed to blocks[0:depth/2] (the "first
            # half"), the other to blocks[depth/2:depth] (the "second
            # half") -- see the self.blocks construction below. depth must
            # split evenly so "first half"/"second half" is unambiguous.
            assert depth % 2 == 0, (
                f"{ffn_kind} needs an even depth to split into two equal "
                f"halves, got depth={depth}")
            if use_cuda_act:
                # CUDA-kernel FourierActivation is fixed at K=2 (see
                # cuda_fact_k2/fact_k2_module.py) -- only swap it in when
                # the variant's own K matches, so a caller sweeping K
                # doesn't silently get a different K than requested.
                assert k_2act == 2, (
                    f"use_cuda_act=True requires K=2, got fact_k{k_2act}_"
                    "global_2act (the CUDA kernel only supports K=2)")
                from cuda_fact_k2.fact_k2_module import FourierActivationK2CUDA
                shared_act_1 = FourierActivationK2CUDA(
                    d_ff_standard, ref=act_ref, init=act_init, shared=True,
                    init_scale=act_init_scale)
                shared_act_2 = FourierActivationK2CUDA(
                    d_ff_standard, ref=act_ref, init=act_init, shared=True,
                    init_scale=act_init_scale)
            else:
                shared_act_1 = FourierActivation(
                    d_ff_standard, K=k_2act, ref=act_ref, init=act_init,
                    shared=True, init_scale=act_init_scale,
                    anneal_harmonics=anneal_harmonics, anneal_rate=anneal_rate)
                shared_act_2 = FourierActivation(
                    d_ff_standard, K=k_2act, ref=act_ref, init=act_init,
                    shared=True, init_scale=act_init_scale,
                    anneal_harmonics=anneal_harmonics, anneal_rate=anneal_rate)
        # "<base_kind>_wshare" (any base_kind, e.g. fact_k2_global_wshare or
        # standard_wshare): ALSO build the FFN itself (fc1, fc2, and its
        # activation -- shared_act by reference if the base kind has one)
        # once and share that same module across every block, instead of
        # each block getting an independent fc1/fc2 -- see Block. Generic
        # suffix check (not just parse_global_wshare's fact_kK_global regex)
        # so a GELU baseline (standard_wshare) can isolate whether tying the
        # linear weights alone -- without a shared activation -- explains
        # part of fact_k2_global_wshare's story.
        if ffn_kind.endswith("_wshare"):
            shared_ffn = build_ffn(ffn_kind, embed_dim, d_ff_standard,
                                   dropout, act=shared_act,
                                   ntied_zero_init_weights=ntied_zero_init_weights,
                                   ntied_rank=ntied_rank,
                                   ntied_const_init=ntied_const_init,
                                   ntied_const_init_all=ntied_const_init_all)
        self.shared_act = shared_act
        # fact_kK_global_2act only -- see the elif k_2act branch above and
        # the self.blocks construction below. Each is its own top-level
        # attribute (not an alias of self.shared_act, which stays None for
        # this variant), so this is safe from the state_dict-double-entry
        # aliasing quirk noted below for embed_act_shared.
        self.shared_act_1 = shared_act_1
        self.shared_act_2 = shared_act_2
        self.shared_ffn = shared_ffn

        # fact_kK_embed: apply THE SAME shared activation to the input
        # embeddings as well, residually, just before the block stack (see
        # forward) -- one module for the entire network, not a second one.
        # shared=True makes its coefficients (1,) and (1,K), so the single
        # instance broadcasts over the embed_dim-wide embeddings and the
        # d_ff-wide FFN hidden layer alike. Activation cost is therefore
        # IDENTICAL to fact_kK_global's (1+2K params), and the two variants
        # differ only in where that one nonlinearity is applied -- no
        # parameter confound in the comparison.
        #
        # Until now the path from patch_embed to blocks[0] was entirely linear
        # -- patchify, cat [CLS], add pos_embed, dropout -- so this adds a
        # nonlinearity where the architecture had none, rather than replacing
        # an existing one.
        #
        # Recorded as a bool, NOT as a second attribute pointing at
        # shared_act: registering one module under two names writes it to
        # state_dict twice (the aliasing quirk that already bit the ACON
        # coefficient-evolution extraction), which would silently corrupt any
        # statistic computed over the saved coefficients.
        self.embed_act_shared = parse_embed_k(ffn_kind) is not None
        assert not (self.embed_act_shared and shared_act is None), (
            f"{ffn_kind} needs the shared FourierActivation built above; "
            "parse_global_k must match this kind too")

        if k_2act is not None:
            # first half (blocks 0..depth/2-1, i.e. "up to depth-6" at
            # depth=12) gets shared_act_1; second half (depth/2..depth-1,
            # "depth-7 to depth-12") gets shared_act_2.
            half = depth // 2
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, ffn_kind, d_ff_standard, dropout=dropout,
                      attn_drop=attn_drop,
                      shared_act=(shared_act_1 if i < half else shared_act_2),
                      shared_ffn=shared_ffn)
                for i in range(depth)
            ])
        else:
            self.blocks = nn.ModuleList([
                Block(embed_dim, num_heads, ffn_kind, d_ff_standard, dropout=dropout,
                      attn_drop=attn_drop, shared_act=shared_act, shared_ffn=shared_ffn)
                for _ in range(depth)
            ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

        self.apply(self._init_weights)

        # Zero-init the FFN's own Linear weights (fc1.weight, fc2.weight in
        # StandardFFN/FactFFN), applied AFTER _init_weights so it overrides
        # that pass's trunc_normal_(std=0.02) for these two matrices only --
        # every other Linear (qkv/attn-proj/head), the shared FourierActivation
        # coefficients, LayerNorm, and biases (already zeroed by _init_weights)
        # are left untouched. At init this makes every FFN block's output
        # identically zero for every input (fc1(x)=0 since weight AND bias are
        # both zero, so fc2(act(0))=0 too), an ablation of whether the FFN
        # pathway can "grow" from nothing purely via gradients once fc2 starts
        # picking up a nonzero constant-input gradient (see run instructions).
        self.ffn_zero_init_weights = ffn_zero_init_weights
        if ffn_zero_init_weights:
            seen = set()
            for blk in self.blocks:
                ffn = blk.ffn
                if id(ffn) in seen:
                    continue
                seen.add(id(ffn))
                if hasattr(ffn, "fc1"):
                    nn.init.zeros_(ffn.fc1.weight)
                if hasattr(ffn, "fc2"):
                    nn.init.zeros_(ffn.fc2.weight)

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

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        # fact_kK_embed only: the SAME shared_act every block's FFN uses,
        # applied here to the embeddings as well. Residual, matching the
        # earlier NMT study's conv_gelu_inact convention
        # (`x = x + enc_input_act(x)`) and the source ViT study it was ported
        # from: at init the FourierActivation is a GELU fit, so a bare
        # x = shared_act(x) would half-rectify the embeddings before the first
        # block ever sees them -- a much larger perturbation than the
        # activation itself is meant to be.
        if self.embed_act_shared:
            x = x + self.shared_act(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_vit(dataset, ffn_kind, embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
              num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
              dropout=0.1, attn_drop=0.0, act_ref="gelu", act_init="true",
              act_init_scale=1.0, anneal_harmonics=False, anneal_rate=0.5,
              ntied_zero_init_weights=False, ffn_zero_init_weights=False,
              ntied_rank=1, ntied_const_init=None, ntied_const_init_all=False,
              use_cuda_act=False):
    cfg = DATASET_CFG[dataset]
    return VisionTransformer(embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                              mlp_ratio=mlp_ratio, ffn_kind=ffn_kind,
                              dropout=dropout, attn_drop=attn_drop, act_ref=act_ref,
                              act_init=act_init, act_init_scale=act_init_scale,
                              anneal_harmonics=anneal_harmonics, anneal_rate=anneal_rate,
                              ntied_zero_init_weights=ntied_zero_init_weights,
                              ffn_zero_init_weights=ffn_zero_init_weights,
                              ntied_rank=ntied_rank, ntied_const_init=ntied_const_init,
                              ntied_const_init_all=ntied_const_init_all,
                              use_cuda_act=use_cuda_act,
                              **cfg)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # id()-based dedup (not name-based): fact_kK_global_wshare's blocks all
    # point at the SAME ffn module, so named_parameters()'s own dedup would
    # otherwise attribute every one of its params to whichever attribute
    # (shared_ffn vs. blocks.i.ffn) happens to be registered first, hiding
    # them from a ".ffn." name-substring filter.
    seen, ffn_train = set(), 0
    for blk in model.blocks:
        for p in blk.ffn.parameters():
            if p.requires_grad and id(p) not in seen:
                seen.add(id(p))
                ffn_train += p.numel()
    # No fact_kK_embed special case here on purpose: it reuses the FFNs' own
    # shared activation at the embedding rather than adding a module, so its
    # parameter count is identical to fact_kK_global's and the blocks-walk
    # above already accounts for every activation parameter in the network.
    return {"total": total, "trainable": train, "ffn_trainable": ffn_train}
