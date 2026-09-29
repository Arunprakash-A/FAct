"""Fourier feed-forward network -- a drop-in replacement for the position-wise
FFN (the per-token MLP) inside a standard Transformer block.

    standard FFN : y = W2 @ GELU(W1 @ x + b1) + b2          (W1, W2 learned)
    Fourier FFN  : y = Wout @ phi( DFT(x) )                 (DFT frozen, phi learned)

Mapping the FourierNet recipe onto the FFN:
  * proj_in (DFT)   -- a FROZEN truncated-DFT projection (`DFTLinear`).  Because
                       DFTLinear is a *truncated* DFT it needs d_ff <= d_model, so
                       we use d_ff = d_model (a full, frozen d_model-point DFT).
                       use_imag=False keeps the signed real (cosine) projection;
                       use_imag=True takes the |DFT| magnitude (folds sign).
  * act (phi)       -- a per-channel LEARNABLE Fourier-series activation
                       (`FourierActivation`), initialised from GELU.
  * out (Wout)      -- learnable projection back to d_model: the "channel mix"
                       the conv study found to be the decisive ingredient.  With
                       freeze_out=True it is instead a FROZEN inverse-DFT
                       synthesis matrix (the fully-frozen-linear ablation).

Only the FFN changes; attention, embeddings, LayerNorm, residuals are untouched.
"""
import math
import numpy as np
import torch
import torch.nn as nn

from fourier_layers import (AconC, DCTActivation, DCTLinear, DFTLinear,
                            FourierActivation, HadamardLinear,
                            HadamardLinearTrunc, parse_acon_global,
                            parse_pau_global,
                            parse_global_2act_k, parse_global_conv,
                            parse_global_k, parse_global_ntied)


class StandardFFN(nn.Module):
    """The ordinary Transformer FFN: Linear -> GELU -> Linear."""

    def __init__(self, d_model, d_ff, dropout=0.1, act="gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU() if act == "gelu" else nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class FactFFN(nn.Module):
    """Standard FFN (fully learnable W1, W2, d_ff = 4*d_model) with GELU
    replaced by a learnable per-neuron `FourierActivation` ("FACT") --
    unlike `FourierFFN` below, proj_in stays an ordinary learned `nn.Linear`,
    not a frozen DFT matrix; only the pointwise nonlinearity changes. This
    isolates the effect of the activation function alone, holding the rest
    of the standard FFN (width, both linear maps) fixed. K=2 matches this
    study's `mnist-mlp-vs-fact-fact-k2` naming convention.

        FACT FFN : y = W2 @ FourierAct_K2( W1 @ x + b1 ) + b2   (W1, W2 learned)

    shared=True ties ONE set of Fourier coefficients across all d_ff neurons
    in the layer (`fact_k2_shared`), instead of each neuron learning its own
    independent {a0, a_k, b_k} (`fact_k2`) -- isolates whether per-neuron
    specialisation of the activation shape matters, at (1+2K) activation
    params per layer instead of d_ff*(1+2K).

    Pass an existing `act` module to make TWO OR MORE FactFFN instances
    literally share the same activation (same nn.Module, same parameter
    tensors) rather than each building its own -- this is how
    `fact_k2_global` (see build_ffn / TransformerNMT) ties one Fourier-K2
    activation across every layer of the whole network, not just across
    neurons within one layer. When `act` is given, `K`/`shared` are ignored
    (the passed-in module's own K/shared already determine its shape).

    pre_act_conv=True (the "fact_kK_global_conv" variants) additionally gives
    THIS layer its own learnable 1-D convolution -- a single shared
    Conv1d(1,1,kernel_size) kernel sliding along the d_ff axis, the same
    banded/Toeplitz recipe as `ConvGELUFFN`'s conv_gelu -- applied to the
    pre-activations (fc1(x)) immediately before `self.act`. Unlike `act`,
    this conv is built fresh per layer (NOT shared by reference), so with a
    globally-shared activation the network still gets `depth` independent
    small channel-mixing kernels, one per layer, feeding into the one global
    nonlinearity.
    """

    def __init__(self, d_model, d_ff, K=2, dropout=0.1, shared=False, act=None,
                 pre_act_conv=False):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = act if act is not None else FourierActivation(
            d_ff, K=K, ref="gelu", init="true", shared=shared)
        self.drop = nn.Dropout(dropout)
        self.pre_act_conv = None
        if pre_act_conv:
            k = conv_kernel_size(d_ff)
            self.pre_act_conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=True)

    def forward(self, x):
        h = self.fc1(x)
        if self.pre_act_conv is not None:
            shape = h.shape
            z = h.reshape(-1, 1, shape[-1])
            # forced fp32 -- this degenerate in=1/out=1-channel Conv1d has no
            # efficient bf16 kernel on this study's GPUs (see ConvGELUFFN).
            with torch.autocast(device_type=z.device.type, enabled=False):
                z = self.pre_act_conv(z.float())
            h = z.reshape(*shape).to(h.dtype)
        return self.fc2(self.drop(self.act(h)))


class NeuronTiedFFN(nn.Module):
    """Extreme within-layer neuron weight-sharing: instead of fc1/fc2 being
    full (d_ff, d_model) / (d_model, d_ff) matrices with one independent
    weight vector per hidden neuron, ALL d_ff hidden neurons share the SAME
    incoming weight vector (fc1 collapses to rank 1) and the SAME outgoing
    weight vector (fc2 collapses to rank 1 too):

        h_i = <x, w_in> + b1_i     for i = 1..d_ff   (same scalar projection
                                    for every neuron -- only the per-neuron
                                    bias b1_i and, downstream, the
                                    nonlinearity give neurons distinct
                                    outputs before they're summed back down)
        y   = w_out * sum_i act(h_i) + b2

    Params: d_model (w_in) + d_ff (b1) + d_model (w_out) + d_model (b2) =
    3*d_model + d_ff, independent of depth -- versus a standard FFN's
    2*d_model*d_ff + d_ff + d_model. At this study's d_model=192/d_ff=768
    that is 1,344 vs. 295,872 params, a ~220x reduction, BEFORE combining
    with "_wshare" (see cv_vit.py) to also tie this one FFN instance across
    every block's depth.

    `act` may be a plain nn.GELU() ("standard_ntied_wshare") or a shared
    FourierActivation instance passed in by the caller ("fact_kK_global_ntied"
    / "fact_kK_global_ntied_wshare" -- see build_ffn's parse_global_ntied
    branch), exactly like every other FFN variant in this file.

    zero_init_weights=True zero-initialises the two WEIGHT vectors (w_in,
    w_out) instead of nn.Linear's default uniform init -- biases (b1, b2)
    are unaffected, still uniform. At init this makes the linear pathway
    input-independent (proj = x @ w_in = 0, so h_i = b1_i for every neuron;
    y = 0*w_out + b2 = b2, a constant), but gradients w.r.t. both w_in and
    w_out are non-zero from the first step (dL/dw_in involves x, dL/dw_out
    involves the activation sum), so both move away from zero immediately --
    this is a "start from nothing, let the linear pathway grow" ablation,
    analogous to zero-init residual-branch tricks elsewhere in deep learning.

    rank>1 (default 1, exactly the mechanism above): generalises the single
    shared vector on each side to R shared BASIS vectors (U_in, V_out, each
    d_model x R) plus a small per-neuron R-dim COMBINATION vector on each
    side (C_in, D_out, each d_ff x R), so every neuron's effective incoming/
    outgoing weight is its own linear combination of the R shared basis
    vectors instead of being IDENTICAL across all d_ff neurons:

        proj = x @ U_in                        # (..., R), shared by all neurons
        h_i  = <proj, c_i> + b1_i              # c_i = C_in[i]: (R,) per-neuron
        s    = sum_i act(h_i) * d_i            # d_i = D_out[i]: (R,) per-neuron
        y    = s @ V_out.T + b2

    The effective (d_ff, d_model) weight matrix on each side is C @ U.T (resp.
    D @ V.T), rank <= R by construction -- R=1 with c_i/d_i fixed at 1 for
    every neuron (not learnable) is exactly the mechanism above; this class
    keeps that R=1 case as a separate, byte-identical code path (so every
    existing "rank-1" result is unaffected) and only takes the low-rank path
    for R>1. zero_init_weights at R>1 zeros just the shared bases (U_in,
    V_out), leaving the per-neuron combination vectors (C_in, D_out) at their
    normal random init -- proj is then always 0 regardless of C_in, so every
    neuron's h_i collapses to b1_i exactly as in the R=1 zero-init case.
    Unlike R=1's single one-step lag (only w_in is gradient-blocked at step
    1), the extra indirection here produces a 3-stage cascade verified
    empirically: at step 1 only V_out has a non-zero gradient (it multiplies
    the already non-zero s=act(b1)@D_out directly); at step 2, once V_out has
    moved, U_in and D_out both unlock (D_out via dy/ds now non-zero, U_in via
    dL/dh now non-zero); C_in unlocks last, at step 3, once U_in has moved
    and proj=x@U_in becomes non-zero (C_in's only path to the loss is
    dh/dC_in = proj).

    const_init_weights=<float> (default None, mutually exclusive with
    zero_init_weights) is zero_init_weights' generalisation: instead of
    filling the shared weight vectors (w_in/w_out at rank=1, U_in/V_out at
    rank>1) with EXACTLY 0, fill every entry with this one small non-zero
    constant c via nn.init.constant_ (biases and, at rank>1, the per-neuron
    combination vectors C_in/D_out are unaffected, same random-uniform init
    as always). A constant-filled matrix has rank exactly 1 for any c != 0
    (every row is a copy of the same value), so this sits strictly between
    "true rank 0" (the exact zero matrix, i.e. zero_init_weights=True) and a
    normal random init: as c -> 0 it approaches rank 0. The point is to
    break the exact-zero gradient-blocking property zero_init_weights relies
    on (at rank>1, several parameters see EXACTLY zero gradient at step 1/2
    purely because some upstream factor is exactly 0 -- see the cascade
    above) while keeping the same "start near nothing" flavour, to check
    whether the zero-init rescue effect needs the gradient block itself or
    just needs to start small.

    const_init_all=True (only meaningful with const_init_weights set and
    rank>1; at rank=1 const_init_weights already covers every weight, since
    w_in/w_out are the only two) extends the constant fill from just the
    shared bases (U_in, V_out) to EVERY weight tensor including the
    per-neuron combination vectors (C_in, D_out) -- i.e. the entire FFN
    weight parameter set starts at the same single scalar c, with only the
    biases (b1, b2) left at their normal random-uniform init. Note this is a
    materially different starting point from the U_in/V_out-only constant
    fill: with C_in/D_out also equal to c, proj = x @ U_in has all R entries
    identical (= c * sum(x)), and h_i = <proj, c_i> + b1_i collapses to the
    SAME scalar (R * c^2 * sum(x)) for every neuron before the bias -- much
    closer in spirit to the rank=1 mechanism (one shared scalar projection)
    than to a genuine rank<=R start, despite having R-dimensional factors.
    """

    def __init__(self, d_model, d_ff, dropout=0.1, act=None,
                 zero_init_weights=False, rank=1, const_init_weights=None,
                 const_init_all=False):
        super().__init__()
        assert not (zero_init_weights and const_init_weights is not None), (
            "zero_init_weights and const_init_weights are mutually exclusive "
            "(const_init_weights=0.0 is equivalent to zero_init_weights=True)")
        assert not (const_init_all and const_init_weights is None), (
            "const_init_all requires const_init_weights to be set")
        self.d_model, self.d_ff, self.rank = d_model, d_ff, rank
        if rank == 1:
            self.w_in = nn.Parameter(torch.empty(d_model))
            self.b1 = nn.Parameter(torch.empty(d_ff))
            self.w_out = nn.Parameter(torch.empty(d_model))
            self.b2 = nn.Parameter(torch.empty(d_model))
        else:
            self.U_in = nn.Parameter(torch.empty(d_model, rank))
            self.C_in = nn.Parameter(torch.empty(d_ff, rank))
            self.b1 = nn.Parameter(torch.empty(d_ff))
            self.D_out = nn.Parameter(torch.empty(d_ff, rank))
            self.V_out = nn.Parameter(torch.empty(d_model, rank))
            self.b2 = nn.Parameter(torch.empty(d_model))
        self.act = act if act is not None else nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.reset_parameters(zero_init_weights=zero_init_weights,
                              const_init_weights=const_init_weights,
                              const_init_all=const_init_all)

    def reset_parameters(self, zero_init_weights=False, const_init_weights=None,
                         const_init_all=False):
        # mirrors nn.Linear's default init (uniform, bound = 1/sqrt(fan_in))
        # applied to a single weight vector (rank=1) or low-rank factor
        # (rank>1) instead of a full matrix.
        bound_in = 1.0 / math.sqrt(self.d_model)
        bound_out = 1.0 / math.sqrt(self.d_ff)
        if self.rank == 1:
            if zero_init_weights:
                nn.init.zeros_(self.w_in)
                nn.init.zeros_(self.w_out)
            elif const_init_weights is not None:
                nn.init.constant_(self.w_in, const_init_weights)
                nn.init.constant_(self.w_out, const_init_weights)
            else:
                nn.init.uniform_(self.w_in, -bound_in, bound_in)
                nn.init.uniform_(self.w_out, -bound_out, bound_out)
            nn.init.uniform_(self.b1, -bound_in, bound_in)
            nn.init.uniform_(self.b2, -bound_out, bound_out)
        else:
            bound_rank = 1.0 / math.sqrt(self.rank)
            if zero_init_weights:
                nn.init.zeros_(self.U_in)
                nn.init.zeros_(self.V_out)
            elif const_init_weights is not None:
                nn.init.constant_(self.U_in, const_init_weights)
                nn.init.constant_(self.V_out, const_init_weights)
            else:
                nn.init.uniform_(self.U_in, -bound_in, bound_in)
                nn.init.uniform_(self.V_out, -bound_rank, bound_rank)
            if const_init_all:
                nn.init.constant_(self.C_in, const_init_weights)
                nn.init.constant_(self.D_out, const_init_weights)
            else:
                nn.init.uniform_(self.C_in, -bound_rank, bound_rank)
                nn.init.uniform_(self.D_out, -bound_out, bound_out)
            nn.init.uniform_(self.b1, -bound_in, bound_in)
            nn.init.uniform_(self.b2, -bound_out, bound_out)

    def forward(self, x):
        if self.rank == 1:
            proj = x @ self.w_in                 # (...,)      one scalar per token
            h = proj.unsqueeze(-1) + self.b1      # (..., d_ff)
            a = self.drop(self.act(h))
            s = a.sum(dim=-1, keepdim=True)       # (..., 1)
            return s * self.w_out + self.b2       # (..., d_model)
        proj = x @ self.U_in                      # (..., rank)
        h = proj @ self.C_in.t() + self.b1        # (..., d_ff)
        a = self.drop(self.act(h))
        s = a @ self.D_out                        # (..., rank)
        return s @ self.V_out.t() + self.b2       # (..., d_model)


class FourierFFN(nn.Module):
    """Frozen-DFT + learnable-Fourier-activation FFN."""

    def __init__(self, d_model, d_ff=None, K=4, use_imag=False,
                 freeze_out=False, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model
        assert d_ff <= d_model, (
            f"FourierFFN uses a truncated-DFT up-projection (out<=in); need "
            f"d_ff<=d_model, got d_ff={d_ff} > d_model={d_model}.")
        self.d_model, self.d_ff, self.freeze_out = d_model, d_ff, freeze_out
        self.proj_in = DFTLinear(d_model, d_ff, use_imag=use_imag)     # FROZEN
        self.act = FourierActivation(d_ff, K=K, ref="gelu", init="true")  # learned
        self.drop = nn.Dropout(dropout)
        if freeze_out:
            # frozen inverse-DFT (real-part synthesis) matrix, d_ff -> d_model
            n = np.arange(d_model)
            m = np.arange(d_ff)
            ang = 2.0 * np.pi * np.outer(n, m) / d_model            # (d_model, d_ff)
            Wout = (np.cos(ang) / math.sqrt(d_ff)).astype(np.float32)
            self.register_buffer("Wout", torch.from_numpy(Wout))
        else:
            self.out = nn.Linear(d_ff, d_model)                     # learned mix-back

    def forward(self, x):
        h = self.proj_in(x)          # frozen DFT features
        h = self.act(h)              # learnable Fourier-series activation
        h = self.drop(h)
        if self.freeze_out:
            return h @ self.Wout.t()
        return self.out(h)


class HadamardFFN(nn.Module):
    """Frozen-Hadamard + learnable-Fourier-activation FFN: the same recipe
    as FourierFFN's `freeze_out=True` case, but the frozen projection is a
    Sylvester-Hadamard orthogonal matrix (`HadamardLinear`) instead of a
    truncated-DFT matrix (`DFTLinear`). Checked against a reference
    `HadamardLinear` for the frozen-orthogonal-transform recipe; unlike
    that module's Paley-I construction (needed there for a
    non-power-of-2 embed_dim), this uses the simpler Sylvester construction
    since d_model=512=2^9 is already a power of 2.

    Only the FFN (this class) changes -- attention's own output projection
    (W_O, internal to `nn.MultiheadAttention`) is untouched, exactly as in
    every other FFN variant in this study.

    Square-only (d_ff must equal d_model): Hadamard matrices have no
    truncated-projection analogue to DFTLinear's out_dim < in_dim.
    """

    def __init__(self, d_model, d_ff=None, K=4, freeze_out=False, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model
        assert d_ff == d_model, (
            f"HadamardFFN's frozen projection is square-only; need "
            f"d_ff==d_model, got d_ff={d_ff} != d_model={d_model}.")
        self.d_model, self.d_ff, self.freeze_out = d_model, d_ff, freeze_out
        self.proj_in = HadamardLinear(d_model)                            # FROZEN
        self.act = FourierActivation(d_ff, K=K, ref="gelu", init="true")  # learned
        self.drop = nn.Dropout(dropout)
        if freeze_out:
            # Sylvester Hadamard matrices are symmetric & orthogonal, so
            # this second frozen copy is already its own inverse/synthesis
            # transform -- the direct analogue of FourierFFN's frozen
            # inverse-DFT `Wout`.
            self.proj_out = HadamardLinear(d_model)
        else:
            self.out = nn.Linear(d_ff, d_model)                           # learned mix-back

    def forward(self, x):
        h = self.proj_in(x)          # frozen Hadamard features
        h = self.act(h)              # learnable Fourier-series activation
        h = self.drop(h)
        if self.freeze_out:
            return self.proj_out(h)
        return self.out(h)


class DCTFFN(nn.Module):
    """Frozen-DCT + learnable-DCT-series-activation FFN: swaps BOTH halves of
    the FourierNet recipe relative to HadamardFFN -- the frozen projection is
    an orthonormal DCT-II matrix (`DCTLinear`) instead of Hadamard/DFT, AND
    the learnable activation is a pure cosine series (`DCTActivation`, K=5
    by default) instead of the full sin+cos Fourier series (K=4).

    Square-only. DCTLinear is orthogonal but NOT symmetric, so (unlike
    HadamardFFN) the frozen synthesis step calls `proj_out.inverse(h)`
    (= h @ C, the true DCT inverse), not a second plain forward pass.
    """

    def __init__(self, d_model, d_ff=None, K=5, freeze_out=False, dropout=0.1):
        super().__init__()
        d_ff = d_ff or d_model
        assert d_ff == d_model, (
            f"DCTFFN's frozen projection is square-only; need "
            f"d_ff==d_model, got d_ff={d_ff} != d_model={d_model}.")
        self.d_model, self.d_ff, self.freeze_out = d_model, d_ff, freeze_out
        self.proj_in = DCTLinear(d_model)                                # FROZEN
        self.act = DCTActivation(d_ff, K=K, ref="gelu", init="true")     # learned
        self.drop = nn.Dropout(dropout)
        if freeze_out:
            self.proj_out = DCTLinear(d_model)
        else:
            self.out = nn.Linear(d_ff, d_model)                          # learned mix-back

    def forward(self, x):
        h = self.proj_in(x)          # frozen DCT features
        h = self.act(h)              # learnable cosine-series activation
        h = self.drop(h)
        if self.freeze_out:
            return self.proj_out.inverse(h)
        return self.out(h)


class MaxoutScalarFFN(nn.Module):
    """Fully-learnable, channel-mixing-free maxout FFN -- ported verbatim
    from a reference `MaxoutScalarKMLP` ("maxout_scalar10"),
    which there replaces a ViT's *entire* MLP block the same way this
    replaces the Transformer FFN's entire proj_in -> act -> proj_out
    pipeline. Unlike every other variant above, there is no frozen matrix
    AND no channel mixing at all: for each channel i, K independent
    learnable scalar multipliers w_0[i]..w_{K-1}[i] are applied to that SAME
    scalar x_i, and the output is their elementwise max --

        FFN(x)_i = max_{k=0..K-1} (w_k[i] * x_i)

    `w` is one nn.Parameter of shape (K, d_model), randn-init (matching the
    source exactly). This is the extreme end of the FourierNet spectrum:
    K*d_model learnable parameters and nothing else -- no DFT/Hadamard/DCT
    buffer, no up- or down-projection. Mechanistically it collapses to a
    learnable per-channel leaky-ReLU/PReLU (only max_k/min_k(w_k[i]) ever
    matter), included un-simplified for a direct, faithful comparison against
    the frozen-orthogonal-basis variants.
    """

    def __init__(self, d_model, K=10, dropout=0.1):
        super().__init__()
        self.d_model, self.K = d_model, K
        self.w = nn.Parameter(torch.randn(K, d_model))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        candidates = x.unsqueeze(-2) * self.w    # (..., K, d_model)
        out = candidates.max(dim=-2).values
        return self.drop(out)


class ConvGELUFFN(nn.Module):
    """The entire FFN replaced by a single, heavily weight-shared 1-D
    convolution across the d_model axis, followed by a plain fixed GELU --
    ported from the baseline study's `ConvGELUMLP` ("conv_gelu"),
    the mirror image of the DFT/Hadamard/DCT/Maxout variants above: there the
    linear part was FROZEN (or absent, for Maxout) and the nonlinearity did
    the work; here the linear part is genuinely LEARNABLE again, but
    deliberately tiny -- one shared (kernel_size + 1)-number kernel (weights
    + bias) slides along the d_model axis, mixing each channel only with its
    nearest neighbours -- a banded/Toeplitz linear map, the opposite extreme
    from a dense d_model x d_model matrix (frozen or not):

        FFN(x) = GELU( Conv1d_k(x) )

    Every one of the d_model output positions reuses the SAME kernel, so
    this "linear part" has only kernel_size+1 learnable parameters per block
    regardless of d_model, and GELU contributes zero activation parameters.
    kernel_size defaults to 5 (the source study's d_model=192 ViT setup) but
    `build_ffn` scales it up for wider d_model, since a fixed-width kernel
    covers a shrinking fraction of a bigger channel axis otherwise.

    The conv always runs in fp32 (autocast forced off around it): this
    degenerate in=1/out=1-channel Conv1d has no efficient bf16 kernel on the
    GPUs used in this study -- left under bf16 autocast it measured ~300x
    slower than fp32 (236ms vs 0.8ms per iteration at this shape), a
    training-throughput cliff, not a numerical-precision choice.
    """

    def __init__(self, d_model, kernel_size=5, dropout=0.1):
        super().__init__()
        self.d_model, self.kernel_size = d_model, kernel_size
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=True)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shape = x.shape
        z = x.reshape(-1, 1, self.d_model)
        with torch.autocast(device_type=z.device.type, enabled=False):
            z = self.conv(z.float())
        z = z.reshape(*shape).to(x.dtype)
        return self.drop(self.act(z))


class ConvMoEFFN(nn.Module):
    """Mixture of Conv (Kernel) Experts -- extends `ConvGELUFFN` the same way
    a Transformer FFN is turned into a Switch/GShard-style MoE FFN: instead
    of ONE shared tiny conv kernel, there are `num_experts` independent tiny
    conv kernels (each identical in shape to conv_gelu's own
    Conv1d(1,1,kernel_size) + GELU), and a lightweight per-token linear
    router picks a sparse top-k subset of experts for every token,
    softmax-normalises the gate logits of just that subset, and combines
    their outputs:

        FFN(x)_token = sum_{e in topk(router(x_token))}
                       softmax(router(x_token))_e * GELU( Conv1d_k^(e)(x_token) )

    Routing is per-token exactly like standard MoE (the "tokens" here are
    the flattened (batch*seq) positions the FFN is applied to, same as
    every other FFN variant in this file); `router` is a plain
    d_model -> num_experts nn.Linear, the smallest possible gate.

    Unlike a heavy MLP-expert MoE (where sparse dispatch exists to avoid
    computing unused experts' expensive matmuls), each expert here is only
    kernel_size+1 parameters and the conv itself is forced to fp32 for
    correctness (see `ConvGELUFFN`), so it is cheaper to compute all E
    experts densely and mask/combine after routing than to build a genuine
    scatter/gather dispatch -- mathematically identical top-k-softmax
    combine, just implemented dense-then-select. This keeps FLOPs
    negligible either way while the trainable parameter count still grows
    only linearly in num_experts (no d_model x d_model matrices anywhere).

    top_k < num_experts keeps the routing genuinely sparse (unselected
    experts get zero gradient signal for that token, matching real MoE
    semantics); top_k == num_experts degenerates to a dense soft mixture.
    """

    def __init__(self, d_model, kernel_size=5, num_experts=4, top_k=2,
                dropout=0.1):
        super().__init__()
        assert 1 <= top_k <= num_experts, (
            f"ConvMoEFFN needs 1 <= top_k <= num_experts, got "
            f"top_k={top_k}, num_experts={num_experts}")
        self.d_model, self.kernel_size = d_model, kernel_size
        self.num_experts, self.top_k = num_experts, top_k
        self.experts = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=kernel_size, padding=kernel_size // 2,
                     bias=True)
            for _ in range(num_experts)])
        self.act = nn.GELU()
        self.router = nn.Linear(d_model, num_experts)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        shape = x.shape
        flat = x.reshape(-1, 1, self.d_model)          # (N, 1, d_model)
        logits = self.router(flat.squeeze(1))            # (N, E)
        top_logits, top_idx = logits.topk(self.top_k, dim=-1)   # (N, k)
        gates = torch.softmax(top_logits, dim=-1)         # (N, k), sums to 1

        with torch.autocast(device_type=flat.device.type, enabled=False):
            z32 = flat.float()
            expert_out = torch.stack(
                [e(z32).squeeze(1) for e in self.experts], dim=1)  # (N, E, d_model)
        expert_out = self.act(expert_out).to(x.dtype)

        idx = top_idx.unsqueeze(-1).expand(-1, -1, self.d_model)   # (N, k, d_model)
        picked = expert_out.gather(1, idx)                # (N, k, d_model)
        mixed = (picked * gates.unsqueeze(-1)).sum(dim=1)  # (N, d_model)
        return self.drop(mixed.reshape(*shape))


class HadamardBottleneckFFN(nn.Module):
    """Frozen truncated-Hadamard DOWN-projection (d_model -> d_ff, d_ff <
    d_model) + learnable Fourier-series activation (in the bottleneck) +
    LEARNABLE up-projection back to d_model -- the asymmetric-width
    analogue of FourierFFN(freeze_out=False). Every other frozen-projection
    variant above (fourier/fourier_mag/hadamard_frozenout/dct_frozenout)
    keeps the frozen transform SQUARE (d_ff == d_model): it's a change of
    basis at fixed width, not a real dimensionality reduction. This one
    genuinely reduces dimensionality through the frozen matrix (a real
    bottleneck), and asks the up-projection -- a real learnable Linear, not
    a second frozen transform -- to do the work of expanding back out.

    Needs d_ff < d_model explicitly (e.g. d_model=768, d_ff=256); the
    frozen HadamardLinearTrunc handles the non-power-of-2 d_model=768 case
    via `fourier_layers.hadamard_matrix`'s Kronecker-factoring fallback.
    """

    def __init__(self, d_model, d_ff, K=4, dropout=0.1):
        super().__init__()
        assert d_ff < d_model, (
            f"HadamardBottleneckFFN needs d_ff < d_model (a genuine "
            f"bottleneck), got d_ff={d_ff} >= d_model={d_model}.")
        self.d_model, self.d_ff = d_model, d_ff
        self.proj_in = HadamardLinearTrunc(d_model, d_ff)                 # FROZEN
        self.act = FourierActivation(d_ff, K=K, ref="gelu", init="true")  # learned
        self.drop = nn.Dropout(dropout)
        self.out = nn.Linear(d_ff, d_model)                               # learned up-projection

    def forward(self, x):
        h = self.proj_in(x)           # frozen Hadamard down-projection
        h = self.act(h)               # learnable Fourier-series activation
        h = self.drop(h)
        return self.out(h)            # learned up-projection


def conv_kernel_size(d_model):
    """Odd kernel width that grows with d_model: 5 at d_model=256 (matching
    the baseline study's d_model=192 default), 9 at 512, 17 at 1024
    -- a fixed-width-5 kernel would cover a shrinking fraction of a wider
    channel axis, so scale it roughly with d_model // 64 (forced odd)."""
    return max(5, (d_model // 64) | 1)


# --------------------------------------------------------------------------- #
FFN_VARIANTS = [
    "standard",             # d_ff = 4*d_model, learned FFN, GELU  (reference)
    "standard_narrow",      # d_ff = d_model,   learned FFN, GELU  (width-matched)
    "standard_wshare",      # d_ff = 4*d_model, GELU (fixed, zero-param, exactly
                            # like `standard`), but fc1/fc2 are built ONCE and
                            # shared by reference across all `depth` blocks --
                            # the GELU-baseline control for fact_k2_global_wshare's
                            # weight-tying: isolates whether tying the FFN's
                            # linear weights alone (no shared activation) already
                            # explains any of that variant's accuracy/param
                            # story, versus needing the shared FourierActivation
                            # too. Any "<base_kind>_wshare" name works the same
                            # way (see cv_vit.py: any ffn_kind ending in
                            # "_wshare" gets its FFN built once and shared).
    "fact_k2",              # d_ff = 4*d_model, learned FFN, GELU -> learnable
                            # FourierActivation K=2 ("FACT") -- W1/W2 stay fully
                            # learned; isolates the activation swap alone (vs.
                            # `fourier`, which also freezes proj_in as a DFT)
    "fact_k2_narrow",       # width-matched fact_k2: d_ff = d_model (no 4x
                            # expansion) -- the FACT analogue of
                            # `standard_narrow`, isolating the activation swap
                            # at fixed (narrow) width rather than fixed 4x width
    "fact_k2_shared",       # d_ff = 4*d_model, same as fact_k2 but the
                            # FourierActivation coefficients are TIED across
                            # all d_ff neurons in a layer (1 set of {a0,a,b}
                            # instead of d_ff independent sets) -- isolates
                            # whether per-neuron specialisation matters
    "fact_k2_global",       # d_ff = 4*d_model; ONE single FourierActivation
                            # K=2 (tied across neurons too) is instantiated
                            # ONCE and shared BY REFERENCE across all 6
                            # encoder+decoder FFN layers -- the whole
                            # network applies the exact same nonlinearity
                            # everywhere. W1/W2 (linear parts) stay
                            # per-layer/independent, same as every other
                            # FACT variant; only the activation is global.
    "fact_k2_embed",        # fact_k2_global exactly -- ONE shared
                            # FourierActivation(K=2) across all 6 FFNs -- with
                            # that SAME module additionally applied to the
                            # input embeddings, residually, right before the
                            # first block: x = x + shared_act(x). Not a second
                            # activation: shared=True makes the coefficients
                            # (1,)/(1,K), so the single instance broadcasts
                            # over the 192-wide embeddings and the 768-wide
                            # FFN hidden alike. Activation cost is therefore
                            # IDENTICAL to fact_k2_global's (5 params), so the
                            # pair isolates WHERE the one global nonlinearity
                            # is applied, with no parameter confound. Adds a
                            # nonlinearity where the architecture had none --
                            # patch_embed -> +pos_embed -> dropout ->
                            # blocks[0] is otherwise entirely linear.
                            # Model-level change, see cv_vit.VisionTransformer;
                            # the FFN is byte-identical to fact_k2_global's.
    "fact_k2_global_wshare", # identical recipe to fact_k2_global (one
                            # globally-shared FourierActivation), PLUS the
                            # FFN's own linear weights (fc1, fc2) are ALSO
                            # built ONCE and shared by reference across all
                            # `depth` blocks -- so the entire FFN sub-module
                            # (linear weights + activation) is a single set
                            # of parameters reused at every layer, a
                            # universal-transformer-style weight-tying
                            # applied only to the FFN (attention stays
                            # per-layer/independent). Any
                            # "fact_kK_global_wshare" name works the same way
                            # for other K (see parse_global_wshare in
                            # fourier_layers.py).
    "standard_ntied_wshare", # GELU (fixed, zero activation params), but fc1/fc2
                            # collapse to a single shared weight vector reused
                            # by every hidden neuron WITHIN a layer (rank-1
                            # linear part, see NeuronTiedFFN), AND that one FFN
                            # instance is built ONCE and shared across all
                            # `depth` blocks (the "_wshare" mechanism) -- the
                            # GELU-baseline control for
                            # fact_k2_global_ntied_wshare, isolating whether
                            # this much more aggressive weight-sharing (within
                            # AND across layers) already explains any of that
                            # variant's story, without a shared activation.
    "fact_k2_global_ntied_wshare", # identical recipe to fact_k2_global_wshare
                            # (one globally-shared FourierActivation, one FFN
                            # instance shared across all `depth` blocks), PLUS
                            # that one FFN's fc1/fc2 collapse to a single
                            # shared weight vector reused by every hidden
                            # neuron WITHIN the layer too (see NeuronTiedFFN) --
                            # the most aggressive point in this study's
                            # FFN-weight-sharing sweep: neurons tied to each
                            # other within a layer AND the whole FFN tied
                            # across every layer. Any "fact_kK_global_ntied
                            # (_wshare)?" name works the same way for other K
                            # (see parse_global_ntied in fourier_layers.py).
    "fact_k3_global",       # identical recipe to fact_k2_global, but the one
                            # globally-shared FourierActivation uses K=3
                            # harmonics instead of K=2 -- isolates the effect
                            # of series order on the single global nonlinearity.
    "fact_k1_global",       # identical recipe to fact_k2_global, but K=1
                            # (a single harmonic) -- the coarsest global
                            # approximation of GELU this study's naming
                            # convention supports.
    "fact_k2_global_phase", # identical recipe to fact_k2_global (one globally-shared
                            # FourierActivation, harmonics {a0,a,b} tied across neurons),
                            # PLUS every neuron adds its own small learnable phase-shifted
                            # sinusoid eps*cos(w*t + phi_i) on top of that shared curve --
                            # a richer neuron-specific signal than a per-neuron constant,
                            # and unlike "_routed" (a mix of the existing harmonics) a
                            # genuinely new oscillatory degree of freedom. See
                            # FourierActivation's `phase` arg / parse_global_phase in
                            # fourier_layers.py. Any "fact_kK_global_phase" name works
                            # the same way for other K.
    "fact_k1_global_conv",  # fact_k1_global, PLUS every layer's own FFN gets
                            # its own learnable 1-D convolution (banded,
                            # shared kernel across the d_ff axis, same recipe
                            # as conv_gelu) applied to the pre-activations
                            # (fc1(x)) right before the one globally-shared
                            # K=1 FourierActivation -- isolates whether cheap
                            # per-layer channel mixing helps a single global
                            # nonlinearity. Any "fact_kK_global_conv" name
                            # works the same way for other K (see
                            # parse_global_k/parse_global_conv in
                            # fourier_layers.py).
    "acon_c",               # d_ff = 4*d_model, learned FFN, GELU -> ACON-C
                            # (Ma et al., CVPR 2021): f(x) = (p1-p2)x*sigmoid(
                            # beta(p1-p2)x) + p2*x with CHANNEL-WISE learnable
                            # {p1,p2,beta} over the d_ff axis, one independent
                            # AconC per block -- the faithful port of the
                            # official acon.py (github.com/nmaac/acon), whose
                            # (1,width,1,1) NCHW params become (width,) here.
                            # Channel-wise is the paper's own best ablation
                            # setting. Init is the paper's beta=p1=1, p2=0
                            # (starts exactly at Swish), not acon.py's randn.
    "acon_c_global",        # acon_c with ONE {p1,p2,beta} triple (scalars,
                            # tied across neurons) shared BY REFERENCE across
                            # every block -- 3 activation params for the whole
                            # network. The ACON analogue of fact_k2_global's
                            # sharing convention, and the paper's "layer-wise"
                            # setting taken to its global extreme, so the two
                            # globally-shared activations are comparable.
    "fourier",              # frozen real-DFT + Fourier act + learned mix-back
    "fourier_mag",          # frozen |DFT| magnitude + Fourier act + learned mix-back
    "fourier_frozenout",    # frozen real-DFT + Fourier act + FROZEN inverse-DFT out
    "hadamard_frozenout",   # frozen Hadamard + Fourier act + FROZEN Hadamard out
    "dct_frozenout",        # frozen DCT-II + cosine-series act (K=5) + FROZEN DCT out
    "maxout_scalar10",      # no frozen matrix, no channel mixing: learnable K=10 per-channel scalar maxout
    "conv_gelu",            # learnable weight-shared 1-D conv (banded, tiny) + fixed GELU
    "conv_gelu_inact",      # conv_gelu FFN + one learnable FourierActivation(K=4) applied
                            # residually to the raw input embeddings (enc + dec), before
                            # the first block -- ported from the baseline study's
                            # best-performing `conv_gelu_inact` ViT variant. Model-level change,
                            # made in the model file; the FFN itself is identical to conv_gelu.
    "conv_gelu_inact_k2",   # same as conv_gelu_inact, but the input FourierActivation
                            # uses K=2 -- the exact value used in the source ViT study,
                            # instead of this study's own K=4 convention.
    "hadamard_bottleneck",  # frozen truncated-Hadamard down-proj (d_model->d_ff,
                            # d_ff<d_model) + learned Fourier act + LEARNABLE
                            # up-proj back to d_model. Needs d_ff < d_model
                            # explicitly (e.g. --d-model 768 --d-ff 256), unlike
                            # every other variant here (which use d_ff==d_model).
    "moce",                 # Mixture of Conv (Kernel) Experts: conv_gelu's single
                            # shared tiny conv kernel replaced by num_experts=4
                            # independent tiny conv kernels + fixed GELU, with a
                            # per-token linear router that top_k=2-routes each
                            # token and softmax-combines the selected experts'
                            # outputs -- the MoE recipe applied to conv_gelu.
]


def build_ffn(kind, d_model, d_ff_standard, dropout, K=4, act=None,
              ntied_zero_init_weights=False, ntied_rank=1, ntied_const_init=None,
              ntied_const_init_all=False):
    if kind == "standard":
        return StandardFFN(d_model, d_ff_standard, dropout)
    if kind == "standard_narrow":
        return StandardFFN(d_model, d_model, dropout)
    if kind == "standard_wshare":
        # Identical to "standard" (fixed GELU, zero activation params) -- the
        # weight-tying itself is a cv_vit.py-level concern (building this ONE
        # instance once and sharing it by reference), not something this
        # module needs to know about.
        return StandardFFN(d_model, d_ff_standard, dropout)
    if kind == "standard_ntied_wshare":
        # Fixed GELU, but fc1/fc2 collapse to a single shared vector reused
        # by every hidden neuron within the layer (NeuronTiedFFN) -- the
        # depth-wise sharing (this ONE instance reused across all `depth`
        # blocks) is, as with standard_wshare, a cv_vit.py-level concern.
        return NeuronTiedFFN(d_model, d_ff_standard, dropout=dropout, act=nn.GELU(),
                             zero_init_weights=ntied_zero_init_weights, rank=ntied_rank,
                             const_init_weights=ntied_const_init,
                             const_init_all=ntied_const_init_all)
    if kind == "acon_c":
        # ACON-C (Ma et al., CVPR 2021) in place of GELU, everything else in
        # the FFN untouched -- channel-wise {p1, p2, beta} over the d_ff
        # hidden axis, built FRESH here so each block gets its own, which is
        # the faithful port of the official acon.py (one AconC per layer, not
        # shared across depth). FactFFN is reused purely as the host for an
        # injected activation: with `act` given it is exactly
        # fc2(drop(act(fc1(x)))), i.e. the standard FFN with the nonlinearity
        # swapped, which is the only difference this comparison is about.
        return FactFFN(d_model, d_ff_standard, dropout=dropout,
                       act=AconC(d_ff_standard))
    if parse_acon_global(kind):
        # ONE AconC (shared=True, so {p1, p2, beta} are scalars tied across
        # neurons too) built once by the caller and reused by every block --
        # 3 learnable activation params for the whole network, the same
        # sharing convention fact_kK_global uses, so the two are comparable.
        assert act is not None, (
            f"{kind} needs a shared `act` module built once by the caller "
            "(see cv_vit.py's VisionTransformer.__init__) and passed to "
            "every layer")
        return FactFFN(d_model, d_ff_standard, dropout=dropout, act=act)
    if parse_pau_global(kind):
        # ONE PAU (shared=True, so {a, b} are (1, m+1)/(1, n) tied across
        # neurons too) built once by the caller and reused by every block --
        # the same sharing convention fact_kK_global/acon_c_global use, so
        # all three globally-shared activations are comparable.
        assert act is not None, (
            f"{kind} needs a shared `act` module built once by the caller "
            "(see cv_vit.py's VisionTransformer.__init__) and passed to "
            "every layer")
        return FactFFN(d_model, d_ff_standard, dropout=dropout, act=act)
    if kind == "fact_k2":
        # K=2 is a deliberate per-variant choice (this study's `fact_k2`
        # naming convention), not the generic K passed in from the caller.
        return FactFFN(d_model, d_ff_standard, K=2, dropout=dropout)
    if kind == "fact_k2_narrow":
        return FactFFN(d_model, d_model, K=2, dropout=dropout)
    if kind == "fact_k2_shared":
        return FactFFN(d_model, d_ff_standard, K=2, dropout=dropout, shared=True)
    if parse_global_k(kind) is not None:
        # `act` must be the ONE FourierActivation instance the caller
        # (TransformerNMT / VisionTransformer) built once and is reusing
        # across every layer -- this variant has no meaning per-layer in
        # isolation. Its K is fixed by however the caller built it.
        assert act is not None, (
            f"{kind} needs a shared `act` module built once by the caller "
            "(see TransformerNMT.__init__ / VisionTransformer.__init__) "
            "and passed to every layer")
        if parse_global_ntied(kind):
            # fc1/fc2 collapse to a single shared vector reused by every
            # hidden neuron within the layer, on top of the passed-in
            # globally-shared activation -- see NeuronTiedFFN.
            return NeuronTiedFFN(d_model, d_ff_standard, dropout=dropout, act=act,
                                zero_init_weights=ntied_zero_init_weights, rank=ntied_rank,
                                const_init_weights=ntied_const_init,
                                const_init_all=ntied_const_init_all)
        return FactFFN(d_model, d_ff_standard, dropout=dropout, act=act,
                       pre_act_conv=parse_global_conv(kind))
    if parse_global_2act_k(kind) is not None:
        # `act` is whichever of the two per-half shared FourierActivation
        # instances the caller assigned THIS block to (see cv_vit.py's
        # VisionTransformer.__init__) -- same FactFFN host as fact_kK_global,
        # just with a different act passed in per block instead of the same
        # one for every block.
        assert act is not None, (
            f"{kind} needs a shared `act` module (one of the two per-half "
            "activations) built by the caller and passed per block "
            "(see cv_vit.py's VisionTransformer.__init__)")
        return FactFFN(d_model, d_ff_standard, dropout=dropout, act=act)
    if kind == "fourier":
        return FourierFFN(d_model, d_model, K=K, use_imag=False,
                          freeze_out=False, dropout=dropout)
    if kind == "fourier_mag":
        return FourierFFN(d_model, d_model, K=K, use_imag=True,
                          freeze_out=False, dropout=dropout)
    if kind == "fourier_frozenout":
        return FourierFFN(d_model, d_model, K=K, use_imag=False,
                          freeze_out=True, dropout=dropout)
    if kind == "hadamard_frozenout":
        return HadamardFFN(d_model, d_model, K=K, freeze_out=True, dropout=dropout)
    if kind == "dct_frozenout":
        # K=5 is a deliberate per-variant choice (cosine-only series needs a
        # different harmonic count than the sin+cos Fourier series), not the
        # generic K passed in from the caller -- every other variant here
        # still runs at its own established K=4.
        return DCTFFN(d_model, d_model, K=5, freeze_out=True, dropout=dropout)
    if kind == "maxout_scalar10":
        return MaxoutScalarFFN(d_model, K=10, dropout=dropout)
    if kind in ("conv_gelu", "conv_gelu_inact", "conv_gelu_inact_k2"):
        # conv_gelu_inact's per-layer FFN is identical to conv_gelu; the extra
        # input-side activation it adds is a model-level change handled in
        # transformer_nmt.TransformerNMT, not here.
        return ConvGELUFFN(d_model, kernel_size=conv_kernel_size(d_model), dropout=dropout)
    if kind == "hadamard_bottleneck":
        return HadamardBottleneckFFN(d_model, d_ff_standard, K=K, dropout=dropout)
    if kind == "moce":
        # num_experts=4, top_k=2 is this variant's own established convention
        # (classic top-2-of-N MoE routing), not the generic K passed in.
        return ConvMoEFFN(d_model, kernel_size=conv_kernel_size(d_model),
                          num_experts=4, top_k=2, dropout=dropout)
    raise ValueError(f"unknown ffn kind: {kind}")
