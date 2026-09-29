"""Core building blocks for the frozen-transform + learnable-activation FFN.

Two ideas:

1. DFTLinear  -- the weight matrix is the basis of an M x N DFT matrix and is
   FROZEN.  Because the DFT matrix is complex, the input is pushed through the
   real part (cos) AND the imaginary part (-sin); the two projections are
   combined and we take the magnitude |z| = sqrt(re^2 + im^2).  This is exactly
   the magnitude of the (truncated) discrete Fourier transform of the input.

2. FourierActivation -- the pointwise nonlinearity is written as a truncated
   Fourier series  phi(t) = a0 + sum_k a_k cos(k t) + b_k sin(k t).  The
   coefficients {a0, a_k, b_k} are the LEARNABLE parameters (per neuron), and
   they are INITIALISED FROM THE TRUE FOURIER COEFFICIENTS of a reference
   activation (GELU by default), not from random values.
"""
import math
import re

import numpy as np
import torch
import torch.nn as nn

# np.trapz was removed in NumPy 2.0 in favor of np.trapezoid (same function,
# renamed); this keeps true_fourier_coeffs working on NumPy versions on
# either side of that split.
_trapz = getattr(np, "trapezoid", None) or np.trapz

# --------------------------------------------------------------------------- #
#  "fact_kK_global[_conv][_ntied][_wshare]" variant-name parsing (K = any
#  positive int, not just the 2/3 this study has used so far) -- shared by
#  cv_vit.py (which builds the ONE shared FourierActivation, and, for
#  _wshare, the ONE shared FFN module) and fourier_ffn.build_ffn (which
#  dispatches every layer's FFN to reuse it), so a caller sweeping K (e.g.
#  backing off fact_k3_global -> fact_k2_global -> fact_k1_global when
#  validation loss rises) doesn't need a new hardcoded branch per K value.
#  The optional "_conv" suffix (e.g. fact_k1_global_conv) additionally gives
#  every layer's own FFN a learnable, per-layer 1-D convolution across the
#  d_ff pre-activation axis, applied right before the one globally-shared
#  FourierActivation -- see FactFFN/build_ffn in fourier_ffn.py.
#  The optional "_ntied" suffix (e.g. fact_k2_global_ntied) additionally
#  collapses the FFN's own linear weights to a single shared vector reused
#  by every one of the d_ff hidden neurons WITHIN a layer (rank-1 fc1/fc2,
#  diversity between neurons coming only from a per-neuron bias) -- see
#  NeuronTiedFFN/build_ffn in fourier_ffn.py. This is a different sharing
#  axis than "_wshare" below: "_ntied" ties neurons to each other within one
#  layer; "_wshare" ties one layer's FFN to every other layer's.
#  The optional "_wshare" suffix (e.g. fact_k2_global_wshare) additionally
#  ties the FFN's own linear weights (fc1, fc2) across all `depth` blocks --
#  built ONCE in cv_vit.VisionTransformer.__init__ and handed by reference to
#  every Block, the same convention already used for the shared activation
#  itself. Attention stays per-layer/independent either way; only the FFN
#  (linear weights + activation) is tied. "_ntied" and "_wshare" combine
#  freely (fact_k2_global_ntied_wshare): weight sharing within a layer AND
#  across layers at once.
#  The optional "_routed" suffix (e.g. fact_k5_global_routed_wshare)
#  additionally gives the ONE globally-shared FourierActivation a learnable,
#  per-neuron ROUTING gate over its K harmonics (see FourierActivation's
#  `route` argument): the K harmonic coefficients {a_k, b_k} themselves stay
#  literally global (one shared set, identical to every other "_global"
#  variant), but each of the d_ff neurons gets its own learnable K-way
#  softmax weighting of how much each harmonic contributes to ITS output --
#  so neurons can specialise their MIX of the shared harmonics without ever
#  copying/duplicating the harmonics' amplitudes. Orthogonal to "_ntied"
#  (which collapses the FFN's own linear weights, not the activation) and
#  "_wshare" (which ties the linear weights across depth); combines freely
#  with either, e.g. fact_k5_global_routed_wshare.
#  The optional "_phase" suffix (e.g. fact_k2_global_phase_wshare)
#  additionally gives the ONE globally-shared FourierActivation a small,
#  richer per-neuron SIGNAL instead of just a routing MIX of the existing
#  harmonics: every neuron i adds its own phase-shifted unit sinusoid
#  eps * cos(w*t + phi_i) on top of the (still literally global) series
#  phi(t) = a0 + sum_k a_k cos(k w t) + b_k sin(k w t), where phi_i (one
#  learnable angle per neuron, randn-init) and eps (one learnable scalar,
#  shared by every neuron, small init) are the only new parameters -- see
#  FourierActivation's `phase` argument. Unlike "_routed" (a K-way softmax
#  MIX of the shared harmonics, which cannot express anything outside their
#  span), this literally adds a new per-neuron oscillatory degree of freedom
#  at the fundamental frequency, so neurons can diverge from the shared
#  curve's shape, not just re-weight it. Orthogonal to every other suffix;
#  combines freely, e.g. fact_k5_global_phase_wshare.
# --------------------------------------------------------------------------- #
GLOBAL_VARIANT_RE = re.compile(
    r"^fact_k(\d+)_global(_conv)?(_ntied)?(_routed)?(_phase)?(_wshare)?$")

# fact_kK_embed: fact_kK_global's FFN stack (ONE shared FourierActivation
# across all blocks) PLUS a second, independent shared FourierActivation
# applied to the input embeddings before the first block. Deliberately its own
# regex rather than an `_embed` suffix on GLOBAL_VARIANT_RE: every suffix there
# MODIFIES the single global activation, whereas this ADDS a second, separate
# one, and each parse_global_* flag must keep returning False for it.
EMBED_VARIANT_RE = re.compile(r"^fact_k(\d+)_embed$")


def parse_global_k(kind):
    m = GLOBAL_VARIANT_RE.match(kind)
    if m:
        return int(m.group(1))
    # fact_kK_embed's FFNs ARE fact_kK_global's -- same K, one shared
    # activation across depth -- so build_ffn's dispatch and cv_vit.py's
    # shared_act construction both need to see the K from here. The extra
    # embedding activation is parse_embed_k's business, not this one's.
    m = EMBED_VARIANT_RE.match(kind)
    return int(m.group(1)) if m else None


def parse_embed_k(kind):
    """K iff `kind` is a fact_kK_embed variant, else None.

    Signals that the MODEL (not the FFN) also applies the network's ONE shared
    FourierActivation to the input embeddings, residually, immediately before
    the block stack: `x = x + shared_act(x)`.

    It is literally the same module the FFNs use, not a second copy. shared=True
    makes the coefficients (1,) and (1,K), so a single instance broadcasts over
    the embed_dim-wide token embeddings and the d_ff-wide FFN hidden layer
    alike. fact_kK_embed therefore has EXACTLY as many activation parameters as
    fact_kK_global (1+2K, i.e. 5 at K=2) -- the two differ only in where that
    one nonlinearity is applied, so the comparison carries no parameter
    confound.

    The residual form -- not a bare `x = shared_act(x)` -- is this repo's
    established convention for an input activation: see
    transformer_nmt.TransformerNMT's conv_gelu_inact
    (`x = x + enc_input_act(x)`), itself ported from the baseline study's
    best-performing ViT variant. It leaves
    the identity path intact at init, so the comparison reflects what the added
    application point contributes instead of a large init-time shift in the
    embedding distribution.

    See cv_vit.VisionTransformer.__init__ and .forward.
    """
    m = EMBED_VARIANT_RE.match(kind)
    return int(m.group(1)) if m else None


# fact_kK_global_2act: like fact_kK_global (ONE globally-shared K-term
# activation across the whole network), but split into TWO independent
# globally-shared activations -- shared_act_1 covers the first half of the
# network's blocks (indices 0..depth/2-1), shared_act_2 the second half
# (depth/2..depth-1). Each half still ties its own {a0, a_k, b_k} across
# every neuron AND every block within that half, exactly like fact_kK_global
# does across the whole network -- so this asks whether the two halves of a
# deep network want different nonlinearity shapes, at 2*(1+2K) activation
# params instead of fact_kK_global's 1+2K. Its own regex (like
# EMBED_VARIANT_RE) rather than a GLOBAL_VARIANT_RE suffix: every suffix
# there modifies the SINGLE global activation, whereas this builds two
# separate ones, and every parse_global_* flag must keep returning False for
# it. See cv_vit.VisionTransformer.__init__.
TWOACT_VARIANT_RE = re.compile(r"^fact_k(\d+)_global_2act$")


def parse_global_2act_k(kind):
    """K iff `kind` is a fact_kK_global_2act variant, else None."""
    m = TWOACT_VARIANT_RE.match(kind)
    return int(m.group(1)) if m else None


def parse_global_conv(kind):
    """True iff `kind` is a fact_kK_global_conv variant -- i.e. the FFN also
    applies its own (per-layer, NOT shared) learnable 1-D convolution across
    the d_ff pre-activation axis, right before the globally-shared
    FourierActivation. Isolates whether cheap per-layer channel mixing on
    top of a single global nonlinearity helps -- the same channel-mixing
    question `conv_gelu`/ConvGELUFFN already asks of a fixed GELU."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return bool(m and m.group(2))


def parse_global_ntied(kind):
    """True iff `kind` is a fact_kK_global_ntied(_wshare)? variant -- i.e.
    the FFN's fc1/fc2 collapse to a single shared weight vector reused by
    every hidden neuron WITHIN one layer (see NeuronTiedFFN in
    fourier_ffn.py), on top of the globally-shared activation. Orthogonal to
    parse_global_wshare (which ties a layer's FFN to every OTHER layer's,
    not neurons to each other within one layer)."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return bool(m and m.group(3))


def parse_global_routed(kind):
    """True iff `kind` is a fact_kK_global_routed(_wshare)? variant -- i.e.
    the one globally-shared FourierActivation additionally routes: a
    learnable, per-neuron softmax gate over the K harmonics, layered on top
    of the literally-global {a0, a_k, b_k} coefficients every "_global"
    variant already shares. See FourierActivation's `route` argument."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return bool(m and m.group(4))


def parse_global_phase(kind):
    """True iff `kind` is a fact_kK_global_phase(_wshare)? variant -- i.e.
    the one globally-shared FourierActivation additionally adds a small,
    learnable, per-neuron phase-shifted sinusoid `eps * cos(w*t + phi_i)` on
    top of the literally-global {a0, a_k, b_k} series every "_global" variant
    already shares -- a richer, oscillatory neuron-specific signal, not just
    a constant offset or a re-weighting of the existing harmonics. See
    FourierActivation's `phase` argument."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return bool(m and m.group(5))


def parse_global_wshare(kind):
    """True iff `kind` is a fact_kK_global_wshare variant -- i.e. the FFN's
    own linear weights (fc1, fc2), not just its activation, are built ONCE
    and shared by reference across all `depth` blocks. See cv_vit.py's
    VisionTransformer.__init__ (builds the one shared FFN module) and
    Block.__init__ (accepts it via `shared_ffn` instead of building its
    own)."""
    m = GLOBAL_VARIANT_RE.match(kind)
    return bool(m and m.group(6))


def parse_acon_global(kind):
    """True iff `kind` is "acon_c_global" -- the ACON-C activation (see AconC
    below) with ONE {p1, p2, beta} triple shared across every neuron AND
    every block, mirroring fact_kK_global's sharing convention exactly so the
    two globally-shared activations are directly comparable. Deliberately a
    plain equality check rather than a GLOBAL_VARIANT_RE-style regex: ACON-C
    has no _conv/_ntied/_routed/_phase/_wshare suffix family to parse, and
    GLOBAL_VARIANT_RE only ever matches fact_kK_global*, so the two naming
    schemes cannot collide."""
    return kind == "acon_c_global"


def parse_pau_global(kind):
    """True iff `kind` is "pau_global" -- the Pade Activation Unit (see PAU
    below) with ONE {a, b} coefficient pair shared across every neuron AND
    every block, mirroring fact_kK_global's and acon_c_global's sharing
    convention exactly so all three globally-shared activations are directly
    comparable. Plain equality check for the same reason parse_acon_global
    is: PAU has no suffix family to parse, and GLOBAL_VARIANT_RE only ever
    matches fact_kK_global*, so the naming schemes cannot collide."""
    return kind == "pau_global"


# --------------------------------------------------------------------------- #
#  True Fourier coefficients of a reference activation on [-pi, pi]
# --------------------------------------------------------------------------- #
def gelu_np(t):
    # exact (erf) GELU
    from scipy.special import erf
    return 0.5 * t * (1.0 + erf(t / math.sqrt(2.0)))


def gelu_exp_np(t):
    # sigmoid/exp approximation of GELU: x * sigmoid(1.702 x)
    # = x / (1 + exp(-1.702 x)) -- an alternative to the tanh approximation,
    # used e.g. in the original GPT/BERT implementations in place of the
    # exact erf-based GELU above.
    return t / (1.0 + np.exp(-1.702 * t))


def leaky_relu_np(t, slope=0.01):
    return np.where(t >= 0, t, slope * t)


def linear_np(t):
    # identity -- reference for a "linear approximation" init: the truncated
    # Fourier series of f(t)=t on [-pi,pi] is the classic sawtooth expansion
    # (a0=0, a_k=0, b_k = 2*(-1)^(k+1)/k), i.e. every neuron starts as a
    # (K-term-truncated) straight line instead of a copy of GELU.
    return t


def relu6_np(t):
    # MobileNetV2's actual activation (Sandler et al. 2018): ReLU capped at
    # 6, i.e. min(max(t, 0), 6) -- used as the FAct init reference for CNN
    # studies that replace a network's real ReLU6 layers with a shared
    # Fourier activation, so the K-term series starts as a genuine
    # approximation of what it's replacing rather than a mismatched
    # transformer-style GELU reference.
    return np.clip(t, 0.0, 6.0)


REFERENCE_ACTS = {"gelu": gelu_np, "gelu_exp": gelu_exp_np,
                   "leaky_relu": leaky_relu_np, "linear": linear_np,
                   "relu6": relu6_np}


def true_fourier_coeffs(K, ref="gelu", L=math.pi, n_grid=4096):
    """Return (a0, a[K], b[K]): the true Fourier-series coefficients of the
    reference activation on [-L, L], computed by numerical (trapezoid)
    integration of the Euler formulas.

        a0  = 1/(2L) * int_{-L}^{L} f(t) dt
        a_k = 1/L    * int_{-L}^{L} f(t) cos(k*w*t) dt      (w = pi/L)
        b_k = 1/L    * int_{-L}^{L} f(t) sin(k*w*t) dt

    With L = pi the base frequency w = 1, so phi(t) uses cos(k t), sin(k t).
    """
    f = REFERENCE_ACTS[ref]
    t = np.linspace(-L, L, n_grid)
    ft = f(t)
    w = math.pi / L
    a0 = _trapz(ft, t) / (2 * L)
    a = np.zeros(K)
    b = np.zeros(K)
    for k in range(1, K + 1):
        a[k - 1] = _trapz(ft * np.cos(k * w * t), t) / L
        b[k - 1] = _trapz(ft * np.sin(k * w * t), t) / L
    return float(a0), a.astype(np.float32), b.astype(np.float32), float(w)


def true_pade_coeffs(m=5, n=4, ref="gelu", L=math.pi, n_grid=4096):
    """Return (a[m+1], b[n]): numerator/denominator coefficients of a degree
    (m, n) Pade rational approximation to the reference activation on
    [-L, L], P(t) = sum_{j=0}^m a_j t^j, Q(t) = 1 + sum_{k=1}^n b_k t^k
    (PAU's forward pass takes |b_k| in Q -- see PAU below -- so this fit,
    which allows either sign, is only a starting point, not the guaranteed
    pole-free form).

    Solved as a LINEAR least-squares problem, the standard trick for fitting
    a rational function without iterating: the target relation
    f(t) = P(t)/Q(t) rearranges to P(t) - f(t)*Q(t) = 0, i.e.

        sum_j a_j t^j  -  sum_k b_k (f(t) t^k)  =  f(t)

    which is linear in the unknowns {a_j, b_k} even though the fitted
    function itself is not.
    """
    f = REFERENCE_ACTS[ref]
    t = np.linspace(-L, L, n_grid)
    ft = f(t)
    cols = [t ** j for j in range(m + 1)] + [-ft * (t ** k) for k in range(1, n + 1)]
    A = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(A, ft, rcond=None)
    a = coef[:m + 1].astype(np.float32)
    b = coef[m + 1:].astype(np.float32)
    return a, b


def fourier_series_eval(t, a0, a, b, w=1.0):
    """Evaluate the truncated series on a numpy array (used for plots/tests)."""
    out = np.full_like(t, a0, dtype=np.float64)
    for k in range(1, len(a) + 1):
        out += a[k - 1] * np.cos(k * w * t) + b[k - 1] * np.sin(k * w * t)
    return out


# --------------------------------------------------------------------------- #
#  Frozen DFT projection layer
# --------------------------------------------------------------------------- #
class DFTLinear(nn.Module):
    """Frozen truncated-DFT magnitude projection R^N -> R^M.

    Wr[m, n] = cos(2*pi * m * n / N)      (real part of the DFT basis)
    Wi[m, n] = -sin(2*pi * m * n / N)     (imaginary part)

    forward(x):  zr = x Wr^T ; zi = x Wi^T ; return sqrt(zr^2 + zi^2)/sqrt(N)

    Row m selects the m-th frequency, so we keep the M lowest frequencies of the
    input (a truncated DFT).  Requires M <= N so every row is a genuine DFT
    frequency below Nyquist.  The 1/sqrt(N) keeps the magnitude scale sane.
    """

    def __init__(self, in_dim, out_dim, use_imag=True):
        super().__init__()
        assert out_dim <= in_dim, (
            f"DFTLinear needs out_dim <= in_dim (got {out_dim} > {in_dim}); "
            "otherwise rows would repeat frequencies above Nyquist.")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.use_imag = use_imag                 # False -> real (cos) part only
        n = np.arange(in_dim)
        m = np.arange(out_dim)
        ang = 2.0 * np.pi * np.outer(m, n) / in_dim        # (M, N)
        Wr = np.cos(ang).astype(np.float32)
        Wi = -np.sin(ang).astype(np.float32)
        # frozen buffers (saved with state_dict, never optimised)
        self.register_buffer("Wr", torch.from_numpy(Wr))
        self.register_buffer("Wi", torch.from_numpy(Wi))
        self.scale = 1.0 / math.sqrt(in_dim)
        self.eps = 1e-6

    def forward(self, x):
        zr = x @ self.Wr.t()          # (B, M) real projection
        if not self.use_imag:
            return zr * self.scale    # real (cosine) part only
        zi = x @ self.Wi.t()          # (B, M) imaginary projection
        mag = torch.sqrt(zr * zr + zi * zi + self.eps)
        return mag * self.scale


# --------------------------------------------------------------------------- #
#  Frozen orthonormal Hadamard projection (ablation: swap the frozen DFT
#  basis for a different fixed orthonormal basis with no frequency ordering)
# --------------------------------------------------------------------------- #
class HadamardLinear(nn.Module):
    """Frozen orthonormal Hadamard projection R^N -> R^N (square only).

    Uses the classic Sylvester construction (`scipy.linalg.hadamard`, valid
    for any power-of-2 N -- d_model=512=2^9 in this study qualifies),
    normalised by 1/sqrt(N) so `C` is a genuine orthogonal matrix
    (C C^T = I), registered as a buffer with requires_grad=False -- never
    touched by the optimizer, exactly like DFTLinear above, just a
    different fixed +-1 orthonormal basis instead of the complex-exponential
    DFT basis. Sylvester matrices are symmetric (C^T == C) by construction,
    so this same frozen matrix is also its own inverse: used as `proj_in`
    it is a lossless change of basis, and reused as `proj_out` (see
    `HadamardFFN`) it exactly undoes that change of basis.
    """

    def __init__(self, dim):
        super().__init__()
        from scipy.linalg import hadamard as _sylvester_hadamard
        assert dim > 0 and (dim & (dim - 1)) == 0, (
            f"HadamardLinear (Sylvester construction) needs dim to be a "
            f"power of 2, got dim={dim}")
        self.dim = dim
        H = _sylvester_hadamard(dim).astype(np.float32)
        C = H / math.sqrt(dim)
        self.register_buffer("C", torch.from_numpy(C))

    def forward(self, x):
        return x @ self.C.t()


# --------------------------------------------------------------------------- #
#  Genuine Hadamard matrices at orders that aren't a power of 2 (needed for a
#  frozen DOWN-projection, e.g. d_model=768 -> d_ff=256, where 768 isn't
#  reachable by Sylvester's construction alone)
# --------------------------------------------------------------------------- #
def _is_prime(k):
    if k < 2:
        return False
    for d in range(2, int(math.isqrt(k)) + 1):
        if k % d == 0:
            return False
    return True


_PALEY_HADAMARD_CACHE = {}


def paley_hadamard(n):
    """Genuine order-n Hadamard matrix (+-1 entries, H @ H.T == n*I) via the
    Paley-I construction: valid whenever q = n-1 is prime and q = 3 (mod 4).
    For a prime q, the Jacobsthal matrix Q[i,j] = chi(j-i) over GF(q) (chi =
    Legendre symbol, chi(0)=0) is skew-symmetric and satisfies Q Q^T = qI-J.
    Bordering it with a row/column of ones and subtracting I from the core
    gives a genuine order-(q+1) Hadamard matrix (verified numerically below,
    not just asserted). Cached per n.
    """
    if n in _PALEY_HADAMARD_CACHE:
        return _PALEY_HADAMARD_CACHE[n]
    q = n - 1
    assert _is_prime(q) and q % 4 == 3, (
        f"Paley-I construction needs n-1 prime and =3 mod 4; got n={n}, q={q}")
    a = np.arange(q)
    chi = np.array([0] + [1 if pow(int(x), (q - 1) // 2, q) == 1 else -1
                          for x in a[1:]])
    idx = (a.reshape(-1, 1) - a.reshape(1, -1)) % q     # idx[i,j] = (i-j) mod q
    Q = chi[(-idx) % q]                                  # Q[i,j] = chi(j-i)
    H = np.ones((n, n), dtype=np.int64)
    H[1:, 1:] = Q - np.eye(q, dtype=np.int64)
    assert np.array_equal(H @ H.T, n * np.eye(n, dtype=np.int64)), (
        f"Paley-I construction failed the orthogonality check for n={n}")
    _PALEY_HADAMARD_CACHE[n] = H
    return H


_HADAMARD_MATRIX_CACHE = {}


def hadamard_matrix(n):
    """Genuine +-1 order-n Hadamard matrix for n not necessarily a power of
    2 -- needed by `HadamardLinearTrunc`'s frozen down-projection at (e.g.)
    n=768=2^8*3, which Sylvester's construction (power-of-2 only) cannot
    reach directly. Tries, in order: (1) Sylvester, if n is a power of 2;
    (2) direct Paley-I, if n-1 is prime and =3 (mod 4); (3) the smallest
    power-of-2 divisor p of n whose cofactor q=n/p is itself a valid
    Hadamard order (1, 2, or Paley-I-compatible), combining Sylvester(p)
    tensor Hadamard(q) -- the Kronecker product of two Hadamard matrices is
    itself Hadamard, of order equal to the product of the two orders
    (standard fact). n=768 resolves via this third route as 2*384 (384-1=
    383 is prime and =3 mod 4, so Paley-I(384) is valid).
    """
    if n in _HADAMARD_MATRIX_CACHE:
        return _HADAMARD_MATRIX_CACHE[n]
    from scipy.linalg import hadamard as _sylvester_hadamard
    if n & (n - 1) == 0:                                  # power of 2
        H = _sylvester_hadamard(n).astype(np.int64)
    elif _is_prime(n - 1) and (n - 1) % 4 == 3:
        H = paley_hadamard(n)
    else:
        # Search increasing powers-of-2 divisors p of n for the smallest one
        # whose cofactor q=n/p is itself a valid Hadamard order (1, 2, or
        # Paley-I-compatible), then combine Sylvester(p) tensor Hadamard(q).
        p, q = None, None
        cand = 1
        while cand <= n:
            if n % cand == 0:
                c = n // cand
                if c == 1 or c == 2 or (_is_prime(c - 1) and (c - 1) % 4 == 3):
                    p, q = cand, c
                    break
            cand *= 2
        if p is None:
            raise ValueError(f"no known Hadamard construction for n={n}")
        Hp = _sylvester_hadamard(p).astype(np.int64) if p > 1 else np.ones((1, 1), dtype=np.int64)
        if q == 1:
            Hq = np.ones((1, 1), dtype=np.int64)
        elif q == 2:
            Hq = np.array([[1, 1], [1, -1]], dtype=np.int64)
        else:
            Hq = paley_hadamard(q)
        H = np.kron(Hp, Hq)
    assert np.array_equal(H @ H.T, n * np.eye(n, dtype=np.int64)), (
        f"hadamard_matrix failed the orthogonality check for n={n}")
    _HADAMARD_MATRIX_CACHE[n] = H
    return H


class HadamardLinearTrunc(nn.Module):
    """Frozen truncated Hadamard projection R^N -> R^M (M <= N, non-square)
    -- the rectangular analogue of `HadamardLinear` above, needed when the
    frozen projection genuinely REDUCES dimensionality (e.g. d_model=768 ->
    d_ff=256) rather than just changing basis at a fixed width. Takes the
    first M rows of a genuine order-N Hadamard matrix (`hadamard_matrix`,
    not restricted to powers of 2), normalised by 1/sqrt(N) -- those M rows
    are themselves exactly orthonormal (rows of an orthogonal matrix), even
    though the resulting M x N matrix isn't itself square/invertible.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        assert out_dim <= in_dim, (
            f"HadamardLinearTrunc needs out_dim <= in_dim, got "
            f"out_dim={out_dim} > in_dim={in_dim}.")
        self.in_dim, self.out_dim = in_dim, out_dim
        H = hadamard_matrix(in_dim).astype(np.float32)
        C = H[:out_dim] / math.sqrt(in_dim)
        self.register_buffer("C", torch.from_numpy(C))

    def forward(self, x):
        return x @ self.C.t()


# --------------------------------------------------------------------------- #
#  Frozen orthonormal DCT-II projection (ablation: a third fixed orthonormal
#  basis, paired below with a cosine-only activation instead of full Fourier)
# --------------------------------------------------------------------------- #
class DCTLinear(nn.Module):
    """Frozen orthonormal DCT-II projection R^N -> R^N (square only).

    C is the standard orthonormal DCT-II basis matrix (`scipy.fft.dct`,
    norm="ortho"), registered as a buffer with requires_grad=False -- never
    touched by the optimizer, same recipe as DFTLinear/HadamardLinear, just
    a third fixed orthonormal basis. Unlike Sylvester Hadamard, C is
    orthogonal but NOT symmetric (C^T != C), so forward() (x @ C^T) and
    inverse() (x @ C) are genuinely different operations -- inverse() is the
    exact synthesis/undo of forward() because C^-1 = C^T for an orthogonal
    matrix, so (C^T)^-1 = C.
    """

    def __init__(self, dim):
        super().__init__()
        from scipy.fft import dct as _scipy_dct
        self.dim = dim
        C = _scipy_dct(np.eye(dim, dtype=np.float64), type=2, norm="ortho",
                       axis=0).astype(np.float32)
        self.register_buffer("C", torch.from_numpy(C))

    def forward(self, x):
        return x @ self.C.t()

    def inverse(self, x):
        return x @ self.C


# --------------------------------------------------------------------------- #
#  ACON-C activation (Ma et al., "Activate or Not: Learning Customized
#  Activation", CVPR 2021 -- github.com/nmaac/acon)
# --------------------------------------------------------------------------- #
class AconC(nn.Module):
    """ACON-C: f(x) = (p1 - p2) x * sigmoid(beta (p1 - p2) x) + p2 x.

    A smooth maximum of the two linear functions p1*x and p2*x, with beta
    controlling how sharply the transition between them happens -- so the
    family subsumes ReLU (p1=1, p2=0, beta->inf), PReLU (p2 = the negative
    slope), and Swish/SiLU (= ACON-A, p1=1, p2=0, beta=1). All three of
    {p1, p2, beta} are learnable; the paper's own ablation (layer-wise 36.3 /
    channel-wise 34.8 / pixel-wise 37.2 top-1 err, ShuffleNetV2 0.5x on
    ImageNet) found CHANNEL-WISE parameters best, which is what the official
    acon.py implements as shape (1, width, 1, 1) over an NCHW feature map.

    Here the tensors are token sequences (B, N, d_ff), not images, so the
    channel axis is the LAST one and the parameters are (width,) -- ordinary
    broadcasting then ties each parameter across batch and tokens and varies
    it across hidden units, exactly as (1, width, 1, 1) ties across batch and
    spatial positions in the conv case.

    Initialisation is the paper's (beta = p1 = 1, p2 = 0), NOT the official
    repo's, which draws p1, p2 ~ N(0, 1) and would start the network from a
    random activation shape. The paper's init makes ACON-C start exactly at
    Swish/SiLU, so this variant begins from a known-sane nonlinearity -- the
    same courtesy FourierActivation's init="true" extends to the FAct
    variants (they start as a faithful GELU) and that the fixed-GELU baseline
    gets for free. Recorded as a deliberate deviation from acon.py.

    shared=True ties ONE {p1, p2, beta} triple across all `width` neurons
    (param shape (1,) instead of (width,)) -- i.e. the paper's "layer-wise"
    setting. Combined with cv_vit.py building a single instance and handing
    it to every block by reference ("acon_c_global"), this makes ACON-C a
    literally global activation: 3 learnable parameters for the entire
    network, the same sharing convention fact_kK_global uses for its
    {a0, a_k, b_k}, so the two are directly comparable.
    """

    def __init__(self, width, shared=False):
        super().__init__()
        self.width = width
        self.shared = shared
        shape = (1,) if shared else (width,)
        self.p1 = nn.Parameter(torch.ones(shape))
        self.p2 = nn.Parameter(torch.zeros(shape))
        self.beta = nn.Parameter(torch.ones(shape))

    def forward(self, x):
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x

    def extra_repr(self):
        return f"width={self.width}, shared={self.shared}"


# --------------------------------------------------------------------------- #
#  Pade Activation Unit baseline (per neuron, or one shared curve)
# --------------------------------------------------------------------------- #
class PAU(nn.Module):
    """Pade Activation Unit (Molina et al., "Pade Activation Units: End-to-end
    Learning of Flexible Activation Functions in Deep Networks", ICLR 2020):

        f(t) = P(t) / Q(t),   P(t) = sum_{j=0}^m a_j t^j,
                               Q(t) = 1 + sum_{k=1}^n |b_k| t^k

    Order (m, n) = (5, 4) is the paper's own default -- a degree-5 numerator
    over a degree-4 denominator, 10 learnable shape parameters per
    shared/per-neuron copy -- the numerator has m+1 = 6 coefficients, not m
    (vs FourierActivation K=2's 5, AconC's 3).

    NOT THE "SAFE PAU". The absolute value here is taken per coefficient,
    1 + sum_k |b_k| t^k, not around the whole sum, 1 + |sum_k b_k t^k|. Only
    the latter is >= 1 for every real t and therefore provably pole-free;
    this one is not, because the ODD powers are negative for t < 0:

      * Q dips below 1 for t < 0. On the ImageNet-1K pau_global seed-1 curve
        the minimum is Q = 0.855 at t = -0.427.
      * Q can reach 0. Starting from the GELU fit below and moving b_3 alone
        to 0.1 puts a real root at t = -3.19; at b_3 = 0.2 it is at t = -2.09,
        where |f| exceeds 8e4 and changes sign.

    The runs in this repository stayed clear of a pole -- the fit leaves
    b_1 = b_3 = 0, so Q starts even in t, and the trained curve keeps
    min Q = 0.855 -- but nothing in the parameterisation enforces that, and
    the b_k are ordinary nn.Parameters. Switching to 1 + |sum_k b_k t^k|
    would restore the guarantee at the cost of changing the baseline, so it
    is left alone and documented rather than silently altered; cuda_pau/
    implements the same 1 + sum_k |b_k| t^k, so the two paths agree.

    Set init="true" (default) to start from the least-squares Pade fit to
    `ref` on [-L, L] (see true_pade_coeffs) -- the same "start as a faithful
    copy of a known-good activation" courtesy FourierActivation's init="true"
    and AconC's Swish-start init extend to their own families. Note the fit
    itself does not enforce Q(t) > 0 (it solves for signed b_k, see
    true_pade_coeffs' docstring); forward() takes |b_k| regardless, so the
    realised initial curve is the fit's P(t) over that rectified Q, a close
    but not exact reproduction of the unconstrained fit. Set init="random" for
    small-random coefficients instead (the ablation that isolates the value
    of that fit, exactly as for FourierActivation).

    shared=True ties ONE {a, b} pair across all `width` neurons (param shape
    (1, m+1)/(1, n) instead of (width, m+1)/(width, n)) -- the same
    "_global" sharing convention as fact_kK_global and AconC(shared=True),
    so all three "one global learnable shape" activations are directly
    comparable.
    """

    def __init__(self, width, m=5, n=4, shared=False, init="true", ref="gelu",
                 init_scale=1.0, L=math.pi):
        super().__init__()
        self.width, self.m, self.n, self.shared = width, m, n, shared
        param_dim = 1 if shared else width
        if init == "true":
            a_true, b_true = true_pade_coeffs(m, n, ref=ref, L=L)
            self.a = nn.Parameter(torch.from_numpy(a_true).repeat(param_dim, 1).clone())
            self.b = nn.Parameter(torch.from_numpy(b_true).repeat(param_dim, 1).clone())
        elif init == "random":
            std = 0.1 * init_scale
            self.a = nn.Parameter(torch.randn(param_dim, m + 1) * std)
            self.b = nn.Parameter(torch.randn(param_dim, n) * std)
        else:
            raise ValueError(init)
        self.register_buffer("jvec", torch.arange(0, m + 1).float())
        self.register_buffer("kvec", torch.arange(1, n + 1).float())

    def forward(self, t):
        # t: (B, width) -> powers (B, width, m+1) / (B, width, n), broadcast
        # against self.a/self.b of shape (1 or width, m+1)/(1 or width, n) --
        # exactly the same broadcasting convention as FourierActivation's
        # (cos, sin) angle tensors against its (1 or M, K) coefficients.
        tp = t.unsqueeze(-1) ** self.jvec
        tq = t.unsqueeze(-1) ** self.kvec
        P = (tp * self.a).sum(-1)
        Q = 1.0 + (tq * self.b.abs()).sum(-1)
        return P / Q

    def extra_repr(self):
        return f"width={self.width}, m={self.m}, n={self.n}, shared={self.shared}"


# --------------------------------------------------------------------------- #
#  Learnable Fourier-series activation (per neuron)
# --------------------------------------------------------------------------- #
class FourierActivation(nn.Module):
    """phi(t) = a0 + sum_{k=1..K} a_k cos(k w t) + b_k sin(k w t), per neuron.

    Parameters (learnable): a0 (M,), a (M,K), b (M,K).  They are initialised
    from the true Fourier coefficients of the reference activation, broadcast
    across all M neurons, so every neuron starts as a faithful copy of (say)
    GELU and is then free to specialise during training.

    Set init="random" to start from small random coefficients instead -- this
    is the ablation that isolates the value of true-coefficient initialisation.
    init_scale multiplies the base std (0.1) of that random init -- e.g.
    init_scale=5 gives std=0.5, putting the initial random amplitude in the
    same ballpark as the true GELU coefficients' magnitude (~0.7-1.0 for
    a0/a1/b1 at K=2) instead of an order of magnitude smaller. Has no effect
    when init="true".

    Set init="variance_preserving" to use the closed-form init from
    "Variance-preserving coefficient initialization of Fourier activation",
    matching torchortho.FourierActivation's default init exactly
    (github.com/K-H-Ismail/torchortho): every harmonic's cos/sin amplitude
    starts at (1/sqrt(I_0(2)))/sqrt(2) ~= 0.4685 (I_0 = modified Bessel
    function of the first kind, order 0; independent of the reference
    activation `ref`), and the DC term is a0 = (1/sqrt(I_0(2))) *
    sqrt(1 - (1/K!)^2), so the series starts at a fixed, K-dependent
    variance rather than either GELU's true-fit shape ("true") or a small
    random draw ("random"). Unlike "true", this ignores `ref` entirely (a0/a/b
    are a pure function of K); unlike "random"/"random_learnable_amp", it is a
    fixed (non-random) starting point, so a single seed already reproduces it
    exactly. This matches torchortho's init *values* under our
    fixed-integer-harmonic parameterisation, not its structure -- torchortho
    itself uses a learnable per-harmonic magnitude/phase and a learnable
    harmonic-frequency ("grid") parameter, which this init does not add.

    Set init="random_learnable_amp" to factor the coefficients as
    a = amp * a_dir, b = amp * b_dir, where a_dir/b_dir are randn(K) unit-
    scale directions and amp is a SEPARATE learnable scalar (shared by a and
    b), initialised to 0.1 * init_scale. Unlike plain "random" -- where the
    initial std is a fixed, one-time hyperparameter and every coefficient's
    magnitude has to shrink/grow independently and in sync via its own
    gradient -- this gives the optimizer one direct scalar knob for overall
    amplitude, decoupled from the K-dimensional shape. self.a/self.b are
    still exposed as read-only properties (amp * a_dir / amp * b_dir) so
    forward() and every coefficient-logging call site are unaffected.

    Set shared=True to tie ONE set of coefficients {a0, a, b} across all M
    neurons in the layer (param_dim=1 instead of M) -- every neuron applies
    the exact same learned Fourier-series activation, instead of each neuron
    specialising independently. Relies on ordinary broadcasting in forward():
    self.a/self.b of shape (1, K) broadcast against the (B, M, K) angle
    tensor exactly like (M, K) would, just tied across the M axis instead of
    varying over it, so forward() needs no shared-specific branch.

    Set route=True (requires shared=True) to additionally give each of the
    `num_features` neurons its own learnable ROUTING gate over the K
    harmonics: route_logits (num_features, K), passed through a softmax over
    K, so every neuron's gate is a genuine distribution (sums to 1) over
    "how much of harmonic k to use". The harmonics' own amplitudes {a0, a_k,
    b_k} stay a SINGLE global set (param_dim=1, exactly like plain
    shared=True) -- routing only decides each neuron's individual MIX of
    those shared harmonics, so neurons can specialise without ever getting
    their own copy of the coefficients themselves. route_init_std scales the
    randn init of route_logits (softmax of near-zero logits starts close to
    uniform routing, i.e. every neuron initially mixes all K harmonics
    roughly equally, then specialises during training). forward() computes
    phi_i(t) = a0 + sum_k route_i_k * (a_k cos(k w t) + b_k sin(k w t)).

    Set phase=True (requires shared=True) to instead give every neuron a
    small, richer additive signal rather than a re-weighting of the shared
    harmonics: a phase-shifted unit sinusoid at the fundamental frequency,

        phi_i(t) = [a0 + sum_k a_k cos(k w t) + b_k sin(k w t)] + eps * cos(w t + phi_i)

    where phi_i (num_features,), one learnable angle per neuron, is
    initialised uniformly in [0, 2*pi), and eps is a SINGLE learnable scalar
    shared by every neuron (phase_eps_init, default 0.05, keeps the term a
    small perturbation at init rather than swamping the shared curve). Unlike
    route=True -- whose K-way softmax mix can only ever reproduce a convex
    combination of the K shared harmonics, i.e. stays inside their span --
    this term is a genuinely new per-neuron oscillatory degree of freedom
    layered on top of the still-literally-global {a0, a_k, b_k}, orthogonal
    to (and combinable with) route. self.phi/self.phase_eps are read
    directly by forward(); no gate/anneal interaction (anneal_harmonics only
    touches self.a/self.b's own K harmonics, not this separate term).

    Set factorial_scale=True to change the series' functional form itself
    (not just its init): every harmonic k's (cos, sin) terms are additionally
    divided by k!,

        phi(t) = a0 + sum_{k=1..K} [a_k cos(k w t) + b_k sin(k w t)] / k!

    instead of the plain (unscaled) sum every other init/variant here uses.
    This is independent of `init` -- e.g. combine with init="variance_preserving"
    to get both the paper's closed-form starting coefficients AND its
    factorial-decayed series form, or with init="true"/"random" to apply just
    the functional-form change on its own. Composes with anneal_harmonics
    (both scale a/b by a K-vector; the two multiply together) and route
    (routing weights apply to the same, now-factorial-scaled, a/b). Reflected
    in effective_a()/effective_b() too, so coefficient-logging call sites see
    the true per-harmonic contribution including the 1/k! factor.

    Set anneal_harmonics=True to multiply harmonic k's (cos, sin) terms by a
    gate g_k(p) = exp(-anneal_rate * (k-1) * p), where p in [0,1] is training
    progress (set externally via set_progress(), e.g. once per optimizer
    step -- see cv_train.train_one). g_1(p)=1 always (the fundamental never
    decays); higher k decays faster (rate scales with k-1), so as training
    proceeds the effective series smoothly narrows from the full K harmonics
    down toward just the lowest few, without ever changing the underlying
    a_k/b_k parameters themselves (they keep training throughout -- only
    their CONTRIBUTION to the forward pass is gated away). Motivated by:
    richer K (e.g. 11) gives a much better initial global fit to the
    reference activation than a low truncation, but this study found
    low-order (K~1-2) truncations easier for the rank-1 neuron-tied FFN to
    optimize -- annealing lets training start from the rich, accurate curve
    and settle toward the empirically more stable low-order one. p=0 (start
    of training) always gives all gates=1 (full K harmonics active,
    identical to anneal_harmonics=False). Has no effect on self.a/self.b
    themselves, only on forward()'s use of them, so coefficient-logging call
    sites (which read self.a/self.b directly) still see the raw, un-gated
    values -- use effective_a()/effective_b() to log the gated values instead.
    """

    def __init__(self, num_features, K=8, ref="gelu", init="true", w=1.0,
                shared=False, init_scale=1.0, anneal_harmonics=False,
                anneal_rate=0.5, route=False, route_init_std=0.5,
                phase=False, phase_eps_init=0.05, factorial_scale=False):
        super().__init__()
        self.M, self.K, self.w, self.shared = num_features, K, w, shared
        self.learnable_amp = (init == "random_learnable_amp")
        self.factorial_scale = factorial_scale
        self.anneal_harmonics = anneal_harmonics
        self.anneal_rate = anneal_rate
        self.route = route
        if route:
            assert shared, (
                "FourierActivation(route=True) requires shared=True -- "
                "routing distributes a SINGLE global set of harmonics "
                "across neurons via a per-neuron gate; it has no meaning "
                "against already-per-neuron (shared=False) coefficients.")
            self.route_logits = nn.Parameter(
                torch.randn(num_features, K) * route_init_std)
        self.phase = phase
        if phase:
            assert shared, (
                "FourierActivation(phase=True) requires shared=True -- the "
                "per-neuron phase-shifted sinusoid is meant to add "
                "neuron-specific diversity ON TOP OF an otherwise literally "
                "global series; it has no meaning against already-per-neuron "
                "(shared=False) coefficients.")
            self.phi = nn.Parameter(torch.rand(num_features) * (2.0 * math.pi))
            self.phase_eps = nn.Parameter(torch.tensor(float(phase_eps_init)))
        param_dim = 1 if shared else num_features
        a0, a, b, w_true = true_fourier_coeffs(K, ref=ref)
        self.w = w_true
        if init == "true":
            self.a0 = nn.Parameter(torch.full((param_dim,), float(a0)))
            self._a = nn.Parameter(torch.from_numpy(a).repeat(param_dim, 1).clone())
            self._b = nn.Parameter(torch.from_numpy(b).repeat(param_dim, 1).clone())
        elif init == "random":
            std = 0.1 * init_scale
            self.a0 = nn.Parameter(torch.zeros(param_dim))
            self._a = nn.Parameter(torch.randn(param_dim, K) * std)
            self._b = nn.Parameter(torch.randn(param_dim, K) * std)
        elif init == "random_learnable_amp":
            self.a0 = nn.Parameter(torch.zeros(param_dim))
            self.a_dir = nn.Parameter(torch.randn(param_dim, K))
            self.b_dir = nn.Parameter(torch.randn(param_dim, K))
            self.amp = nn.Parameter(torch.tensor(0.1 * init_scale))
        elif init == "variance_preserving":
            # Variance-preserving init, matching torchortho.FourierActivation's
            # default init exactly (github.com/K-H-Ismail/torchortho,
            # fourier_activation.py): every harmonic's magnitude is
            # 1/sqrt(I_0(2)) (I_0 = modified Bessel function of the first
            # kind, order 0; I_0(2) ~= 2.2796 -> scale ~= 0.66233), split
            # evenly between its cos/sin (a_k, b_k) components per that
            # reference's phase=pi/4 init convention (magnitude*cos(pi/4) ==
            # magnitude*sin(pi/4) == magnitude/sqrt(2)); a0 is the same
            # magnitude scaled by sqrt(1-(1/K!)^2) so the DC term preserves
            # total series variance. NOTE: this reproduces torchortho's
            # *values* under our fixed-integer-harmonic (a_k, b_k) parameterisation;
            # torchortho itself instead parameterises each harmonic as a
            # learnable magnitude/phase pair with a learnable frequency
            # ("grid"), which this init does not replicate structurally.
            from scipy.special import iv
            bessel_scale = 1.0 / math.sqrt(iv(0, 2.0))   # == 0.66232645879
            a0_vp = bessel_scale * math.sqrt(1.0 - (1.0 / math.factorial(K)) ** 2)
            ak_vp = bessel_scale / math.sqrt(2.0)
            self.a0 = nn.Parameter(torch.full((param_dim,), float(a0_vp)))
            self._a = nn.Parameter(torch.full((param_dim, K), float(ak_vp)))
            self._b = nn.Parameter(torch.full((param_dim, K), float(ak_vp)))
        else:
            raise ValueError(init)
        self.register_buffer("kvec", torch.arange(1, K + 1).float())
        if factorial_scale:
            kfact_inv = torch.tensor([1.0 / math.factorial(k) for k in range(1, K + 1)],
                                      dtype=torch.float32)
            self.register_buffer("kfact_inv", kfact_inv)
        self.register_buffer("progress", torch.tensor(0.0))

    @property
    def a(self):
        return self.amp * self.a_dir if self.learnable_amp else self._a

    @property
    def b(self):
        return self.amp * self.b_dir if self.learnable_amp else self._b

    def set_progress(self, p):
        self.progress.fill_(float(p))

    def harmonic_gate(self):
        if not self.anneal_harmonics:
            return None
        rate = self.anneal_rate * (self.kvec - 1.0)   # 0 for k=1, grows with k
        return torch.exp(-rate * self.progress)        # (K,), gate_1==1 always

    def effective_a(self):
        gate = self.harmonic_gate()
        a = self.a if gate is None else self.a * gate
        return a * self.kfact_inv if self.factorial_scale else a

    def effective_b(self):
        gate = self.harmonic_gate()
        b = self.b if gate is None else self.b * gate
        return b * self.kfact_inv if self.factorial_scale else b

    def route_weights(self):
        """(num_features, K) softmax-normalised per-neuron routing gate --
        only valid when route=True."""
        return torch.softmax(self.route_logits, dim=-1)

    def forward(self, t):
        # t: (B, M) ; build (B, M, K) angles = k * w * t
        ang = t.unsqueeze(-1) * (self.kvec * self.w)          # (B, M, K)
        cos, sin = torch.cos(ang), torch.sin(ang)
        gate = self.harmonic_gate()
        a, b = (self.a, self.b) if gate is None else (self.a * gate, self.b * gate)
        if self.factorial_scale:
            # phi(t) = a0 + sum_k [a_k cos(k w t) + b_k sin(k w t)] / k!
            a, b = a * self.kfact_inv, b * self.kfact_inv
        # a, b: (1, K) (shared) or (M, K) broadcast over batch
        if self.route:
            r = self.route_weights()                          # (M, K)
            out = (cos * a * r).sum(-1) + (sin * b * r).sum(-1) + self.a0
        else:
            out = (cos * a).sum(-1) + (sin * b).sum(-1) + self.a0
        if self.phase:
            # eps * cos(w*t + phi_i) -- t: (B, M), self.phi: (M,) broadcasts
            # over the batch axis; fundamental frequency (w), not k*w, since
            # this is a standalone per-neuron signal, not another harmonic.
            out = out + self.phase_eps * torch.cos(t * self.w + self.phi)
        return out


# --------------------------------------------------------------------------- #
#  Learnable DCT-series (cosine-only) activation (per neuron)
# --------------------------------------------------------------------------- #
class DCTActivation(nn.Module):
    """phi(t) = a0 + sum_{k=1..K} a_k cos(k w t), per neuron -- the same
    per-neuron truncated-series activation as FourierActivation, but with
    the sine terms dropped: a pure cosine series, the continuous analogue of
    a K-term DCT reconstruction (DCT-II is the discrete/sampled cosine
    transform; this is its per-neuron, continuously-parameterised
    counterpart). Coefficients are the TRUE cosine-series coefficients of
    the reference activation -- reusing `true_fourier_coeffs` and discarding
    its sine half, since that function already computes a0/a_k via the same
    integral a cosine-only (DCT-style) series would use -- broadcast across
    every neuron and then free to specialise during training.
    """

    def __init__(self, num_features, K=5, ref="gelu", init="true", w=1.0):
        super().__init__()
        self.M, self.K, self.w = num_features, K, w
        a0, a, _b, w_true = true_fourier_coeffs(K, ref=ref)
        self.w = w_true
        if init == "true":
            self.a0 = nn.Parameter(torch.full((num_features,), float(a0)))
            self.a = nn.Parameter(torch.from_numpy(a).repeat(num_features, 1).clone())
        elif init == "random":
            std = 0.1
            self.a0 = nn.Parameter(torch.zeros(num_features))
            self.a = nn.Parameter(torch.randn(num_features, K) * std)
        else:
            raise ValueError(init)
        self.register_buffer("kvec", torch.arange(1, K + 1).float())

    def forward(self, t):
        ang = t.unsqueeze(-1) * (self.kvec * self.w)          # (B, M, K)
        cos = torch.cos(ang)
        out = (cos * self.a).sum(-1) + self.a0
        return out
