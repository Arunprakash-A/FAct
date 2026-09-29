"""Drop-in CUDA-accelerated replacement for fourier_layers.FourierActivation
at K=2 -- the truncation the paper's source-network runs use, and the one the
transferred coefficients in `fact/coefficients/` were learned at.

phi(t) = a0 + a1*cos(w*t) + b1*sin(w*t) + a2*cos(2*w*t) + b2*sin(2*w*t)

The extension is JIT-compiled on first import via torch.utils.cpp_extension.load
(NOT setup.py's build_ext -- nvcc and the torch build commonly disagree on CUDA
version, and build_ext's _check_cuda_version hard-fails on that mismatch even
when the toolchain itself compiles and runs fine; load() skips that check). Compiled output is cached under torch's extensions dir, so
only the first import per machine pays the compile cost.

Usage (mirrors FourierActivation's constructor for the subset it supports --
init in {"true", "random"}, no route/phase/anneal/factorial_scale):

    from fact_k2_module import FourierActivationK2CUDA
    act = FourierActivationK2CUDA(d_ff, ref="gelu", init="true", shared=True).cuda()
    y = act(x)   # x: (..., d_ff) CUDA float32/float64 tensor

`shared=True` matches every "_global" variant (one set of 5 coefficients for
the whole network); `shared=False` matches per-neuron coefficients (P=M).
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_layers import true_fourier_coeffs  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
fact_k2_cuda = load(
    name="fact_k2_cuda",
    sources=[os.path.join(_THIS_DIR, "fact_k2_cuda_kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class _FactK2Fn(torch.autograd.Function):
    """Autograd hookup for the K=2 kernel.

    `w` may be a plain Python float (every fixed-w variant) OR a 0-dim
    Parameter (the `_lw` learnable-frequency variants). Only the tensor case
    asks the kernel for a d(phi)/dw reduction, and the kernel templates that
    branch away at compile time, so a fixed-w call costs exactly what it did
    before this was added.

    Nothing but `t` and the five coefficients is stashed for backward: the
    kernel recomputes sin/cos from `t` inside the backward pass rather than
    keeping the (B, M, K) cos/sin intermediates autograd would otherwise save.
    That recompute is the whole point of using the kernel here -- at WMT14
    shapes those intermediates are what makes the pure-PyTorch path cost
    ~5 GiB more than GELU.
    """

    @staticmethod
    def forward(ctx, t, a0, a, b, w):
        t_c = t.contiguous()
        a0_c, a_c, b_c = a0.contiguous(), a.contiguous(), b.contiguous()
        w_val = float(w)
        out = fact_k2_cuda.forward(t_c, a0_c, a_c, b_c, w_val)
        ctx.save_for_backward(t_c, a0_c, a_c, b_c)
        ctx.w = w_val
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t, a0, a, b = ctx.saved_tensors
        need_grad_w = ctx.needs_input_grad[4]
        grad_input, grad_a0, grad_a, grad_b, grad_w = fact_k2_cuda.backward(
            grad_output.contiguous(), t, a0, a, b, ctx.w, need_grad_w)
        return (grad_input, grad_a0, grad_a, grad_b,
                grad_w if need_grad_w else None)


class FourierActivationK2CUDA(nn.Module):
    """CUDA-kernel FourierActivation, K fixed at 2. See module docstring."""

    K = 2

    def __init__(self, num_features, ref="gelu", init="true", w=None,
                 shared=False, init_scale=1.0, learnable_w=False, w_init=None):
        super().__init__()
        assert init in ("true", "random"), (
            f"FourierActivationK2CUDA supports init in "
            f"('true', 'random'), got {init!r}")
        self.M = num_features
        self.shared = shared
        self.learnable_w = learnable_w
        param_dim = 1 if shared else num_features
        a0, a, b, w_true = true_fourier_coeffs(self.K, ref=ref)
        # Mirrors fourier_layers.FourierActivation's learnable_w/w_init
        # contract exactly, so the two implementations stay swappable: w_init
        # only means something with learnable_w=True, and leaving it None
        # starts at w_true so the arm is identical to the fixed-w arm at
        # step 0. See that class for why a non-default w_init gives up the
        # "GELU at init" property.
        assert w_init is None or learnable_w, (
            "w_init only means something with learnable_w=True")
        assert not (learnable_w and not shared), (
            "learnable_w requires shared=True -- w is ONE scalar for the whole "
            "network, and the kernel's grad-wrt-w reduction is only defined "
            "for the shared (P == 1) parameterisation")
        w_start = float(w_true if w is None else w)
        self.w_init = float(w_start if w_init is None else w_init)
        if learnable_w:
            self.w = nn.Parameter(torch.tensor(self.w_init))
        else:
            self.w = self.w_init
        if init == "true":
            self.a0 = nn.Parameter(torch.full((param_dim,), float(a0)))
            self._a = nn.Parameter(torch.from_numpy(a).repeat(param_dim, 1).clone())
            self._b = nn.Parameter(torch.from_numpy(b).repeat(param_dim, 1).clone())
        else:
            std = 0.1 * init_scale
            self.a0 = nn.Parameter(torch.zeros(param_dim))
            self._a = nn.Parameter(torch.randn(param_dim, self.K) * std)
            self._b = nn.Parameter(torch.randn(param_dim, self.K) * std)

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def forward(self, t):
        assert t.is_cuda, "FourierActivationK2CUDA requires a CUDA input tensor"
        assert t.size(-1) == self.M or self.shared, (
            f"input last dim ({t.size(-1)}) must equal num_features ({self.M})")
        if t.dtype in (torch.float16, torch.bfloat16):
            # The kernel has no half/bfloat16 forward (fact_k2_cuda_kernel.cu
            # only dispatches scalar_t in {half, bfloat16, float, double} for
            # the OUTPUT copy, but accumulates/launches trig math in acc_t --
            # actually it's the CUDA op itself that has no registered half
            # kernel here) and this autograd.Function carries no autocast
            # policy, so a caller training under AMP (e.g. cv_train.py) would
            # otherwise hand it a fp16 `t` and hard-crash. Upcast for the
            # call and cast back, matching what the pure-PyTorch
            # FourierActivation already does under autocast (its `mul`
            # against the fp32 kvec buffer promotes a half input to fp32
            # before any trig runs).
            with torch.autocast(device_type=t.device.type, enabled=False):
                out = _FactK2Fn.apply(t.float(), self.a0, self._a, self._b, self.w)
            return out.to(t.dtype)
        return _FactK2Fn.apply(t, self.a0, self._a, self._b, self.w)

    def extra_repr(self):
        return (f"num_features={self.M}, K={self.K}, w={float(self.w):.6f}, "
                f"shared={self.shared}, learnable_w={self.learnable_w}")


class PerTokenFourierActivationK2CUDA(nn.Module):
    """CUDA-kernel version of fourier_layers.PerTokenFourierActivation
    (K=2 only): one independent curve per SEQUENCE POSITION instead of per
    feature -- same fact_k2_cuda extension and _FactK2Fn autograd.Function
    as FourierActivationK2CUDA's shared=False ("per-feature", P=M) path
    above, just transposed so POSITION is what the kernel treats as the
    axis to index coefficients by, instead of the channel axis.

    fact_k2_cuda_kernel.cu's backward recomputes sin/cos from the saved
    input tensor inside the kernel -- ctx.save_for_backward keeps only
    (t, a0, a, b), never the (B, T, M, K) angle/cos/sin intermediates the
    pure-PyTorch fourier_layers.PerTokenFourierActivation's autograd graph
    would otherwise keep alive between forward and backward. That recompute-
    not-store backward is the entire reason this class exists: at WMT14
    scale (B ~ 300-400, T <= 128, M = d_ff = 2048) those intermediates are
    hundreds of millions of elements each, several of them simultaneously.

    forward(t) expects t of shape (B, T, M) with T <= max_positions -- the
    kernel itself only knows "index by the last dim", so this class
    transposes T to the end before the call and back after. Unlike
    FourierActivationK2CUDA(shared=False), which requires the input's last
    dim to equal num_features EXACTLY, this supports T < max_positions (a
    torch.autograd-friendly slice of the (max_positions,)/(max_positions,2)
    parameters down to the first T rows) -- needed for both training (T is
    whatever the batch's target length is) and autoregressive decoding
    (T grows by one every step, starting from 1).
    """

    K = 2

    def __init__(self, max_positions, ref="gelu", init="true", w=None):
        super().__init__()
        assert init == "true", (
            f"PerTokenFourierActivationK2CUDA supports init='true' only, got {init!r}")
        self.max_positions = max_positions
        a0, a, b, w_true = true_fourier_coeffs(self.K, ref=ref)
        self.w = float(w_true if w is None else w)
        self.a0 = nn.Parameter(torch.full((max_positions,), float(a0)))
        self._a = nn.Parameter(torch.from_numpy(a).repeat(max_positions, 1).clone())
        self._b = nn.Parameter(torch.from_numpy(b).repeat(max_positions, 1).clone())

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    def forward(self, t):
        assert t.is_cuda, "PerTokenFourierActivationK2CUDA requires a CUDA input tensor"
        T = t.size(1)
        assert T <= self.max_positions, (
            f"PerTokenFourierActivationK2CUDA built for max_positions="
            f"{self.max_positions}, got a sequence of length {T}")
        t_swapped = t.transpose(1, 2)              # (B, M, T) -- position now last
        a0_T, a_T, b_T = self.a0[:T], self._a[:T], self._b[:T]
        if t.dtype in (torch.float16, torch.bfloat16):
            # see FourierActivationK2CUDA.forward's identical guard: the
            # kernel only dispatches float/double, and this autograd.Function
            # carries no autocast policy.
            with torch.autocast(device_type=t.device.type, enabled=False):
                out = _FactK2Fn.apply(t_swapped.float().contiguous(), a0_T, a_T, b_T, self.w)
            out = out.to(t.dtype)
        else:
            out = _FactK2Fn.apply(t_swapped.contiguous(), a0_T, a_T, b_T, self.w)
        return out.transpose(1, 2)                 # back to (B, T, M)

    def coeffs(self):
        """Snapshot for logging -- identical shape/keys to
        fourier_layers.PerTokenFourierActivation.coeffs() (mean/std across
        positions plus a full per_position table), so both implementations
        feed the same downstream plotting/analysis code unchanged."""
        a0 = self.a0.detach().reshape(-1).cpu().numpy()
        a = self._a.detach().cpu().numpy()
        b = self._b.detach().cpu().numpy()
        out = {"w": self.w, "K": self.K, "a0": float(a0.mean()), "a0_std": float(a0.std())}
        for i in range(self.K):
            out[f"a{i+1}"] = float(a[:, i].mean())
            out[f"a{i+1}_std"] = float(a[:, i].std())
            out[f"b{i+1}"] = float(b[:, i].mean())
            out[f"b{i+1}_std"] = float(b[:, i].std())
        out["per_position"] = [
            [float(a0[t])] + [float(a[t, k]) for k in range(self.K)]
                            + [float(b[t, k]) for k in range(self.K)]
            for t in range(self.max_positions)
        ]
        return out

    def extra_repr(self):
        c = self.coeffs()
        return (f"K={self.K}, max_positions={self.max_positions}, w={self.w:.6f}, "
                f"a0={c['a0']:.6f}+/-{c['a0_std']:.6f} (mean+/-std over positions), "
                f"cuda_kernel=True")
