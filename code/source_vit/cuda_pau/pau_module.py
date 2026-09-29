"""Drop-in CUDA-accelerated replacement for fourier_layers.PAU -- direct
analogue of cuda_fact_k2/fact_k2_module.py for the Pade Activation Unit, see
pau_cuda_kernel.cu for the kernel-level design/gradient derivation.

f(t) = P(t)/Q(t),  P(t) = sum_{j=0..5} a_j t^j,  Q(t) = 1 + sum_{k=1..4} |b_k| t^k

The extension is JIT-compiled on first import via torch.utils.cpp_extension.load
(same reasoning as cuda_fact_k2/fact_k2_module.py). Compiled output is cached under
torch's extensions dir, so only the first import per machine pays the compile
cost.

Parameters are named `self.a` / `self.b` -- NOT `self._a`/`self._b` behind a
property, unlike FourierActivationK*CUDA -- because fourier_layers.PAU itself
exposes `self.a`/`self.b` directly (no property indirection), and matching
that attribute name is what makes a state_dict trained with the pure-PyTorch
PAU load straight into this module (and vice versa): both register their
learnable tensors under the same key, `shared_act.a`/`shared_act.b`.

Usage (mirrors PAU's constructor for the subset it supports -- order fixed at
the kernel's compiled (m, n) = (5, 4), init in {"true", "random"}):

    from pau_module import PAUCUDA
    act = PAUCUDA(d_ff, ref="gelu", init="true", shared=True).cuda()
    y = act(x)   # x: (..., d_ff) CUDA float32/float64 tensor

`shared=True` matches "pau_global" (ONE {a, b} pair for the whole network,
10 params total); `shared=False` matches per-neuron coefficients (P=M).
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fourier_layers import true_pade_coeffs  # noqa: E402

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
pau_cuda = load(
    name="pau_cuda",
    sources=[os.path.join(_THIS_DIR, "pau_cuda_kernel.cu")],
    extra_cflags=["-O3"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class _PAUFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, t, a, b):
        t_c = t.contiguous()
        a_c, b_c = a.contiguous(), b.contiguous()
        out = pau_cuda.forward(t_c, a_c, b_c)
        ctx.save_for_backward(t_c, a_c, b_c)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t, a, b = ctx.saved_tensors
        grad_input, grad_a, grad_b = pau_cuda.backward(grad_output.contiguous(), t, a, b)
        return grad_input, grad_a, grad_b


class PAUCUDA(nn.Module):
    """CUDA-kernel PAU, order fixed at (m, n) = (5, 4). See module docstring."""

    M_ORDER = 5
    N_ORDER = 4

    def __init__(self, num_features, ref="gelu", init="true", shared=False,
                 init_scale=1.0, L=None):
        super().__init__()
        assert init in ("true", "random"), (
            f"PAUCUDA supports init in ('true', 'random'), got {init!r}")
        import math
        L = math.pi if L is None else L
        self.M = num_features
        self.shared = shared
        param_dim = 1 if shared else num_features
        if init == "true":
            a_true, b_true = true_pade_coeffs(self.M_ORDER, self.N_ORDER, ref=ref, L=L)
            self.a = nn.Parameter(torch.from_numpy(a_true).repeat(param_dim, 1).clone())
            self.b = nn.Parameter(torch.from_numpy(b_true).repeat(param_dim, 1).clone())
        else:
            std = 0.1 * init_scale
            self.a = nn.Parameter(torch.randn(param_dim, self.M_ORDER + 1) * std)
            self.b = nn.Parameter(torch.randn(param_dim, self.N_ORDER) * std)

    def forward(self, t):
        assert t.is_cuda, "PAUCUDA requires a CUDA input tensor"
        assert t.size(-1) == self.M or self.shared, (
            f"input last dim ({t.size(-1)}) must equal num_features ({self.M})")
        if t.dtype in (torch.float16, torch.bfloat16):
            # Same AMP upcast as FourierActivationK2/K9CUDA -- the kernel has
            # no half/bfloat16 dispatch (AT_DISPATCH_FLOATING_TYPES only
            # covers float/double), so match the pure-PyTorch PAU's implicit
            # fp32 promotion under autocast explicitly instead of hard-crashing.
            with torch.autocast(device_type=t.device.type, enabled=False):
                out = _PAUFn.apply(t.float(), self.a, self.b)
            return out.to(t.dtype)
        return _PAUFn.apply(t, self.a, self.b)

    def extra_repr(self):
        return f"num_features={self.M}, m={self.M_ORDER}, n={self.N_ORDER}, shared={self.shared}"
