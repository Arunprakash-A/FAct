"""The learned FAct K=2 activation, frozen into a FIXED activation function.

This study's premise: the imagenet1k_100ep run of `fact_k2_global` trained ONE
Fourier-series activation

    phi(t) = a0 + a1*cos(w t) + b1*sin(w t) + a2*cos(2w t) + b2*sin(2w t)

end-to-end alongside the network, shared across every neuron and every block,
starting from the true GELU Fourier fit. It converged to a specific curve. That
curve -- not the mechanism that learned it -- is what gets tested here, as a
plain drop-in nonlinearity with NO learnable parameters, against ReLU, GELU,
SiLU and the rest of the standard fixed-activation menu.

So the coefficients live in `register_buffer`, not `nn.Parameter`:
  * every variant in this study has EXACTLY zero activation parameters, so the
    parameter counts across all 11 activations are identical to the byte and
    the comparison carries no capacity confound;
  * the optimizer never sees them, so weight decay can't drift them;
  * `.state_dict()` still carries them, so a checkpoint is self-describing.

Coefficients come from learned_fact_k2_coeffs.json, extracted from the
source-network ImageNet-1k checkpoint (seed 1, 100 epochs, top-1 0.6512 --
the same checkpoint the frozen-activation ablation freezes to). That JSON
also records what every completed seed of that run
converged to; seeds 1-5 agree to ~3 decimal places on all five coefficients, so
"the learned FAct activation" is a reproducible curve rather than one seed's
accident.

CUDA path (the default, per the study instruction to use the custom kernel):
`cuda_fixed_fact_k2/fixed_fact_k2_kernel.cu`, a fork of the trainable
`cuda_fact_k2` kernel that adds a `backward_input` entry point. The parent
kernel's backward always also reduces the {a0, a, b} gradients -- tens of
millions of summands into five scalars -- which for a frozen activation is
computed and immediately discarded by autograd. The fork skips it. Forward
numerics are bit-identical to the parent kernel (same kernel body); see
test_fact_fixed.py, which checks both against the pure-PyTorch reference.
"""
import json
import os

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
COEFF_JSON = os.path.join(_THIS_DIR, "learned_fact_k2_coeffs.json")

_ext = None


def _load_ext():
    """JIT-compile (once per machine; cached under torch's extensions dir) the
    forked kernel. Deliberately torch.utils.cpp_extension.load and NOT
    setup.py build_ext, for the same reason
    ../source_vit/cuda_fact_k2/fact_k2_module.py gives: nvcc and the torch
    build commonly disagree on CUDA version, and build_ext's
    _check_cuda_version hard-fails on that even though the toolchain
    compiles and runs fine."""
    global _ext
    if _ext is None:
        from torch.utils.cpp_extension import load
        _ext = load(
            name="fixed_fact_k2_cuda",
            sources=[os.path.join(_THIS_DIR, "cuda_fixed_fact_k2",
                                  "fixed_fact_k2_kernel.cu")],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _ext


def load_coeffs(path=COEFF_JSON):
    with open(path) as f:
        d = json.load(f)
    assert d["K"] == 2, f"this module is K=2 only, got K={d['K']}"
    return d


class _FixedFactK2Fn(torch.autograd.Function):
    """Forward through the CUDA kernel; backward through its frozen-coefficient
    `backward_input` path (grad wrt the input only -- there is no coefficient
    gradient to return, so none is computed)."""

    @staticmethod
    def forward(ctx, t, a0, a, b, w):
        ext = _load_ext()
        t_c = t.contiguous()
        a0_c, a_c, b_c = a0.contiguous(), a.contiguous(), b.contiguous()
        out = ext.forward(t_c, a0_c, a_c, b_c, w)
        ctx.save_for_backward(t_c, a_c, b_c)
        ctx.w = w
        return out

    @staticmethod
    def backward(ctx, grad_output):
        t, a, b = ctx.saved_tensors
        grad_input = _load_ext().backward_input(
            grad_output.contiguous(), t, a, b, ctx.w)
        return grad_input, None, None, None, None


class FixedFAct(nn.Module):
    """phi(t) with the coefficients frozen at the learned ImageNet-1k values.

    use_cuda_kernel=True (default) routes through the forked CUDA kernel and
    requires a CUDA tensor. use_cuda_kernel=False evaluates the same series in
    plain PyTorch -- kept as the correctness reference (test_fact_fixed.py) and
    as the only path that works on CPU, never as a silent fallback: a run that
    asked for the kernel and quietly got pure PyTorch would be a different
    wall-clock measurement wearing the same variant name.
    """

    K = 2

    def __init__(self, coeff_path=COEFF_JSON, use_cuda_kernel=True):
        super().__init__()
        d = load_coeffs(coeff_path)
        self.use_cuda_kernel = use_cuda_kernel
        self.w = float(d["w"])
        self.coeff_source = d.get("source_checkpoint", coeff_path)
        # (1,) / (1, K): the "shared" layout the kernel calls P=1 -- one set of
        # coefficients broadcast over every feature, matching how
        # fact_k2_global trained it (tied across neurons AND across depth).
        self.register_buffer("a0", torch.tensor([d["a0"]], dtype=torch.float32))
        self.register_buffer("a", torch.tensor([d["a"]], dtype=torch.float32))
        self.register_buffer("b", torch.tensor([d["b"]], dtype=torch.float32))
        # (K,) frequency multipliers for the pure-PyTorch path.
        self.register_buffer("kvec", torch.arange(1, self.K + 1, dtype=torch.float32))

    def _torch_forward(self, t):
        ang = t.unsqueeze(-1) * (self.kvec * self.w)
        return (self.a0
                + (ang.cos() * self.a).sum(-1)
                + (ang.sin() * self.b).sum(-1))

    def forward(self, t):
        if not self.use_cuda_kernel:
            return self._torch_forward(t)
        assert t.is_cuda, (
            "FixedFAct(use_cuda_kernel=True) needs a CUDA tensor; pass "
            "use_cuda_kernel=False for the pure-PyTorch reference path")
        if t.dtype in (torch.float16, torch.bfloat16):
            # The kernel dispatches over AT_DISPATCH_FLOATING_TYPES (float/
            # double only), and this autograd.Function carries no autocast
            # policy, so under AMP an unguarded call would hard-crash on a
            # half input. Upcast for the call and cast back -- exactly what
            # cuda_fact_k2/fact_k2_module.py does, and what the pure-PyTorch
            # FourierActivation does implicitly by promoting against its fp32
            # kvec buffer.
            with torch.autocast(device_type=t.device.type, enabled=False):
                out = _FixedFactK2Fn.apply(t.float(), self.a0, self.a, self.b, self.w)
            return out.to(t.dtype)
        return _FixedFactK2Fn.apply(t, self.a0, self.a, self.b, self.w)

    def extra_repr(self):
        return (f"K={self.K}, w={self.w:.6f}, frozen=True, "
                f"cuda_kernel={self.use_cuda_kernel}, "
                f"a0={self.a0.item():.6f}, "
                f"a={[round(v, 6) for v in self.a.flatten().tolist()]}, "
                f"b={[round(v, 6) for v in self.b.flatten().tolist()]}")
