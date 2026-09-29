"""A 3-segment (up-down-up) piecewise-linear approximation of the learned
FAct-K2 curve, frozen the same way `fact_fixed.py`'s FixedFAct is frozen.

Where the coefficients come from: `fact_fixed.FixedFAct` evaluates the full
Fourier series

    phi(t) = a0 + a1*cos(t) + b1*sin(t) + a2*cos(2t) + b2*sin(2t)

at the imagenet1k_100ep seed1-final coefficients. Restricted to the FFN
pre-activation p99 operating range (|t| <= 1.2112, measured over 464.8M FFN
pre-activation values from 512 ImageNet-1k val images), that curve is exactly
up-down-up: two interior turning points, a local max at t=-0.874 and a local
min at t=+0.186 (its other two turning points, at t=-2.46 and t=+1.91, sit
outside this range). A 3-segment interpolating PWL was fit through that
window with the two interior knots FREE (minimizing weighted RMSE against the
real pre-activation histogram, not pinned to the exact extrema -- doing so
cuts weighted RMSE by 26% because error is dominated by the steep, convex
final climb toward the true peak just past the right edge). The resulting
knots ship next to this module in `three_segment_breakpoints.json`, whose
`optimized` block records the fit and the error it achieves.

    t   = [-1.21125, -1.05679,  0.33076,  1.21125]
    phi = [ 0.19049,  0.27843, -0.09685,  0.96750]
    weighted RMSE in range = 0.0743, max abs error in range = 0.1597

Like FixedFAct, the four knots live in `register_buffer`s, never
`nn.Parameter`s -- zero learnable activation parameters, so it drops into
this study's ACT_KINDS registry under the same equal-capacity guarantee.

No custom CUDA kernel. `fixed_fact_k2_kernel.cu` exists because the PARENT
Fourier kernel's backward always reduces a0/a/b gradients across the whole
tensor -- millions of summands into 5 scalars -- and skipping that reduction
for a frozen curve is the whole win. A piecewise-linear function has no such
reduction to skip: its backward is just `grad_output * slope[segment(t)]`,
already O(N) with no trig and no cross-element reduction, so plain PyTorch
(bucketize + gather, both single fused CUDA kernels via ATen) is not leaving
an obvious kernel-fusion win on the table the way the Fourier path was.
"""
import json
import os

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#: Source of truth for the knots, shipped alongside this module so a stale
#: copy can't drift silently -- edit that file, not the fallback list below.
BREAKPOINTS_JSON = os.path.join(_THIS_DIR, "three_segment_breakpoints.json")

# Fallback, verbatim from three_segment_breakpoints.json's "optimized" block,
# used only if that file is missing (e.g. the module was vendored on its own).
_FALLBACK_T = [-1.2112500000000015, -1.0567904484324053,
               0.33075634635765444, 1.2112500000000015]
_FALLBACK_PHI = [0.19048779940803617, 0.27842990670242457,
                 -0.09684871631617309, 0.9674956611260993]


def load_breakpoints(path=BREAKPOINTS_JSON):
    try:
        with open(path) as f:
            d = json.load(f)["optimized"]
        return d["t"], d["phi"]
    except FileNotFoundError:
        return _FALLBACK_T, _FALLBACK_PHI


class FixedFactPWL3(nn.Module):
    """3-knot (4-point) frozen piecewise-linear activation. Interpolates
    between the knots; linearly EXTRAPOLATES beyond them using the boundary
    segment's own slope (via `torch.bucketize`, which does not clamp), rather
    than saturating -- a small ~100K-param ViT's pre-activation distribution
    on a new dataset/arch is not guaranteed to sit inside the range this was
    fit on, and a flat/clamped tail would zero the gradient out there."""

    def __init__(self, breakpoints_path=BREAKPOINTS_JSON):
        super().__init__()
        t, phi = load_breakpoints(breakpoints_path)
        assert len(t) == 4 and len(phi) == 4, "expected 4 knots (3 segments)"
        t = torch.tensor(t, dtype=torch.float32)
        phi = torch.tensor(phi, dtype=torch.float32)
        slope = (phi[1:] - phi[:-1]) / (t[1:] - t[:-1])
        intercept = phi[:-1] - slope * t[:-1]
        self.register_buffer("knots_t", t)                # (4,)
        self.register_buffer("knots_phi", phi)             # (4,)
        self.register_buffer("boundaries", t[1:-1].contiguous())  # (2,)
        self.register_buffer("slope", slope)                # (3,)
        self.register_buffer("intercept", intercept)        # (3,)

    def forward(self, t):
        idx = torch.bucketize(t, self.boundaries)  # values in {0, 1, 2}
        return self.slope[idx] * t + self.intercept[idx]

    def extra_repr(self):
        return (f"frozen=True, segments=3, "
                f"t={[round(v, 5) for v in self.knots_t.tolist()]}, "
                f"phi={[round(v, 5) for v in self.knots_phi.tolist()]}")
