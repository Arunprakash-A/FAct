"""A 5-segment piecewise-linear approximation of the learned FAct-K2 curve,
frozen the same way `fact_fixed.py`'s FixedFAct and `fact_fixed_pwl3.py`'s
FixedFactPWL3 are frozen.

Direct extension of the 3-segment fit: a K=2 Fourier series has exactly 4
critical points per period (2 local max, 2 local min). fact_fixed_pwl3 only
captured the innermost pair (local max at t=-0.874, local min at t=+0.186),
which are the two that fall inside the FFN pre-activation p99 range -- its
domain edges (+-1.21125) were just that percentile boundary, not extrema.
This module adds the outer pair (local min at t~=-2.46, local max at
t~=+1.91) as two more interior knots, widened to the p99.99 pre-activation
range (+-3.2116) so both new extrema sit properly inside it rather than
almost on the boundary. 6 knots, 5 segments, shape down-up-down-up-down (the
domain starts below the new left minimum and ends above the new right
maximum, so the boundary segments slope INTO those extrema from outside
rather than away from them -- unlike the 3-segment fit's up-down-up, whose
boundary segments slope away from its interior extrema).

    t   = [-3.21157, -2.60319, -0.98408,  0.26741,  2.18947,  3.21157]
    phi = [-0.17977, -1.03266,  0.30150, -0.11324,  1.52431, -0.46117]
    weighted RMSE in range = 0.0600, max abs error in range = 0.4848

The knots ship next to this module in `five_segment_breakpoints.json`, whose
`optimized` block records the fit -- same weighted-RMSE-against-the-empirical-
histogram criterion as the 3-segment fit, just with 4 free interior knots
instead of 2.

Like FixedFactPWL3, the six knots live in `register_buffer`s, never
`nn.Parameter`s -- zero learnable activation parameters. No custom CUDA
kernel, same reasoning as fact_fixed_pwl3.py: the backward is just
`grad_output * slope[segment(t)]`, already a single fused bucketize+gather
pair with no trig and no cross-element reduction to skip.
"""
import json
import os

import torch
import torch.nn as nn

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
#: Source of truth for the knots, shipped alongside this module so a stale
#: copy can't drift silently -- edit that file, not the fallback list below.
BREAKPOINTS_JSON = os.path.join(_THIS_DIR, "five_segment_breakpoints.json")

# Fallback, verbatim from five_segment_breakpoints.json's "optimized" block,
# used only if that file is missing (e.g. the module was vendored on its own).
_FALLBACK_T = [-3.2115682102908254, -2.603186509398306, -0.9840829583727969,
               0.26741081568760683, 2.1894667228836817, 3.2115682102908254]
_FALLBACK_PHI = [-0.17977104127806043, -1.0326629687035087, 0.30149783656904394,
                  -0.11323833574265785, 1.524311147522553, -0.46116812719340716]


def load_breakpoints(path=BREAKPOINTS_JSON):
    try:
        with open(path) as f:
            d = json.load(f)["optimized"]
        return d["t"], d["phi"]
    except FileNotFoundError:
        return _FALLBACK_T, _FALLBACK_PHI


class FixedFactPWL5(nn.Module):
    """5-segment (6-knot) frozen piecewise-linear activation. Interpolates
    between the knots; linearly EXTRAPOLATES beyond them using the boundary
    segment's own slope (via `torch.bucketize`, which does not clamp) --
    same non-saturating design as FixedFactPWL3, verified the same way."""

    def __init__(self, breakpoints_path=BREAKPOINTS_JSON):
        super().__init__()
        t, phi = load_breakpoints(breakpoints_path)
        assert len(t) == 6 and len(phi) == 6, "expected 6 knots (5 segments)"
        t = torch.tensor(t, dtype=torch.float32)
        phi = torch.tensor(phi, dtype=torch.float32)
        slope = (phi[1:] - phi[:-1]) / (t[1:] - t[:-1])
        intercept = phi[:-1] - slope * t[:-1]
        self.register_buffer("knots_t", t)                # (6,)
        self.register_buffer("knots_phi", phi)             # (6,)
        self.register_buffer("boundaries", t[1:-1].contiguous())  # (4,)
        self.register_buffer("slope", slope)                # (5,)
        self.register_buffer("intercept", intercept)        # (5,)

    def forward(self, t):
        idx = torch.bucketize(t, self.boundaries)  # values in {0, 1, 2, 3, 4}
        return self.slope[idx] * t + self.intercept[idx]

    def extra_repr(self):
        return (f"frozen=True, segments=5, "
                f"t={[round(v, 5) for v in self.knots_t.tolist()]}, "
                f"phi={[round(v, 5) for v in self.knots_phi.tolist()]}")
