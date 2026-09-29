"""The same 3-segment PWL as fact_fixed_pwl3, reformulated as a base line plus
a sum of scaled ReLU hinges -- i.e. built directly out of `max(0, x)` the way
ReLU itself is, instead of `torch.bucketize` + segment lookup.

    phi(t) = m0*t + c0 + d1*ReLU(t - t1) + d2*ReLU(t - t2)

where m0/c0 are the LEFTMOST segment's own slope/intercept (the line the
function follows for all t < t1, extrapolated), t1/t2 are the two interior
knots, and d1/d2 are the SLOPE CHANGES at each knot (d_i = slope_i -
slope_{i-1}). Standard PWL-as-hinge-sum identity -- verified bit-for-bit
against FixedFactPWL3's bucketize implementation over t in [-8, 8] (max abs
diff ~4e-8, floating-point noise, i.e. exact), unlike a naive nested
min/max-of-lines form, which only reproduces the true function on t < ~2.15
and silently reverts toward the leftmost segment's line beyond that -- min/max
of GLOBAL affine lines is only exact everywhere for a convex/concave function,
and this one (up-down-up, unbounded both tails) is neither.

Same knots, same source (three_segment_breakpoints.json's "optimized" block,
same fallback), same zero-parameter freezing as fact_fixed_pwl3.py -- this
module exists to test whether the hinge-sum form (plain arithmetic, no
data-dependent branching) is faster or slower on GPU than bucketize+gather,
not to change the function computed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from fact_fixed_pwl3 import load_breakpoints, BREAKPOINTS_JSON


class FixedFactPWL3ReLU(nn.Module):
    """Hinge-sum reformulation of FixedFactPWL3. Same phi(t), different
    arithmetic: no bucketize/gather, just two ReLUs and an affine map."""

    def __init__(self, breakpoints_path=BREAKPOINTS_JSON):
        super().__init__()
        t, phi = load_breakpoints(breakpoints_path)
        assert len(t) == 4 and len(phi) == 4, "expected 4 knots (3 segments)"
        t = torch.tensor(t, dtype=torch.float32)
        phi = torch.tensor(phi, dtype=torch.float32)
        slope = (phi[1:] - phi[:-1]) / (t[1:] - t[:-1])   # (3,)
        intercept = phi[:-1] - slope * t[:-1]              # (3,)

        self.register_buffer("m0", slope[0].clone())
        self.register_buffer("c0", intercept[0].clone())
        self.register_buffer("t1", t[1].clone())
        self.register_buffer("t2", t[2].clone())
        self.register_buffer("d1", (slope[1] - slope[0]).clone())
        self.register_buffer("d2", (slope[2] - slope[1]).clone())
        # Kept only for extra_repr/debugging parity with FixedFactPWL3.
        self.register_buffer("knots_t", t)
        self.register_buffer("knots_phi", phi)

    def forward(self, t):
        return (self.m0 * t + self.c0
                + self.d1 * F.relu(t - self.t1)
                + self.d2 * F.relu(t - self.t2))

    def extra_repr(self):
        return (f"frozen=True, segments=3, formulation=relu-hinge-sum, "
                f"t={[round(v, 5) for v in self.knots_t.tolist()]}, "
                f"phi={[round(v, 5) for v in self.knots_phi.tolist()]}")
