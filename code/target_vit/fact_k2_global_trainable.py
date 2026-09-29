"""The K=2 global FAct activation, LEARNED IN PLACE on the target dataset --
"Evolving FAct" (EFAct) in the activation-comparison figures, as opposed to
`fact_fixed.FixedFAct` ("Frozen FAct" / FFAct), which is the same functional
form but with its coefficients frozen at ImageNet-1K-learned values.

    phi(t) = a0 + a1*cos(w t) + b1*sin(w t) + a2*cos(2w t) + b2*sin(2w t)

Coefficients are `nn.Parameter`s (not buffers): exactly five learnable
scalars, ONE set shared by reference across every block and every neuron in
the network (matching how fact_k2_global is trained in the source-network
runs), initialised at the true K=2
Fourier fit of GELU so training starts as a faithful GELU copy and is free to
specialise from there.

This is the ONLY activation in this study's zoo that contributes learnable
parameters -- deliberately. It exists to make the FFAct-vs-fixed-activation
comparison (all zero-param, see fixed_acts.py) into a three-way comparison
against a matched-architecture, matched-budget "trained in place" arm, so
that the EFAct numbers reported alongside FFAct are no longer confounded by
FFAct's depth-2/100K backbone vs. EFAct's previous depth-6/2.7M one.

No CUDA kernel: `cuda_fixed_fact_k2`'s fork deliberately drops the
coefficient-gradient reduction (see fact_fixed.py's docstring) because it was
built for a FROZEN activation -- using it here would silently return zero
coefficient gradient and this activation would never move. Pure PyTorch
autograd is the only correct path for a trainable global FAct at this
parameter count (five scalars, batched-broadcast forward/backward is not the
bottleneck at this ~100K-parameter model scale).
"""
import math

import numpy as np
import torch
import torch.nn as nn


def gelu_np(t):
    from scipy.special import erf
    return 0.5 * t * (1.0 + erf(t / math.sqrt(2.0)))


def true_fourier_coeffs(K, ref="gelu", L=math.pi, n_grid=4096):
    """Numerically-integrated Fourier series coefficients of `ref` truncated to
    K harmonics on [-L, L], as the target-scale runs computed them: a left
    Riemann sum on an endpoint-excluded grid.

    NOT identical to fourier_layers.true_fourier_coeffs, which integrates the
    same formulas by trapezoid with endpoints included and is the more accurate
    of the two -- at K=2 the two disagree by ~8e-4 on a1/a2 (this one returns
    0.0253 for a2, that one 0.0261, the exact value). Each reproduces the
    initialisation its own scale's published runs used, which is why both are
    kept; the coefficients are trainable from step 1 either way."""
    assert ref == "gelu", f"only ref='gelu' is wired up here, got {ref!r}"
    t = np.linspace(-L, L, n_grid, endpoint=False)
    dt = t[1] - t[0]
    f = gelu_np(t)
    w = math.pi / L
    a0 = (f.sum() * dt) / (2.0 * L)
    a = np.array([(f * np.cos(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
    b = np.array([(f * np.sin(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
    return float(a0), a.astype(np.float32), b.astype(np.float32), w


class FactK2GlobalTrainable(nn.Module):
    """phi(t) with a0/a/b as learnable nn.Parameters, GELU-fit initialised,
    ONE set shared (by reference, via this single module instance) across
    every activation site in the network -- exactly five trainable scalars
    for the whole model, verified by fixed_acts.act_param_count."""

    K = 2

    def __init__(self, ref="gelu"):
        super().__init__()
        a0, a, b, w = true_fourier_coeffs(self.K, ref=ref)
        self.w = w
        self.a0 = nn.Parameter(torch.tensor([a0], dtype=torch.float32))
        self.a = nn.Parameter(torch.from_numpy(a).unsqueeze(0).clone())  # (1, K)
        self.b = nn.Parameter(torch.from_numpy(b).unsqueeze(0).clone())  # (1, K)
        self.register_buffer("kvec", torch.arange(1, self.K + 1, dtype=torch.float32))

    def forward(self, t):
        # t: (..., M); (1, K) a/b broadcast against (..., M, K) angles --
        # tied across every feature and every call site, matching how
        # fact_k2_global was trained everywhere else in this repo.
        ang = t.unsqueeze(-1) * (self.kvec * self.w)
        return (self.a0
                + (ang.cos() * self.a).sum(-1)
                + (ang.sin() * self.b).sum(-1))

    def extra_repr(self):
        return (f"K={self.K}, w={self.w:.6f}, frozen=False (trainable), "
                f"a0={self.a0.item():.6f}, "
                f"a={[round(v, 6) for v in self.a.flatten().tolist()]}, "
                f"b={[round(v, 6) for v in self.b.flatten().tolist()]}")
