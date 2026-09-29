"""The activation menu under test: one learned-then-frozen FAct curve vs. ten
standard fixed activations.

Every entry here is parameter-free. That is the design constraint that makes
this study a clean comparison -- swapping the activation changes the function
and nothing else, so every variant's ViT has an identical parameter count and
an identical optimizer state (verified at run time by train_static.py, which
asserts the count against a reference variant).

The ten baselines cover the families a fixed activation can belong to:
  piecewise-linear   relu, leaky_relu, hardswish (piecewise-linear gate)
  smooth self-gated  gelu, silu (Swish-1), mish
  saturating         tanh, elu, selu
  smooth-positive    softplus
and `fact_fixed` is the learned Fourier curve -- periodic, non-monotone, and
unbounded in neither direction: structurally unlike all ten.

Defaults are each activation's canonical/library default (LeakyReLU
negative_slope=0.01, ELU alpha=1.0, Softplus beta=1, SELU's fixed
lambda/alpha), i.e. what someone reaching for the activation would actually
get, not a per-dataset tuned value -- tuning one baseline and not the others
is exactly the confound this study is meant to avoid.
"""
import torch.nn as nn

from fact_fixed import FixedFAct
from fact_fixed_pwl3 import FixedFactPWL3
from fact_fixed_pwl3_relu import FixedFactPWL3ReLU
from fact_fixed_pwl5 import FixedFactPWL5
from fact_k2_global_trainable import FactK2GlobalTrainable

#: Every activation under test. Order is the canonical reporting order. Every
#: entry here is parameter-free (see act_param_count) -- this is the
#: zero-capacity zoo this study's equal-capacity guarantee rests
#: on. "fact_k2_global" (Evolving FAct / EFAct, five learnable parameters,
#: build_act below) is deliberately NOT in this tuple: it is a separate,
#: standalone comparison against this same architecture and budget, not an
#: addition to the zero-param sweep.
ACT_KINDS = (
    "fact_fixed",
    "fact_fixed_pwl3",
    "fact_fixed_pwl3_udu",
    "fact_fixed_pwl3_relu",
    "fact_fixed_pwl5",
    "relu",
    "leaky_relu",
    "gelu",
    "silu",
    "elu",
    "tanh",
    "mish",
    "softplus",
    "hardswish",
    "selu",
)

#: Human-readable labels for figures/tables.
ACT_LABELS = {
    "fact_fixed": "FAct-K2 (learned, frozen)",
    "fact_fixed_pwl3": "FAct-K2 3-seg PWL (learned, frozen)",
    "fact_fixed_pwl3_udu": "FAct-K2 3-seg PWL-UdU (learned, frozen)",
    "fact_fixed_pwl3_relu": "FAct-K2 3-seg PWL, ReLU-hinge form (learned, frozen)",
    "fact_fixed_pwl5": "FAct-K2 5-seg PWL (learned, frozen)",
    "relu": "ReLU",
    "leaky_relu": "LeakyReLU(0.01)",
    "gelu": "GELU",
    "silu": "SiLU / Swish-1",
    "elu": "ELU(1.0)",
    "tanh": "Tanh",
    "mish": "Mish",
    "softplus": "Softplus(1.0)",
    "hardswish": "Hardswish",
    "selu": "SELU",
}

_BUILTINS = {
    "relu": lambda: nn.ReLU(),
    "leaky_relu": lambda: nn.LeakyReLU(negative_slope=0.01),
    "gelu": lambda: nn.GELU(),
    "silu": lambda: nn.SiLU(),
    "elu": lambda: nn.ELU(alpha=1.0),
    "tanh": lambda: nn.Tanh(),
    "mish": lambda: nn.Mish(),
    "softplus": lambda: nn.Softplus(beta=1.0, threshold=20.0),
    "hardswish": lambda: nn.Hardswish(),
    "selu": lambda: nn.SELU(),
}


def build_act(kind, use_cuda_kernel=True):
    """One activation module. `use_cuda_kernel` applies to `fact_fixed` only
    (the ten builtins are single fused ATen ops already, and fact_fixed_pwl3
    has no CUDA kernel of its own -- see fact_fixed_pwl3.py for why); it is
    accepted for every kind so callers never have to branch on the variant
    name."""
    if kind == "fact_fixed":
        return FixedFAct(use_cuda_kernel=use_cuda_kernel)
    if kind == "fact_fixed_pwl3":
        return FixedFactPWL3()
    if kind == "fact_fixed_pwl3_udu":
        # Same module/knots as fact_fixed_pwl3 -- verified (see FixedFactPWL3's
        # docstring) to already extrapolate linearly and unboundedly on both
        # outer segments, i.e. no saturating/clamped tail exists to remove.
        # Registered under this explicit name so the up-down-up, non-saturating
        # design is unambiguous in results/labels rather than implicit.
        return FixedFactPWL3()
    if kind == "fact_fixed_pwl3_relu":
        return FixedFactPWL3ReLU()
    if kind == "fact_fixed_pwl5":
        return FixedFactPWL5()
    if kind == "fact_k2_global":
        # Evolving FAct (EFAct): the only kind in this registry with
        # learnable parameters. Not in ACT_KINDS -- see the note above it.
        return FactK2GlobalTrainable()
    if kind in _BUILTINS:
        return _BUILTINS[kind]()
    raise ValueError(f"unknown activation kind: {kind!r} (expected one of "
                     f"{ACT_KINDS} or 'fact_k2_global')")


def act_param_count(kind, use_cuda_kernel=True):
    """Learnable parameters an activation of this kind contributes. Must be 0
    for every kind in ACT_KINDS -- FixedFAct's coefficients are buffers, not
    parameters. Used by the run drivers' equal-capacity assertion. The one
    exception is 'fact_k2_global' (Evolving FAct), which is expected to
    return exactly 5 (a0, a1, b1, a2, b2) -- it is deliberately outside
    ACT_KINDS and callers must not assert it against 0."""
    return sum(p.numel() for p in build_act(kind, use_cuda_kernel).parameters()
               if p.requires_grad)
