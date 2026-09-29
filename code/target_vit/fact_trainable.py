"""Promote a FixedFAct's frozen coefficients to trainable full-precision
parameters, and freeze everything else in the network.

This is the mechanism behind the "can the activation repair a quantized
network?" experiment. Three properties matter and each is asserted rather than
assumed:

  GLOBALLY SHARED   FixedFAct already stores ONE coefficient set -- shape (1,)
                    and (1, K), broadcast over every feature and handed by
                    reference to every activation site in the network, tied
                    across neurons AND across depth (that is how
                    fact_k2_global trained the curve in the first place). So
                    promoting the buffers yields FIVE trainable scalars for the
                    whole model, not five per layer. The count is checked.

  FULL PRECISION    The promoted parameters are fp32 and are deliberately NOT
                    registered with the quantizer -- no weight quantization
                    touches them, and no activation observer is attached to
                    them. Five fp32 scalars is a rounding error against a
                    ~100K-parameter INT4 network (0.005% of the weights, and
                    they cost 20 bytes), so keeping them in float costs
                    essentially nothing of what quantization was meant to buy.

  EVERYTHING ELSE FROZEN   fc1 and fc2 -- the two layers the activation sits
                    between -- along with every other weight, bias and norm in
                    the network, keep requires_grad=False and keep the
                    quantized values PTQ gave them. The optimizer is handed the
                    five coefficients and nothing else.

The CUDA kernel cannot be used here. Its autograd Function implements
backward_input only: the fork deliberately dropped the coefficient-gradient
reduction because the activation was frozen, so a tuning run on the kernel path
would silently receive no coefficient gradient at all. Callers must build the
model with use_cuda_kernel=False; make_fact_trainable refuses otherwise rather
than producing a run whose loss never moves.
"""
import torch
import torch.nn as nn

#: The FixedFAct buffers that become parameters. kvec is a constant frequency
#: index, not a coefficient, so it stays a buffer.
COEFF_NAMES = ("a0", "a", "b")


def make_fact_trainable(model, freeze_rest=True):
    """Returns (params, info). `params` is the list handed to the optimizer."""
    act = getattr(model, "shared_act", None)
    if act is None:
        raise ValueError("model has no shared_act -- this expects the study's "
                         "globally-shared activation layout")
    if not hasattr(act, "kvec"):
        raise ValueError(f"shared_act is {type(act).__name__}, not FixedFAct -- "
                         "only the Fourier activation has coefficients to tune")
    if getattr(act, "use_cuda_kernel", False):
        raise ValueError(
            "shared_act uses the CUDA kernel, whose backward returns an input "
            "gradient only (no coefficient gradient) -- build the model with "
            "use_cuda_kernel=False before tuning, or the coefficients would "
            "silently never update")

    if freeze_rest:
        for p in model.parameters():
            p.requires_grad_(False)

    params = []
    for name in COEFF_NAMES:
        buf = getattr(act, name)
        # Swap buffer -> Parameter in place, keeping the value and dtype.
        del act._buffers[name]
        p = nn.Parameter(buf.detach().clone().float())
        setattr(act, name, p)
        params.append(p)

    n_tunable = sum(p.numel() for p in params)
    n_total = sum(p.numel() for p in model.parameters())
    n_other_trainable = sum(p.numel() for p in model.parameters()
                            if p.requires_grad) - n_tunable
    # The whole premise is "five shared scalars, nothing else": if the layout
    # ever stopped being globally shared this would catch it, instead of
    # quietly tuning one curve per layer.
    assert n_tunable == 1 + 2 * act.K, (
        f"expected {1 + 2*act.K} globally-shared coefficients, got {n_tunable} "
        "-- shared_act is no longer one tied set")
    if freeze_rest:
        assert n_other_trainable == 0, (
            f"{n_other_trainable} non-activation parameters are still trainable")

    return params, {"n_tunable": n_tunable, "n_total": n_total,
                    "frac_of_model": n_tunable / n_total,
                    "n_other_trainable": n_other_trainable,
                    "init": coeffs_of(act)}


def coeffs_of(act):
    return {"a0": act.a0.detach().flatten().tolist(),
            "a": act.a.detach().flatten().tolist(),
            "b": act.b.detach().flatten().tolist(),
            "w": float(act.w)}


@torch.no_grad()
def curve(act, lo=-8.0, hi=8.0, n=257, device=None):
    """phi sampled on a grid -- how the tuned activation actually differs in
    shape from the frozen one, which is research question 4's object."""
    t = torch.linspace(lo, hi, n, device=device or act.a0.device)
    return t.tolist(), act(t).detach().flatten().tolist()
