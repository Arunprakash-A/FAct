"""FAct: one globally shared learnable nonlinearity.

A network using FAct instantiates the activation **once** and applies that same
module at every activation site. With ``K`` harmonics the whole network's
nonlinearity is ``2K + 1`` scalars:

.. math::

    \\phi(t) = a_0 + \\sum_{k=1}^{K} a_k \\cos(k \\omega t) + b_k \\sin(k \\omega t)

``omega`` is fixed at ``pi / L`` with ``L = pi``, i.e. ``omega = 1``, and is a
plain attribute rather than a parameter: it is exactly redundant with the scale
of the affine map feeding the activation, so learning it adds no expressivity.
Only ``a0``, ``a`` and ``b`` are learnable.

Two modules are provided:

``FAct``
    the learnable form, initialised at the truncated Fourier series of a
    reference activation (GELU by default) so training starts as a faithful
    copy of it and is free to move away.

``FrozenFAct``
    the same functional form with the coefficients held in buffers rather than
    parameters, so it contributes exactly zero learnable parameters and the
    optimizer never touches it. Constructed with no argument it loads the
    coefficients this paper transferred -- the curve learned by a depth-6,
    3.05M-parameter ViT on ImageNet-1K, seed 1, 100 epochs.

Sharing is by object identity, so pass one instance everywhere::

    act = FAct(K=2)                     # 5 learnable scalars for the whole net
    blocks = [Block(dim, act=act) for _ in range(depth)]

Constructing one module per block would give each block its own curve and is
*not* what this paper studies; :func:`count_activation_parameters` is the cheap
way to assert you did it right.

The implementations here are consolidated from the study code under ``code/``
for use in other projects; ``tests/test_fact.py`` checks them against those
originals element-wise. A CUDA kernel for the frozen K=2 case lives at
``code/target_vit/cuda_fixed_fact_k2/``, and one for the learnable case at
``code/source_vit/cuda_fact_k2/``.
"""
import json
import math
import os

import numpy as np
import torch
import torch.nn as nn

__all__ = ["FAct", "FrozenFAct", "fourier_fit", "load_coefficients",
           "TRANSFERRED_K2_PATH", "count_activation_parameters"]

#: Coefficients of the nonlinearity this paper transfers: learned end-to-end by
#: a depth-6 ViT on ImageNet-1K and then frozen. Seeds 1-5 of that run end within
#: 0.033 of each other on every one of the five values (tightest on a0/a1/a2,
#: ~0.006-0.009; loosest on b1, 0.033) -- small next to the 0.94 they travelled
#: from the GELU fit, so this is a reproducible curve rather than one seed's
#: accident. The file records every seed's endpoint.
TRANSFERRED_K2_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "coefficients", "imagenet1k_vit_d6_k2.json")

_REFERENCES = {
    "gelu": lambda t: 0.5 * t * (1.0 + _erf(t / math.sqrt(2.0))),
    "relu": lambda t: np.maximum(t, 0.0),
    "silu": lambda t: t / (1.0 + np.exp(-t)),
    "tanh": np.tanh,
}


def _erf(x):
    from scipy.special import erf
    return erf(x)


def fourier_fit(K, ref="gelu", L=math.pi, n_grid=4096):
    """Truncated Fourier series of a reference activation on ``[-L, L]``.

    Returns ``(a0, a, b, omega)`` with ``a`` and ``b`` of shape ``(K,)`` and
    ``omega = pi / L``. Numerically integrated on a uniform ``n_grid`` grid --
    the same computation the study code performs, kept identical so a network
    initialised here starts from the same point the published runs did.

    Note which quadrature this is. It is a left Riemann sum (``endpoint=False``)
    over a function that is not periodic on ``[-L, L]``, so its error falls off
    as O(dt): at the default ``n_grid`` and ``K=2`` it returns ``(0.7057,
    -0.7041, 0.0253, 1.0000, -0.5000)`` for GELU. The study code is not uniform
    here -- ``code/source_vit/fourier_layers.true_fourier_coeffs`` integrates by
    trapezoid *with* endpoints and lands on ``(0.7061, -0.7049, 0.0261, 1.0000,
    -0.5000)``, the exact values, which are what the paper prints and what the
    ImageNet-1K source runs started from. The two differ by ~8e-4.

    This function reproduces the target-scale implementation
    (``code/target_vit/fact_k2_global_trainable.true_fourier_coeffs``), which is
    what ``tests/test_fact.py`` pins it to. If you want the source-scale
    initialisation, take it from ``gelu_fit_init`` in the coefficients JSON.
    Either way these are five trainable scalars from the first step, so the
    difference is an initialisation detail, not a change of method.
    """
    if ref not in _REFERENCES:
        raise ValueError(f"unknown reference {ref!r}; choose from {sorted(_REFERENCES)}")
    t = np.linspace(-L, L, n_grid, endpoint=False)
    dt = t[1] - t[0]
    f = _REFERENCES[ref](t)
    w = math.pi / L
    a0 = (f.sum() * dt) / (2.0 * L)
    a = np.array([(f * np.cos(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
    b = np.array([(f * np.sin(k * w * t)).sum() * dt / L for k in range(1, K + 1)])
    return float(a0), a.astype(np.float32), b.astype(np.float32), w


def load_coefficients(path=TRANSFERRED_K2_PATH):
    """Read a coefficient JSON, returning ``(a0, a, b, omega, meta)``."""
    with open(path) as fh:
        d = json.load(fh)
    a = np.asarray(d["a"], dtype=np.float32)
    b = np.asarray(d["b"], dtype=np.float32)
    if not (len(a) == len(b) == d["K"]):
        raise ValueError(f"{path}: K={d['K']} but {len(a)} cosine / {len(b)} sine terms")
    return float(d["a0"]), a, b, float(d["w"]), d


class _FourierSeries(nn.Module):
    """Shared evaluation of ``phi``; subclasses decide parameter vs. buffer."""

    def _series(self, t):
        # (..., 1) * (K,) -> (..., K) angles, summed back to the input's shape.
        ang = t.unsqueeze(-1) * (self.kvec * self.w)
        return self.a0 + (ang.cos() * self.a).sum(-1) + (ang.sin() * self.b).sum(-1)

    def curve(self, lo=-math.pi, hi=math.pi, n=512):
        """``(t, phi(t))`` as numpy arrays -- for plotting the learned shape."""
        t = torch.linspace(lo, hi, n, dtype=torch.float32,
                           device=self.a0.device)
        with torch.no_grad():
            return t.cpu().numpy(), self(t).cpu().numpy()

    def extra_repr(self):
        return (f"K={self.K}, omega={self.w:.6f}, "
                f"a0={self.a0.item():.6f}, "
                f"a={[round(v, 6) for v in self.a.flatten().tolist()]}, "
                f"b={[round(v, 6) for v in self.b.flatten().tolist()]}")


class FAct(_FourierSeries):
    """The learnable globally shared nonlinearity: ``2K + 1`` scalars in total.

    Args:
        K: number of harmonics. The paper's source runs use ``K=2``.
        ref: activation whose Fourier fit initialises the coefficients; the
            default ``"gelu"`` needs SciPy for ``erf``. ``None``
            starts from zeros, which trains far worse; the paper's runs all use
            the GELU fit (see "GELU Initialization and Optimization").

    One instance is one nonlinearity. Share it across every block::

        act = FAct(K=2)
        assert count_activation_parameters(model) == 5
    """

    def __init__(self, K=2, ref="gelu"):
        super().__init__()
        self.K = int(K)
        if ref is None:
            a0, a, b, w = 0.0, np.zeros(self.K, np.float32), np.zeros(self.K, np.float32), 1.0
        else:
            a0, a, b, w = fourier_fit(self.K, ref=ref)
        self.w = w
        self.ref = ref
        self.a0 = nn.Parameter(torch.tensor([a0], dtype=torch.float32))
        self.a = nn.Parameter(torch.from_numpy(np.asarray(a)).clone().unsqueeze(0))
        self.b = nn.Parameter(torch.from_numpy(np.asarray(b)).clone().unsqueeze(0))
        self.register_buffer("kvec", torch.arange(1, self.K + 1, dtype=torch.float32))

    def forward(self, t):
        return self._series(t)

    def freeze(self):
        """Return a :class:`FrozenFAct` carrying this module's current curve.

        This is the transfer step: the returned module has no parameters, so it
        drops into a different network as an ordinary fixed activation.
        """
        with torch.no_grad():
            return FrozenFAct.from_values(self.a0.item(),
                                          self.a.flatten().cpu().numpy(),
                                          self.b.flatten().cpu().numpy(), self.w)


class FrozenFAct(_FourierSeries):
    """A FAct curve held fixed: zero learnable parameters, drop-in for GELU.

    With no arguments it loads the coefficients this paper transfers, so a
    target network gets the published nonlinearity in one line::

        model = ViT(act=FrozenFAct())

    The coefficients live in buffers, so ``state_dict()`` still carries them and
    a checkpoint stays self-describing, while the optimizer never sees them and
    weight decay cannot drift them.
    """

    def __init__(self, path=TRANSFERRED_K2_PATH):
        super().__init__()
        a0, a, b, w, meta = load_coefficients(path)
        self.K = len(a)
        self.w = w
        self.source = meta.get("source_run", {})
        self._init_buffers(a0, a, b)

    def _init_buffers(self, a0, a, b):
        self.register_buffer("a0", torch.tensor([a0], dtype=torch.float32))
        self.register_buffer("a", torch.tensor(np.asarray(a, np.float32)).unsqueeze(0))
        self.register_buffer("b", torch.tensor(np.asarray(b, np.float32)).unsqueeze(0))
        self.register_buffer("kvec", torch.arange(1, self.K + 1, dtype=torch.float32))

    @classmethod
    def from_values(cls, a0, a, b, w=1.0):
        """Build from explicit coefficients rather than from a JSON file."""
        self = cls.__new__(cls)
        nn.Module.__init__(self)
        self.K = len(a)
        self.w = float(w)
        self.source = {}
        self._init_buffers(a0, a, b)
        return self

    def forward(self, t):
        return self._series(t)


def count_activation_parameters(module):
    """Learnable parameters contributed by FAct modules anywhere in ``module``.

    A correctly shared FAct reports ``2K + 1`` however deep the network is; a
    per-block instantiation reports a multiple of it. Cheap assertion to keep in
    a model's constructor.
    """
    seen, total = set(), 0
    for m in module.modules():
        if isinstance(m, _FourierSeries) and id(m) not in seen:
            seen.add(id(m))
            total += sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
    return total
