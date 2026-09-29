"""FAct -- a globally shared learnable nonlinearity, and the curve this paper
learned on ImageNet-1K and transferred to smaller networks.

    from fact import FAct, FrozenFAct

    act = FAct(K=2)          # learn one nonlinearity jointly with the network
    act = FrozenFAct()       # or install the published, already-learned curve

See :mod:`fact.activation` for the full contract, and ``code/`` for the
training code behind the paper's results.
"""
from .activation import (FAct, FrozenFAct, count_activation_parameters,
                         fourier_fit, load_coefficients, TRANSFERRED_K2_PATH)

__all__ = ["FAct", "FrozenFAct", "fourier_fit", "load_coefficients",
           "count_activation_parameters", "TRANSFERRED_K2_PATH"]
__version__ = "1.0.0"
