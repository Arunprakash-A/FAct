"""Pin ``fact/`` to the implementations that produced the paper's numbers.

``fact/activation.py`` is a consolidated, dependency-light version of code that
lives in two places under ``code/target_vit/``. Consolidation is only safe if it is
numerically exact, so these tests compare it element-wise against both originals
rather than re-deriving the formula.

    pytest tests/ -q
"""
import importlib.util
import math
import os
import sys

import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from fact import (FAct, FrozenFAct, count_activation_parameters,  # noqa: E402
                  fourier_fit, load_coefficients)

ZOO = os.path.join(ROOT, "code", "target_vit")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def study_trainable():
    """The learnable FAct exactly as the target-scale runs instantiated it."""
    sys.path.insert(0, ZOO)
    return _load("study_trainable", os.path.join(ZOO, "fact_k2_global_trainable.py"))


@pytest.fixture(scope="module")
def study_frozen():
    """The frozen FAct exactly as the transfer runs instantiated it."""
    sys.path.insert(0, ZOO)
    return _load("study_frozen", os.path.join(ZOO, "fact_fixed.py"))


T = torch.linspace(-8.0, 8.0, 4001, dtype=torch.float32)


def test_learnable_matches_study_implementation(study_trainable):
    ours, theirs = FAct(K=2, ref="gelu"), study_trainable.FactK2GlobalTrainable()
    assert ours.w == theirs.w
    torch.testing.assert_close(ours.a0, theirs.a0)
    torch.testing.assert_close(ours.a, theirs.a)
    torch.testing.assert_close(ours.b, theirs.b)
    torch.testing.assert_close(ours(T), theirs(T))


def test_frozen_matches_study_implementation(study_frozen):
    # use_cuda_kernel=False is the study's own pure-PyTorch reference path
    ours = FrozenFAct()
    theirs = study_frozen.FixedFAct(use_cuda_kernel=False)
    assert ours.w == theirs.w and ours.K == theirs.K
    torch.testing.assert_close(ours.a0, theirs.a0)
    torch.testing.assert_close(ours.a, theirs.a)
    torch.testing.assert_close(ours.b, theirs.b)
    torch.testing.assert_close(ours(T), theirs(T))


def test_transferred_coefficients_are_the_published_ones():
    """Table/figure captions quote these five numbers; they must not drift."""
    a0, a, b, w, meta = load_coefficients()
    assert w == 1.0 and meta["K"] == 2
    assert round(a0, 4) == 0.2182
    assert [round(float(v), 4) for v in a] == [0.1193, -0.4237]
    assert [round(float(v), 4) for v in b] == [0.8315, -0.5918]
    assert meta["source_run"]["imagenet1k_test_top1"] == 0.6512


def test_all_source_seeds_converged_to_the_same_curve():
    """The transferred curve is a property of the run, not of seed 1.

    All five seeds of the source run end within 0.033 of seed 1 on every one of
    the five coefficients -- small next to the distance from the GELU fit they
    all started at, which is 0.94 on b1 alone.
    """
    a0, a, b, _, meta = load_coefficients()
    spread = 0.0
    for seed, rec in meta["all_seeds_converged"].items():
        deltas = ([abs(rec["a0"] - a0)]
                  + [abs(x - y) for x, y in zip(rec["a"], a)]
                  + [abs(x - y) for x, y in zip(rec["b"], b)])
        spread = max(spread, max(deltas))
    assert spread < 0.05, f"seeds disagree by {spread:.4f}"

    init = meta["gelu_fit_init"]
    travelled = max([abs(init["a0"] - a0)]
                    + [abs(x - y) for x, y in zip(init["a"], a)]
                    + [abs(x - y) for x, y in zip(init["b"], b)])
    assert travelled > 10 * spread, "the curve barely moved from its GELU init"


def test_gelu_fit_is_the_best_k_harmonic_approximation():
    """FAct(ref="gelu") starts at the truncated Fourier series of GELU.

    Two harmonics is a coarse fit -- max error 0.21 over [-1.5, 1.5], which
    comfortably covers the source model's pre-activation range -- so the
    initialisation is a starting point, not the object of study. What must hold
    is that it *is* the truncation, so every added harmonic fits better.
    """
    t = torch.linspace(-1.5, 1.5, 2000)
    target = torch.nn.functional.gelu(t)
    errs = [(FAct(K=k, ref="gelu")(t) - target).abs().max().item()
            for k in (1, 2, 4, 8, 16)]
    assert errs == sorted(errs, reverse=True), errs
    assert errs[-1] < 0.05, f"K=16 fit is off by {errs[-1]:.3f}"

    # and the coefficients are exactly what fourier_fit returns
    a0, a, b, w = fourier_fit(2, ref="gelu")
    m = FAct(K=2, ref="gelu")
    assert m.w == w and abs(m.a0.item() - a0) < 1e-7
    torch.testing.assert_close(m.a.flatten(), torch.tensor(a))


def test_frozen_has_no_learnable_parameters():
    assert list(FrozenFAct().parameters()) == []
    assert FrozenFAct().state_dict()["a"].shape == (1, 2)   # still self-describing


def test_learnable_parameter_count_is_2k_plus_1():
    for k in (1, 2, 4, 8):
        assert sum(p.numel() for p in FAct(K=k).parameters()) == 2 * k + 1


def test_sharing_is_by_identity():
    """One instance reused across depth stays 2K+1; one per block does not."""
    act = FAct(K=2)
    shared = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(4, 4), act)
                                   for _ in range(6)])
    per_block = torch.nn.Sequential(*[torch.nn.Sequential(torch.nn.Linear(4, 4), FAct(K=2))
                                      for _ in range(6)])
    assert count_activation_parameters(shared) == 5
    assert count_activation_parameters(per_block) == 30


def test_gradient_reaches_the_shared_coefficients_from_every_site():
    act = FAct(K=2)
    net = torch.nn.Sequential(torch.nn.Linear(4, 4), act, torch.nn.Linear(4, 4), act)
    net(torch.randn(8, 4)).sum().backward()
    assert act.a0.grad is not None and act.a.grad.abs().sum() > 0


def test_freeze_round_trips():
    act = FAct(K=2)
    with torch.no_grad():
        act.a += 0.3           # move away from the GELU fit
    frozen = act.freeze()
    torch.testing.assert_close(frozen(T), act(T))
    assert list(frozen.parameters()) == []


def test_shape_and_dtype_are_preserved():
    act = FrozenFAct()
    for shape in [(3,), (2, 5), (4, 7, 11)]:
        x = torch.randn(*shape)
        assert act(x).shape == x.shape
    assert act(torch.randn(3, dtype=torch.float64)).dtype == torch.float64
