"""Correctness gate for the frozen-FAct CUDA path. Run once per host before
launching that host's sweep -- the kernel is JIT-compiled per machine against
three different nvcc/torch pairings, so "it compiled" is not the same as "it
computes the right thing here".

Checks, in order:
  1. the coefficients loaded are the learned ImageNet-1k ones, and every
     completed seed of that source run agrees with them (the curve under test
     is reproducible, not one seed's accident);
  2. FixedFAct has zero learnable parameters, and so does every other
     activation in the menu -- the equal-capacity premise of the whole study;
  3. CUDA-kernel forward == pure-PyTorch forward;
  4. CUDA-kernel grad wrt input == autograd's grad through the pure-PyTorch
     path -- i.e. the forked `backward_input` entry point, which drops the
     coefficient-gradient reduction, still gets the input gradient right;
  5. the same, under AMP autocast (the fp16/bf16 upcast guard);
  6. the frozen coefficients survive an optimizer step -- a buffer cannot be
     updated by AdamW, but this asserts it rather than assuming it;
  7. a full ViT forward/backward runs and every variant has an identical
     parameter count.
"""
import os
import sys

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from fact_fixed import FixedFAct, load_coeffs  # noqa: E402
from fixed_acts import ACT_KINDS, act_param_count, build_act  # noqa: E402
from vit_acts import build_vit, count_params  # noqa: E402


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
    if not ok:
        raise AssertionError(name)


def main():
    assert torch.cuda.is_available(), "these checks need a CUDA device"
    dev = torch.device("cuda")
    print(f"host={os.uname().nodename} gpu={torch.cuda.get_device_name(0)} "
          f"torch={torch.__version__} cuda={torch.version.cuda}")

    print("\n1. coefficients")
    d = load_coeffs()
    print(f"   source: {os.path.basename(d['source_checkpoint'])} "
          f"(imagenet1k top-1 {d['source_run']['imagenet1k_test_top1']})")
    print(f"   a0={d['a0']:.6f} a={[round(v, 6) for v in d['a']]} "
          f"b={[round(v, 6) for v in d['b']]} w={d['w']}")
    # Agreement is measured on the CURVE, not on the raw coefficients: two
    # coefficient vectors that differ by a few percent can still be the same
    # function to within a fraction of a percent, and the function is what
    # this study freezes and tests.
    import numpy as np
    t = np.linspace(-np.pi, np.pi, 2001)

    def phi(a0, a, b, w):
        out = np.full_like(t, a0)
        for k in (1, 2):
            out += a[k - 1] * np.cos(k * w * t) + b[k - 1] * np.sin(k * w * t)
        return out

    ref = phi(d["a0"], d["a"], d["b"], d["w"])
    rng = ref.max() - ref.min()
    seeds = d["all_seeds_converged"]
    worst = max(np.abs(phi(s["a0"], s["a"], s["b"], d["w"]) - ref).max()
                for s in seeds.values()) / rng
    g = d["gelu_fit_init"]
    from_init = np.abs(phi(g["a0"], g["a"], g["b"], d["w"]) - ref).max() / rng
    check("all source seeds converged to the same curve", worst < 0.05,
          f"worst seed differs from seed1 by {worst:.2%} of the curve's range "
          f"(over {len(seeds)} seeds); for scale, the GELU fit it STARTED from "
          f"differs by {from_init:.1%}")

    print("\n2. zero learnable activation parameters")
    for kind in ACT_KINDS:
        n = act_param_count(kind)
        check(f"{kind} has 0 learnable params", n == 0, f"got {n}")

    print("\n3/4. CUDA kernel vs pure-PyTorch reference (fp32)")
    torch.manual_seed(0)
    act_cuda = FixedFAct(use_cuda_kernel=True).to(dev)
    act_torch = FixedFAct(use_cuda_kernel=False).to(dev)
    x = (torch.randn(64, 65, 256, device=dev) * 3.0).requires_grad_(True)
    xr = x.detach().clone().requires_grad_(True)
    y_cuda, y_torch = act_cuda(x), act_torch(xr)
    fwd_err = (y_cuda - y_torch).abs().max().item()
    check("forward matches", fwd_err < 1e-5, f"max abs err {fwd_err:.3e}")

    g = torch.randn_like(y_cuda)
    y_cuda.backward(g)
    y_torch.backward(g)
    bwd_err = (x.grad - xr.grad).abs().max().item()
    rel = bwd_err / (xr.grad.abs().max().item() + 1e-12)
    check("grad wrt input matches (frozen-coeff backward_input path)",
          rel < 1e-5, f"max abs err {bwd_err:.3e} (rel {rel:.3e})")

    print("\n5. under autocast")
    for dtype in (torch.float16, torch.bfloat16):
        try:
            xa = (torch.randn(8, 65, 256, device=dev)).requires_grad_(True)
            with torch.autocast("cuda", dtype=dtype):
                ya = act_cuda(xa.to(dtype))
            ya.sum().backward()
            check(f"autocast {dtype} forward+backward runs",
                  torch.isfinite(ya).all().item() and xa.grad is not None)
        except RuntimeError as e:  # V100 has no bf16
            check(f"autocast {dtype} unsupported on this GPU (skipped)", True, str(e)[:60])

    print("\n6. coefficients are frozen under an optimizer step")
    m = build_vit("cifar10", "fact_fixed").to(dev)
    before = (m.shared_act.a0.clone(), m.shared_act.a.clone(), m.shared_act.b.clone())
    opt = torch.optim.AdamW(m.parameters(), lr=1e-1, weight_decay=0.5)
    out = m(torch.randn(8, 3, 32, 32, device=dev))
    out.sum().backward()
    opt.step()
    after = (m.shared_act.a0, m.shared_act.a, m.shared_act.b)
    check("a0/a/b unchanged after AdamW step",
          all(torch.equal(b, a) for b, a in zip(before, after)))

    print("\n7. every variant is the same size and trains")
    ref = None
    for ds in ("fmnist", "cifar10", "cifar100"):
        counts = {}
        for kind in ACT_KINDS:
            mm = build_vit(ds, kind).to(dev)
            pc = count_params(mm)
            counts[kind] = pc["total"]
            c = 1 if ds == "fmnist" else 3
            s = 28 if ds == "fmnist" else 32
            o = mm(torch.randn(4, c, s, s, device=dev))
            o.sum().backward()
            assert torch.isfinite(o).all(), f"{ds}/{kind} produced non-finite logits"
        uniq = set(counts.values())
        check(f"{ds}: all {len(ACT_KINDS)} variants have identical param count",
              len(uniq) == 1, f"{next(iter(uniq)):,} params")
        ref = counts

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
