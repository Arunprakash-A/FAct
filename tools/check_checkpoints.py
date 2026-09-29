"""Check the shipped checkpoints.

Each one is rebuilt from the code in code/, loaded strictly, and checked against
the accuracy and config stored inside it. The whole frozen-curve family -- every
target checkpoint on all four datasets -- is swept for the same five scalars and
for per-seed and mean accuracy against the paper. The CIFAR-10 seed-0 target is
additionally re-evaluated end to end and must reproduce its recorded accuracy.

    python tools/check_checkpoints.py                  # downloads CIFAR-10
    python tools/check_checkpoints.py --data-root DIR  # reuse a copy
    python tools/check_checkpoints.py --skip-eval      # no dataset needed

Exits non-zero if any check fails.
"""
import argparse
import glob
import json
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_CODE = os.path.join(ROOT, "code", "source_vit")
TGT_CODE = os.path.join(ROOT, "code", "target_vit")
COEFFS = os.path.join(ROOT, "fact", "coefficients", "imagenet1k_vit_d6_k2.json")

FACT_D6 = os.path.join(ROOT, "checkpoints", "vit_d6_imagenet1k_fact_k2_seed1_final.pt")
GELU_D6 = os.path.join(ROOT, "checkpoints", "vit_d6_imagenet1k_gelu_seed1_final.pt")
FFACT_D2 = os.path.join(ROOT, "checkpoints", "vit_d2_cifar10_frozen_fact_seed0_best.pt")
FFACT_GLOB = os.path.join(ROOT, "checkpoints", "vit_d2_*_frozen_fact_seed*_best.pt")

#: The paper's parameter counts. The difference is the point: +5 for the curve.
SRC_PARAMS = {"standard": 3_048_232, "fact_k2_global": 3_048_237}

#: The paper's ImageNet-1K means over 5 seeds, for a sanity bound on seed 1.
PAPER_IN1K = {"fact_k2_global": (64.38, 0.44), "gelu": (62.45, 0.13)}

#: The paper's target-scale Frozen FAct results -- (mean, s.d., seeds averaged),
#: shared hyper-parameters (lr 1e-3, wd 0.05), 100 epochs. Every mean here is
#: over SIX seeds; where fewer than six ship, the mean check is loosened to one
#: s.d. rather than silently comparing a different average.
PAPER_TARGET = {"fmnist": (90.95, 0.10, 6), "cifar10": (75.45, 0.40, 6),
                "cifar100": (47.79, 0.52, 6), "food101": (35.82, 0.33, 6)}

_results, _failed = [], []


def check(label, ok, detail=""):
    _results.append(ok)
    if not ok:
        _failed.append(label)
    print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def build_source(ckpt):
    sys.path.insert(0, SRC_CODE)
    import cv_vit  # noqa: E402
    c = ckpt["config"]
    m = cv_vit.build_vit("imagenet1k", c["variant"], depth=c["depth"], dropout=c["dropout"])
    sys.path.remove(SRC_CODE)
    return m


def build_target(ckpt, use_cuda_kernel=None):
    """The frozen activation has a CUDA kernel and an equivalent PyTorch
    fallback; default to whichever the run recorded, but allow forcing it."""
    sys.path.insert(0, TGT_CODE)
    import vit_acts  # noqa: E402
    c = ckpt["config"]
    if use_cuda_kernel is None:
        use_cuda_kernel = c.get("use_cuda_kernel", True)
    m = vit_acts.build_vit(c["dataset"], c["act_kind"], embed_dim=c["embed_dim"],
                           depth=c["depth"], num_heads=c["num_heads"],
                           mlp_ratio=c["mlp_ratio"], dropout=c["dropout"],
                           use_cuda_kernel=use_cuda_kernel)
    sys.path.remove(TGT_CODE)
    return m


def coeffs_of(state_dict, prefix="shared_act"):
    """The five scalars (a0, a1, b1, a2, b2). The two code trees spell the
    tensors differently -- leading underscores in one -- so accept either."""
    def get(name):
        for k in (f"{prefix}.{name}", f"{prefix}._{name}"):
            if k in state_dict:
                return state_dict[k].flatten()
        raise KeyError(f"{prefix}.{name}")
    a0, a, b = get("a0"), get("a"), get("b")
    return torch.tensor([a0[0], a[0], b[0], a[1], b[1]], dtype=torch.float64)


def check_structure(fact6, gelu6, ffact2):
    print("\n[1] structure -- rebuild from code/ and load strictly")
    counts = {}
    for name, ckpt, builder in (("fact_k2_global", fact6, build_source),
                                ("standard", gelu6, build_source),
                                ("fact_fixed", ffact2, build_target)):
        model = builder(ckpt)
        try:
            model.load_state_dict(ckpt["state_dict"], strict=True)
            ok, why = True, ""
        except RuntimeError as e:
            ok, why = False, str(e).split("\n")[0]
        check(f"{name}: loads with strict=True", ok, why)
        counts[name] = (
            sum(p.numel() for p in model.parameters() if p.requires_grad),
            sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and (".act" in n or n.startswith("shared_act"))))

    for variant, want in SRC_PARAMS.items():
        check(f"{variant}: {want:,} trainable parameters",
              counts[variant][0] == want, f"{counts[variant][0]:,}")
    check("one shared curve costs exactly +5 parameters",
          counts["fact_k2_global"][0] - counts["standard"][0] == 5)
    check("Learnable FAct: 5 trainable activation parameters",
          counts["fact_k2_global"][1] == 5)
    check("GELU: 0 trainable activation parameters", counts["standard"][1] == 0)

    recorded = ffact2["config"]["params"]["total"]
    check(f"target: {recorded:,} trainable parameters, as its config records",
          counts["fact_fixed"][0] == recorded, f"{counts['fact_fixed'][0]:,}")
    check("target: 0 trainable activation parameters -- the curve is installed, "
          "not learned", counts["fact_fixed"][1] == 0)


def check_curve(fact6, ffact2):
    print("\n[2] the transferred curve -- bit-exact across source, file and target")
    src, tgt = coeffs_of(fact6["state_dict"]), coeffs_of(ffact2["state_dict"])
    pub = json.load(open(COEFFS))
    published = torch.tensor([pub["a0"], pub["a"][0], pub["b"][0],
                              pub["a"][1], pub["b"][1]], dtype=torch.float64)

    print("       " + "  ".join(f"{n:>12}" for n in ("a0", "a1", "b1", "a2", "b2")))
    for label, v in (("source", src), ("published", published), ("target", tgt)):
        print(f"  {label:>9} " + "  ".join(f"{x:12.9f}" for x in v))

    check("source checkpoint == fact/coefficients/", torch.equal(src, published))
    check("target checkpoint == source checkpoint", torch.equal(tgt, src))

    for label, ckpt in (("source", fact6), ("target", ffact2)):
        sd = ckpt["state_dict"]
        blocks = sorted({k.split(".ffn.act.")[0] for k in sd if ".ffn.act." in k})
        per_block = [coeffs_of(sd, f"{b}.ffn.act") for b in blocks]
        check(f"{label}: all {len(blocks)} blocks hold the same five numbers",
              all(torch.equal(c, per_block[0]) for c in per_block)
              and torch.equal(per_block[0], coeffs_of(sd)))


def check_reference(ffact2):
    print("\n[3] fact.FrozenFAct against the checkpoint's own activation")
    sys.path.insert(0, ROOT)
    from fact import FrozenFAct, count_activation_parameters  # noqa: E402
    sys.path.remove(ROOT)

    published = FrozenFAct()
    t = torch.linspace(-8, 8, 20_001)

    model = build_target(ffact2, use_cuda_kernel=False)
    model.load_state_dict(ffact2["state_dict"], strict=True)
    with torch.no_grad():
        gap = (published(t) - model.blocks[0].ffn.act(t)).abs().max().item()
    check("matches elementwise (PyTorch path)", gap < 1e-6, f"max diff {gap:.2e}")

    # The CUDA kernel is the path that actually ran; check it where possible.
    if torch.cuda.is_available() and ffact2["config"].get("use_cuda_kernel"):
        cm = build_target(ffact2, use_cuda_kernel=True).cuda()
        cm.load_state_dict(ffact2["state_dict"], strict=True)
        with torch.no_grad():
            g = (published(t).cuda() - cm.blocks[0].ffn.act(t.cuda())).abs().max().item()
        check("matches the CUDA kernel the run used", g < 1e-6, f"max diff {g:.2e}")
    else:
        print("       (CUDA kernel not checked -- no GPU)")

    check("FrozenFAct has zero learnable parameters",
          count_activation_parameters(published) == 0)


def check_recorded(fact6, gelu6):
    print("\n[4] recorded accuracy against the paper's 5-seed means")
    for variant, ckpt in (("fact_k2_global", fact6), ("gelu", gelu6)):
        mean, sd = PAPER_IN1K[variant]
        got = ckpt["test_acc"] * 100
        check(f"{variant} seed 1 within the paper's mean +- 3 sd",
              abs(got - mean) <= 3 * sd, f"{got:.2f}% vs {mean:.2f} +- {sd:.2f}%")
    check("the source gap has the paper's sign and size",
          fact6["test_acc"] - gelu6["test_acc"] > 0.015,
          f"{(fact6['test_acc'] - gelu6['test_acc']) * 100:+.2f} pp, seed 1")


def target_family():
    """Every shipped target checkpoint, grouped by dataset: {ds: {seed: path}}."""
    family = {}
    for path in sorted(glob.glob(FFACT_GLOB)):
        stem = os.path.basename(path)[len("vit_d2_"):-len("_best.pt")]
        ds, _, seed = stem.partition("_frozen_fact_seed")
        family.setdefault(ds, {})[int(seed)] = path
    return family


def check_family():
    """The frozen curve is one object; these are the networks it was installed
    in. Build every one from code/, load it strictly, and confirm it carries the
    published five scalars and the accuracy the paper reports.

    Built on the pure-PyTorch path throughout: the state dict is the same either
    way (check_reference pins the two against each other), and this keeps the
    sweep free of a CUDA toolchain."""
    print("\n[5] the frozen-curve family -- every target checkpoint")
    pub = json.load(open(COEFFS))
    published = torch.tensor([pub["a0"], pub["a"][0], pub["b"][0],
                              pub["a"][1], pub["b"][1]], dtype=torch.float64)

    for ds, seeds in sorted(target_family().items()):
        if ds not in PAPER_TARGET:
            check(f"{ds}: has a published mean to check against", False,
                  "not in PAPER_TARGET")
            continue
        mean, sd, n_paper = PAPER_TARGET[ds]
        loaded, curves, act_params, accs, strays = [], [], [], [], []

        for seed, path in sorted(seeds.items()):
            ck = load(path)
            model = build_target(ck, use_cuda_kernel=False)
            try:
                model.load_state_dict(ck["state_dict"], strict=True)
                loaded.append(True)
            except RuntimeError as e:
                loaded.append(False)
                strays.append(f"seed {seed}: {str(e).splitlines()[0]}")
            curves.append(torch.equal(coeffs_of(ck["state_dict"]), published))
            act_params.append(sum(
                p.numel() for n, p in model.named_parameters()
                if p.requires_grad and (".act" in n or n.startswith("shared_act"))))
            # A checkpoint saved from the mid-run flush has no test_acc: it is
            # not the best-val model the paper scored, and must not be averaged.
            accs.append(None if "test_acc" not in ck else ck["test_acc"] * 100)

        got = [a for a in accs if a is not None]
        shown = ", ".join("--" if a is None else f"{a:.2f}" for a in accs)
        print(f"  {ds:9} seeds {sorted(seeds)}  ->  {shown}")

        check(f"{ds}: all {len(seeds)} load with strict=True", all(loaded),
              "; ".join(strays))
        check(f"{ds}: all {len(seeds)} carry the published curve, bit-exact",
              all(curves))
        check(f"{ds}: all {len(seeds)} have 0 trainable activation parameters",
              all(p == 0 for p in act_params))
        check(f"{ds}: every seed records a test accuracy", len(got) == len(accs),
              f"{len(accs) - len(got)} missing")
        check(f"{ds}: every seed within the paper's mean +- 3 s.d.",
              bool(got) and all(abs(a - mean) <= 3 * sd for a in got),
              f"paper {mean:.2f} +- {sd:.2f}%")

        avg = sum(got) / len(got) if got else float("nan")
        if len(got) == n_paper:
            check(f"{ds}: the {len(got)}-seed mean reproduces the paper's",
                  abs(avg - mean) < 0.005, f"{avg:.2f}% vs {mean:.2f}%")
        else:
            check(f"{ds}: the {len(got)}-seed mean is within 1 s.d. of the "
                  f"paper's {n_paper}-seed mean", abs(avg - mean) <= sd,
                  f"{avg:.2f}% vs {mean:.2f}%")


def check_recomputed(ffact2, data_root, device):
    print(f"\n[6] re-evaluating the CIFAR-10 seed-0 target ({device})")
    from torchvision.datasets import CIFAR10

    sys.path.insert(0, TGT_CODE)
    from cv_data import MEAN_STD, TensorImageDataset  # noqa: E402
    sys.path.remove(TGT_CODE)

    ds = CIFAR10(root=data_root, train=False, download=True)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float().div(255.0)
    y = torch.tensor(ds.targets, dtype=torch.long)
    check("CIFAR-10 test split is 10,000 images", len(y) == 10_000)

    # Normalisation and evaluation are the study's own; only loading differs.
    loader = torch.utils.data.DataLoader(
        TensorImageDataset(x, y, *MEAN_STD["cifar10"], train_aug=False),
        batch_size=512, shuffle=False)

    model = build_target(ffact2, use_cuda_kernel=device.startswith("cuda")
                         and ffact2["config"].get("use_cuda_kernel", True))
    model.load_state_dict(ffact2["state_dict"], strict=True)
    model.to(device).eval()

    correct = 0
    with torch.no_grad():
        for xb, yb in loader:
            correct += (model(xb.to(device)).argmax(1).cpu() == yb).sum().item()
    acc = correct / len(y)
    print(f"       recomputed top-1: {acc * 100:.2f}%  ({correct:,}/{len(y):,})")
    check("recomputed == the accuracy stored in the checkpoint",
          abs(acc - ffact2["test_acc"]) < 5e-5,
          f"{acc * 100:.2f}% vs {ffact2['test_acc'] * 100:.2f}%")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.path.join(ROOT, "data"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    for p in (FACT_D6, GELU_D6, FFACT_D2):
        if not os.path.exists(p):
            sys.exit(f"missing checkpoint: {os.path.relpath(p, ROOT)}")

    fact6, gelu6, ffact2 = load(FACT_D6), load(GELU_D6), load(FFACT_D2)
    print(f"torch {torch.__version__}")
    for label, ck in (("source, Learnable FAct", fact6), ("source, GELU", gelu6),
                      ("target, Frozen FAct", ffact2)):
        c = ck["config"]
        print(f"  {label:24} {c.get('dataset')}, depth {c.get('depth')}, "
              f"seed {c.get('seed')}, {c.get('epochs')} epochs")

    check_structure(fact6, gelu6, ffact2)
    check_curve(fact6, ffact2)
    check_reference(ffact2)
    check_recorded(fact6, gelu6)
    check_family()
    if args.skip_eval:
        print("\n[6] re-evaluation skipped (--skip-eval)")
    else:
        check_recomputed(ffact2, args.data_root, args.device)

    print(f"\n{sum(_results)}/{len(_results)} checks passed")
    if _failed:
        for f in _failed:
            print("  failed:", f)
        sys.exit(1)


if __name__ == "__main__":
    main()
