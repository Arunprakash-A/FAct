"""Train a ~100K model on FMNIST / CIFAR-10 / CIFAR-100, one activation per run.

Three architectures share this trainer, selected with --arch:

  vit           ~100K ViT (vit_acts.py), embed=64 depth=2 heads=4 mlp=4.0
  convmlpmixer  ~100K ConvMLPMixer+GAP (convmlpmixer_acts.py), channels=(39,78)
  resnet18      11.2M CIFAR-style ResNet-18 (resnet18_acts.py), widths 64-512
  mlp           ~100K plain MLP on flattened input (mlp_acts.py), 784-100x3-10
  mlp_bn        the same MLP with BatchNorm1d before every activation
  mlp_ln        the same MLP with LayerNorm before every activation
  conv1d_mlp_ln       Conv1d(1->100) -> pool(8) -> LayerNorm MLP (104,510)

They differ ONLY in the model constructor, the per-dataset epoch budget (each
architecture keeps the budget its own source experiment used, so its numbers
stay comparable to that experiment) and the dropout rate (the reference
ResNet-18 has none). Optimizer, schedule, batch size, augmentation, validation
split, checkpoint selection and the equal-capacity assertion are shared code,
so an activation ranking measured on one architecture is measured the same way
on the others.

The first two arms were built to a shared ~100K budget; resnet18 is NOT -- it
runs at its own natural size, so it tests whether the ranking survives a ~110x
capacity jump rather than adding a third point at equal capacity. The
equal-capacity assertion below is WITHIN an arm and holds for all three.

Training recipe is a faithful port of the main study's cv_train.train_one, so
these numbers sit on the same footing as every other CV run in this repo:

    AdamW(lr=1e-3, weight_decay=0.05), batch 256, label smoothing 0.1,
    5 warmup epochs then a cosine arc down to 1% of base lr,
    pad-and-random-crop + horizontal flip augmentation (cv_data.py),
    5000-image validation split carved out of train with a FIXED seed (42),
    best-val checkpoint selection, top-1 reported on the real test set.
    Epoch budgets: fmnist 20, cifar10 40, cifar100 60 (cv_train.EPOCHS).

Three deliberate departures from cv_train.train_one, all applied identically to
every variant so they cannot favour one activation:

  1. NO automatic mixed precision -- fp32 everywhere. The three hosts this
     study runs on have three different fast low-precision formats (H200 bf16,
     L4 bf16, V100 fp16-only), and a periodic activation
     evaluated in bf16 is not the same function as one evaluated in fp16.
     Running fp32 makes a CIFAR-100 number computed on the H200 directly
     comparable to a CIFAR-10 number computed on an L4. At ~100K parameters
     the throughput cost is negligible.
  2. The model is vit_acts.build_vit (activation injected as a module), not
     cv_vit.build_vit (activation selected by FFN-variant string).
  3. Only two checkpoints per run are written -- best-val and last -- per the
     study instruction. No per-epoch checkpoints.

Everything needed to plot convergence (per-epoch train/val loss and accuracy,
the LR trace, wall-clock) is written to <run>_result.json.
"""
import argparse
import gc
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy("file_system")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)

from cv_data import get_dataloaders  # noqa: E402
from fixed_acts import ACT_KINDS, act_param_count  # noqa: E402
from vit_acts import (DEPTH_DEFAULT, EMBED_DIM_DEFAULT, MLP_RATIO_DEFAULT,  # noqa: E402
                      NUM_HEADS_DEFAULT, build_vit)
from vit_acts import count_params as vit_count_params  # noqa: E402
from convmlpmixer_acts import (CHANNELS_DEFAULT, FC_HIDDEN_DEFAULT,  # noqa: E402
                               build_convmlpmixer)
from convmlpmixer_acts import count_params as cnn_count_params  # noqa: E402
from resnet18_acts import (LAYERS_DEFAULT, WIDTH_DEFAULT,  # noqa: E402
                           build_resnet18)
from resnet18_acts import count_params as resnet_count_params  # noqa: E402
from resnet18_postln_acts import build_resnet18_postln  # noqa: E402
from resnet18_postln_acts import count_params as resnet_postln_count_params  # noqa: E402
from resnet18_actnorm_acts import build_resnet18_actnorm  # noqa: E402
from resnet18_actnorm_acts import count_params as resnet_actnorm_count_params  # noqa: E402
from convnext_acts import (CHANNELS_DEFAULT as CONVNEXT_CHANNELS_DEFAULT,  # noqa: E402
                           BLOCKS_DEFAULT as CONVNEXT_BLOCKS_DEFAULT, build_convnext)
from convnext_acts import count_params as convnext_count_params  # noqa: E402
from mlp_acts import (DEPTH_DEFAULT as MLP_DEPTH_DEFAULT,  # noqa: E402
                      WIDTH_DEFAULT as MLP_WIDTH_DEFAULT, build_mlp)
from mlp_acts import count_params as mlp_count_params  # noqa: E402
from conv1d_mlp_acts import (WIDTH_DEFAULT as C1D_WIDTH,  # noqa: E402
                             build_conv1d_mlp)
from conv1d_mlp_acts import count_params as c1d_count_params  # noqa: E402

ARCHS = ("vit", "convmlpmixer", "resnet18", "resnet18_postln", "resnet18_actnorm",
         "convnext", "mlp", "mlp_bn", "mlp_ln", "conv1d_mlp_ln")
C1D_WIDTH_BY_ARCH = {"conv1d_mlp_ln": C1D_WIDTH}
#: norm kind per MLP variant -- the only thing separating the three.
MLP_NORM = {"mlp": "none", "mlp_bn": "batch", "mlp_ln": "layer"}

#: Per-architecture epoch budgets. vit: cv_train.EPOCHS (what the ViT arm of
#: this study ran). convmlpmixer: the companion CNN study's epoch map, which
#: gives FMNIST 30 rather than 20 -- kept as-is so these runs are directly
#: comparable to that experiment's learnable-activation numbers.
EPOCHS_BY_ARCH = {
    # food101: 101 fine-grained classes over 75K images -- the hardest of the
    # five, given cifar100's budget. eurosat: 10 easy classes over ~21.6K
    # images, given cifar10's.
    "vit": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
            "food101": 60, "eurosat": 40},
    # food101/eurosat have no ConvMLPMixer source experiment to inherit from,
    # so they take the ViT arm's budgets -- that keeps the two architectures
    # directly comparable on exactly these two datasets.
    "convmlpmixer": {"fmnist": 30, "cifar10": 40, "cifar100": 60,
                     "food101": 60, "eurosat": 40},
    # resnet18 has no source experiment in this repo to inherit from, so it
    # takes the ViT arm's budgets on all five -- the same reasoning already
    # applied to food101/eurosat above, and what makes a per-dataset ranking
    # comparable across the three arms.
    "resnet18": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
                 "food101": 60, "eurosat": 40},
    # resnet18_postln: the resnet18 arch + a LayerNorm2d after each residual
    # addition (see resnet18_postln_acts.py) -- a targeted ablation, not a new
    # capacity-matched arm, so it just inherits resnet18's budgets.
    "resnet18_postln": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
                        "food101": 60, "eurosat": 40},
    # resnet18_actnorm: LayerNorm2d before EVERY activation call (not just
    # post-residual sites) -- inherits resnet18's budgets too.
    "resnet18_actnorm": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
                         "food101": 60, "eurosat": 40},
    # convnext: no source experiment of its own -- takes the vit100k arm's
    # budget, the same reasoning already applied to resnet18/mlp/etc, and
    # what makes the FMNIST comparison requested directly apples-to-apples
    # with vit100k and cnn100k's own FMNIST epoch counts (20 for vit, 30 for
    # cnn -- this uses vit's 20 since it's the more common default and this
    # arch is being compared to both).
    "convnext": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
                "food101": 60, "eurosat": 40},
    # mlp: same reasoning -- no source experiment of its own, so it takes the
    # ViT arm's budget. Only FMNIST is in scope for this arm (an MLP's fc1 is
    # in_features x width, so its parameter count is tied to the input size and
    # would leave the ~100K budget on any other dataset at this width).
    "mlp": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
            "food101": 60, "eurosat": 40},
    "mlp_bn": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
               "food101": 60, "eurosat": 40},
    "mlp_ln": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
               "food101": 60, "eurosat": 40},
    "conv1d_mlp_ln": {"fmnist": 20, "cifar10": 40, "cifar100": 60,
                      "food101": 60, "eurosat": 40},
}
#: Per-architecture dropout. The two ~100K arms use 0.1 (their source
#: experiments' value); the reference ResNet-18 has no dropout at all and gets
#: its regularisation from BatchNorm + weight decay + augmentation, so adding
#: some would be a departure from the published architecture. Applied
#: identically to all 11 activations within an arm either way.
DROPOUT_BY_ARCH = {"vit": 0.1, "convmlpmixer": 0.1, "resnet18": 0.0,
                   "resnet18_postln": 0.0, "resnet18_actnorm": 0.0,
                   "convnext": 0.1,
                   "mlp": 0.1, "mlp_bn": 0.1, "mlp_ln": 0.1,
                   "conv1d_mlp_ln": 0.1}
#: Filename prefix per architecture, so both arms can share a results tree.
NAME_PREFIX = {"vit": "vit100k", "convmlpmixer": "cnn100k",
               "resnet18": "resnet18", "resnet18_postln": "resnet18postln",
               "resnet18_actnorm": "resnet18actnorm",
               "convnext": "convnext100k",
               "mlp": "mlp100k",
               "mlp_bn": "mlpbn100k", "mlp_ln": "mlpln100k",
               "conv1d_mlp_ln": "c1dmlp100k"}
DATASETS = ("fmnist", "cifar10", "cifar100", "food101", "eurosat")


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    """Byte-identical to cv_train.cosine_warmup_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def run_name(dataset, act_kind, seed, arch="vit"):
    return f"{NAME_PREFIX[arch]}_{dataset}_{act_kind}_seed{seed}"


@torch.no_grad()
def evaluate(model, loader, device, lossf):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        out = model(x)
        loss = lossf(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_one(dataset, act_kind, data_root, out_dir, epochs=None, seed=0,
              batch_size=256, lr=1e-3, wd=0.05, warmup_epochs=5,
              label_smoothing=0.1, dropout=None, num_workers=4,
              embed_dim=EMBED_DIM_DEFAULT, depth=DEPTH_DEFAULT,
              num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
              use_cuda_kernel=True, device=None, log=print,
              log_every=50, arch="vit", channels=CHANNELS_DEFAULT,
              fc_hidden=FC_HIDDEN_DEFAULT, width=WIDTH_DEFAULT,
              layers=LAYERS_DEFAULT, mlp_width=MLP_WIDTH_DEFAULT,
              mlp_depth=MLP_DEPTH_DEFAULT):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs = epochs or EPOCHS_BY_ARCH[arch][dataset]
    dropout = DROPOUT_BY_ARCH[arch] if dropout is None else dropout
    set_seed(seed)

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset, data_root, batch_size=batch_size, num_workers=num_workers)

    if arch == "vit":
        model = build_vit(dataset, act_kind, embed_dim=embed_dim, depth=depth,
                          num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                          use_cuda_kernel=use_cuda_kernel).to(device)
        pc = vit_count_params(model)
        shape_str = (f"embed_dim={embed_dim} depth={depth} heads={num_heads} "
                     f"mlp_ratio={mlp_ratio}")
    elif arch == "convmlpmixer":
        model = build_convmlpmixer(dataset, act_kind, dropout=dropout,
                                   channels=channels, fc_hidden=fc_hidden,
                                   mlp_ratio=mlp_ratio,
                                   use_cuda_kernel=use_cuda_kernel).to(device)
        pc = cnn_count_params(model)
        shape_str = (f"channels={tuple(channels)} fc_hidden={fc_hidden} "
                     f"mlp_ratio={mlp_ratio}")
    elif arch == "resnet18":
        model = build_resnet18(dataset, act_kind, dropout=dropout, width=width,
                               layers=tuple(layers),
                               use_cuda_kernel=use_cuda_kernel).to(device)
        pc = resnet_count_params(model)
        shape_str = f"width={width} layers={tuple(layers)}"
    elif arch == "resnet18_postln":
        model = build_resnet18_postln(dataset, act_kind, dropout=dropout, width=width,
                                      layers=tuple(layers),
                                      use_cuda_kernel=use_cuda_kernel).to(device)
        pc = resnet_postln_count_params(model)
        shape_str = f"width={width} layers={tuple(layers)} +LayerNorm2d-post-residual"
    elif arch == "resnet18_actnorm":
        model = build_resnet18_actnorm(dataset, act_kind, dropout=dropout, width=width,
                                       layers=tuple(layers),
                                       use_cuda_kernel=use_cuda_kernel).to(device)
        pc = resnet_actnorm_count_params(model)
        shape_str = f"width={width} layers={tuple(layers)} +LayerNorm2d-before-every-act"
    elif arch == "convnext":
        model = build_convnext(dataset, act_kind, dropout=dropout,
                               channels=CONVNEXT_CHANNELS_DEFAULT,
                               blocks=CONVNEXT_BLOCKS_DEFAULT, mlp_ratio=mlp_ratio,
                               use_cuda_kernel=use_cuda_kernel).to(device)
        pc = convnext_count_params(model)
        shape_str = (f"channels={CONVNEXT_CHANNELS_DEFAULT} blocks={CONVNEXT_BLOCKS_DEFAULT} "
                    f"mlp_ratio={mlp_ratio}")
    elif arch in C1D_WIDTH_BY_ARCH:
        w = C1D_WIDTH_BY_ARCH[arch]
        model = build_conv1d_mlp(dataset, act_kind, dropout=dropout, width=w,
                                 depth=mlp_depth,
                                 use_cuda_kernel=use_cuda_kernel).to(device)
        pc = c1d_count_params(model)
        shape_str = (f"conv1d(1->100,k=9,s=4)->pool8 width={w} "
                     f"hidden_layers={mlp_depth} norm=layer")
    else:
        model = build_mlp(dataset, act_kind, dropout=dropout, width=mlp_width,
                          depth=mlp_depth, norm=MLP_NORM[arch],
                          use_cuda_kernel=use_cuda_kernel).to(device)
        pc = mlp_count_params(model)
        shape_str = (f"in_features={model.in_features} width={mlp_width} "
                     f"hidden_layers={mlp_depth} norm={MLP_NORM[arch]}")
    # The equal-capacity guarantee the ACT_KINDS comparison rests on: no
    # activation in ACT_KINDS may contribute a single learnable parameter, so
    # every variant's ViT has the identical parameter count. Asserted per run
    # rather than trusted, because it is the one way an "activation-only"
    # comparison can silently stop being one. "fact_k2_global" (Evolving
    # FAct) is the deliberate, documented exception -- see fixed_acts.py --
    # and is asserted against 5 (a0, a1, b1, a2, b2) instead of 0.
    expect_act_params = 5 if act_kind == "fact_k2_global" else 0
    assert pc["act_trainable"] == expect_act_params, (
        f"activation {act_kind!r} contributed {pc['act_trainable']} learnable "
        f"parameters, expected {expect_act_params}")

    name = run_name(dataset, act_kind, seed, arch)
    log(f"[{name}] arch={arch} device={device} params total={pc['total']:,} "
        f"trainable={pc['trainable']:,} ffn={pc['ffn_trainable']:,} "
        f"act={pc['act_trainable']} "
        f"{shape_str} epochs={epochs} bs={batch_size} lr={lr} wd={wd} "
        f"dropout={dropout} amp=off(fp32) cuda_kernel={use_cuda_kernel} "
        f"n_train={meta['n_train']} n_val={meta['n_val']} n_test={meta['n_test']}")
    log(f"[{name}] activation: {model.shared_act}")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
               "lr": [], "epoch_wall_s": []}
    best_val_acc, best_state, best_epoch = -1.0, None, -1
    global_step, lr_now = 0, lr
    t0 = time.time()

    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    run_config = dict(arch=arch, dataset=dataset, act_kind=act_kind, seed=seed, epochs=epochs,
                      batch_size=batch_size, lr=lr, weight_decay=wd,
                      warmup_epochs=warmup_epochs, label_smoothing=label_smoothing,
                      dropout=dropout, optimizer="adamw",
                      lr_scheduler="cosine_warmup", amp=False,
                      embed_dim=embed_dim, depth=depth, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, use_cuda_kernel=use_cuda_kernel,
                      channels=tuple(channels) if arch == "convmlpmixer" else None,
                      fc_hidden=fc_hidden if arch == "convmlpmixer" else None,
                      width=width if arch in ("resnet18", "resnet18_postln", "resnet18_actnorm") else None,
                      layers=tuple(layers) if arch in ("resnet18", "resnet18_postln", "resnet18_actnorm") else None,
                      mlp_width=mlp_width if arch in MLP_NORM else None,
                      mlp_depth=mlp_depth if arch in MLP_NORM else None,
                      mlp_norm=MLP_NORM.get(arch),
                      params=pc)

    for ep in range(1, epochs + 1):
        ep_t0 = time.time()
        model.train()
        ep_loss, ep_correct, ep_n = 0.0, 0, 0
        for step_in_ep, (x, y) in enumerate(train_loader, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr_now = cosine_warmup_lr(global_step, total_steps, warmup_steps, lr)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            opt.zero_grad(set_to_none=True)
            out = model(x)
            loss = lossf(out, y)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * x.size(0)
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_n += x.size(0)
            global_step += 1
            if log_every and (step_in_ep % log_every == 0 or step_in_ep == 1):
                log(f"[{name}] epoch {ep} step {step_in_ep}/{steps_per_epoch} "
                    f"loss={loss.item():.4f} ({time.time() - t0:.1f}s elapsed)")

        train_loss, train_acc = ep_loss / ep_n, ep_correct / ep_n
        val_loss, val_acc = evaluate(model, val_loader, device, lossf)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["epoch_wall_s"].append(time.time() - ep_t0)
        log(f"[{name}] epoch {ep}/{epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} lr={lr_now:.2e} ({time.time() - t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc, best_epoch = val_acc, ep
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            # Flushed to disk the moment it improves, so a crash mid-run still
            # leaves a usable best checkpoint (test_acc is filled in below).
            torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                        "epoch": ep, "config": run_config},
                       os.path.join(ckpt_dir, f"{name}_best.pt"))

    train_time_s = time.time() - t0
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss_best, test_acc_best = evaluate(model, test_loader, device, lossf)
    model.load_state_dict(final_state)
    test_loss_final, test_acc_final = evaluate(model, test_loader, device, lossf)

    torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                "epoch": best_epoch, "test_acc": test_acc_best,
                "config": run_config}, os.path.join(ckpt_dir, f"{name}_best.pt"))
    torch.save({"state_dict": final_state, "epoch": epochs,
                "test_acc": test_acc_final, "config": run_config},
               os.path.join(ckpt_dir, f"{name}_last.pt"))

    result = {
        "name": name, "arch": arch, "dataset": dataset, "act_kind": act_kind, "seed": seed,
        "epochs": epochs, "params": pc, "meta": meta, "config": run_config,
        "best_val_acc": best_val_acc, "best_epoch": best_epoch,
        "test_acc": test_acc_best, "test_loss": test_loss_best,
        "test_acc_last_epoch": test_acc_final, "test_loss_last_epoch": test_loss_final,
        "wall_s": train_time_s, "history": history,
        "host": os.uname().nodename,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, f"{name}_result.json"), "w") as f:
        json.dump(result, f, indent=2)

    log(f"[{name}] DONE test_acc(best-val)={test_acc_best:.4f} "
        f"test_acc(last)={test_acc_final:.4f} best_val_acc={best_val_acc:.4f} "
        f"@epoch{best_epoch} time={train_time_s:.1f}s")

    del train_loader, val_loader, test_loader, opt, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--arch", default="vit", choices=ARCHS)
    ap.add_argument("--act", required=True,
                    choices=list(ACT_KINDS) + ["fact_k2_global"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05,
                    help="AdamW weight decay. The common configuration uses 0.05; the "
                         "activation-specific condition searches it jointly with --lr "
                         "over {0, 0.05, 0.2}, so reproducing those runs needs this.")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--data-root", default=os.environ.get("FACT_DATA_ROOT", os.path.expanduser("~/datasets")))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM_DEFAULT)
    ap.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    ap.add_argument("--num-heads", type=int, default=NUM_HEADS_DEFAULT)
    ap.add_argument("--mlp-ratio", type=float, default=MLP_RATIO_DEFAULT)
    ap.add_argument("--channels", type=int, nargs=2, default=list(CHANNELS_DEFAULT),
                    help="convmlpmixer only: the two stage widths (c1, c2)")
    ap.add_argument("--fc-hidden", type=int, default=FC_HIDDEN_DEFAULT,
                    help="convmlpmixer only: classifier hidden width")
    ap.add_argument("--width", type=int, default=WIDTH_DEFAULT,
                    help="resnet18 only: stem width (stages are w, 2w, 4w, 8w)")
    ap.add_argument("--layers", type=int, nargs=4, default=list(LAYERS_DEFAULT),
                    help="resnet18 only: blocks per stage (2 2 2 2 = ResNet-18)")
    ap.add_argument("--mlp-width", type=int, default=MLP_WIDTH_DEFAULT,
                    help="mlp only: hidden width (100 -> 99,710 params on FMNIST)")
    ap.add_argument("--mlp-depth", type=int, default=MLP_DEPTH_DEFAULT,
                    help="mlp only: number of hidden layers")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override the per-architecture default "
                         "(vit/convmlpmixer 0.1, resnet18 0.0)")
    ap.add_argument("--no-cuda-kernel", action="store_true",
                    help="evaluate fact_fixed in pure PyTorch instead of the "
                         "custom CUDA kernel (reference path; no effect on the "
                         "ten builtin activations)")
    ap.add_argument("--log-every", type=int, default=50)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    expect = 5 if args.act == "fact_k2_global" else 0
    assert act_param_count(args.act, not args.no_cuda_kernel) == expect
    train_one(args.dataset, args.act, args.data_root, args.out_dir,
              epochs=args.epochs, seed=args.seed, batch_size=args.batch_size,
              lr=args.lr, wd=args.wd, num_workers=args.num_workers,
              embed_dim=args.embed_dim, depth=args.depth,
              num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
              use_cuda_kernel=not args.no_cuda_kernel,
              log_every=args.log_every, arch=args.arch, dropout=args.dropout,
              channels=tuple(args.channels), fc_hidden=args.fc_hidden,
              width=args.width, layers=tuple(args.layers),
              mlp_width=args.mlp_width, mlp_depth=args.mlp_depth)


if __name__ == "__main__":
    main()
