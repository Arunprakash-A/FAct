"""Train one cv_vit FFN variant on one CV dataset to convergence: AdamW +
cosine LR decay with linear warmup, AMP mixed precision, label smoothing,
standard random-crop+flip augmentation. Identical recipe (same
epochs/lr/schedule/augmentation/batch size) to the earlier convolutional-FFN
baseline study, so results are directly comparable to its existing
"standard"/"conv_gelu" numbers -- only the model
(cv_vit.build_vit, FFN swapped via fourier_ffn.build_ffn) differs.

Beyond that recipe, this one also supports:
  * epoch_hook(ep, model, val_acc) -- called after every epoch, used by
    cv_run.py's fact_k2_global seed-0 run to snapshot the single shared
    activation's coefficients for the training-evolution gif, without a
    separate retraining pass.
  * return_model=True -- return the trained model alongside the result dict.
  * Saves BOTH best-val and final-epoch checkpoints for every run (matching
    the baseline study's convention), for every variant, not just
    fact_k2_global.
"""
import gc
import json
import math
import os
import time

import torch
import torch.multiprocessing
import torch.nn as nn

torch.multiprocessing.set_sharing_strategy("file_system")

from cv_data import get_dataloaders
from cv_vit import (DEPTH_DEFAULT, EMBED_DIM_DEFAULT, NUM_HEADS_DEFAULT,
                    MLP_RATIO_DEFAULT, build_vit, count_params)

EPOCHS = {"fmnist": 20, "cifar10": 40, "cifar100": 60, "tinyimagenet": 120,
          "imagenet1k": 30,
          # food101: 75,750 train images (between cifar100's 100K and
          # tinyimagenet's 100K, but 101 fine-grained classes vs cifar100's
          # 100 coarse ones) -- matches cifar100's epoch budget, and the
          # only prior Food-101 ViT run also used 60.
          "food101": 60, "food101_224": 50}


def set_seed(s):
    import random
    import numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def run_name(dataset, variant, seed=0):
    base = f"cv_{dataset}_{variant}"
    return base if seed == 0 else f"{base}_seed{seed}"


def cosine_warmup_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cos = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * cos)


def linear_warmup_linear_decay_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(1, total_steps - warmup_steps), 1.0)
    return base_lr * (1.0 - (1.0 - min_lr_ratio) * progress)


def linear_warmup_step_decay_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01,
                                 milestones_frac=(1 / 3, 2 / 3), gamma=0.1):
    """Same linear warmup as every other scheduler here, then a piecewise-
    constant LR that drops by `gamma` at each fraction of the POST-warmup
    schedule in `milestones_frac` -- e.g. the default (1/3, 2/3) drops once
    a third of the way through the remaining epochs and again two-thirds of
    the way through. Unlike the continuous schedules, this one has sharp
    discontinuities, so convergence curves are expected to show visible
    kinks right at the drop points."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    post_total = max(1, total_steps - warmup_steps)
    post_progress = min((step - warmup_steps) / post_total, 1.0)
    lr = base_lr
    for m in milestones_frac:
        if post_progress >= m:
            lr *= gamma
    return max(lr, base_lr * min_lr_ratio)


def linear_warmup_constant_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01):
    """Linear warmup, then held flat at base_lr for the rest of training --
    no decay at all. min_lr_ratio is accepted (for a uniform call signature
    across every LR_SCHEDULERS entry) but unused."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    return base_lr


def onecycle_lr(step, total_steps, warmup_steps, base_lr, min_lr_ratio=0.01,
                 final_frac=0.1, final_div_factor=100.0):
    """Warms up to base_lr over warmup_steps (same warmup convention as
    every other scheduler here), cosine-cruises down to base_lr/10 over most
    of the remaining schedule, then linearly anneals further down to
    base_lr/final_div_factor over a short final `final_frac` of the
    remaining steps -- the two-stage "cruise then annihilate" shape from
    Smith's 1cycle policy, distinct from cosine_warmup's single smooth
    cosine arc spanning the whole post-warmup schedule down to only
    min_lr_ratio * base_lr."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    post_total = max(1, total_steps - warmup_steps)
    post_step = step - warmup_steps
    cruise_total = post_total * (1 - final_frac)
    if post_step < cruise_total:
        progress = post_step / max(1, cruise_total)
        cos = 0.5 * (1 + math.cos(math.pi * progress))
        lo = base_lr * 0.1
        return lo + (base_lr - lo) * cos
    else:
        progress = min((post_step - cruise_total) / max(1, post_total - cruise_total), 1.0)
        lo_start = base_lr * 0.1
        lo_end = base_lr / final_div_factor
        return lo_start + (lo_end - lo_start) * progress


LR_SCHEDULERS = {
    "cosine_warmup": cosine_warmup_lr,
    "linear_warmup_linear_decay": linear_warmup_linear_decay_lr,
    "linear_warmup_step_decay": linear_warmup_step_decay_lr,
    "linear_warmup_constant": linear_warmup_constant_lr,
    "onecycle": onecycle_lr,
}


@torch.no_grad()
def evaluate(model, loader, device, lossf):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    use_cuda = device.type == "cuda"
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_cuda):
            out = model(x)
            loss = lossf(out, y)
        total_loss += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total_loss / n, correct / n


def train_one(dataset, variant, data_root, out_dir, epochs=None, batch_size=256,
              lr=1e-3, wd=0.05, warmup_epochs=5, label_smoothing=0.1,
              lr_scheduler="cosine_warmup",
              seed=0, num_workers=8, device=None, log=print,
              epoch_hook=None, return_model=False, save_checkpoints=True,
              save_epoch_ckpts=False, optimizer="adamw", momentum=0.9,
              grad_clip=None, name_suffix="",
              early_stop_patience=None, early_stop_min_delta=1e-4,
              on_train_start=None, coeff_reg_lambda=0.0, coeff_reg_type="l2",
              depth=DEPTH_DEFAULT, embed_dim=EMBED_DIM_DEFAULT,
              num_heads=NUM_HEADS_DEFAULT, mlp_ratio=MLP_RATIO_DEFAULT,
              freeze_linear=False, freeze_coeffs=False, freeze_pretrained_blocks=0, linear_probe=False,
              dropout=0.1, act_ref="gelu",
              act_init="true", act_init_scale=1.0, anneal_harmonics=False,
              anneal_rate=0.5, ntied_zero_init_weights=False,
              ffn_zero_init_weights=False, ntied_rank=1, ntied_const_init=None,
              ntied_const_init_all=False, use_cuda_act=False,
              init_state_dict=None, init_best_state=None, init_best_val_acc=-1.0,
              epoch_offset=0, schedule_epochs=None):
    """lr_scheduler: which per-step LR schedule to use after the linear
    warmup (warmup_epochs, identical across every choice below so warmup
    itself is never the variable being compared) -- one of
    LR_SCHEDULERS.keys():
      * "cosine_warmup" (default, matches every prior run in this study):
        smooth single cosine arc down to lr * 0.01.
      * "linear_warmup_linear_decay": straight-line decay to lr * 0.01
        instead of a cosine arc.
      * "linear_warmup_step_decay": piecewise-constant, dropping by 10x at
        1/3 and 2/3 of the post-warmup schedule -- has sharp discontinuities.
      * "linear_warmup_constant": no decay at all after warmup.
      * "onecycle": cosine-cruise to lr/10 over most of the schedule, then a
        short final linear anneal down to lr/100 (Smith's 1cycle shape).
    Raises KeyError immediately (via LR_SCHEDULERS[lr_scheduler]) if given
    an unrecognized name, before any training happens.

    depth: number of Transformer blocks (default DEPTH_DEFAULT=6, matching
    every prior run in this study). Callers comparing a global-activation
    variant at reduced depth (fewer, wider-effect layers) against the
    standard-depth baseline pass this explicitly per call.

    embed_dim / num_heads / mlp_ratio: passed straight through to build_vit,
    default to this study's usual ViT-Ti/16-scale values (192/6/4.0). A
    caller comparing at ViT-Base/16 scale (embed_dim=768, depth=12,
    num_heads=12, mlp_ratio=4.0, per Dosovitskiy et al. 2020) passes these
    explicitly -- d_ff (the FFN hidden width) is int(embed_dim * mlp_ratio)
    either way, computed inside build_vit/VisionTransformer.

    init_state_dict: if given, model weights are loaded from this state_dict
    right after construction, to CONTINUE a previous run (e.g. 10 more epochs
    on top of an already-trained checkpoint) instead of training from scratch.

    epoch_offset / schedule_epochs: when resuming, `epochs` is the number of
    ADDITIONAL epochs to run in THIS call, while schedule_epochs is the total
    intended run length (normally epoch_offset + epochs) that the
    cosine-warmup LR schedule is computed against, and epoch_offset is how
    many epochs were already completed in the prior call being resumed --
    together these let the LR schedule and the epoch numbers used in
    logs/history/checkpoint filenames continue seamlessly across the resume
    boundary (e.g. epoch 11..20 of one 20-epoch cosine schedule) instead of
    restarting a fresh warmup+cosine cycle. schedule_epochs defaults to
    `epochs` (i.e. no resume) if not given.

    init_best_val_acc / init_best_state: the best-val accuracy/weights carried
    over from the run being resumed, so the "_best.pt" checkpoint this call
    writes stays the best across the FULL training run, not just this call's
    epochs -- without these, resuming would let this call's first-epoch
    val_acc silently overwrite a genuinely better earlier checkpoint.

    optimizer: "adamw" (default, matches every prior run in this study) or
    "sgd" (SGD+momentum, optionally Nesterov via momentum>0 -- see caller).
    grad_clip: if not None, clip gradient global-norm to this value before
    the optimizer step (unscaling first under AMP) -- SGD-from-scratch on a
    from-scratch ViT (no conv stem / no adaptive per-parameter LR) is more
    prone to occasional loss spikes than AdamW, so callers using
    optimizer="sgd" should normally pass this.

    coeff_reg_lambda: if > 0 and the model has a shared FourierActivation
    (model.shared_act is not None, i.e. any fact_kK_global variant), an L2
    penalty lambda * sum(a_k^2 + b_k^2) over the K *harmonic* coefficients
    (a0, the series' DC/mean term, is deliberately excluded -- penalizing it
    would bias the learned activation's mean away from GELU's rather than
    just capping its oscillation amplitude) is added to the training loss
    every step. This directly targets "large coefficients" -- the harmonic
    amplitudes that both control how far outside [-pi, pi]-fitted territory
    the series can swing and, per the unregularized SGD run, were implicated
    in the K=3/K=2 divergences this study's K-backoff had to route around.
    No-op for "standard" (model.shared_act is None) or when 0.0 (default).
    coeff_reg_type: "l2" (default, sum of squares -- the above) or "l1" (sum
    of absolute values, sum(|a_k| + |b_k|)) -- L1 pushes small harmonic
    coefficients toward exactly zero (sparsity) rather than just shrinking
    all of them, so a high-K series regularized this way can end up
    effectively using far fewer than K harmonics.

    dropout: passed straight through to build_vit -- applied to the FFN's
    internal dropout, each Block's attention-output projection dropout, and
    the patch/pos-embedding dropout (see cv_vit.build_vit/VisionTransformer).
    Default 0.1 matches every prior run in this study; a caller raising this
    (e.g. 0.5) is trading capacity for regularization against overfitting.

    act_ref: which reference activation the shared FourierActivation's K
    harmonic coefficients are initialised from (fourier_layers.REFERENCE_ACTS:
    "gelu" default, matching every prior run; "leaky_relu"; or "linear" --
    the truncated Fourier series of f(t)=t, i.e. every neuron starts as a
    straight-line approximation instead of a copy of GELU). No-op for
    "standard" (model.shared_act is None).

    freeze_linear: if True, every nn.Linear module's weight and bias is
    frozen (requires_grad=False) immediately after construction, at its
    random init -- so only non-Linear parameters (the shared FourierActivation
    coefficients, LayerNorm affine params, the patch-embed Conv2d, cls_token,
    pos_embed) receive gradient updates. Isolates how much a global
    activation function alone can adapt the network when every Linear
    projection (qkv/attn-proj/FFN fc1+fc2/head) is stuck at its
    initialization. The optimizer is built over only the still-trainable
    parameters, so frozen Linear weights get no state/momentum either.

    freeze_coeffs: if True and the model has a shared FourierActivation
    (model.shared_act is not None, i.e. any fact_kK_global variant), its
    a0/a/b coefficients are frozen (requires_grad_(False)) right after
    construction, at their init values -- so the K-term Fourier series stays
    fixed at its true-coefficient (or whatever act_init gives) starting
    shape for the entire run, and every other parameter (the FFN's fc1/fc2,
    attention, LayerNorm, embeddings, head) still trains normally. This is
    the opposite ablation from freeze_linear: isolates whether *learning*
    the shared activation's coefficients matters, vs. just having a fixed
    (but still initially GELU-shaped, since act_init defaults to "true")
    Fourier-series activation in place of a learned one. No-op for
    "standard" (model.shared_act is None). coeff_reg_lambda is forced to 0
    when frozen (an L2/L1 penalty on coefficients with no gradient is
    meaningless). Same naming/behaviour convention as the companion CNN
    studies' `freeze_coeffs`.

    freeze_pretrained_blocks: if > 0, freezes every parameter of
    model.blocks[:freeze_pretrained_blocks] (attention + FFN, i.e. every
    Linear AND LayerNorm inside those blocks) plus the final model.norm,
    right after construction (so at whatever values init_state_dict loaded
    them to, e.g. a transplanted pretrained checkpoint -- see
    ../food101/finetune_food101.py). For a fact_kK_global variant, since every block's
    ffn.act is the SAME shared_act module by reference, freezing blocks
    0..freeze_pretrained_blocks-1 already freezes the shared FourierActivation
    coefficients too (pass freeze_coeffs=True as well purely so run_config
    records the intent explicitly; it's otherwise redundant here). Blocks
    from index freeze_pretrained_blocks onward (e.g. a newly-added block in a
    deeper target model that has no pretrained counterpart), plus
    patch_embed/pos_embed/cls_token/head (never covered by this flag -- they
    have no pretrained values to freeze at and must learn the new dataset's
    input/output geometry regardless), stay trainable. Meant for "grow the
    depth by N and train only the new block(s)" experiments: pair with
    build_init_state_dict's transplant so the frozen blocks sit at their
    pretrained weights, not a random init that would then never move.

    epoch_hook(ep, model, val_acc, val_loss), if given, is called after every
    epoch; if it returns a truthy value, training stops early (e.g. a caller
    backing off a fact_kK_global run's K when val_loss keeps rising).

    early_stop_patience: if not None, stop training once val_loss has failed
    to improve by >= early_stop_min_delta for this many consecutive epochs
    (classic patience-based early stopping on the monitored validation
    metric) -- independent of epoch_hook, so a run can be cut short either
    because it plateaued (early stopping) or because a caller's epoch_hook
    detected instability; `result["stop_reason"]` distinguishes the two.
    Disabled (None) by default so existing callers are unaffected; the
    best-val checkpoint is unaffected either way since it is tracked/saved
    every epoch regardless of how/when the loop ends.

    name_suffix is appended to the run name used for saved files (not the
    `variant` field recorded in results/config), so e.g. an SGD run of the
    same variant can be saved alongside an AdamW run without colliding.

    on_train_start(model), if given, is called once right after the model is
    built (before epoch 1) -- for callers that need to attach instrumentation
    (e.g. a forward hook) before any training step runs, which epoch_hook
    (first called only after epoch 1 finishes) cannot do.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if lr_scheduler not in LR_SCHEDULERS:
        raise ValueError(f"unknown lr_scheduler: {lr_scheduler!r}, "
                          f"expected one of {sorted(LR_SCHEDULERS)}")
    set_seed(seed)
    epochs = epochs or EPOCHS[dataset]

    train_loader, val_loader, test_loader, meta = get_dataloaders(
        dataset, data_root, batch_size=batch_size, num_workers=num_workers)

    model = build_vit(dataset, variant, depth=depth, embed_dim=embed_dim,
                      num_heads=num_heads, mlp_ratio=mlp_ratio, dropout=dropout,
                      act_ref=act_ref, act_init=act_init,
                      act_init_scale=act_init_scale,
                      anneal_harmonics=anneal_harmonics,
                      anneal_rate=anneal_rate,
                      ntied_zero_init_weights=ntied_zero_init_weights,
                      ffn_zero_init_weights=ffn_zero_init_weights,
                      ntied_rank=ntied_rank, ntied_const_init=ntied_const_init,
                      ntied_const_init_all=ntied_const_init_all,
                      use_cuda_act=use_cuda_act).to(device)
    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    if freeze_linear:
        n_frozen = 0
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.weight.requires_grad_(False)
                if m.bias is not None:
                    m.bias.requires_grad_(False)
                n_frozen += 1

    if freeze_coeffs and model.shared_act is not None:
        for p in (model.shared_act.a0, model.shared_act.a, model.shared_act.b):
            p.requires_grad_(False)
        coeff_reg_lambda = 0.0

    if freeze_pretrained_blocks > 0:
        assert freeze_pretrained_blocks <= len(model.blocks), (
            f"freeze_pretrained_blocks={freeze_pretrained_blocks} > depth={len(model.blocks)}")
        for blk in model.blocks[:freeze_pretrained_blocks]:
            for p in blk.parameters():
                p.requires_grad_(False)
        for p in model.norm.parameters():
            p.requires_grad_(False)

    # linear_probe: treat the transplanted network as a FIXED feature
    # extractor and train ONLY the freshly-initialized head. Freezes every
    # parameter not under "head." -- patch_embed/pos_embed/cls_token, all
    # blocks, the final norm, and (for fact_kK_global) the shared
    # FourierActivation's a0/a/b, which therefore stay at exactly their
    # ImageNet-1K-learned values. Strictly stronger than
    # freeze_pretrained_blocks=depth, which leaves the input stem trainable.
    # Nothing upstream of head requires grad, so autograd never walks back
    # into the body: no activation storage, no backward through the blocks.
    n_probe_frozen = 0
    if linear_probe:
        for pname, p in model.named_parameters():
            if not pname.startswith("head."):
                p.requires_grad_(False)
                n_probe_frozen += 1
        coeff_reg_lambda = 0.0
        assert any(p.requires_grad for p in model.parameters()), \
            "linear_probe froze every parameter -- no head.* found"

    pc = count_params(model)
    name = run_name(dataset, variant, seed) + name_suffix
    log(f"[{name}] depth={depth} params total={pc['total']:,} trainable={pc['trainable']:,} "
        f"ffn_trainable={pc['ffn_trainable']:,} n_train={meta['n_train']} "
        f"n_classes={meta['n_classes']} dropout={dropout} act_ref={act_ref} act_init={act_init} "
        f"act_init_scale={act_init_scale} anneal_harmonics={anneal_harmonics} "
        f"anneal_rate={anneal_rate} ntied_zero_init_weights={ntied_zero_init_weights} "
        f"ffn_zero_init_weights={ffn_zero_init_weights} ntied_rank={ntied_rank} "
        f"ntied_const_init={ntied_const_init} ntied_const_init_all={ntied_const_init_all}"
        + (f" freeze_linear=True (froze {n_frozen} nn.Linear modules)" if freeze_linear else "")
        + (" freeze_coeffs=True (shared FourierActivation a0/a/b frozen at init)"
           if freeze_coeffs and model.shared_act is not None else "")
        + (f" freeze_pretrained_blocks={freeze_pretrained_blocks} (froze blocks[0:"
           f"{freeze_pretrained_blocks}] + final norm; blocks[{freeze_pretrained_blocks}:"
           f"{depth}] + patch_embed/pos_embed/cls_token/head stay trainable)"
           if freeze_pretrained_blocks > 0 else "")
        + (f" linear_probe=True (froze {n_probe_frozen} non-head tensors; only "
           f"head.* trainable = {pc['trainable']:,} params)" if linear_probe else ""))
    if on_train_start is not None:
        on_train_start(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer == "adamw":
        opt = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=wd)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(trainable_params, lr=lr, momentum=momentum,
                              nesterov=momentum > 0, weight_decay=wd)
    else:
        raise ValueError(f"unknown optimizer: {optimizer}")
    lossf = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    schedule_epochs = schedule_epochs or epochs
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * schedule_epochs
    warmup_steps = steps_per_epoch * warmup_epochs

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": [],
               "coeff_reg_loss": []}
    best_val_acc, best_state = init_best_val_acc, init_best_state
    global_step, lr_now = steps_per_epoch * epoch_offset, lr
    t0 = time.time()

    epoch_ckpt_dir = None
    if save_epoch_ckpts:
        epoch_ckpt_dir = os.path.join(out_dir, "checkpoints", f"{name}_epochs")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)

    reg_active = coeff_reg_lambda > 0 and model.shared_act is not None
    run_config = dict(dataset=dataset, variant=variant, seed=seed,
                       epoch_offset=epoch_offset, schedule_epochs=schedule_epochs,
                       depth=depth, batch_size=batch_size, lr=lr, weight_decay=wd,
                       warmup_epochs=warmup_epochs, label_smoothing=label_smoothing,
                       optimizer=optimizer, lr_scheduler=lr_scheduler,
                       coeff_reg_lambda=coeff_reg_lambda,
                       coeff_reg_type=coeff_reg_type, freeze_linear=freeze_linear,
                       freeze_coeffs=freeze_coeffs, freeze_pretrained_blocks=freeze_pretrained_blocks,
                       linear_probe=linear_probe,
                       dropout=dropout, act_ref=act_ref, act_init=act_init,
                       act_init_scale=act_init_scale, anneal_harmonics=anneal_harmonics,
                       anneal_rate=anneal_rate, ntied_zero_init_weights=ntied_zero_init_weights,
                       ffn_zero_init_weights=ffn_zero_init_weights, ntied_rank=ntied_rank,
                       ntied_const_init=ntied_const_init, ntied_const_init_all=ntied_const_init_all,
                       use_cuda_act=use_cuda_act,
                       **({"momentum": momentum, "grad_clip": grad_clip}
                          if optimizer == "sgd" else {}))

    ckpt_dir = None
    if save_checkpoints:
        ckpt_dir = os.path.join(out_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

    stopped_early = False
    stop_reason = None
    es_best_val_loss, es_counter = float("inf"), 0
    for ep in range(epochs):
        abs_ep = epoch_offset + ep + 1
        model.train()
        ep_loss, ep_correct, ep_n, ep_reg = 0.0, 0, 0, 0.0
        step_in_ep = 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            lr_now = LR_SCHEDULERS[lr_scheduler](global_step, total_steps, warmup_steps, lr)
            for pg in opt.param_groups:
                pg["lr"] = lr_now
            if model.shared_act is not None and getattr(model.shared_act, "anneal_harmonics", False):
                model.shared_act.set_progress(global_step / total_steps)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(x)
                loss = lossf(out, y)
                if reg_active:
                    act = model.shared_act
                    if coeff_reg_type == "l1":
                        reg = coeff_reg_lambda * (act.a.abs().sum() + act.b.abs().sum())
                    else:
                        reg = coeff_reg_lambda * (act.a.pow(2).sum() + act.b.pow(2).sum())
                    loss = loss + reg
                    ep_reg += reg.item() * x.size(0)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            ep_loss += loss.item() * x.size(0)
            ep_correct += (out.argmax(1) == y).sum().item()
            ep_n += x.size(0)
            global_step += 1
            step_in_ep += 1
            if step_in_ep % 10 == 0 or step_in_ep == 1:
                log(f"[{name}] epoch {abs_ep} step {step_in_ep}/{steps_per_epoch} "
                    f"loss={loss.item():.4f} ({time.time()-t0:.1f}s elapsed)")

        train_loss, train_acc = ep_loss / ep_n, ep_correct / ep_n
        reg_loss = ep_reg / ep_n if reg_active else 0.0
        val_loss, val_acc = evaluate(model, val_loader, device, lossf)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr_now)
        history["coeff_reg_loss"].append(reg_loss)
        reg_str = f" reg_loss={reg_loss:.4e}" if reg_active else ""
        log(f"[{name}] epoch {abs_ep}/{schedule_epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
            f"lr={lr_now:.2e}{reg_str} ({time.time()-t0:.1f}s)")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if ckpt_dir is not None:
                # Written to disk immediately (not only after the full loop
                # finishes) so an unexpected failure mid-run (OOM, node
                # preemption, ...) still leaves a usable best checkpoint --
                # "test_acc" is filled in below once training completes.
                torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                            "epoch": abs_ep, "config": run_config},
                           os.path.join(ckpt_dir, f"{name}_best.pt"))

        stop_signal = None
        if epoch_hook is not None:
            stop_signal = epoch_hook(abs_ep, model, val_acc, val_loss)

        if epoch_ckpt_dir is not None:
            torch.save({"state_dict": model.state_dict(), "epoch": abs_ep,
                        "val_acc": val_acc, "train_acc": train_acc},
                       os.path.join(epoch_ckpt_dir, f"epoch{abs_ep:03d}.pt"))

        if stop_signal:
            stopped_early, stop_reason = True, f"epoch_hook: {stop_signal}"
            log(f"[{name}] epoch_hook signalled early stop after epoch {abs_ep}: "
                f"{stop_signal}")
            break

        if early_stop_patience is not None:
            if val_loss < es_best_val_loss - early_stop_min_delta:
                es_best_val_loss, es_counter = val_loss, 0
            else:
                es_counter += 1
                if es_counter >= early_stop_patience:
                    stopped_early, stop_reason = True, "early_stopping"
                    log(f"[{name}] early stopping after epoch {abs_ep}: val_loss hasn't "
                        f"improved by >= {early_stop_min_delta} for "
                        f"{early_stop_patience} epochs (best={es_best_val_loss:.4f})")
                    break

    actual_epochs = len(history["train_loss"])
    run_config["epochs_this_call"] = actual_epochs
    run_config["epochs"] = epoch_offset + actual_epochs
    run_config["stopped_early"] = stopped_early
    run_config["stop_reason"] = stop_reason

    train_time_s = time.time() - t0
    final_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    test_loss_best, test_acc_best = evaluate(model, test_loader, device, lossf)
    model.load_state_dict(final_state)
    test_loss_final, test_acc_final = evaluate(model, test_loader, device, lossf)

    if save_checkpoints:
        torch.save({"state_dict": best_state, "val_acc": best_val_acc,
                    "test_acc": test_acc_best, "config": run_config},
                   os.path.join(ckpt_dir, f"{name}_best.pt"))
        torch.save({"state_dict": final_state, "test_acc": test_acc_final,
                    "config": run_config},
                   os.path.join(ckpt_dir, f"{name}_final.pt"))

    result = {
        "variant": variant, "seed": seed, "epochs": epoch_offset + actual_epochs,
        "epochs_this_call": actual_epochs, "epoch_offset": epoch_offset,
        "dataset": dataset, "depth": depth, "coeff_reg_lambda": coeff_reg_lambda,
        "coeff_reg_type": coeff_reg_type, "freeze_linear": freeze_linear,
        "freeze_coeffs": freeze_coeffs, "dropout": dropout,
        "act_ref": act_ref, "ffn_zero_init_weights": ffn_zero_init_weights,
        "batch_size": batch_size, "lr": lr, "weight_decay": wd,
        "warmup_epochs": warmup_epochs, "label_smoothing": label_smoothing,
        "optimizer": optimizer, "lr_scheduler": lr_scheduler,
        "stopped_early": stopped_early, "stop_reason": stop_reason,
        "meta": meta, "params": pc,
        "best_val_acc": best_val_acc,
        "test_acc": test_acc_best, "test_loss": test_loss_best,
        "test_acc_final_epoch": test_acc_final, "test_loss_final_epoch": test_loss_final,
        "history": history, "wall_s": train_time_s,
    }
    log(f"[{name}] DONE test_acc(best-val-ckpt)={test_acc_best:.4f} "
        f"best_val_acc={best_val_acc:.4f} time={train_time_s:.1f}s")

    model.load_state_dict(best_state)  # leave caller holding the best-val model
    del train_loader, val_loader, test_loader, opt
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return (result, model) if return_model else result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["fmnist", "cifar10", "cifar100", "tinyimagenet",
                              "imagenet1k", "food101", "food101_224"])
    ap.add_argument("--variant", default="standard")
    ap.add_argument("--data-root", default=os.environ.get("FACT_DATA_ROOT", os.path.expanduser("~/datasets")))
    ap.add_argument("--out-dir", default=os.path.dirname(__file__))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    ap.add_argument("--embed-dim", type=int, default=EMBED_DIM_DEFAULT)
    ap.add_argument("--num-heads", type=int, default=NUM_HEADS_DEFAULT)
    ap.add_argument("--mlp-ratio", type=float, default=MLP_RATIO_DEFAULT)
    ap.add_argument("--coeff-reg", type=float, default=0.0)
    ap.add_argument("--coeff-reg-type", choices=["l1", "l2"], default="l2")
    ap.add_argument("--freeze-linear", action="store_true")
    ap.add_argument("--freeze-coeffs", action="store_true",
                     help="freeze the shared FourierActivation's a0/a/b coefficients "
                          "at their init values for the whole run (no-op for "
                          "variant=standard); opposite ablation from --freeze-linear")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--act-ref", choices=["gelu", "gelu_exp", "leaky_relu", "linear"], default="gelu")
    ap.add_argument("--ffn-zero-init-weights", action="store_true")
    ap.add_argument("--lr-scheduler", choices=sorted(LR_SCHEDULERS), default="cosine_warmup")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--act-init", choices=["true", "random", "random_learnable_amp"], default="true")
    ap.add_argument("--act-init-scale", type=float, default=1.0)
    ap.add_argument("--use-cuda-act", action="store_true",
                     help="use the custom CUDA kernel (cuda_fact_k2) instead of the "
                          "pure-PyTorch FourierActivation for any K=2 globally-shared "
                          "activation (fact_k2_global, fact_k2_global_2act); no-op "
                          "for variants without a shared K=2 FourierActivation")
    ap.add_argument("--save-epoch-ckpts", action="store_true")
    ap.add_argument("--resume-from", default=None,
                     help="path to a per-epoch checkpoint (e.g. .../epoch037.pt) to "
                          "continue training from; --epochs is then the number of "
                          "ADDITIONAL epochs to run, and schedule_epochs is set to "
                          "epoch_offset + epochs so the cosine LR schedule stays "
                          "continuous across the resume boundary")
    args = ap.parse_args()

    init_state_dict = init_best_state = None
    init_best_val_acc = -1.0
    epoch_offset = 0
    schedule_epochs = args.epochs
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location="cpu")
        init_state_dict = ckpt["state_dict"]
        epoch_offset = ckpt["epoch"]
        ckpt_dir = os.path.dirname(args.resume_from)
        if os.path.basename(ckpt_dir).endswith("_epochs"):
            ckpt_dir = os.path.dirname(ckpt_dir)
        best_path = os.path.join(ckpt_dir, f"{run_name(args.dataset, args.variant, args.seed)}_best.pt")
        if os.path.exists(best_path):
            best_ckpt = torch.load(best_path, map_location="cpu")
            init_best_state, init_best_val_acc = best_ckpt["state_dict"], best_ckpt["val_acc"]
        else:
            init_best_state, init_best_val_acc = init_state_dict, ckpt.get("val_acc", -1.0)
        schedule_epochs = epoch_offset + args.epochs
        print(f"resuming from {args.resume_from}: epoch_offset={epoch_offset}, "
              f"schedule_epochs={schedule_epochs}, init_best_val_acc={init_best_val_acc:.4f}")

    r = train_one(args.dataset, args.variant, args.data_root, args.out_dir,
                  epochs=args.epochs, seed=args.seed, depth=args.depth,
                  embed_dim=args.embed_dim, num_heads=args.num_heads,
                  mlp_ratio=args.mlp_ratio,
                  coeff_reg_lambda=args.coeff_reg, coeff_reg_type=args.coeff_reg_type,
                  freeze_linear=args.freeze_linear, freeze_coeffs=args.freeze_coeffs,
                  dropout=args.dropout,
                  act_ref=args.act_ref, lr_scheduler=args.lr_scheduler,
                  ffn_zero_init_weights=args.ffn_zero_init_weights,
                  num_workers=args.num_workers, batch_size=args.batch_size,
                  act_init=args.act_init, act_init_scale=args.act_init_scale,
                  use_cuda_act=args.use_cuda_act,
                  init_state_dict=init_state_dict, init_best_state=init_best_state,
                  init_best_val_acc=init_best_val_acc, epoch_offset=epoch_offset,
                  schedule_epochs=schedule_epochs,
                  save_epoch_ckpts=args.save_epoch_ckpts)
    print(json.dumps({k: v for k, v in r.items() if k != "history"}, indent=2))
