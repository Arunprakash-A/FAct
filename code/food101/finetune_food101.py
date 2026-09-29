"""Fine-tune the two already-trained 100-epoch ImageNet-1K checkpoints
(standard/GELU, fact_k2_global) on a small downstream CV dataset (CIFAR-10
by default, or any other cv_vit.DATASET_CFG entry via --dataset, e.g.
cifar100) for a short run, to compare convergence speed under transfer.

ImageNet-1K's 224x224/patch16 grid (197 tokens) and CIFAR-10/100's
32x32/patch4 grid (65 tokens, see cv_vit.DATASET_CFG) are architecturally
incompatible at the input/output ends, so a straight state_dict load can't
work. Instead this transplants only the pretrained TRANSFORMER BODY --
every block's attention + FFN (for fact_k2_global, including the trained
shared FourierActivation: 5 scalar coefficients, dataset-independent by
construction) and the final LayerNorm -- into
a freshly-built target-dataset-shaped model, leaving patch_embed /
pos_embed / cls_token / head at their normal random init (those have to
learn the target dataset's own input/output geometry from scratch either
way -- CIFAR-100's head is 100-way instead of CIFAR-10's 10-way, everything
else about the transplant is identical).

Reuses cv_train.train_one's existing init_state_dict mechanism (built for
resuming a run; repurposed here for a cross-dataset warm start), so the
per-epoch train/val logging, best-checkpoint tracking, and final test-acc
evaluation are byte-identical to every other run in this study --directly
comparable convergence curves between the two variants.
"""
import argparse
import json
import os

import torch

from cv_train import train_one
from cv_vit import DEPTH_DEFAULT, build_vit

# patch_embed/pos_embed/cls_token are listed here but only actually land
# when the target geometry matches the checkpoint's (food101_224 does;
# the 64px food101 does not, and build_init_state_dict's shape check
# drops them into skipped_shape instead). "head." is never transplanted:
# 1000-way -> 101-way.
TRANSPLANT_PREFIXES = ("blocks.", "shared_act.", "norm.",
                        "patch_embed.", "pos_embed", "cls_token")


def build_init_state_dict(dataset, variant, pretrained_ckpt_path, depth,
                           use_cuda_act=False):
    target = build_vit(dataset, variant, depth=depth,
                        use_cuda_act=use_cuda_act).state_dict()
    src = torch.load(pretrained_ckpt_path, map_location="cpu")["state_dict"]
    loaded, skipped_prefix, skipped_shape = [], [], []
    for k, v in src.items():
        if not any(k.startswith(p) for p in TRANSPLANT_PREFIXES):
            skipped_prefix.append(k)
            continue
        if k in target and target[k].shape == v.shape:
            target[k] = v
            loaded.append(k)
        else:
            skipped_shape.append(k)
    # fact_kK_global: every block's ffn.act IS the shared_act module BY
    # REFERENCE (see cv_vit.VisionTransformer), so .state_dict() emits its
    # a0/_a/_b under BOTH "shared_act.<suffix>" and "blocks.<i>.ffn.act.
    # <suffix>" for every block i -- these are aliases of the SAME tensor,
    # not independent copies. When depth (target) > the source checkpoint's
    # depth, the newly added blocks' "blocks.<i>.ffn.act.*" aliases above
    # were never touched by the transplant loop (no matching source key)
    # and are still target's own fresh/untrained init. Since
    # model.load_state_dict() applies state_dict entries in the model's
    # attribute-registration order -- shared_act is registered before
    # blocks in VisionTransformer.__init__, so "blocks.*" keys are applied
    # AFTER "shared_act.*" -- that stale untransplanted alias would
    # silently clobber the correctly-transplanted shared value post-load.
    # Force every "blocks.<i>.ffn.act.<suffix>" alias back in sync with its
    # (now-transplanted) "shared_act.<suffix>" counterpart so the final
    # loaded value is correct regardless of key-processing order.
    for k in list(target):
        if k.startswith("blocks.") and ".ffn.act." in k:
            shared_key = "shared_act." + k.split(".ffn.act.", 1)[1]
            if shared_key in target:
                target[k] = target[shared_key]
    info = {"loaded": loaded, "skipped_prefix": skipped_prefix, "skipped_shape": skipped_shape}
    return target, info


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10",
                     choices=["cifar10", "cifar100", "fmnist", "tinyimagenet", "food101",
                              "food101_224"])
    ap.add_argument("--variant", required=True,
                     choices=["standard", "fact_k2_global", "pau_global"])
    ap.add_argument("--pretrained-ckpt", default=None,
                     help="ImageNet-1K checkpoint to transplant the transformer body from "
                          "(the initial cross-dataset warm start). Not needed with --resume-from.")
    ap.add_argument("--resume-from", default=None,
                     help="path to THIS script's own '..._best.pt' (cifar10-shaped) to continue "
                          "training from -- --epochs is then the number of ADDITIONAL epochs, and "
                          "schedule_epochs = epoch_offset + epochs so the cosine LR schedule stays "
                          "continuous across the resume boundary, same convention as cv_train.py's "
                          "own --resume-from.")
    ap.add_argument("--data-root",
                     default=os.environ.get("FACT_DATA_ROOT",
                                            os.path.expanduser("~/datasets")),
                     help="dataset root (default: $FACT_DATA_ROOT, else ~/datasets) -- "
                          "the same convention as cv_train.py and train_static.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--depth", type=int, default=DEPTH_DEFAULT)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--use-cuda-act", action="store_true",
                     help="run the shared K=2 FAct through the fused CUDA kernel, whose "
                          "backward recomputes sin/cos from the saved input rather than "
                          "storing the cos/sin intermediates -- materially lower activation "
                          "memory, numerically equivalent to the pure-PyTorch path "
                          "(the frozen-kernel case is checked in tools/check_checkpoints.py). "
                          "No effect on variant=standard.")
    ap.add_argument("--freeze-coeffs", action="store_true",
                     help="freeze the transplanted shared FourierActivation's a0/a/b "
                          "coefficients at their ImageNet-1K-learned values for the whole "
                          "finetune (fact_k2_global only) -- isolates whether adapting the "
                          "activation's SHAPE to the new dataset matters, vs. just reusing "
                          "the pretrained shape and adapting every other weight. No effect "
                          "on variant=standard (no shared_act to freeze).")
    ap.add_argument("--linear-probe", action="store_true",
                     help="freeze EVERY transplanted tensor (patch_embed/pos_embed/"
                          "cls_token/blocks/norm, and for fact_k2_global the five shared "
                          "FourierActivation coefficients) and train only the fresh "
                          "101-way head -- a linear probe on the ImageNet-1K features, "
                          "measuring how separable the target classes already are in "
                          "the source representation rather than how well it adapts.")
    ap.add_argument("--freeze-pretrained-blocks", type=int, default=0,
                     help="freeze blocks[:N] (attention+FFN+LayerNorm) plus the final norm "
                          "at their transplanted ImageNet-1K values; only blocks[N:] (e.g. a "
                          "newly added block in a --depth deeper than the pretrained "
                          "checkpoint) plus patch_embed/pos_embed/cls_token/head stay "
                          "trainable. For fact_k2_global this also freezes the shared "
                          "FourierActivation coefficients as a side effect (block 0's "
                          "ffn.act IS shared_act by reference) -- pass --freeze-coeffs too "
                          "so run_config records that explicitly.")
    args = ap.parse_args()

    data_root = os.path.expanduser(args.data_root)
    os.makedirs(args.out_dir, exist_ok=True)
    history_path = os.path.join(
        args.out_dir, f"{args.dataset}_finetune_{args.variant}_history.json")

    info = None
    if args.resume_from:
        assert not args.pretrained_ckpt, "pass either --resume-from or --pretrained-ckpt, not both"
        ckpt = torch.load(args.resume_from, map_location="cpu")
        init_sd = ckpt["state_dict"]
        # a "_best.pt" written mid-run (train_one line ~447) has a top-level
        # "epoch" key; the one written after the run completes (line ~492)
        # is overwritten without it, but its "config" dict always carries
        # the true absolute epoch count under "epochs" -- fall back to that.
        epoch_offset = ckpt.get("epoch", ckpt.get("config", {}).get("epochs"))
        assert epoch_offset is not None, (
            f"can't determine epoch_offset from {args.resume_from}: no 'epoch' key "
            f"and no config['epochs'] either")
        init_best_state, init_best_val_acc = ckpt["state_dict"], ckpt["val_acc"]
        schedule_epochs = epoch_offset + args.epochs
        print(f"[{args.variant}] resuming from {args.resume_from}: epoch_offset={epoch_offset}, "
              f"schedule_epochs={schedule_epochs}, init_best_val_acc={init_best_val_acc:.4f}",
              flush=True)
    else:
        assert args.pretrained_ckpt, "--pretrained-ckpt required unless --resume-from is given"
        init_sd, info = build_init_state_dict(
            args.dataset, args.variant, args.pretrained_ckpt, args.depth,
            use_cuda_act=args.use_cuda_act)
        print(f"[{args.variant}] transplanted {len(info['loaded'])} tensors from the pretrained "
              f"ImageNet-1K checkpoint (transformer body); left {len(info['skipped_prefix'])} "
              f"dataset-specific tensors (patch_embed/pos_embed/cls_token/head) at random init.",
              flush=True)
        epoch_offset, schedule_epochs = 0, None
        init_best_state, init_best_val_acc = None, -1.0

    result = train_one(
        dataset=args.dataset, variant=args.variant, data_root=data_root, out_dir=args.out_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, wd=args.wd,
        warmup_epochs=args.warmup_epochs, label_smoothing=0.1, lr_scheduler="cosine_warmup",
        seed=args.seed, num_workers=args.num_workers, depth=args.depth,
        init_state_dict=init_sd, init_best_state=init_best_state,
        init_best_val_acc=init_best_val_acc, epoch_offset=epoch_offset,
        schedule_epochs=schedule_epochs, name_suffix="_finetune_imagenet1k",
        save_checkpoints=True, freeze_coeffs=args.freeze_coeffs,
        freeze_pretrained_blocks=args.freeze_pretrained_blocks,
        linear_probe=args.linear_probe,
        use_cuda_act=args.use_cuda_act,
    )
    if info is not None:
        result["transplant_counts"] = {k: len(v) for k, v in info.items()}
        result["transplanted_keys"] = info["loaded"]
        result["pretrained_ckpt"] = args.pretrained_ckpt
    if args.resume_from and os.path.exists(history_path):
        with open(history_path) as f:
            prev = json.load(f)
        result["history"] = {k: prev["history"][k] + result["history"][k]
                              for k in result["history"]}
        result["resumed_from"] = args.resume_from

    with open(history_path, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                       if k not in ("transplanted_keys", "history")}, indent=2))
    print(f"wrote {history_path}", flush=True)
