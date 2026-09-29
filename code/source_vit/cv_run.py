"""Sweep FFN variants x seeds on one CV dataset, write
results/cv_<dataset>_<host>.json. Resumable: existing (variant, seed)
skipped.

For variant=="fact_k2_global" seed==0, also snapshots the single shared
FourierActivation's coefficients after every epoch to
figures/fact_k2_global_epoch_coeffs_cv_<dataset>.json and its final
coefficients to figures/fact_k2_global_coeffs_cv_<dataset>.json -- the inputs
to the paper's coefficient-trajectory plots, captured inline via
cv_train.train_one's epoch_hook, so no separate retraining pass is needed.
"""
import argparse
import json
import os
import socket
import time

import torch

from cv_train import train_one
from fourier_layers import parse_global_k

HERE = os.path.dirname(__file__)
CV_FFN_VARIANTS = ["standard", "standard_narrow", "fact_k2", "fact_k2_narrow",
                   "fact_k2_shared", "fact_k2_global"]
# Any "fact_kK_global" or "fact_kK_global_conv" name (K = any positive int),
# not a hardcoded per-K set -- fixes a gap where fact_k1_global (and any
# future K) silently skipped epoch-coefficient logging/save_epoch_ckpts
# below because it wasn't in a hand-maintained {"fact_k2_global", ...} set.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                     choices=["fmnist", "cifar10", "cifar100", "tinyimagenet",
                              "food101"])
    ap.add_argument("--variants", nargs="+", default=CV_FFN_VARIANTS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--data-root", default=os.environ.get("FACT_DATA_ROOT", os.path.expanduser("~/datasets")))
    ap.add_argument("--out", default=None)
    ap.add_argument("--epochs", type=int, default=None,
                     help="override cv_train's per-dataset default epoch count")
    args = ap.parse_args()

    host = socket.gethostname().split(".")[0]
    out = args.out or os.environ.get(
        "CV_RESULTS_OUT", os.path.join(HERE, "results", f"cv_{args.dataset}_{host}.json"))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    os.makedirs(os.path.join(HERE, "figures"), exist_ok=True)
    results = []
    if os.path.exists(out):
        results = json.load(open(out))
    done = {(r["variant"], r["seed"]) for r in results}

    dev_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"host={host} dataset={args.dataset} out={out} device={dev_name}", flush=True)

    for v in args.variants:
        for s in args.seeds:
            if (v, s) in done:
                print(f"skip {v} seed{s} (already done)", flush=True)
                continue
            t0 = time.time()

            epoch_coeffs = []
            epoch_hook = None
            if parse_global_k(v) is not None and s == 0:
                def epoch_hook(ep, model, val_acc, val_loss):
                    act = model.shared_act
                    epoch_coeffs.append({
                        "epoch": ep, "val_acc": val_acc,
                        "a0": act.a0.detach().cpu().tolist(),
                        "a": act.a.detach().cpu().tolist(),
                        "b": act.b.detach().cpu().tolist(),
                    })

            save_epoch_ckpts = (parse_global_k(v) is not None and s == 0)
            r, model = train_one(args.dataset, v, args.data_root, HERE, seed=s,
                                 epochs=args.epochs,
                                 log=lambda *a: print(*a, flush=True),
                                 epoch_hook=epoch_hook, return_model=True,
                                 save_epoch_ckpts=save_epoch_ckpts)
            results.append(r)
            json.dump(results, open(out, "w"), indent=2)
            print(f"=== saved {v} seed{s} in {time.time()-t0:.1f}s "
                  f"(test_acc={r['test_acc']:.4f}) ===\n", flush=True)

            if parse_global_k(v) is not None and s == 0:
                act = model.shared_act
                final = {"variant": v, "dataset": args.dataset, "seed": s,
                         "test_acc": r["test_acc"], "best_val_acc": r["best_val_acc"],
                         "a0": act.a0.detach().cpu().tolist(),
                         "a": act.a.detach().cpu().tolist(),
                         "b": act.b.detach().cpu().tolist(),
                         "w": float(act.w), "K": act.K}
                coeffs_path = os.path.join(HERE, "figures",
                                           f"{v}_coeffs_cv_{args.dataset}.json")
                json.dump(final, open(coeffs_path, "w"), indent=2)
                epoch_path = os.path.join(HERE, "figures",
                                          f"{v}_epoch_coeffs_cv_{args.dataset}.json")
                json.dump({"dataset": args.dataset, "seed": s, "w": float(act.w),
                           "K": act.K, "epochs": epoch_coeffs},
                          open(epoch_path, "w"), indent=2)
                print(f"wrote {coeffs_path} and {epoch_path} ({len(epoch_coeffs)} epochs)",
                     flush=True)

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
