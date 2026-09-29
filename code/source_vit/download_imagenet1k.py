"""Download the ImageNet-1K (ILSVRC/imagenet-1k) train+validation parquet
shards from HuggingFace into <data-root>/hf/imagenet1k, the layout cv_data.py
expects (and the same <data-root>/hf layout used for CIFAR-10/100 and
Tiny-ImageNet).

<data-root> defaults to $FACT_DATA_ROOT, else ~/datasets -- the same default
cv_train.py's --data-root uses. Pass --data-root to override.

The test split (13.6GB, unlabeled) is skipped: it has no public labels and is
not used for training or eval, which is why cv_data.py reports top-1 on the
validation split. README.md and classes.py (label names) are pulled too, since
the loaders need the class list.

Expect ~150GB and several hours. The download resumes from .incomplete files,
so re-running after an interruption is safe.

ILSVRC/imagenet-1k is a GATED dataset: you must accept its terms on the
HuggingFace page and be logged in (`hf auth login`, or set HF_TOKEN) before
this will fetch anything. A permission failure is not retried -- it is raised
straight away with that instruction, since no amount of waiting fixes it.
"""
import argparse
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # xet backend hangs on concurrent downloads

import time

from huggingface_hub import snapshot_download

try:                                  # huggingface_hub >= 0.23
    from huggingface_hub.errors import (EntryNotFoundError, GatedRepoError,
                                        RepositoryNotFoundError)
except ImportError:                   # older layout
    from huggingface_hub.utils import (EntryNotFoundError, GatedRepoError,
                                       RepositoryNotFoundError)

#: Failures that no amount of retrying will fix -- almost always "you have not
#: accepted the dataset's terms" or "you are not logged in".
PERMANENT_ERRORS = (GatedRepoError, RepositoryNotFoundError, EntryNotFoundError)

DEFAULT_DATA_ROOT = os.environ.get("FACT_DATA_ROOT", os.path.expanduser("~/datasets"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                    help="dataset root; shards land in <data-root>/hf/imagenet1k "
                         f"(default: {DEFAULT_DATA_ROOT})")
    ap.add_argument("--max-workers", type=int, default=8)
    ap.add_argument("--max-attempts", type=int, default=20,
                    help="give up after this many transient failures (default: 20)")
    args = ap.parse_args()

    local_dir = os.path.join(args.data_root, "hf", "imagenet1k")
    print(f"downloading ImageNet-1K parquet shards -> {local_dir}")

    t0 = time.time()
    path = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            path = snapshot_download(
                repo_id="ILSVRC/imagenet-1k",
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=["data/train-*.parquet", "data/validation-*.parquet",
                                "README.md", "classes.py"],
                max_workers=args.max_workers,
            )
            break
        except PERMANENT_ERRORS as e:
            raise SystemExit(
                f"{type(e).__name__}: {e}\n\n"
                "ILSVRC/imagenet-1k is gated. Accept its terms at\n"
                "  https://huggingface.co/datasets/ILSVRC/imagenet-1k\n"
                "then authenticate with `hf auth login` (or set HF_TOKEN) and\n"
                "re-run. Retrying will not help.")
        except Exception as e:
            if attempt == args.max_attempts:
                raise SystemExit(
                    f"giving up after {attempt} attempts; last error was "
                    f"{type(e).__name__}: {e}\n"
                    "Partial shards are kept as .incomplete files, so re-running "
                    "resumes where this left off.")
            print(f"attempt {attempt}/{args.max_attempts} failed "
                  f"({type(e).__name__}: {e}); resuming from .incomplete files in 10s")
            time.sleep(10)

    print(f"done in {(time.time() - t0)/3600:.2f}h -> {path}")


if __name__ == "__main__":
    main()
