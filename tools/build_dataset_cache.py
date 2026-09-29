"""Build the pre-decoded CIFAR-10/100 tensors the training code reads.

``code/{source,target}_vit/cv_data.py`` does not decode CIFAR itself: it
``torch.load``s ``<data-root>/hf/{c10,c100}_{train,test}_decoded.pt``, one
float32 ``(N, 3, 32, 32)`` tensor in ``[0, 1]`` plus int64 labels. Reading the
same decoded tensor for every activation is what makes the variants comparable,
but it also means a fresh clone has nothing to train on until this script runs.

    export FACT_DATA_ROOT=/path/to/datasets
    python tools/build_dataset_cache.py              # cifar10 + cifar100
    python tools/build_dataset_cache.py --verify     # check what is there

Source is the HuggingFace parquet release of each dataset (PNG bytes + label),
decoded in stored row order -- *not* torchvision's ``CIFAR10``, whose row order
differs, which would silently change the validation carve-out and every
augmentation draw. The SHA-256 digests below are of the published runs' own
tensors, so a rebuilt cache is checked to be byte-identical to the one behind
the paper's numbers rather than merely plausible.

The other datasets need nothing from this script: Fashion-MNIST and Food-101 are
pulled by torchvision on demand, and ImageNet-1K has its own downloader,
``code/source_vit/download_imagenet1k.py``. (``--dataset tinyimagenet`` is code
the paper does not use; it expects ``<data-root>/hf/tiny_imagenet_{train,valid}
.parquet`` to be fetched by hand.)
"""
import argparse
import hashlib
import io
import os
import sys

import numpy as np
import torch

#: repo, parquet path, image column, label column, cache prefix
SOURCES = {
    "cifar10": ("uoft-cs/cifar10", "plain_text/{split}-00000-of-00001.parquet",
                "img", "label", "c10"),
    "cifar100": ("uoft-cs/cifar100", "cifar100/{split}-00000-of-00001.parquet",
                 "img", "fine_label", "c100"),
}

#: SHA-256 of the published tensors' raw bytes, (x, y) per split. Measured on
#: the caches the paper's runs read; a rebuild that matches these is the same
#: data in the same order, down to the byte.
DIGESTS = {
    ("cifar10", "train"): ("cd85adb8ebe4aa6cf89aa22f6d2047de4d8a64554f555a8d44a6a3b2f445a1cd",
                           "d6d2fe5c7528f766adb2e13bf983d1e4707936c06b4ccf240ef3d2dcd72c986d"),
    ("cifar10", "test"): ("53e7254b7fbf84fd84d2bedbff142db6cd7d44c6b4c353b4836b2bd1fa851c7d",
                          "cbb7365de8ed11f05cc4c3a1e7f78144127c5e851efd83762fb18202461230bb"),
    ("cifar100", "train"): ("39033954a2b22b1bd30986ea3072cc4354d85ce6bcc3d195c9b242036cdcbc5d",
                            "8bf4935e7c77a3096270043c6f3b221b6ecb9c63d1a08608e9af66cff17e977a"),
    ("cifar100", "test"): ("872c85d4e96899a58f6ecb4ae883f203cd8371548862a77f9eba62bf6f8b5ef0",
                           "80031c23f8300724c6fbc588e4bccbe41c6392041782148727bb497b0132c4ad"),
}

N_ROWS = {"train": 50_000, "test": 10_000}


def _sha256(t):
    return hashlib.sha256(np.ascontiguousarray(t.numpy()).tobytes()).hexdigest()


def cache_path(data_root, dataset, split):
    which = SOURCES[dataset][4]
    return os.path.join(data_root, "hf", f"{which}_{split}_decoded.pt")


def decode(dataset, split):
    """Decode one split's parquet into ``(x, y)`` in its stored row order."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo, pattern, img_col, lbl_col, _ = SOURCES[dataset]
    path = hf_hub_download(repo, pattern.format(split=split), repo_type="dataset")
    table = pq.read_table(path)
    rows = table.column(img_col).to_pylist()
    y = torch.tensor(table.column(lbl_col).to_numpy(zero_copy_only=False),
                     dtype=torch.long)
    arr = np.zeros((len(rows), 32, 32, 3), dtype=np.uint8)
    for i, rec in enumerate(rows):
        arr[i] = np.array(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    return x.contiguous(), y


def check(dataset, split, x, y):
    """True iff this split matches the published rows, shape and digests."""
    want_x, want_y = DIGESTS[(dataset, split)]
    ok = True
    if len(y) != N_ROWS[split] or tuple(x.shape[1:]) != (3, 32, 32):
        print(f"    FAIL shape {tuple(x.shape)}, expected ({N_ROWS[split]}, 3, 32, 32)")
        ok = False
    for name, got, want in (("images", _sha256(x), want_x), ("labels", _sha256(y), want_y)):
        if got == want:
            print(f"    ok   {name} sha256 {got[:16]}... matches the published cache")
        else:
            print(f"    FAIL {name} sha256 {got[:16]}... != published {want[:16]}...")
            ok = False
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root",
                    default=os.environ.get("FACT_DATA_ROOT", os.path.expanduser("~/datasets")),
                    help="dataset root; caches land in <data-root>/hf "
                         "(default: $FACT_DATA_ROOT, else ~/datasets)")
    ap.add_argument("--datasets", nargs="+", default=sorted(SOURCES),
                    choices=sorted(SOURCES))
    ap.add_argument("--verify", action="store_true",
                    help="only check the caches already on disk; build nothing")
    ap.add_argument("--force", action="store_true",
                    help="re-decode and overwrite a cache that already exists")
    args = ap.parse_args()

    failed = []
    for dataset in args.datasets:
        for split in ("train", "test"):
            path = cache_path(args.data_root, dataset, split)
            rel = os.path.join("<data-root>", "hf", os.path.basename(path))
            exists = os.path.exists(path)
            print(f"\n{dataset} {split} -> {rel}")

            if args.verify or (exists and not args.force):
                if not exists:
                    print("    missing -- run without --verify to build it")
                    failed.append(f"{dataset}/{split}")
                    continue
                d = torch.load(path, weights_only=False)
                print("    already built" if not args.verify else "    on disk")
                if not check(dataset, split, d["x"], d["y"]):
                    failed.append(f"{dataset}/{split}")
                continue

            print("    decoding from the HuggingFace parquet release...")
            x, y = decode(dataset, split)
            if not check(dataset, split, x, y):
                failed.append(f"{dataset}/{split}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save({"x": x, "y": y}, path)
            print(f"    wrote {os.path.getsize(path) / 1e6:.0f} MB")

    print("\n" + ("failed: " + ", ".join(failed) if failed else "all caches present and byte-identical to the published runs'"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
