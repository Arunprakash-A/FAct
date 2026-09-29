"""Data loading for Fashion-MNIST / CIFAR-10 / CIFAR-100 / Tiny-ImageNet /
ImageNet-1K.

All paths below are relative to `--data-root` ($FACT_DATA_ROOT, default
~/datasets). Fashion-MNIST is downloaded on demand into
<data-root>/FashionMNIST/raw; CIFAR-10/100 are read as pre-decoded tensors
from <data-root>/hf/{c10,c100}_{train,test}_decoded.pt and Tiny-ImageNet
from <data-root>/hf/tinyimagenet_{train,valid}_decoded.pt. Reading the same
decoded tensors for every arm, with identical augmentation/normalisation, is
what makes the activation variants directly comparable to each other.

ImageNet-1K is the odd one out: at ~150GB of JPEG-encoded parquet shards it
cannot be pre-decoded into one in-memory tensor the way the other four
datasets are (see _load_cifar/_decode_tinyimagenet_parquet above -- those
eagerly materialise x as a single float tensor and cache it as a .pt file).
Instead ImageNet1kDataset wraps a `datasets` (HuggingFace) Arrow-backed,
memory-mapped table built once from the raw parquet shards in
<data-root>/hf/imagenet1k/data/ -- JPEG bytes are decoded lazily per __getitem__
(one image at a time, in DataLoader worker processes), exactly the on-the-fly
decoding this module's docstring above says isn't used for the other four
datasets. See get_imagenet1k_dataloaders / ImageNet1kDataset below.
"""
import glob
import os
import torch
from torch.utils.data import Dataset, DataLoader

MEAN_STD = {
    "fmnist": ((0.2860,), (0.3530,)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "tinyimagenet": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    # standard ImageNet-1K per-channel mean/std (millions of prior models'
    # convention, e.g. torchvision's pretrained ImageNet weights).
    "imagenet1k": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    # food101's mean/std is filled in by _load_food101/_load_cached below
    # (computed from its own train split, same convention as every other
    # entry here being a fixed published constant) -- this placeholder is
    # only what a caller would see if it read MEAN_STD before ever loading
    # the dataset once.
}
PAD_DEFAULT = {"fmnist": 2, "cifar10": 4, "cifar100": 4, "tinyimagenet": 4,
               "food101": 4}
N_CLASSES = {"fmnist": 10, "cifar10": 10, "cifar100": 100, "tinyimagenet": 200,
             "imagenet1k": 1000, "food101": 101, "food101_224": 101}
IMAGENET1K_IMG_SIZE = 224

#: Side length Food-101 is cached at -- carried over from the only prior
#: Food-101 run, which chose 64 to match tinyimagenet's own
#: resolution so PAD_DEFAULT=4 and a patch=8 ViT config carry over unchanged.
FOOD101_IMG_SIZE = 64


def _load_fmnist(data_root):
    from torchvision.datasets import FashionMNIST
    train = FashionMNIST(root=data_root, train=True, download=True)
    test = FashionMNIST(root=data_root, train=False, download=True)
    xtr = train.data.float().div(255.0).unsqueeze(1)
    ytr = train.targets.long()
    xte = test.data.float().div(255.0).unsqueeze(1)
    yte = test.targets.long()
    return xtr, ytr, xte, yte


def _load_cifar(data_root, which):
    tr = torch.load(os.path.join(data_root, "hf", f"{which}_train_decoded.pt"),
                     weights_only=False)
    te = torch.load(os.path.join(data_root, "hf", f"{which}_test_decoded.pt"),
                     weights_only=False)
    return tr["x"].float(), tr["y"].long(), te["x"].float(), te["y"].long()


def _decode_tinyimagenet_parquet(data_root, split):
    import io
    import numpy as np
    import pandas as pd
    from PIL import Image
    hf_dir = os.path.join(data_root, "hf")
    cache = os.path.join(hf_dir, f"tinyimagenet_{split}_decoded.pt")
    if os.path.exists(cache):
        d = torch.load(cache, weights_only=False)
        return d["x"], d["y"]
    df = pd.read_parquet(os.path.join(hf_dir, f"tiny_imagenet_{split}.parquet"))
    n = len(df)
    arr = np.zeros((n, 64, 64, 3), dtype=np.uint8)
    for i, rec in enumerate(df["image"]):
        arr[i] = np.array(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"))
    x = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
    y = torch.tensor(df["label"].to_numpy(), dtype=torch.long)
    try:
        torch.save({"x": x, "y": y}, cache)
    except OSError:
        pass
    return x, y


def _load_tinyimagenet(data_root):
    xtr, ytr = _decode_tinyimagenet_parquet(data_root, "train")
    xte, yte = _decode_tinyimagenet_parquet(data_root, "valid")
    return xtr, ytr, xte, yte


def _food101_cache_path(data_root):
    return os.path.join(data_root, "_cache", "food101_64px_uint8.pt")


def _decode_food101_to_uint8(ds, size, log=print):
    """PIL images -> one (N, 3, size, size) uint8 tensor. uint8 (not
    float32) because Food-101 is 101,000 images: at float32/64x64 that's
    ~5GB resident, at uint8 ~1.2GB -- TensorImageDataset converts per-sample
    at __getitem__ time either way, so the values the model sees are
    identical."""
    from torchvision.transforms.functional import resize, to_tensor
    n = len(ds)
    x = torch.empty((n, 3, size, size), dtype=torch.uint8)
    y = torch.empty(n, dtype=torch.long)
    for i in range(n):
        img, label = ds[i]
        img = img.convert("RGB")
        if img.size != (size, size):
            img = resize(img, [size, size])
        x[i] = (to_tensor(img) * 255).round().clamp(0, 255).to(torch.uint8)
        y[i] = label
        if i % 10000 == 0:
            log(f"    decoded {i}/{n}")
    return x, y


def _load_food101(data_root, log=print):
    """Official Food-101 splits: 75,750 train / 25,250 test, 101 classes,
    decoded+cached once to `_cache/food101_64px_uint8.pt` (per-channel
    mean/std computed from the TRAIN split only and stored alongside, then
    read back into MEAN_STD). Once the cache exists this normally just loads
    it straight off disk rather than re-decoding 101K JPEGs."""
    path = _food101_cache_path(data_root)
    if os.path.exists(path):
        blob = torch.load(path, map_location="cpu")
    else:
        from torchvision.datasets import Food101
        os.makedirs(os.path.dirname(path), exist_ok=True)
        log(f"[cv_data] building food101 cache at {path} (first use only)")
        blob = {}
        for split, kx, ky in (("train", "xtr", "ytr"), ("test", "xte", "yte")):
            ds = Food101(root=data_root, split=split, download=True)
            log(f"[cv_data] decoding Food101 [{split}] ({len(ds)} images)")
            blob[kx], blob[ky] = _decode_food101_to_uint8(
                ds, FOOD101_IMG_SIZE, log=log)
        f = blob["xtr"].float().div(255.0)
        blob["mean"] = f.mean(dim=(0, 2, 3)).tolist()
        blob["std"] = f.std(dim=(0, 2, 3)).tolist()
        del f
        torch.save(blob, path)
        log(f"[cv_data] food101 cache written: train={blob['xtr'].shape} "
            f"test={blob['xte'].shape} mean={blob['mean']} std={blob['std']}")
    MEAN_STD["food101"] = (tuple(blob["mean"]), tuple(blob["std"]))
    # xtr/xte are cached as uint8 [0,255] (see _decode_food101_to_uint8);
    # every other loader here (_load_fmnist, _load_cifar) hands
    # TensorImageDataset a [0,1]-scaled float, and MEAN_STD["food101"] above
    # was itself computed on the /255-scaled train split, so this must match.
    return (blob["xtr"].float().div(255.0), blob["ytr"].long(),
            blob["xte"].float().div(255.0), blob["yte"].long())


def load_raw(dataset, data_root):
    if dataset == "fmnist":
        return _load_fmnist(data_root)
    if dataset == "cifar10":
        return _load_cifar(data_root, "c10")
    if dataset == "cifar100":
        return _load_cifar(data_root, "c100")
    if dataset == "tinyimagenet":
        return _load_tinyimagenet(data_root)
    if dataset == "food101":
        return _load_food101(data_root)
    raise ValueError(dataset)


def _load_imagenet1k_hf_split(data_root, split):
    """Build (once; cached by the `datasets` library under
    ~/.cache/huggingface/datasets on first call) a memory-mapped Arrow table
    over every train-*.parquet / validation-*.parquet shard -- gives true
    O(1) random-access __getitem__ (needed for DataLoader(shuffle=True))
    without ever materialising the whole 150GB in RAM. `split` is "train" or
    "validation" (there is no "test" split -- ImageNet-1K's actual test set
    has no public labels, so it is deliberately not downloaded; see
    download_imagenet1k.py. The standard benchmark protocol reports top-1 on
    "validation" -- see get_imagenet1k_dataloaders, which uses it as this
    study's held-out `test_loader`)."""
    from datasets import load_dataset
    hf_dir = os.path.join(data_root, "hf", "imagenet1k", "data")
    pattern = "train-*.parquet" if split == "train" else "validation-*.parquet"
    files = sorted(glob.glob(os.path.join(hf_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"no {pattern} files under {hf_dir} -- run download_imagenet1k.py first")
    return load_dataset("parquet", data_files={split: files}, split=split, num_proc=8)


class ImageNet1kDataset(Dataset):
    """Wraps an Arrow-backed `datasets.Dataset` (see
    _load_imagenet1k_hf_split): __getitem__ decodes ONE JPEG (via PIL, inside
    whichever DataLoader worker process calls it) and applies torchvision
    transforms, unlike TensorImageDataset above which slices a pre-decoded
    in-memory tensor. `indices`, if given, restricts this view to a subset of
    rows (e.g. carving a held-out val split out of the 1.28M-row train table
    without copying any image data -- see get_imagenet1k_dataloaders).

    train_aug=True: RandomResizedCrop(224) + horizontal flip, the standard
    ImageNet ViT-from-scratch recipe -- NOT the pad-then-random-crop scheme
    TensorImageDataset._augment uses for the other (fixed-size, pre-decoded)
    datasets, since raw ImageNet JPEGs have no fixed size for padding to make
    sense of. train_aug=False: Resize(256) + CenterCrop(224), the standard
    ImageNet eval preprocessing."""

    def __init__(self, hf_ds, mean, std, train_aug=False, img_size=IMAGENET1K_IMG_SIZE,
                 indices=None):
        import torchvision.transforms as T
        self.ds = hf_ds
        self.indices = indices
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.train_aug = train_aug
        if train_aug:
            self.tf = T.Compose([
                T.RandomResizedCrop(img_size, scale=(0.5, 1.0)),
                T.RandomHorizontalFlip(),
            ])
        else:
            self.tf = T.Compose([
                T.Resize(int(img_size * 256 / 224)),
                T.CenterCrop(img_size),
            ])
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.ds)

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx]) if self.indices is not None else idx
        rec = self.ds[real_idx]
        img = rec["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        img = self.tf(img)
        x = self.to_tensor(img)
        x = (x - self.mean) / self.std
        return x, int(rec["label"])


def get_imagenet1k_dataloaders(data_root, batch_size=256, val_size=10000, seed=42,
                                num_workers=16):
    """ImageNet-1K analogue of get_dataloaders below, kept as a separate
    function (rather than another load_raw branch) because it can't share
    get_dataloaders' "slice a pre-loaded (x, y) tensor pair" logic -- see
    ImageNet1kDataset's docstring. train/val is the usual random split
    carved out of the 1.28M-row train table (val_size held out for
    best-checkpoint selection / early stopping, same role val plays for
    every other dataset here); test is the real, official 50K-image
    ImageNet-1K validation split -- the standard benchmark's actual reported
    top-1 metric, the one the standard benchmark reports -- so
    train_one's "test_acc" for this dataset IS the standard ImageNet-1K
    validation accuracy other papers report."""
    mean, std = MEAN_STD["imagenet1k"]
    train_hf = _load_imagenet1k_hf_split(data_root, "train")
    val_hf = _load_imagenet1k_hf_split(data_root, "validation")

    n = len(train_hf)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = ImageNet1kDataset(train_hf, mean, std, train_aug=True, indices=train_idx)
    val_ds = ImageNet1kDataset(train_hf, mean, std, train_aug=False, indices=val_idx)
    test_ds = ImageNet1kDataset(val_hf, mean, std, train_aug=False)

    common = dict(num_workers=num_workers, pin_memory=True,
                  persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "n_classes": N_CLASSES["imagenet1k"]}
    return train_loader, val_loader, test_loader, meta


class TensorImageDataset(Dataset):
    def __init__(self, x, y, mean, std, train_aug=False, pad=4):
        self.x, self.y = x, y
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        self.train_aug = train_aug
        self.pad = pad

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.train_aug:
            img = self._augment(img)
        img = (img - self.mean) / self.std
        return img, self.y[idx]

    def _augment(self, img):
        C, H, W = img.shape
        p = self.pad
        padded = img.new_zeros(C, H + 2 * p, W + 2 * p)
        padded[:, p:p + H, p:p + W] = img
        top = int(torch.randint(0, 2 * p + 1, (1,)))
        left = int(torch.randint(0, 2 * p + 1, (1,)))
        img = padded[:, top:top + H, left:left + W]
        if torch.rand(()) < 0.5:
            img = img.flip(-1)
        return img



class _Food101Records:
    """torchvision Food101 -> the {"image": PIL, "label": int} record shape
    ImageNet1kDataset.__getitem__ expects, so the 224px Food-101 pipeline
    reuses that class's transforms rather than duplicating them."""

    def __init__(self, tv_ds):
        self.ds = tv_ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        img, label = self.ds[i]
        return {"image": img, "label": label}


def get_food101_224_dataloaders(data_root, batch_size=256, val_size=5000,
                                 seed=42, num_workers=16):
    """Food-101 at 224x224, decoded per-sample from the raw JPEGs (no
    pre-decoded cache: 101K images at 224 would be ~15GB). Official 75,750
    train / 25,250 test split, with val_size rows held out of train for
    validation -- same split convention as get_dataloaders' tensor path."""
    from torchvision.datasets import Food101
    train_full = _Food101Records(Food101(root=data_root, split="train"))
    test_recs = _Food101Records(Food101(root=data_root, split="test"))
    mean, std = MEAN_STD["food101_224"]

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_full), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = ImageNet1kDataset(train_full, mean, std, train_aug=True,
                                  img_size=IMAGENET1K_IMG_SIZE, indices=train_idx)
    val_ds = ImageNet1kDataset(train_full, mean, std, train_aug=False,
                                img_size=IMAGENET1K_IMG_SIZE, indices=val_idx)
    test_ds = ImageNet1kDataset(test_recs, mean, std, train_aug=False,
                                 img_size=IMAGENET1K_IMG_SIZE)

    common = dict(num_workers=num_workers, pin_memory=True,
                   persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "n_classes": N_CLASSES["food101_224"]}
    return train_loader, val_loader, test_loader, meta


def get_dataloaders(dataset, data_root, batch_size=256, val_size=5000, seed=42,
                     num_workers=4, pad=None):
    if dataset == "food101_224":
        nw = num_workers if num_workers != 4 else 16
        return get_food101_224_dataloaders(
            data_root, batch_size=batch_size, val_size=val_size, seed=seed,
            num_workers=nw)
    if dataset == "imagenet1k":
        # val_size default (5000) is sized for the other four datasets'
        # train sets (60K-100K rows); scale it up for ImageNet's 1.28M-row
        # train set unless the caller explicitly overrode val_size.
        vs = 10000 if val_size == 5000 else val_size
        nw = num_workers if num_workers != 4 else 16
        return get_imagenet1k_dataloaders(data_root, batch_size=batch_size,
                                          val_size=vs, seed=seed, num_workers=nw)
    xtr_full, ytr_full, xte, yte = load_raw(dataset, data_root)
    mean, std = MEAN_STD[dataset]
    pad = PAD_DEFAULT[dataset] if pad is None else pad

    g = torch.Generator().manual_seed(seed)
    n = xtr_full.shape[0]
    perm = torch.randperm(n, generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_ds = TensorImageDataset(xtr_full[train_idx], ytr_full[train_idx],
                                   mean, std, train_aug=True, pad=pad)
    val_ds = TensorImageDataset(xtr_full[val_idx], ytr_full[val_idx],
                                 mean, std, train_aug=False)
    test_ds = TensorImageDataset(xte, yte, mean, std, train_aug=False)

    common = dict(num_workers=num_workers, pin_memory=True,
                   persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)

    meta = {"n_train": len(train_ds), "n_val": len(val_ds), "n_test": len(test_ds),
            "n_classes": N_CLASSES[dataset]}
    return train_loader, val_loader, test_loader, meta
