from __future__ import annotations

import os
from pathlib import Path
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import sklearn.preprocessing as skp
import torch
from bs4 import BeautifulSoup
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer


# -------------------- MFeat --------------------
MFEAT_VIEWS = ["fou", "fac", "kar", "pix", "zer", "mor"]


def load_mfeat(root: str) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    root = Path(root)
    views = []
    for v in MFEAT_VIEWS:
        fp = root / f"mfeat-{v}"
        X = np.loadtxt(fp).astype(np.float32)
        views.append(X)
    n = views[0].shape[0]
    if not all(x.shape[0] == n for x in views):
        raise ValueError("Row mismatch across MFeat views.")
    y_true = np.repeat(np.arange(10), 200).astype(np.int64)
    class_names = [str(i) for i in range(10)]
    return views, y_true, class_names


# -------------------- Reuters --------------------
def _iter_reuters_docs(reuters_dir: str, splits=("TRAIN", "TEST")):
    reuters_dir = Path(reuters_dir)
    sgm_files = sorted(reuters_dir.glob("reut2-*.sgm"))
    if not sgm_files:
        raise FileNotFoundError(f"No reut2-*.sgm found in: {reuters_dir}")

    for fp in sgm_files:
        soup = BeautifulSoup(fp.read_text(errors="ignore"), "html.parser")
        for r in soup.find_all("reuters"):
            split_tag = (r.get("lewissplit") or "").upper()
            if split_tag and split_tag not in splits:
                continue

            topics_node = r.find("topics")
            topics = []
            if topics_node:
                topics = [d.get_text(strip=True) for d in topics_node.find_all("d")]

            title = r.find("title")
            body = r.find("body")
            text = " ".join([
                title.get_text(" ", strip=True) if title else "",
                body.get_text(" ", strip=True) if body else "",
            ]).strip()

            if text:
                yield text, topics, split_tag


def load_reuters21578(
    reuters_dir: str,
    splits=("TRAIN",),
    top_k_topics: int = 10,
    svd_dim: int = 300,
    max_features_word: int = 50000,
    max_features_char: int = 80000,
    random_state: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    texts = []
    single_topics = []

    for text, topics, _split in _iter_reuters_docs(reuters_dir, splits=splits):
        if len(topics) == 1:
            texts.append(text)
            single_topics.append(topics[0])

    if not texts:
        raise ValueError("No single-topic Reuters documents found.")

    freq = Counter(single_topics)
    topic_names = [t for t, _ in freq.most_common(top_k_topics)]
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    keep_idx = [i for i, t in enumerate(single_topics) if t in topic_to_id]

    texts = [texts[i] for i in keep_idx]
    y_true = np.array([topic_to_id[single_topics[i]] for i in keep_idx], dtype=np.int64)

    word_vec = TfidfVectorizer(
        stop_words="english",
        lowercase=True,
        max_df=0.9,
        min_df=2,
        max_features=max_features_word,
    )
    Xw_sparse = word_vec.fit_transform(texts)
    Xw = TruncatedSVD(n_components=min(svd_dim, Xw_sparse.shape[1] - 1), random_state=random_state)
    Xw = Xw.fit_transform(Xw_sparse).astype(np.float32)

    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        lowercase=True,
        max_df=0.95,
        min_df=2,
        max_features=max_features_char,
    )
    Xc_sparse = char_vec.fit_transform(texts)
    Xc = TruncatedSVD(n_components=min(svd_dim, Xc_sparse.shape[1] - 1), random_state=random_state)
    Xc = Xc.fit_transform(Xc_sparse).astype(np.float32)

    return [Xw, Xc], y_true, topic_names


# -------------------- AwA2 --------------------
def read_class_names(fp: Path) -> List[str]:
    names: List[str] = []
    for raw in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            names.append(parts[1].strip())
        else:
            names.append(parts[0].strip())
    return names


def load_awa2(
    root: str,
    split: str = "all",
    max_per_class: Optional[int] = 200,
    use_continuous: bool = True,
    random_state: int = 0,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    root = Path(root)

    # ---------- cache paths ----------
    cache_dir = root / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_tag = f"{split}_max{max_per_class}_cont{int(use_continuous)}_rs{random_state}"
    ximg_fp = cache_dir / f"X_img_{cache_tag}.npy"
    xattr_fp = cache_dir / f"X_attr_{cache_tag}.npy"
    y_fp = cache_dir / f"y_true_{cache_tag}.npy"
    names_fp = cache_dir / f"class_names_{cache_tag}.npy"

    # ---------- fast path: load cached arrays ----------
    if ximg_fp.exists() and xattr_fp.exists() and y_fp.exists() and names_fp.exists():
        X_img = np.load(ximg_fp)
        X_attr = np.load(xattr_fp)
        y_true = np.load(y_fp)
        class_names = np.load(names_fp, allow_pickle=True).tolist()
        print(f"[AwA2] Loaded cached features from {cache_dir}")
        return [X_img, X_attr], y_true, class_names

    # ---------- original build path ----------
    jpeg_dir = root / "JPEGImages"
    classes_fp = root / "classes.txt"
    train_fp = root / "trainclasses.txt"
    test_fp = root / "testclasses.txt"
    attr_fp = root / ("predicate-matrix-continuous.txt" if use_continuous else "predicate-matrix-binary.txt")

    all_classes = read_class_names(classes_fp)
    if split.lower() == "train":
        class_names = read_class_names(train_fp)
    elif split.lower() == "test":
        class_names = read_class_names(test_fp)
    elif split.lower() == "all":
        class_names = all_classes
    else:
        raise ValueError("split must be one of: 'train', 'test', 'all'")

    class_to_global_idx = {c: i for i, c in enumerate(all_classes)}
    attr_mat = np.loadtxt(attr_fp).astype(np.float32)

    rng = np.random.default_rng(random_state)
    img_paths: List[Path] = []
    y_list: List[int] = []
    attr_list: List[np.ndarray] = []

    for local_label, cname in enumerate(class_names):
        cdir = jpeg_dir / cname
        files = sorted([p for p in cdir.glob("*.jpg")])
        if max_per_class is not None and len(files) > max_per_class:
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in idx]
        gidx = class_to_global_idx[cname]
        c_attr = attr_mat[gidx]
        for p in files:
            img_paths.append(p)
            y_list.append(local_label)
            attr_list.append(c_attr)

    y_true = np.asarray(y_list, dtype=np.int64)
    X_attr = np.vstack(attr_list).astype(np.float32)

    import torch
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image

    class ImgDS(Dataset):
        def __init__(self, paths: List[Path], tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert("RGB")
            return self.tfm(img)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval().to(device)

    ds = ImgDS(img_paths, tfm)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    feats = []
    with torch.no_grad():
        for xb in dl:
            xb = xb.to(device, non_blocking=True)
            fb = model(xb).detach().cpu().numpy().astype(np.float32)
            feats.append(fb)

    X_img = np.vstack(feats)

    # ---------- save cache ----------
    np.save(ximg_fp, X_img)
    np.save(xattr_fp, X_attr)
    np.save(y_fp, y_true)
    np.save(names_fp, np.array(class_names, dtype=object))
    print(f"[AwA2] Saved cached features to {cache_dir}")

    return [X_img, X_attr], y_true, class_names


# -------------------- normalization / dataset wrappers --------------------

def _normalize_views(data_X: List[np.ndarray], norm: str) -> List[np.ndarray]:
    if norm == "standard":
        return [skp.scale(x).astype(np.float32) for x in data_X]
    if norm == "l2-norm":
        return [skp.normalize(x).astype(np.float32) for x in data_X]
    if norm == "min-max":
        return [skp.minmax_scale(x).astype(np.float32) for x in data_X]
    raise ValueError(f"Unknown normalization: {norm}")


def load_custom_dataset(args):
    root = Path(args.data_root)
    if args.dataset == "MFeat":
        data_X, label_y, class_names = load_mfeat(str(root / "Dataset" / "Handwritten"))
    elif args.dataset == "Reuters":
        data_X, label_y, class_names = load_reuters21578(str(root / "Dataset" / "Reuters"))
    elif args.dataset == "AwA2":
        data_X, label_y, class_names = load_awa2(str(root / "Dataset" / "AwA" / "Animals_with_Attributes2"))
    else:
        raise KeyError(f"Unsupported custom dataset: {args.dataset}")

    data_X = _normalize_views(data_X, args.data_norm)
    args.n_views = len(data_X)
    args.n_sample = data_X[0].shape[0]
    args.n_clusters = len(np.unique(label_y))
    args.class_names = class_names
    return data_X, label_y, class_names


class CustomMultiviewDataset(torch.utils.data.Dataset):
    def __init__(self, data_X: List[np.ndarray], labels: np.ndarray, pseudo_label: Optional[np.ndarray] = None):
        super().__init__()
        self.n_views = len(data_X)
        self.data = data_X
        self.labels = labels.astype(np.int64) - np.min(labels)
        self.pseudo_label = pseudo_label

    def __len__(self):
        return self.data[0].shape[0]

    def __getitem__(self, idx):
        data = [torch.tensor(self.data[i][idx].astype("float32")) for i in range(self.n_views)]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.pseudo_label is None:
            return idx, data, label
        pseudo = torch.tensor(self.pseudo_label[idx], dtype=torch.long)
        return idx, data, label, pseudo


def get_pseudo_label_path(args):
    return os.path.join(args.output_dir, f"{args.dataset}_warmup_pseudo_labels_seed{args.seed}.npy")


def load_dataset(args, use_pseudo_labels=False):
    data, labels, _ = load_custom_dataset(args)
    if use_pseudo_labels:
        pseudo_path = get_pseudo_label_path(args)
        pseudo_labels = np.load(pseudo_path)
        dataset = CustomMultiviewDataset(data, labels, pseudo_label=pseudo_labels)
    else:
        dataset = CustomMultiviewDataset(data, labels)
    input_dims = [x.shape[1] for x in data]
    return dataset, input_dims
