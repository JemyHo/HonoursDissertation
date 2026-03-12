
from __future__ import annotations

from pathlib import Path
from collections import Counter
from typing import List, Optional, Sequence, Tuple
import html
import re

import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


class MultiViewArrayDataset(Dataset):
    def __init__(self, views: Sequence[np.ndarray], labels: np.ndarray):
        if len(views) < 2:
            raise ValueError('Need at least 2 views.')
        n = int(views[0].shape[0])
        for i, v in enumerate(views):
            if v.shape[0] != n:
                raise ValueError(f'View {i} row count {v.shape[0]} != {n}')
        labels = np.asarray(labels).reshape(-1)
        if labels.shape[0] != n:
            raise ValueError(f'Labels row count {labels.shape[0]} != {n}')
        self.views = [np.asarray(v, dtype=np.float32) for v in views]
        self.labels = labels.astype(np.int64)

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        xs = [torch.from_numpy(v[idx]) for v in self.views]
        y = int(self.labels[idx])
        return xs, y, torch.tensor(idx, dtype=torch.long)


def _zscore_views(views: Sequence[np.ndarray]) -> List[np.ndarray]:
    out: List[np.ndarray] = []
    for v in views:
        v = np.asarray(v, dtype=np.float32)
        scaler = StandardScaler(with_mean=True, with_std=True)
        out.append(scaler.fit_transform(v).astype(np.float32))
    return out


def load_mfeat(root: str = 'Dataset/Handwritten') -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    root = Path(root)
    names = ['fou', 'fac', 'kar', 'pix', 'zer', 'mor']
    files = [root / f'mfeat-{n}' for n in names]
    for fp in files:
        if not fp.exists():
            raise FileNotFoundError(f'Missing file: {fp}')
    views = [np.loadtxt(fp, dtype=np.float32) for fp in files]
    n = views[0].shape[0]
    if any(v.shape[0] != n for v in views):
        raise ValueError('MFeat views do not share the same number of rows.')
    # UCI Multiple Features: 200 samples per digit class in order 0..9
    y_true = np.repeat(np.arange(10, dtype=np.int64), n // 10)
    class_names = [str(i) for i in range(10)]
    return _zscore_views(views), y_true, class_names


REUTERS_FILE_RE = re.compile(r'reut2-\d{3}\.sgm$')


def _iter_reuters_docs(reuters_dir: str, splits=("TRAIN",)):
    reuters_dir = Path(reuters_dir)
    sgm_files = sorted(reuters_dir.glob('reut2-*.sgm'))
    if not sgm_files:
        raise FileNotFoundError(f'No reut2-*.sgm found in: {reuters_dir}')

    from bs4 import BeautifulSoup

    for fp in sgm_files:
        soup = BeautifulSoup(fp.read_text(encoding='latin-1', errors='ignore'), 'html.parser')
        for r in soup.find_all('reuters'):
            split_tag = (r.get('lewissplit') or '').upper()
            if split_tag and split_tag not in splits:
                continue

            topics_node = r.find('topics')
            topics = []
            if topics_node:
                topics = [d.get_text(strip=True) for d in topics_node.find_all('d')]

            title = r.find('title')
            body = r.find('body')
            text = ' '.join([
                title.get_text(' ', strip=True) if title else '',
                body.get_text(' ', strip=True) if body else '',
            ]).strip()

            if text:
                yield text, topics, split_tag


def load_reuters(
    reuters_dir: str = 'Dataset/Reuters',
    splits=("TRAIN",),
    top_k_topics: int = 10,
    svd_dim: int = 300,
    max_features_word: int = 50000,
    max_features_char: int = 80000,
    random_state: int = 0,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    texts: List[str] = []
    single_topics: List[str] = []
    for text, topics, _split in _iter_reuters_docs(reuters_dir, splits=splits):
        if len(topics) == 1:
            texts.append(text)
            single_topics.append(topics[0])

    if not texts:
        raise ValueError('No single-topic Reuters documents found.')

    freq = Counter(single_topics)
    topic_names = [t for t, _ in freq.most_common(top_k_topics)]
    topic_to_id = {t: i for i, t in enumerate(topic_names)}
    keep_idx = [i for i, t in enumerate(single_topics) if t in topic_to_id]

    texts = [texts[i] for i in keep_idx]
    y_true = np.array([topic_to_id[single_topics[i]] for i in keep_idx], dtype=np.int64)

    word_vec = TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_df=0.9,
        min_df=5,
        max_features=max_features_word,
    )
    Xw_sparse = word_vec.fit_transform(texts)
    Xw = TruncatedSVD(n_components=min(svd_dim, Xw_sparse.shape[1] - 1), random_state=random_state)        .fit_transform(Xw_sparse).astype(np.float32)

    char_vec = TfidfVectorizer(
        analyzer='char_wb',
        ngram_range=(3, 5),
        lowercase=True,
        sublinear_tf=True,
        min_df=5,
        max_features=max_features_char,
    )
    Xc_sparse = char_vec.fit_transform(texts)
    Xc = TruncatedSVD(n_components=min(svd_dim, Xc_sparse.shape[1] - 1), random_state=random_state)        .fit_transform(Xc_sparse).astype(np.float32)

    return _zscore_views([Xw, Xc]), y_true, topic_names


def _read_class_names(fp: Path) -> List[str]:
    names: List[str] = []
    for raw in fp.read_text(encoding='utf-8', errors='ignore').splitlines():
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
    root: str = 'Dataset/AwA/Animals_with_Attributes2',
    split: str = 'all',
    max_per_class: Optional[int] = 200,
    use_continuous: bool = True,
    random_state: int = 0,
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    root = Path(root)
    jpeg_dir = root / 'JPEGImages'
    classes_fp = root / 'classes.txt'
    train_fp = root / 'trainclasses.txt'
    test_fp = root / 'testclasses.txt'
    attr_fp = root / ('predicate-matrix-continuous.txt' if use_continuous else 'predicate-matrix-binary.txt')

    all_classes = _read_class_names(classes_fp)
    if split.lower() == 'train':
        class_names = _read_class_names(train_fp)
    elif split.lower() == 'test':
        class_names = _read_class_names(test_fp)
    elif split.lower() == 'all':
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
        if not cdir.exists():
            raise FileNotFoundError(f'Missing class folder: {cdir}')
        files = sorted(cdir.glob('*.jpg'))
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
    from torch.utils.data import Dataset as TorchDataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image

    class ImgDS(TorchDataset):
        def __init__(self, paths, tfm):
            self.paths = paths
            self.tfm = tfm
        def __len__(self):
            return len(self.paths)
        def __getitem__(self, i):
            img = Image.open(self.paths[i]).convert('RGB')
            return self.tfm(img)

    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
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

    return _zscore_views([X_img, X_attr]), y_true, class_names


def load_data(dataset_name: str, data_root: str = '.'):
    name = dataset_name.lower()
    if name == 'mfeat':
        views, y_true, class_names = load_mfeat(Path(data_root) / 'Dataset' / 'Handwritten')
    elif name == 'reuters':
        views, y_true, class_names = load_reuters(Path(data_root) / 'Dataset' / 'Reuters')
    elif name == 'awa2':
        views, y_true, class_names = load_awa2(Path(data_root) / 'Dataset' / 'AwA' / 'Animals_with_Attributes2')
    else:
        raise NotImplementedError(f'Unsupported dataset: {dataset_name}')

    dataset = MultiViewArrayDataset(views, y_true)
    dims = [int(v.shape[1]) for v in views]
    view = len(views)
    data_size = int(y_true.shape[0])
    class_num = int(len(np.unique(y_true)))
    return dataset, dims, view, data_size, class_num, class_names
