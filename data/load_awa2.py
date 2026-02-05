# data/load_awa2.py
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

def read_class_names(fp: Path) -> List[str]:
    names: List[str] = []
    for raw in fp.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue

        # handle: "1<TAB>antelope" OR "1 antelope"
        parts = line.split(None, 1)
        if len(parts) == 2 and parts[0].isdigit():
            names.append(parts[1].strip())
        else:
            names.append(parts[0].strip())
    return names

    
def load_awa2(
    root: str = "Dataset/AwA/Animals_with_Attributes2",
    split: str = "all",                 # "train", "test", "all"
    max_per_class: Optional[int] = 200, # set None for full dataset (bigger)
    use_continuous: bool = True,        # continuous vs binary attributes
    random_state: int = 0,
    feature_dim: int = 512,             # resnet18 gives 512
    batch_size: int = 64,
    num_workers: int = 2,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Loads AwA2 into your KMeans pipeline format.

    Returns:
      views: [X_img, X_attr]
        - X_img : (N, feature_dim)  visual embeddings from a pretrained CNN
        - X_attr: (N, A)            attribute vector per image (class-level, repeated)
      y_true: (N,) integer class labels (0..K-1) for evaluation only
      class_names: list of selected class names (length K)
    """
    root = Path(root)
    assert root.exists(), f"AwA2 root not found: {root}"

    # --- Required paths in AwA2 ---
    jpeg_dir = root / "JPEGImages"
    classes_fp = root / "classes.txt"
    train_fp = root / "trainclasses.txt"
    test_fp  = root / "testclasses.txt"

    attr_fp = root / ("predicate-matrix-continuous.txt" if use_continuous else "predicate-matrix-binary.txt")

    # --- Read class list (AwA2 has 50 classes) ---
    all_classes = read_class_names(classes_fp)

    # --- Select classes based on split file ---
    if split.lower() == "train":
      class_names = read_class_names(train_fp)
    elif split.lower() == "test":
        class_names = read_class_names(test_fp)
    elif split.lower() == "all":
        class_names = all_classes
    else:
        raise ValueError("split must be one of: 'train', 'test', 'all'")

    print("First 5 class_names:", class_names[:5])
    print("Example folder exists?:", (jpeg_dir / class_names[0]).exists())


    # Map class name -> row in attribute matrix (based on classes.txt order)
    class_to_global_idx = {c: i for i, c in enumerate(all_classes)}
    missing = [c for c in class_names if c not in class_to_global_idx]
    if missing:
        raise ValueError(f"Some split classes not found in classes.txt: {missing[:5]}")

    # --- Load class-level attribute matrix ---
    attr_mat = np.loadtxt(attr_fp).astype(np.float32)  # expected shape ~ (50, num_attributes)
    if attr_mat.shape[0] != len(all_classes):
        raise ValueError(
            f"Attribute matrix row count {attr_mat.shape[0]} != #classes {len(all_classes)}. "
            "Check file format."
        )

    # --- Gather image paths + labels + attributes per image ---
    rng = np.random.default_rng(random_state)

    img_paths: List[Path] = []
    y_list: List[int] = []
    attr_list: List[np.ndarray] = []

    for local_label, cname in enumerate(class_names):
        cdir = jpeg_dir / cname
        if not cdir.exists():
            raise FileNotFoundError(f"Missing class folder: {cdir}")

        files = sorted([p for p in cdir.glob("*.jpg")])
        if max_per_class is not None and len(files) > max_per_class:
            # sample to keep runtime reasonable
            idx = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in idx]

        gidx = class_to_global_idx[cname]
        c_attr = attr_mat[gidx]  # (A,)

        for p in files:
            img_paths.append(p)
            y_list.append(local_label)
            attr_list.append(c_attr)

    y_true = np.asarray(y_list, dtype=np.int64)
    X_attr = np.vstack(attr_list).astype(np.float32)

    # --- Extract visual features with torchvision ResNet18 ---
    # (kept inside the loader so your notebook stays clean)
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
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Pretrained model -> feature extractor
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = torch.nn.Identity()  # output becomes (B, 512)
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
    if X_img.shape[0] != X_attr.shape[0]:
        raise ValueError(f"Row mismatch: X_img {X_img.shape} vs X_attr {X_attr.shape}")

    # sanity print
    print("AwA2 loaded:", "X_img", X_img.shape, "X_attr", X_attr.shape, "y", y_true.shape)

    return [X_img, X_attr], y_true, class_names
