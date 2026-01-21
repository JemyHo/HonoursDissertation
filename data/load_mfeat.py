from pathlib import Path
import numpy as np

MFEAT_VIEWS = ["fou", "fac", "kar", "pix", "zer", "mor"]

def load_mfeat(root="Dataset/Handwritten"):
    root = Path(root)

    views = []
    for v in MFEAT_VIEWS:
        fp = root / f"mfeat-{v}"
        X = np.loadtxt(fp).astype(np.float32)
        views.append(X)

    n = views[0].shape[0]
    if not all(x.shape[0] == n for x in views):
        raise ValueError("Row mismatch across views.")

    # Labels are for evaluation only (digits 0..9, 200 each)
    y_true = np.repeat(np.arange(10), 200).astype(np.int64)
    if len(y_true) != n:
        raise ValueError(f"Expected 2000 samples, got {n}.")

    return views, y_true