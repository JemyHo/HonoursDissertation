import os
import numpy as np
from scipy.io import savemat

# Expected raw files (UCI Multiple Features) in current folder or DATA_DIR
FILES = [
    ("mfeat-fou", "fou"),
    ("mfeat-fac", "fac"),
    ("mfeat-kar", "kar"),
    ("mfeat-pix", "pix"),
    ("mfeat-zer", "zer"),
    ("mfeat-mor", "mor"),
]


def load_txt(path: str) -> np.ndarray:
    # UCI files are whitespace-separated numeric text
    return np.loadtxt(path, dtype=np.float32)


def main(data_dir: str = ".", out_path: str = "datasets/MFeat.mat"):
    views = []
    for fname, tag in FILES:
        fpath = os.path.join(data_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(f"Missing {fpath}")
        X = load_txt(fpath)
        views.append(X)
        print(f"Loaded {fname}: {X.shape}")

    # Labels: 200 samples per class 0..9 in order
    y = np.repeat(np.arange(10, dtype=np.int32), 200)
    y = y.reshape(-1, 1)

    # Build MATLAB-style cell array: shape (1, V)
    X_cell = np.empty((1, len(views)), dtype=object)
    for i, X in enumerate(views):
        X_cell[0, i] = X

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # MATLAB v5 format is enough for our generated files
    savemat(out_path, {"X": X_cell, "Y": y})
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=".")
    ap.add_argument("--out", default="datasets/MFeat.mat")
    args = ap.parse_args()
    main(args.data_dir, args.out)
