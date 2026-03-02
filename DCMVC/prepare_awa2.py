import os
import numpy as np
from scipy.io import savemat

"""
This script expects you already have numeric features for each view of AwA2.
Typical setup:
  - view 1: CNN image embedding (e.g., ResNet-50 pool5)  [N, d1]
  - view 2: attribute vector or another embedding          [N, d2]
  - labels: class id 0..K-1                                [N]

Save them as .npy files and point this script to them.
"""


def main(view1: str, view2: str, labels: str, out_path: str = "datasets/AwA2.mat"):
    X1 = np.load(view1).astype(np.float32)
    X2 = np.load(view2).astype(np.float32)
    y = np.load(labels).astype(np.int32).reshape(-1, 1)

    assert X1.shape[0] == X2.shape[0] == y.shape[0], "N mismatch"

    X_cell = np.empty((1, 2), dtype=object)
    X_cell[0, 0] = X1
    X_cell[0, 1] = X2

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    savemat(out_path, {"X": X_cell, "Y": y})
    print(f"Saved to {out_path} with N={y.shape[0]}, d1={X1.shape[1]}, d2={X2.shape[1]}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--view1", required=True)
    ap.add_argument("--view2", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="datasets/AwA2.mat")
    args = ap.parse_args()

    main(args.view1, args.view2, args.labels, args.out)
