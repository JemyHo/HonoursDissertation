from pathlib import Path
import numpy as np

# The Multiple Features dataset provides 6 different "views" (feature sets) per sample.
# Each view is stored as a separate file: mfeat-fou, mfeat-fac, mfeat-kar, mfeat-pix, mfeat-zer, mfeat-mor.
MFEAT_VIEWS = ["fou", "fac", "kar", "pix", "zer", "mor"]

def load_mfeat(root="Dataset/Handwritten"):

  # root : str - Directory containing the mfeat-* files.
  root = Path(root)

  views = []
  for v in MFEAT_VIEWS:
    # Build the path to the view file, e.g. Dataset/Handwritten/mfeat-fou
      fp = root / f"mfeat-{v}"
      # Load the whitespace-delimited numeric table:
      # - rows   = samples (2000 total)
      # - cols   = features for this particular view
      # Cast to float32 to reduce memory use and speed up computation.
      X = np.loadtxt(fp).astype(np.float32)
      views.append(X)

  n = views[0].shape[0]
  # Sanity check: all views must contain the same number of samples (rows)
  if not all(x.shape[0] == n for x in views):
      raise ValueError("Row mismatch across views.")

  # The dataset is ordered with 200 samples per class for digits 0..9 (total 2000 samples).
  # We create labels for evaluation ONLY (K-means itself does not use y_true).
  y_true = np.repeat(np.arange(10), 200).astype(np.int64)
  if len(y_true) != n:
      raise ValueError(f"Expected 2000 samples, got {n}.")

  return views, y_true