from __future__ import annotations

from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset


class MultiViewDataset(Dataset):
    """A simple dataset wrapper that returns (x_views, index, y).

    - x_views: list[Tensor], one tensor per view
    - index: global row index (needed to align with epoch-level KMeans pseudo-labels)
    - y: true label if provided (for evaluation only), else -1
    """

    def __init__(self, views: List[np.ndarray], labels: Optional[np.ndarray] = None):
        assert len(views) >= 2, "DCMVC needs >=2 views."
        n = views[0].shape[0]
        for v in views:
            assert v.shape[0] == n, "All views must have same N."

        self.views = [torch.from_numpy(v.astype(np.float32)) for v in views]
        self.labels = None if labels is None else torch.from_numpy(labels.astype(np.int64))
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], int, int]:
        x_views = [v[idx] for v in self.views]
        y = -1 if self.labels is None else int(self.labels[idx])
        return x_views, idx, y
