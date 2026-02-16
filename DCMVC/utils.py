import random
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def standardize_views(views: list[np.ndarray]) -> list[np.ndarray]:
    out = []
    for v in views:
        scaler = StandardScaler(with_mean=True, with_std=True)
        out.append(scaler.fit_transform(v))
    return out
