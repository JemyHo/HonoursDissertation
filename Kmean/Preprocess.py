import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def preprocess_views(views, pca_dim=None, pca_random_state=0):
    proc = []
    for X in views:
        X = np.asarray(X, dtype=np.float32)
        Xs = StandardScaler().fit_transform(X)  # per-view z-score
        if pca_dim is not None and pca_dim < Xs.shape[1]:
            Xs = PCA(n_components=pca_dim, random_state=pca_random_state).fit_transform(Xs)
        proc.append(Xs.astype(np.float32))
    return proc

def early_fuse(views, view_weights=None):
    if view_weights is None:
        return np.concatenate(views, axis=1)
    w = np.asarray(view_weights, dtype=np.float32)
    assert len(w) == len(views)
    return np.concatenate([views[i] * w[i] for i in range(len(views))], axis=1)
