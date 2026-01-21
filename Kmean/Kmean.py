import numpy as np
from sklearn.cluster import KMeans
from .Preprocess import preprocess_views, early_fuse
from .Metrics import clustering_scores

def run_kmeans_baseline(
    views,
    y_true,
    n_clusters,
    seeds=(0, 1, 2, 3, 4),
    pca_dim=None,
    view_weights=None,
    n_init=20,
    max_iter=300
):
    """
    Implements: per-view standardize (+ optional PCA) -> early fusion -> KMeans++ multi-restart
    and returns mean ± std over seeds for ACC/NMI/ARI.
    """
    views_p = preprocess_views(views, pca_dim=pca_dim)
    X = early_fuse(views_p, view_weights=view_weights)

    per_seed = []
    for s in seeds:
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=int(s),
        )
        pred = km.fit_predict(X)
        per_seed.append(clustering_scores(y_true, pred))

    # aggregate
    keys = per_seed[0].keys()
    mean = {k: float(np.mean([d[k] for d in per_seed])) for k in keys}
    std  = {k: float(np.std([d[k] for d in per_seed], ddof=1)) for k in keys}  # sample std
    return {"per_seed": per_seed, "mean": mean, "std": std}