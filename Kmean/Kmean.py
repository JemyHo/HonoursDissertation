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
    per-view standardize (+ optional PCA) -> early fusion -> KMeans
    Returns:
      {
        "per_seed": [ {seed, ACC, NMI, ARI, y_pred, inertia, n_iter}, ... ],
        "mean": {...},
        "std": {...}
      }
    """
    views_p = preprocess_views(views, pca_dim=pca_dim)
    X = early_fuse(views_p, view_weights=view_weights)

    per_seed = []
    metric_keys = None

    for s in seeds:
        s = int(s)
        km = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            n_init=n_init,
            max_iter=max_iter,
            random_state=s,
        )
        y_pred = km.fit_predict(X).astype(np.int64)

        scores = clustering_scores(y_true, y_pred)  # dict: ACC/NMI/ARI (and whatever you return)
        if metric_keys is None:
            metric_keys = list(scores.keys())

        per_seed.append({
            "seed": s,
            **scores,
            "y_pred": y_pred,
            "inertia": float(getattr(km, "inertia_", np.nan)),
            "n_iter": int(getattr(km, "n_iter_", 0)),
        })

    # aggregate only metrics (NOT y_pred / inertia / n_iter)
    mean = {k: float(np.mean([d[k] for d in per_seed])) for k in metric_keys}

    if len(per_seed) > 1:
        std = {k: float(np.std([d[k] for d in per_seed], ddof=1)) for k in metric_keys}
    else:
        std = {k: 0.0 for k in metric_keys}

    return {"per_seed": per_seed, "mean": mean, "std": std}
