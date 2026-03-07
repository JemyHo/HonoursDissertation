import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

def cluster_composition_table(y_true, y_pred, class_names=None, topk=5):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    K = int(y_pred.max()) + 1
    C = int(y_true.max()) + 1

    # contingency: rows=cluster, cols=true class
    M = np.zeros((K, C), dtype=int)
    for c, t in zip(y_pred, y_true):
        M[int(c), int(t)] += 1

    rows = []
    for k in range(K):
        counts = M[k]
        size = counts.sum()
        if size == 0:
            rows.append({"cluster": k, "size": 0, "majority_class": None, "purity": 0.0, "top_classes": ""})
            continue

        maj = int(counts.argmax())
        purity = float(counts[maj] / size)

        top_idx = np.argsort(counts)[::-1][:topk]
        top_parts = []
        for j in top_idx:
            if counts[j] == 0:
                break
            name = class_names[j] if class_names is not None else str(j)
            top_parts.append(f"{name}:{counts[j]}")
        rows.append({
            "cluster": k,
            "size": int(size),
            "majority_class": (class_names[maj] if class_names is not None else str(maj)),
            "purity": purity,
            "top_classes": ", ".join(top_parts)
        })

    df = pd.DataFrame(rows).sort_values(["size", "purity"], ascending=False).reset_index(drop=True)
    return df, M

def cluster_class_count_matrix(y_true, y_pred, n_clusters=None, n_classes=None):
  y_true = np.asarray(y_true, dtype=int)
  y_pred = np.asarray(y_pred, dtype=int)
  K = int(n_clusters if n_clusters is not None else (y_pred.max() + 1))
  C = int(n_classes  if n_classes  is not None else (y_true.max() + 1))
  cm = np.zeros((K, C), dtype=int)
  for p, t in zip(y_pred, y_true):
      if 0 <= p < K and 0 <= t < C:
          cm[p, t] += 1
  return cm

def plot_heatmap(cm, title, xlabels=None, ylabels=None, max_xticks=20,
                 figsize=(7,5), vmin=0, vmax=None):
    plt.figure(figsize=figsize)
    plt.imshow(cm, aspect="auto", vmin=vmin, vmax=vmax)
    plt.title(title)
    plt.xlabel("True class")
    plt.ylabel("Cluster")

    if xlabels is not None:
        if len(xlabels) <= max_xticks:
            plt.xticks(range(len(xlabels)), xlabels, rotation=90, fontsize=8)
        else:
            step = max(1, len(xlabels)//max_xticks)
            idx = list(range(0, len(xlabels), step))
            plt.xticks(idx, [xlabels[i] for i in idx], rotation=90, fontsize=8)

    if ylabels is not None:
        if len(ylabels) <= max_xticks:
            plt.yticks(range(len(ylabels)), ylabels, fontsize=8)

    plt.colorbar(label="count")
    plt.tight_layout()
    plt.show()


def hungarian_reorder(cm):
    """
    cm: [K, C] counts for (cluster, class)
    Returns:
      row_order: clusters reordered so best-matched classes go left-to-right
      col_order: classes reordered to match the same ordering (optional)
      mapping: dict cluster -> matched class
    """
    # Hungarian: maximize sum of matched counts => minimize (-cm)
    r, c = linear_sum_assignment(-cm)
    mapping = {int(rr): int(cc) for rr, cc in zip(r, c)}

    # Sort pairs by matched class index to create a diagonal view
    pairs = sorted(zip(r, c), key=lambda x: x[1])
    row_order = [int(rr) for rr, _ in pairs]
    col_order = [int(cc) for _, cc in pairs]
    return row_order, col_order, mapping

def plot_2d(X, y_pred, method="pca", title="2D plot", sample=3000, random_state=0):
    import numpy as np
    import matplotlib.pyplot as plt

    X = np.asarray(X)
    y_pred = np.asarray(y_pred)

    n = X.shape[0]
    if sample is not None and n > sample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(n, size=sample, replace=False)
        Xs = X[idx]
        ys = y_pred[idx]
    else:
        Xs = X
        ys = y_pred

    method = method.lower().strip()

    if method == "pca":
        from sklearn.decomposition import PCA
        Z = PCA(n_components=2, random_state=random_state).fit_transform(Xs)

    elif method in ("tsne", "t-sne"):
        from sklearn.manifold import TSNE
        Z = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=random_state).fit_transform(Xs)

    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca' or 'tsne'.")

    plt.figure(figsize=(7, 6))
    plt.scatter(Z[:, 0], Z[:, 1], c=ys, s=6)
    plt.title(title)
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")
