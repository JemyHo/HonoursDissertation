import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def plot_heatmap_clusters_vs_true(M, class_names=None, title="Clusters × True Classes", max_ticks=25):
    K, C = M.shape
    plt.figure(figsize=(10, 7))
    plt.imshow(M, aspect="auto")
    plt.colorbar(label="count")
    plt.xlabel("True class")
    plt.ylabel("Cluster")
    plt.title(title)

    # ticks (don’t spam 50 labels unless you want chaos)
    if class_names is not None and C <= max_ticks:
        plt.xticks(range(C), class_names, rotation=90, fontsize=8)
    elif class_names is not None:
        step = max(1, C // 10)
        xs = list(range(0, C, step))
        plt.xticks(xs, [class_names[i] for i in xs], rotation=90, fontsize=8)
    plt.tight_layout()
    plt.show()

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
