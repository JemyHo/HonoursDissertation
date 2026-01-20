import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def cluster_acc_hungarian(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    true_labels = np.unique(y_true)
    pred_labels = np.unique(y_pred)

    true_map = {lab: i for i, lab in enumerate(true_labels)}
    pred_map = {lab: i for i, lab in enumerate(pred_labels)}

    yt = np.vectorize(true_map.get)(y_true)
    yp = np.vectorize(pred_map.get)(y_pred)

    D = max(yt.max(), yp.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(yt.size):
        w[yp[i], yt[i]] += 1

    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    correct = w[row_ind, col_ind].sum()
    return correct / yt.size

def clustering_scores(y_true, y_pred):
    acc = cluster_acc_hungarian(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    ari = adjusted_rand_score(y_true, y_pred)
    return {"ACC": float(acc), "NMI": float(nmi), "ARI": float(ari)}
