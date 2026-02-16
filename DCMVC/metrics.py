import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def clustering_acc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return w[r, c].sum() / y_pred.size

def nmi(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(normalized_mutual_info_score(y_true, y_pred))

def ari(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(adjusted_rand_score(y_true, y_pred))
