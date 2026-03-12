
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from torch.utils.data import DataLoader


def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(int(y_pred.max()), int(y_true.max())) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    r, c = linear_sum_assignment(w.max() - w)
    return float(sum(w[i, j] for i, j in zip(r, c)) / y_pred.size)


def purity(y_true, y_pred):
    y_true = np.asarray(y_true).copy()
    y_pred = np.asarray(y_pred)
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels) + 1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return float(accuracy_score(y_true, y_voted_labels))


def evaluate(label, pred):
    nmi = float(normalized_mutual_info_score(label, pred))
    ari = float(adjusted_rand_score(label, pred))
    acc = float(cluster_acc(label, pred))
    pur = float(purity(label, pred))
    return nmi, ari, acc, pur


def valid(model, device, dataset, view, data_size, class_num, eval_h=False,
          epoch: Optional[int] = None, dataset_name: Optional[str] = None,
          seed: Optional[int] = None, save_dir: str = 'preds'):
    test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
    for _, (xs, y, _) in enumerate(test_loader):
        for v in range(view):
            xs[v] = xs[v].to(device)
    labels = y.cpu().detach().data.numpy().squeeze()

    with torch.no_grad():
        hs, qs, zs, xrs, H = model(xs)
        qs, preds = model.forward_cluster(xs)
        q = sum(qs) / view
        q = np.argmax(q.cpu().detach().numpy(), axis=1)
        H_np = H.detach().cpu().numpy()

    if dataset_name is not None and seed is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f'{dataset_name}_seed{seed}_ypred.npy'), q)
        np.save(os.path.join(save_dir, f'{dataset_name}_seed{seed}_H.npy'), H_np)
        np.save(os.path.join(save_dir, f'{dataset_name}_seed{seed}_ytrue.npy'), labels)

    if eval_h:
        print('Clustering results on semantic labels: ' + str(labels.shape[0]))

    nmi, ari, acc, pur = evaluate(labels, q)
    print('ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR={:.4f}'.format(acc, nmi, ari, pur))
    return acc, nmi, pur, ari
