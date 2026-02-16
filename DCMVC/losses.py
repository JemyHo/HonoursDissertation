from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors


# ---------------------------
# Basic utilities
# ---------------------------

def recon_loss_mse(x_views: List[torch.Tensor], xh_views: List[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for x, xh in zip(x_views, xh_views):
        loss = loss + F.mse_loss(xh, x)
    return loss / len(x_views)


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, dim=1)
    b_n = F.normalize(b, dim=1)
    return a_n @ b_n.T


# ---------------------------
# DCD: centroids + loss
# ---------------------------

@torch.no_grad()
def compute_global_centroids(
    z_all: torch.Tensor,                 # [N, D]
    z_views_all: List[torch.Tensor],     # V * [N, D]
    assign_all: np.ndarray,              # [N]
    K: int,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Epoch-level centroids (detached).

    Used as a stable fallback when a mini-batch misses some clusters.
    """
    device = z_all.device
    D = z_all.shape[1]
    mu = torch.zeros((K, D), device=device, dtype=z_all.dtype)
    mu_v = [torch.zeros((K, D), device=device, dtype=z_all.dtype) for _ in z_views_all]

    assign_t = torch.from_numpy(assign_all.astype(np.int64)).to(device)

    for k in range(K):
        mask = (assign_t == k)
        if mask.any():
            mu[k] = F.normalize(z_all[mask].sum(dim=0), dim=0)
            for v in range(len(z_views_all)):
                mu_v[v][k] = F.normalize(z_views_all[v][mask].sum(dim=0), dim=0)
        else:
            # empty cluster from KMeans is rare but possible; use random unit vector
            mu[k] = F.normalize(torch.randn(D, device=device), dim=0)
            for v in range(len(z_views_all)):
                mu_v[v][k] = F.normalize(torch.randn(D, device=device), dim=0)

    return mu.detach(), [m.detach() for m in mu_v]


def dcd_loss(
    z_cons: torch.Tensor,                 # [M, D]
    z_views: List[torch.Tensor],          # V*[M, D]
    assign_batch: torch.Tensor,           # [M]
    K: int,
    tauC: float,
    mu_global: torch.Tensor,              # [K, D]
    mu_v_global: List[torch.Tensor],      # V*[K, D]
    eps: float = 1e-12,
) -> torch.Tensor:
    """Dynamic Cluster Diffusion loss.

    Implementation matches the paper structure:
      - pos: (mu_k, mu_k^v)
      - neg: (mu_k, mu_j), j != k

    We compute centroids from the batch when possible; otherwise fall back to epoch-global centroids.
    """
    V = len(z_views)

    # batch centroids with fallback
    mu = mu_global.clone()
    mu_v = [mv.clone() for mv in mu_v_global]

    for k in range(K):
        mask = (assign_batch == k)
        if mask.any():
            mu[k] = F.normalize(z_cons[mask].sum(dim=0), dim=0)
            for v in range(V):
                mu_v[v][k] = F.normalize(z_views[v][mask].sum(dim=0), dim=0)

    mu = F.normalize(mu, dim=1)
    sim_mm = mu @ mu.T  # [K,K]

    loss = 0.0
    for v in range(V):
        muv = F.normalize(mu_v[v], dim=1)
        pos_sim = torch.sum(mu * muv, dim=1)           # [K]
        pos = torch.exp(pos_sim / tauC)                # [K]

        neg = torch.exp(sim_mm / tauC)                 # [K,K]
        neg = neg.sum(dim=1) - torch.exp(torch.diag(sim_mm) / tauC)  # exclude self

        loss_v = -torch.log((pos + eps) / (pos + neg + eps))
        loss = loss + loss_v.mean()

    return loss / V


# ---------------------------
# RNGPA: kNN graph + loss
# ---------------------------

def build_knn_lists_full(X: np.ndarray, k: int) -> List[np.ndarray]:
    """Build full-dataset kNN lists (cosine distance)."""
    nn_model = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), metric="cosine", n_jobs=-1)
    nn_model.fit(X)
    _, inds = nn_model.kneighbors(X, return_distance=True)

    out: List[np.ndarray] = []
    for i in range(X.shape[0]):
        neigh = inds[i]
        neigh = neigh[neigh != i]
        out.append(neigh[:k])
    return out


def batch_adj_from_knn(batch_indices: np.ndarray, knn_lists: List[np.ndarray]) -> torch.Tensor:
    """Build mutual-kNN adjacency for a batch using precomputed full-data knn lists."""
    M = batch_indices.shape[0]
    pos = {int(idx): i for i, idx in enumerate(batch_indices.tolist())}
    adj = torch.zeros((M, M), dtype=torch.bool)

    for i, gidx in enumerate(batch_indices.tolist()):
        for n in knn_lists[int(gidx)]:
            n = int(n)
            if n in pos:
                adj[i, pos[n]] = True

    adj = adj | adj.T
    adj.fill_diagonal_(False)
    return adj


def batch_adj_from_features(x: torch.Tensor, k: int) -> torch.Tensor:
    """Build mutual-kNN adjacency inside a mini-batch using cosine similarity."""
    M = x.shape[0]
    x_n = F.normalize(x, dim=1)
    sim = x_n @ x_n.T
    sim.fill_diagonal_(-1e9)

    k_eff = min(k, M - 1)
    _, inds = torch.topk(sim, k=k_eff, dim=1, largest=True, sorted=False)

    adj = torch.zeros((M, M), dtype=torch.bool, device=x.device)
    row = torch.arange(M, device=x.device).unsqueeze(1).expand_as(inds)
    adj[row, inds] = True
    adj = adj | adj.T
    adj.fill_diagonal_(False)
    return adj


def rngpa_loss(
    z_cons: torch.Tensor,               # [M, D]
    z_views: List[torch.Tensor],        # V*[M, D]
    x_views: List[torch.Tensor],        # V*[M, d_v]
    assign_batch: torch.Tensor,         # [M]
    tauI: float,
    k: int,
    full_knn_lists: Optional[List[List[np.ndarray]]] = None,  # [V][N]
    batch_indices: Optional[np.ndarray] = None,               # [M]
    eps: float = 1e-12,
) -> torch.Tensor:
    """Reliable Neighbor-Guided Positive Alignment loss.

    Guardrails included to avoid empty positive/negative sets producing NaNs.
    """
    device = z_cons.device
    M = z_cons.shape[0]
    V = len(z_views)

    same = (assign_batch.unsqueeze(1) == assign_batch.unsqueeze(0))  # [M,M]
    eye = torch.eye(M, dtype=torch.bool, device=device)

    total = 0.0
    for v in range(V):
        # neighbor graph G^v
        if full_knn_lists is not None and batch_indices is not None:
            G = batch_adj_from_knn(batch_indices, full_knn_lists[v]).to(device)
        else:
            G = batch_adj_from_features(x_views[v], k=k).to(device)

        P = (G & same) & (~eye)
        N = ((~G) & (~same)) & (~eye)

        sim_z_zv = cosine_sim_matrix(z_cons, z_views[v])
        sim_zv_zv = cosine_sim_matrix(z_views[v], z_views[v])

        loss_v = 0.0
        for i in range(M):
            p_mask = P[i]
            n_mask = N[i]

            # guardrails
            if not p_mask.any():
                p_mask = same[i] & (~eye[i])
            if not n_mask.any():
                n_mask = (~same[i]) & (~eye[i])
                if not n_mask.any():
                    n_mask = ~eye[i]

            pos_sum = (sim_z_zv[i] * p_mask.float()).sum()
            numer = torch.exp(pos_sum / tauI)

            den1 = torch.exp(sim_z_zv[i][n_mask] / tauI).sum()
            den2 = torch.exp(sim_zv_zv[i][n_mask] / tauI).sum()
            denom = den1 + den2

            loss_v = loss_v + (-torch.log((numer + eps) / (denom + eps)))

        total = total + (loss_v / M)

    return total / V
