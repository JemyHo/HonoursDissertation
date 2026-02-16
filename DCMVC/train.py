from __future__ import annotations

from typing import List, Optional, Dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans

from .config import Config
from .utils import set_seed, standardize_views
from .data import MultiViewDataset
from .model import DCMVC
from .losses import (
    recon_loss_mse,
    compute_global_centroids,
    dcd_loss,
    rngpa_loss,
    build_knn_lists_full,
)
from .metrics import clustering_acc, nmi, ari


@torch.no_grad()
def encode_all(model: DCMVC, loader: DataLoader, device: torch.device):
    """Encode the full dataset to get consensus and per-view embeddings."""
    model.eval()

    z_chunks = []
    z_v_chunks = None
    idxs = []

    for x_views, idx, _ in loader:
        x_views = [x.to(device) for x in x_views]
        z_views, _, z_cons, _ = model(x_views)

        z_chunks.append(z_cons.detach().cpu())
        if z_v_chunks is None:
            z_v_chunks = [[] for _ in range(len(z_views))]
        for v in range(len(z_views)):
            z_v_chunks[v].append(z_views[v].detach().cpu())

        idxs.append(idx.numpy())

    z_all = torch.cat(z_chunks, dim=0)
    z_views_all = [torch.cat(ch, dim=0) for ch in z_v_chunks]
    idx_all = np.concatenate(idxs, axis=0)
    return z_all, z_views_all, idx_all


def train_dcmvc(
    views: List[np.ndarray],
    n_clusters: int,
    labels: Optional[np.ndarray] = None,
    cfg: Config = Config(),
) -> Dict[str, object]:
    """Train DCMVC in two stages:

    1) Warm-up: reconstruction only
    2) Fine-tune: EM loop
        - E-step: KMeans on consensus embedding -> pseudo labels
        - M-step: minimize L = Lrec + alpha*Ldcd + beta*Lrngpa
    """

    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    views = standardize_views(views)
    N = views[0].shape[0]
    in_dims = [v.shape[1] for v in views]

    ds = MultiViewDataset(views, labels=labels)

    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )

    full_loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    model = DCMVC(in_dims=in_dims, latent_dim=cfg.latent_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    # RNGPA neighbor graph strategy
    use_full_graph = (N <= 10_000)
    k = cfg.k_small if use_full_graph else cfg.k_large

    full_knn_lists = None
    if use_full_graph:
        full_knn_lists = []
        for v in range(len(views)):
            full_knn_lists.append(build_knn_lists_full(views[v], k=k))

    # ----------------------
    # Warm-up
    # ----------------------
    model.train()
    for ep in range(cfg.warmup_epochs):
        total_loss = 0.0
        for x_views, _, _ in loader:
            x_views = [x.to(device) for x in x_views]
            z_views, xh_views, z_cons, w = model(x_views)

            loss = recon_loss_mse(x_views, xh_views)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu())

        if (ep + 1) % 20 == 0:
            print(
                f"[Warm-up] epoch {ep+1:03d}/{cfg.warmup_epochs} "
                f"Lrec={total_loss/len(loader):.6f} "
                f"w={w.detach().cpu().numpy()}"
            )

    # ----------------------
    # Fine-tune (EM)
    # ----------------------
    assign_all = None
    mu_global = None
    mu_v_global = None

    for ep in range(cfg.finetune_epochs):
        # E-step: KMeans on full consensus embeddings
        z_all_cpu, z_views_all_cpu, idx_all = encode_all(model, full_loader, device)
        z_all = z_all_cpu.to(device)
        z_views_all = [zv.to(device) for zv in z_views_all_cpu]

        km = KMeans(n_clusters=n_clusters, n_init=20, max_iter=300, random_state=cfg.seed)
        km.fit(z_all_cpu.numpy())
        assign_all = km.labels_

        # epoch-level centroids (fallback when batch misses clusters)
        mu_global, mu_v_global = compute_global_centroids(z_all, z_views_all, assign_all, n_clusters)

        # M-step
        model.train()
        totals = {"rec": 0.0, "dcd": 0.0, "rng": 0.0, "all": 0.0}

        for x_views, idx, _ in loader:
            x_views = [x.to(device) for x in x_views]
            idx_np = idx.numpy()
            assign_batch = torch.from_numpy(assign_all[idx_np].astype(np.int64)).to(device)

            z_views, xh_views, z_cons, w = model(x_views)

            l_rec = recon_loss_mse(x_views, xh_views)
            l_dcd = dcd_loss(
                z_cons=z_cons,
                z_views=z_views,
                assign_batch=assign_batch,
                K=n_clusters,
                tauC=cfg.tauC,
                mu_global=mu_global,
                mu_v_global=mu_v_global,
            )
            l_rng = rngpa_loss(
                z_cons=z_cons,
                z_views=z_views,
                x_views=x_views,
                assign_batch=assign_batch,
                tauI=cfg.tauI,
                k=k,
                full_knn_lists=full_knn_lists,
                batch_indices=idx_np if use_full_graph else None,
            )

            loss = l_rec + cfg.alpha * l_dcd + cfg.beta * l_rng

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            totals["rec"] += float(l_rec.detach().cpu())
            totals["dcd"] += float(l_dcd.detach().cpu())
            totals["rng"] += float(l_rng.detach().cpu())
            totals["all"] += float(loss.detach().cpu())

        if (ep + 1) % 10 == 0:
            print(
                f"[Fine-tune] epoch {ep+1:03d}/{cfg.finetune_epochs} "
                f"L={totals['all']/len(loader):.6f} "
                f"Lrec={totals['rec']/len(loader):.6f} "
                f"Ldcd={totals['dcd']/len(loader):.6f} "
                f"Lrng={totals['rng']/len(loader):.6f} "
                f"w={w.detach().cpu().numpy()}"
            )

    # final encode + KMeans
    z_all_cpu, _, _ = encode_all(model, full_loader, device)
    km_final = KMeans(n_clusters=n_clusters, n_init=50, max_iter=500, random_state=cfg.seed)
    pred = km_final.fit_predict(z_all_cpu.numpy())

    results: Dict[str, object] = {"model": model, "z": z_all_cpu.numpy(), "pred": pred}

    if labels is not None:
        results["acc"] = clustering_acc(labels, pred)
        results["nmi"] = nmi(labels, pred)
        results["ari"] = ari(labels, pred)
        print(f"[Final] ACC={results['acc']:.4f} NMI={results['nmi']:.4f} ARI={results['ari']:.4f}")

    return results
