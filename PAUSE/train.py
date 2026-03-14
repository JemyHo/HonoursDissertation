from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np
import torch

from custom_dataset_loader import load_dataset
from custom_train_epoch import train_warmup_epoch, train_universum_epoch
from model import MultiViewAutoencoderWithClustering
from utils import evaluate_with_seed, save_pseudo_labels, get_pseudo_label_path, set_random_seed


def build_parser():
    p = argparse.ArgumentParser(description="PAUSE adapted for MFeat / Reuters / AwA2")
    p.add_argument("--dataset", choices=["MFeat", "Reuters", "AwA2"], required=True)
    p.add_argument("--data_root", type=str, required=True, help="Project root containing Dataset/")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--warmup_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup_lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--warmup_weight_decay", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--beta", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.05)
    p.add_argument("--ratio", type=float, default=2.5)
    p.add_argument("--drop_rate", type=float, default=0.5)
    p.add_argument("--feature_drop_rate", type=float, default=0.4)
    p.add_argument("--augmentation_strategy", type=str, default="feature_dropout", choices=["feature_dropout", "selective_dropout"])
    p.add_argument("--data_norm", type=str, default="min-max", choices=["standard", "min-max", "l2-norm"])
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=1024)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--print_freq", type=int, default=1)
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--output_log_dir", type=str, default="./logs")
    p.add_argument("--save_ckpt", action="store_true")
    return p


def build_model_dims(input_dims, hidden_dim, latent_dim):
    recognition = [[d, hidden_dim, hidden_dim, latent_dim] for d in input_dims]
    generative = [[latent_dim, hidden_dim, hidden_dim, d] for d in input_dims]
    return recognition, generative


def save_best_artifacts(output_dir, dataset, seed, epoch, acc, nmi, ari, y_pred, y_true, H):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    preds_dir = Path(output_dir) / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    with open(Path(output_dir) / f"{dataset}_seed{seed}_best.json", "w", encoding="utf-8") as f:
        json.dump({"ACC": float(acc), "NMI": float(nmi), "ARI": float(ari), "epoch": int(epoch)}, f, indent=2)

    np.save(preds_dir / f"{dataset}_seed{seed}_ypred.npy", y_pred)
    np.save(preds_dir / f"{dataset}_seed{seed}_ytrue.npy", y_true)
    np.save(preds_dir / f"{dataset}_seed{seed}_H.npy", H)


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    set_random_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.output_log_dir, exist_ok=True)

    dataset, input_dims = load_dataset(args, use_pseudo_labels=False)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    recognition_dims, generative_dims = build_model_dims(input_dims, args.hidden_dim, args.latent_dim)
    model = MultiViewAutoencoderWithClustering(
        n_views=args.n_views,
        recognition_model_dims=recognition_dims,
        generative_model_dims=generative_dims,
        temperature=args.temperature,
        n_clusters=args.n_clusters,
        drop_rate=args.drop_rate,
        args=args,
    ).to(args.device)

    history_path = Path(args.output_dir) / f"{args.dataset}_seed{args.seed}_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f_hist:
        writer = csv.DictWriter(f_hist, fieldnames=["epoch", "stage", "train_loss", "ACC", "NMI", "ARI"])
        writer.writeheader()

        best = {"acc": -1.0, "nmi": -1.0, "ari": -1.0, "epoch": -1, "y_pred": None, "y_true": None, "H": None}

        # warmup
        optimizer = torch.optim.Adam(model.parameters(), lr=args.warmup_lr, weight_decay=args.warmup_weight_decay)
        for epoch in range(1, args.warmup_epochs + 1):
            loss = train_warmup_epoch(model, train_loader, optimizer, epoch, args.device, args)
            acc, nmi, ari, pred_labels = evaluate_with_seed(model, test_loader, args.device, args.n_clusters, args.seed)
            print(f"Warmup epoch {epoch}/{args.warmup_epochs} | loss={loss:.4f} | ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
            writer.writerow({"epoch": epoch, "stage": "warmup", "train_loss": loss, "ACC": acc, "NMI": nmi, "ARI": ari})
            if epoch == args.warmup_epochs:
                save_pseudo_labels(pred_labels, get_pseudo_label_path(args))
            if acc > best["acc"]:
                # extract embeddings + labels for saving
                model.eval()
                feats, true = [], []
                with torch.no_grad():
                    for _, views, labels in test_loader:
                        views = [v.to(args.device) for v in views]
                        z, _ = model(*views)
                        feats.append(torch.cat(z, dim=1).cpu().numpy())
                        true.append(labels.cpu().numpy())
                H = np.vstack(feats)
                y_true = np.concatenate(true)
                best.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch, "y_pred": pred_labels, "y_true": y_true, "H": H})

        # universum
        dataset_pseudo, _ = load_dataset(args, use_pseudo_labels=True)
        train_loader = torch.utils.data.DataLoader(dataset_pseudo, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.warmup_epochs + 1, args.epochs + 1):
            loss = train_universum_epoch(model, train_loader, optimizer, epoch, args.device, args)
            acc, nmi, ari, pred_labels = evaluate_with_seed(model, test_loader, args.device, args.n_clusters, args.seed)
            print(f"Epoch {epoch}/{args.epochs} | loss={loss:.4f} | ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f}")
            writer.writerow({"epoch": epoch, "stage": "universum", "train_loss": loss, "ACC": acc, "NMI": nmi, "ARI": ari})
            if acc > best["acc"]:
                model.eval()
                feats, true = [], []
                with torch.no_grad():
                    for _, views, labels in test_loader:
                        views = [v.to(args.device) for v in views]
                        z, _ = model(*views)
                        feats.append(torch.cat(z, dim=1).cpu().numpy())
                        true.append(labels.cpu().numpy())
                H = np.vstack(feats)
                y_true = np.concatenate(true)
                best.update({"acc": acc, "nmi": nmi, "ari": ari, "epoch": epoch, "y_pred": pred_labels, "y_true": y_true, "H": H})

    save_best_artifacts(args.output_dir, args.dataset, args.seed, best["epoch"], best["acc"], best["nmi"], best["ari"], best["y_pred"], best["y_true"], best["H"])
    print(f"Best epoch={best['epoch']} ACC={best['acc']:.4f} NMI={best['nmi']:.4f} ARI={best['ari']:.4f}")


if __name__ == "__main__":
    main()
