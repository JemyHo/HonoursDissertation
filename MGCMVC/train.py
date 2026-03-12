
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from dataloader import load_data
from loss import Loss
from metric import valid
from network import Network


def parse_args():
    parser = argparse.ArgumentParser(description='Train MGCMVC on adapted datasets')
    parser.add_argument('--dataset', default='MFeat', choices=['MFeat', 'Reuters', 'AwA2'])
    parser.add_argument('--data_root', default='.')
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--mse_epochs', default=200, type=int)
    parser.add_argument('--con_epochs', default=100, type=int)
    parser.add_argument('--temperature_f', default=0.5, type=float)
    parser.add_argument('--temperature_l', default=1.0, type=float)
    parser.add_argument('--learning_rate', default=3e-4, type=float)
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--feature_dim', default=128, type=int)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--save_model', action='store_true', default=False)
    parser.add_argument('--out_dir', default='outputs')
    return parser.parse_args()


def setup_seed(seed: int):
    print('seed:', seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_view_value(rs, H, view):
    w = []
    H_np = H.detach().cpu().numpy().flatten()
    for v in range(view):
        w.append(torch.exp(-torch.tensor(wasserstein_distance(H_np, rs[v].detach().cpu().numpy().flatten()))))
    w = torch.stack(w)
    w = w / torch.sum(w)
    return w.squeeze()


def save_results_table(results, filename):
    df = pd.DataFrame(results, columns=['Epoch', 'Loss', 'ACC', 'NMI', 'PUR', 'ARI'])
    df.to_csv(filename, index=False)
    print(f'Results saved to {filename}')


def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    setup_seed(args.seed)

    dataset, input_dims, view, data_size, class_num, class_names = load_data(args.dataset, data_root=args.data_root)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=min(args.batch_size, data_size),
        shuffle=True,
        drop_last=True,
    )

    low_feature_dim = 128
    high_feature_dim = 256
    dims = [500, 500, 2000]
    model = Network(view, input_dims, low_feature_dim, high_feature_dim, dims, class_num, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(min(args.batch_size, data_size), class_num, args.temperature_f, args.temperature_l, device).to(device)
    mse = torch.nn.MSELoss()

    out_dir = Path(args.out_dir)
    pred_dir = out_dir / 'preds'
    model_dir = out_dir / 'models'
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(epoch: int):
        model.train()
        tot_loss = 0.0
        for xs, _, _ in data_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            _, _, _, xrs, _ = model(xs)
            loss = sum(mse(xs[v], xrs[v]) for v in range(view))
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
        avg = tot_loss / max(len(data_loader), 1)
        print(f'Pretrain Epoch {epoch} Loss:{avg:.6f}')

    def contrastive_train(epoch: int):
        model.train()
        tot_loss = 0.0
        for xs, _, _ in data_loader:
            for v in range(view):
                xs[v] = xs[v].to(device)
            optimizer.zero_grad()
            hs, qs, zs, xrs, H = model(xs)
            with torch.no_grad():
                weights = compute_view_value(hs, H, view)
            loss_list = []
            for v in range(view):
                for w in range(v + 1, view):
                    loss_list.append((weights[v] + weights[w]) * criterion.forward_feature(hs[v], hs[w]))
                    loss_list.append(criterion.forward_label(qs[v], qs[w]))
                loss_list.append(mse(xs[v], xrs[v]))
            loss = sum(loss_list)
            loss.backward()
            optimizer.step()
            tot_loss += float(loss.item())
        avg = tot_loss / max(len(data_loader), 1)
        print(f'Contrastive Epoch {epoch} Loss:{avg:.6f}')
        return avg

    print('==================================')
    print('Args:', args)
    print('Device:', device)
    print('Input dims:', input_dims, 'views:', view, 'samples:', data_size, 'classes:', class_num)
    print('==================================')

    for epoch in range(1, args.mse_epochs + 1):
        pretrain(epoch)

    results = []
    best_acc = -1.0
    best = None
    best_state = None

    for epoch in range(args.mse_epochs + 1, args.mse_epochs + args.con_epochs + 1):
        loss = contrastive_train(epoch)
        acc, nmi, pur, ari = valid(
            model, device, dataset, view, data_size, class_num,
            eval_h=False, epoch=epoch, dataset_name=args.dataset,
            seed=args.seed, save_dir=str(pred_dir)
        )
        results.append([epoch, loss, acc, nmi, pur, ari])
        if acc > best_acc:
            best_acc = acc
            best = {'ACC': acc, 'NMI': nmi, 'PUR': pur, 'ARI': ari, 'epoch': epoch}
            if args.save_model:
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if args.save_model and best_state is not None:
        torch.save(best_state, model_dir / f'{args.dataset}_seed{args.seed}.pth')

    print('Best clustering performance: ACC = {:.4f} NMI = {:.4f} ARI = {:.4f} PUR = {:.4f} @ epoch {}'.format(
        best['ACC'], best['NMI'], best['ARI'], best['PUR'], best['epoch']))

    save_results_table(results, out_dir / f'{args.dataset}_seed{args.seed}_history.csv')
    with open(out_dir / f'{args.dataset}_seed{args.seed}_best.json', 'w', encoding='utf-8') as f:
        json.dump(best, f, indent=2)


if __name__ == '__main__':
    main()
