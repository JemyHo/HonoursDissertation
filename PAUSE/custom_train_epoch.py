import torch
from itertools import combinations
from utils import Info_Nce_Loss, compute_single_uni_loss_new, data_augmentation


def _mean_pairwise(losses):
    if not losses:
        return torch.tensor(0.0)
    return sum(losses) / len(losses)


def train_warmup_epoch(model, train_loader, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    mse_loss_fn = torch.nn.MSELoss()

    for _, views, _labels in train_loader:
        views = [v.to(device) for v in views]
        optimizer.zero_grad()
        z, reconstructed = model(*views)
        z_aug = [data_augmentation(zi, args) for zi in z]

        batch_size = z[0].size(0)
        info_nce_loss_fn = Info_Nce_Loss(batch_size=batch_size, device=device, temperature=args.temperature)

        intra_losses = [info_nce_loss_fn(z[i], z_aug[i]) for i in range(len(z))]
        inter_losses = [info_nce_loss_fn(z[i], z[j]) for i, j in combinations(range(len(z)), 2)]
        reconstruction_loss = sum(mse_loss_fn(reconstructed[j], views[j]) for j in range(model.n_views))

        total_loss = args.alpha * _mean_pairwise(inter_losses) + reconstruction_loss + args.beta * _mean_pairwise(intra_losses)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    return running_loss / max(len(train_loader), 1)


def train_universum_epoch(model, train_loader, optimizer, epoch, device, args):
    model.train()
    running_loss = 0.0
    mse_loss_fn = torch.nn.MSELoss()

    for _, views, _labels, pseudo_labels in train_loader:
        views = [v.to(device) for v in views]
        pseudo_labels = pseudo_labels.to(device)

        optimizer.zero_grad()
        z, reconstructed = model(*views)
        z_aug = [data_augmentation(zi, args) for zi in z]

        reconstruction_loss = sum(mse_loss_fn(reconstructed[j], views[j]) for j in range(model.n_views))
        inter_uni_losses = [compute_single_uni_loss_new(z[i], z[j], pseudo_labels, temperature=args.temperature)
                            for i, j in combinations(range(len(z)), 2)]
        intra_uni_losses = [compute_single_uni_loss_new(z[i], z_aug[i], pseudo_labels, temperature=args.temperature)
                            for i in range(len(z))]

        gamma2 = args.gamma / args.ratio
        total_loss = reconstruction_loss + args.gamma * _mean_pairwise(inter_uni_losses) + gamma2 * _mean_pairwise(intra_uni_losses)
        total_loss.backward()
        optimizer.step()
        running_loss += total_loss.item()

    return running_loss / max(len(train_loader), 1)
