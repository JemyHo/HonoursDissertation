from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP_AE(nn.Module):
    def __init__(self, in_dim: int, latent_dim: int = 256, hidden=(500, 500, 2000)):
        super().__init__()
        h1, h2, h3 = hidden
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(True),
            nn.Linear(h1, h2), nn.ReLU(True),
            nn.Linear(h2, h3), nn.ReLU(True),
            nn.Linear(h3, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, h3), nn.ReLU(True),
            nn.Linear(h3, h2), nn.ReLU(True),
            nn.Linear(h2, h1), nn.ReLU(True),
            nn.Linear(h1, in_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat

class DCMVC(nn.Module):
    def __init__(self, in_dims: List[int], latent_dim: int = 256):
        super().__init__()
        self.V = len(in_dims)
        self.aes = nn.ModuleList([MLP_AE(d, latent_dim=latent_dim) for d in in_dims])
        self.view_logits = nn.Parameter(torch.zeros(self.V))

    def forward(self, x_views: List[torch.Tensor]):
        z_views, xh_views = [], []
        for ae, x in zip(self.aes, x_views):
            z_v, xh_v = ae(x)
            z_views.append(z_v)
            xh_views.append(xh_v)

        w = F.softmax(self.view_logits, dim=0)
        z_cons = 0.0
        for v in range(self.V):
            z_cons = z_cons + w[v] * z_views[v]
        return z_views, xh_views, z_cons, w
