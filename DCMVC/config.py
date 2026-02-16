from dataclasses import dataclass
import torch

@dataclass
class Config:
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 0

    latent_dim: int = 256
    lr: float = 1e-4
    batch_size: int = 256

    warmup_epochs: int = 200
    finetune_epochs: int = 100

    tauC: float = 0.5
    tauI: float = 0.5
    alpha: float = 0.1
    beta: float = 0.1

    k_small: int = 10   # N <= 10k
    k_large: int = 3    # N > 10k

    num_workers: int = 0
