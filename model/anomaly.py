"""Normal-behaviour anomaly model: a temporal autoencoder trained on normal shoppers only.

Input is the per-frame normalised pose (first 34 feature channels = 17 joints x (x, y))
plus joint confidences (channels 34:51) used to mask the reconstruction loss.
The anomaly score of a window is the confidence-weighted reconstruction error.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

XY = slice(0, 34)
CONF = slice(34, 51)


class PoseAutoencoder(nn.Module):
    def __init__(self, hidden: int = 128, latent: int = 32, in_dim: int = 34):
        super().__init__()
        self.in_dim = in_dim
        self.enc = nn.GRU(in_dim, hidden, batch_first=True)
        self.to_latent = nn.Linear(hidden, latent)
        self.from_latent = nn.Linear(latent, hidden)
        self.dec = nn.GRU(hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, T, 34)
        _, h = self.enc(x)
        z = self.to_latent(h[-1])
        h0 = self.from_latent(z).unsqueeze(1).expand(-1, x.shape[1], -1)
        y, _ = self.dec(h0)
        return self.out(y)


def prepare(X: np.ndarray, clip: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    """Return (xy, weight) with xy clipped to a sane range and joint-confidence weights."""
    xy = np.clip(np.nan_to_num(X[:, :, XY]), -clip, clip).astype(np.float32)
    conf = np.clip(np.nan_to_num(X[:, :, CONF]), 0.0, 1.0).astype(np.float32)
    w = np.repeat(conf, 2, axis=2)  # (N, T, 34): same weight for x and y
    return xy, w


def weighted_mse(pred: torch.Tensor, target: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    err = ((pred - target) ** 2) * w
    return err.sum(dim=(1, 2)) / w.sum(dim=(1, 2)).clamp(min=1e-6)


def fit(
    X: np.ndarray,
    epochs: int = 25,
    batch: int = 256,
    lr: float = 2e-3,
    seed: int = 0,
    device: str | None = None,
    noise: float = 0.02,
) -> PoseAutoencoder:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    xy, w = prepare(X)
    xy_t, w_t = torch.from_numpy(xy), torch.from_numpy(w)
    model = PoseAutoencoder().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(xy_t)
    for ep in range(epochs):
        perm = torch.randperm(n)
        total = 0.0
        model.train()
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            xb, wb = xy_t[idx].to(device), w_t[idx].to(device)
            pred = model(xb + noise * torch.randn_like(xb))
            loss = weighted_mse(pred, xb, wb).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(idx)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1}/{epochs} loss {total / n:.4f}", flush=True)
    return model.eval()


@torch.no_grad()
def score(model: PoseAutoencoder, X: np.ndarray, batch: int = 1024) -> np.ndarray:
    device = next(model.parameters()).device
    xy, w = prepare(X)
    out = []
    for i in range(0, len(xy), batch):
        xb = torch.from_numpy(xy[i : i + batch]).to(device)
        wb = torch.from_numpy(w[i : i + batch]).to(device)
        out.append(weighted_mse(model(xb), xb, wb).cpu().numpy())
    return np.concatenate(out) if out else np.empty(0)
