"""Small temporal CNN: three 1-D convolutions along the window, then one output layer."""
from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

if TYPE_CHECKING:
    from ..data import Samples


class TemporalCNN(nn.Module):
    """Three Conv1d layers over time, then one linear layer.

    With `pool`, the linear layer sees the newest step plus the max and mean over the whole
    window. Without it, it sees the newest step only, which reaches back just
    3 * (kernel // 2) steps and ignores the rest of the window.
    """

    def __init__(self, in_channels: int, hidden: int = 64, kernel: int = 3, dropout: float = 0.1,
                 pool: bool = True):
        super().__init__()
        layers, c = [], in_channels
        for _ in range(3):
            layers += [nn.Conv1d(c, hidden, kernel, padding=kernel // 2), nn.ReLU(), nn.Dropout(dropout)]
            c = hidden
        self.convs = nn.Sequential(*layers)
        self.pool = pool
        self.head = nn.Linear(hidden * 3 if pool else hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, history, C] -> [B] logits."""
        h = self.convs(x.transpose(1, 2))
        out = h[:, :, -1]
        if self.pool:
            out = torch.cat([out, h.amax(dim=2), h.mean(dim=2)], dim=1)
        return self.head(out).squeeze(-1)


class CNNMonitor:
    """Trains a TemporalCNN and keeps the epoch with the lowest validation loss."""

    def __init__(self, seed: int = 0, hidden: int = 64, kernel: int = 3, dropout: float = 0.1,
                 pool: bool = True, lr: float = 1e-3, weight_decay: float = 1e-4, epochs: int = 20,
                 batch_size: int = 512, device: str | None = None):
        self.seed, self.epochs, self.batch_size = seed, epochs, batch_size
        self.lr, self.weight_decay = lr, weight_decay
        self.net_kwargs = dict(hidden=hidden, kernel=kernel, dropout=dropout, pool=pool)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, train: Samples, val: Samples) -> CNNMonitor:
        torch.manual_seed(self.seed)
        # Inputs are normalised per batch on the device, so the data is never copied whole.
        self.mean = torch.from_numpy(train.x.mean(axis=(0, 1), keepdims=True)).to(self.device)
        self.std = torch.from_numpy(train.x.std(axis=(0, 1), keepdims=True) + 1e-6).to(self.device)
        self.net = TemporalCNN(train.x.shape[-1], **self.net_kwargs).to(self.device)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        x = torch.from_numpy(train.x)
        y = torch.from_numpy(train.y.astype(np.float32))
        gen = torch.Generator().manual_seed(self.seed)
        best_loss, best_state = float("inf"), None
        for _ in range(self.epochs):
            self.net.train()
            for idx in torch.randperm(len(x), generator=gen).split(self.batch_size):
                xb, yb = self._normalise(x[idx]), y[idx].to(self.device)
                loss = nn.functional.binary_cross_entropy_with_logits(self.net(xb), yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
            p = np.clip(self.predict_proba(val), 1e-6, 1 - 1e-6)
            val_loss = float(-np.mean(val.y * np.log(p) + (1 - val.y) * np.log(1 - p)))
            if val_loss < best_loss:
                best_loss, best_state = val_loss, copy.deepcopy(self.net.state_dict())
        self.net.load_state_dict(best_state)
        return self

    def _normalise(self, batch: torch.Tensor) -> torch.Tensor:
        return (batch.to(self.device) - self.mean) / self.std

    @torch.no_grad()
    def predict_proba(self, samples: Samples) -> np.ndarray:
        """Probability of completion for each sample."""
        self.net.eval()
        x = torch.from_numpy(samples.x)
        out = [torch.sigmoid(self.net(self._normalise(xb))).cpu() for xb in x.split(4096)]
        return torch.cat(out).numpy()
