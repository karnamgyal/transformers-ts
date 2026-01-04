# Synthetic Dataset for Time-Series Data 
import math
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


class StructuredSyntheticDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, n_channels: int, seed: int = 41):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        t = torch.linspace(0, 1, self.seq_len)

        x = torch.empty(self.n_channels, self.seq_len)

        base_freqs = torch.randint(2, 10, (self.n_channels,), generator=self.g).float()
        phases = 2 * math.pi * torch.rand(self.n_channels, generator=self.g)
        amps = 0.6 + 0.8 * torch.rand(self.n_channels, generator=self.g)

        trend_slope = (torch.rand(self.n_channels, generator=self.g) - 0.5) * 0.8
        trend = trend_slope.unsqueeze(1) * t.unsqueeze(0)

        for c in range(self.n_channels):
            f = base_freqs[c].item()
            sig = amps[c] * torch.sin(2 * math.pi * f * t + phases[c])
            sig = sig + 0.3 * torch.sin(2 * math.pi * (f * 0.5) * t + phases[c] * 0.7)

            noise = 0.10 * torch.randn(self.seq_len, generator=self.g)

            ar = torch.zeros(self.seq_len)
            ar_eps = 0.05 * torch.randn(self.seq_len, generator=self.g)
            for i in range(1, self.seq_len):
                ar[i] = 0.8 * ar[i - 1] + ar_eps[i]

            x[c] = sig + trend[c] + noise + ar

        x = (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)
        return x.float()


class StructuredForecastDataset(Dataset):
    def __init__(
        self,
        n_samples: int,
        context_len: int,
        horizon: int,
        n_channels: int,
        seed: int = 41,
    ):
        self.n_samples = n_samples
        self.context_len = context_len
        self.horizon = horizon
        self.n_channels = n_channels
        self.total_len = context_len + horizon
        self.base = StructuredSyntheticDataset(
            n_samples=n_samples,
            seq_len=self.total_len,
            n_channels=n_channels,
            seed=seed,
        )

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        full = self.base[idx]  # [C, context+horizon]
        x = full[:, : self.context_len]                 # [C, L]
        y = full[:, self.context_len : self.total_len]  # [C, T]
        return x, y


def build_loader(cfg: dict):
    data = cfg["data"]
    ds = StructuredSyntheticDataset(
        n_samples=int(data.get("n_samples", 2000)),
        seq_len=int(data["seq_len"]),
        n_channels=int(data["n_channels"]),
        seed=int(cfg.get("seed", 41)),
    )
    return DataLoader(
        ds,
        batch_size=int(data["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(data.get("num_workers", 0)),
    )


def build_forecast_loaders(cfg: dict):
    data = cfg["data"]
    fcfg = cfg["forecast"]

    ds = StructuredForecastDataset(
        n_samples=int(data.get("n_samples", 2000)),
        context_len=int(fcfg["context_len"]),
        horizon=int(fcfg["horizon"]),
        n_channels=int(data["n_channels"]),
        seed=int(cfg.get("seed", 41)),
    )

    n = len(ds)
    n_train = int(0.8 * n)
    n_val = n - n_train

    train_ds, val_ds = random_split(
        ds,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.get("seed", 41)),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(data["batch_size"]),
        shuffle=True,
        drop_last=True,
        num_workers=int(data.get("num_workers", 0)),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(data["batch_size"]),
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
    )

    return train_loader, val_loader
