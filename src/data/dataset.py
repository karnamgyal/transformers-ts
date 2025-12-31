# Synthetic Dataset for Time-Series Data
import torch
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, n_samples: int, seq_len: int, n_channels: int, seed: int = 41):
        self.n_samples = n_samples
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.g = torch.Generator().manual_seed(seed)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Random time-series: [C, L]
        x = torch.randn(self.n_channels, self.seq_len, generator=self.g)
        # Optional: normalize per-sample (helps training stability)
        x = (x - x.mean()) / (x.std() + 1e-6)
        return x.float()

def build_loader(cfg: dict):
    data = cfg["data"]
    ds = SyntheticDataset(
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
