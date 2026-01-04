import torch
import torch.nn as nn

class ForecastHead(nn.Module):
    def __init__(self, d_model: int, n_channels: int, horizon: int):
        super().__init__()
        self.n_channels = n_channels
        self.horizon = horizon
        self.proj = nn.Linear(d_model, n_channels * horizon)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = z.mean(dim=1)                      # [B, D]
        out = self.proj(x)                     # [B, C*T]
        return out.view(out.size(0), self.n_channels, self.horizon)  # [B, C, T]
