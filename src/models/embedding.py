# Embedding modules for time-series data.
import math
import torch
import torch.nn as nn

from src.data.transforms import patchify

class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.
    Input / Output: [B, N, D]
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                 # [N, D]
        pos = torch.arange(max_len).float().unsqueeze(1)   # [N, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, D]
        return x + self.pe[: x.size(1)].unsqueeze(0)
    
class PatchEmbedding(nn.Module):
    """
    Time-series patch embedding.

    [B, C, L]
      -> patchify
      -> [B, C, N, P]
      -> flatten
      -> [B, N, C*P]
      -> Linear
      -> [B, N, D]
      -> + PositionalEncoding
    """
    def __init__(
        self,
        n_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        max_patches: int = 10000,
        use_positional_encoding: bool = True,
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(n_channels * patch_len, d_model)
        self.pos = (
            PositionalEncoding(d_model, max_patches)
            if use_positional_encoding
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        patches = patchify(x, self.patch_len, self.stride) # [B, C, N, P]
        B, C, N, P = patches.shape
        patches = patches.permute(0, 2, 1, 3)              # [B, N, C, P]
        patches = patches.reshape(B, N, C * P)             # [B, N, C*P]
        tokens = self.proj(patches)                        # [B, N, D]
        return self.pos(tokens)
