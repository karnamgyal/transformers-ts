# Model wrapping embedding and transformer
import torch
import torch.nn as nn   
from src.models.embedding import PatchEmbedding
from src.models.transformer import TransformerEncoder

class TimeSeriesTransformer(nn.Module):
    """
    Time-series transformer model.

    Input: [B, C, L]
      -> PatchEmbedding
      -> [B, N, D]
      -> TransformerEncoder
      -> [B, N, D]
    """
    def __init__(
        self,
        n_channels: int,
        patch_len: int,
        stride: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        d_ff: int = None,
        dropout: float = 0.1,
        max_patches: int = 10000,
        use_positional_encoding: bool = True,
    ):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = PatchEmbedding(
            n_channels,
            patch_len,
            stride,
            d_model,
            max_patches,
            use_positional_encoding,
        )
        self.encoder = TransformerEncoder(
            n_layers,
            d_model,
            n_heads,
            d_ff,
            dropout,
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # x: [B, C, L]
        x = self.embedding(x)  # [B, N, D]
        x = self.encoder(x, attn_mask=attn_mask)  # [B, N, D]
        return x
