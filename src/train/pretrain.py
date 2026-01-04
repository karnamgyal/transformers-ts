# Self-supervised pretraining with masked patch reconstruction 
import os
os.makedirs("checkpoints", exist_ok=True)

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import build_loader
from src.models.model import TimeSeriesTransformer


def make_patch_targets(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    Convert raw time-series into flattened patch targets.

    Args:
        x: Input tensor of shape [B, C, L]
        patch_len: Length of each patch
        stride: Patch stride

    Returns:
        targets: Tensor of shape [B, N, C * patch_len]
    """
    patches = x.unfold(dimension=2, size=patch_len, step=stride)
    patches = patches.permute(0, 2, 1, 3).contiguous()
    targets = patches.view(patches.size(0), patches.size(1), -1)
    return targets


def mask_tokens(tokens: torch.Tensor, mask_ratio: float):
    """
    Randomly mask a subset of patch tokens.

    Args:
        tokens: Patch tokens of shape [B, N, D]
        mask_ratio: Fraction of patches to mask

    Returns:
        tokens_masked: Tokens with masked positions zeroed out
        mask: Boolean mask indicating which patches were masked
    """
    B, N, _ = tokens.shape
    mask = torch.rand(B, N, device=tokens.device) < mask_ratio
    tokens_masked = tokens.clone()
    tokens_masked[mask] = 0.0
    return tokens_masked, mask


def main(cfg: dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    patch_len = int(cfg["model"]["patch_len"])
    stride = int(cfg["model"]["stride"])
    d_model = int(cfg["model"]["d_model"])
    n_heads = int(cfg["model"]["n_heads"])
    n_layers = int(cfg["model"]["n_layers"])

    n_channels = int(cfg["data"]["n_channels"])
    seq_len = int(cfg["data"]["seq_len"])

    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])
    log_interval = int(cfg["train"].get("log_interval", 10))

    mask_ratio = cfg.get("ssl", {}).get("mask_ratio", 0.5)

    loader = build_loader(cfg)

    model = TimeSeriesTransformer(
        n_channels=n_channels,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    patch_dim = n_channels * patch_len
    decoder = nn.Linear(d_model, patch_dim).to(device)

    optimizer = optim.AdamW(
        list(model.parameters()) + list(decoder.parameters()),
        lr=lr
    )
    loss_fn = nn.MSELoss()

    model.train()
    decoder.train()

    step = 0
    for epoch in range(epochs):
        for x in loader:
            x = x.to(device)

            targets = make_patch_targets(x, patch_len, stride)

            tokens = model.embedding(x)
            tokens_masked, mask = mask_tokens(tokens, mask_ratio)

            z = model.encoder(tokens_masked)
            pred = decoder(z)

            loss = loss_fn(pred[mask], targets[mask])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.6f}")

            step += 1

    torch.save(
    {
        "model": model.state_dict(),
        "decoder": decoder.state_dict(),
        "cfg": cfg,
    },
    "checkpoints/pretrain.pt",
    )
    print("pretraining done")

if __name__ == "__main__":
    from src.utils.config import load_config
    cfg = load_config("configs/pretrain.yaml")
    main(cfg)
