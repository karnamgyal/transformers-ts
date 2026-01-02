import torch
from src.models.model import TimeSeriesTransformer  # adjust import to your file name

def main():
    model = TimeSeriesTransformer(
        n_channels=3,
        patch_len=16,
        stride=8,
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
    )

    x = torch.randn(2, 3, 256)
    y = model(x)

    print("y shape:", y.shape)
    y.mean().backward()
    print("backward: OK")

if __name__ == "__main__":
    main()
