import torch
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import build_forecast_loaders
from src.models.model import TimeSeriesTransformer
from src.models.heads import ForecastHead


def load_pretrained(path: str, model: nn.Module, device: str):
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)


def main(cfg: dict):
    cfg_device = cfg.get("device", "auto")
    device = "cuda" if (cfg_device == "cuda" or (cfg_device == "auto" and torch.cuda.is_available())) else "cpu"

    patch_len = int(cfg["model"]["patch_len"])
    stride = int(cfg["model"]["stride"])
    d_model = int(cfg["model"]["d_model"])
    n_heads = int(cfg["model"]["n_heads"])
    n_layers = int(cfg["model"]["n_layers"])

    n_channels = int(cfg["data"]["n_channels"])

    lr = float(cfg["train"]["lr"])
    epochs = int(cfg["train"]["epochs"])
    log_interval = int(cfg["train"].get("log_interval", 10))

    horizon = int(cfg["forecast"]["horizon"])
    freeze_backbone = bool(cfg["forecast"].get("freeze_backbone", False))

    train_loader, val_loader = build_forecast_loaders(cfg)

    model = TimeSeriesTransformer(
        n_channels=n_channels,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
    ).to(device)

    pretrained_path = cfg["forecast"].get("pretrained_path", None)
    if pretrained_path:
        load_pretrained(pretrained_path, model, device)
        print("loaded pretrained:", pretrained_path)
    else:
        print("training from scratch")

    head = ForecastHead(d_model=d_model, n_channels=n_channels, horizon=horizon).to(device)

    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

    params = list(head.parameters()) + [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=lr)
    loss_fn = nn.MSELoss()

    def eval_val():
        model.eval()
        head.eval()
        total = 0.0
        count = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                z = model(x)
                pred = head(z)
                loss = loss_fn(pred, y)
                total += loss.item()
                count += 1
        model.train()
        head.train()
        return total / max(1, count)

    model.train()
    head.train()

    step = 0
    best_val = float("inf")

    for epoch in range(epochs):
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            z = model(x)
            pred = head(z)

            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % log_interval == 0:
                val_loss = eval_val()
                if val_loss < best_val:
                    best_val = val_loss
                print(
                    f"epoch {epoch} step {step} "
                    f"train_loss {loss.item():.6f} val_loss {val_loss:.6f} best_val {best_val:.6f}"
                )

            step += 1

    print("finetune done")


if __name__ == "__main__":
    from src.utils.config import load_config
    cfg = load_config("configs/finetune_forecast.yaml")
    main(cfg)
