# Transforms for Time-Series Data
# Patching from paper: "A Time-Series is Worth 64 Words"
import torch

def patchify(x: torch.Tensor, patch_len: int, stride: int) -> torch.Tensor:
    """
    Convert time-series into patches.
    Args:
        x: Input tensor of shape [B, C, L]
        patch_len: Length of each patch
        stride: Step between patches
    Returns:
        Patches tensor of shape [B, C, N_patches, patch_len]
    """
    B, C, L = x.shape
    patches = x.unfold(dimension=2, size=patch_len, step=stride)
    return patches
