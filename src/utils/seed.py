# Seed Utility
import random
import numpy as np
import torch

# Seed Setter
def set_seed(seed: int = 41):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
