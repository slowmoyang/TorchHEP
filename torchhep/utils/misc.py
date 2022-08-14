import os
import random
import numpy as np
import torch

def make_seed():
    return int.from_bytes(os.urandom(4), byteorder="big")

def seed_all(seed: int) -> None:
    """https://pytorch.org/docs/stable/notes/randomness.html"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_reproducible(seed: int,
                      use_deterministic_algorithms: bool = False) -> None:
    seed_all(seed)
    torch.use_deterministic_algorithms(use_deterministic_algorithms)
