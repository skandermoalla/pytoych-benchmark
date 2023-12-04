import random

import numpy as np
import torch


def generate_random_seed():
    """Generate a random seed."""
    return random.randint(0, 2**32 - 1)


# Update this function whenever you have a library that needs to be seeded.
def seed_everything(config):
    """Seed all random generators.
    This is not strict reproducibility, but it should be enough for most cases.
    https://pytorch.org/docs/stable/notes/randomness.html
    """
    random.seed(config.seed)

    # This is for legacy numpy.
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html
    np.random.seed(config.seed)
    # New code should make a Generator out of the config.seed directly.

    torch.manual_seed(config.seed)
