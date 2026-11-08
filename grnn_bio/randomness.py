"""Order-independent named RNG streams; no reliance on randomized Python hash()."""
import hashlib
import numpy as np


def seed_for(seed, namespace):
    digest = hashlib.sha256(f'grnn-v7:{int(seed)}:{namespace}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def rng_for(seed, namespace):
    return np.random.default_rng(seed_for(seed, namespace))
