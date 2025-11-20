import os
import torch
import numpy as np
import random

# Paths
PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEMMA_SAE_CACHE_PATH = os.path.join(PROJECT_PATH, "crisp", "gemma_sae_cache")
LLAMA_3_1_SAE_CACHE_PATH = os.path.join(PROJECT_PATH, "crisp", "llama_sae_cache")
SAE_LLAMA_3_1_8B = "fnlp/Llama3_1-8B-Base-LXR-8x"
SAE_GEMMA_2_2B = "google/gemma-2-2b"
LLAMA_3_1_8B ="meta-llama/Llama-3.1-8B"
GEMMA_2_2B = "google/gemma-2-2b"

SEED = 0

def set_seed(seed=SEED, deterministic_cudnn=True):
    """
    Set seeds for reproducibility.

    Args:
        seed: Integer seed for random number generators
        deterministic_cudnn: Whether to make cuDNN deterministic (may reduce performance)
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Make PyTorch operations deterministic
    if deterministic_cudnn:
        # For complete determinism (may slow down training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # PyTorch 1.8+ settings for determinism
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        try:
            # Available from PyTorch 1.8
            torch.use_deterministic_algorithms(True, warn_only=True)
        except AttributeError:
            # Older PyTorch versions
            torch.set_deterministic(True)
