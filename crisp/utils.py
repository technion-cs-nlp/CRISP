import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
import datetime
import time
import functools
import pickle
import random
import os
import json
import numpy as np
import torch
from pathlib import Path
from globals import PROJECT_PATH, LLAMA_3_1_8B, GEMMA_2_2B, SEED
import requests

LLAMA_CACHE_DIR = f"{PROJECT_PATH}/crisp/crisp_cache/llama_3_8b_processed_features"
LLAMA_3_1_CACHE_DIR = f"{PROJECT_PATH}/crisp/crisp_cache/llama_3_1_8b_processed_features"
GEMMA_CACHE_DIR = f"{PROJECT_PATH}/crisp/crisp_cache/gemma_2_2b_processed_features"


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"{func.__name__} took {elapsed_time:.4f} seconds")
        return result
    return wrapper

def get_cache_filename(layer: int, data_config):
    """Generate a consistent and readable cache filename based on config parameters."""
    if data_config is None:
        return None

    # Common parameters for all config types
    params = [f"layer_{layer}"]

    # Get config type name to include in filename
    config_type = data_config.__class__.__name__.lower().replace("config", "")
    params.append(f"type_{config_type}")

    # Handle specific config types
    if hasattr(data_config, 'forget_dataset_name') and hasattr(data_config, 'retain_dataset_name'):
        # HPDataConfig specific parameters
        forget_name = data_config.forget_dataset_name.split('/')[-1]
        retain_name = data_config.retain_dataset_name.split('/')[-1]
        params.append(f"forget_{forget_name}")
        params.append(f"retain_{retain_name}")


    elif hasattr(data_config, 'forget_type') and hasattr(data_config, 'retain_type'):
        # WMDPDataConfig specific parameters
        params.append(f"forget_{data_config.forget_type}")
        params.append(f"retain_{data_config.retain_type}")

    # # Add common processing parameters
    # if hasattr(data_config, 'n_examples'):
    #     params.append(f"n_examples_{data_config.n_examples}")

    # always load the full features analysis
    if config_type == "hpdata":
        params.append(f"n_examples_2500")
    else:
        params.append(f"n_examples_None")

    if hasattr(data_config, 'max_length'):
        params.append(f"max_len_{data_config.max_length}")

    # Create a filename-safe string
    filename = "_".join(params)
    return filename

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Redirect old module paths to new ones
        if module == 'crisp.crisp':
            module = 'crisp'  # Use your correct module path
        return super().find_class(module, name)

def load_cached_features(layer: int, data_config, model_name: str):
    if data_config is None:
        return None

    filename = get_cache_filename(layer, data_config)

    # Select cache directory based on model name
    if LLAMA_3_1_8B == model_name:
        cache_dir = LLAMA_3_1_CACHE_DIR
    elif GEMMA_2_2B == model_name:
        cache_dir = GEMMA_CACHE_DIR
    else:
        raise ValueError(f"Unknown model type: {model_name}. Cannot determine cache directory.")

    cache_path = f"{cache_dir}/{filename}.pkl"
    config_path = f"{cache_dir}/{filename}_config.json"

    if os.path.exists(cache_path) and os.path.exists(config_path):
        with open(cache_path, "rb") as f:
            features = CustomUnpickler(f).load()
        with open(config_path, "r") as f:
            cached_config = json.load(f)
        return features
    return None

def save_cached_features(layer: int, data_config, features, model_name: str):
    if data_config is None:
        return

    # Select cache directory based on model name
    if LLAMA_3_1_8B == model_name:
        cache_dir = LLAMA_3_1_CACHE_DIR
    elif GEMMA_2_2B == model_name:
        cache_dir = GEMMA_CACHE_DIR
    else:
        raise ValueError(f"Unknown model type: {model_name}. Cannot determine cache directory.")

    os.makedirs(cache_dir, exist_ok=True)
    filename = get_cache_filename(layer, data_config)
    cache_path = f"{cache_dir}/{filename}.pkl"
    config_path = f"{cache_dir}/{filename}_config.json"

    with open(cache_path, "wb") as f:
        pickle.dump(features, f)
    with open(config_path, "w") as f:
        json.dump(data_config.__dict__, f, indent=4)

def get_gpu_memory_info():
    """Get available memory for each GPU in bytes."""
    if not torch.cuda.is_available():
        return {}

    available_gpus = {}
    for i in range(torch.cuda.device_count()):
        free_memory = torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)
        available_gpus[i] = free_memory

    return available_gpus

def get_ram_memory_info():
    """Get available RAM memory in bytes."""
    if sys.platform.startswith('linux'):
        with open('/proc/meminfo', 'r') as f:
            meminfo = f.readlines()
        mem_total = int([x for x in meminfo if 'MemTotal' in x][0].split()[1]) * 1024
        mem_free = int([x for x in meminfo if 'MemFree' in x][0].split()[1]) * 1024
        return {'total': mem_total, 'free': mem_free}
    else:
        raise NotImplementedError("RAM memory info is only implemented for Linux.")

def print_memory_info():
    """Print available GPU and RAM memory information."""
    gpu_memory = get_gpu_memory_info()
    ram_memory = get_ram_memory_info()

    print("Available GPU Memory:")
    for gpu, mem in gpu_memory.items():
        print(f"GPU {gpu}: {mem / (1024 ** 3):.2f} GB")

    print("Available RAM Memory:")
    print(f"Total: {ram_memory['total'] / (1024 ** 3):.2f} GB, Free: {ram_memory['free'] / (1024 ** 3):.2f} GB")


def save_model(model, configs: dict, path: str, results: dict = None, tokenizer = None):
    """
    Save the model and configurations to disk.

    Args:
        model: The model to save
        configs (dict): Dictionary containing configuration objects
        path (str): Path to save the model and configurations
        results (dict, optional): Dictionary containing experiment results
    """

    os.makedirs(path, exist_ok=True)

    configs_to_save = {}
    for config_name, config_obj in configs.items():
        if hasattr(config_obj, 'to_dict'):
            configs_to_save[config_name] = config_obj.to_dict()
        elif hasattr(config_obj, '__dict__'):
            configs_to_save[config_name] = config_obj.__dict__
        else:
            configs_to_save[config_name] = config_obj

    if 'SEED' in globals():
        configs_to_save["seed"] = SEED

    config_path = os.path.join(path, "configs.json")
    with open(config_path, 'w') as f:
        json.dump(configs_to_save, f, indent=2)

    if results is not None:
        results_path = os.path.join(path, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {results_path}")

    if tokenizer is not None:
        tokenizer.save_pretrained(path)
        print(f"Tokenizer saved to {path}")

    model.save_pretrained(path)
    print(f"Model saved to {path}")

    print(f"Configurations saved to {config_path}")

def get_feature_tokens(model_id, layer, feature_index, top_k=5):
    """
    Get the top activating tokens for a specific feature from Neuronpedia.
    
    Args:
        model_id: Model identifier (e.g., "llama3.1-8b")
        layer: Layer number
        feature_index: Feature index
        top_k: Number of top tokens to return (default 5)
    
    Returns:
        Feature data including top activating tokens
    """
    if "llama" in model_id:
        source = f"{layer}-llamascope-res-32k"
    elif "gemma" in model_id:
        source = f"{layer}-gemmascope-res-16k"
    else:
        print(f"Warning: Unknown model_id {model_id}, defaulting to llama source format")
        source = f"{layer}-llamascope-res-32k"

    url = f"https://www.neuronpedia.org/api/feature/{model_id}/{source}/{feature_index}"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        print(f"Error: {response.status_code}")
        return None
