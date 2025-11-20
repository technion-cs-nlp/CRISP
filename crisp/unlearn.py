from dataclasses import dataclass
from typing import List
from functools import partial
import torch
from torch import Tensor
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json
import os

from crisp import CRISP
from data import genenrate_hp_eval_text, generate_bio_eval_text, generate_cyber_eval_text, DataConfig, wmdp_bio_coherency_prompts, wmdp_cyber_coherency_prompts, hp_coherency_prompts
from eval import get_mcq_accuracy
from globals import set_seed, SEED, PROJECT_PATH
from utils import save_model

@dataclass
class UnlearnConfig:
    """Configuration for unlearning process"""

    data_type: str # hp, bio, or cyber

    # Learning parameters
    learning_rate: float = 1e-5
    num_epochs: int = 1
    batch_size: int = 1

    # Feature selection
    k_features: int | List[int] = 10  # Number of top salient features, if list, specify for each layer
    alpha: float = 5  # Target activation factor for the salient features, alpha * mean(acts)
    beta: float = 0.99  # Regularization parameter for the loss function
    gamma: float = 0.01  # Coherency loss parameter

    # Lora parameters
    lora_rank: int = 4

    # Model saving
    save_model: bool = False  # Whether to save the model after unlearning
    save_path: str = f"{PROJECT_PATH}/CRISP/saved_models/crisp"

    verbose: str = None

    def __post_init__(self):
        assert self.data_type in ["hp", "bio", "cyber"]

        if self.verbose:
            self.verbose = self.data_type

        if self.save_model and not self.save_path:
            raise ValueError("save_path must be provided when save_model=True")

    def to_dict(self):
        return {
            "learning_rate": self.learning_rate,
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "k_features": self.k_features,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "lora_rank": self.lora_rank,
            "save_model": self.save_model,
            "save_path": self.save_path,
            "data_type": self.data_type,
            "verbose": self.verbose,
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


def get_params(model: AutoModelForCausalLM, layer_ids, params_name=None):
    params = []
    for layer_id in layer_ids:
        for name, param in model.model.layers[layer_id].named_parameters():
            if params_name is None or any(pn in name for pn in params_name):
                params.append(param)
    return params


def get_features_acts(model: CRISP, outputs, k, mode: str, alpha: float, topk_filter: bool = True, original_outputs=None) -> dict:
    """
    Extracts and processes feature activations from a given model's layers based on the specified mode.

    Args:
        model (CRISP): The model from which to extract feature activations.
        outputs: The outputs from the model, containing hidden states.
        k (int): The number of salient features to consider.
        mode (str): The mode of processing activations. Valid modes are:
            - 'acts': Uses raw activations.
            - 'acts_alpha': Adds alpha*mean(acts) directly to the activations.
        alpha (float): The scaling factor for activations.
        original_outputs: outputs from the original (non-LoRA) model.

    Returns:
        dict: A dictionary where keys are layer indices and values are the processed activations for the salient features.
    """

    features_acts = {}

    for i, layer_idx in enumerate(model.layers):
        salient_features = model.get_salient_features(layer_idx, k, topk_filter)
        replacement_features = None
        if salient_features.numel() == 0:
            continue
        # Get the specific device for this layer's SAE
        layer_name = f"layers.{layer_idx}"
        layer_device = model.model_saes.get_layer_device(layer_name)

        # Move to the specific device
        hidden_states = outputs.hidden_states[layer_idx].to(layer_device)

        if mode == 'acts':
            encoded = model.model_saes.encode(hidden_states, layers=layer_idx)

        elif mode == "acts_alpha":
            mask = None
            if original_outputs:
                original_hidden_states = original_outputs.hidden_states[layer_idx].to(layer_device)
                original_encoded = model.model_saes.encode(original_hidden_states, layer=layer_idx)
                mask = original_encoded[:, :, salient_features] > 0

            def add_alpha_to_salient(pre_acts, features, alpha, mask=None):
                if features.numel() > 0:
                    new_act_value = pre_acts[:, :, features] + torch.abs(alpha * pre_acts.mean(dim=-1, keepdim=True))
                    if mask is not None:
                        pre_acts[:, :, features] = torch.where(mask, new_act_value, pre_acts[:, :, features])
                    else:
                        pre_acts[:, :, features] = new_act_value
                return pre_acts
            encoded = model.model_saes.encode_hook(hidden_states, layer_idx, add_alpha_to_salient, salient_features, alpha, mask)

        else:
            raise ValueError(f"Unknown mode: {mode}. Valid modes: ['acts', 'acts_alpha']")

        if salient_features.numel() > 0:
            features_acts[layer_idx] = encoded[:, :, salient_features]

    return features_acts


def loss_features_acts(crisp: CRISP, outputs, config: UnlearnConfig, original_outputs=None) -> Tensor:
    """
        Calculate the loss based on the activations of features in the model's layers.

        This function computes the activations of features in the model's layers using the provided configuration.
        It then calculates the loss as the mean activation across all layers and prints detailed information about
        the activations for each layer.

        Args:
            model (CRISP): The model from which to extract feature activations.
            outputs: The outputs from the model's forward pass.
            config (UnlearnConfig): Configuration object containing parameters for feature extraction.

        Returns:
            float: The mean activation loss across all layers.
    """
    # Get a device from the first layer to use for the initial loss tensor
    first_layer = crisp.layers[0] if crisp.layers else 0
    layer_name = f"layers.{first_layer}"
    initial_device = crisp.model_saes.get_layer_device(layer_name)

    loss = torch.tensor(0.0, device=initial_device)
    features_acts = get_features_acts(crisp, outputs, k=config.k_features, mode="acts_alpha", alpha=config.alpha, topk_filter=True, original_outputs=original_outputs)
    attention_mask = outputs.attention_mask  # shape [b, seq_len]

    for layer, acts in features_acts.items():
        # acts: shape [b, seq_len, d_model]
        # skip pad tokens (mask=0)
        valid_mask = attention_mask.bool()
        # skip BOS index 0
        valid_mask[:, 0] = False
        # zero out invalid positions while keeping original dimensions
        valid_mask_expanded = valid_mask.unsqueeze(-1)  # reshape to [b, seq_len, 1] for broadcasting
        valid_acts = acts * valid_mask_expanded.to(acts.device)  # apply mask, preserving [b, seq_len, d_model] shape
        non_zero_acts = valid_acts[valid_acts > 0]
        if non_zero_acts.numel() > 0:
            loss += non_zero_acts.abs().mean().to(loss.device)
        # print(f"Layer {layer}: num activated features: {acts.size(1)}, mean acts: {acts.mean().item():.2f}, sum acts: {acts.sum().item():.2f}")

    # Mean the loss by the number of layers
    if len(features_acts) > 0:
        return loss / len(features_acts)
    return loss


def loss_acts_diff(crisp: CRISP, edited_outputs, target_outputs, config: UnlearnConfig, layers: str = "sae"):
    """
    Loss based on the difference between the activations of the original model residual stream x
    and the edited model activation residual stream x_hat.
    Uses tensor stacking for efficient calculation while properly masking BOS and PAD tokens.
    """
    # Get attention mask to ignore pad tokens
    attention_mask = edited_outputs.attention_mask  # shape [b, seq_len]
    valid_mask = attention_mask.bool()
    # skip BOS index 0
    valid_mask[:, 0] = False
    # Expand mask for broadcasting with hidden dimension
    valid_mask_expanded = valid_mask.unsqueeze(-1)  # [b, seq_len, 1]

    # Determine which layers to collect activations from
    if layers == "last":
        # Just get the last layer
        x_target = target_outputs.hidden_states[-1].unsqueeze(0)
        x_edit = edited_outputs.hidden_states[-1].unsqueeze(0)
    elif layers == "sae":
        # Get layers specified in crisp.layers and stack them
        x_target = torch.stack([target_outputs.hidden_states[layer] for layer in crisp.layers], dim=0)
        x_edit = torch.stack([edited_outputs.hidden_states[layer] for layer in crisp.layers], dim=0)
    elif layers == "all":
        # Get all layers (except first two) and stack them
        x_target = torch.stack([h for h in target_outputs.hidden_states[2:]], dim=0)
        x_edit = torch.stack([h for h in edited_outputs.hidden_states[2:]], dim=0)

    # Apply masks to zero out BOS and padding tokens
    # For [num_layers, batch_size, seq_len, hidden_dim] tensors
    # Need to expand mask to [1, batch_size, seq_len, 1] for broadcasting
    mask_for_stacked = valid_mask_expanded.unsqueeze(0)
    x_target = x_target * mask_for_stacked
    x_edit = x_edit * mask_for_stacked

    # Calculate divergence on masked stacked tensor at once
    loss = calc_divergence(x_target, x_edit, method="mse")

    return loss


def calc_divergence(p: Tensor, q: Tensor, method: str):
    """
        Calculate the divergence between two tensors.

        Args:
            p (Tensor): The first tensor.
            q (Tensor): The second tensor.
            mode (str): The mode of divergence to use. Valid modes are:
                - 'kl': Kullback-Leibler divergence.
                - 'js': Jensen-Shannon divergence.
                - 'mse': Mean Squared Error.

        Returns:
            Tensor: The divergence between p and q.
    """
    if method == "mse":
        divergence = ((p - q) ** 2).mean()
    elif method == "sse":
        divergence = ((p - q) ** 2).sum()
    else:
        raise ValueError(f"Unknown mode: {method}. Valid methods: ['kl', 'mse', 'js']")

    # if divergence is nan, return 0
    if torch.isnan(divergence).any():
        return torch.tensor(0.0, device=p.device)
    return divergence


def get_loss_func(original_outputs) -> callable:
    """
    Returns the loss function with original outputs bound via partial application.
    
    Args:
        original_outputs: The original model outputs to use for features_activated loss
        
    Returns:
        Callable: The loss function with original_outputs pre-bound
    """
    return partial(loss_features_acts, original_outputs=original_outputs)


def unlearn_lora(crisp: CRISP, text_target, text_benign, config: UnlearnConfig, data_config: DataConfig = None):
    """
    Unlearns a concept using LoRA adapters instead of modifying base model weights.

    Args:
        crisp (CRISP): The crisp model to adapt
        # text_target (str or list of str): The target text(s) representing concept to unlearn
        text_benign (str or list of str): The benign text(s) used for processing
        config (UnlearnConfig): Configuration object for unlearning process
        data_config (DataConfig, optional): Configuration for data processing
    """
    set_seed(SEED)
    if not crisp.features_dict:
        crisp.process_multi_texts_batch(text_target=text_target, text_benign=text_benign, data_config=data_config, batch_size=config.batch_size)
    print(f"Unlearn Config: {config}")
    print(f"CRISP Config: {crisp.config}")
    print(f"Data Config: {data_config}")
    print(f"SEED: {SEED}")

    if not isinstance(crisp.model, PeftModel):
        # Configure LoRA
        target_modules = ['down_proj', 'gate_proj', 'up_proj', 'q_proj', 'v_proj', 'k_proj', 'o_proj']

        lora_config = LoraConfig(
            init_lora_weights=True,
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_rank,
            lora_alpha=2*config.lora_rank,
            bias="none",
            target_modules=target_modules,
            layers_to_transform=list(range(3, 10)),
            lora_dropout=0.05,
        )

        # Add LoRA adapters to model
        crisp.model = get_peft_model(crisp.model, lora_config)

    optimizer = torch.optim.AdamW(crisp.model.parameters(), lr=config.learning_rate, amsgrad=True)

    # Convert inputs to lists if necessary
    if isinstance(text_target, str):
        text_target = [text_target]
    if isinstance(text_benign, str):
        text_benign = [text_benign]

    # Create batches of data
    n_samples = len(text_target)
    batch_size = min(config.batch_size, n_samples)
    n_batches = (n_samples + batch_size - 1) // batch_size  # ceiling division

    for _ in tqdm(range(config.num_epochs), desc="Epochs", total=config.num_epochs):
        # Shuffle data each epoch for better training
        indices = torch.randperm(n_samples).tolist()

        for batch_idx in tqdm(range(n_batches), desc="Batches", leave=False):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]

            batch_target = [text_target[i] for i in batch_indices]
            batch_benign = [text_benign[i] for i in batch_indices]

            optimizer.zero_grad()

            # Process entire target batch at once
            unlearn_edited_outputs = crisp(batch_target, requires_grad=True)

            # Calculate unlearning loss on batch using features_activated
            with crisp.model.disable_adapter():
                unlearn_original_outputs = crisp(batch_target, requires_grad=False)
            
            loss_func = get_loss_func(unlearn_original_outputs)
            loss_unlearn = loss_func(crisp, unlearn_edited_outputs, config)

            # Regularization - always use benign text with acts_diff
            reg_edited_outputs = crisp(batch_benign, requires_grad=True)
            with crisp.model.disable_adapter():
                reg_original_outputs = crisp(batch_benign, requires_grad=False)
            reg_loss = loss_acts_diff(crisp, reg_edited_outputs, reg_original_outputs, config)

            # Coherency loss - always enabled
            # Select coherency prompts based on target type
            if config.data_type == "cyber":
                coherency_prompts = wmdp_cyber_coherency_prompts
            elif config.data_type == "hp":
                coherency_prompts = hp_coherency_prompts
            elif config.data_type == "bio":
                coherency_prompts = wmdp_bio_coherency_prompts
            else:
                raise ValueError(f"Unknown data type: {config.data_type}. Valid options: ['hp', 'bio', 'cyber']")

            text_idx = batch_idx % len(coherency_prompts)
            coher_text = coherency_prompts[text_idx]

            with crisp.model.disable_adapter():
                coher_original_outputs = crisp(coher_text, requires_grad=False)
            unlearn_outputs = crisp(coher_text, requires_grad=True)
            coher_loss = loss_acts_diff(crisp, unlearn_outputs, coher_original_outputs, config, layers="last")

            # Apply separate backward passes for memory efficiency
            loss_unlearn = (1-config.beta) * loss_unlearn
            if loss_unlearn.item() != 0:
                loss_unlearn.backward()

            reg_loss = config.beta * reg_loss.to(loss_unlearn.device)
            if reg_loss.item() != 0:
                reg_loss.backward()

            coher_loss = config.beta * config.gamma * coher_loss.to(loss_unlearn.device)
            if coher_loss.item() != 0:
                coher_loss.backward()

            batch_loss = loss_unlearn + reg_loss + coher_loss

            optimizer.step()

            # Print batch stats
            if (batch_idx + 1) % 25 == 0:
                str_print_loss = f"Batch {batch_idx+1}/{n_batches}, Loss: {batch_loss.item():.2e} " + \
                      f"(Unlearn: {loss_unlearn.item():.2e}, " + \
                      f"Reg: {reg_loss.item():.2e}, " + \
                      f"Coherency: {coher_loss.item():.2e})"
                print(str_print_loss)

            if (batch_idx + 1) % 200 == 0 and config.verbose:
                if config.data_type == "hp":
                    genenrate_hp_eval_text(crisp)
                    get_mcq_accuracy(crisp, type="hp")
                elif config.data_type == "bio":
                    generate_bio_eval_text(crisp)
                    get_mcq_accuracy(crisp, type="wmdp_bio")
                    get_mcq_accuracy(crisp, type="high_school_bio")
                    get_mcq_accuracy(crisp, type="college_bio")
                elif config.data_type == "cyber":
                    generate_cyber_eval_text(crisp)
                    get_mcq_accuracy(crisp, type="wmdp_cyber")
                    get_mcq_accuracy(crisp, type="high_school_cs")
                    get_mcq_accuracy(crisp, type="college_cs")

        if config.save_model:
            configs = {
                "crisp_config": crisp.config.to_dict(),
                "unlearn_config": config.to_dict(),
                "data_config": data_config.to_dict() if data_config else None
            }
            model_name = "llama_3_1" if "llama" in crisp.config.model_name else "gemma"
            path = f"{PROJECT_PATH}/CRISP/saved_models/crisp/{model_name}/{config.data_type}"
            save_model(crisp.model, configs=configs, path=path, tokenizer=crisp.tokenizer)