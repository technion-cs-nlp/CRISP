from typing import List, Literal
import json
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import ModelSaes, EncoderOutput
from hidden_state_ops import hs_to_logits
from collections import OrderedDict
from torch import Tensor
from functools import partial
from dataclasses import dataclass
from globals import LLAMA_3_1_8B, GEMMA_2_2B, SAE_GEMMA_2_2B, SAE_LLAMA_3_1_8B
from utils import load_cached_features, save_cached_features, print_memory_info
from peft import PeftModel


@dataclass
class Feature:
    index: int
    target_count: int
    benign_count: int
    target_acts: int
    benign_acts: int
    target_acts_relative: float = None
    benign_acts_relative: float = None
    freq_diff: int = None

    def __post_init__(self):
        self.target_acts_relative = self.target_acts / (self.benign_acts + 1)
        self.benign_acts_relative = self.benign_acts / (self.target_acts + 1)
        self.freq_diff = int(self.target_count - self.benign_count)

    def __contains__(self, feature_idx):
        return feature_idx == self.index

    def __str__(self):
        index = str(self.index) + ";"
        return (f"Feature(index={index:<8} tcount={int(self.target_count):<8} bcount={int(self.benign_count):<8}"
                f"freq diff={self.freq_diff:<8}"
                f"target acts rel={self.target_acts_relative:<10.2f} benign acts rel={self.benign_acts_relative:<10.2f}"
                f"target acts={self.target_acts:<10.1e} benign acts={self.benign_acts:<10.1e}"
                )

    def __repr__(self):
        return str(self)


class LayerFeatures:
    def __init__(self, features: List[Feature] = None):
        self.features = OrderedDict()
        self.similar_features = OrderedDict()
        if features:
            self.add_features(features)

    def add_features(self, features: List[Feature]):
        # then sort the features
        sorted_features = self._sort_features(features)
        # add the features to the dict
        self.features = OrderedDict((f.index, f) for f in sorted_features)

    def _sort_features(self, features: List[Feature]) -> List[Feature]:
        # Sort by score_freq_diff which requires minimum target frequency
        sorted_features = sorted(features, key=lambda f: f.freq_diff, reverse=True)
        return sorted_features

    def __getitem__(self, index):
        if isinstance(index, slice):
            keys = list(self.features.keys())[index]
            return list(self.features[k] for k in keys)
        return self.features[index]

    def __contains__(self, index):
        return index in self.features

    def __str__(self):
        return "\n".join(str(f) for f in self.features.values())

    def __repr__(self):
        return str(self)

    def topk(self, k: int):
        return list(self.features.values())[:k]

    def topk_filtered(self, k: int, model_name: str = None):
        candidate_features = self.topk(k*10)
        threshold = 3
        filtered_features = [f for f in candidate_features if f.target_acts_relative >= threshold]
        return filtered_features[:k]

    def bottomk_filtered(self, k: int):
        candidate_features = self.bottomk(k*10)
        threshold = 3
        filtered_features = [f for f in candidate_features if f.benign_acts_relative >= threshold]
        return filtered_features[:k]

    def bottomk(self, k: int):
        return list(self.features.values())[-k:][::-1]

    def clear(self):
        self.features.clear()

    def __iter__(self):
        return iter(self.features.values())

    def __len__(self):
        return len(self.features)

@dataclass
class CRISPConfig:
    layers: List[int]
    model_name: Literal["gemma", "llama_3_1", "llama"]
    bf16: bool = True
    saes_model_name: str = None

    def __post_init__(self):
        model_name_lower = self.model_name.lower()

        if model_name_lower == "gemma" or self.model_name == GEMMA_2_2B:
            self.model_name = GEMMA_2_2B
        elif model_name_lower == "llama_3_1" or model_name_lower == "llama" or self.model_name == LLAMA_3_1_8B:
            self.model_name = LLAMA_3_1_8B
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}. Supported models are 'llama', 'gemma', and 'llama_3_1'.")

        # Set SAE model name
        if self.model_name == GEMMA_2_2B:
            self.saes_model_name = SAE_GEMMA_2_2B
        elif self.model_name == LLAMA_3_1_8B:
            self.saes_model_name = SAE_LLAMA_3_1_8B
        else:
            raise ValueError(f"Unsupported model name: {self.model_name}. Supported models are 'llama', 'gemma', and 'llama_3_1'.")

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "layers": self.layers,
            "saes_model_name": self.saes_model_name,
            "bf16": self.bf16
        }

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)


class CRISP:
    def __init__(self, config: CRISPConfig):
        self.config = config
        self.model_saes = ModelSaes(
            sae_type=config.saes_model_name, 
            layers=config.layers,
            device='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(config.model_name, device_map='auto', torch_dtype=torch.bfloat16 if config.bf16 else torch.float32)
        self.model.config.use_cache = False if config.model_name == GEMMA_2_2B else True  # Gemma 2 has a bug with use_cache=True
        self.device_model = torch.device(self.model.device)

        self.features_dict = {}
        self.layers = config.layers
        self.hook_handles = []
        self.hook_function = None

    def set_hook_function(self, hook_function):
        self.hook_function = hook_function
        self.register_hooks()

    def hook(self, layer_idx, module, input, output):
        """Hook with layer index built in"""
        if self.hook_function is not None:
            return self.hook_function(module, input, output, layer_idx)
        return output

    def register_hooks(self):
        # if model wrapped with PeftModel, we need to get the inner model
        if isinstance(self.model, PeftModel):
            model = self.model.model.model
        else:
            model = self.model.model
        for layer_idx in self.layers:
            layer = model.layers[layer_idx]
            bound_hook = partial(self.hook, layer_idx)
            handle = layer.register_forward_hook(bound_hook)
            self.hook_handles.append(handle)

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    def tokenize(self, text, max_len=1024):
        return self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(self.device_model)

    def get_model_outputs(self, inputs, requires_grad=False):
        if requires_grad:
            outputs = self.model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

        # Add attention mask to outputs for loss calculation
        if 'attention_mask' in inputs:
            outputs.attention_mask = inputs['attention_mask']
        return outputs

    def get_logits(self, hidden_state):
        with torch.no_grad():
            return hs_to_logits(self.model, hidden_state[:, -1, :])

    def get_probs(self, hidden_state):
        with torch.no_grad():
            return torch.softmax(self.get_logits(hidden_state), dim=-1)

    def process_multi_texts_batch(self, text_target, text_benign, data_config=None, batch_size=16):
        # Clear previous features if any
        if self.features_dict:
            print("Clearing previous features.")
            self.features_dict.clear()
            torch.cuda.empty_cache()

        # Check cached features for all layers (without breaking early)
        cached_layer_features = {}
        uncached_layers = []

        for layer in self.layers:
            cached_features = load_cached_features(
                layer, 
                data_config, 
                model_name=self.config.model_name
            )
            if cached_features:
                cached_layer_features[layer] = cached_features
            else:
                uncached_layers.append(layer)
        print(f"Found {len(cached_layer_features)} cached layers and {len(uncached_layers)} uncached layers.")

        # If all layers are cached, skip batch processing
        if not uncached_layers:
            print("All layers have cached features.")
            for layer in self.layers:
                self.features_dict[layer] = LayerFeatures(cached_layer_features[layer])
            return
        else:
            print(f"Need to process {len(uncached_layers)} uncached layers: {uncached_layers}")

        # Continue with batch processing for uncached layers
        n_features = list(self.model_saes.saes.values())[0].d_sae
        agg_encoded_acts_targets = torch.zeros(len(uncached_layers), n_features).cpu()
        agg_encoded_acts_benigns = torch.zeros(len(uncached_layers), n_features).cpu()
        agg_encoded_counts_targets = torch.zeros(len(uncached_layers), n_features).cpu()
        agg_encoded_counts_benigns = torch.zeros(len(uncached_layers), n_features).cpu()
        num_samples = len(text_target)

        # Process in batches
        for batch_start in tqdm(range(0, num_samples, batch_size), desc="Processing text batches"):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_target = text_target[batch_start:batch_end]
            batch_benign = text_benign[batch_start:batch_end]

            # Tokenize all texts in batch at once
            inputs_target = self.tokenize(batch_target)
            inputs_benign = self.tokenize(batch_benign)

            # Create valid masks that exclude BOS and PAD tokens
            target_mask = inputs_target['attention_mask'].bool()
            benign_mask = inputs_benign['attention_mask'].bool()
            # Skip BOS token (index 0)
            target_mask[:, 0] = False
            benign_mask[:, 0] = False

            # Expand masks for broadcasting with hidden dimension later
            target_mask_expanded = target_mask.unsqueeze(-1)  # [batch, seq_len, 1]
            benign_mask_expanded = benign_mask.unsqueeze(-1)  # [batch, seq_len, 1]

            # Get model outputs for the batch
            outputs_target = self.get_model_outputs(inputs_target).hidden_states
            outputs_benign = self.get_model_outputs(inputs_benign).hidden_states

            batch_acts_target, batch_acts_benign = [], []
            batch_masks_target, batch_masks_benign = [], []

            for layer in uncached_layers:
                # Get the specific device for this layer's SAE
                layer_name = f"layers.{layer}"
                layer_device = self.model_saes.get_layer_device(layer_name)
                # Move tensors to the appropriate device
                batch_acts_target.append(outputs_target[layer].to(layer_device))
                batch_acts_benign.append(outputs_benign[layer].to(layer_device))
                # Also store masks for later use
                batch_masks_target.append(target_mask_expanded.to(layer_device))
                batch_masks_benign.append(benign_mask_expanded.to(layer_device))

            # Only encode if we have layers to process
            if batch_acts_target:
                # Encode all activations in the batch at once
                batch_encoded_target = [self.model_saes.encode(batch_acts_target[i], layer=layer) for i, layer in enumerate(uncached_layers)]
                batch_encoded_benign = [self.model_saes.encode(batch_acts_benign[i], layer=layer) for i, layer in enumerate(uncached_layers)]

                # Add batch features to the aggregated tensors, applying masks
                for i, layer in enumerate(uncached_layers):
                    # Apply masks to encoded activations
                    masked_encoded_target = batch_encoded_target[i] * batch_masks_target[i]
                    masked_encoded_benign = batch_encoded_benign[i] * batch_masks_benign[i]

                    # Sum activations only for valid tokens
                    agg_encoded_acts_targets[i] += masked_encoded_target.sum(dim=(0, 1)).detach().cpu()
                    agg_encoded_acts_benigns[i] += masked_encoded_benign.sum(dim=(0, 1)).detach().cpu()

                    # Count non-zero activations only for valid tokens
                    valid_target = (masked_encoded_target > 0)
                    valid_benign = (masked_encoded_benign > 0)
                    agg_encoded_counts_targets[i] += valid_target.sum(dim=(0, 1)).detach().cpu()
                    agg_encoded_counts_benigns[i] += valid_benign.sum(dim=(0, 1)).detach().cpu()

                # Explicitly delete tensors and empty cache
                del batch_encoded_target, batch_encoded_benign, masked_encoded_target, masked_encoded_benign

            # Clean up batch tensors
            del inputs_target, inputs_benign, outputs_target, outputs_benign
            del batch_acts_target, batch_acts_benign, batch_masks_target, batch_masks_benign
            torch.cuda.empty_cache()

        # Process features for each layer
        uncached_idx = 0  # Separate index for uncached layers
        for layer in tqdm(self.layers, desc="Processing layers"):
            # Use cached features if available
            if layer in cached_layer_features:
                print(f"Using cached features for layer {layer}.")
                features = cached_layer_features[layer]
            else:
                print(f"Processing features for layer {layer}.")
                features = self.process_features(
                    agg_encoded_acts_targets[uncached_idx],
                    agg_encoded_acts_benigns[uncached_idx],
                    agg_encoded_counts_targets[uncached_idx],
                    agg_encoded_counts_benigns[uncached_idx]
                )
                uncached_idx += 1  # Increment only for uncached layers
                if data_config:
                    print(f"Saving features for layer {layer} to cache.\n")
                    save_cached_features(
                        layer, 
                        data_config, 
                        features, 
                        model_name=self.config.model_name
                    )

            layer_features = LayerFeatures(features)
            self.features_dict[layer] = layer_features

    @torch.no_grad()
    def process_features(self, features_stats_target, features_stats_benign, features_stats_target_counts, features_stats_benign_counts) -> List[Feature]:
        """
        Process feature statistics to create Feature objects.

        Args:
            features_stats_target: Tensor of activation sums for target features
            features_stats_benign: Tensor of activation sums for benign features
            features_stats_target_counts: Tensor of activation counts for target features
            features_stats_benign_counts: Tensor of activation counts for benign features

        Returns:
            List of Feature objects with statistics
        """
        # Get indices where either target or benign has non-zero counts
        active_features = torch.logical_or(
            features_stats_target_counts > 0,
            features_stats_benign_counts > 0
        )
        active_indices = torch.nonzero(active_features).squeeze(1)

        # Create feature objects for all active features
        features = []
        for idx in active_indices:
            idx_item = idx.item()
            target_count = features_stats_target_counts[idx_item].item()
            benign_count = features_stats_benign_counts[idx_item].item()
            target_acts = features_stats_target[idx_item].item()
            benign_acts = features_stats_benign[idx_item].item()

            feature = Feature(
                index=idx_item,
                target_count=target_count,
                benign_count=benign_count,
                target_acts=target_acts,
                benign_acts=benign_acts
            )
            features.append(feature)

        return features

    def __call__(self, text: str, features_ablation: bool = False, k_features_ablate: int = None, alpha: float = None,
                 requires_grad: bool = False, error=False, topk_filter=True):

        inputs = self.tokenize(text)
        if not features_ablation:
            return self.get_model_outputs(inputs, requires_grad=requires_grad)
        else:
            try:
                outputs = self.get_model_outputs_with_ablation(inputs, k_features_ablate=k_features_ablate, layers=self.layers,
                    alpha=alpha,  requires_grad=requires_grad, error=error, topk_filter=topk_filter)
                return outputs
            finally:
                self.remove_hooks()
                torch.cuda.empty_cache()

    def generate(self, prompt: str, max_tokens: int = 20) -> str:

        inputs = self.tokenize(prompt)

        generation_config = {
            "max_new_tokens": max_tokens,
            "do_sample": False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "top_p": False,
        }

        with torch.no_grad():
            outputs = self.model.generate(
                inputs["input_ids"],
                **generation_config
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

    def get_salient_features(self, layer_idx, k_features, topk_filter: bool = True):
        layer_features = self.features_dict[layer_idx]
        if topk_filter:
            top_features = [f.index for f in layer_features.topk_filtered(k_features, self.config.model_name)]
        else:
            top_features = [f.index for f in layer_features.topk(k_features)]
        features_indices = torch.tensor(top_features[:k_features])
        return features_indices

    def ablation_hook(self, module, input, output, layer_idx: int, k_features_ablate: int, layers: List[int], alpha: float, error: bool = False, topk_filter: bool = True):
        if layer_idx not in layers:
            return output

        if not isinstance(output, tuple):
            return output

        hidden_states = output[0]
        hidden_states = self.edit_acts(hidden_states, layer_idx, k_features_ablate, alpha, error=error, topk_filter=topk_filter)
        outputs = (hidden_states,) + output[1:]
        return outputs

    def edit_acts(self, x, layer_idx: int, k_features_ablate: int, alpha: float, error: bool = False, topk_filter: bool = True) -> Tensor:

        original_dtype = x.dtype
        layer_name = f"layers.{layer_idx}"
        sae_device = self.model_saes.get_layer_device(layer_name)

        # Move to the specific SAE device
        x = x.to(sae_device, dtype=original_dtype)

        # Get original encoded representation
        encoded_original, mean, std = self.model_saes.encode_with_stats(x, layer=layer_idx)

        # Create clone for ablation
        encoded_ablated = encoded_original.clone()

        # Get features to ablate and perform ablation
        features_to_ablate = self.get_salient_features(
            layer_idx, k_features=k_features_ablate, topk_filter=topk_filter).to(encoded_ablated.device)

        features_to_ablate = torch.tensor(features_to_ablate).to(encoded_ablated.device)

        encoded_ablated = self.ablate_features(encoded_ablated, features_to_ablate, alpha, mean)

        # Decode both versions
        decoded_original = self.model_saes.decode(encoded_original, layer=layer_idx)
        decoded_ablated = self.model_saes.decode(encoded_ablated, layer=layer_idx)

        # Calculate just the difference caused by ablation
        ablation_effect = decoded_ablated - decoded_original

        # If error is True, don't add the error term between the original and decoded activations
        if error:
            x_error = x - decoded_original.to(x.device, dtype=original_dtype)
            ablation_effect = -x_error

        # Apply just the ablation effect to original activations
        # This maintains maximum fidelity to the original
        result = x + ablation_effect.to(x.device, dtype=original_dtype)

        return result.to(self.device_model, dtype=original_dtype)

    def ablate_features(self, encoded, features_to_ablate, alpha, mean, replacement_features=None) -> EncoderOutput:
        # Check if features_to_ablate not empty
        if features_to_ablate.numel() > 0:
            # Create a mask for positive activations (> 0)
            mask = encoded[:, 1:, features_to_ablate] > 0  # skip first token ([BOS] token)
            # Only modify values where mask is True (activation > 0)
            new_act_value = -torch.abs((mean * alpha).unsqueeze(-1)[:,1:]) # remove first token ([BOS] token)
            encoded[:, 1:, features_to_ablate] = torch.where(mask, new_act_value, encoded[:, 1:, features_to_ablate])
        return encoded

    def unload_lora(self):
        try:
            self.model = self.model.unload()
        except:
            pass
        finally:
            with torch.no_grad():
                torch.cuda.empty_cache()