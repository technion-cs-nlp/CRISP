import json
import abc
import os
from fnmatch import fnmatch
from pathlib import Path
from typing import NamedTuple, Dict, Optional, Union, Type, TypeVar, Any
from tqdm.auto import tqdm
# import einops
import torch
import numpy as np
from huggingface_hub import snapshot_download
from natsort import natsorted
from safetensors.torch import load_model, save_model
from torch import Tensor, nn
from globals import GEMMA_SAE_CACHE_PATH, LLAMA_3_1_SAE_CACHE_PATH
from utils import get_gpu_memory_info

from dataclasses import dataclass
from simple_parsing import Serializable


@dataclass
class SaeConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    expansion_factor: int = 32
    """Multiple of the input dimension to use as the SAE dimension."""

    normalize_decoder: bool = True
    """Normalize the decoder weights to have unit norm."""

    num_latents: int = 0
    """Number of latents to use. If 0, use `expansion_factor`."""

    k: int = 32
    """Number of nonzero features."""

    multi_topk: bool = False
    """Use Multi-TopK loss."""

    @classmethod
    def from_dict(cls, d: dict, *, drop_extra_fields: bool = False) -> "SaeConfig":
        if drop_extra_fields:
            # Silently remove extra fields by only keeping known fields
            known_fields = cls.__dataclass_fields__.keys()
            d = {k: v for k, v in d.items() if k in known_fields}
        return cls(**d)


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    """Activations of the top-k latents."""

    top_indices: Tensor
    """Indices of the top-k features."""


class ForwardOutput(NamedTuple):
    sae_out: Tensor

    latent_acts: Tensor
    """Activations of the top-k latents."""

    latent_indices: Tensor
    """Indices of the top-k features."""

    fvu: Tensor
    """Fraction of variance unexplained."""

    auxk_loss: Tensor
    """AuxK loss, if applicable."""

    multi_topk_fvu: Tensor
    """Multi-TopK FVU, if applicable."""


class Sae(nn.Module, abc.ABC):
    """Base class for Sparse Autoencoders."""

    def __init__(
        self,
        d_in: int,
        d_sae: int | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

    @property
    @abc.abstractmethod
    def device(self) -> torch.device:
        """Get device where the model parameters are stored."""
        pass

    @property
    @abc.abstractmethod
    def dtype(self) -> torch.dtype:
        """Get data type of the model parameters."""
        pass

    @abc.abstractmethod
    def encode_pre_relu(self, x: Tensor, **kwargs) -> Tensor:
        """Compute the pre-ReLU activations."""
        pass

    @abc.abstractmethod
    def encode_hook(self, x:Tensor, intervention_fn=None, *intervention_args, **intervention_kwargs) -> Tensor:
        """Encode input tensor with optional intervention."""
        pass

    @abc.abstractmethod
    def encode_activate(self, pre_relu: Tensor, **kwargs) -> Tensor:
        """Activate the pre-ReLU activations."""
        pass

    @abc.abstractmethod
    def encode(self, x: Tensor, **kwargs) -> Tensor:
        """Encode input tensor to latent representation."""
        pass

    @abc.abstractmethod
    def encode_top(self, x: Tensor, **kwargs) -> EncoderOutput:
        """Encode input tensor and select top-k latents."""
        pass

    @abc.abstractmethod
    def decode(self, encoded: Any, **kwargs) -> Tensor:
        """Decode latent representation back to input space."""
        pass

    @abc.abstractmethod
    def forward(self, x: Tensor, **kwargs) -> Any:
        """Forward pass through the autoencoder."""
        pass

    @abc.abstractmethod
    def save_to_disk(self, path: Path | str) -> None:
        """Save model to disk."""
        pass

    @classmethod
    @abc.abstractmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
    ) -> "Sae":
        """Load model from disk."""
        pass

    @staticmethod
    def get_sae_class(sae_type: str) -> Type["Sae"]:
        """Get SAE class by type string."""
        if sae_type == "identity":
            return IdentitySae
        elif "llama" in sae_type.lower():
            return TopkSae
        elif "gemma" in sae_type.lower():
            return JumpReLUSAE
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}")


class TopkSae(Sae):
    """Sparse Autoencoder with top-k feature selection."""

    def __init__(
        self,
        d_in: int,
        cfg: SaeConfig,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        *,
        decoder: bool = True,
    ):
        super().__init__(d_in, cfg.num_latents, device, dtype)
        self.cfg = cfg
        self.num_latents = cfg.num_latents or d_in * cfg.expansion_factor
        self.d_sae = self.num_latents

        self.encoder = nn.Linear(d_in, self.num_latents, device=device, dtype=dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone()) if decoder else None
        if decoder and self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(d_in, dtype=dtype, device=device))

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        decoder: bool = True,
    ) -> "TopkSae":
        path = Path(path)

        # Check if this is a LLaMA SAE by looking for the checkpoints directory and hyperparams.json
        sae_path = path / "checkpoints" / "final.safetensors"
        config_path = path / "hyperparams.json"
        is_llama_sae = sae_path.exists() and config_path.exists()
        
        if is_llama_sae:
            # Load LLaMA 3.1 SAE from the downloaded structure
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found at: {config_path}")
            with open(config_path, "r") as f:
                sae_config = json.load(f)

            if not sae_path.exists():
                raise FileNotFoundError(f"LLaMA 3.1 checkpoint not found at: {sae_path}")

            # Load the model weights directly
            from safetensors import safe_open

            with safe_open(sae_path, framework="pt", device=str(device)) as f:

                # Create config - you may need to adjust these defaults based on your needs
                cfg = SaeConfig(
                    num_latents= sae_config['d_sae'],
                    k=sae_config['top_k'],  # top k for these SAE
                    normalize_decoder=sae_config["sparsity_include_decoder_norm"],
                    expansion_factor=sae_config['expansion_factor']  # Not used when num_latents is specified
                )

                # Create SAE instance
                sae = TopkSae(sae_config['d_model'], cfg, device=device, decoder=decoder)

                # Load the weights
                sae.encoder.weight.data = f.get_tensor("encoder.weight")
                if "encoder.bias" in f.keys():
                    sae.encoder.bias.data = f.get_tensor("encoder.bias")

                if decoder and "W_dec" in f.keys():
                    sae.W_dec.data = f.get_tensor("W_dec")
                elif decoder and "decoder.weight" in f.keys():
                    sae.W_dec.data = f.get_tensor("decoder.weight").T

                if "b_dec" in f.keys():
                    sae.b_dec.data = f.get_tensor("b_dec")
                elif "decoder.bias" in f.keys():
                    sae.b_dec.data = f.get_tensor("decoder.bias")

            return sae
        else:
            with open(path / "hyperparams.json", "r") as f:
                cfg_dict = json.load(f)
                d_in = cfg_dict.pop("d_sae")

                cfg = SaeConfig.from_dict(cfg_dict, drop_extra_fields=True)
                sae = TopkSae(d_in, cfg, device=device, decoder=decoder)
                load_model(
                    model=sae,
                    filename=str(path / "sae.safetensors"),
                    device=str(device),
                    strict=decoder,
                )
                return sae

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    **self.cfg.to_dict(),
                    "d_in": self.d_in,
                    "sae_type": "topk",
                },
                f,
            )

    @property
    def device(self):
        return self.encoder.weight.device

    @property
    def dtype(self):
        return self.encoder.weight.dtype

    def pre_acts(self, x: Tensor) -> Tensor:
        # Remove decoder bias as per Anthropic
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)
        out = nn.functional.relu(out)
        return out

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=True))

    def encode_pre_relu(self, x: Tensor) -> Tensor:
        """Compute the pre-ReLU activations."""
        sae_in = x.to(self.dtype) - self.b_dec
        out = self.encoder(sae_in)
        return out

    def encode_hook(self, x: Tensor, intervention_fn=None, *intervention_args, **intervention_kwargs) -> Tensor:
        """
        Encode input with an intervention before activation.

        Args:
            x: Input tensor
            intervention_fn: Function to apply to the pre-activation tensor
            intervention_args: Additional positional arguments to pass to intervention_fn
            intervention_kwargs: Additional keyword arguments to pass to intervention_fn

        Returns:
            Encoded tensor after intervention and activation
        """
        # Compute pre-activation
        pre_relu = self.encode_pre_relu(x)

        # Apply intervention if provided
        if intervention_fn is not None:
            pre_relu = intervention_fn(pre_relu, *intervention_args, **intervention_kwargs)

        # Apply activation function
        acts = nn.functional.relu(pre_relu)
        # zero all acts that are not in the top-k
        sparse_acts = torch.zeros_like(acts)
        topk_acts, topk_indices = acts.topk(self.cfg.k, sorted=False)
        sparse_acts.scatter_(-1, topk_indices, topk_acts)
        return sparse_acts

    def encode_activate(self, pre_relu: Tensor) -> Tensor:
        acts = nn.functional.relu(pre_relu)
        # zero all acts that are not in the top-k
        sparse_acts = torch.zeros_like(acts)
        topk_acts, topk_indices = acts.topk(self.cfg.k, sorted=False)
        sparse_acts.scatter_(-1, topk_indices, topk_acts)
        return sparse_acts

    def encode(self, x: Tensor) -> Tensor:
        """Internal encoding implementation."""
        acts = self.pre_acts(x)
        # zero all acts that are not in the top-k
        sparse_acts = torch.zeros_like(acts)
        topk_acts, topk_indices = acts.topk(self.cfg.k, sorted=False)
        sparse_acts.scatter_(-1, topk_indices, topk_acts)
        return sparse_acts

    def encode_top(self, x: Tensor) -> EncoderOutput:
        """Encode the input and select the top-k latents."""
        out = self.pre_acts(x)
        out = self.select_topk(out)
        return out

    def decode(self, acts: Tensor) -> Tensor:
        """Decode the activations."""
        acts = acts.to(self.dtype)
        acts = acts @ self.W_dec + self.b_dec
        return acts

    def decode_top(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        assert self.W_dec is not None, "Decoder weight was not initialized."

        top_acts = top_acts.to(self.dtype)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.W_dec.mT.shape[-1],))
        buf.scatter_add_(-1, top_indices, top_acts)
        acts = buf @ self.W_dec + self.b_dec
        return acts

    @staticmethod
    def find_zero_indices(top_indices, zero_features):
        # Create list to store indices to zero
        zero_indices = []

        # Iterate through top_indices
        for i, idx in enumerate(top_indices):
            # Check if current index appears in zero_features
            if idx in zero_features:
                zero_indices.append(i)

        return zero_indices

    def forward(
        self, x: Tensor, y: Tensor | None = None, *, dead_mask: Tensor | None = None
    ) -> ForwardOutput:
        pre_acts = self.pre_acts(x)

        # If we aren't given a distinct target, we're autoencoding
        if y is None:
            y = x

        # Decode and compute residual
        top_acts, top_indices = self.select_topk(pre_acts)
        sae_out = self.decode(top_acts, top_indices)
        e = sae_out - y

        # Used as a denominator for putting everything on a reasonable scale
        total_variance = (y - y.mean(0)).pow(2).sum()

        # Second decoder pass for AuxK loss
        if dead_mask is not None and (num_dead := int(dead_mask.sum())) > 0:
            # Heuristic from Appendix B.1 in the paper
            k_aux = y.shape[-1] // 2

            # Reduce the scale of the loss if there are a small number of dead latents
            scale = min(num_dead / k_aux, 1.0)
            k_aux = min(k_aux, num_dead)

            # Don't include living latents in this loss
            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)

            # Top-k dead latents
            auxk_acts, auxk_indices = auxk_latents.topk(k_aux, sorted=False)

            # Encourage the top ~50% of dead latents to predict the residual of the
            # top k living latents
            e_hat = self.decode(auxk_acts, auxk_indices)
            auxk_loss = (e_hat - e).pow(2).sum()
            auxk_loss = scale * auxk_loss / total_variance
        else:
            auxk_loss = sae_out.new_tensor(0.0)

        l2_loss = e.pow(2).sum()
        fvu = l2_loss / total_variance

        if self.cfg.multi_topk:
            top_acts, top_indices = pre_acts.topk(4 * self.cfg.k, sorted=False)
            sae_out = self.decode(top_acts, top_indices)

            multi_topk_fvu = (sae_out - y).pow(2).sum() / total_variance
        else:
            multi_topk_fvu = sae_out.new_tensor(0.0)

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            fvu,
            auxk_loss,
            multi_topk_fvu,
        )


    @classmethod
    def download_and_save(
        cls,
        save_path: Path | str,
        layer: str | int,
        repo_id: str = "fnlp/Llama3_1-8B-Base-LXR-8x",
        device: str | torch.device = "cuda",
        decoder: bool = True,
        force_download: bool = False,
        pattern: str | None = None,
    ) -> "TopkSae":
        """
        Download SAE weights from Hugging Face Hub and save them locally in a format
        compatible with load_from_disk.

        Args:
            save_path: Path to save the downloaded weights
            layer: Layer name (e.g., "layers.10") or layer number (e.g., 10)
            repo_id: Repository ID on Hugging Face Hub
            device: Device to load the model onto
            decoder: Whether to load decoder weights
            force_download: Whether to force download the weights
            pattern: Optional pattern to filter layer names

        Returns:
            The loaded TopkSae instance
        """
        from huggingface_hub import snapshot_download
        import shutil

        # Convert layer to layer number
        if isinstance(layer, int):
            layer_num = layer
        else:
            # Extract layer number from layer_name (e.g., "layers.10" -> 10)
            layer_num = int(layer.split(".")[-1])

        # For LLaMA SAEs, the repo structure uses Llama3_1-8B-Base-L{layer}R-8x format
        llama_dir_pattern = f"Llama3_1-8B-Base-L{layer_num}R-8x/*"

        # Download from the HuggingFace Hub
        repo_path = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=llama_dir_pattern,
                force_download=force_download,
            )
        )

        # The downloaded structure is: repo_path/Llama3_1-8B-Base-L{layer_num}R-8x/
        llama_model_dir = repo_path / f"Llama3_1-8B-Base-L{layer_num}R-8x"
        
        if not llama_model_dir.exists():
            raise FileNotFoundError(f"Downloaded directory not found at: {llama_model_dir}")

        # Create local directory structure using layer_x format
        save_path = Path(save_path)
        save_dir = save_path / f"layer_{layer_num}"
        
        # Copy the entire structure to our cache directory
        if save_dir.exists() and not force_download:
            print(f"SAE for layer {layer_num} already exists in cache at {save_dir}")
        else:
            save_dir.mkdir(parents=True, exist_ok=True)
            if save_dir.exists():
                shutil.rmtree(save_dir)
            shutil.copytree(llama_model_dir, save_dir)
            print(f"SAE for layer {layer_num} downloaded and saved to {save_dir}")

        # Load the SAE from the saved directory
        sae = cls.load_from_disk(save_dir, device=device, decoder=decoder)

        return sae

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps


class JumpReLUSAE(Sae):
    """Jump ReLU Sparse Autoencoder."""

    def __init__(self, d_in: int, d_sae: int, device: str | torch.device = "cpu", dtype: torch.dtype | None = None):
        super().__init__(d_in, d_sae, device, dtype)
        self._device = torch.device(device)
        self.d_sae = d_sae

        # Note that we initialize these to zeros because we're loading in pre-trained weights
        self.W_enc = nn.Parameter(torch.zeros(d_in, d_sae, device=device, dtype=dtype))
        self.W_dec = nn.Parameter(torch.zeros(d_sae, d_in, device=device, dtype=dtype))
        self.threshold = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.b_enc = nn.Parameter(torch.zeros(d_sae, device=device, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(d_in, device=device, dtype=dtype))

    @property
    def device(self) -> torch.device:
        """Implement the abstract device property from Sae base class."""
        return self._device

    @property
    def dtype(self):
        return self.W_enc.dtype

    def encode_pre_relu(self, input_acts: Tensor) -> Tensor:
        """Compute the pre-ReLU activations."""
        input_acts = input_acts.to(self.dtype)
        input_acts = input_acts.to(self.device)
        pre_acts = input_acts @ self.W_enc + self.b_enc
        return pre_acts

    def encode_hook(
        self,
        input_acts: Tensor,
        intervention_fn=None,
        *intervention_args,
        **intervention_kwargs,
    ) -> Tensor:
        """
        Encode input with an intervention before activation.

        Args:
            input_acts: Input tensor
            intervention_fn: Function to apply to the pre-activation tensor
            intervention_args: Additional positional arguments to pass to intervention_fn
            intervention_kwargs: Additional keyword arguments to pass to intervention_fn

        Returns:
            Encoded tensor after intervention and activation
        """
        # Compute pre-activation
        pre_relu = self.encode_pre_relu(input_acts)

        # Apply intervention if provided
        if intervention_fn is not None:
            pre_relu = intervention_fn(pre_relu, *intervention_args, **intervention_kwargs)

        # Apply activation function
        acts = torch.nn.functional.relu(pre_relu)
        mask = (pre_relu > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_relu)
        return acts

    def encode_activate(self, pre_relu: Tensor) -> Tensor:
        acts = torch.nn.functional.relu(pre_relu)
        mask = (pre_relu > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_relu)
        return acts

    def encode(self, input_acts: Tensor) -> Tensor:
        """Internal encoding implementation."""
        input_acts = input_acts.to(self.dtype)
        input_acts = input_acts.to(self.device)
        pre_acts = input_acts @ self.W_enc + self.b_enc
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return acts

    def encode_top(self, input_acts: Tensor) -> EncoderOutput:
        """
        Encode using sparse scatter indices."""
        acts = self.encode(input_acts)
        return EncoderOutput(acts, acts)

    def decode(self, acts: Tensor) -> Tensor:
        """Decode the activations."""
        acts = acts.to(self.dtype)
        acts = acts.to(self.device)
        acts = acts @ self.W_dec + self.b_dec
        return acts

    def decode_top(self, top_acts: Tensor, top_indices: Tensor) -> Tensor:
        """Decode from EncoderOutput format to compatibility with TopkSae."""
        assert self.W_dec is not None, "Decoder weight was not initialized."

        top_acts = top_acts.to(self.dtype)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.W_dec.mT.shape[-1],))
        buf.scatter_add_(-1, top_indices, top_acts)
        acts = buf @ self.W_dec + self.b_dec
        return acts

    def forward(self, acts: Tensor, **kwargs) -> Tensor:
        acts = acts.to(self.device)
        encoded = self.encode(acts)
        recon = self.decode(encoded)
        return recon

    def save_to_disk(self, path: Path | str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        save_model(self, str(path / "sae.safetensors"))
        with open(path / "cfg.json", "w") as f:
            json.dump(
                {
                    "d_in": self.d_in,
                    "d_sae": self.d_sae,
                    "sae_type": "jumprelu",
                },
                f,
            )

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device,
    ) -> "JumpReLUSAE":

        # if . exists in layer replace with _
        path = str(path).replace("layers.", "layer_")
        path = Path(path)
        path = path / "width_16k"

        # get file name in the path
        files = os.listdir(path)
        path = path / files[0] / "params.npz"

        params = np.load(path)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}
        sae = JumpReLUSAE(params['W_enc'].shape[0], params['W_enc'].shape[1], device=device)
        sae.load_state_dict(pt_params)
        return sae

    @classmethod
    def download_and_save(
        cls,
        save_path: Path | str,
        layer: int,
        repo_id: str = "google/gemma-scope-2b-pt-res",
        width: str = "width_16k",
        device: str | torch.device = "cuda",
        force_download: bool = False,
    ) -> "JumpReLUSAE":
        """
        Download SAE weights from Hugging Face Hub and save them locally in a format
        compatible with load_from_disk.

        Args:
            save_path: Path to save the downloaded weights
            average: Average type (e.g., "mean", "median")
            layer: Layer number
            repo_id: Repository ID on Hugging Face Hub
            width: Width of the model (e.g., "width_16k")
            device: Device to load the model onto
            force_download: Whether to force download the weights
        Returns:
            The loaded JumpReLUSAE instance
        """
        from huggingface_hub import hf_hub_download

        layer_to_average = {
            0: "average_l0_105",
            1: "average_l0_102",
            2: "average_l0_141",
            3: "average_l0_59",
            4: "average_l0_124",
            5: "average_l0_68",
            6: "average_l0_70",
            7: "average_l0_69",
            8: "average_l0_71",
            9: "average_l0_73",
            10: "average_l0_77",
            11: "average_l0_80",
            12: "average_l0_82",
            13: "average_l0_84",
            14: "average_l0_84",
            15: "average_l0_78",
            16: "average_l0_78",
            17: "average_l0_77",
            18: "average_l0_74",
            19: "average_l0_73",
            20: "average_l0_71",
            21: "average_l0_70",
            22: "average_l0_72",
            23: "average_l0_75",
            24: "average_l0_73",
            25: "average_l0_116"
        }

        average = layer_to_average[layer]
        # Download the parameters
        path_to_params = hf_hub_download(
            repo_id=repo_id,
            filename=f"layer_{layer}/{width}/{average}/params.npz",
            force_download=force_download,
        )

        # Load parameters
        params = np.load(path_to_params)
        pt_params = {k: torch.from_numpy(v).to(device) for k, v in params.items()}

        # Create and load SAE
        sae = cls(params['W_enc'].shape[0], params['W_enc'].shape[1], device=device)
        sae.load_state_dict(pt_params)

        # Create local directory structure
        save_path = Path(save_path)
        save_dir = save_path / f"layer_{layer}" / width / average
        os.makedirs(save_dir, exist_ok=True)

        # Save locally
        np.savez(save_dir / "params.npz", **{k: v.cpu().numpy() for k, v in pt_params.items()})

        return sae


class ModelSaes:
    def __init__(
        self,
        sae_type: str,
        layers: list[str] | list[int] | tuple[int, int] | None = None,
        device: str | torch.device = 'auto',
    ):
        """Initialize ModelSaes by loading SAEs from repository or cache.

        Args:
            sae_type: Repository name or local path to load SAEs from
            layers: Can be:
                - None (load all layers)
                - List of string layer names (["layers.10", "layers.11"])
                - List of layer numbers ([10, 11])
                - Tuple of (start, end) layer numbers inclusive ((10, 12) for layers 10-12)
            device: Device to load the SAEs onto. Use 'auto' to distribute across GPUs
        """

        # Convert layer specifications to layer names
        if isinstance(layers, (tuple)):
            if len(layers) == 2 and isinstance(layers[0], int) and isinstance(layers[1], int):
                layer_names = self._layer_range_to_names(layers[0], layers[1])
        elif isinstance(layers, (list)) and all(isinstance(x, int) for x in layers):
            layer_names = [self._layer_num_to_name(x) for x in layers]
        else:
            layer_names = layers

        self.model_name = sae_type
        self.layer_device_map = {}  # Track which device each layer is on
        self.saes = {}
        self.layer_device_map = {}
        self._layer_names = []

        # Setup device allocation strategy
        self.distribute_layers = device == 'auto' and torch.cuda.is_available() and torch.cuda.device_count() > 1
        self.default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if self.distribute_layers:
            print(f"Distributing SAEs across {torch.cuda.device_count()} GPUs")
            self.gpu_memory_info = get_gpu_memory_info()
            self.gpus_by_memory = sorted(self.gpu_memory_info.keys(),
                                        key=lambda x: self.gpu_memory_info[x],
                                        reverse=True)
            self.current_gpu_idx = 0  # Start with GPU with most memory

        if "gemma" in sae_type.lower():
            cache_dir = GEMMA_SAE_CACHE_PATH
        elif "llama" in sae_type.lower():
            cache_dir = LLAMA_3_1_SAE_CACHE_PATH
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}. Please specify a valid cache path.")

        if cache_dir and Path(cache_dir).exists():
            self.cache_dir = Path(cache_dir)
            try:
                print(f"Loading from cache: {self.cache_dir}")
                cached = self._load_distributed_from_disk(self.model_name, layers=layer_names)
                if layers is None or all(layer in cached.layer_names for layer in layer_names):
                    self.saes = cached.saes
                    self.layer_device_map = cached.layer_device_map
                    self._layer_names = cached.layer_names
                    if layers:
                        self.saes = {k: self.saes[k] for k in layer_names}
                        self.layer_device_map = {k: self.layer_device_map[k] for k in layer_names}
                        self._layer_names = natsorted(self.saes.keys())
                    return
            except Exception as e:
                print(f"Cache load failed: {e}")
            else:
                print("Cache load failed, loading from repo")
        else:
            raise FileNotFoundError(f"Cache directory {cache_dir} does not exist.")

    def _get_next_device(self, estimated_size_mb=300):
        """Get the next appropriate device for loading a layer based on memory availability."""
        if not self.distribute_layers:
            return self.default_device

        # Simple round-robin strategy with some memory tracking
        device_id = self.gpus_by_memory[self.current_gpu_idx]
        device_str = f"cuda:{device_id}"

        # Track estimated memory usage
        self.gpu_memory_info[device_id] -= estimated_size_mb * 1024 * 1024  # Convert MB to bytes

        # Move to next GPU for next layer
        self.current_gpu_idx = (self.current_gpu_idx + 1) % len(self.gpus_by_memory)

        return device_str

    def _load_distributed_from_disk(
        self,
        sae_type: str,
        layers: list[str] | None = None
    ) -> "ModelSaes":
        """Load ModelSaes from disk with distribution across GPUs."""

        # Create instance
        instance = ModelSaes.__new__(ModelSaes)
        instance.model_name = sae_type
        instance.saes = {}
        instance.layer_device_map = {}
        instance._layer_names = []

        # Configure device distribution
        instance.distribute_layers = self.distribute_layers
        instance.default_device = self.default_device

        if instance.distribute_layers:
            instance.gpu_memory_info = get_gpu_memory_info()
            instance.gpus_by_memory = sorted(instance.gpu_memory_info.keys(),
                                            key=lambda x: instance.gpu_memory_info[x],
                                            reverse=True)
            instance.current_gpu_idx = 0

        # Load requested SAEs
        for layer in tqdm(layers, desc="Loading SAEs"):
            device = instance._get_next_device()
            print(f"Loading {layer} on {device}")
            # Convert layers.X format to layer_X format for directory name
            layer_dir = layer.replace("layers.", "layer_")
            layer_path = Path(self.cache_dir) / layer_dir

            sae_class = Sae.get_sae_class(sae_type)
            sae = sae_class.load_from_disk(layer_path, device=device)
            instance.saes[layer] = sae
            instance.layer_device_map[layer] = device

        instance._layer_names = natsorted(instance.saes.keys())
        return instance

    def get_layer_device(self, layer_name: str) -> str:
        """Get the device where a specific layer is loaded."""
        if isinstance(layer_name, int):
            layer_name = self._layer_num_to_name(layer_name)
        return self.layer_device_map.get(layer_name, self.default_device)

    def __getitem__(self, layer: str | int) -> Sae:
        layer_name = self._layer_num_to_name(layer) if isinstance(layer, int) else layer
        return self.saes[layer_name]

    def __iter__(self):
        return iter(self.saes.items())

    def __len__(self) -> int:
        return len(self.saes)

    def __str__(self) -> str:
        layers_str = '\n    '.join(self.layer_names)
        return f"ModelSaes\n  Model: {self.model_name}\n  Layers:\n    {layers_str}, \n  Devices:\n    {self.layer_device_map}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def layer_names(self) -> list[str]:
        return self._layer_names

    def encode_hook(
        self,
        hidden_states: torch.Tensor,
        layer: int,
        intervention_fn=None,
        *intervention_args,
        **intervention_kwargs,
    ) -> Tensor:
        """Encode hidden states using SAE with optional intervention."""

        layer_device = self.get_layer_device(layer)
        layer_name = self._layer_num_to_name(layer)
        hidden_states = hidden_states.to(layer_device)
        sae = self.saes[layer_name]
        return sae.encode_hook(hidden_states, intervention_fn, *intervention_args, **intervention_kwargs)

    def encode(
        self,
        hidden_states: torch.Tensor,
        layer: int,
    ) -> Tensor:
        """Encode hidden states using SAE.

        This method handles different SAE types transparently.
        """
        layer_device = self.get_layer_device(layer)
        layer_name = self._layer_num_to_name(layer)
        hidden_states = hidden_states.to(layer_device)
        sae = self.saes[layer_name]
        return sae.encode(hidden_states)

    def encode_with_stats(
        self,
        hidden_states: torch.Tensor,
        layer: int,
    ) -> Tensor:
        """Compute the pre-ReLU activations for the specified layers."""

        layer_device = self.get_layer_device(layer)
        layer_name = self._layer_num_to_name(layer)
        hidden_states = hidden_states.to(layer_device)
        sae = self.saes[layer_name]
        pre_relu = sae.encode_pre_relu(hidden_states)
        mean = pre_relu.mean(dim=-1)
        std = pre_relu.std(dim=-1)
        encoded = sae.encode_activate(pre_relu)
        return encoded, mean, std

    def decode(
        self,
        encoded_state: Tensor,
        layer: int,
    ) -> list[torch.Tensor] | torch.Tensor:
        """Decode SAE activations back to hidden states."""

        if isinstance(layer, int):
            layer = self._layer_num_to_name(layer)

        # Get the layer's SAE
        sae = self.saes[layer]
        layer_device = self.get_layer_device(layer)

        if isinstance(encoded_state, EncoderOutput):
            top_acts = encoded_state.top_acts.to(layer_device)
            top_indices = encoded_state.top_indices.to(layer_device)
            return sae.decode_top(top_acts, top_indices)
        elif isinstance(encoded_state, torch.Tensor):
            encoded_state = encoded_state.to(layer_device)
            return sae.decode(encoded_state)
        else:
            raise ValueError(f"Unsupported encoded state type: {type(encoded_state)}, expected Tensor or EncoderOutput")

    def to(self, device: str | torch.device) -> "ModelSaes":
        for sae in self.saes.values():
            sae.to(device)
        return self

    def save_to_disk(self, path: Path | str):
        """Save all SAEs and metadata to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save metadata
        with open(path / "model_config.json", "w") as f:
            json.dump({
                "model_name": self.model_name,
                "layers": self.layer_names,
            }, f)

        # Save each SAE in its own subdirectory
        for layer_name, sae in tqdm(self.saes.items(), desc="Saving SAEs", leave=False):
            tqdm.write(f"Saving {layer_name}...")
            layer_path = path / layer_name
            sae.save_to_disk(layer_path)

    @classmethod
    def load_from_disk(
        cls,
        path: Path | str,
        device: str | torch.device = "cpu",
        decoder: bool = True,
        layers: list[str] | None = None
    ) -> "ModelSaes":
        """Load ModelSaes from disk."""
        path = Path(path)

        # Load metadata
        with open(path / "model_config.json", "r") as f:
            config = json.load(f)

        # Validate requested layers exist in saved config
        if layers is not None:
            missing = [layer for layer in layers if layer not in config["layers"]]
            if missing:
                raise ValueError(f"Requested layers not found in cache: {missing}")
            load_layers = layers
        else:
            load_layers = config["layers"]

        # Load requested SAEs
        saes = {}
        for layer in tqdm(load_layers, desc="Loading SAEs"):
            tqdm.write(f"Loading {layer}...")

            # Determine SAE type from config
            layer_path = path / layer
            with open(layer_path / "cfg.json", "r") as f:
                cfg_dict = json.load(f)
                sae_type = cfg_dict.get("sae_type", "topk")  # Default to topk for backward compatibility

            sae_class = Sae.get_sae_class(sae_type)
            sae = sae_class.load_from_disk(layer_path, device=device, decoder=decoder)
            saes[layer] = sae

        # Create instance
        instance = cls.__new__(cls)
        instance.model_name = config["model_name"]
        instance.saes = saes
        instance._layer_names = natsorted(saes.keys())
        instance.cache_dir = path
        instance.layer_device_map = {layer: str(device) for layer in saes}
        instance.default_device = str(device)
        instance.distribute_layers = False

        return instance

    @staticmethod
    def _layer_num_to_name(num: int) -> str:
        return f"layers.{num}"

    @staticmethod
    def _layer_range_to_names(start: int, end: int) -> list[str]:
        return [f"layers.{i}" for i in range(start, end + 1)]
