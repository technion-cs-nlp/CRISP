# CRISP: Persistent Concept Unlearning via Sparse Autoencoders

[![Website](https://img.shields.io/badge/Website-CRISP-blue)](https://technion-cs-nlp.github.io/CRISP/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red)](https://arxiv.org/abs/2508.13650)

Welcome to the official repository for **CRISP**, a parameter-efficient method for persistent concept unlearning in large language models using sparse autoencoders (SAEs).

**🚨 The Problem**: Large language models (LLMs) memorize harmful or sensitive knowledge. Existing unlearning methods often degrade general utility or fail to permanently remove the knowledge, allowing it to resurface under specific prompting or attacks.

**✅ Our Solution**: CRISP leverages Sparse Autoencoders (SAEs) to automatically identify and suppress specific features activated by harmful knowledge. By fine-tuning with LoRA to suppress these features, CRISP achieves persistent removal while preserving the model's fluency and benign capabilities.

## CRISP Main Method

![CRISP Method Overview](assets/method-main.png)

(1) **Feature Selection**: Identify SAE features active on target data but not benign data.
(2) **Model Optimization**: Fine-tune (LoRA) to suppress these features.
(3) **Result**: Persistent unlearning with minimal collateral damage.

## 🎯 Key Features

- **🛡️ Persistent Unlearning**: Modifies model weights (via LoRA) rather than just steering inference, ensuring permanent removal.
- **🧠 Interpretable**: Uses Sparse Autoencoders to identify semantically meaningful features associated with the concept to be unlearned.
- **⚡ Parameter-Efficient**: Updates only a small fraction of parameters using Low-Rank Adaptation (LoRA).
- **📊 High Precision**: Disentangles harmful concepts from benign ones, preserving general model capabilities and fluency.

## 🚀 Quick Start

### Installation

Set up your environment using the provided configuration:

```bash
# Clone the repository
git clone https://github.com/tomerashuach/CRISP.git
cd CRISP
```

1. Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate crisp_env
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

## 📚 Demo

We provide a demo notebook showcasing the unlearning of the "Harry Potter" concept:

```bash
jupyter notebook demo_unlearn_hp.ipynb
```

This demo illustrates how CRISP identifies and suppresses features related to specific knowledge.

### Supported Models

- **Llama-3.1-8B**
- **Gemma-2-2B**

*Uses publicly available SAEs from LlamaScope and GemmaScope.*

## 📊 Datasets

The repository supports evaluation and unlearning on:

1.  **WMDP (Weapons of Mass Destruction Proxies)**:
    *   **Biosecurity**: Virology knowledge vs. general biology.
    *   **Cybersecurity**: Harmful cyber instructions vs. general computer science.
    
    *Note: Due to the WMDP policy, this repo does not contain the WMDP dataset. To use CRISP on WMDP, one needs to request access via [https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus](https://huggingface.co/datasets/cais/wmdp-bio-forget-corpus).*

2.  **Harry Potter**: Used for demonstration and analysis of copyright/book knowledge unlearning.

## 🔬 Method Overview

CRISP operates in two key phases:

### 1. 🎯 Feature Selection
*   **Activation Statistics**: Compute SAE feature activations on a **Target Corpus** (harmful) and a **Retain Corpus** (benign).
*   **Filtering**: Select features with high activation frequency and high relative activation ratio on the target set.

### 2. ✂️ Model Optimization
*   **Suppression**: Fine-tune the model using LoRA adapters.
*   **Objective**: Minimize activations of selected features on the target corpus while preserving original hidden states on the retain corpus.
*   **Loss Function**: Combines unlearning loss, retention loss, and coherence loss.

## 📖 Paper Citation

If you use CRISP in your research, please cite:

```bibtex
@article{ashuach2025crisp,
  title={CRISP: Persistent Concept Unlearning via Sparse Autoencoders},
  author={Ashuach, Tomer and Arad, Dana and Mueller, Aaron and Tutek, Martin and Belinkov, Yonatan},
  journal={arXiv preprint arXiv:2508.13650},
  year={2025},
  url={https://arxiv.org/abs/2508.13650}, 
}
```
