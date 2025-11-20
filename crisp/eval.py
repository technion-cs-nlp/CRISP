import os
import json
import numpy as np
import random
import torch
from tqdm.auto import tqdm

from globals import SEED, set_seed
from crisp import CRISP

set_seed()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_mcq_accuracy(model, type, tokenizer=None, split_data=False, val_ratio=0.5, seed=SEED, batch_size=8,
                        feature_ablation=False, error=False, topk_filter=True, k_features_ablate=None,
                        layers=None, alpha=None, verbose=True, return_answers=False):

    if isinstance(model, CRISP):
        tokenizer = model.tokenizer
        model = model.model
    elif tokenizer is None:
        raise ValueError("If model is not CRISP, tokenizer must be provided")
    elif feature_ablation:
        raise ValueError("Feature ablation is only supported with CRISP model")

    def prepare_data(data, batch_size):
        """
        Return a list of batches of the form (text_batch, answers_batch)
        """
        batches = []
        batch = []

        # Extract the questions, answers, and choices from the new format
        questions = data.get("questions", [])
        answers = data.get("answers", [])
        choices_lists = data.get("choices", [])

        # Ensure all lists have the same length
        n_items = min(len(questions), len(answers), len(choices_lists))

        for i in range(n_items):
            question = f"""
The following is a multiple choice question (with answer).

{questions[i]}
A. {choices_lists[i][0]}
B. {choices_lists[i][1]}
C. {choices_lists[i][2]}
D. {choices_lists[i][3]}
Answer:
"""
            ans = answers[i]
            batch.append((question, ans))
            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
        if batch:
            batches.append(batch)
        return batches

    def get_accuracy(model, tokenizer, batches):
        # get token idxs for A, B, C, D
        A_idx = tokenizer.encode("A", add_special_tokens=False)[-1]
        B_idx = tokenizer.encode("B", add_special_tokens=False)[-1]
        C_idx = tokenizer.encode("C", add_special_tokens=False)[-1]
        D_idx = tokenizer.encode("D", add_special_tokens=False)[-1]

        # set padding side to left for getting logits the last token in the sequence
        tokenizer.padding_side = "left"
        tokenizer.pad_token_id = tokenizer.eos_token_id

        choice_idxs = torch.tensor([A_idx, B_idx, C_idx, D_idx]).to(model.device)
        corrects = []

        pbar = tqdm(batches, desc="Evaluating batches", total=len(batches))
        for batch in pbar:
            texts = [x[0] for x in batch]
            answers = torch.tensor([x[1] for x in batch]).to(model.device)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                if feature_ablation:  # If using CRISP with feature ablation
                    outputs = model.get_model_outputs_with_ablation(
                        inputs,
                        k_features_ablate=k_features_ablate,
                        layers=layers,
                        alpha=alpha,
                        error=error,
                        topk_filter=topk_filter,
                    ).logits[:, -1, choice_idxs]
                else:
                    outputs = model(**inputs).logits[:, -1, choice_idxs]
            predictions = outputs.argmax(dim=-1)
            batch_correct = (predictions == answers).tolist()
            corrects.extend(batch_correct)
            batch_acc = sum(batch_correct) / len(batch_correct)
            total_acc = sum(corrects) / len(corrects)
            pbar.set_postfix({'Batch Acc': f'{batch_acc:.3f}', 'Total Acc': f'{total_acc:.3f}'})
        return corrects

    if type == "wmdp_bio":
        data_path = "data/wmdp/bio/bio_mcq.json"
    elif type == "college_bio":
        data_path = "data/wmdp/bio//college_bio_mcq.json"
    elif type == "high_school_bio":
        data_path = "data/wmdp/bio//high_school_bio_mcq.json"
    elif type == "wmdp_cyber":
        data_path = "data/wmdp/cyber/cyber_mcq.json"
    elif type =="high_school_cs":
        data_path = "data/wmdp/cyber/high_school_computer_science_mcq.json"
    elif type == "college_cs":
        data_path = "data/wmdp/cyber/college_computer_science_mcq.json"
    elif type == "hp":
        data_path = "data/hp/hp_mcq.json"
    elif type == "mmlu":
        data_path = "data/mmlu/mmlu.json"
    elif type == "mmlu_10":  # 10 examples per subject
        data_path = "data/mmlu/mmlu_10.json"
    else:
        raise ValueError("Invalid type specified")

    corrects = {}
    with open(data_path, "r") as fp:
        reader = json.load(fp)

    batches = prepare_data(reader, batch_size=batch_size)
    corrects[data_path] = get_accuracy(model, tokenizer, batches)

    all_corrects = [x for sublist in corrects.values() for x in sublist]
    all_acc = sum(all_corrects) / len(all_corrects)

    if verbose:
        print(f"Overall accuracy for {os.path.basename(data_path).replace('.json','')}: {all_acc:.3f}")

    # Handle split data if requested
    if split_data:
        # Set seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)

        # Create indices for val/test split
        n_samples = len(all_corrects)
        indices = list(range(n_samples))
        random.shuffle(indices)

        val_size = int(n_samples * val_ratio)
        val_indices = set(indices[:val_size])
        test_indices = set(indices[val_size:])

        # Split results
        val_corrects = [all_corrects[i] for i in range(n_samples) if i in val_indices]
        test_corrects = [all_corrects[i] for i in range(n_samples) if i in test_indices]

        val_acc = sum(val_corrects) / len(val_corrects) if val_corrects else 0.0
        test_acc = sum(test_corrects) / len(test_corrects) if test_corrects else 0.0

        if verbose:
            print(f"Val accuracy ({len(val_corrects)} samples): {val_acc:.3f}")
            print(f"Test accuracy ({len(test_corrects)} samples): {test_acc:.3f}")

        if return_answers:
            return {"val_corrects": val_corrects, "test_corrects": test_corrects, "all_corrects": all_corrects}
        else:
            return {"val_acc": val_acc, "test_acc": test_acc, "all_acc": all_acc}

    # Original behavior when split_data=False
    if return_answers:
        return all_corrects
    else:
        return all_acc