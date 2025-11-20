import torch

# ~~~~~~~~ Transforms ~~~~~~~~

def hs_to_logits(model, hs, requires_grad=False):
    '''
    hs: tensor of shape (num_neurons, embed_dim)
    requires_grad: whether the hs tensor requires gradients. Default is False.
    '''
    if model.config.model_type == 'gemma2':
        try:
            layer_norm = model.model.norm
            lm_head = model.lm_head
        except AttributeError:
            layer_norm = model.model.model.norm
            lm_head = model.lm_head
    elif model.config.model_type == 'llama':
        try:
            layer_norm = model.model.norm
            lm_head = model.lm_head
        except AttributeError:
            layer_norm = model.model.model.norm
            lm_head = model.lm_head
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    # Ensure hs has the same dtype as model parameters
    model_dtype = next(model.parameters()).dtype
    if hs.dtype != model_dtype:
        hs = hs.to(dtype=model_dtype)

    if requires_grad:
        logits = lm_head(layer_norm(hs))
    else:
        with torch.no_grad():
            logits = lm_head(layer_norm(hs))

    return logits

def hs_to_probs(model, hs, requires_grad=False):
    '''
    hs: tensor of shape (num_neurons, embed_dim)
    requires_grad: whether the hs tensor requires gradients. Default is False.
    '''
    logits = hs_to_logits(model, hs, requires_grad=requires_grad)
    if requires_grad:
        probs = torch.softmax(logits, dim=-1)
    else:
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
    return probs

@torch.no_grad()
def invert_lm_head(logits, lm_head_weight, lm_head_bias, lm_head_weight_inv=None):
    if lm_head_weight_inv is None:
        lm_head_weight_inv = torch.pinverse(lm_head_weight)
    hs = (logits - lm_head_bias) @ lm_head_weight_inv.T
    return hs

@torch.no_grad()
def invert_llama_layer_norm(normed_hs, mean, var, rmsnorm):
    # Extract the scale (gamma) parameter from the RMSNorm module
    gamma = rmsnorm.weight
    eps = rmsnorm.variance_epsilon
    # Compute the root mean square
    rms = torch.sqrt(var + eps)
    # Invert the RMSNorm operation
    x = (normed_hs  * rms.unsqueeze(1)) / gamma + mean.unsqueeze(1)
    return x

@torch.no_grad()
def logits_to_hs(model, logits, original_mean, original_var, lm_head_weight_inv=None):
    if model.config.model_type == 'gptj':
        layer_norm = model.transformer.ln_f
        lm_head_weight = model.lm_head.weight
        lm_head_bias = model.lm_head.bias
        invert_layer_norm = invert_gptj_layer_norm
    elif model.config.model_type == 'llama':
        layer_norm = model.model.norm
        lm_head_weight = model.lm_head.weight
        lm_head_bias = torch.zeros_like(lm_head_weight.T[0])  # llama doesn't have a bias
        invert_layer_norm = invert_llama_layer_norm
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    normed_hs = invert_lm_head(logits, lm_head_weight=lm_head_weight, lm_head_bias=lm_head_bias, lm_head_weight_inv=lm_head_weight_inv)
    hs = invert_layer_norm(normed_hs, original_mean, original_var, layer_norm)
    return hs

@torch.no_grad()
def test_logits_to_hs(model):
    if model.config.model_type == 'gptj':
        hs = model.transformer.h[0].mlp.fc_out.weight[:, :10].detach().clone().to(device).T
    elif model.config.model_type == 'llama':
        hs = model.model.layers[0].mlp.down_proj.weight[:, :10].detach().clone().to(device).T
    else:
        raise ValueError(f"Model type not supported {model.config.model_type}, supported models are 'gptj' and 'llama'")

    mean, var = hs.mean(dim=-1), hs.var(dim=-1)
    logits = hs_to_logits(model, hs)
    hs_restored = logits_to_hs(model, logits, mean, var)
    result = torch.allclose(hs, hs_restored, atol=1e-3, rtol=1e-3)

    if result:
        print("\n\t~~~~~~~~~~~ Test PASSED ~~~~~~~~~~~")
    else:
        print("\n\t~~~~~~~~~~~ Test FAILED ~~~~~~~~~~~")
    return hs, hs_restored


# ~~~~~~~~ Token Ranks ~~~~~~~~
@torch.no_grad()
def get_topk_token_ranks_in_vocab(vocab_distribution, top_k, largest=True):
    '''
    Return the top k tokens ranks and scores for each item in batch.
    Args:
        vocab_distribution: tensor of shape [batch_size, vocab_size] or [vocab_size]
        top_k: number of top tokens to return
        largest: if True return largest values, else smallest
    Returns:
        topk_token_indices: tensor of shape [batch_size, top_k] or [top_k]
        topk_token_logits: tensor of shape [batch_size, top_k] or [top_k]
    '''
    if len(vocab_distribution.shape) not in [1, 2]:
        raise ValueError("vocab_distribution should be 1 or 2 dimensional")

    # Handle both 1D and 2D cases with same code by adding batch dim if needed
    if len(vocab_distribution.shape) == 1:
        vocab_distribution = vocab_distribution.unsqueeze(0)

    topk_results = torch.topk(vocab_distribution, top_k, dim=1, largest=largest)
    topk_token_indices = topk_results.indices
    topk_token_logits = topk_results.values

    # Remove batch dimension if input was 1D
    if len(vocab_distribution.shape) == 1:
        topk_token_indices = topk_token_indices.squeeze(0)
        topk_token_logits = topk_token_logits.squeeze(0)

    return topk_token_indices, topk_token_logits

@torch.no_grad()
def get_token_logit_in_vocab(vocab_distribution, token_id):
    '''
    Return the logit of a given token id in the vocab distribution.
    Args:
        vocab_distribution: tensor of shape [batch_size, vocab_size] or [vocab_size]
        token_id: the token id to find the logit for
    Returns:
        token_score: the logit of the token in the vocab distribution
    '''
    if len(vocab_distribution.shape) not in [1, 2]:
        raise ValueError("vocab_distribution should be 1 or 2 dimensional")

    # Handle both 1D and 2D cases with same code by adding batch dim if needed
    if len(vocab_distribution.shape) == 1:
        vocab_distribution = vocab_distribution.unsqueeze(0)

    token_logits = vocab_distribution[:, token_id]

    # Remove batch dimension if input was 1D
    if len(vocab_distribution.shape) == 1:
        token_logits = token_logits.squeeze(0)

    return token_logits

@torch.no_grad()
def get_token_info_in_vocab(vocab_distribution, token_ids):
    '''
    Return the logit, probability, and rank of given token ids in the vocab distribution.
    Args:
        vocab_distribution: tensor of shape [batch_size, vocab_size] or [vocab_size]
        token_ids: the token id or list of token ids to find the information for
    Returns:
        token_logits: the logits of the tokens in the vocab distribution
        token_probs: the probabilities of the tokens in the vocab distribution
        token_ranks: the ranks of the tokens in the vocab distribution
    '''
    if len(vocab_distribution.shape) not in [1, 2]:
        raise ValueError("vocab_distribution should be 1 or 2 dimensional")

    # Handle both 1D and 2D cases with same code by adding batch dim if needed
    if len(vocab_distribution.shape) == 1:
        vocab_distribution = vocab_distribution.unsqueeze(0)

    if isinstance(token_ids, int):
        token_ids = [token_ids]

    token_logits = vocab_distribution[:, token_ids]
    token_probs = torch.softmax(vocab_distribution, dim=-1)[:, token_ids]
    token_ranks = (vocab_distribution.unsqueeze(1) >= token_logits.unsqueeze(-1)).sum(dim=-1) - 1

    # Remove batch dimension if input was 1D
    if len(vocab_distribution.shape) == 1:
        token_logits = token_logits.squeeze(0)
        token_probs = token_probs.squeeze(0)
        token_ranks = token_ranks.squeeze(0)

    return token_logits, token_probs, token_ranks