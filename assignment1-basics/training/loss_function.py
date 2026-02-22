import torch
import torch.nn.functional as F
from transformer import softmax

def cross_entropy(
	inputs: torch.Tensor, 
	targets: torch.Tensor,
	) -> torch.Tensor:
    """
    Inputs:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            raw logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Cross entropy loss for each example. Tensor of shape (batch_size,)
    """
    vocab_size = inputs.shape[-1]

    # Don't use this—softmax doesn't handle numerical overflow
    # Convert logits into probability distr that sum to 1 across row
    # probs = softmax.softmax(inputs, dim_i=-1)
    # log_probs = -torch.log(probs)

    # Use this instead
    log_probs = softmax.neg_log_softmax(inputs, dim_i=-1)

    # Expand ith position (number) into one hot encoding
    one_hot_encoding = F.one_hot(targets, num_classes=vocab_size).float()

    # Only look at "how off" for the word that should have been predicted
    masked = log_probs * one_hot_encoding

    loss_for_each_example = torch.sum(masked, dim=-1)

    return loss_for_each_example


def avg_cross_entropy(
	inputs: torch.Tensor, 
	targets: torch.Tensor,
	) -> torch.Tensor:
    """Compute average cross entropy across batches"""
    return torch.mean(cross_entropy(inputs,targets), dim=-1)



	