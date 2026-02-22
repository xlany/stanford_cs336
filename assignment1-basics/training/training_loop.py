import torch
import optimizer as optimizer_lib

def training_loop(
	weights: torch.Tensor, 
	optimizer: torch.optim.Optimizer, 
	training_steps: int) -> None:
	"""Directly copied over from assignment 4.2.1."""

	for t in range(training_steps):
		optimizer.zero_grad() # Reset the gradients for all learnable parameters. 
		loss = (weights**2).mean() # Compute a scalar loss value. 
		print(loss.cpu().item())
		loss.backward() # Run backward pass, which computes gradients. 
		optimizer.step() # Run optimizer step.

