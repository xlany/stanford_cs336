import torch
import numpy as np

from transformer import transformer as trans
from training import (
	data_loading,
	optimizer as opt,
	loss_function,
)

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


def train(
	filepath: str,
	vocab_size: str,
	batch_size: int,
	context_length: int,
	d_model: int,
	d_ff: int,
	num_layers: int,
	num_heads: int,
	rope_theta: int,
	lr: float,
	weight_decay: float,
	betas: tuple[float, float],
	training_steps: int,
	device: str,
):
	dataset = np.load(filepath, mmap_mode='r')
	print(f'Loaded dataset from {filepath}')
	inputs, targets = data_loading.load_batched_data(
		dataset=dataset,
		batch_size=batch_size,
		context_length=context_length,
		device=device,
	)
	print(f'Loaded inputs and targets, length {len(inputs)}')

	model = trans.Transformer(
		vocab_size=vocab_size,
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		rope_theta=rope_theta
	)

	optimizer = opt.AdamW(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay,
		betas=betas,
	)
	print(f'Initialized model and optimizer')

	for step in range(training_steps):
		print(f'Starting step {step}')
		optimizer.zero_grad()
		loss = loss_function.cross_entropy(
			inputs=inputs,
			targets=targets,
		)
		print(loss.cpu().item())
		loss.backward()
		optimizer.step()


if __name__=='__main__':
	filepath = './data/TinyStoriesV2-GPT4-valid_tokenized.npy'

	train(
		filepath=filepath,
		vocab_size=10_000,
		batch_size=32,
		context_length=256,
		d_model=512,
		d_ff=1344,
		num_layers=4,
		num_heads=16,
		rope_theta=10_000,
		lr=1e-3,
		weight_decay=0.01,
		betas=[0.9, 0.95],
		training_steps=100,
		device='mps'
	)