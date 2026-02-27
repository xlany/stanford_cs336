import torch
import numpy as np
import os
import wandb

from transformer import transformer as trans
from training import (
	data_loading,
	optimizer as opt,
	loss_function,
	gradient_clipping,
	checkpointing,
)

LOG_PER_STEP = 100
CHECKPOINT_PER_STEP = 500

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
	run_name: str,
	dataset_filepath: str,
	checkpoint_filepath: str,
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
	max_l2_norm: float,
	training_steps: int,
	device: torch.device,
	dtype: torch.dtype,
):
	"""Main training function."""
	# Load long list of tokens via virtual memory mapping
	# Thus it's still on disk rather than load into RAM
	# When data_loading asks for a specific slice, the OS loads that page on-demand
	dataset = np.load(dataset_filepath, mmap_mode='r')
	print(f'Loaded dataset from {dataset_filepath}')

	# Initialize model and optimizer
	model = trans.Transformer(
		vocab_size=vocab_size,
		context_length=context_length,
		d_model=d_model,
		num_layers=num_layers,
		num_heads=num_heads,
		d_ff=d_ff,
		rope_theta=rope_theta,
		device=device,
		dtype=dtype,
	)
	optimizer = opt.AdamW(
		model.parameters(),
		lr=lr,
		weight_decay=weight_decay,
		betas=betas,
	)
	print(f'Initialized model and optimizer')

	# Resume run if checkpoints already exist
	steps_already_run = 0
	wandb_run_id = None
	if os.path.exists(checkpoint_filepath):
		print(f'Resuming training...')
		wandb_run_id, steps_already_run = checkpointing.load_latest_checkpoint(
			folder=checkpoint_filepath,
			model=model,
			optimizer=optimizer,
		)
	os.makedirs(checkpoint_filepath, exist_ok=True)

	if wandb_run_id:
		run = wandb.init(
			project='stanford_cs336_assignment1',
			name=run_name,
			id=wandb_run_id
		)
	else:
		run = wandb.init(
			project='stanford_cs336_assignment1',
			name=run_name,
		)
		wandb_run_id = run.id

	# Start training!
	for step in range(steps_already_run, training_steps):
		if step % LOG_PER_STEP == 0:
			print(f'Starting step {step}')
		# At each step, sample a new batch of data from the entire dataset
		# Inputs: [batch_size sequence_len] = [32, 256]
		# Model forward pass: [batch_size sequence_len vocab_size] = [32, 256, 10000] with logits of each of the 10000 possible words
		# Loss function inputs: [word_position vocab_size] = [32 * 256, 10000]. 
		# 	Flatten the previous output so that each row is the logits of the 10000 possible words for the position in [batch_size, sequence_len]
		inputs, targets = data_loading.load_batched_data(
			dataset=dataset,
			batch_size=batch_size,
			context_length=context_length,
			device=device,
		)
		# Forward pass
		optimizer.zero_grad()  # Reset the optimizer, ready to accumulate next round of gradients
		logits = model.forward(inputs)  # Model step, model looks at tokens -> outputs predicted logits
		loss = loss_function.avg_cross_entropy(  # Calculate loss via avg cross entropy
			inputs=logits.view(-1, logits.size(-1)),
			targets=targets.view(-1),
		)
		run.log({'loss': loss.cpu().item()})
		if step % LOG_PER_STEP == 0:
			print(loss.cpu().item())
		loss.backward()  # Calculate the gradients
		gradient_clipping.clip_gradients(parameters=model.parameters(), max_l2_norm=max_l2_norm)  # Clip the gradients
		optimizer.step()  # Optimizer step

		if step % CHECKPOINT_PER_STEP == 0 and step != 0:
			print(f'Saving checkpoint for step {step}')
			checkpointing.save_checkpoint(
				model=model,
				optimizer=optimizer,
				iteration=step,
				wandb_run_id=wandb_run_id,
				out=f'{checkpoint_filepath}/{step}.pt' 
			)
	
	checkpointing.save_checkpoint(
		model=model,
		optimizer=optimizer,
		iteration=step,
		wandb_run_id=wandb_run_id,
		out=f'{checkpoint_filepath}/{step}.pt' 
	)
	run.finish()
	
if __name__=='__main__':
	run_name = 'baseline_owt'
	# dataset_filepath = './data/TinyStoriesV2-GPT4-valid_tokenized.npy'
	dataset_filepath = './data/owt_train_tokenized.npy'
	checkpoint_filepath = './checkpoints'

	train(
		run_name=run_name,
		dataset_filepath=dataset_filepath,
		checkpoint_filepath=f'{checkpoint_filepath}/{run_name}/',
		vocab_size=10_000,
		batch_size=32,
		context_length=256,
		d_model=512,
		d_ff=1344,
		num_layers=4,
		num_heads=16,
		rope_theta=4_000,
		lr=1e-3,
		weight_decay=0.01,
		betas=[0.9, 0.95],
		max_l2_norm=1.0,
		training_steps=5_000,
		device=torch.device("mps"),
		dtype=torch.float32,
	)