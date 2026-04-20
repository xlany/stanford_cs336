import cs336_basics.transformer.transformer as trans
import cs336_basics.training.loss_function  as loss_function
import cs336_basics.training.data_loading as data_loading

import torch
import numpy as np
import timeit
import statistics
import argparse


def benchmark_forward_backwards(
	model: trans.Transformer,
	train_inputs: tuple[torch.Tensor, torch.Tensor],
	train_targets: tuple[torch.Tensor, torch.Tensor],
	num_warmups: int,
	num_trials: int,
) -> tuple[list[float], list[float]]:
	"""Measure times of the forward and backward model passes."""
	forward_times = []
	backward_times = []

	def run_forward_backward_pass(
		record_time: bool
	):
		if record_time:
			forward_start_time = timeit.default_timer()
		train_logits = model.forward(train_inputs)
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		if record_time:
			forward_end_time = timeit.default_timer()
			forward_times.append(forward_end_time - forward_start_time)

		train_loss = loss_function.avg_cross_entropy(
			inputs=train_logits.view(-1, train_logits.size(-1)),
			targets=train_targets.view(-1),
		)
		if torch.cuda.is_available():
			torch.cuda.synchronize()

		if record_time:
			backward_start_time = timeit.default_timer()
		train_loss.backward()
		if torch.cuda.is_available():
			torch.cuda.synchronize()
		if record_time:
			backward_end_time = timeit.default_timer()
			backward_times.append(backward_end_time - backward_start_time)
	
	for _ in range(num_warmups):
		run_forward_backward_pass(record_time=False)

	for _ in range(num_trials):
		run_forward_backward_pass(record_time=True)

	return forward_times, backward_times


def random_batched_data(
	vocab_size: int,
	batch_size: int, 
	context_length: int, 
	device: torch.device,
	) -> tuple[torch.Tensor, torch.Tensor]:
	random_dataset = np.random.randint(low=0, high=vocab_size, size=500_000)
	return data_loading.load_batched_data(
		dataset=random_dataset,
		batch_size=batch_size,
		context_length=context_length,
		device=device,
    )
	

def run_transformer_benchmark(
	num_warmups: int,
	num_trials: int,
	vocab_size: str,
	batch_size: int,
	context_length: int,
	d_model: int,
	d_ff: int,
	num_layers: int,
	num_heads: int,
	rope_theta: int | None,
	device: torch.device,
	dtype: torch.dtype,
) -> None:
	
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
	
	train_inputs, train_targets = random_batched_data(
		vocab_size=vocab_size,
		batch_size=batch_size,
		context_length=context_length,
		device=device,
    )

	forward_pass, backward_pass = benchmark_forward_backwards(
		model=model,
		train_inputs=train_inputs,
		train_targets=train_targets,
		num_warmups=num_warmups,
		num_trials=num_trials,
		)
	
	print(f'Forward pass: mean {statistics.mean(forward_pass):.3g}, std {statistics.stdev(forward_pass):.3g}')
	print(f'Backward pass: {statistics.mean(backward_pass):.3g}, std {statistics.stdev(backward_pass):.3g}')
	
	
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--num_warmups', type=int)
	parser.add_argument('--num_trials', type=int)
	parser.add_argument('--vocab_size', type=int)
	parser.add_argument('--batch_size', type=int)
	parser.add_argument('--context_length', type=int)

	parser.add_argument('--d_model', type=int)
	parser.add_argument('--d_ff', type=int)
	parser.add_argument('--num_layers', type=int)
	parser.add_argument('--num_heads', type=int)

	args = parser.parse_args()

	run_transformer_benchmark(
		num_warmups=args.num_warmups,
		num_trials=args.num_trials,
		vocab_size=args.vocab_size,
		batch_size=args.batch_size,
		context_length=args.context_length,
		d_model=args.d_model,
		d_ff=args.d_ff,
		num_layers=args.num_layers,
		num_heads=args.num_heads,
		rope_theta=10_000,
		device=torch.device("mps"),
		dtype=torch.float32,
	)

