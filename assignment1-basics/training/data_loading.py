import numpy as np
import torch
import random


def load_batched_data(
	dataset: np.typing.NDArray, 
	batch_size: int, 
	context_length: int, 
	device: str
	) -> tuple[torch.Tensor, torch.Tensor]:
	"""Turn the entire dataset into samples of input -> output where
		input = window[start:end]
		output = window[start+1:end+1]
		These are used to train next token prediction!

	We randomly sample batch_size numbers of these pairs.
	"""
	inputs: list[torch.Tensor] = []
	outputs: list[torch.Tensor] = []
	for start in range(len(dataset)-context_length):
		end = start+context_length
		inputs.append(torch.from_numpy(dataset[start:end]))
		outputs.append(torch.from_numpy(dataset[start+1:end+1]))

	possible_indicies = [i for i in range(len(inputs))]
	chosen_indicies = random.sample(possible_indicies, batch_size)
	chosen_inputs = torch.stack([inputs[i] for i in chosen_indicies]).to(device)
	chosen_outputs = torch.stack([outputs[i] for i in chosen_indicies]).to(device)
	return (chosen_inputs, chosen_outputs)
