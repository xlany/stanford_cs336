import numpy as np
import torch
import random



def load_batched_data(
	dataset: np.typing.NDArray, 
	batch_size: int, 
	context_length: int, 
	device: torch.device,
	) -> tuple[torch.Tensor, torch.Tensor]:
	"""We randomly sample batch_size numbers of (input sentence, output sentence) pairs.

	input = window[start:end]
	output = window[start+1:end+1]
	These are used to train next token prediction!
	"""
	chosen_starts = random.sample(range(len(dataset)-context_length), batch_size)
	inputs = [torch.from_numpy(dataset[start:start+context_length]) for start in chosen_starts]
	outputs = [torch.from_numpy(dataset[start+1:start+context_length+1]) for start in chosen_starts]
	chosen_inputs = torch.stack(inputs).to(device)
	chosen_outputs = torch.stack(outputs).to(device)
	return (chosen_inputs, chosen_outputs)