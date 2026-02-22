from collections.abc import Callable, Iterable 
from typing import Optional
import torch
import math


class SGD(torch.optim.Optimizer):
	"""Directly copied over from assignment 4.2.1."""

	def __init__(self, params, lr=1e-3):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		defaults = {"lr": lr}
		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None): 
		loss = None if closure is None else closure() 
		for group in self.param_groups:
			lr = group["lr"] # Get the learning rate.
			for p in group["params"]: 
				if p.grad is None:
					continue

				state = self.state[p] # Get state associated with p.
				t = state.get("t", 0) # Get iteration number from the state, or initial value. 
				grad = p.grad.data # Get the gradient of loss with respect to p.
				p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
				state["t"] = t + 1 # Increment iteration number.
		return loss



class AdamW(torch.optim.Optimizer):
	"""AdamW optimizer is Adam + weight decay."""

	def __init__(
		self, 
		params: torch.nn.Parameter,
		lr: float, 
		weight_decay: float, 
		betas: tuple[float, float],
		eps: float):

		# Surprisingly we cannot do params = [params] because that is a generator that gets consumed by m/v setup
		# Instead we gotta explicitly turn into an iterable via list()
		params = list(params)

		# Start off the moment vectors = zeros b/c we haven't taken any steps
		m = torch.zeros_like(params[0])
		v = torch.zeros_like(params[0])

		defaults = {
			"lr": lr,
			"weight_decay": weight_decay,
			"betas": betas,
			"moments": (m, v),
			"eps": eps,
			}

		super().__init__(params, defaults)

	def step(self, closure: Optional[Callable] = None): 
		"""Take a step according to formula on pg 32.

		Resource accounting of Transformer + AdamW

		Parameter calculation
			I don't want to use simplified FLOPs calculation for transformer when I worked so hard to calc the exact formula.
			Transformer
				transformer blocks = num_layers * (4 * d_model * d_model + 3 * d_model * d_ff + 2 * d_model)
				rmsnorm = d_model
				output embedding = vocab_size * d_model
				Total = vocab_size * d_model + 4 * num_layers * d_model * d_model + 3 * num_layers * d_model * 4 * d_model + (2 * num_layer + 1) * d_model
					= 16 * num_layers * d_model * d_model + (2 * num_layer + vocab_size + 1) * d_model 
			AdamW
				m = parameters
				v = parameters
				p.data = batch_size * parameters

			Total = 3 * [16 * num_layers * d_model * d_model + (2 * num_layer + vocab_size + 1) * d_model]
				= 42 * num_layers * d_model * d_model + (6 * num_layer + 3 * vocab_size + 3) * d_model

		Activations = 0

		FLOP calculations
			Pretend p.grad.data already computed
			m = 3 * batch_size * sequence_len * d_model
			v = 4 * batch_size * sequence_len * d_model
			p.data update 1 = 5 * batch_size * sequence_len * d_model
			p.data update 2 = 2 * batch_size * sequence_len * d_model
			Total = 14 * batch_size * sequence_len * d_model
		"""
		loss = None if closure is None else closure() 
		for group in self.param_groups:

			# Get all the params
			lr = group["lr"] 
			weight_decay = group["weight_decay"]
			eps = group["eps"]
			beta1, beta2 = group["betas"]

			# For each batch_t...
			for p in group["params"]: 
				if p.grad is None:
					continue

				state = self.state[p]
				t = state.get("t", 1)  # Start from 1
				m, v = group["moments"]

				grad = p.grad.data  # Get the gradient of loss with respect to p.
				m = beta1 * m + (1 - beta1) * grad
				v = beta2 * v + (1 - beta2) * grad ** 2
				lr_t = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)

				p.data -= lr_t * m / (torch.sqrt(v) + eps)  # Step w regard to learning
				p.data -= lr * weight_decay * p.data		# Step w regard to weight decay

				group["moments"] = (m, v)
				state["t"] = t + 1  # Increment iteration number.

		return loss






