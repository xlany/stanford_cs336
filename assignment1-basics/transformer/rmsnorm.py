import torch
from einops import rearrange, einsum, reduce
import math


class RMSNorm(torch.nn.Module):
	"""Normalize the logit distr for each word.

	word = [logit, logit, logit]
		Normalize this distr

	Also have learnable weight so some embedding positions adjusts to higher/lower prob.
	"""
	def __init__(
		self, 
		d_model: int, 
		eps: float = 1e-5,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None):

		# Any subclass of torch.nn.Module needs to call the 
		# superclass before we can do anything
		super().__init__()
		self.d_model = d_model
		self.eps = eps

		# Initialize to all ones at first
		self.g = torch.ones(self.d_model)

		# Make tensor trainable
		self.g = torch.nn.Parameter(self.g)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Normalize inputs.

		Resource accounting
			Parameters = d_model
			Activations = batch_size * sequence_len * d_model
			FLOPs = 4 * batch_size * sequence_len * d_model

		Activation calculations
			Reuse activation memory when possible
			x_normalized = batch_size * sequence_len * d_model
			Total = batch_size * sequence_len * d_model


		FLOPs calculations
			Input: [batch_size sequence_len d_model]
			Square = batch_size * sequence_len * d_model
			Mean_squared = batch_size * sequence_len * d_model
				Because adding them all up is d_model - 1 FLOPs
				But division is +1 FLOPs
			x_normalized = batch_size * sequence_len * d_model
			rms_norm = batch_size * sequence_len * d_model
			Total = 4 * batch_size * sequence_len * d_model

			Things that don't count: add, sqrt, upscaling (memory not computation)
		"""
		# Upscale to higher precision
		in_dtype = x.dtype
		x = x.to(torch.float32)

		x_squared = x ** 2  			# square is element-wise, still [batch_size sequence_len d_model]
		mean_squared = reduce(
			x_squared, 
			'batch_size sequence_len d_model -> batch_size sequence_len 1',  # avg over d_model dimensions
			'mean')						
		rms = mean_squared + self.eps 	# addition is element-wise, self.eps is scalar thus broadcast. Still [batch_size sequence_len 1]
		rms = torch.sqrt(rms)  			# torch.sqrt is element-wise. Still [batch_size sequence_len 1]
		x_normalized = x / rms 			# [batch_size sequence_len d_model] / [batch_size sequence_len 1] = [batch_size sequence_len d_model]
										# This makes the vector have unit norm
		rms_norm = einsum(x_normalized, self.g, 'batch_size sequence_len d_model, d_model -> batch_size sequence_len d_model')
			# think multiply by diagonal matrix [g1 ... ...]
			#									[... g2 ...]
			#									[... ... g3]
			# multiplication becomes element-wise [a1g1, a2g2, a3g3]

		# Downscale to original precision
		return rms_norm.to(in_dtype)


