import torch
from einops import rearrange, einsum
import math

class Linear(torch.nn.Module):
	def __init__(
		self, 
		in_features: int, 
		out_features: int, 
		device: torch.device | None = None, 
		dtype: torch.dtype | None = None):
		"""Create randomized W."""

		# Any subclass of torch.nn.Module needs to call the 
		# superclass before we can do anything
		super().__init__()

		# Make tensor shape
		self.W = torch.empty(out_features, in_features)

		# Make tensor trainable
		self.W = torch.nn.Parameter(self.W)

		# Set initial weights
		# Normal distribution with truncation
		std_squared = 2/(out_features + in_features)
		std = math.sqrt(std_squared)
		torch.nn.init.trunc_normal_(
			self.W, 
			mean=0, 
			std=std,
			a=-3*std,
			b=3*std
			)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Calculate y = W_t * x.

		Resource accounting
			Num of parameters = d_in * d_out
			FLOPs = 2 * ... * d_in * d_out
		"""
		W_t = rearrange(
			self.W,
			"d_out d_in -> d_in d_out"
			)
		return einsum(x, W_t, "... d_in, d_in d_out -> ... d_out")


	def from_weights(self, W: torch.Tensor) -> None:
		"""Initialize from weights."""
		self.W = torch.nn.Parameter(W)


	def weights(self) -> torch.Tensor:
		"""Return weights."""
		return self.W

