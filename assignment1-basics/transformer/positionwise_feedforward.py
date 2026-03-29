import torch
from einops import einsum
import math
import transformer.linear as linear

def _SiLU(x: torch.Tensor):
	"""Element-wise.

	Resource accounting
		Parameters = 0
		Activations = batch_size * sequence_length * d_ff
		FLOPs = 2 * batch_size * sequence_length * d_ff

	Activations calculation
		output = batch_size * sequence_length * d_ff

	FLOPs accounting
		Input: [batch_size sequence_length d_ff]
		sigmoid = batch_size * sequence_length * d_ff
		multiply = batch_size * sequence_length * d_ff
		Total = 2 * batch_size * sequence_length * d_ff
	"""
	return x * torch.sigmoid(x)

class SwiGLU(torch.nn.Module):
	def __init__(
		self,
		d_model: int,
		d_ff: int | None = None,
		device: torch.device | None = None, 
		dtype: torch.dtype | None = None,
	):

		super().__init__()

		self.d_model = d_model
		if d_ff:
			self.d_ff = d_ff
		else:
			# Calculate d_ff by (8/3 * self.d_model)
			# Round up to the nearest multiple of 64
			multiple_64 = math.ceil((8/3 * self.d_model)/64)
			self.d_ff = multiple_64 * 64

		self.W1 = linear.Linear(self.d_model, self.d_ff, device, dtype)
		self.W3 = linear.Linear(self.d_model, self.d_ff, device, dtype)
		self.W2 = linear.Linear(self.d_ff, self.d_model, device, dtype)


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Go through FFN.

		Resource accounting
			Parameters = 3 * d_model * d_ff
			Activations = 4 * batch_size * sequence_length * d_ff + batch_size * sequence_length * d_model
			FLOPs = 6 * batch_size * sequence_length * d_model * d_ff 
				  + 2 * batch_size * sequence_length * d_ff

		Activations calculation
			Simplified according to instructions on pg 33
			W1_x = batch_size * sequence_length * d_ff
			silu_W1_x = batch_size * sequence_length * d_ff
			output = batch_size * sequence_length * d_model
			Total = 2 * batch_size * sequence_length * d_ff + batch_size * sequence_length * d_model

		FLOPs calculation
			Input [batch_size, sequence_length, d_model] x [d_model d_ff]
			W1_x = 2 * batch_size * sequence_length * d_model * d_ff
			W3_x = 2 * batch_size * sequence_length * d_model * d_ff
			silu_W1_x = 2 * batch_size * sequence_length * d_ff
			inner = batch_size * sequence_length * d_ff
			W2_x = [batch_size sequence_length d_ff] x [d_ff d_model]
				= 2 * batch_size * sequence_length * d_model * d_ff
			Total = 6 * batch_size * sequence_length * d_model * d_ff + 3 * batch_size * sequence_length * d_ff

		"""
		W1_x = self.W1.forward(x)		# [batch_size, sequence_length, d_ff]
		silu_W1_x = _SiLU(W1_x)			# [batch_size, sequence_length, d_ff]
		W3_x = self.W3.forward(x)		# [batch_size, sequence_length, d_ff]

		inner = silu_W1_x * W3_x		# Same shape so element-wise [batch_size, sequence_length, d_ff]
		
		return self.W2.forward(inner)


class SiLU(torch.nn.Module):
	def __init__(
		self,
		d_model: int,
		d_ff: int | None = None,
		device: torch.device | None = None, 
		dtype: torch.dtype | None = None,
	):

		super().__init__()

		self.d_model = d_model
		if d_ff:
			self.d_ff = d_ff
		else:
			self.d_ff = 4 * self.d_model

		self.W1 = linear.Linear(self.d_model, self.d_ff, device, dtype)
		self.W2 = linear.Linear(self.d_ff, self.d_model, device, dtype)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Go through FFN."""
		W1_x = self.W1.forward(x)		# [batch_size, sequence_length, d_ff]
		silu_W1_x = _SiLU(W1_x)			# [batch_size, sequence_length, d_ff]	
		return self.W2.forward(silu_W1_x)









