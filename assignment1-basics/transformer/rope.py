
import torch
from einops import einsum
import math
import transformer.linear as linear
import dataclasses

@dataclasses.dataclass
class RopeConfig:
	theta: float
	d_k: int 
	max_seq_len: int


class RotaryPositionalEmbedding(torch.nn.Module):
	def __init__(
		self, 
		theta: float, 
		d_k: int,		
		max_seq_len: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None):
		"""
		theta = how much to "spread out" each row in the cos/sin matrix
		d_k = number of dimensions each attention head looks at
		max_seq_len = the longest a sentence can be
			Determines how big we need to precompute our cos/sin matrix
		"""
		super().__init__()

		self.theta = theta
		self.device = device
		self.dtype = dtype
		max_seq_len = max_seq_len
		rope_cache = self.precompute_rotary_emb(dim=d_k, max_positions=max_seq_len)
		self.register_buffer("rope_cache", rope_cache, persistent=False)


	def precompute_rotary_emb(self, dim: int, max_positions: int):
		"""Precompute the cos/sin matrix.

		Adapted from my code for CS224N.
		"""
		rope_cache = torch.zeros(max_positions, dim//2, 2, device=self.device, dtype=self.dtype)

		for i in range(1, dim//2 + 1):  
			theta_i = self.theta ** (-2*(i-1)/dim)
			
			for t in range(max_positions):
				rope_cache[t, i-1, 0] = torch.cos(torch.tensor(t * theta_i))
				rope_cache[t, i-1, 1] = torch.sin(torch.tensor(t * theta_i))

		return rope_cache

	def apply_rotary_emb(self, x: torch.Tensor, rope_cache: torch.Tensor):
		"""Apply the RoPE to the input tensor x.

		Adapted from my code for CS224N.
		"""
		batch_size, sequence_len, d_k = x.shape
				
		# Reshape x to pair up consecutive dimensions for complex representation
		x_pairs = x.reshape(batch_size, sequence_len, d_k // 2, 2)
		
		# Convert to complex representation
		x_complex = torch.view_as_complex(x_pairs) 
		rope_complex = torch.view_as_complex(rope_cache)
		
		# Apply rotation via complex multiplication
		# rope_complex broadcasts across batch and heads dimensions
		rotated_complex = x_complex * rope_complex
		
		# Convert back to real representation
		rotated_real = torch.view_as_real(rotated_complex)
		
		# Reshape back to original dimensions
		rotated_x = rotated_real.reshape(*x.shape)
		
		return rotated_x

	def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
		"""Perform RoPE calculation on input.

		Only get the cos/sin positions that we need (given by token_positions)
		Because RoPE cache is recomputed to be max_seq_len long but the input only has sequence_len.

		Resource accounting
			Parameters = 0
			FLOPs = 3 * batch_size * sequence_len * d_k

		FLOPs accounting
			Input: [batch_size sequence_len d_k]
			rope_complex = [x_complex] x [rope_complex]
				= [batch_size sequence_len (d_k // 2)] x [batch_size sequence_len (d_k // 2)] complex
				Each complex multiplication is 6 FLOPs
					(a + bi) * (c + di) = (ac - bd) + (ad + bc)i
				= 6 * batch_size * sequence_len * (d_k // 2)
				= 3 * batch_size * sequence_len * d_k

		Don't count: rotations (they are memory operations)
		"""

		rope_cache = self.rope_cache[token_positions]

		return self.apply_rotary_emb(x, rope_cache)



