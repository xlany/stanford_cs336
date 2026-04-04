import torch
from einops import rearrange, einsum
import torch.nn.functional as F


class Embedding(torch.nn.Module):
	"""Convert from full vocab size to embedding in d_model space.

	vocab_size -> d_model
	"""


	def __init__(
		self, 
		num_embeddings: int,
		embeddings_dim: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None):

		# Any subclass of torch.nn.Module needs to call the 
		# superclass before we can do anything
		super().__init__()

		# Make tensor shape
		self.vocab_size = num_embeddings
		self.d_model = embeddings_dim
		self.W = torch.empty(self.vocab_size, self.d_model, device=device, dtype=dtype)

		# Make tensor trainable
		self.W = torch.nn.Parameter(self.W)

		# Set initial weights
		# Normal distribution with truncation
		std = 1
		torch.nn.init.trunc_normal_(
			self.W, 
			mean=0, 
			std=std,
			a=-3*std,
			b=3*std
			)

	def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
		"""Token_ids come in as [batch_size, sequence_len] with the token directly ended as number (i.e. 9000).

		We need to make it [batch_size, sequence_len, vocab_size] by performing one hot encoding (i.e. [0, ..., 1, 0, ...]).

		Then calculate y = x * W.

		Resource accounting
			Parameters = vocab_size * d_model
			FLOPs = [batch_size sequence_len vocab_size] x [vocab_size, d_model] = [batch_size sequence_len d_model]
				= batch_size * 2(sequence_len)(vocab_size)(d_model)
				= 2 * batch_size * sequence_len * vocab_size * d_model
		"""
		one_hot_encoding = F.one_hot(token_ids, num_classes=self.vocab_size).float()


		return einsum(one_hot_encoding, self.W, "... vocab_size, vocab_size d_model -> ... d_model")

