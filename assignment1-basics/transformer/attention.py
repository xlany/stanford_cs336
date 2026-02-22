import torch
from einops import einsum, rearrange
import transformer.linear as linear
import math
import transformer.softmax as softmax
import transformer.rope as rope

def scaled_dot_product_attention(
	Q: torch.Tensor,
	K: torch.Tensor,
	V: torch.Tensor,
	mask: torch.Tensor | None = None
	) -> torch.Tensor:
	"""Calculate attention.

	1. Find similarity between Q & K which is dot product of each word in the sequence.
	2. Scale back the scores by how long the embedding length (d_k) is. Because doesn't make sense for words to get infinitely more similar 
		(bigger dot product) just from the embedding length increasing
	3. If there is a mask, we convert False -> -inf score
	4. Put it through the softmax


	Resource accounting
		Parameters = 0
		Activation = 2 * batch_size * sequence_len_q * sequence_len_k + batch_size * sequence_len_q * d_v
		FLOPs = 4 * batch_size * sequence_len_q * sequence_len_k * d_v/d_k + 7 * batch_size * sequence_len_q * sequence_len_k

	Activation calculation
		Q_Kt = batch_size * sequence_len_q * sequence_len_k 
			Based on output size
		softmax = batch_size * sequence_len_q * sequence_len_k
		output = batch_size * sequence_len_q * d_v

	FLOPs calculation
		Q_Kt = [batch_size ... sequence_len_q d_k] x [batch_size ... d_k sequence_len_k]
			= 2 * batch_size * sequence_len_q * sequence_len_k * d_k
		scores = batch_size * sequence_len_q * sequence_len_k
		mask = batch_size * sequence_len_q * sequence_len_k
		softmax = 5 * batch_size * sequence_len_q * sequence_len_k
		V = [batch_size ... sequence_len_q sequence_len_k] x [batch_size ... sequence_len_k d_v]
			= 2 * batch_size * sequence_len_q * sequence_len_k * d_v
		Total = 2 * batch_size * sequence_len_q * sequence_len_k * (d_v + d_k) + 7 * batch_size * sequence_len_q * sequence_len_k
	"""
	
	d_k = Q.shape[-1]
	K_transpose = rearrange(K, "batch_size ... sequence_len d_k -> batch_size ... d_k sequence_len")
	Q_Kt = einsum(Q, K_transpose, "batch_size ... sequence_len_q d_k, batch_size ... d_k sequence_len_k -> batch_size ... sequence_len_q sequence_len_k")
	scores = Q_Kt / math.sqrt(d_k)
	if mask is not None: 
		scores = scores.masked_fill(~mask, float('-inf')) # Invert the mask first so that True = turn into -inf
														  # Then apply mask in-place

	# Why dim_i=-1?
	# Q_Kt[..., i, 0] = score between query i and key 0
	# Q_Kt[..., i, 1] = score between query i and key 1
	# Q_Kt[..., i, 2] = score between query i and key 2
	# dim_i=-1 puts the keys score into probability distribution
	softmaxxed = softmax.softmax(scores, dim_i=-1)
	return einsum(softmaxxed, V, "batch_size ... sequence_len_q sequence_len_k, batch_size ... sequence_len_k d_v -> batch_size ... sequence_len_q d_v")
	

class CausalMultiHeadSelfAttention(torch.nn.Module):
	"""Use multiple attention heads."""

	def __init__(
			self, 
			d_model: int, 
			num_heads: int,
			rope_config: rope.RopeConfig | None = None
			):
		"""Initialize Q, W, V, combination layer.


		Rather than using full matricies, split between num_heads attention heads. 
		Treat each attention head as having its own separate Q[i], K[i], V[i].
		There is one final W_O layer (full matrix) that learns how to combine together these separate heads. 
		"""
		super().__init__()

		self.d_model = d_model
		self.num_heads = num_heads
		self.d_k = int(d_model / num_heads)
		self.d_v = int(d_model / num_heads)

		self.all_WQ = torch.nn.ModuleList([linear.Linear(self.d_k, self.d_model) for _ in range(self.num_heads)])
		self.all_WK = torch.nn.ModuleList([linear.Linear(self.d_k, self.d_model) for _ in range(self.num_heads)])
		self.all_WV = torch.nn.ModuleList([linear.Linear(self.d_v, self.d_model) for _ in range(self.num_heads)])
		self.W_O = linear.Linear(self.num_heads * self.d_v, self.d_model)

		self.rope = None
		if rope_config:
			self.rope = rope.RotaryPositionalEmbedding(
				theta=rope_config.theta,
				d_k=rope_config.d_k,
				max_seq_len=rope_config.max_seq_len)


	def from_weights(self, q_weights: torch.Tensor, k_weights: torch.Tensor, v_weights: torch.Tensor, o_weights: torch.Tensor) -> None:
		"""Initialize Q, K, V, combination layer from weights.

		Since this is multi-head attention, we need to split the full Q, K, V matricies into separate Q[i], K[i], V[i].

		Args:
			q_weights: [d_model, d_model]
			k_weights: [d_model, d_model]
			v_weights: [d_model, d_model]
			o_weights: [d_model, d_model]
		"""
		q_chunks = torch.chunk(q_weights, self.num_heads, dim=0)  # Split [d_model, d_model] into a bunch of [d_k, d_model]
		k_chunks = torch.chunk(k_weights, self.num_heads, dim=0)
		v_chunks = torch.chunk(v_weights, self.num_heads, dim=0)
		for chunks, linear_layers in zip([q_chunks, k_chunks, v_chunks], [self.all_WQ, self.all_WK, self.all_WV]):
			for chunk, linear_layer in zip(chunks, linear_layers):
				linear_layer.from_weights(chunk)
				# linear_layer.load_state_dict({'W': chunk}) -> this doesn't work for some reason

		self.W_O.load_state_dict({'W': o_weights})


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		"""Perform the attention calculation.

		Args:
			x: [batch_size sequence_len d_model]

		Resource accounting
			Parameters = 4 * d_model * d_model
			Activations = 4 * batch_size * sequence_len * d_model + 4 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v
			FLOPs = 6 * batch_size * sequence_len * d_model * d_model
				  + 3 * batch_size * sequence_len * d_model
				  + 4 * batch_size * sequence_len * sequence_len * d_model
				  + 7 * batch_size * sequence_len * sequence_len * num_heads

		Activations calculation
			Q = batch_size * sequence_len * d_model
			K = batch_size * sequence_len * d_model
			V = batch_size * sequence_len * d_model
			attention = num_heads * (2 * batch_size * sequence_len_q * sequence_len_k + batch_size * sequence_len_q * d_v)
			output = batch_size * sequence_len * d_model
			Total = 4 * batch_size * sequence_len * d_model + 2 * num_heads * batch_size * sequence_len_q * sequence_len_k + num_heads * batch_size * sequence_len_q * d_v

		FLOPs calculation
			Q = [batch_size sequence_len d_model] x [d_model, d_model] 
				= 2 * batch_size * sequence_len * d_model * d_model
			K = 2 * batch_size * sequence_len * d_model * d_model
			V = 2 * batch_size * sequence_len * d_model * d_model
			rope_Q = 3 * batch_size * sequence_len * d_model
			rope_K = 3 * batch_size * sequence_len * d_model
			Causal mask = sequence_len * sequence_len -> too small
			all attention = num_heads * each attention 
				= num_heads * (4 * batch_size * sequence_len * sequence_len * {d_v or d_k} + 7 * batch_size * sequence_len * sequence_len)
				= 4 * batch_size * sequence_len * sequence_len * d_model + 7 * batch_size * sequence_len * sequence_len * num_heads
			W_0 forward = [batch_size sequence_len d_model] x [d_model d_model]
				= 2 * batch_size * sequence_len * d_model * d_model
			Total = 8 * batch_size * sequence_len * d_model * d_model
				  + 6 * batch_size * sequence_len * d_model
				  + 4 * batch_size * sequence_len * sequence_len * d_model
				  + 7 * batch_size * sequence_len * sequence_len * num_heads
		"""
		_, sequence_len, _ = x.shape
		
		# Every attention head looks at the full x. They each produce a different view of it. 
		Q = [self.all_WQ[i].forward(x) for i in range(self.num_heads)]  # [batch_size sequence_length d_model] x [d_model, d_k] = [batch_size sequence_length d_k]
		K = [self.all_WK[i].forward(x) for i in range(self.num_heads)]  # [batch_size sequence_length d_model] x [d_model, d_k] = [batch_size sequence_length d_k]
		V = [self.all_WV[i].forward(x) for i in range(self.num_heads)]  # [batch_size sequence_length d_model] x [d_model, d_v] = [batch_size sequence_length d_v]


		if self.rope:
			# Example query or key shape: 		[batch_size sequence_length d_k] = [4, 12, 16]
			# Example token_positions shape: 	[sequence_length] = [12]
			# Example token_positions: 		 	[0, 1, 2, ..., 11]
			token_positions = torch.arange(0, sequence_len)
			Q = [self.rope.forward(q, token_positions) for q in Q]	
			K = [self.rope.forward(k, token_positions) for k in K]

		all_true = torch.full((sequence_len, sequence_len), True, dtype=torch.bool)  # [sequence_length sequence_length]
		causal_mask = torch.tril(all_true, diagonal=0)		# Make the bottom triangle all True because each token can only look at itself and the ones before it
															# T F F
															# T T F
															# T T T

		attention = [scaled_dot_product_attention(Q[i], K[i], V[i], causal_mask) for i in range(self.num_heads)]  # Output: [batch_size sequence_length d_v] per head

		# We concatenate each attention head's view of the input -> make a full view
		multihead_attention = torch.cat(attention, dim=-1)  # Concatenate multiple [batch_size sequence_length d_v] -> [batch_size sequence_length d_model]

		return self.W_O.forward(multihead_attention)		# Apply the combination layer -> [batch_size sequence_length d_model]





